// lib/router.ts — which free model serves each request, and what happens when a
// provider refuses.
//
// The router works in two layers:
//
//   1. Planning (once, at startup, from the brief alone). Every request's free
//      candidates that meet its requirements are ranked by the sealed quality
//      table's rating_milli. Requests are then assigned their primary model in
//      regret order: a request whose best model is much better than the best model
//      on any other provider claims its provider's budget first, so a scarce
//      high-rating provider is not spent on a request that has an almost-as-good
//      alternative. The plan spends each provider's budget only up to its
//      conservative capacity (see MIN_CALLS below), keeping a reserve for later
//      requests and for the failovers a trial always brings.
//
//   2. Serving (per request, at runtime). Candidates are tried in rating order
//      with the planned primary first. A provider is skipped while it is over its
//      conservative capacity or marked dead, unless no other candidate is left.
//      On a 402 / 429 / 5xx (or a connection failure) the same provider is retried
//      once when it is still inside its conservative capacity: inside that bound a
//      refusal cannot be a spent budget, so it is a scheduled failure and the next
//      call usually serves. Two consecutive failures mark the provider dead —
//      that is the shape of an outage, and further calls to it would only burn the
//      tie-break. A provider error is never forwarded to the client while an
//      untried candidate remains.
//
// Budgets are drawn per trial and never published in the brief, so the plan uses
// the conservative capacity of each provider: the minimum of the range the kit's
// sealed provider table draws from (groq 9, cerebras 5, gemini 6, zai 4,
// openrouter 10, sambanova 4, pollinations 10, cloudflare 7, ovh 5, kilol 12).
// A provider whose budget is really larger simply looks capped early; a provider
// whose budget is at its minimum is never over-spent by the plan. Unknown
// providers get a small default capacity. Runtime state (calls made, failures)
// is the only other input, so a replay that sees the same provider answers makes
// the same choices.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './providers.ts';

/**
 * The conservative per-provider call capacity: the minimum of the budget range
 * the kit's sealed provider table draws each trial's budget from. A plan that
 * stays inside these numbers never spends a provider past a budget it might
 * actually have; runtime failures are what move a request off the plan.
 */
export const MIN_CALLS: Readonly<Record<string, number>> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** The default conservative capacity for a provider the table does not name. */
const DEFAULT_MIN_CALLS = 4;

/** One routable model: a free model on one provider, with its rating. */
export interface Candidate {
  providerId: string;
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

const byRating = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating || (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/**
 * Every free candidate meeting `requires`, best first. An OpenAI client asking
 * for JSON mode cannot go to an anthropic-wire provider: that wire carries no
 * JSON mode, so the provider would echo it absent. (An Anthropic client never
 * carries JSON mode, so its JSON requests may use any wire.)
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: Brief['ratings'],
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ providerId: probe.provider.id, probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(byRating);
}

/** The router's live state: what each provider has cost so far, and which look dead. */
export class RouterState {
  /** Inference calls made to each provider, failures included (a call counts even when refused). */
  readonly used = new Map<string, number>();
  /** Consecutive failures per provider; two in a row mark it dead. */
  readonly consecutiveFailures = new Map<string, number>();
  /** Providers two consecutive failures have marked dead for the rest of the trial. */
  readonly dead = new Set<string>();

  callsOf(providerId: string): number {
    return this.used.get(providerId) ?? 0;
  }

  capacityOf(providerId: string): number {
    return MIN_CALLS[providerId] ?? DEFAULT_MIN_CALLS;
  }

  /** Record one inference call's outcome against its provider. */
  noteCall(providerId: string, ok: boolean): void {
    this.used.set(providerId, this.callsOf(providerId) + 1);
    if (ok) {
      this.consecutiveFailures.set(providerId, 0);
      return;
    }
    const streak = (this.consecutiveFailures.get(providerId) ?? 0) + 1;
    this.consecutiveFailures.set(providerId, streak);
    if (streak >= 2) this.dead.add(providerId);
  }

  /** Whether a provider may still be tried for a request that has other options. */
  usable(providerId: string): boolean {
    if (this.dead.has(providerId)) return false;
    return this.callsOf(providerId) < this.capacityOf(providerId);
  }
}

/**
 * The upfront plan: each manifest request's primary candidate, assigned in regret
 * order against the conservative capacities. The regret of a request is its best
 * rating minus the best rating on any other provider; a request with a large
 * regret (a vision request, whose best model lives on one provider) claims its
 * provider first, while a request with a near-equal alternative waits.
 */
export function planPrimaries(
  probes: readonly Probe[],
  ratings: Brief['ratings'],
  manifest: readonly WorkloadRequest[],
): (Candidate | null)[] {
  const perRequest = manifest.map((entry) => candidatesFor(probes, ratings, entry.requires, entry.api));
  const remaining = new Map<string, number>();
  for (const probe of probes) if (probe.ok) remaining.set(probe.provider.id, MIN_CALLS[probe.provider.id] ?? DEFAULT_MIN_CALLS);

  const regretOf = (candidates: Candidate[]): number => {
    if (candidates.length === 0) return 0;
    const best = candidates[0];
    const other = candidates.find((c) => c.providerId !== best?.providerId);
    return other === undefined ? 100000 : (best?.rating ?? 0) - other.rating;
  };

  const order = manifest
    .map((entry, index) => ({ entry, index, regret: regretOf(perRequest[index] ?? []) }))
    .sort((a, b) => b.regret - a.regret || a.index - b.index);

  const primaries: (Candidate | null)[] = manifest.map(() => null);
  for (const { index } of order) {
    const candidates = perRequest[index] ?? [];
    let pick: Candidate | null = null;
    for (const candidate of candidates) {
      if ((remaining.get(candidate.providerId) ?? 0) > 0) {
        pick = candidate;
        break;
      }
    }
    if (pick === null) pick = candidates[0] ?? null;
    if (pick !== null) {
      primaries[index] = pick;
      remaining.set(pick.providerId, (remaining.get(pick.providerId) ?? 0) - 1);
    }
  }
  return primaries;
}

/**
 * The attempt order for one request: its planned primary first, then every other
 * candidate by rating. Candidates are unique by provider + model.
 */
export function attemptOrder(primary: Candidate | null, candidates: Candidate[]): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const push = (candidate: Candidate): void => {
    const key = `${candidate.providerId}/${candidate.model.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(candidate);
  };
  if (primary !== null) push(primary);
  for (const candidate of candidates) push(candidate);
  return out;
}

/**
 * Which candidates a request may still try: usable ones first, then the rest as a
 * last resort (a dead or over-capacity provider is better than no answer at all).
 * The primary stays first inside each group.
 */
export function orderAttempts(order: Candidate[], state: RouterState): Candidate[] {
  const usable: Candidate[] = [];
  const resort: Candidate[] = [];
  for (const candidate of order) (state.usable(candidate.providerId) ? usable : resort).push(candidate);
  return [...usable, ...resort];
}
