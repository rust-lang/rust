//! Computes a normalizes-to (projection) goal for trait associated types and consts.
//!
//! We sadly can't simply take all possible candidates for normalization goals
//! and check whether they result in the same constraints. We want to make sure
//! that trying to normalize an alias doesn't result in constraints which aren't
//! otherwise required.
//!
//! Most notably, when proving a trait goal by via a where-bound, we should not
//! normalize via impls which have stricter region constraints than the where-bound:
//!
//! ```rust
//! trait Trait<'a> {
//!     type Assoc;
//! }
//!
//! impl<'a, T: 'a> Trait<'a> for T {
//!     type Assoc = u32;
//! }
//!
//! fn with_bound<'a, T: Trait<'a>>(_value: T::Assoc) {}
//! ```
//!
//! The where-bound of `with_bound` doesn't specify the associated type, so we would
//! only be able to normalize `<T as Trait<'a>>::Assoc` by using the impl. This impl
//! adds a `T: 'a` bound however, which would result in a region error. Given that the
//! user explicitly wrote that `T: Trait<'a>` holds, this is undesirable and we instead
//! treat the alias as rigid.
//!
//! See trait-system-refactor-initiative#124 for more details.
use rustc_type_ir::inherent::*;
use rustc_type_ir::{self as ty, Interner};
use tracing::{debug, instrument};

use crate::delegate::SolverDelegate;
use crate::solve::assembly::{AllowInferenceConstraints, AssembleCandidatesFrom};
use crate::solve::inspect::ProbeKind;
use crate::solve::trait_goals::TraitGoalProvenVia;
use crate::solve::{
    CandidateSource, Certainty, EvalCtxt, Goal, MaybeCause, NoSolution, QueryResult,
};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_projection_term(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        // Fast path via preferred env and alias bound candidates.
        if let Some(result) = self.assemble_and_merge_env_candidates(goal) {
            return result;
        }

        // Decides whether we need to assemble impl candidates.
        // We only assemble them if the trait goal is also proved via impl candidates.
        let cx = self.cx();
        let trait_ref = goal.predicate.alias.trait_ref(cx);
        let (_, proven_via) = self.probe(|_| ProbeKind::ShadowedEnvProbing).enter(|ecx| {
            let trait_goal: Goal<I, ty::TraitPredicate<I>> = goal.with(cx, trait_ref);
            ecx.compute_trait_goal(trait_goal)
        })?;
        if let Some(proven_via) = proven_via {
            match proven_via {
                TraitGoalProvenVia::ParamEnv | TraitGoalProvenVia::AliasBound => {
                    // This is somewhat inconsistent and may make #57893 slightly easier to exploit.
                    // However, it matches the behavior of the old solver. See
                    // `tests/ui/traits/next-solver/normalization-shadowing/use_object_if_empty_env.rs`.
                    // FIXME: predicates with opaque self type rely on assembly call to force
                    // ambiguous fallback candidate. It happens to be this object assembly call
                    // here.
                    let (candidates, _) =
                        self.assemble_and_evaluate_candidates(goal, AssembleCandidatesFrom::Object);
                    if !candidates.is_empty() {
                        debug_assert_eq!(candidates.len(), 1);
                        return Ok(candidates[0].result);
                    }
                    // If the trait goal has been proven by using the environment, we want to treat
                    // aliases as rigid if there are no applicable projection bounds in the environment.
                    self.probe(|&result| ProbeKind::RigidAlias { result }).enter(|this| {
                        this.structurally_instantiate_normalizes_to_term(
                            goal,
                            goal.predicate.alias,
                        );
                        this.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    })
                }
                TraitGoalProvenVia::Misc => self.assemble_and_merge_impl_candidates(goal),
            }
        } else {
            // We don't care about overflow. If proving the trait goal overflowed, then
            // it's enough to report an overflow error for that, we don't also have to
            // overflow during normalization.
            //
            // We use `forced_ambiguity` here over `make_ambiguous_response_no_constraints`
            // because the former will also record a built-in candidate in the inspector.
            self.forced_ambiguity(MaybeCause::Ambiguity).map(|cand| cand.result)
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn assemble_and_merge_env_candidates(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> Option<QueryResult<I>> {
        // Even when a trait bound has been proven using a where-bound, we
        // still need to consider alias-bounds for normalization, see
        // `tests/ui/next-solver/alias-bound-shadowed-by-env.rs`.
        let (mut candidates, _) = self
            .assemble_and_evaluate_candidates(goal, AssembleCandidatesFrom::EnvAndBoundsFastPath);
        debug!(?candidates);

        if candidates.is_empty() {
            return None;
        }

        // FIXME(generic_associated_types): Addresses aggressive inference in #92917.
        // If we're normalizing an GAT, we bail if using a where-bound would constrain
        // its generic arguments.
        //
        // If this type is a GAT with currently unconstrained arguments, we do not
        // want to normalize it via a candidate which only applies for a specific
        // instantiation. We could otherwise keep the GAT as rigid and succeed this way.
        // See tests/ui/generic-associated-types/no-incomplete-gat-arg-inference.rs.
        //
        // This only avoids normalization if a GAT argument is fully unconstrained.
        // This is quite arbitrary but fixing it causes some ambiguity, see #125196.
        for arg in goal.predicate.alias.own_args(self.cx()).iter() {
            let Some(term) = arg.as_term() else {
                continue;
            };
            match self.structurally_normalize_term(goal.param_env, term) {
                Ok(term) => {
                    if term.is_infer() {
                        return Some(self.evaluate_added_goals_and_make_canonical_response(
                            Certainty::AMBIGUOUS,
                        ));
                    }
                }
                Err(NoSolution) => return Some(Err(NoSolution)),
            }
        }

        // We still need to prefer where-bounds over alias-bounds however.
        // See `tests/ui/winnowing/norm-where-bound-gt-alias-bound.rs`.
        if candidates.iter().any(|c| matches!(c.source, CandidateSource::ParamEnv(_))) {
            candidates.retain(|c| matches!(c.source, CandidateSource::ParamEnv(_)));
        }

        let result = if let Some((response, _)) = self.try_merge_candidates(&candidates) {
            Ok(response)
        } else {
            self.flounder(&candidates)
        };
        Some(result)
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn assemble_and_merge_impl_candidates(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        // We already looked for param env and alias bound candidates on the fast path
        // so we don't have to assemble them again.
        let (mut candidates, _) =
            self.assemble_and_evaluate_candidates(goal, AssembleCandidatesFrom::Impl);

        // We drop specialized impls to allow normalization via a final impl here. In case
        // the specializing impl has different inference constraints from the specialized
        // impl, proving the trait goal is already ambiguous, so we never get here. This
        // means we can just ignore inference constraints and don't have to special-case
        // constraining the normalized-to `term`.
        self.filter_specialized_impls(AllowInferenceConstraints::Yes, &mut candidates);
        if let Some((response, _)) = self.try_merge_candidates(&candidates) {
            Ok(response)
        } else {
            self.flounder(&candidates)
        }
    }
}
