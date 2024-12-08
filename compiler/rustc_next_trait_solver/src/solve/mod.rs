//! The next-generation trait solver, currently still WIP.
//!
//! As a user of rust, you can use `-Znext-solver` to enable the new trait solver.
//!
//! As a developer of rustc, you shouldn't be using the new trait
//! solver without asking the trait-system-refactor-initiative, but it can
//! be enabled with `InferCtxtBuilder::with_next_trait_solver`. This will
//! ensure that trait solving using that inference context will be routed
//! to the new trait solver.
//!
//! For a high-level overview of how this solver works, check out the relevant
//! section of the rustc-dev-guide.

mod alias_relate;
mod assembly;
mod effect_goals;
mod eval_ctxt;
pub mod inspect;
mod normalizes_to;
mod project_goals;
mod search_graph;
mod trait_goals;

use rustc_type_ir::inherent::*;
pub use rustc_type_ir::solve::*;
use rustc_type_ir::{self as ty, Interner, TypingMode};
use tracing::instrument;

pub use self::eval_ctxt::{EvalCtxt, GenerateProofTree, SolverDelegateEvalExt};
use crate::delegate::SolverDelegate;

/// How many fixpoint iterations we should attempt inside of the solver before bailing
/// with overflow.
///
/// We previously used  `cx.recursion_limit().0.checked_ilog2().unwrap_or(0)` for this.
/// However, it feels unlikely that uncreasing the recursion limit by a power of two
/// to get one more itereation is every useful or desirable. We now instead used a constant
/// here. If there ever ends up some use-cases where a bigger number of fixpoint iterations
/// is required, we can add a new attribute for that or revert this to be dependant on the
/// recursion limit again. However, this feels very unlikely.
const FIXPOINT_STEP_LIMIT: usize = 8;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum GoalEvaluationKind {
    Root,
    Nested,
}

/// Whether evaluating this goal ended up changing the
/// inference state.
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy)]
pub enum HasChanged {
    Yes,
    No,
}

// FIXME(trait-system-refactor-initiative#117): we don't detect whether a response
// ended up pulling down any universes.
fn has_no_inference_or_external_constraints<I: Interner>(
    response: ty::Canonical<I, Response<I>>,
) -> bool {
    let ExternalConstraintsData {
        ref region_constraints,
        ref opaque_types,
        ref normalization_nested_goals,
    } = *response.value.external_constraints;
    response.value.var_values.is_identity()
        && region_constraints.is_empty()
        && opaque_types.is_empty()
        && normalization_nested_goals.is_empty()
}

impl<'a, D, I> EvalCtxt<'a, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "trace", skip(self))]
    fn compute_type_outlives_goal(
        &mut self,
        goal: Goal<I, ty::OutlivesPredicate<I, I::Ty>>,
    ) -> QueryResult<I> {
        let ty::OutlivesPredicate(ty, lt) = goal.predicate;
        self.register_ty_outlives(ty, lt);
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    #[instrument(level = "trace", skip(self))]
    fn compute_region_outlives_goal(
        &mut self,
        goal: Goal<I, ty::OutlivesPredicate<I, I::Region>>,
    ) -> QueryResult<I> {
        let ty::OutlivesPredicate(a, b) = goal.predicate;
        self.register_region_outlives(a, b);
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    #[instrument(level = "trace", skip(self))]
    fn compute_coerce_goal(&mut self, goal: Goal<I, ty::CoercePredicate<I>>) -> QueryResult<I> {
        self.compute_subtype_goal(Goal {
            param_env: goal.param_env,
            predicate: ty::SubtypePredicate {
                a_is_expected: false,
                a: goal.predicate.a,
                b: goal.predicate.b,
            },
        })
    }

    #[instrument(level = "trace", skip(self))]
    fn compute_subtype_goal(&mut self, goal: Goal<I, ty::SubtypePredicate<I>>) -> QueryResult<I> {
        if goal.predicate.a.is_ty_var() && goal.predicate.b.is_ty_var() {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
        } else {
            self.sub(goal.param_env, goal.predicate.a, goal.predicate.b)?;
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        }
    }

    fn compute_dyn_compatible_goal(&mut self, trait_def_id: I::DefId) -> QueryResult<I> {
        if self.cx().trait_is_dyn_compatible(trait_def_id) {
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn compute_well_formed_goal(&mut self, goal: Goal<I, I::GenericArg>) -> QueryResult<I> {
        match self.well_formed_goals(goal.param_env, goal.predicate) {
            Some(goals) => {
                self.add_goals(GoalSource::Misc, goals);
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            None => self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS),
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn compute_const_evaluatable_goal(
        &mut self,
        Goal { param_env, predicate: ct }: Goal<I, I::Const>,
    ) -> QueryResult<I> {
        match ct.kind() {
            ty::ConstKind::Unevaluated(uv) => {
                // We never return `NoSolution` here as `evaluate_const` emits an
                // error itself when failing to evaluate, so emitting an additional fulfillment
                // error in that case is unnecessary noise. This may change in the future once
                // evaluation failures are allowed to impact selection, e.g. generic const
                // expressions in impl headers or `where`-clauses.

                // FIXME(generic_const_exprs): Implement handling for generic
                // const expressions here.
                if let Some(_normalized) = self.evaluate_const(param_env, uv) {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                } else {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
            }
            ty::ConstKind::Infer(_) => {
                self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
            }
            ty::ConstKind::Placeholder(_)
            | ty::ConstKind::Value(_, _)
            | ty::ConstKind::Error(_) => {
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            // We can freely ICE here as:
            // - `Param` gets replaced with a placeholder during canonicalization
            // - `Bound` cannot exist as we don't have a binder around the self Type
            // - `Expr` is part of `feature(generic_const_exprs)` and is not implemented yet
            ty::ConstKind::Param(_) | ty::ConstKind::Bound(_, _) | ty::ConstKind::Expr(_) => {
                panic!("unexpected const kind: {:?}", ct)
            }
        }
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn compute_const_arg_has_type_goal(
        &mut self,
        goal: Goal<I, (I::Const, I::Ty)>,
    ) -> QueryResult<I> {
        let (ct, ty) = goal.predicate;

        let ct_ty = match ct.kind() {
            ty::ConstKind::Infer(_) => {
                return self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
            }
            ty::ConstKind::Error(_) => {
                return self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
            }
            ty::ConstKind::Unevaluated(uv) => {
                self.cx().type_of(uv.def).instantiate(self.cx(), uv.args)
            }
            ty::ConstKind::Expr(_) => unimplemented!(
                "`feature(generic_const_exprs)` is not supported in the new trait solver"
            ),
            ty::ConstKind::Param(_) => {
                unreachable!("`ConstKind::Param` should have been canonicalized to `Placeholder`")
            }
            ty::ConstKind::Bound(_, _) => panic!("escaping bound vars in {:?}", ct),
            ty::ConstKind::Value(ty, _) => ty,
            ty::ConstKind::Placeholder(placeholder) => {
                self.cx().find_const_ty_from_env(goal.param_env, placeholder)
            }
        };

        self.eq(goal.param_env, ct_ty, ty)?;
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    /// Try to merge multiple possible ways to prove a goal, if that is not possible returns `None`.
    ///
    /// In this case we tend to flounder and return ambiguity by calling `[EvalCtxt::flounder]`.
    #[instrument(level = "trace", skip(self), ret)]
    fn try_merge_responses(
        &mut self,
        responses: &[CanonicalResponse<I>],
    ) -> Option<CanonicalResponse<I>> {
        if responses.is_empty() {
            return None;
        }

        // FIXME(-Znext-solver): We should instead try to find a `Certainty::Yes` response with
        // a subset of the constraints that all the other responses have.
        let one = responses[0];
        if responses[1..].iter().all(|&resp| resp == one) {
            return Some(one);
        }

        responses
            .iter()
            .find(|response| {
                response.value.certainty == Certainty::Yes
                    && has_no_inference_or_external_constraints(**response)
            })
            .copied()
    }

    /// If we fail to merge responses we flounder and return overflow or ambiguity.
    #[instrument(level = "trace", skip(self), ret)]
    fn flounder(&mut self, responses: &[CanonicalResponse<I>]) -> QueryResult<I> {
        if responses.is_empty() {
            return Err(NoSolution);
        }

        let Certainty::Maybe(maybe_cause) =
            responses.iter().fold(Certainty::AMBIGUOUS, |certainty, response| {
                certainty.unify_with(response.value.certainty)
            })
        else {
            panic!("expected flounder response to be ambiguous")
        };

        Ok(self.make_ambiguous_response_no_constraints(maybe_cause))
    }

    /// Normalize a type for when it is structurally matched on.
    ///
    /// This function is necessary in nearly all cases before matching on a type.
    /// Not doing so is likely to be incomplete and therefore unsound during
    /// coherence.
    #[instrument(level = "trace", skip(self, param_env), ret)]
    fn structurally_normalize_ty(
        &mut self,
        param_env: I::ParamEnv,
        ty: I::Ty,
    ) -> Result<I::Ty, NoSolution> {
        if let ty::Alias(..) = ty.kind() {
            let normalized_ty = self.next_ty_infer();
            let alias_relate_goal = Goal::new(
                self.cx(),
                param_env,
                ty::PredicateKind::AliasRelate(
                    ty.into(),
                    normalized_ty.into(),
                    ty::AliasRelationDirection::Equate,
                ),
            );
            self.add_goal(GoalSource::Misc, alias_relate_goal);
            self.try_evaluate_added_goals()?;
            Ok(self.resolve_vars_if_possible(normalized_ty))
        } else {
            Ok(ty)
        }
    }

    /// Normalize a const for when it is structurally matched on, or more likely
    /// when it needs `.try_to_*` called on it (e.g. to turn it into a usize).
    ///
    /// This function is necessary in nearly all cases before matching on a const.
    /// Not doing so is likely to be incomplete and therefore unsound during
    /// coherence.
    #[instrument(level = "trace", skip(self, param_env), ret)]
    fn structurally_normalize_const(
        &mut self,
        param_env: I::ParamEnv,
        ct: I::Const,
    ) -> Result<I::Const, NoSolution> {
        if let ty::ConstKind::Unevaluated(..) = ct.kind() {
            let normalized_ct = self.next_const_infer();
            let alias_relate_goal = Goal::new(
                self.cx(),
                param_env,
                ty::PredicateKind::AliasRelate(
                    ct.into(),
                    normalized_ct.into(),
                    ty::AliasRelationDirection::Equate,
                ),
            );
            self.add_goal(GoalSource::Misc, alias_relate_goal);
            self.try_evaluate_added_goals()?;
            Ok(self.resolve_vars_if_possible(normalized_ct))
        } else {
            Ok(ct)
        }
    }

    fn opaque_type_is_rigid(&self, def_id: I::DefId) -> bool {
        match self.typing_mode() {
            // Opaques are never rigid outside of analysis mode.
            TypingMode::Coherence | TypingMode::PostAnalysis => false,
            // During analysis, opaques are rigid unless they may be defined by
            // the current body.
            TypingMode::Analysis { defining_opaque_types: non_rigid_opaques }
            | TypingMode::PostBorrowckAnalysis { defined_opaque_types: non_rigid_opaques } => {
                !def_id.as_local().is_some_and(|def_id| non_rigid_opaques.contains(&def_id))
            }
        }
    }
}

fn response_no_constraints_raw<I: Interner>(
    cx: I,
    max_universe: ty::UniverseIndex,
    variables: I::CanonicalVars,
    certainty: Certainty,
) -> CanonicalResponse<I> {
    ty::Canonical {
        max_universe,
        variables,
        value: Response {
            var_values: ty::CanonicalVarValues::make_identity(cx, variables),
            // FIXME: maybe we should store the "no response" version in cx, like
            // we do for cx.types and stuff.
            external_constraints: cx.mk_external_constraints(ExternalConstraintsData::default()),
            certainty,
        },
    }
}
