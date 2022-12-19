//! The new trait solver, currently still WIP.
//!
//! As a user of the trait system, you can use `TyCtxt::evaluate_goal` to
//! interact with this solver.
//!
//! For a high-level overview of how this solver works, check out the relevant
//! section of the rustc-dev-guide.
//!
//! FIXME(@lcnr): Write that section. If you read this before then ask me
//! about it on zulip.

// FIXME: Instead of using `infcx.canonicalize_query` we have to add a new routine which
// preserves universes and creates a unique var (in the highest universe) for each
// appearance of a region.

// FIXME: `CanonicalVarValues` should be interned and `Copy`.

// FIXME: uses of `infcx.at` need to enable deferred projection equality once that's implemented.

use std::mem;

use rustc_infer::infer::canonical::OriginalQueryValues;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::Obligation;
use rustc_middle::infer::canonical::{Canonical, CanonicalVarValues};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{RegionOutlivesPredicate, ToPredicate, TypeOutlivesPredicate};
use rustc_span::DUMMY_SP;

use self::infcx_ext::InferCtxtExt;

mod assembly;
mod cache;
mod fulfill;
mod infcx_ext;
mod overflow;
mod project_goals;
mod trait_goals;

pub use fulfill::FulfillmentCtxt;

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub struct Goal<'tcx, P> {
    param_env: ty::ParamEnv<'tcx>,
    predicate: P,
}

impl<'tcx, P> Goal<'tcx, P> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: impl ToPredicate<'tcx, P>,
    ) -> Goal<'tcx, P> {
        Goal { param_env, predicate: predicate.to_predicate(tcx) }
    }

    /// Updates the goal to one with a different `predicate` but the same `param_env`.
    fn with<Q>(self, tcx: TyCtxt<'tcx>, predicate: impl ToPredicate<'tcx, Q>) -> Goal<'tcx, Q> {
        Goal { param_env: self.param_env, predicate: predicate.to_predicate(tcx) }
    }
}

impl<'tcx, P> From<Obligation<'tcx, P>> for Goal<'tcx, P> {
    fn from(obligation: Obligation<'tcx, P>) -> Goal<'tcx, P> {
        Goal { param_env: obligation.param_env, predicate: obligation.predicate }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, TypeFoldable, TypeVisitable)]
pub struct Response<'tcx> {
    pub var_values: CanonicalVarValues<'tcx>,
    /// Additional constraints returned by this query.
    pub external_constraints: ExternalConstraints<'tcx>,
    pub certainty: Certainty,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub enum Certainty {
    Yes,
    Maybe(MaybeCause),
}

impl Certainty {
    /// When proving multiple goals using **AND**, e.g. nested obligations for an impl,
    /// use this function to unify the certainty of these goals
    pub fn unify_and(self, other: Certainty) -> Certainty {
        match (self, other) {
            (Certainty::Yes, Certainty::Yes) => Certainty::Yes,
            (Certainty::Yes, Certainty::Maybe(_)) => other,
            (Certainty::Maybe(_), Certainty::Yes) => self,
            (Certainty::Maybe(MaybeCause::Overflow), Certainty::Maybe(MaybeCause::Overflow)) => {
                Certainty::Maybe(MaybeCause::Overflow)
            }
            // If at least one of the goals is ambiguous, hide the overflow as the ambiguous goal
            // may still result in failure.
            (Certainty::Maybe(MaybeCause::Ambiguity), Certainty::Maybe(_))
            | (Certainty::Maybe(_), Certainty::Maybe(MaybeCause::Ambiguity)) => {
                Certainty::Maybe(MaybeCause::Ambiguity)
            }
        }
    }
}

/// Why we failed to evaluate a goal.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub enum MaybeCause {
    /// We failed due to ambiguity. This ambiguity can either
    /// be a true ambiguity, i.e. there are multiple different answers,
    /// or we hit a case where we just don't bother, e.g. `?x: Trait` goals.
    Ambiguity,
    /// We gave up due to an overflow, most often by hitting the recursion limit.
    Overflow,
}

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, TypeFoldable, TypeVisitable)]
pub struct ExternalConstraints<'tcx> {
    // FIXME: implement this.
    regions: (),
    opaque_types: Vec<(Ty<'tcx>, Ty<'tcx>)>,
}

type CanonicalGoal<'tcx, T = ty::Predicate<'tcx>> = Canonical<'tcx, Goal<'tcx, T>>;
type CanonicalResponse<'tcx> = Canonical<'tcx, Response<'tcx>>;
/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<'tcx> = Result<CanonicalResponse<'tcx>, NoSolution>;

pub trait TyCtxtExt<'tcx> {
    fn evaluate_goal(self, goal: CanonicalGoal<'tcx>) -> QueryResult<'tcx>;
}

impl<'tcx> TyCtxtExt<'tcx> for TyCtxt<'tcx> {
    fn evaluate_goal(self, goal: CanonicalGoal<'tcx>) -> QueryResult<'tcx> {
        let mut cx = EvalCtxt::new(self);
        cx.evaluate_canonical_goal(goal)
    }
}

struct EvalCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,

    provisional_cache: cache::ProvisionalCache<'tcx>,
    overflow_data: overflow::OverflowData,
}

impl<'tcx> EvalCtxt<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> EvalCtxt<'tcx> {
        EvalCtxt {
            tcx,
            provisional_cache: cache::ProvisionalCache::empty(),
            overflow_data: overflow::OverflowData::new(tcx),
        }
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution> {
        let mut orig_values = OriginalQueryValues::default();
        let canonical_goal = infcx.canonicalize_query(goal, &mut orig_values);
        let canonical_response = self.evaluate_canonical_goal(canonical_goal)?;
        Ok((
            true, // FIXME: check whether `var_values` are an identity substitution.
            fixme_instantiate_canonical_query_response(infcx, &orig_values, canonical_response),
        ))
    }

    fn evaluate_canonical_goal(&mut self, goal: CanonicalGoal<'tcx>) -> QueryResult<'tcx> {
        match self.try_push_stack(goal) {
            Ok(()) => {}
            // Our goal is already on the stack, eager return.
            Err(response) => return response,
        }

        // We may have to repeatedly recompute the goal in case of coinductive cycles,
        // check out the `cache` module for more information.
        //
        // FIXME: Similar to `evaluate_all`, this has to check for overflow.
        loop {
            let result = self.compute_goal(goal);

            // FIXME: `Response` should be `Copy`
            if self.try_finalize_goal(goal, result.clone()) {
                return result;
            }
        }
    }

    fn compute_goal(&mut self, canonical_goal: CanonicalGoal<'tcx>) -> QueryResult<'tcx> {
        // WARNING: We're looking at a canonical value without instantiating it here.
        //
        // We have to be incredibly careful to not change the order of bound variables or
        // remove any. As we go from `Goal<'tcx, Predicate>` to `Goal` with the variants
        // of `PredicateKind` this is the case and it is and faster than instantiating and
        // recanonicalizing.
        let Goal { param_env, predicate } = canonical_goal.value;
        if let Some(kind) = predicate.kind().no_bound_vars() {
            match kind {
                ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => self.compute_trait_goal(
                    canonical_goal.unchecked_rebind(Goal { param_env, predicate }),
                ),
                ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => self
                    .compute_projection_goal(
                        canonical_goal.unchecked_rebind(Goal { param_env, predicate }),
                    ),
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(predicate)) => self
                    .compute_type_outlives_goal(
                        canonical_goal.unchecked_rebind(Goal { param_env, predicate }),
                    ),
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(predicate)) => self
                    .compute_region_outlives_goal(
                        canonical_goal.unchecked_rebind(Goal { param_env, predicate }),
                    ),
                // FIXME: implement these predicates :)
                ty::PredicateKind::WellFormed(_)
                | ty::PredicateKind::ObjectSafe(_)
                | ty::PredicateKind::ClosureKind(_, _, _)
                | ty::PredicateKind::Subtype(_)
                | ty::PredicateKind::Coerce(_)
                | ty::PredicateKind::ConstEvaluatable(_)
                | ty::PredicateKind::ConstEquate(_, _)
                | ty::PredicateKind::TypeWellFormedFromEnv(_)
                | ty::PredicateKind::Ambiguous => unimplemented!(),
            }
        } else {
            let (infcx, goal, var_values) =
                self.tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &canonical_goal);
            let kind = infcx.replace_bound_vars_with_placeholders(goal.predicate.kind());
            let goal = goal.with(self.tcx, ty::Binder::dummy(kind));
            let (_, certainty) = self.evaluate_goal(&infcx, goal)?;
            infcx.make_canonical_response(var_values, certainty)
        }
    }

    fn compute_type_outlives_goal(
        &mut self,
        _goal: CanonicalGoal<'tcx, TypeOutlivesPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        todo!()
    }

    fn compute_region_outlives_goal(
        &mut self,
        _goal: CanonicalGoal<'tcx, RegionOutlivesPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        todo!()
    }
}

impl<'tcx> EvalCtxt<'tcx> {
    fn evaluate_all(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        mut goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> Result<Certainty, NoSolution> {
        let mut new_goals = Vec::new();
        self.repeat_while_none(|this| {
            let mut has_changed = Err(Certainty::Yes);
            for goal in goals.drain(..) {
                let (changed, certainty) = match this.evaluate_goal(infcx, goal) {
                    Ok(result) => result,
                    Err(NoSolution) => return Some(Err(NoSolution)),
                };

                if changed {
                    has_changed = Ok(());
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => {
                        new_goals.push(goal);
                        has_changed = has_changed.map_err(|c| c.unify_and(certainty));
                    }
                }
            }

            match has_changed {
                Ok(()) => {
                    mem::swap(&mut new_goals, &mut goals);
                    None
                }
                Err(certainty) => Some(Ok(certainty)),
            }
        })
    }
}

fn fixme_instantiate_canonical_query_response<'tcx>(
    _: &InferCtxt<'tcx>,
    _: &OriginalQueryValues<'tcx>,
    _: CanonicalResponse<'tcx>,
) -> Certainty {
    unimplemented!()
}
