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

use rustc_infer::infer::canonical::{Canonical, CanonicalVarKind, CanonicalVarValues};
use rustc_infer::infer::canonical::{OriginalQueryValues, QueryRegionConstraints, QueryResponse};
use rustc_infer::infer::{InferCtxt, InferOk, TyCtxtInferExt};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::Obligation;
use rustc_middle::infer::canonical::Certainty as OldCertainty;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{RegionOutlivesPredicate, ToPredicate, TypeOutlivesPredicate};
use rustc_span::DUMMY_SP;

use crate::traits::ObligationCause;

mod assembly;
mod fulfill;
mod infcx_ext;
mod project_goals;
mod search_graph;
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
#[derive(Debug, PartialEq, Eq, Clone, Hash, TypeFoldable, TypeVisitable, Default)]
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
        let mut search_graph = search_graph::SearchGraph::new(self);
        EvalCtxt::evaluate_canonical_goal(self, &mut search_graph, goal)
    }
}

struct EvalCtxt<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    var_values: CanonicalVarValues<'tcx>,

    search_graph: &'a mut search_graph::SearchGraph<'tcx>,
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    /// Creates a new evaluation context outside of the trait solver.
    ///
    /// With this solver making a canonical response doesn't make much sense.
    /// The `search_graph` for this solver has to be completely empty.
    fn new_outside_solver(
        infcx: &'a InferCtxt<'tcx>,
        search_graph: &'a mut search_graph::SearchGraph<'tcx>,
    ) -> EvalCtxt<'a, 'tcx> {
        assert!(search_graph.is_empty());
        EvalCtxt { infcx, var_values: CanonicalVarValues::dummy(), search_graph }
    }

    #[instrument(level = "debug", skip(tcx, search_graph), ret)]
    fn evaluate_canonical_goal(
        tcx: TyCtxt<'tcx>,
        search_graph: &'a mut search_graph::SearchGraph<'tcx>,
        canonical_goal: CanonicalGoal<'tcx>,
    ) -> QueryResult<'tcx> {
        match search_graph.try_push_stack(tcx, canonical_goal) {
            Ok(()) => {}
            // Our goal is already on the stack, eager return.
            Err(response) => return response,
        }

        // We may have to repeatedly recompute the goal in case of coinductive cycles,
        // check out the `cache` module for more information.
        //
        // FIXME: Similar to `evaluate_all`, this has to check for overflow.
        loop {
            let (ref infcx, goal, var_values) =
                tcx.infer_ctxt().build_with_canonical(DUMMY_SP, &canonical_goal);
            let mut ecx = EvalCtxt { infcx, var_values, search_graph };
            let result = ecx.compute_goal(goal);

            // FIXME: `Response` should be `Copy`
            if search_graph.try_finalize_goal(tcx, canonical_goal, result.clone()) {
                return result;
            }
        }
    }

    fn make_canonical_response(&self, certainty: Certainty) -> QueryResult<'tcx> {
        let external_constraints = take_external_constraints(self.infcx)?;

        Ok(self.infcx.canonicalize_response(Response {
            var_values: self.var_values.clone(),
            external_constraints,
            certainty,
        }))
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution> {
        let mut orig_values = OriginalQueryValues::default();
        let canonical_goal = self.infcx.canonicalize_query(goal, &mut orig_values);
        let canonical_response =
            EvalCtxt::evaluate_canonical_goal(self.tcx(), self.search_graph, canonical_goal)?;
        Ok((
            !canonical_response.value.var_values.is_identity(),
            instantiate_canonical_query_response(self.infcx, &orig_values, canonical_response),
        ))
    }

    fn compute_goal(&mut self, goal: Goal<'tcx, ty::Predicate<'tcx>>) -> QueryResult<'tcx> {
        let Goal { param_env, predicate } = goal;
        let kind = predicate.kind();
        if let Some(kind) = kind.no_bound_vars() {
            match kind {
                ty::PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                    self.compute_trait_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) => {
                    self.compute_projection_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(predicate)) => {
                    self.compute_type_outlives_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(predicate)) => {
                    self.compute_region_outlives_goal(Goal { param_env, predicate })
                }
                // FIXME: implement these predicates :)
                ty::PredicateKind::WellFormed(_)
                | ty::PredicateKind::ObjectSafe(_)
                | ty::PredicateKind::ClosureKind(_, _, _)
                | ty::PredicateKind::Subtype(_)
                | ty::PredicateKind::Coerce(_)
                | ty::PredicateKind::ConstEvaluatable(_)
                | ty::PredicateKind::ConstEquate(_, _)
                | ty::PredicateKind::TypeWellFormedFromEnv(_)
                | ty::PredicateKind::Ambiguous => self.make_canonical_response(Certainty::Yes),
            }
        } else {
            let kind = self.infcx.replace_bound_vars_with_placeholders(kind);
            let goal = goal.with(self.tcx(), ty::Binder::dummy(kind));
            let (_, certainty) = self.evaluate_goal(goal)?;
            self.make_canonical_response(certainty)
        }
    }

    fn compute_type_outlives_goal(
        &mut self,
        _goal: Goal<'tcx, TypeOutlivesPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.make_canonical_response(Certainty::Yes)
    }

    fn compute_region_outlives_goal(
        &mut self,
        _goal: Goal<'tcx, RegionOutlivesPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.make_canonical_response(Certainty::Yes)
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    fn evaluate_all(
        &mut self,
        mut goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> Result<Certainty, NoSolution> {
        let mut new_goals = Vec::new();
        self.repeat_while_none(|this| {
            let mut has_changed = Err(Certainty::Yes);
            for goal in goals.drain(..) {
                let (changed, certainty) = match this.evaluate_goal(goal) {
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

#[instrument(level = "debug", skip(infcx), ret)]
fn take_external_constraints<'tcx>(
    infcx: &InferCtxt<'tcx>,
) -> Result<ExternalConstraints<'tcx>, NoSolution> {
    let region_obligations = infcx.take_registered_region_obligations();
    let opaque_types = infcx.take_opaque_types_for_query_response();
    Ok(ExternalConstraints {
        // FIXME: Now that's definitely wrong :)
        //
        // Should also do the leak check here I think
        regions: drop(region_obligations),
        opaque_types,
    })
}

fn instantiate_canonical_query_response<'tcx>(
    infcx: &InferCtxt<'tcx>,
    original_values: &OriginalQueryValues<'tcx>,
    response: CanonicalResponse<'tcx>,
) -> Certainty {
    let Ok(InferOk { value, obligations }) = infcx
        .instantiate_query_response_and_region_obligations(
            &ObligationCause::dummy(),
            ty::ParamEnv::empty(),
            original_values,
            &response.unchecked_map(|resp| QueryResponse {
                var_values: resp.var_values,
                region_constraints: QueryRegionConstraints::default(),
                certainty: match resp.certainty {
                    Certainty::Yes => OldCertainty::Proven,
                    Certainty::Maybe(_) => OldCertainty::Ambiguous,
                },
                opaque_types: resp.external_constraints.opaque_types,
                value: resp.certainty,
            }),
        ) else { bug!(); };
    assert!(obligations.is_empty());
    value
}

pub(super) fn response_no_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: Canonical<'tcx, impl Sized>,
    certainty: Certainty,
) -> QueryResult<'tcx> {
    let var_values = goal
        .variables
        .iter()
        .enumerate()
        .map(|(i, info)| match info.kind {
            CanonicalVarKind::Ty(_) | CanonicalVarKind::PlaceholderTy(_) => {
                tcx.mk_ty(ty::Bound(ty::INNERMOST, ty::BoundVar::from_usize(i).into())).into()
            }
            CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => {
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(i),
                    kind: ty::BrAnon(i as u32, None),
                };
                tcx.mk_region(ty::ReLateBound(ty::INNERMOST, br)).into()
            }
            CanonicalVarKind::Const(_, ty) | CanonicalVarKind::PlaceholderConst(_, ty) => tcx
                .mk_const(ty::ConstKind::Bound(ty::INNERMOST, ty::BoundVar::from_usize(i)), ty)
                .into(),
        })
        .collect();

    Ok(Canonical {
        max_universe: goal.max_universe,
        variables: goal.variables,
        value: Response {
            var_values: CanonicalVarValues { var_values },
            external_constraints: Default::default(),
            certainty,
        },
    })
}
