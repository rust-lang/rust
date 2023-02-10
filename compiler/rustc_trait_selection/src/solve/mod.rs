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

// FIXME: uses of `infcx.at` need to enable deferred projection equality once that's implemented.

use std::mem;

use rustc_hir::def_id::DefId;
use rustc_infer::infer::canonical::{Canonical, CanonicalVarValues};
use rustc_infer::infer::canonical::{OriginalQueryValues, QueryRegionConstraints, QueryResponse};
use rustc_infer::infer::{InferCtxt, InferOk, TyCtxtInferExt};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::Obligation;
use rustc_middle::infer::canonical::Certainty as OldCertainty;
use rustc_middle::traits::solve::{ExternalConstraints, ExternalConstraintsData};
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::ty::{
    CoercePredicate, RegionOutlivesPredicate, SubtypePredicate, ToPredicate, TypeOutlivesPredicate,
};
use rustc_span::DUMMY_SP;

use crate::traits::ObligationCause;

mod assembly;
mod fulfill;
mod infcx_ext;
mod project_goals;
mod search_graph;
mod trait_goals;

pub use fulfill::FulfillmentCtxt;

use self::infcx_ext::InferCtxtExt;

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
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub struct Response<'tcx> {
    pub var_values: CanonicalVarValues<'tcx>,
    /// Additional constraints returned by this query.
    pub external_constraints: ExternalConstraints<'tcx>,
    pub certainty: Certainty,
}

trait CanonicalResponseExt {
    fn has_no_inference_or_external_constraints(&self) -> bool;
}

impl<'tcx> CanonicalResponseExt for Canonical<'tcx, Response<'tcx>> {
    fn has_no_inference_or_external_constraints(&self) -> bool {
        // so that we get a compile error when regions are supported
        // so this code can be checked for being correct
        let _: () = self.value.external_constraints.regions;

        self.value.var_values.is_identity()
            && self.value.external_constraints.opaque_types.is_empty()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub enum Certainty {
    Yes,
    Maybe(MaybeCause),
}

impl Certainty {
    pub const AMBIGUOUS: Certainty = Certainty::Maybe(MaybeCause::Ambiguity);

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

type CanonicalGoal<'tcx, T = ty::Predicate<'tcx>> = Canonical<'tcx, Goal<'tcx, T>>;
type CanonicalResponse<'tcx> = Canonical<'tcx, Response<'tcx>>;
/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<'tcx> = Result<CanonicalResponse<'tcx>, NoSolution>;

pub trait InferCtxtEvalExt<'tcx> {
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution>;
}

impl<'tcx> InferCtxtEvalExt<'tcx> for InferCtxt<'tcx> {
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution> {
        let mut search_graph = search_graph::SearchGraph::new(self.tcx);

        let result = EvalCtxt {
            search_graph: &mut search_graph,
            infcx: self,
            var_values: CanonicalVarValues::dummy(),
            in_projection_eq_hack: false,
        }
        .evaluate_goal(goal);

        assert!(search_graph.is_empty());
        result
    }
}

struct EvalCtxt<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    var_values: CanonicalVarValues<'tcx>,

    search_graph: &'a mut search_graph::SearchGraph<'tcx>,

    /// This field is used by a debug assertion in [`EvalCtxt::evaluate_goal`],
    /// see the comment in that method for more details.
    in_projection_eq_hack: bool,
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    /// The entry point of the solver.
    ///
    /// This function deals with (coinductive) cycles, overflow, and caching
    /// and then calls [`EvalCtxt::compute_goal`] which contains the actual
    /// logic of the solver.
    ///
    /// Instead of calling this function directly, use either [EvalCtxt::evaluate_goal]
    /// if you're inside of the solver or [InferCtxtEvalExt::evaluate_root_goal] if you're
    /// outside of it.
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
            let mut ecx =
                EvalCtxt { infcx, var_values, search_graph, in_projection_eq_hack: false };
            let result = ecx.compute_goal(goal);

            if search_graph.try_finalize_goal(tcx, canonical_goal, result) {
                return result;
            }
        }
    }

    fn make_canonical_response(&self, certainty: Certainty) -> QueryResult<'tcx> {
        let external_constraints = compute_external_query_constraints(self.infcx)?;

        Ok(self.infcx.canonicalize_response(Response {
            var_values: self.var_values,
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

        let has_changed = !canonical_response.value.var_values.is_identity();
        let certainty =
            instantiate_canonical_query_response(self.infcx, &orig_values, canonical_response);

        // Check that rerunning this query with its inference constraints applied
        // doesn't result in new inference constraints and has the same result.
        //
        // If we have projection goals like `<T as Trait>::Assoc == u32` we recursively
        // call `exists<U> <T as Trait>::Assoc == U` to enable better caching. This goal
        // could constrain `U` to `u32` which would cause this check to result in a
        // solver cycle.
        if cfg!(debug_assertions) && has_changed && !self.in_projection_eq_hack {
            let mut orig_values = OriginalQueryValues::default();
            let canonical_goal = self.infcx.canonicalize_query(goal, &mut orig_values);
            let canonical_response =
                EvalCtxt::evaluate_canonical_goal(self.tcx(), self.search_graph, canonical_goal)?;
            assert!(canonical_response.value.var_values.is_identity());
            assert_eq!(certainty, canonical_response.value.certainty);
        }

        Ok((has_changed, certainty))
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
                ty::PredicateKind::Subtype(predicate) => {
                    self.compute_subtype_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Coerce(predicate) => {
                    self.compute_coerce_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::ClosureKind(def_id, substs, kind) => self
                    .compute_closure_kind_goal(Goal {
                        param_env,
                        predicate: (def_id, substs, kind),
                    }),
                ty::PredicateKind::ObjectSafe(trait_def_id) => {
                    self.compute_object_safe_goal(trait_def_id)
                }
                ty::PredicateKind::WellFormed(arg) => {
                    self.compute_well_formed_goal(Goal { param_env, predicate: arg })
                }
                ty::PredicateKind::Ambiguous => self.make_canonical_response(Certainty::AMBIGUOUS),
                // FIXME: implement these predicates :)
                ty::PredicateKind::ConstEvaluatable(_) | ty::PredicateKind::ConstEquate(_, _) => {
                    self.make_canonical_response(Certainty::Yes)
                }
                ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                    bug!("TypeWellFormedFromEnv is only used for Chalk")
                }
                ty::PredicateKind::AliasEq(lhs, rhs) => {
                    self.compute_alias_eq_goal(Goal { param_env, predicate: (lhs, rhs) })
                }
            }
        } else {
            let kind = self.infcx.instantiate_binder_with_placeholders(kind);
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

    fn compute_coerce_goal(
        &mut self,
        goal: Goal<'tcx, CoercePredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        self.compute_subtype_goal(Goal {
            param_env: goal.param_env,
            predicate: SubtypePredicate {
                a_is_expected: false,
                a: goal.predicate.a,
                b: goal.predicate.b,
            },
        })
    }

    fn compute_subtype_goal(
        &mut self,
        goal: Goal<'tcx, SubtypePredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        if goal.predicate.a.is_ty_var() && goal.predicate.b.is_ty_var() {
            // FIXME: Do we want to register a subtype relation between these vars?
            // That won't actually reflect in the query response, so it seems moot.
            self.make_canonical_response(Certainty::AMBIGUOUS)
        } else {
            let InferOk { value: (), obligations } = self
                .infcx
                .at(&ObligationCause::dummy(), goal.param_env)
                .sub(goal.predicate.a, goal.predicate.b)?;
            self.evaluate_all_and_make_canonical_response(
                obligations.into_iter().map(|pred| pred.into()).collect(),
            )
        }
    }

    fn compute_closure_kind_goal(
        &mut self,
        goal: Goal<'tcx, (DefId, ty::SubstsRef<'tcx>, ty::ClosureKind)>,
    ) -> QueryResult<'tcx> {
        let (_, substs, expected_kind) = goal.predicate;
        let found_kind = substs.as_closure().kind_ty().to_opt_closure_kind();

        let Some(found_kind) = found_kind else {
            return self.make_canonical_response(Certainty::AMBIGUOUS);
        };
        if found_kind.extends(expected_kind) {
            self.make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn compute_object_safe_goal(&mut self, trait_def_id: DefId) -> QueryResult<'tcx> {
        if self.tcx().check_is_object_safe(trait_def_id) {
            self.make_canonical_response(Certainty::Yes)
        } else {
            Err(NoSolution)
        }
    }

    fn compute_well_formed_goal(
        &mut self,
        goal: Goal<'tcx, ty::GenericArg<'tcx>>,
    ) -> QueryResult<'tcx> {
        match crate::traits::wf::unnormalized_obligations(
            self.infcx,
            goal.param_env,
            goal.predicate,
        ) {
            Some(obligations) => self.evaluate_all_and_make_canonical_response(
                obligations.into_iter().map(|o| o.into()).collect(),
            ),
            None => self.make_canonical_response(Certainty::AMBIGUOUS),
        }
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn compute_alias_eq_goal(
        &mut self,
        goal: Goal<'tcx, (ty::Term<'tcx>, ty::Term<'tcx>)>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();

        let evaluate_normalizes_to = |ecx: &mut EvalCtxt<'_, 'tcx>, alias, other| {
            debug!("evaluate_normalizes_to(alias={:?}, other={:?})", alias, other);
            let r = ecx.infcx.probe(|_| {
                let (_, certainty) = ecx.evaluate_goal(goal.with(
                    tcx,
                    ty::Binder::dummy(ty::ProjectionPredicate {
                        projection_ty: alias,
                        term: other,
                    }),
                ))?;
                ecx.make_canonical_response(certainty)
            });
            debug!("evaluate_normalizes_to(..) -> {:?}", r);
            r
        };

        if goal.predicate.0.is_infer() || goal.predicate.1.is_infer() {
            bug!(
                "`AliasEq` goal with an infer var on lhs or rhs which should have been instantiated"
            );
        }

        match (
            goal.predicate.0.to_alias_term_no_opaque(tcx),
            goal.predicate.1.to_alias_term_no_opaque(tcx),
        ) {
            (None, None) => bug!("`AliasEq` goal without an alias on either lhs or rhs"),
            (Some(alias), None) => evaluate_normalizes_to(self, alias, goal.predicate.1),
            (None, Some(alias)) => evaluate_normalizes_to(self, alias, goal.predicate.0),
            (Some(alias_lhs), Some(alias_rhs)) => {
                debug!("compute_alias_eq_goal: both sides are aliases");

                let mut candidates = Vec::with_capacity(3);

                // Evaluate all 3 potential candidates for the alias' being equal
                candidates.push(evaluate_normalizes_to(self, alias_lhs, goal.predicate.1));
                candidates.push(evaluate_normalizes_to(self, alias_rhs, goal.predicate.0));
                candidates.push(self.infcx.probe(|_| {
                    debug!("compute_alias_eq_goal: alias defids are equal, equating substs");
                    let nested_goals = self.infcx.eq(goal.param_env, alias_lhs, alias_rhs)?;
                    self.evaluate_all_and_make_canonical_response(nested_goals)
                }));

                debug!(?candidates);

                self.try_merge_responses(candidates.into_iter())
            }
        }
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    // Recursively evaluates a list of goals to completion, returning the certainty
    // of all of the goals.
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

    // Recursively evaluates a list of goals to completion, making a query response.
    //
    // This is just a convenient way of calling [`EvalCtxt::evaluate_all`],
    // then [`EvalCtxt::make_canonical_response`].
    fn evaluate_all_and_make_canonical_response(
        &mut self,
        goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> QueryResult<'tcx> {
        self.evaluate_all(goals).and_then(|certainty| self.make_canonical_response(certainty))
    }

    fn try_merge_responses(
        &mut self,
        responses: impl Iterator<Item = QueryResult<'tcx>>,
    ) -> QueryResult<'tcx> {
        let candidates = responses.into_iter().flatten().collect::<Box<[_]>>();

        if candidates.is_empty() {
            return Err(NoSolution);
        }

        // FIXME(-Ztreat-solver=next): We should instead try to find a `Certainty::Yes` response with
        // a subset of the constraints that all the other responses have.
        let one = candidates[0];
        if candidates[1..].iter().all(|resp| resp == &one) {
            return Ok(one);
        }

        if let Some(response) = candidates.iter().find(|response| {
            response.value.certainty == Certainty::Yes
                && response.has_no_inference_or_external_constraints()
        }) {
            return Ok(response.clone());
        }

        let certainty = candidates.iter().fold(Certainty::AMBIGUOUS, |certainty, response| {
            certainty.unify_and(response.value.certainty)
        });
        // FIXME(-Ztrait-solver=next): We should take the intersection of the constraints on all the
        // responses and use that for the constraints of this ambiguous response.
        let response = self.make_canonical_response(certainty);
        if let Ok(response) = &response {
            assert!(response.has_no_inference_or_external_constraints());
        }

        response
    }
}

#[instrument(level = "debug", skip(infcx), ret)]
fn compute_external_query_constraints<'tcx>(
    infcx: &InferCtxt<'tcx>,
) -> Result<ExternalConstraints<'tcx>, NoSolution> {
    let region_obligations = infcx.take_registered_region_obligations();
    let opaque_types = infcx.take_opaque_types_for_query_response();
    Ok(infcx.tcx.intern_external_constraints(ExternalConstraintsData {
        // FIXME: Now that's definitely wrong :)
        //
        // Should also do the leak check here I think
        regions: drop(region_obligations),
        opaque_types,
    }))
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
                // FIXME: This to_owned makes me sad, but we should eventually impl
                // `instantiate_query_response_and_region_obligations` separately
                // instead of piggybacking off of the old implementation.
                opaque_types: resp.external_constraints.opaque_types.to_owned(),
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
    Ok(Canonical {
        max_universe: goal.max_universe,
        variables: goal.variables,
        value: Response {
            var_values: CanonicalVarValues::make_identity(tcx, goal.variables),
            // FIXME: maybe we should store the "no response" version in tcx, like
            // we do for tcx.types and stuff.
            external_constraints: tcx
                .intern_external_constraints(ExternalConstraintsData::default()),
            certainty,
        },
    })
}
