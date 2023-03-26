use rustc_hir::def_id::DefId;
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{
    DefineOpaqueTypes, InferCtxt, InferOk, LateBoundRegionConversionTime, TyCtxtInferExt,
};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::solve::{CanonicalGoal, Certainty, MaybeCause, QueryResult};
use rustc_infer::traits::ObligationCause;
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor,
};
use rustc_span::DUMMY_SP;
use std::ops::ControlFlow;

use super::search_graph::{self, OverflowHandler};
use super::SolverMode;
use super::{search_graph::SearchGraph, Goal};

pub struct EvalCtxt<'a, 'tcx> {
    // FIXME: should be private.
    pub(super) infcx: &'a InferCtxt<'tcx>,
    pub(super) var_values: CanonicalVarValues<'tcx>,
    /// The highest universe index nameable by the caller.
    ///
    /// When we enter a new binder inside of the query we create new universes
    /// which the caller cannot name. We have to be careful with variables from
    /// these new universes when creating the query response.
    ///
    /// Both because these new universes can prevent us from reaching a fixpoint
    /// if we have a coinductive cycle and because that's the only way we can return
    /// new placeholders to the caller.
    pub(super) max_input_universe: ty::UniverseIndex,

    pub(super) search_graph: &'a mut SearchGraph<'tcx>,

    pub(super) nested_goals: NestedGoals<'tcx>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum IsNormalizesToHack {
    Yes,
    No,
}

#[derive(Debug, Clone)]
pub(super) struct NestedGoals<'tcx> {
    pub(super) normalizes_to_hack_goal: Option<Goal<'tcx, ty::ProjectionPredicate<'tcx>>>,
    pub(super) goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
}

impl NestedGoals<'_> {
    pub(super) fn new() -> Self {
        Self { normalizes_to_hack_goal: None, goals: Vec::new() }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.normalizes_to_hack_goal.is_none() && self.goals.is_empty()
    }
}

pub trait InferCtxtEvalExt<'tcx> {
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution>;
}

impl<'tcx> InferCtxtEvalExt<'tcx> for InferCtxt<'tcx> {
    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution> {
        let mode = if self.intercrate { SolverMode::Coherence } else { SolverMode::Normal };
        let mut search_graph = search_graph::SearchGraph::new(self.tcx, mode);

        let mut ecx = EvalCtxt {
            search_graph: &mut search_graph,
            infcx: self,
            // Only relevant when canonicalizing the response.
            max_input_universe: ty::UniverseIndex::ROOT,
            var_values: CanonicalVarValues::dummy(),
            nested_goals: NestedGoals::new(),
        };
        let result = ecx.evaluate_goal(IsNormalizesToHack::No, goal);

        assert!(
            ecx.nested_goals.is_empty(),
            "root `EvalCtxt` should not have any goals added to it"
        );

        assert!(search_graph.is_empty());
        result
    }
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    pub(super) fn solver_mode(&self) -> SolverMode {
        self.search_graph.solver_mode()
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
        // Deal with overflow, caching, and coinduction.
        //
        // The actual solver logic happens in `ecx.compute_goal`.
        search_graph.with_new_goal(tcx, canonical_goal, |search_graph| {
            let intercrate = match search_graph.solver_mode() {
                SolverMode::Normal => false,
                SolverMode::Coherence => true,
            };
            let (ref infcx, goal, var_values) = tcx
                .infer_ctxt()
                .intercrate(intercrate)
                .build_with_canonical(DUMMY_SP, &canonical_goal);
            let mut ecx = EvalCtxt {
                infcx,
                var_values,
                max_input_universe: canonical_goal.max_universe,
                search_graph,
                nested_goals: NestedGoals::new(),
            };
            ecx.compute_goal(goal)
        })
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        is_normalizes_to_hack: IsNormalizesToHack,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution> {
        let (orig_values, canonical_goal) = self.canonicalize_goal(goal);
        let canonical_response =
            EvalCtxt::evaluate_canonical_goal(self.tcx(), self.search_graph, canonical_goal)?;

        let has_changed = !canonical_response.value.var_values.is_identity();
        let (certainty, nested_goals) = self.instantiate_and_apply_query_response(
            goal.param_env,
            orig_values,
            canonical_response,
        )?;

        // Check that rerunning this query with its inference constraints applied
        // doesn't result in new inference constraints and has the same result.
        //
        // If we have projection goals like `<T as Trait>::Assoc == u32` we recursively
        // call `exists<U> <T as Trait>::Assoc == U` to enable better caching. This goal
        // could constrain `U` to `u32` which would cause this check to result in a
        // solver cycle.
        if cfg!(debug_assertions)
            && has_changed
            && is_normalizes_to_hack == IsNormalizesToHack::No
            && !self.search_graph.in_cycle()
        {
            debug!("rerunning goal to check result is stable");
            let (_orig_values, canonical_goal) = self.canonicalize_goal(goal);
            let canonical_response =
                EvalCtxt::evaluate_canonical_goal(self.tcx(), self.search_graph, canonical_goal)?;
            if !canonical_response.value.var_values.is_identity() {
                bug!("unstable result: {goal:?} {canonical_goal:?} {canonical_response:?}");
            }
            assert_eq!(certainty, canonical_response.value.certainty);
        }

        Ok((has_changed, certainty, nested_goals))
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
                ty::PredicateKind::Clause(ty::Clause::ConstArgHasType(ct, ty)) => {
                    self.compute_const_arg_has_type_goal(Goal { param_env, predicate: (ct, ty) })
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
                ty::PredicateKind::Ambiguous => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
                // FIXME: implement these predicates :)
                ty::PredicateKind::ConstEvaluatable(_) | ty::PredicateKind::ConstEquate(_, _) => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
                ty::PredicateKind::TypeWellFormedFromEnv(..) => {
                    bug!("TypeWellFormedFromEnv is only used for Chalk")
                }
                ty::PredicateKind::AliasRelate(lhs, rhs, direction) => self
                    .compute_alias_relate_goal(Goal {
                        param_env,
                        predicate: (lhs, rhs, direction),
                    }),
            }
        } else {
            let kind = self.infcx.instantiate_binder_with_placeholders(kind);
            let goal = goal.with(self.tcx(), ty::Binder::dummy(kind));
            self.add_goal(goal);
            self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        }
    }

    // Recursively evaluates all the goals added to this `EvalCtxt` to completion, returning
    // the certainty of all the goals.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn try_evaluate_added_goals(&mut self) -> Result<Certainty, NoSolution> {
        let mut goals = core::mem::replace(&mut self.nested_goals, NestedGoals::new());
        let mut new_goals = NestedGoals::new();

        let response = self.repeat_while_none(
            |_| Ok(Certainty::Maybe(MaybeCause::Overflow)),
            |this| {
                let mut has_changed = Err(Certainty::Yes);

                if let Some(goal) = goals.normalizes_to_hack_goal.take() {
                    let (_, certainty, nested_goals) = match this.evaluate_goal(
                        IsNormalizesToHack::Yes,
                        goal.with(this.tcx(), ty::Binder::dummy(goal.predicate)),
                    ) {
                        Ok(r) => r,
                        Err(NoSolution) => return Some(Err(NoSolution)),
                    };
                    new_goals.goals.extend(nested_goals);

                    if goal.predicate.projection_ty
                        != this.resolve_vars_if_possible(goal.predicate.projection_ty)
                    {
                        has_changed = Ok(())
                    }

                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe(_) => {
                            let goal = this.resolve_vars_if_possible(goal);

                            // The rhs of this `normalizes-to` must always be an unconstrained infer var as it is
                            // the hack used by `normalizes-to` to ensure that every `normalizes-to` behaves the same
                            // regardless of the rhs.
                            //
                            // However it is important not to unconditionally replace the rhs with a new infer var
                            // as otherwise we may replace the original unconstrained infer var with a new infer var
                            // and never propagate any constraints on the new var back to the original var.
                            let term = this
                                .term_is_fully_unconstrained(goal)
                                .then_some(goal.predicate.term)
                                .unwrap_or_else(|| {
                                    this.next_term_infer_of_kind(goal.predicate.term)
                                });
                            let projection_pred = ty::ProjectionPredicate {
                                term,
                                projection_ty: goal.predicate.projection_ty,
                            };
                            new_goals.normalizes_to_hack_goal =
                                Some(goal.with(this.tcx(), projection_pred));

                            has_changed = has_changed.map_err(|c| c.unify_and(certainty));
                        }
                    }
                }

                for nested_goal in goals.goals.drain(..) {
                    let (changed, certainty, nested_goals) =
                        match this.evaluate_goal(IsNormalizesToHack::No, nested_goal) {
                            Ok(result) => result,
                            Err(NoSolution) => return Some(Err(NoSolution)),
                        };
                    new_goals.goals.extend(nested_goals);

                    if changed {
                        has_changed = Ok(());
                    }

                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe(_) => {
                            new_goals.goals.push(nested_goal);
                            has_changed = has_changed.map_err(|c| c.unify_and(certainty));
                        }
                    }
                }

                core::mem::swap(&mut new_goals, &mut goals);
                match has_changed {
                    Ok(()) => None,
                    Err(certainty) => Some(Ok(certainty)),
                }
            },
        );

        self.nested_goals = goals;
        response
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn probe<T>(&mut self, f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> T) -> T {
        let mut ecx = EvalCtxt {
            infcx: self.infcx,
            var_values: self.var_values,
            max_input_universe: self.max_input_universe,
            search_graph: self.search_graph,
            nested_goals: self.nested_goals.clone(),
        };
        self.infcx.probe(|_| f(&mut ecx))
    }

    pub(super) fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub(super) fn next_ty_infer(&self) -> Ty<'tcx> {
        self.infcx.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: DUMMY_SP,
        })
    }

    pub(super) fn next_const_infer(&self, ty: Ty<'tcx>) -> ty::Const<'tcx> {
        self.infcx.next_const_var(
            ty,
            ConstVariableOrigin { kind: ConstVariableOriginKind::MiscVariable, span: DUMMY_SP },
        )
    }

    /// Returns a ty infer or a const infer depending on whether `kind` is a `Ty` or `Const`.
    /// If `kind` is an integer inference variable this will still return a ty infer var.
    pub(super) fn next_term_infer_of_kind(&self, kind: ty::Term<'tcx>) -> ty::Term<'tcx> {
        match kind.unpack() {
            ty::TermKind::Ty(_) => self.next_ty_infer().into(),
            ty::TermKind::Const(ct) => self.next_const_infer(ct.ty()).into(),
        }
    }

    /// Is the projection predicate is of the form `exists<T> <Ty as Trait>::Assoc = T`.
    ///
    /// This is the case if the `term` is an inference variable in the innermost universe
    /// and does not occur in any other part of the predicate.
    pub(super) fn term_is_fully_unconstrained(
        &self,
        goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>,
    ) -> bool {
        let term_is_infer = match goal.predicate.term.unpack() {
            ty::TermKind::Ty(ty) => {
                if let &ty::Infer(ty::TyVar(vid)) = ty.kind() {
                    match self.infcx.probe_ty_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == self.universe(),
                    }
                } else {
                    false
                }
            }
            ty::TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
                    match self.infcx.probe_const_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == self.universe(),
                    }
                } else {
                    false
                }
            }
        };

        // Guard against `<T as Trait<?0>>::Assoc = ?0>`.
        struct ContainsTerm<'a, 'tcx> {
            term: ty::Term<'tcx>,
            infcx: &'a InferCtxt<'tcx>,
        }
        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsTerm<'_, 'tcx> {
            type BreakTy = ();
            fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                if let Some(vid) = t.ty_vid()
                    && let ty::TermKind::Ty(term) = self.term.unpack()
                    && let Some(term_vid) = term.ty_vid()
                    && self.infcx.root_var(vid) == self.infcx.root_var(term_vid)
                {
                    ControlFlow::Break(())
                } else if t.has_non_region_infer() {
                    t.super_visit_with(self)
                } else {
                    ControlFlow::Continue(())
                }
            }

            fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = c.kind()
                    && let ty::TermKind::Const(term) = self.term.unpack()
                    && let ty::ConstKind::Infer(ty::InferConst::Var(term_vid)) = term.kind()
                    && self.infcx.root_const_var(vid) == self.infcx.root_const_var(term_vid)
                {
                    ControlFlow::Break(())
                } else if c.has_non_region_infer() {
                    c.super_visit_with(self)
                } else {
                    ControlFlow::Continue(())
                }
            }
        }

        let mut visitor = ContainsTerm { infcx: self.infcx, term: goal.predicate.term };

        term_is_infer
            && goal.predicate.projection_ty.visit_with(&mut visitor).is_continue()
            && goal.param_env.visit_with(&mut visitor).is_continue()
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    pub(super) fn eq<T: ToTrace<'tcx>>(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<(), NoSolution> {
        self.infcx
            .at(&ObligationCause::dummy(), param_env)
            .eq(DefineOpaqueTypes::No, lhs, rhs)
            .map(|InferOk { value: (), obligations }| {
                self.add_goals(obligations.into_iter().map(|o| o.into()));
            })
            .map_err(|e| {
                debug!(?e, "failed to equate");
                NoSolution
            })
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    pub(super) fn sub<T: ToTrace<'tcx>>(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        sub: T,
        sup: T,
    ) -> Result<(), NoSolution> {
        self.infcx
            .at(&ObligationCause::dummy(), param_env)
            .sub(DefineOpaqueTypes::No, sub, sup)
            .map(|InferOk { value: (), obligations }| {
                self.add_goals(obligations.into_iter().map(|o| o.into()));
            })
            .map_err(|e| {
                debug!(?e, "failed to subtype");
                NoSolution
            })
    }

    /// Equates two values returning the nested goals without adding them
    /// to the nested goals of the `EvalCtxt`.
    ///
    /// If possible, try using `eq` instead which automatically handles nested
    /// goals correctly.
    #[instrument(level = "debug", skip(self, param_env), ret)]
    pub(super) fn eq_and_get_goals<T: ToTrace<'tcx>>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        self.infcx
            .at(&ObligationCause::dummy(), param_env)
            .eq(DefineOpaqueTypes::No, lhs, rhs)
            .map(|InferOk { value: (), obligations }| {
                obligations.into_iter().map(|o| o.into()).collect()
            })
            .map_err(|e| {
                debug!(?e, "failed to equate");
                NoSolution
            })
    }

    pub(super) fn instantiate_binder_with_infer<T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.infcx.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            value,
        )
    }

    pub(super) fn instantiate_binder_with_placeholders<T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.infcx.instantiate_binder_with_placeholders(value)
    }

    pub(super) fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.infcx.resolve_vars_if_possible(value)
    }

    pub(super) fn fresh_substs_for_item(&self, def_id: DefId) -> ty::SubstsRef<'tcx> {
        self.infcx.fresh_substs_for_item(DUMMY_SP, def_id)
    }

    pub(super) fn universe(&self) -> ty::UniverseIndex {
        self.infcx.universe()
    }
}
