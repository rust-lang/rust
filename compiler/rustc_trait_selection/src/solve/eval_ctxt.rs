use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{
    DefineOpaqueTypes, InferCtxt, InferOk, LateBoundRegionConversionTime, RegionVariableOrigin,
    TyCtxtInferExt,
};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::ObligationCause;
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::traits::solve::inspect::{self, CandidateKind};
use rustc_middle::traits::solve::{
    CanonicalInput, CanonicalResponse, Certainty, IsNormalizesToHack, MaybeCause,
    PredefinedOpaques, PredefinedOpaquesData, QueryResult,
};
use rustc_middle::traits::DefiningAnchor;
use rustc_middle::ty::{
    self, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor,
};
use rustc_span::DUMMY_SP;
use std::ops::ControlFlow;

use crate::traits::specialization_graph;

use super::inspect::ProofTreeBuilder;
use super::search_graph::{self, OverflowHandler};
use super::SolverMode;
use super::{search_graph::SearchGraph, Goal};

mod canonical;
mod probe;

pub struct EvalCtxt<'a, 'tcx> {
    /// The inference context that backs (mostly) inference and placeholder terms
    /// instantiated while solving goals.
    ///
    /// NOTE: The `InferCtxt` that backs the `EvalCtxt` is intentionally private,
    /// because the `InferCtxt` is much more general than `EvalCtxt`. Methods such
    /// as  `take_registered_region_obligations` can mess up query responses,
    /// using `At::normalize` is totally wrong, calling `evaluate_root_goal` can
    /// cause coinductive unsoundness, etc.
    ///
    /// Methods that are generally of use for trait solving are *intentionally*
    /// re-declared through the `EvalCtxt` below, often with cleaner signatures
    /// since we don't care about things like `ObligationCause`s and `Span`s here.
    /// If some `InferCtxt` method is missing, please first think defensively about
    /// the method's compatibility with this solver, or if an existing one does
    /// the job already.
    infcx: &'a InferCtxt<'tcx>,

    pub(super) var_values: CanonicalVarValues<'tcx>,

    predefined_opaques_in_body: PredefinedOpaques<'tcx>,

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

    // Has this `EvalCtxt` errored out with `NoSolution` in `try_evaluate_added_goals`?
    //
    // If so, then it can no longer be used to make a canonical query response,
    // since subsequent calls to `try_evaluate_added_goals` have possibly dropped
    // ambiguous goals. Instead, a probe needs to be introduced somewhere in the
    // evaluation code.
    tainted: Result<(), NoSolution>,

    inspect: ProofTreeBuilder<'tcx>,
}

#[derive(Debug, Clone)]
pub(super) struct NestedGoals<'tcx> {
    /// This normalizes-to goal that is treated specially during the evaluation
    /// loop. In each iteration we take the RHS of the projection, replace it with
    /// a fresh inference variable, and only after evaluating that goal do we
    /// equate the fresh inference variable with the actual RHS of the predicate.
    ///
    /// This is both to improve caching, and to avoid using the RHS of the
    /// projection predicate to influence the normalizes-to candidate we select.
    ///
    /// This is not a 'real' nested goal. We must not forget to replace the RHS
    /// with a fresh inference variable when we evaluate this goal. That can result
    /// in a trait solver cycle. This would currently result in overflow but can be
    /// can be unsound with more powerful coinduction in the future.
    pub(super) normalizes_to_hack_goal: Option<Goal<'tcx, ty::ProjectionPredicate<'tcx>>>,
    /// The rest of the goals which have not yet processed or remain ambiguous.
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

#[derive(PartialEq, Eq, Debug, Hash, HashStable, Clone, Copy)]
pub enum GenerateProofTree {
    Yes,
    No,
}

pub trait InferCtxtEvalExt<'tcx> {
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        generate_proof_tree: GenerateProofTree,
    ) -> (
        Result<(bool, Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution>,
        Option<inspect::GoalEvaluation<'tcx>>,
    );
}

impl<'tcx> InferCtxtEvalExt<'tcx> for InferCtxt<'tcx> {
    #[instrument(level = "debug", skip(self), ret)]
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        generate_proof_tree: GenerateProofTree,
    ) -> (
        Result<(bool, Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution>,
        Option<inspect::GoalEvaluation<'tcx>>,
    ) {
        let mode = if self.intercrate { SolverMode::Coherence } else { SolverMode::Normal };
        let mut search_graph = search_graph::SearchGraph::new(self.tcx, mode);

        let mut ecx = EvalCtxt {
            search_graph: &mut search_graph,
            infcx: self,
            // Only relevant when canonicalizing the response,
            // which we don't do within this evaluation context.
            predefined_opaques_in_body: self
                .tcx
                .mk_predefined_opaques_in_body(PredefinedOpaquesData::default()),
            // Only relevant when canonicalizing the response.
            max_input_universe: ty::UniverseIndex::ROOT,
            var_values: CanonicalVarValues::dummy(),
            nested_goals: NestedGoals::new(),
            tainted: Ok(()),
            inspect: (self.tcx.sess.opts.unstable_opts.dump_solver_proof_tree
                || matches!(generate_proof_tree, GenerateProofTree::Yes))
            .then(ProofTreeBuilder::new_root)
            .unwrap_or_else(ProofTreeBuilder::new_noop),
        };
        let result = ecx.evaluate_goal(IsNormalizesToHack::No, goal);

        let tree = ecx.inspect.finalize();
        if let Some(tree) = &tree {
            // module to allow more granular RUSTC_LOG filtering to just proof tree output
            super::inspect::dump::print_tree(tree);
        }

        assert!(
            ecx.nested_goals.is_empty(),
            "root `EvalCtxt` should not have any goals added to it"
        );

        assert!(search_graph.is_empty());
        (result, tree)
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
    #[instrument(level = "debug", skip(tcx, search_graph, goal_evaluation), ret)]
    fn evaluate_canonical_goal(
        tcx: TyCtxt<'tcx>,
        search_graph: &'a mut search_graph::SearchGraph<'tcx>,
        canonical_input: CanonicalInput<'tcx>,
        mut goal_evaluation: &mut ProofTreeBuilder<'tcx>,
    ) -> QueryResult<'tcx> {
        goal_evaluation.canonicalized_goal(canonical_input);

        // Deal with overflow, caching, and coinduction.
        //
        // The actual solver logic happens in `ecx.compute_goal`.
        search_graph.with_new_goal(
            tcx,
            canonical_input,
            goal_evaluation,
            |search_graph, goal_evaluation| {
                let intercrate = match search_graph.solver_mode() {
                    SolverMode::Normal => false,
                    SolverMode::Coherence => true,
                };
                let (ref infcx, input, var_values) = tcx
                    .infer_ctxt()
                    .intercrate(intercrate)
                    .with_next_trait_solver(true)
                    .with_opaque_type_inference(canonical_input.value.anchor)
                    .build_with_canonical(DUMMY_SP, &canonical_input);

                let mut ecx = EvalCtxt {
                    infcx,
                    var_values,
                    predefined_opaques_in_body: input.predefined_opaques_in_body,
                    max_input_universe: canonical_input.max_universe,
                    search_graph,
                    nested_goals: NestedGoals::new(),
                    tainted: Ok(()),
                    inspect: goal_evaluation.new_goal_evaluation_step(input),
                };

                for &(key, ty) in &input.predefined_opaques_in_body.opaque_types {
                    ecx.insert_hidden_type(key, input.goal.param_env, ty)
                        .expect("failed to prepopulate opaque types");
                }

                if !ecx.nested_goals.is_empty() {
                    panic!(
                        "prepopulating opaque types shouldn't add goals: {:?}",
                        ecx.nested_goals
                    );
                }

                let result = ecx.compute_goal(input.goal);
                ecx.inspect.query_result(result);
                goal_evaluation.goal_evaluation_step(ecx.inspect);

                // When creating a query response we clone the opaque type constraints
                // instead of taking them. This would cause an ICE here, since we have
                // assertions against dropping an `InferCtxt` without taking opaques.
                // FIXME: Once we remove support for the old impl we can remove this.
                if input.anchor != DefiningAnchor::Error {
                    let _ = infcx.take_opaque_types();
                }

                result
            },
        )
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        is_normalizes_to_hack: IsNormalizesToHack,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution> {
        let (orig_values, canonical_goal) = self.canonicalize_goal(goal);
        let mut goal_evaluation = self.inspect.new_goal_evaluation(goal, is_normalizes_to_hack);
        let canonical_response = EvalCtxt::evaluate_canonical_goal(
            self.tcx(),
            self.search_graph,
            canonical_goal,
            &mut goal_evaluation,
        );
        goal_evaluation.query_result(canonical_response);
        let canonical_response = match canonical_response {
            Err(e) => {
                self.inspect.goal_evaluation(goal_evaluation);
                return Err(e);
            }
            Ok(response) => response,
        };

        let has_changed = !canonical_response.value.var_values.is_identity()
            || !canonical_response.value.external_constraints.opaque_types.is_empty();
        let (certainty, nested_goals) = match self.instantiate_and_apply_query_response(
            goal.param_env,
            orig_values,
            canonical_response,
        ) {
            Err(e) => {
                self.inspect.goal_evaluation(goal_evaluation);
                return Err(e);
            }
            Ok(response) => response,
        };
        goal_evaluation.returned_goals(&nested_goals);
        self.inspect.goal_evaluation(goal_evaluation);

        if !has_changed && !nested_goals.is_empty() {
            bug!("an unchanged goal shouldn't have any side-effects on instantiation");
        }

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
            let new_canonical_response = EvalCtxt::evaluate_canonical_goal(
                self.tcx(),
                self.search_graph,
                canonical_goal,
                // FIXME(-Ztrait-solver=next): we do not track what happens in `evaluate_canonical_goal`
                &mut ProofTreeBuilder::new_noop(),
            )?;
            // We only check for modulo regions as we convert all regions in
            // the input to new existentials, even if they're expected to be
            // `'static` or a placeholder region.
            if !new_canonical_response.value.var_values.is_identity_modulo_regions() {
                bug!(
                    "unstable result: re-canonicalized goal={canonical_goal:#?} \
                    first_response={canonical_response:#?} \
                    second_response={new_canonical_response:#?}"
                );
            }
            if certainty != new_canonical_response.value.certainty {
                bug!(
                    "unstable certainty: {certainty:#?} re-canonicalized goal={canonical_goal:#?} \
                     first_response={canonical_response:#?} \
                     second_response={new_canonical_response:#?}"
                );
            }
        }

        Ok((has_changed, certainty, nested_goals))
    }

    fn compute_goal(&mut self, goal: Goal<'tcx, ty::Predicate<'tcx>>) -> QueryResult<'tcx> {
        let Goal { param_env, predicate } = goal;
        let kind = predicate.kind();
        if let Some(kind) = kind.no_bound_vars() {
            match kind {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)) => {
                    self.compute_trait_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(predicate)) => {
                    self.compute_projection_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(predicate)) => {
                    self.compute_type_outlives_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(predicate)) => {
                    self.compute_region_outlives_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
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
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => {
                    self.compute_well_formed_goal(Goal { param_env, predicate: arg })
                }
                ty::PredicateKind::Ambiguous => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
                // FIXME: implement this predicate :)
                ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(_)) => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
                ty::PredicateKind::ConstEquate(_, _) => {
                    bug!("ConstEquate should not be emitted when `-Ztrait-solver=next` is active")
                }
                ty::PredicateKind::Clause(ty::ClauseKind::TypeWellFormedFromEnv(..)) => {
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
        let inspect = self.inspect.new_evaluate_added_goals();
        let inspect = core::mem::replace(&mut self.inspect, inspect);

        let mut goals = core::mem::replace(&mut self.nested_goals, NestedGoals::new());
        let mut new_goals = NestedGoals::new();

        let response = self.repeat_while_none(
            |_| Ok(Certainty::Maybe(MaybeCause::Overflow)),
            |this| {
                this.inspect.evaluate_added_goals_loop_start();

                let mut has_changed = Err(Certainty::Yes);

                if let Some(goal) = goals.normalizes_to_hack_goal.take() {
                    // Replace the goal with an unconstrained infer var, so the
                    // RHS does not affect projection candidate assembly.
                    let unconstrained_rhs = this.next_term_infer_of_kind(goal.predicate.term);
                    let unconstrained_goal = goal.with(
                        this.tcx(),
                        ty::Binder::dummy(ty::ProjectionPredicate {
                            projection_ty: goal.predicate.projection_ty,
                            term: unconstrained_rhs,
                        }),
                    );

                    let (_, certainty, instantiate_goals) =
                        match this.evaluate_goal(IsNormalizesToHack::Yes, unconstrained_goal) {
                            Ok(r) => r,
                            Err(NoSolution) => return Some(Err(NoSolution)),
                        };
                    new_goals.goals.extend(instantiate_goals);

                    // Finally, equate the goal's RHS with the unconstrained var.
                    // We put the nested goals from this into goals instead of
                    // next_goals to avoid needing to process the loop one extra
                    // time if this goal returns something -- I don't think this
                    // matters in practice, though.
                    match this.eq_and_get_goals(
                        goal.param_env,
                        goal.predicate.term,
                        unconstrained_rhs,
                    ) {
                        Ok(eq_goals) => {
                            goals.goals.extend(eq_goals);
                        }
                        Err(NoSolution) => return Some(Err(NoSolution)),
                    };

                    // We only look at the `projection_ty` part here rather than
                    // looking at the "has changed" return from evaluate_goal,
                    // because we expect the `unconstrained_rhs` part of the predicate
                    // to have changed -- that means we actually normalized successfully!
                    if goal.predicate.projection_ty
                        != this.resolve_vars_if_possible(goal.predicate.projection_ty)
                    {
                        has_changed = Ok(())
                    }

                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe(_) => {
                            // We need to resolve vars here so that we correctly
                            // deal with `has_changed` in the next iteration.
                            new_goals.normalizes_to_hack_goal =
                                Some(this.resolve_vars_if_possible(goal));
                            has_changed = has_changed.map_err(|c| c.unify_with(certainty));
                        }
                    }
                }

                for goal in goals.goals.drain(..) {
                    let (changed, certainty, instantiate_goals) =
                        match this.evaluate_goal(IsNormalizesToHack::No, goal) {
                            Ok(result) => result,
                            Err(NoSolution) => return Some(Err(NoSolution)),
                        };
                    new_goals.goals.extend(instantiate_goals);

                    if changed {
                        has_changed = Ok(());
                    }

                    match certainty {
                        Certainty::Yes => {}
                        Certainty::Maybe(_) => {
                            new_goals.goals.push(goal);
                            has_changed = has_changed.map_err(|c| c.unify_with(certainty));
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

        self.inspect.eval_added_goals_result(response);

        if response.is_err() {
            self.tainted = Err(NoSolution);
        }

        let goal_evaluations = std::mem::replace(&mut self.inspect, inspect);
        self.inspect.added_goals_evaluation(goal_evaluations);

        self.nested_goals = goals;
        response
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub(super) fn next_ty_infer(&self) -> Ty<'tcx> {
        self.infcx.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: DUMMY_SP,
        })
    }

    pub(super) fn next_region_infer(&self) -> ty::Region<'tcx> {
        self.infcx.next_region_var(RegionVariableOrigin::MiscVariable(DUMMY_SP))
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
                        Err(universe) => universe == self.infcx.universe(),
                    }
                } else {
                    false
                }
            }
            ty::TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
                    match self.infcx.probe_const_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == self.infcx.universe(),
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
    #[instrument(level = "trace", skip(self, param_env), ret)]
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

    pub(super) fn translate_substs(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        source_impl: DefId,
        source_substs: ty::SubstsRef<'tcx>,
        target_node: specialization_graph::Node,
    ) -> ty::SubstsRef<'tcx> {
        crate::traits::translate_substs(
            self.infcx,
            param_env,
            source_impl,
            source_substs,
            target_node,
        )
    }

    pub(super) fn register_ty_outlives(&self, ty: Ty<'tcx>, lt: ty::Region<'tcx>) {
        self.infcx.register_region_obligation_with_cause(ty, lt, &ObligationCause::dummy());
    }

    pub(super) fn register_region_outlives(&self, a: ty::Region<'tcx>, b: ty::Region<'tcx>) {
        // `b : a` ==> `a <= b`
        // (inlined from `InferCtxt::region_outlives_predicate`)
        self.infcx.sub_regions(
            rustc_infer::infer::SubregionOrigin::RelateRegionParamBound(DUMMY_SP),
            b,
            a,
        );
    }

    /// Computes the list of goals required for `arg` to be well-formed
    pub(super) fn well_formed_goals(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        arg: ty::GenericArg<'tcx>,
    ) -> Option<impl Iterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>> {
        crate::traits::wf::unnormalized_obligations(self.infcx, param_env, arg)
            .map(|obligations| obligations.into_iter().map(|obligation| obligation.into()))
    }

    pub(super) fn is_transmutable(
        &self,
        src_and_dst: rustc_transmute::Types<'tcx>,
        scope: Ty<'tcx>,
        assume: rustc_transmute::Assume,
    ) -> Result<Certainty, NoSolution> {
        use rustc_transmute::Answer;
        // FIXME(transmutability): This really should be returning nested goals for `Answer::If*`
        match rustc_transmute::TransmuteTypeEnv::new(self.infcx).is_transmutable(
            ObligationCause::dummy(),
            src_and_dst,
            scope,
            assume,
        ) {
            Answer::Yes => Ok(Certainty::Yes),
            Answer::No(_) | Answer::If(_) => Err(NoSolution),
        }
    }

    pub(super) fn can_define_opaque_ty(&mut self, def_id: LocalDefId) -> bool {
        self.infcx.opaque_type_origin(def_id).is_some()
    }

    pub(super) fn insert_hidden_type(
        &mut self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
    ) -> Result<(), NoSolution> {
        let mut obligations = Vec::new();
        self.infcx.insert_hidden_type(
            opaque_type_key,
            &ObligationCause::dummy(),
            param_env,
            hidden_ty,
            true,
            &mut obligations,
        )?;
        self.add_goals(obligations.into_iter().map(|o| o.into()));
        Ok(())
    }

    pub(super) fn add_item_bounds_for_hidden_type(
        &mut self,
        opaque_def_id: DefId,
        opaque_substs: ty::SubstsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
    ) {
        let mut obligations = Vec::new();
        self.infcx.add_item_bounds_for_hidden_type(
            opaque_def_id,
            opaque_substs,
            ObligationCause::dummy(),
            param_env,
            hidden_ty,
            &mut obligations,
        );
        self.add_goals(obligations.into_iter().map(|o| o.into()));
    }

    // Do something for each opaque/hidden pair defined with `def_id` in the
    // current inference context.
    pub(super) fn unify_existing_opaque_tys(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        key: ty::OpaqueTypeKey<'tcx>,
        ty: Ty<'tcx>,
    ) -> Vec<CanonicalResponse<'tcx>> {
        // FIXME: Super inefficient to be cloning this...
        let opaques = self.infcx.clone_opaque_types_for_query_response();

        let mut values = vec![];
        for (candidate_key, candidate_ty) in opaques {
            if candidate_key.def_id != key.def_id {
                continue;
            }
            values.extend(
                self.probe(|r| CandidateKind::Candidate {
                    name: "opaque type storage".into(),
                    result: *r,
                })
                .enter(|ecx| {
                    for (a, b) in std::iter::zip(candidate_key.substs, key.substs) {
                        ecx.eq(param_env, a, b)?;
                    }
                    ecx.eq(param_env, candidate_ty, ty)?;
                    ecx.add_item_bounds_for_hidden_type(
                        candidate_key.def_id.to_def_id(),
                        candidate_key.substs,
                        param_env,
                        candidate_ty,
                    );
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }),
            );
        }
        values
    }

    // Try to evaluate a const, or return `None` if the const is too generic.
    // This doesn't mean the const isn't evaluatable, though, and should be treated
    // as an ambiguity rather than no-solution.
    pub(super) fn try_const_eval_resolve(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        unevaluated: ty::UnevaluatedConst<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<ty::Const<'tcx>> {
        use rustc_middle::mir::interpret::ErrorHandled;
        match self.infcx.try_const_eval_resolve(param_env, unevaluated, ty, None) {
            Ok(ct) => Some(ct),
            Err(ErrorHandled::Reported(e)) => Some(self.tcx().const_error(ty, e.into())),
            Err(ErrorHandled::TooGeneric) => None,
        }
    }
}
