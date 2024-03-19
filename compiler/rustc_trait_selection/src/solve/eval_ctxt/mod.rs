use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{
    BoundRegionConversionTime, DefineOpaqueTypes, InferCtxt, InferOk, TyCtxtInferExt,
};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::solve::{MaybeCause, NestedNormalizationGoals};
use rustc_infer::traits::ObligationCause;
use rustc_middle::infer::canonical::CanonicalVarInfos;
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::traits::solve::inspect;
use rustc_middle::traits::solve::{
    CanonicalInput, CanonicalResponse, Certainty, PredefinedOpaques, PredefinedOpaquesData,
    QueryResult,
};
use rustc_middle::traits::specialization_graph;
use rustc_middle::ty::{
    self, InferCtxtLike, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_session::config::DumpSolverProofTree;
use rustc_span::DUMMY_SP;
use std::io::Write;
use std::iter;
use std::ops::ControlFlow;

use crate::traits::vtable::{count_own_vtable_entries, prepare_vtable_segments, VtblSegment};

use super::inspect::ProofTreeBuilder;
use super::{search_graph, GoalEvaluationKind, FIXPOINT_STEP_LIMIT};
use super::{search_graph::SearchGraph, Goal};
use super::{GoalSource, SolverMode};
pub use select::InferCtxtSelectExt;

mod canonical;
mod commit_if_ok;
mod probe;
mod select;

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

    /// The variable info for the `var_values`, only used to make an ambiguous response
    /// with no constraints.
    variables: CanonicalVarInfos<'tcx>,
    /// Whether we're currently computing a `NormalizesTo` goal. Unlike other goals,
    /// `NormalizesTo` goals act like functions with the expected term always being
    /// fully unconstrained. This would weaken inference however, as the nested goals
    /// never get the inference constraints from the actual normalized-to type. Because
    /// of this we return any ambiguous nested goals from `NormalizesTo` to the caller
    /// when then adds these to its own context. The caller is always an `AliasRelate`
    /// goal so this never leaks out of the solver.
    is_normalizes_to_goal: bool,
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

    pub(super) inspect: ProofTreeBuilder<'tcx>,
}

#[derive(Default, Debug, Clone)]
pub(super) struct NestedGoals<'tcx> {
    /// These normalizes-to goals are treated specially during the evaluation
    /// loop. In each iteration we take the RHS of the projection, replace it with
    /// a fresh inference variable, and only after evaluating that goal do we
    /// equate the fresh inference variable with the actual RHS of the predicate.
    ///
    /// This is both to improve caching, and to avoid using the RHS of the
    /// projection predicate to influence the normalizes-to candidate we select.
    ///
    /// Forgetting to replace the RHS with a fresh inference variable when we evaluate
    /// this goal results in an ICE..
    pub(super) normalizes_to_goals: Vec<Goal<'tcx, ty::NormalizesTo<'tcx>>>,
    /// The rest of the goals which have not yet processed or remain ambiguous.
    pub(super) goals: Vec<(GoalSource, Goal<'tcx, ty::Predicate<'tcx>>)>,
}

impl<'tcx> NestedGoals<'tcx> {
    pub(super) fn new() -> Self {
        Self { normalizes_to_goals: Vec::new(), goals: Vec::new() }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.normalizes_to_goals.is_empty() && self.goals.is_empty()
    }

    pub(super) fn extend(&mut self, other: NestedGoals<'tcx>) {
        self.normalizes_to_goals.extend(other.normalizes_to_goals);
        self.goals.extend(other.goals)
    }
}

#[derive(PartialEq, Eq, Debug, Hash, HashStable, Clone, Copy)]
pub enum GenerateProofTree {
    Yes,
    IfEnabled,
    Never,
}

#[extension(pub trait InferCtxtEvalExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        generate_proof_tree: GenerateProofTree,
    ) -> (Result<(bool, Certainty), NoSolution>, Option<inspect::GoalEvaluation<'tcx>>) {
        EvalCtxt::enter_root(self, generate_proof_tree, |ecx| {
            ecx.evaluate_goal(GoalEvaluationKind::Root, GoalSource::Misc, goal)
        })
    }
}

impl<'a, 'tcx> EvalCtxt<'a, 'tcx> {
    pub(super) fn solver_mode(&self) -> SolverMode {
        self.search_graph.solver_mode()
    }

    pub(super) fn set_is_normalizes_to_goal(&mut self) {
        self.is_normalizes_to_goal = true;
    }

    /// Creates a root evaluation context and search graph. This should only be
    /// used from outside of any evaluation, and other methods should be preferred
    /// over using this manually (such as [`InferCtxtEvalExt::evaluate_root_goal`]).
    fn enter_root<R>(
        infcx: &InferCtxt<'tcx>,
        generate_proof_tree: GenerateProofTree,
        f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> R,
    ) -> (R, Option<inspect::GoalEvaluation<'tcx>>) {
        let mode = if infcx.intercrate { SolverMode::Coherence } else { SolverMode::Normal };
        let mut search_graph = search_graph::SearchGraph::new(mode);

        let mut ecx = EvalCtxt {
            infcx,
            search_graph: &mut search_graph,
            nested_goals: NestedGoals::new(),
            inspect: ProofTreeBuilder::new_maybe_root(infcx.tcx, generate_proof_tree),

            // Only relevant when canonicalizing the response,
            // which we don't do within this evaluation context.
            predefined_opaques_in_body: infcx
                .tcx
                .mk_predefined_opaques_in_body(PredefinedOpaquesData::default()),
            max_input_universe: ty::UniverseIndex::ROOT,
            variables: ty::List::empty(),
            var_values: CanonicalVarValues::dummy(),
            is_normalizes_to_goal: false,
            tainted: Ok(()),
        };
        let result = f(&mut ecx);

        let tree = ecx.inspect.finalize();
        if let (Some(tree), DumpSolverProofTree::Always) = (
            &tree,
            infcx.tcx.sess.opts.unstable_opts.next_solver.map(|c| c.dump_tree).unwrap_or_default(),
        ) {
            let mut lock = std::io::stdout().lock();
            let _ = lock.write_fmt(format_args!("{tree:?}\n"));
            let _ = lock.flush();
        }

        assert!(
            ecx.nested_goals.is_empty(),
            "root `EvalCtxt` should not have any goals added to it"
        );

        assert!(search_graph.is_empty());
        (result, tree)
    }

    /// Creates a nested evaluation context that shares the same search graph as the
    /// one passed in. This is suitable for evaluation, granted that the search graph
    /// has had the nested goal recorded on its stack ([`SearchGraph::with_new_goal`]),
    /// but it's preferable to use other methods that call this one rather than this
    /// method directly.
    ///
    /// This function takes care of setting up the inference context, setting the anchor,
    /// and registering opaques from the canonicalized input.
    fn enter_canonical<R>(
        tcx: TyCtxt<'tcx>,
        search_graph: &'a mut search_graph::SearchGraph<'tcx>,
        canonical_input: CanonicalInput<'tcx>,
        canonical_goal_evaluation: &mut ProofTreeBuilder<'tcx>,
        f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>, Goal<'tcx, ty::Predicate<'tcx>>) -> R,
    ) -> R {
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
            variables: canonical_input.variables,
            var_values,
            is_normalizes_to_goal: false,
            predefined_opaques_in_body: input.predefined_opaques_in_body,
            max_input_universe: canonical_input.max_universe,
            search_graph,
            nested_goals: NestedGoals::new(),
            tainted: Ok(()),
            inspect: canonical_goal_evaluation.new_goal_evaluation_step(input),
        };

        for &(key, ty) in &input.predefined_opaques_in_body.opaque_types {
            ecx.insert_hidden_type(key, input.goal.param_env, ty)
                .expect("failed to prepopulate opaque types");
        }

        if !ecx.nested_goals.is_empty() {
            panic!("prepopulating opaque types shouldn't add goals: {:?}", ecx.nested_goals);
        }

        let result = f(&mut ecx, input.goal);

        canonical_goal_evaluation.goal_evaluation_step(ecx.inspect);

        // When creating a query response we clone the opaque type constraints
        // instead of taking them. This would cause an ICE here, since we have
        // assertions against dropping an `InferCtxt` without taking opaques.
        // FIXME: Once we remove support for the old impl we can remove this.
        let _ = infcx.take_opaque_types();

        result
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
        goal_evaluation: &mut ProofTreeBuilder<'tcx>,
    ) -> QueryResult<'tcx> {
        let mut canonical_goal_evaluation =
            goal_evaluation.new_canonical_goal_evaluation(canonical_input);

        // Deal with overflow, caching, and coinduction.
        //
        // The actual solver logic happens in `ecx.compute_goal`.
        let result = ensure_sufficient_stack(|| {
            search_graph.with_new_goal(
                tcx,
                canonical_input,
                &mut canonical_goal_evaluation,
                |search_graph, canonical_goal_evaluation| {
                    EvalCtxt::enter_canonical(
                        tcx,
                        search_graph,
                        canonical_input,
                        canonical_goal_evaluation,
                        |ecx, goal| {
                            let result = ecx.compute_goal(goal);
                            ecx.inspect.query_result(result);
                            result
                        },
                    )
                },
            )
        });

        canonical_goal_evaluation.query_result(result);
        goal_evaluation.canonical_goal_evaluation(canonical_goal_evaluation);
        result
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        goal_evaluation_kind: GoalEvaluationKind,
        source: GoalSource,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(bool, Certainty), NoSolution> {
        let (normalization_nested_goals, has_changed, certainty) =
            self.evaluate_goal_raw(goal_evaluation_kind, source, goal)?;
        assert!(normalization_nested_goals.is_empty());
        Ok((has_changed, certainty))
    }

    /// Recursively evaluates `goal`, returning the nested goals in case
    /// the nested goal is a `NormalizesTo` goal.
    ///
    /// As all other goal kinds do not return any nested goals and
    /// `NormalizesTo` is only used by `AliasRelate`, all other callsites
    /// should use [`EvalCtxt::evaluate_goal`] which discards that empty
    /// storage.
    // FIXME(-Znext-solver=coinduction): `_source` is currently unused but will
    // be necessary once we implement the new coinduction approach.
    fn evaluate_goal_raw(
        &mut self,
        goal_evaluation_kind: GoalEvaluationKind,
        _source: GoalSource,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> Result<(NestedNormalizationGoals<'tcx>, bool, Certainty), NoSolution> {
        let (orig_values, canonical_goal) = self.canonicalize_goal(goal);
        let mut goal_evaluation =
            self.inspect.new_goal_evaluation(goal, &orig_values, goal_evaluation_kind);
        let canonical_response = EvalCtxt::evaluate_canonical_goal(
            self.tcx(),
            self.search_graph,
            canonical_goal,
            &mut goal_evaluation,
        );
        let canonical_response = match canonical_response {
            Err(e) => {
                self.inspect.goal_evaluation(goal_evaluation);
                return Err(e);
            }
            Ok(response) => response,
        };

        let (normalization_nested_goals, certainty, has_changed) = self
            .instantiate_response_discarding_overflow(
                goal.param_env,
                orig_values,
                canonical_response,
            );
        self.inspect.goal_evaluation(goal_evaluation);
        // FIXME: We previously had an assert here that checked that recomputing
        // a goal after applying its constraints did not change its response.
        //
        // This assert was removed as it did not hold for goals constraining
        // an inference variable to a recursive alias, e.g. in
        // tests/ui/traits/next-solver/overflow/recursive-self-normalization.rs.
        //
        // Once we have decided on how to handle trait-system-refactor-initiative#75,
        // we should re-add an assert here.

        Ok((normalization_nested_goals, has_changed, certainty))
    }

    fn instantiate_response_discarding_overflow(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        original_values: Vec<ty::GenericArg<'tcx>>,
        response: CanonicalResponse<'tcx>,
    ) -> (NestedNormalizationGoals<'tcx>, Certainty, bool) {
        if let Certainty::Maybe(MaybeCause::Overflow { .. }) = response.value.certainty {
            return (NestedNormalizationGoals::empty(), response.value.certainty, false);
        }

        let has_changed = !response.value.var_values.is_identity_modulo_regions()
            || !response.value.external_constraints.opaque_types.is_empty();

        let (normalization_nested_goals, certainty) =
            self.instantiate_and_apply_query_response(param_env, original_values, response);
        (normalization_nested_goals, certainty, has_changed)
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
                ty::PredicateKind::ObjectSafe(trait_def_id) => {
                    self.compute_object_safe_goal(trait_def_id)
                }
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => {
                    self.compute_well_formed_goal(Goal { param_env, predicate: arg })
                }
                ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(ct)) => {
                    self.compute_const_evaluatable_goal(Goal { param_env, predicate: ct })
                }
                ty::PredicateKind::ConstEquate(_, _) => {
                    bug!("ConstEquate should not be emitted when `-Znext-solver` is active")
                }
                ty::PredicateKind::NormalizesTo(predicate) => {
                    self.compute_normalizes_to_goal(Goal { param_env, predicate })
                }
                ty::PredicateKind::AliasRelate(lhs, rhs, direction) => self
                    .compute_alias_relate_goal(Goal {
                        param_env,
                        predicate: (lhs, rhs, direction),
                    }),
                ty::PredicateKind::Ambiguous => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
            }
        } else {
            self.infcx.enter_forall(kind, |kind| {
                let goal = goal.with(self.tcx(), ty::Binder::dummy(kind));
                self.add_goal(GoalSource::Misc, goal);
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            })
        }
    }

    // Recursively evaluates all the goals added to this `EvalCtxt` to completion, returning
    // the certainty of all the goals.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn try_evaluate_added_goals(&mut self) -> Result<Certainty, NoSolution> {
        let inspect = self.inspect.new_evaluate_added_goals();
        let inspect = core::mem::replace(&mut self.inspect, inspect);

        let mut response = Ok(Certainty::overflow(false));
        for _ in 0..FIXPOINT_STEP_LIMIT {
            // FIXME: This match is a bit ugly, it might be nice to change the inspect
            // stuff to use a closure instead. which should hopefully simplify this a bit.
            match self.evaluate_added_goals_step() {
                Ok(Some(cert)) => {
                    response = Ok(cert);
                    break;
                }
                Ok(None) => {}
                Err(NoSolution) => {
                    response = Err(NoSolution);
                    break;
                }
            }
        }

        self.inspect.eval_added_goals_result(response);

        if response.is_err() {
            self.tainted = Err(NoSolution);
        }

        let goal_evaluations = std::mem::replace(&mut self.inspect, inspect);
        self.inspect.added_goals_evaluation(goal_evaluations);

        response
    }

    /// Iterate over all added goals: returning `Ok(Some(_))` in case we can stop rerunning.
    ///
    /// Goals for the next step get directly added to the nested goals of the `EvalCtxt`.
    fn evaluate_added_goals_step(&mut self) -> Result<Option<Certainty>, NoSolution> {
        let tcx = self.tcx();
        let mut goals = core::mem::take(&mut self.nested_goals);

        self.inspect.evaluate_added_goals_loop_start();

        fn with_misc_source<'tcx>(
            it: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>,
        ) -> impl Iterator<Item = (GoalSource, Goal<'tcx, ty::Predicate<'tcx>>)> {
            iter::zip(iter::repeat(GoalSource::Misc), it)
        }

        // If this loop did not result in any progress, what's our final certainty.
        let mut unchanged_certainty = Some(Certainty::Yes);
        for goal in goals.normalizes_to_goals {
            // Replace the goal with an unconstrained infer var, so the
            // RHS does not affect projection candidate assembly.
            let unconstrained_rhs = self.next_term_infer_of_kind(goal.predicate.term);
            let unconstrained_goal = goal.with(
                tcx,
                ty::NormalizesTo { alias: goal.predicate.alias, term: unconstrained_rhs },
            );

            let (NestedNormalizationGoals(nested_goals), _, certainty) = self.evaluate_goal_raw(
                GoalEvaluationKind::Nested,
                GoalSource::Misc,
                unconstrained_goal,
            )?;
            // Add the nested goals from normalization to our own nested goals.
            goals.goals.extend(nested_goals);

            // Finally, equate the goal's RHS with the unconstrained var.
            // We put the nested goals from this into goals instead of
            // next_goals to avoid needing to process the loop one extra
            // time if this goal returns something -- I don't think this
            // matters in practice, though.
            let eq_goals =
                self.eq_and_get_goals(goal.param_env, goal.predicate.term, unconstrained_rhs)?;
            goals.goals.extend(with_misc_source(eq_goals));

            // We only look at the `projection_ty` part here rather than
            // looking at the "has changed" return from evaluate_goal,
            // because we expect the `unconstrained_rhs` part of the predicate
            // to have changed -- that means we actually normalized successfully!
            let with_resolved_vars = self.resolve_vars_if_possible(goal);
            if goal.predicate.alias != with_resolved_vars.predicate.alias {
                unchanged_certainty = None;
            }

            match certainty {
                Certainty::Yes => {}
                Certainty::Maybe(_) => {
                    self.nested_goals.normalizes_to_goals.push(with_resolved_vars);
                    unchanged_certainty = unchanged_certainty.map(|c| c.unify_with(certainty));
                }
            }
        }

        for (source, goal) in goals.goals {
            let (has_changed, certainty) =
                self.evaluate_goal(GoalEvaluationKind::Nested, source, goal)?;
            if has_changed {
                unchanged_certainty = None;
            }

            match certainty {
                Certainty::Yes => {}
                Certainty::Maybe(_) => {
                    self.nested_goals.goals.push((source, goal));
                    unchanged_certainty = unchanged_certainty.map(|c| c.unify_with(certainty));
                }
            }
        }

        Ok(unchanged_certainty)
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
    /// This is the case if the `term` does not occur in any other part of the predicate
    /// and is able to name all other placeholder and inference variables.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn term_is_fully_unconstrained(
        &self,
        goal: Goal<'tcx, ty::NormalizesTo<'tcx>>,
    ) -> bool {
        let universe_of_term = match goal.predicate.term.unpack() {
            ty::TermKind::Ty(ty) => {
                if let &ty::Infer(ty::TyVar(vid)) = ty.kind() {
                    self.infcx.universe_of_ty(vid).unwrap()
                } else {
                    return false;
                }
            }
            ty::TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
                    self.infcx.universe_of_ct(vid).unwrap()
                } else {
                    return false;
                }
            }
        };

        struct ContainsTermOrNotNameable<'a, 'tcx> {
            term: ty::Term<'tcx>,
            universe_of_term: ty::UniverseIndex,
            infcx: &'a InferCtxt<'tcx>,
        }

        impl<'a, 'tcx> ContainsTermOrNotNameable<'a, 'tcx> {
            fn check_nameable(&self, universe: ty::UniverseIndex) -> ControlFlow<()> {
                if self.universe_of_term.can_name(universe) {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            }
        }

        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsTermOrNotNameable<'_, 'tcx> {
            type Result = ControlFlow<()>;
            fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                match *t.kind() {
                    ty::Infer(ty::TyVar(vid)) => {
                        if let ty::TermKind::Ty(term) = self.term.unpack()
                            && let Some(term_vid) = term.ty_vid()
                            && self.infcx.root_var(vid) == self.infcx.root_var(term_vid)
                        {
                            ControlFlow::Break(())
                        } else {
                            self.check_nameable(self.infcx.universe_of_ty(vid).unwrap())
                        }
                    }
                    ty::Placeholder(p) => self.check_nameable(p.universe),
                    _ => {
                        if t.has_non_region_infer() || t.has_placeholders() {
                            t.super_visit_with(self)
                        } else {
                            ControlFlow::Continue(())
                        }
                    }
                }
            }

            fn visit_const(&mut self, c: ty::Const<'tcx>) -> Self::Result {
                match c.kind() {
                    ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                        if let ty::TermKind::Const(term) = self.term.unpack()
                            && let ty::ConstKind::Infer(ty::InferConst::Var(term_vid)) = term.kind()
                            && self.infcx.root_const_var(vid) == self.infcx.root_const_var(term_vid)
                        {
                            ControlFlow::Break(())
                        } else {
                            self.check_nameable(self.infcx.universe_of_ct(vid).unwrap())
                        }
                    }
                    ty::ConstKind::Placeholder(p) => self.check_nameable(p.universe),
                    _ => {
                        if c.has_non_region_infer() || c.has_placeholders() {
                            c.super_visit_with(self)
                        } else {
                            ControlFlow::Continue(())
                        }
                    }
                }
            }
        }

        let mut visitor = ContainsTermOrNotNameable {
            infcx: self.infcx,
            universe_of_term,
            term: goal.predicate.term,
        };
        goal.predicate.alias.visit_with(&mut visitor).is_continue()
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
                self.add_goals(GoalSource::Misc, obligations.into_iter().map(|o| o.into()));
            })
            .map_err(|e| {
                debug!(?e, "failed to equate");
                NoSolution
            })
    }

    /// This sohuld only be used when we're either instantiating a previously
    /// unconstrained "return value" or when we're sure that all aliases in
    /// the types are rigid.
    #[instrument(level = "debug", skip(self, param_env), ret)]
    pub(super) fn eq_structurally_relating_aliases<T: ToTrace<'tcx>>(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<(), NoSolution> {
        let cause = ObligationCause::dummy();
        let InferOk { value: (), obligations } = self
            .infcx
            .at(&cause, param_env)
            .trace(lhs, rhs)
            .eq_structurally_relating_aliases(lhs, rhs)?;
        assert!(obligations.is_empty());
        Ok(())
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
                self.add_goals(GoalSource::Misc, obligations.into_iter().map(|o| o.into()));
            })
            .map_err(|e| {
                debug!(?e, "failed to subtype");
                NoSolution
            })
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    pub(super) fn relate<T: ToTrace<'tcx>>(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
    ) -> Result<(), NoSolution> {
        self.infcx
            .at(&ObligationCause::dummy(), param_env)
            .relate(DefineOpaqueTypes::No, lhs, variance, rhs)
            .map(|InferOk { value: (), obligations }| {
                self.add_goals(GoalSource::Misc, obligations.into_iter().map(|o| o.into()));
            })
            .map_err(|e| {
                debug!(?e, "failed to relate");
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
            BoundRegionConversionTime::HigherRankedType,
            value,
        )
    }

    pub(super) fn enter_forall<T: TypeFoldable<TyCtxt<'tcx>> + Copy, U>(
        &self,
        value: ty::Binder<'tcx, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        self.infcx.enter_forall(value, f)
    }
    pub(super) fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.infcx.resolve_vars_if_possible(value)
    }

    pub(super) fn fresh_args_for_item(&self, def_id: DefId) -> ty::GenericArgsRef<'tcx> {
        self.infcx.fresh_args_for_item(DUMMY_SP, def_id)
    }

    pub(super) fn translate_args(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        source_impl: DefId,
        source_args: ty::GenericArgsRef<'tcx>,
        target_node: specialization_graph::Node,
    ) -> ty::GenericArgsRef<'tcx> {
        crate::traits::translate_args(self.infcx, param_env, source_impl, source_args, target_node)
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
        assume: rustc_transmute::Assume,
    ) -> Result<Certainty, NoSolution> {
        use rustc_transmute::Answer;
        // FIXME(transmutability): This really should be returning nested goals for `Answer::If*`
        match rustc_transmute::TransmuteTypeEnv::new(self.infcx).is_transmutable(
            ObligationCause::dummy(),
            src_and_dst,
            assume,
        ) {
            Answer::Yes => Ok(Certainty::Yes),
            Answer::No(_) | Answer::If(_) => Err(NoSolution),
        }
    }

    pub(super) fn can_define_opaque_ty(&self, def_id: LocalDefId) -> bool {
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
            &mut obligations,
        )?;
        self.add_goals(GoalSource::Misc, obligations.into_iter().map(|o| o.into()));
        Ok(())
    }

    pub(super) fn add_item_bounds_for_hidden_type(
        &mut self,
        opaque_def_id: DefId,
        opaque_args: ty::GenericArgsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
    ) {
        let mut obligations = Vec::new();
        self.infcx.add_item_bounds_for_hidden_type(
            opaque_def_id,
            opaque_args,
            ObligationCause::dummy(),
            param_env,
            hidden_ty,
            &mut obligations,
        );
        self.add_goals(GoalSource::Misc, obligations.into_iter().map(|o| o.into()));
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
            values.extend(self.probe_misc_candidate("opaque type storage").enter(|ecx| {
                for (a, b) in std::iter::zip(candidate_key.args, key.args) {
                    ecx.eq(param_env, a, b)?;
                }
                ecx.eq(param_env, candidate_ty, ty)?;
                ecx.add_item_bounds_for_hidden_type(
                    candidate_key.def_id.to_def_id(),
                    candidate_key.args,
                    param_env,
                    candidate_ty,
                );
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }));
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
        match self.infcx.try_const_eval_resolve(param_env, unevaluated, ty, DUMMY_SP) {
            Ok(ct) => Some(ct),
            Err(ErrorHandled::Reported(e, _)) => {
                Some(ty::Const::new_error(self.tcx(), e.into(), ty))
            }
            Err(ErrorHandled::TooGeneric(_)) => None,
        }
    }

    /// Walk through the vtable of a principal trait ref, executing a `supertrait_visitor`
    /// for every trait ref encountered (including the principal). Passes both the vtable
    /// base and the (optional) vptr slot.
    pub(super) fn walk_vtable(
        &mut self,
        principal: ty::PolyTraitRef<'tcx>,
        mut supertrait_visitor: impl FnMut(&mut Self, ty::PolyTraitRef<'tcx>, usize, Option<usize>),
    ) {
        let tcx = self.tcx();
        let mut offset = 0;
        prepare_vtable_segments::<()>(tcx, principal, |segment| {
            match segment {
                VtblSegment::MetadataDSA => {
                    offset += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                }
                VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                    let own_vtable_entries = count_own_vtable_entries(tcx, trait_ref);

                    supertrait_visitor(
                        self,
                        trait_ref,
                        offset,
                        emit_vptr.then(|| offset + own_vtable_entries),
                    );

                    offset += own_vtable_entries;
                    if emit_vptr {
                        offset += 1;
                    }
                }
            }
            ControlFlow::Continue(())
        });
    }
}
