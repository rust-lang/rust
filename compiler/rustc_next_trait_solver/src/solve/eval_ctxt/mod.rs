use std::ops::ControlFlow;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir::data_structures::ensure_sufficient_stack;
use rustc_type_ir::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_type_ir::inherent::*;
use rustc_type_ir::relate::Relate;
use rustc_type_ir::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use rustc_type_ir::{self as ty, CanonicalVarValues, InferCtxtLike, Interner};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};
use tracing::{instrument, trace};

use crate::coherence;
use crate::delegate::SolverDelegate;
use crate::solve::inspect::{self, ProofTreeBuilder};
use crate::solve::search_graph::SearchGraph;
use crate::solve::{
    CanonicalInput, CanonicalResponse, Certainty, Goal, GoalEvaluationKind, GoalSource, MaybeCause,
    NestedNormalizationGoals, NoSolution, PredefinedOpaquesData, QueryResult, SolverMode,
    FIXPOINT_STEP_LIMIT,
};

pub(super) mod canonical;
mod probe;

pub struct EvalCtxt<'a, D, I = <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
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
    delegate: &'a D,

    /// The variable info for the `var_values`, only used to make an ambiguous response
    /// with no constraints.
    variables: I::CanonicalVars,
    /// Whether we're currently computing a `NormalizesTo` goal. Unlike other goals,
    /// `NormalizesTo` goals act like functions with the expected term always being
    /// fully unconstrained. This would weaken inference however, as the nested goals
    /// never get the inference constraints from the actual normalized-to type. Because
    /// of this we return any ambiguous nested goals from `NormalizesTo` to the caller
    /// when then adds these to its own context. The caller is always an `AliasRelate`
    /// goal so this never leaks out of the solver.
    is_normalizes_to_goal: bool,
    pub(super) var_values: CanonicalVarValues<I>,

    predefined_opaques_in_body: I::PredefinedOpaques,

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

    pub(super) search_graph: &'a mut SearchGraph<D>,

    nested_goals: NestedGoals<I>,

    // Has this `EvalCtxt` errored out with `NoSolution` in `try_evaluate_added_goals`?
    //
    // If so, then it can no longer be used to make a canonical query response,
    // since subsequent calls to `try_evaluate_added_goals` have possibly dropped
    // ambiguous goals. Instead, a probe needs to be introduced somewhere in the
    // evaluation code.
    tainted: Result<(), NoSolution>,

    pub(super) inspect: ProofTreeBuilder<D>,
}

#[derive_where(Clone, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
// FIXME: This can be made crate-private once `EvalCtxt` also lives in this crate.
pub struct NestedGoals<I: Interner> {
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
    pub normalizes_to_goals: Vec<Goal<I, ty::NormalizesTo<I>>>,
    /// The rest of the goals which have not yet processed or remain ambiguous.
    pub goals: Vec<(GoalSource, Goal<I, I::Predicate>)>,
}

impl<I: Interner> NestedGoals<I> {
    pub fn new() -> Self {
        Self { normalizes_to_goals: Vec::new(), goals: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.normalizes_to_goals.is_empty() && self.goals.is_empty()
    }
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum GenerateProofTree {
    Yes,
    No,
}

pub trait SolverDelegateEvalExt: SolverDelegate {
    fn evaluate_root_goal(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
        generate_proof_tree: GenerateProofTree,
    ) -> (Result<(bool, Certainty), NoSolution>, Option<inspect::GoalEvaluation<Self::Interner>>);

    // FIXME: This is only exposed because we need to use it in `analyse.rs`
    // which is not yet uplifted. Once that's done, we should remove this.
    fn evaluate_root_goal_raw(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
        generate_proof_tree: GenerateProofTree,
    ) -> (
        Result<(NestedNormalizationGoals<Self::Interner>, bool, Certainty), NoSolution>,
        Option<inspect::GoalEvaluation<Self::Interner>>,
    );
}

impl<D, I> SolverDelegateEvalExt for D
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal(
        &self,
        goal: Goal<I, I::Predicate>,
        generate_proof_tree: GenerateProofTree,
    ) -> (Result<(bool, Certainty), NoSolution>, Option<inspect::GoalEvaluation<I>>) {
        EvalCtxt::enter_root(self, generate_proof_tree, |ecx| {
            ecx.evaluate_goal(GoalEvaluationKind::Root, GoalSource::Misc, goal)
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal_raw(
        &self,
        goal: Goal<I, I::Predicate>,
        generate_proof_tree: GenerateProofTree,
    ) -> (
        Result<(NestedNormalizationGoals<I>, bool, Certainty), NoSolution>,
        Option<inspect::GoalEvaluation<I>>,
    ) {
        EvalCtxt::enter_root(self, generate_proof_tree, |ecx| {
            ecx.evaluate_goal_raw(GoalEvaluationKind::Root, GoalSource::Misc, goal)
        })
    }
}

impl<'a, D, I> EvalCtxt<'a, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn solver_mode(&self) -> SolverMode {
        self.search_graph.solver_mode()
    }

    pub(super) fn set_is_normalizes_to_goal(&mut self) {
        self.is_normalizes_to_goal = true;
    }

    /// Creates a root evaluation context and search graph. This should only be
    /// used from outside of any evaluation, and other methods should be preferred
    /// over using this manually (such as [`SolverDelegateEvalExt::evaluate_root_goal`]).
    pub(super) fn enter_root<R>(
        delegate: &D,
        generate_proof_tree: GenerateProofTree,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> R,
    ) -> (R, Option<inspect::GoalEvaluation<I>>) {
        let mut search_graph = SearchGraph::new(delegate.solver_mode());

        let mut ecx = EvalCtxt {
            delegate,
            search_graph: &mut search_graph,
            nested_goals: NestedGoals::new(),
            inspect: ProofTreeBuilder::new_maybe_root(generate_proof_tree),

            // Only relevant when canonicalizing the response,
            // which we don't do within this evaluation context.
            predefined_opaques_in_body: delegate
                .cx()
                .mk_predefined_opaques_in_body(PredefinedOpaquesData::default()),
            max_input_universe: ty::UniverseIndex::ROOT,
            variables: Default::default(),
            var_values: CanonicalVarValues::dummy(),
            is_normalizes_to_goal: false,
            tainted: Ok(()),
        };
        let result = f(&mut ecx);

        let proof_tree = ecx.inspect.finalize();
        assert!(
            ecx.nested_goals.is_empty(),
            "root `EvalCtxt` should not have any goals added to it"
        );

        assert!(search_graph.is_empty());
        (result, proof_tree)
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
        cx: I,
        search_graph: &'a mut SearchGraph<D>,
        canonical_input: CanonicalInput<I>,
        canonical_goal_evaluation: &mut ProofTreeBuilder<D>,
        f: impl FnOnce(&mut EvalCtxt<'_, D>, Goal<I, I::Predicate>) -> R,
    ) -> R {
        let (ref delegate, input, var_values) =
            SolverDelegate::build_with_canonical(cx, search_graph.solver_mode(), &canonical_input);

        let mut ecx = EvalCtxt {
            delegate,
            variables: canonical_input.variables,
            var_values,
            is_normalizes_to_goal: false,
            predefined_opaques_in_body: input.predefined_opaques_in_body,
            max_input_universe: canonical_input.max_universe,
            search_graph,
            nested_goals: NestedGoals::new(),
            tainted: Ok(()),
            inspect: canonical_goal_evaluation.new_goal_evaluation_step(var_values, input),
        };

        for &(key, ty) in &input.predefined_opaques_in_body.opaque_types {
            ecx.delegate.inject_new_hidden_type_unchecked(key, ty);
        }

        if !ecx.nested_goals.is_empty() {
            panic!("prepopulating opaque types shouldn't add goals: {:?}", ecx.nested_goals);
        }

        let result = f(&mut ecx, input.goal);
        ecx.inspect.probe_final_state(ecx.delegate, ecx.max_input_universe);
        canonical_goal_evaluation.goal_evaluation_step(ecx.inspect);

        // When creating a query response we clone the opaque type constraints
        // instead of taking them. This would cause an ICE here, since we have
        // assertions against dropping an `InferCtxt` without taking opaques.
        // FIXME: Once we remove support for the old impl we can remove this.
        // FIXME: Could we make `build_with_canonical` into `enter_with_canonical` and call this at the end?
        delegate.reset_opaque_types();

        result
    }

    /// The entry point of the solver.
    ///
    /// This function deals with (coinductive) cycles, overflow, and caching
    /// and then calls [`EvalCtxt::compute_goal`] which contains the actual
    /// logic of the solver.
    ///
    /// Instead of calling this function directly, use either [EvalCtxt::evaluate_goal]
    /// if you're inside of the solver or [SolverDelegateEvalExt::evaluate_root_goal] if you're
    /// outside of it.
    #[instrument(level = "debug", skip(cx, search_graph, goal_evaluation), ret)]
    fn evaluate_canonical_goal(
        cx: I,
        search_graph: &'a mut SearchGraph<D>,
        canonical_input: CanonicalInput<I>,
        goal_evaluation: &mut ProofTreeBuilder<D>,
    ) -> QueryResult<I> {
        let mut canonical_goal_evaluation =
            goal_evaluation.new_canonical_goal_evaluation(canonical_input);

        // Deal with overflow, caching, and coinduction.
        //
        // The actual solver logic happens in `ecx.compute_goal`.
        let result = ensure_sufficient_stack(|| {
            search_graph.with_new_goal(
                cx,
                canonical_input,
                &mut canonical_goal_evaluation,
                |search_graph, canonical_goal_evaluation| {
                    EvalCtxt::enter_canonical(
                        cx,
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
        goal: Goal<I, I::Predicate>,
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
    pub(super) fn evaluate_goal_raw(
        &mut self,
        goal_evaluation_kind: GoalEvaluationKind,
        _source: GoalSource,
        goal: Goal<I, I::Predicate>,
    ) -> Result<(NestedNormalizationGoals<I>, bool, Certainty), NoSolution> {
        let (orig_values, canonical_goal) = self.canonicalize_goal(goal);
        let mut goal_evaluation =
            self.inspect.new_goal_evaluation(goal, &orig_values, goal_evaluation_kind);
        let canonical_response = EvalCtxt::evaluate_canonical_goal(
            self.cx(),
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
        param_env: I::ParamEnv,
        original_values: Vec<I::GenericArg>,
        response: CanonicalResponse<I>,
    ) -> (NestedNormalizationGoals<I>, Certainty, bool) {
        if let Certainty::Maybe(MaybeCause::Overflow { .. }) = response.value.certainty {
            return (NestedNormalizationGoals::empty(), response.value.certainty, false);
        }

        let has_changed = !response.value.var_values.is_identity_modulo_regions()
            || !response.value.external_constraints.opaque_types.is_empty();

        let (normalization_nested_goals, certainty) =
            self.instantiate_and_apply_query_response(param_env, original_values, response);
        (normalization_nested_goals, certainty, has_changed)
    }

    fn compute_goal(&mut self, goal: Goal<I, I::Predicate>) -> QueryResult<I> {
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
                    panic!("ConstEquate should not be emitted when `-Znext-solver` is active")
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
            self.delegate.enter_forall(kind, |kind| {
                let goal = goal.with(self.cx(), ty::Binder::dummy(kind));
                self.add_goal(GoalSource::InstantiateHigherRanked, goal);
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            })
        }
    }

    // Recursively evaluates all the goals added to this `EvalCtxt` to completion, returning
    // the certainty of all the goals.
    #[instrument(level = "trace", skip(self))]
    pub(super) fn try_evaluate_added_goals(&mut self) -> Result<Certainty, NoSolution> {
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

        if response.is_err() {
            self.tainted = Err(NoSolution);
        }

        response
    }

    /// Iterate over all added goals: returning `Ok(Some(_))` in case we can stop rerunning.
    ///
    /// Goals for the next step get directly added to the nested goals of the `EvalCtxt`.
    fn evaluate_added_goals_step(&mut self) -> Result<Option<Certainty>, NoSolution> {
        let cx = self.cx();
        let mut goals = core::mem::take(&mut self.nested_goals);

        // If this loop did not result in any progress, what's our final certainty.
        let mut unchanged_certainty = Some(Certainty::Yes);
        for goal in goals.normalizes_to_goals {
            // Replace the goal with an unconstrained infer var, so the
            // RHS does not affect projection candidate assembly.
            let unconstrained_rhs = self.next_term_infer_of_kind(goal.predicate.term);
            let unconstrained_goal = goal.with(
                cx,
                ty::NormalizesTo { alias: goal.predicate.alias, term: unconstrained_rhs },
            );

            let (NestedNormalizationGoals(nested_goals), _, certainty) = self.evaluate_goal_raw(
                GoalEvaluationKind::Nested,
                GoalSource::Misc,
                unconstrained_goal,
            )?;
            // Add the nested goals from normalization to our own nested goals.
            trace!(?nested_goals);
            goals.goals.extend(nested_goals);

            // Finally, equate the goal's RHS with the unconstrained var.
            //
            // SUBTLE:
            // We structurally relate aliases here. This is necessary
            // as we otherwise emit a nested `AliasRelate` goal in case the
            // returned term is a rigid alias, resulting in overflow.
            //
            // It is correct as both `goal.predicate.term` and `unconstrained_rhs`
            // start out as an unconstrained inference variable so any aliases get
            // fully normalized when instantiating it.
            //
            // FIXME: Strictly speaking this may be incomplete if the normalized-to
            // type contains an ambiguous alias referencing bound regions. We should
            // consider changing this to only use "shallow structural equality".
            self.eq_structurally_relating_aliases(
                goal.param_env,
                goal.predicate.term,
                unconstrained_rhs,
            )?;

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

    /// Record impl args in the proof tree for later access by `InspectCandidate`.
    pub(crate) fn record_impl_args(&mut self, impl_args: I::GenericArgs) {
        self.inspect.record_impl_args(self.delegate, self.max_input_universe, impl_args)
    }

    pub(super) fn cx(&self) -> I {
        self.delegate.cx()
    }

    #[instrument(level = "trace", skip(self))]
    pub(super) fn add_normalizes_to_goal(&mut self, mut goal: Goal<I, ty::NormalizesTo<I>>) {
        goal.predicate = goal
            .predicate
            .fold_with(&mut ReplaceAliasWithInfer { ecx: self, param_env: goal.param_env });
        self.inspect.add_normalizes_to_goal(self.delegate, self.max_input_universe, goal);
        self.nested_goals.normalizes_to_goals.push(goal);
    }

    #[instrument(level = "debug", skip(self))]
    pub(super) fn add_goal(&mut self, source: GoalSource, mut goal: Goal<I, I::Predicate>) {
        goal.predicate = goal
            .predicate
            .fold_with(&mut ReplaceAliasWithInfer { ecx: self, param_env: goal.param_env });
        self.inspect.add_goal(self.delegate, self.max_input_universe, source, goal);
        self.nested_goals.goals.push((source, goal));
    }

    #[instrument(level = "trace", skip(self, goals))]
    pub(super) fn add_goals(
        &mut self,
        source: GoalSource,
        goals: impl IntoIterator<Item = Goal<I, I::Predicate>>,
    ) {
        for goal in goals {
            self.add_goal(source, goal);
        }
    }

    pub(super) fn next_ty_infer(&mut self) -> I::Ty {
        let ty = self.delegate.next_ty_infer();
        self.inspect.add_var_value(ty);
        ty
    }

    pub(super) fn next_const_infer(&mut self) -> I::Const {
        let ct = self.delegate.next_const_infer();
        self.inspect.add_var_value(ct);
        ct
    }

    /// Returns a ty infer or a const infer depending on whether `kind` is a `Ty` or `Const`.
    /// If `kind` is an integer inference variable this will still return a ty infer var.
    pub(super) fn next_term_infer_of_kind(&mut self, kind: I::Term) -> I::Term {
        match kind.kind() {
            ty::TermKind::Ty(_) => self.next_ty_infer().into(),
            ty::TermKind::Const(_) => self.next_const_infer().into(),
        }
    }

    /// Is the projection predicate is of the form `exists<T> <Ty as Trait>::Assoc = T`.
    ///
    /// This is the case if the `term` does not occur in any other part of the predicate
    /// and is able to name all other placeholder and inference variables.
    #[instrument(level = "trace", skip(self), ret)]
    pub(super) fn term_is_fully_unconstrained(&self, goal: Goal<I, ty::NormalizesTo<I>>) -> bool {
        let universe_of_term = match goal.predicate.term.kind() {
            ty::TermKind::Ty(ty) => {
                if let ty::Infer(ty::TyVar(vid)) = ty.kind() {
                    self.delegate.universe_of_ty(vid).unwrap()
                } else {
                    return false;
                }
            }
            ty::TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
                    self.delegate.universe_of_ct(vid).unwrap()
                } else {
                    return false;
                }
            }
        };

        struct ContainsTermOrNotNameable<'a, D: SolverDelegate<Interner = I>, I: Interner> {
            term: I::Term,
            universe_of_term: ty::UniverseIndex,
            delegate: &'a D,
        }

        impl<D: SolverDelegate<Interner = I>, I: Interner> ContainsTermOrNotNameable<'_, D, I> {
            fn check_nameable(&self, universe: ty::UniverseIndex) -> ControlFlow<()> {
                if self.universe_of_term.can_name(universe) {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            }
        }

        impl<D: SolverDelegate<Interner = I>, I: Interner> TypeVisitor<I>
            for ContainsTermOrNotNameable<'_, D, I>
        {
            type Result = ControlFlow<()>;
            fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
                match t.kind() {
                    ty::Infer(ty::TyVar(vid)) => {
                        if let ty::TermKind::Ty(term) = self.term.kind() {
                            if let ty::Infer(ty::TyVar(term_vid)) = term.kind() {
                                if self.delegate.root_ty_var(vid)
                                    == self.delegate.root_ty_var(term_vid)
                                {
                                    return ControlFlow::Break(());
                                }
                            }
                        }

                        self.check_nameable(self.delegate.universe_of_ty(vid).unwrap())
                    }
                    ty::Placeholder(p) => self.check_nameable(p.universe()),
                    _ => {
                        if t.has_non_region_infer() || t.has_placeholders() {
                            t.super_visit_with(self)
                        } else {
                            ControlFlow::Continue(())
                        }
                    }
                }
            }

            fn visit_const(&mut self, c: I::Const) -> Self::Result {
                match c.kind() {
                    ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                        if let ty::TermKind::Const(term) = self.term.kind() {
                            if let ty::ConstKind::Infer(ty::InferConst::Var(term_vid)) = term.kind()
                            {
                                if self.delegate.root_const_var(vid)
                                    == self.delegate.root_const_var(term_vid)
                                {
                                    return ControlFlow::Break(());
                                }
                            }
                        }

                        self.check_nameable(self.delegate.universe_of_ct(vid).unwrap())
                    }
                    ty::ConstKind::Placeholder(p) => self.check_nameable(p.universe()),
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
            delegate: self.delegate,
            universe_of_term,
            term: goal.predicate.term,
        };
        goal.predicate.alias.visit_with(&mut visitor).is_continue()
            && goal.param_env.visit_with(&mut visitor).is_continue()
    }

    #[instrument(level = "trace", skip(self, param_env), ret)]
    pub(super) fn eq<T: Relate<I>>(
        &mut self,
        param_env: I::ParamEnv,
        lhs: T,
        rhs: T,
    ) -> Result<(), NoSolution> {
        self.relate(param_env, lhs, ty::Variance::Invariant, rhs)
    }

    /// This should be used when relating a rigid alias with another type.
    ///
    /// Normally we emit a nested `AliasRelate` when equating an inference
    /// variable and an alias. This causes us to instead constrain the inference
    /// variable to the alias without emitting a nested alias relate goals.
    #[instrument(level = "trace", skip(self, param_env), ret)]
    pub(super) fn relate_rigid_alias_non_alias(
        &mut self,
        param_env: I::ParamEnv,
        alias: ty::AliasTerm<I>,
        variance: ty::Variance,
        term: I::Term,
    ) -> Result<(), NoSolution> {
        // NOTE: this check is purely an optimization, the structural eq would
        // always fail if the term is not an inference variable.
        if term.is_infer() {
            let cx = self.cx();
            // We need to relate `alias` to `term` treating only the outermost
            // constructor as rigid, relating any contained generic arguments as
            // normal. We do this by first structurally equating the `term`
            // with the alias constructor instantiated with unconstrained infer vars,
            // and then relate this with the whole `alias`.
            //
            // Alternatively we could modify `Equate` for this case by adding another
            // variant to `StructurallyRelateAliases`.
            let identity_args = self.fresh_args_for_item(alias.def_id);
            let rigid_ctor = ty::AliasTerm::new_from_args(cx, alias.def_id, identity_args);
            let ctor_term = rigid_ctor.to_term(cx);
            let obligations =
                self.delegate.eq_structurally_relating_aliases(param_env, term, ctor_term)?;
            debug_assert!(obligations.is_empty());
            self.relate(param_env, alias, variance, rigid_ctor)
        } else {
            Err(NoSolution)
        }
    }

    /// This sohuld only be used when we're either instantiating a previously
    /// unconstrained "return value" or when we're sure that all aliases in
    /// the types are rigid.
    #[instrument(level = "trace", skip(self, param_env), ret)]
    pub(super) fn eq_structurally_relating_aliases<T: Relate<I>>(
        &mut self,
        param_env: I::ParamEnv,
        lhs: T,
        rhs: T,
    ) -> Result<(), NoSolution> {
        let result = self.delegate.eq_structurally_relating_aliases(param_env, lhs, rhs)?;
        assert_eq!(result, vec![]);
        Ok(())
    }

    #[instrument(level = "trace", skip(self, param_env), ret)]
    pub(super) fn sub<T: Relate<I>>(
        &mut self,
        param_env: I::ParamEnv,
        sub: T,
        sup: T,
    ) -> Result<(), NoSolution> {
        self.relate(param_env, sub, ty::Variance::Covariant, sup)
    }

    #[instrument(level = "trace", skip(self, param_env), ret)]
    pub(super) fn relate<T: Relate<I>>(
        &mut self,
        param_env: I::ParamEnv,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
    ) -> Result<(), NoSolution> {
        let goals = self.delegate.relate(param_env, lhs, variance, rhs)?;
        self.add_goals(GoalSource::Misc, goals);
        Ok(())
    }

    /// Equates two values returning the nested goals without adding them
    /// to the nested goals of the `EvalCtxt`.
    ///
    /// If possible, try using `eq` instead which automatically handles nested
    /// goals correctly.
    #[instrument(level = "trace", skip(self, param_env), ret)]
    pub(super) fn eq_and_get_goals<T: Relate<I>>(
        &self,
        param_env: I::ParamEnv,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<I, I::Predicate>>, NoSolution> {
        self.delegate.relate(param_env, lhs, ty::Variance::Invariant, rhs)
    }

    pub(super) fn instantiate_binder_with_infer<T: TypeFoldable<I> + Copy>(
        &self,
        value: ty::Binder<I, T>,
    ) -> T {
        self.delegate.instantiate_binder_with_infer(value)
    }

    pub(super) fn enter_forall<T: TypeFoldable<I> + Copy, U>(
        &self,
        value: ty::Binder<I, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        self.delegate.enter_forall(value, f)
    }

    pub(super) fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<I>,
    {
        self.delegate.resolve_vars_if_possible(value)
    }

    pub(super) fn fresh_args_for_item(&mut self, def_id: I::DefId) -> I::GenericArgs {
        let args = self.delegate.fresh_args_for_item(def_id);
        for arg in args.iter() {
            self.inspect.add_var_value(arg);
        }
        args
    }

    pub(super) fn register_ty_outlives(&self, ty: I::Ty, lt: I::Region) {
        self.delegate.register_ty_outlives(ty, lt);
    }

    pub(super) fn register_region_outlives(&self, a: I::Region, b: I::Region) {
        // `b : a` ==> `a <= b`
        self.delegate.sub_regions(b, a);
    }

    /// Computes the list of goals required for `arg` to be well-formed
    pub(super) fn well_formed_goals(
        &self,
        param_env: I::ParamEnv,
        arg: I::GenericArg,
    ) -> Option<Vec<Goal<I, I::Predicate>>> {
        self.delegate.well_formed_goals(param_env, arg)
    }

    pub(super) fn trait_ref_is_knowable(
        &mut self,
        param_env: I::ParamEnv,
        trait_ref: ty::TraitRef<I>,
    ) -> Result<bool, NoSolution> {
        let delegate = self.delegate;
        let lazily_normalize_ty = |ty| self.structurally_normalize_ty(param_env, ty);
        coherence::trait_ref_is_knowable(&**delegate, trait_ref, lazily_normalize_ty)
            .map(|is_knowable| is_knowable.is_ok())
    }

    pub(super) fn fetch_eligible_assoc_item(
        &self,
        param_env: I::ParamEnv,
        goal_trait_ref: ty::TraitRef<I>,
        trait_assoc_def_id: I::DefId,
        impl_def_id: I::DefId,
    ) -> Result<Option<I::DefId>, NoSolution> {
        self.delegate.fetch_eligible_assoc_item(
            param_env,
            goal_trait_ref,
            trait_assoc_def_id,
            impl_def_id,
        )
    }

    pub(super) fn can_define_opaque_ty(&self, def_id: I::LocalDefId) -> bool {
        self.delegate.defining_opaque_types().contains(&def_id)
    }

    pub(super) fn insert_hidden_type(
        &mut self,
        opaque_type_key: ty::OpaqueTypeKey<I>,
        param_env: I::ParamEnv,
        hidden_ty: I::Ty,
    ) -> Result<(), NoSolution> {
        let mut goals = Vec::new();
        self.delegate.insert_hidden_type(opaque_type_key, param_env, hidden_ty, &mut goals)?;
        self.add_goals(GoalSource::Misc, goals);
        Ok(())
    }

    pub(super) fn add_item_bounds_for_hidden_type(
        &mut self,
        opaque_def_id: I::DefId,
        opaque_args: I::GenericArgs,
        param_env: I::ParamEnv,
        hidden_ty: I::Ty,
    ) {
        let mut goals = Vec::new();
        self.delegate.add_item_bounds_for_hidden_type(
            opaque_def_id,
            opaque_args,
            param_env,
            hidden_ty,
            &mut goals,
        );
        self.add_goals(GoalSource::Misc, goals);
    }

    // Do something for each opaque/hidden pair defined with `def_id` in the
    // current inference context.
    pub(super) fn unify_existing_opaque_tys(
        &mut self,
        param_env: I::ParamEnv,
        key: ty::OpaqueTypeKey<I>,
        ty: I::Ty,
    ) -> Vec<CanonicalResponse<I>> {
        // FIXME: Super inefficient to be cloning this...
        let opaques = self.delegate.clone_opaque_types_for_query_response();

        let mut values = vec![];
        for (candidate_key, candidate_ty) in opaques {
            if candidate_key.def_id != key.def_id {
                continue;
            }
            values.extend(
                self.probe(|result| inspect::ProbeKind::OpaqueTypeStorageLookup {
                    result: *result,
                })
                .enter(|ecx| {
                    for (a, b) in std::iter::zip(candidate_key.args.iter(), key.args.iter()) {
                        ecx.eq(param_env, a, b)?;
                    }
                    ecx.eq(param_env, candidate_ty, ty)?;
                    ecx.add_item_bounds_for_hidden_type(
                        candidate_key.def_id.into(),
                        candidate_key.args,
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
        param_env: I::ParamEnv,
        unevaluated: ty::UnevaluatedConst<I>,
    ) -> Option<I::Const> {
        self.delegate.try_const_eval_resolve(param_env, unevaluated)
    }

    pub(super) fn is_transmutable(
        &mut self,
        param_env: I::ParamEnv,
        dst: I::Ty,
        src: I::Ty,
        assume: I::Const,
    ) -> Result<Certainty, NoSolution> {
        self.delegate.is_transmutable(param_env, dst, src, assume)
    }
}

/// Eagerly replace aliases with inference variables, emitting `AliasRelate`
/// goals, used when adding goals to the `EvalCtxt`. We compute the
/// `AliasRelate` goals before evaluating the actual goal to get all the
/// constraints we can.
///
/// This is a performance optimization to more eagerly detect cycles during trait
/// solving. See tests/ui/traits/next-solver/cycles/cycle-modulo-ambig-aliases.rs.
struct ReplaceAliasWithInfer<'me, 'a, D, I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    ecx: &'me mut EvalCtxt<'a, D>,
    param_env: I::ParamEnv,
}

impl<D, I> TypeFolder<I> for ReplaceAliasWithInfer<'_, '_, D, I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn cx(&self) -> I {
        self.ecx.cx()
    }

    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        match ty.kind() {
            ty::Alias(..) if !ty.has_escaping_bound_vars() => {
                let infer_ty = self.ecx.next_ty_infer();
                let normalizes_to = ty::PredicateKind::AliasRelate(
                    ty.into(),
                    infer_ty.into(),
                    ty::AliasRelationDirection::Equate,
                );
                self.ecx.add_goal(
                    GoalSource::Misc,
                    Goal::new(self.cx(), self.param_env, normalizes_to),
                );
                infer_ty
            }
            _ => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: I::Const) -> I::Const {
        match ct.kind() {
            ty::ConstKind::Unevaluated(..) if !ct.has_escaping_bound_vars() => {
                let infer_ct = self.ecx.next_const_infer();
                let normalizes_to = ty::PredicateKind::AliasRelate(
                    ct.into(),
                    infer_ct.into(),
                    ty::AliasRelationDirection::Equate,
                );
                self.ecx.add_goal(
                    GoalSource::Misc,
                    Goal::new(self.cx(), self.param_env, normalizes_to),
                );
                infer_ct
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, predicate: I::Predicate) -> I::Predicate {
        if predicate.allow_normalization() { predicate.super_fold_with(self) } else { predicate }
    }
}
