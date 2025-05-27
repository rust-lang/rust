use std::mem;
use std::ops::ControlFlow;

#[cfg(feature = "nightly")]
use rustc_macros::HashStable_NoContext;
use rustc_type_ir::data_structures::{HashMap, HashSet, ensure_sufficient_stack};
use rustc_type_ir::fast_reject::DeepRejectCtxt;
use rustc_type_ir::inherent::*;
use rustc_type_ir::relate::Relate;
use rustc_type_ir::relate::solver_relating::RelateExt;
use rustc_type_ir::search_graph::PathKind;
use rustc_type_ir::{
    self as ty, CanonicalVarValues, InferCtxtLike, Interner, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
    TypingMode,
};
use tracing::{debug, instrument, trace};

use super::has_only_region_constraints;
use crate::coherence;
use crate::delegate::SolverDelegate;
use crate::solve::inspect::{self, ProofTreeBuilder};
use crate::solve::search_graph::SearchGraph;
use crate::solve::{
    CanonicalInput, Certainty, FIXPOINT_STEP_LIMIT, Goal, GoalEvaluation, GoalEvaluationKind,
    GoalSource, GoalStalledOn, HasChanged, NestedNormalizationGoals, NoSolution, QueryInput,
    QueryResult,
};

pub(super) mod canonical;
mod probe;

/// The kind of goal we're currently proving.
///
/// This has effects on cycle handling handling and on how we compute
/// query responses, see the variant descriptions for more info.
#[derive(Debug, Copy, Clone)]
enum CurrentGoalKind {
    Misc,
    /// We're proving an trait goal for a coinductive trait, either an auto trait or `Sized`.
    ///
    /// These are currently the only goals whose impl where-clauses are considered to be
    /// productive steps.
    CoinductiveTrait,
    /// Unlike other goals, `NormalizesTo` goals act like functions with the expected term
    /// always being fully unconstrained. This would weaken inference however, as the nested
    /// goals never get the inference constraints from the actual normalized-to type.
    ///
    /// Because of this we return any ambiguous nested goals from `NormalizesTo` to the
    /// caller when then adds these to its own context. The caller is always an `AliasRelate`
    /// goal so this never leaks out of the solver.
    NormalizesTo,
}

impl CurrentGoalKind {
    fn from_query_input<I: Interner>(cx: I, input: QueryInput<I, I::Predicate>) -> CurrentGoalKind {
        match input.goal.predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
                if cx.trait_is_coinductive(pred.trait_ref.def_id) {
                    CurrentGoalKind::CoinductiveTrait
                } else {
                    CurrentGoalKind::Misc
                }
            }
            ty::PredicateKind::NormalizesTo(_) => CurrentGoalKind::NormalizesTo,
            _ => CurrentGoalKind::Misc,
        }
    }
}

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
    variables: I::CanonicalVarKinds,

    /// What kind of goal we're currently computing, see the enum definition
    /// for more info.
    current_goal_kind: CurrentGoalKind,
    pub(super) var_values: CanonicalVarValues<I>,

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
    /// The opaque types from the canonical input. We only need to return opaque types
    /// which have been added to the storage while evaluating this goal.
    pub(super) initial_opaque_types_storage_num_entries:
        <D::Infcx as InferCtxtLike>::OpaqueTypeStorageEntries,

    pub(super) search_graph: &'a mut SearchGraph<D>,

    nested_goals: Vec<(GoalSource, Goal<I, I::Predicate>, Option<GoalStalledOn<I>>)>,

    pub(super) origin_span: I::Span,

    // Has this `EvalCtxt` errored out with `NoSolution` in `try_evaluate_added_goals`?
    //
    // If so, then it can no longer be used to make a canonical query response,
    // since subsequent calls to `try_evaluate_added_goals` have possibly dropped
    // ambiguous goals. Instead, a probe needs to be introduced somewhere in the
    // evaluation code.
    tainted: Result<(), NoSolution>,

    pub(super) inspect: ProofTreeBuilder<D>,
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum GenerateProofTree {
    Yes,
    No,
}

pub trait SolverDelegateEvalExt: SolverDelegate {
    /// Evaluates a goal from **outside** of the trait solver.
    ///
    /// Using this while inside of the solver is wrong as it uses a new
    /// search graph which would break cycle detection.
    fn evaluate_root_goal(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
        generate_proof_tree: GenerateProofTree,
        span: <Self::Interner as Interner>::Span,
        stalled_on: Option<GoalStalledOn<Self::Interner>>,
    ) -> (
        Result<GoalEvaluation<Self::Interner>, NoSolution>,
        Option<inspect::GoalEvaluation<Self::Interner>>,
    );

    /// Check whether evaluating `goal` with a depth of `root_depth` may
    /// succeed. This only returns `false` if the goal is guaranteed to
    /// not hold. In case evaluation overflows and fails with ambiguity this
    /// returns `true`.
    ///
    /// This is only intended to be used as a performance optimization
    /// in coherence checking.
    fn root_goal_may_hold_with_depth(
        &self,
        root_depth: usize,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
    ) -> bool;

    // FIXME: This is only exposed because we need to use it in `analyse.rs`
    // which is not yet uplifted. Once that's done, we should remove this.
    fn evaluate_root_goal_raw(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
        generate_proof_tree: GenerateProofTree,
        stalled_on: Option<GoalStalledOn<Self::Interner>>,
    ) -> (
        Result<
            (NestedNormalizationGoals<Self::Interner>, GoalEvaluation<Self::Interner>),
            NoSolution,
        >,
        Option<inspect::GoalEvaluation<Self::Interner>>,
    );
}

impl<D, I> SolverDelegateEvalExt for D
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal(
        &self,
        goal: Goal<I, I::Predicate>,
        generate_proof_tree: GenerateProofTree,
        span: I::Span,
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> (Result<GoalEvaluation<I>, NoSolution>, Option<inspect::GoalEvaluation<I>>) {
        EvalCtxt::enter_root(self, self.cx().recursion_limit(), generate_proof_tree, span, |ecx| {
            ecx.evaluate_goal(GoalEvaluationKind::Root, GoalSource::Misc, goal, stalled_on)
        })
    }

    fn root_goal_may_hold_with_depth(
        &self,
        root_depth: usize,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
    ) -> bool {
        self.probe(|| {
            EvalCtxt::enter_root(self, root_depth, GenerateProofTree::No, I::Span::dummy(), |ecx| {
                ecx.evaluate_goal(GoalEvaluationKind::Root, GoalSource::Misc, goal, None)
            })
            .0
        })
        .is_ok()
    }

    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal_raw(
        &self,
        goal: Goal<I, I::Predicate>,
        generate_proof_tree: GenerateProofTree,
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> (
        Result<(NestedNormalizationGoals<I>, GoalEvaluation<I>), NoSolution>,
        Option<inspect::GoalEvaluation<I>>,
    ) {
        EvalCtxt::enter_root(
            self,
            self.cx().recursion_limit(),
            generate_proof_tree,
            I::Span::dummy(),
            |ecx| {
                ecx.evaluate_goal_raw(GoalEvaluationKind::Root, GoalSource::Misc, goal, stalled_on)
            },
        )
    }
}

impl<'a, D, I> EvalCtxt<'a, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn typing_mode(&self) -> TypingMode<I> {
        self.delegate.typing_mode()
    }

    /// Computes the `PathKind` for the step from the current goal to the
    /// nested goal required due to `source`.
    ///
    /// See #136824 for a more detailed reasoning for this behavior. We
    /// consider cycles to be coinductive if they 'step into' a where-clause
    /// of a coinductive trait. We will likely extend this function in the future
    /// and will need to clearly document it in the rustc-dev-guide before
    /// stabilization.
    pub(super) fn step_kind_for_source(&self, source: GoalSource) -> PathKind {
        match source {
            // We treat these goals as unknown for now. It is likely that most miscellaneous
            // nested goals will be converted to an inductive variant in the future.
            //
            // Having unknown cycles is always the safer option, as changing that to either
            // succeed or hard error is backwards compatible. If we incorrectly treat a cycle
            // as inductive even though it should not be, it may be unsound during coherence and
            // fixing it may cause inference breakage or introduce ambiguity.
            GoalSource::Misc => PathKind::Unknown,
            GoalSource::NormalizeGoal(path_kind) => path_kind,
            GoalSource::ImplWhereBound => match self.current_goal_kind {
                // We currently only consider a cycle coinductive if it steps
                // into a where-clause of a coinductive trait.
                CurrentGoalKind::CoinductiveTrait => PathKind::Coinductive,
                // While normalizing via an impl does step into a where-clause of
                // an impl, accessing the associated item immediately steps out of
                // it again. This means cycles/recursive calls are not guarded
                // by impls used for normalization.
                //
                // See tests/ui/traits/next-solver/cycles/normalizes-to-is-not-productive.rs
                // for how this can go wrong.
                CurrentGoalKind::NormalizesTo => PathKind::Inductive,
                // We probably want to make all traits coinductive in the future,
                // so we treat cycles involving where-clauses of not-yet coinductive
                // traits as ambiguous for now.
                CurrentGoalKind::Misc => PathKind::Unknown,
            },
            // Relating types is always unproductive. If we were to map proof trees to
            // corecursive functions as explained in #136824, relating types never
            // introduces a constructor which could cause the recursion to be guarded.
            GoalSource::TypeRelating => PathKind::Inductive,
            // Instantiating a higher ranked goal can never cause the recursion to be
            // guarded and is therefore unproductive.
            GoalSource::InstantiateHigherRanked => PathKind::Inductive,
            // These goal sources are likely unproductive and can be changed to
            // `PathKind::Inductive`. Keeping them as unknown until we're confident
            // about this and have an example where it is necessary.
            GoalSource::AliasBoundConstCondition | GoalSource::AliasWellFormed => PathKind::Unknown,
        }
    }

    /// Creates a root evaluation context and search graph. This should only be
    /// used from outside of any evaluation, and other methods should be preferred
    /// over using this manually (such as [`SolverDelegateEvalExt::evaluate_root_goal`]).
    pub(super) fn enter_root<R>(
        delegate: &D,
        root_depth: usize,
        generate_proof_tree: GenerateProofTree,
        origin_span: I::Span,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> R,
    ) -> (R, Option<inspect::GoalEvaluation<I>>) {
        let mut search_graph = SearchGraph::new(root_depth);

        let mut ecx = EvalCtxt {
            delegate,
            search_graph: &mut search_graph,
            nested_goals: Default::default(),
            inspect: ProofTreeBuilder::new_maybe_root(generate_proof_tree),

            // Only relevant when canonicalizing the response,
            // which we don't do within this evaluation context.
            max_input_universe: ty::UniverseIndex::ROOT,
            initial_opaque_types_storage_num_entries: Default::default(),
            variables: Default::default(),
            var_values: CanonicalVarValues::dummy(),
            current_goal_kind: CurrentGoalKind::Misc,
            origin_span,
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
        let (ref delegate, input, var_values) = D::build_with_canonical(cx, &canonical_input);

        for &(key, ty) in &input.predefined_opaques_in_body.opaque_types {
            let prev = delegate.register_hidden_type_in_storage(key, ty, I::Span::dummy());
            // It may be possible that two entries in the opaque type storage end up
            // with the same key after resolving contained inference variables.
            //
            // We could put them in the duplicate list but don't have to. The opaques we
            // encounter here are already tracked in the caller, so there's no need to
            // also store them here. We'd take them out when computing the query response
            // and then discard them, as they're already present in the input.
            //
            // Ideally we'd drop duplicate opaque type definitions when computing
            // the canonical input. This is more annoying to implement and may cause a
            // perf regression, so we do it inside of the query for now.
            if let Some(prev) = prev {
                debug!(?key, ?ty, ?prev, "ignore duplicate in `opaque_types_storage`");
            }
        }

        let initial_opaque_types_storage_num_entries = delegate.opaque_types_storage_num_entries();
        let mut ecx = EvalCtxt {
            delegate,
            variables: canonical_input.canonical.variables,
            var_values,
            current_goal_kind: CurrentGoalKind::from_query_input(cx, input),
            max_input_universe: canonical_input.canonical.max_universe,
            initial_opaque_types_storage_num_entries,
            search_graph,
            nested_goals: Default::default(),
            origin_span: I::Span::dummy(),
            tainted: Ok(()),
            inspect: canonical_goal_evaluation.new_goal_evaluation_step(var_values),
        };

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
        step_kind_from_parent: PathKind,
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
                step_kind_from_parent,
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
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> Result<GoalEvaluation<I>, NoSolution> {
        let (normalization_nested_goals, goal_evaluation) =
            self.evaluate_goal_raw(goal_evaluation_kind, source, goal, stalled_on)?;
        assert!(normalization_nested_goals.is_empty());
        Ok(goal_evaluation)
    }

    /// Recursively evaluates `goal`, returning the nested goals in case
    /// the nested goal is a `NormalizesTo` goal.
    ///
    /// As all other goal kinds do not return any nested goals and
    /// `NormalizesTo` is only used by `AliasRelate`, all other callsites
    /// should use [`EvalCtxt::evaluate_goal`] which discards that empty
    /// storage.
    pub(super) fn evaluate_goal_raw(
        &mut self,
        goal_evaluation_kind: GoalEvaluationKind,
        source: GoalSource,
        goal: Goal<I, I::Predicate>,
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> Result<(NestedNormalizationGoals<I>, GoalEvaluation<I>), NoSolution> {
        // If we have run this goal before, and it was stalled, check that any of the goal's
        // args have changed. Otherwise, we don't need to re-run the goal because it'll remain
        // stalled, since it'll canonicalize the same way and evaluation is pure.
        if let Some(stalled_on) = stalled_on {
            if !stalled_on.stalled_vars.iter().any(|value| self.delegate.is_changed_arg(*value))
                && !self
                    .delegate
                    .opaque_types_storage_num_entries()
                    .needs_reevaluation(stalled_on.num_opaques)
            {
                return Ok((
                    NestedNormalizationGoals::empty(),
                    GoalEvaluation {
                        certainty: Certainty::Maybe(stalled_on.stalled_cause),
                        has_changed: HasChanged::No,
                        stalled_on: Some(stalled_on),
                    },
                ));
            }
        }

        let (orig_values, canonical_goal) = self.canonicalize_goal(goal);
        let mut goal_evaluation =
            self.inspect.new_goal_evaluation(goal, &orig_values, goal_evaluation_kind);
        let canonical_response = EvalCtxt::evaluate_canonical_goal(
            self.cx(),
            self.search_graph,
            canonical_goal,
            self.step_kind_for_source(source),
            &mut goal_evaluation,
        );
        let response = match canonical_response {
            Err(e) => {
                self.inspect.goal_evaluation(goal_evaluation);
                return Err(e);
            }
            Ok(response) => response,
        };

        let has_changed =
            if !has_only_region_constraints(response) { HasChanged::Yes } else { HasChanged::No };

        let (normalization_nested_goals, certainty) =
            self.instantiate_and_apply_query_response(goal.param_env, &orig_values, response);
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

        let stalled_on = match certainty {
            Certainty::Yes => None,
            Certainty::Maybe(stalled_cause) => match has_changed {
                // FIXME: We could recompute a *new* set of stalled variables by walking
                // through the orig values, resolving, and computing the root vars of anything
                // that is not resolved. Only when *these* have changed is it meaningful
                // to recompute this goal.
                HasChanged::Yes => None,
                HasChanged::No => {
                    // Remove the unconstrained RHS arg, which is expected to have changed.
                    let mut stalled_vars = orig_values;
                    if let Some(normalizes_to) = goal.predicate.as_normalizes_to() {
                        let normalizes_to = normalizes_to.skip_binder();
                        let rhs_arg: I::GenericArg = normalizes_to.term.into();
                        let idx = stalled_vars
                            .iter()
                            .rposition(|arg| *arg == rhs_arg)
                            .expect("expected unconstrained arg");
                        stalled_vars.swap_remove(idx);
                    }

                    Some(GoalStalledOn {
                        num_opaques: canonical_goal
                            .canonical
                            .value
                            .predefined_opaques_in_body
                            .opaque_types
                            .len(),
                        stalled_vars,
                        stalled_cause,
                    })
                }
            },
        };

        Ok((normalization_nested_goals, GoalEvaluation { certainty, has_changed, stalled_on }))
    }

    fn compute_goal(&mut self, goal: Goal<I, I::Predicate>) -> QueryResult<I> {
        let Goal { param_env, predicate } = goal;
        let kind = predicate.kind();
        if let Some(kind) = kind.no_bound_vars() {
            match kind {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)) => {
                    self.compute_trait_goal(Goal { param_env, predicate }).map(|(r, _via)| r)
                }
                ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(predicate)) => {
                    self.compute_host_effect_goal(Goal { param_env, predicate })
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
                ty::PredicateKind::DynCompatible(trait_def_id) => {
                    self.compute_dyn_compatible_goal(trait_def_id)
                }
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                    self.compute_well_formed_goal(Goal { param_env, predicate: term })
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
            self.enter_forall(kind, |ecx, kind| {
                let goal = goal.with(ecx.cx(), ty::Binder::dummy(kind));
                ecx.add_goal(GoalSource::InstantiateHigherRanked, goal);
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
        // If this loop did not result in any progress, what's our final certainty.
        let mut unchanged_certainty = Some(Certainty::Yes);
        for (source, goal, stalled_on) in mem::take(&mut self.nested_goals) {
            if let Some(has_changed) = self.delegate.compute_goal_fast_path(goal, self.origin_span)
            {
                if matches!(has_changed, HasChanged::Yes) {
                    unchanged_certainty = None;
                }
                continue;
            }

            // We treat normalizes-to goals specially here. In each iteration we take the
            // RHS of the projection, replace it with a fresh inference variable, and only
            // after evaluating that goal do we equate the fresh inference variable with the
            // actual RHS of the predicate.
            //
            // This is both to improve caching, and to avoid using the RHS of the
            // projection predicate to influence the normalizes-to candidate we select.
            //
            // Forgetting to replace the RHS with a fresh inference variable when we evaluate
            // this goal results in an ICE.
            if let Some(pred) = goal.predicate.as_normalizes_to() {
                // We should never encounter higher-ranked normalizes-to goals.
                let pred = pred.no_bound_vars().unwrap();
                // Replace the goal with an unconstrained infer var, so the
                // RHS does not affect projection candidate assembly.
                let unconstrained_rhs = self.next_term_infer_of_kind(pred.term);
                let unconstrained_goal =
                    goal.with(cx, ty::NormalizesTo { alias: pred.alias, term: unconstrained_rhs });

                let (
                    NestedNormalizationGoals(nested_goals),
                    GoalEvaluation { certainty, stalled_on, has_changed: _ },
                ) = self.evaluate_goal_raw(
                    GoalEvaluationKind::Nested,
                    source,
                    unconstrained_goal,
                    stalled_on,
                )?;
                // Add the nested goals from normalization to our own nested goals.
                trace!(?nested_goals);
                self.nested_goals.extend(nested_goals.into_iter().map(|(s, g)| (s, g, None)));

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
                    pred.term,
                    unconstrained_rhs,
                )?;

                // We only look at the `projection_ty` part here rather than
                // looking at the "has changed" return from evaluate_goal,
                // because we expect the `unconstrained_rhs` part of the predicate
                // to have changed -- that means we actually normalized successfully!
                // FIXME: Do we need to eagerly resolve here? Or should we check
                // if the cache key has any changed vars?
                let with_resolved_vars = self.resolve_vars_if_possible(goal);
                if pred.alias != goal.predicate.as_normalizes_to().unwrap().skip_binder().alias {
                    unchanged_certainty = None;
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => {
                        self.nested_goals.push((source, with_resolved_vars, stalled_on));
                        unchanged_certainty = unchanged_certainty.map(|c| c.and(certainty));
                    }
                }
            } else {
                let GoalEvaluation { certainty, has_changed, stalled_on } =
                    self.evaluate_goal(GoalEvaluationKind::Nested, source, goal, stalled_on)?;
                if has_changed == HasChanged::Yes {
                    unchanged_certainty = None;
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => {
                        self.nested_goals.push((source, goal, stalled_on));
                        unchanged_certainty = unchanged_certainty.map(|c| c.and(certainty));
                    }
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

    #[instrument(level = "debug", skip(self))]
    pub(super) fn add_goal(&mut self, source: GoalSource, mut goal: Goal<I, I::Predicate>) {
        goal.predicate =
            goal.predicate.fold_with(&mut ReplaceAliasWithInfer::new(self, source, goal.param_env));
        self.inspect.add_goal(self.delegate, self.max_input_universe, source, goal);
        self.nested_goals.push((source, goal, None));
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

    pub(super) fn next_region_var(&mut self) -> I::Region {
        let region = self.delegate.next_region_infer();
        self.inspect.add_var_value(region);
        region
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
            cache: HashSet<I::Ty>,
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
                if self.cache.contains(&t) {
                    return ControlFlow::Continue(());
                }

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

                        self.check_nameable(self.delegate.universe_of_ty(vid).unwrap())?;
                    }
                    ty::Placeholder(p) => self.check_nameable(p.universe())?,
                    _ => {
                        if t.has_non_region_infer() || t.has_placeholders() {
                            t.super_visit_with(self)?
                        }
                    }
                }

                assert!(self.cache.insert(t));
                ControlFlow::Continue(())
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
            cache: Default::default(),
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
            let obligations = self.delegate.eq_structurally_relating_aliases(
                param_env,
                term,
                ctor_term,
                self.origin_span,
            )?;
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
        let result = self.delegate.eq_structurally_relating_aliases(
            param_env,
            lhs,
            rhs,
            self.origin_span,
        )?;
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
        let goals = self.delegate.relate(param_env, lhs, variance, rhs, self.origin_span)?;
        for &goal in goals.iter() {
            let source = match goal.predicate.kind().skip_binder() {
                ty::PredicateKind::Subtype { .. } | ty::PredicateKind::AliasRelate(..) => {
                    GoalSource::TypeRelating
                }
                // FIXME(-Znext-solver=coinductive): should these WF goals also be unproductive?
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(_)) => GoalSource::Misc,
                p => unreachable!("unexpected nested goal in `relate`: {p:?}"),
            };
            self.add_goal(source, goal);
        }
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
        Ok(self.delegate.relate(param_env, lhs, ty::Variance::Invariant, rhs, self.origin_span)?)
    }

    pub(super) fn instantiate_binder_with_infer<T: TypeFoldable<I> + Copy>(
        &self,
        value: ty::Binder<I, T>,
    ) -> T {
        self.delegate.instantiate_binder_with_infer(value)
    }

    /// `enter_forall`, but takes `&mut self` and passes it back through the
    /// callback since it can't be aliased during the call.
    pub(super) fn enter_forall<T: TypeFoldable<I>, U>(
        &mut self,
        value: ty::Binder<I, T>,
        f: impl FnOnce(&mut Self, T) -> U,
    ) -> U {
        self.delegate.enter_forall(value, |value| f(self, value))
    }

    pub(super) fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<I>,
    {
        self.delegate.resolve_vars_if_possible(value)
    }

    pub(super) fn eager_resolve_region(&self, r: I::Region) -> I::Region {
        if let ty::ReVar(vid) = r.kind() {
            self.delegate.opportunistic_resolve_lt_var(vid)
        } else {
            r
        }
    }

    pub(super) fn fresh_args_for_item(&mut self, def_id: I::DefId) -> I::GenericArgs {
        let args = self.delegate.fresh_args_for_item(def_id);
        for arg in args.iter() {
            self.inspect.add_var_value(arg);
        }
        args
    }

    pub(super) fn register_ty_outlives(&self, ty: I::Ty, lt: I::Region) {
        self.delegate.register_ty_outlives(ty, lt, self.origin_span);
    }

    pub(super) fn register_region_outlives(&self, a: I::Region, b: I::Region) {
        // `'a: 'b` ==> `'b <= 'a`
        self.delegate.sub_regions(b, a, self.origin_span);
    }

    /// Computes the list of goals required for `arg` to be well-formed
    pub(super) fn well_formed_goals(
        &self,
        param_env: I::ParamEnv,
        term: I::Term,
    ) -> Option<Vec<Goal<I, I::Predicate>>> {
        self.delegate.well_formed_goals(param_env, term)
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
        goal_trait_ref: ty::TraitRef<I>,
        trait_assoc_def_id: I::DefId,
        impl_def_id: I::DefId,
    ) -> Result<Option<I::DefId>, I::ErrorGuaranteed> {
        self.delegate.fetch_eligible_assoc_item(goal_trait_ref, trait_assoc_def_id, impl_def_id)
    }

    pub(super) fn register_hidden_type_in_storage(
        &mut self,
        opaque_type_key: ty::OpaqueTypeKey<I>,
        hidden_ty: I::Ty,
    ) -> Option<I::Ty> {
        self.delegate.register_hidden_type_in_storage(opaque_type_key, hidden_ty, self.origin_span)
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
        self.add_goals(GoalSource::AliasWellFormed, goals);
    }

    // Do something for each opaque/hidden pair defined with `def_id` in the
    // current inference context.
    pub(super) fn probe_existing_opaque_ty(
        &mut self,
        key: ty::OpaqueTypeKey<I>,
    ) -> Option<(ty::OpaqueTypeKey<I>, I::Ty)> {
        // We shouldn't have any duplicate entries when using
        // this function during `TypingMode::Analysis`.
        let duplicate_entries = self.delegate.clone_duplicate_opaque_types();
        assert!(duplicate_entries.is_empty(), "unexpected duplicates: {duplicate_entries:?}");
        let mut matching = self.delegate.clone_opaque_types_lookup_table().into_iter().filter(
            |(candidate_key, _)| {
                candidate_key.def_id == key.def_id
                    && DeepRejectCtxt::relate_rigid_rigid(self.cx())
                        .args_may_unify(candidate_key.args, key.args)
            },
        );
        let first = matching.next();
        let second = matching.next();
        assert_eq!(second, None);
        first
    }

    // Try to evaluate a const, or return `None` if the const is too generic.
    // This doesn't mean the const isn't evaluatable, though, and should be treated
    // as an ambiguity rather than no-solution.
    pub(super) fn evaluate_const(
        &self,
        param_env: I::ParamEnv,
        uv: ty::UnevaluatedConst<I>,
    ) -> Option<I::Const> {
        self.delegate.evaluate_const(param_env, uv)
    }

    pub(super) fn is_transmutable(
        &mut self,
        dst: I::Ty,
        src: I::Ty,
        assume: I::Const,
    ) -> Result<Certainty, NoSolution> {
        self.delegate.is_transmutable(dst, src, assume)
    }
}

/// Eagerly replace aliases with inference variables, emitting `AliasRelate`
/// goals, used when adding goals to the `EvalCtxt`. We compute the
/// `AliasRelate` goals before evaluating the actual goal to get all the
/// constraints we can.
///
/// This is a performance optimization to more eagerly detect cycles during trait
/// solving. See tests/ui/traits/next-solver/cycles/cycle-modulo-ambig-aliases.rs.
///
/// The emitted goals get evaluated in the context of the parent goal; by
/// replacing aliases in nested goals we essentially pull the normalization out of
/// the nested goal. We want to treat the goal as if the normalization still happens
/// inside of the nested goal by inheriting the `step_kind` of the nested goal and
/// storing it in the `GoalSource` of the emitted `AliasRelate` goals.
/// This is necessary for tests/ui/sized/coinductive-1.rs to compile.
struct ReplaceAliasWithInfer<'me, 'a, D, I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    ecx: &'me mut EvalCtxt<'a, D>,
    param_env: I::ParamEnv,
    normalization_goal_source: GoalSource,
    cache: HashMap<I::Ty, I::Ty>,
}

impl<'me, 'a, D, I> ReplaceAliasWithInfer<'me, 'a, D, I>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    fn new(
        ecx: &'me mut EvalCtxt<'a, D>,
        for_goal_source: GoalSource,
        param_env: I::ParamEnv,
    ) -> Self {
        let step_kind = ecx.step_kind_for_source(for_goal_source);
        ReplaceAliasWithInfer {
            ecx,
            param_env,
            normalization_goal_source: GoalSource::NormalizeGoal(step_kind),
            cache: Default::default(),
        }
    }
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
                    self.normalization_goal_source,
                    Goal::new(self.cx(), self.param_env, normalizes_to),
                );
                infer_ty
            }
            _ => {
                if !ty.has_aliases() {
                    ty
                } else if let Some(&entry) = self.cache.get(&ty) {
                    return entry;
                } else {
                    let res = ty.super_fold_with(self);
                    assert!(self.cache.insert(ty, res).is_none());
                    res
                }
            }
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
                    self.normalization_goal_source,
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
