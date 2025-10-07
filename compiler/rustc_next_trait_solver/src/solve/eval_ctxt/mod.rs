use std::mem;
use std::ops::ControlFlow;

#[cfg(feature = "nightly")]
use rustc_macros::HashStable_NoContext;
use rustc_type_ir::data_structures::{HashMap, HashSet};
use rustc_type_ir::inherent::*;
use rustc_type_ir::relate::Relate;
use rustc_type_ir::relate::solver_relating::RelateExt;
use rustc_type_ir::search_graph::{CandidateHeadUsages, PathKind};
use rustc_type_ir::solve::OpaqueTypesJank;
use rustc_type_ir::{
    self as ty, CanonicalVarValues, InferCtxtLike, Interner, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
    TypingMode,
};
use tracing::{debug, instrument, trace};

use super::has_only_region_constraints;
use crate::canonical::{
    canonicalize_goal, canonicalize_response, instantiate_and_apply_query_response,
    response_no_constraints_raw,
};
use crate::coherence;
use crate::delegate::SolverDelegate;
use crate::placeholder::BoundVarReplacer;
use crate::resolve::eager_resolve_vars;
use crate::solve::search_graph::SearchGraph;
use crate::solve::ty::may_use_unstable_feature;
use crate::solve::{
    CanonicalInput, CanonicalResponse, Certainty, ExternalConstraintsData, FIXPOINT_STEP_LIMIT,
    Goal, GoalEvaluation, GoalSource, GoalStalledOn, HasChanged, MaybeCause,
    NestedNormalizationGoals, NoSolution, QueryInput, QueryResult, Response, inspect,
};

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

    pub(super) inspect: inspect::EvaluationStepBuilder<D>,
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
        span: <Self::Interner as Interner>::Span,
        stalled_on: Option<GoalStalledOn<Self::Interner>>,
    ) -> Result<GoalEvaluation<Self::Interner>, NoSolution>;

    /// Checks whether evaluating `goal` may hold while treating not-yet-defined
    /// opaque types as being kind of rigid.
    ///
    /// See the comment on [OpaqueTypesJank] for more details.
    fn root_goal_may_hold_opaque_types_jank(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
    ) -> bool;

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
    fn evaluate_root_goal_for_proof_tree(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
        span: <Self::Interner as Interner>::Span,
    ) -> (
        Result<NestedNormalizationGoals<Self::Interner>, NoSolution>,
        inspect::GoalEvaluation<Self::Interner>,
    );
}

impl<D, I> SolverDelegateEvalExt for D
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[instrument(level = "debug", skip(self), ret)]
    fn evaluate_root_goal(
        &self,
        goal: Goal<I, I::Predicate>,
        span: I::Span,
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> Result<GoalEvaluation<I>, NoSolution> {
        EvalCtxt::enter_root(self, self.cx().recursion_limit(), span, |ecx| {
            ecx.evaluate_goal(GoalSource::Misc, goal, stalled_on)
        })
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn root_goal_may_hold_opaque_types_jank(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
    ) -> bool {
        self.probe(|| {
            EvalCtxt::enter_root(self, self.cx().recursion_limit(), I::Span::dummy(), |ecx| {
                ecx.evaluate_goal(GoalSource::Misc, goal, None)
            })
            .is_ok_and(|r| match r.certainty {
                Certainty::Yes => true,
                Certainty::Maybe { cause: _, opaque_types_jank } => match opaque_types_jank {
                    OpaqueTypesJank::AllGood => true,
                    OpaqueTypesJank::ErrorIfRigidSelfTy => false,
                },
            })
        })
    }

    fn root_goal_may_hold_with_depth(
        &self,
        root_depth: usize,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
    ) -> bool {
        self.probe(|| {
            EvalCtxt::enter_root(self, root_depth, I::Span::dummy(), |ecx| {
                ecx.evaluate_goal(GoalSource::Misc, goal, None)
            })
        })
        .is_ok()
    }

    #[instrument(level = "debug", skip(self))]
    fn evaluate_root_goal_for_proof_tree(
        &self,
        goal: Goal<I, I::Predicate>,
        span: I::Span,
    ) -> (Result<NestedNormalizationGoals<I>, NoSolution>, inspect::GoalEvaluation<I>) {
        evaluate_root_goal_for_proof_tree(self, goal, span)
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
        origin_span: I::Span,
        f: impl FnOnce(&mut EvalCtxt<'_, D>) -> R,
    ) -> R {
        let mut search_graph = SearchGraph::new(root_depth);

        let mut ecx = EvalCtxt {
            delegate,
            search_graph: &mut search_graph,
            nested_goals: Default::default(),
            inspect: inspect::EvaluationStepBuilder::new_noop(),

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
        assert!(
            ecx.nested_goals.is_empty(),
            "root `EvalCtxt` should not have any goals added to it"
        );
        assert!(search_graph.is_empty());
        result
    }

    /// Creates a nested evaluation context that shares the same search graph as the
    /// one passed in. This is suitable for evaluation, granted that the search graph
    /// has had the nested goal recorded on its stack. This method only be used by
    /// `search_graph::Delegate::compute_goal`.
    ///
    /// This function takes care of setting up the inference context, setting the anchor,
    /// and registering opaques from the canonicalized input.
    pub(super) fn enter_canonical<R>(
        cx: I,
        search_graph: &'a mut SearchGraph<D>,
        canonical_input: CanonicalInput<I>,
        proof_tree_builder: &mut inspect::ProofTreeBuilder<D>,
        f: impl FnOnce(&mut EvalCtxt<'_, D>, Goal<I, I::Predicate>) -> R,
    ) -> R {
        let (ref delegate, input, var_values) = D::build_with_canonical(cx, &canonical_input);
        for (key, ty) in input.predefined_opaques_in_body.iter() {
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
            inspect: proof_tree_builder.new_evaluation_step(var_values),
        };

        let result = f(&mut ecx, input.goal);
        ecx.inspect.probe_final_state(ecx.delegate, ecx.max_input_universe);
        proof_tree_builder.finish_evaluation_step(ecx.inspect);

        // When creating a query response we clone the opaque type constraints
        // instead of taking them. This would cause an ICE here, since we have
        // assertions against dropping an `InferCtxt` without taking opaques.
        // FIXME: Once we remove support for the old impl we can remove this.
        // FIXME: Could we make `build_with_canonical` into `enter_with_canonical` and call this at the end?
        delegate.reset_opaque_types();

        result
    }

    pub(super) fn ignore_candidate_head_usages(&mut self, usages: CandidateHeadUsages) {
        self.search_graph.ignore_candidate_head_usages(usages);
    }

    /// Recursively evaluates `goal`, returning whether any inference vars have
    /// been constrained and the certainty of the result.
    fn evaluate_goal(
        &mut self,
        source: GoalSource,
        goal: Goal<I, I::Predicate>,
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> Result<GoalEvaluation<I>, NoSolution> {
        let (normalization_nested_goals, goal_evaluation) =
            self.evaluate_goal_raw(source, goal, stalled_on)?;
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
        source: GoalSource,
        goal: Goal<I, I::Predicate>,
        stalled_on: Option<GoalStalledOn<I>>,
    ) -> Result<(NestedNormalizationGoals<I>, GoalEvaluation<I>), NoSolution> {
        // If we have run this goal before, and it was stalled, check that any of the goal's
        // args have changed. Otherwise, we don't need to re-run the goal because it'll remain
        // stalled, since it'll canonicalize the same way and evaluation is pure.
        if let Some(GoalStalledOn {
            num_opaques,
            ref stalled_vars,
            ref sub_roots,
            stalled_certainty,
        }) = stalled_on
            && !stalled_vars.iter().any(|value| self.delegate.is_changed_arg(*value))
            && !sub_roots
                .iter()
                .any(|&vid| self.delegate.sub_unification_table_root_var(vid) != vid)
            && !self.delegate.opaque_types_storage_num_entries().needs_reevaluation(num_opaques)
        {
            return Ok((
                NestedNormalizationGoals::empty(),
                GoalEvaluation {
                    goal,
                    certainty: stalled_certainty,
                    has_changed: HasChanged::No,
                    stalled_on,
                },
            ));
        }

        // We only care about one entry per `OpaqueTypeKey` here,
        // so we only canonicalize the lookup table and ignore
        // duplicate entries.
        let opaque_types = self.delegate.clone_opaque_types_lookup_table();
        let (goal, opaque_types) = eager_resolve_vars(self.delegate, (goal, opaque_types));

        let (orig_values, canonical_goal) = canonicalize_goal(self.delegate, goal, &opaque_types);
        let canonical_result = self.search_graph.evaluate_goal(
            self.cx(),
            canonical_goal,
            self.step_kind_for_source(source),
            &mut inspect::ProofTreeBuilder::new_noop(),
        );
        let response = match canonical_result {
            Err(e) => return Err(e),
            Ok(response) => response,
        };

        let has_changed =
            if !has_only_region_constraints(response) { HasChanged::Yes } else { HasChanged::No };

        let (normalization_nested_goals, certainty) = instantiate_and_apply_query_response(
            self.delegate,
            goal.param_env,
            &orig_values,
            response,
            self.origin_span,
        );

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
            Certainty::Maybe { .. } => match has_changed {
                // FIXME: We could recompute a *new* set of stalled variables by walking
                // through the orig values, resolving, and computing the root vars of anything
                // that is not resolved. Only when *these* have changed is it meaningful
                // to recompute this goal.
                HasChanged::Yes => None,
                HasChanged::No => {
                    let mut stalled_vars = orig_values;

                    // Remove the unconstrained RHS arg, which is expected to have changed.
                    if let Some(normalizes_to) = goal.predicate.as_normalizes_to() {
                        let normalizes_to = normalizes_to.skip_binder();
                        let rhs_arg: I::GenericArg = normalizes_to.term.into();
                        let idx = stalled_vars
                            .iter()
                            .rposition(|arg| *arg == rhs_arg)
                            .expect("expected unconstrained arg");
                        stalled_vars.swap_remove(idx);
                    }

                    // Remove the canonicalized universal vars, since we only care about stalled existentials.
                    let mut sub_roots = Vec::new();
                    stalled_vars.retain(|arg| match arg.kind() {
                        // Lifetimes can never stall goals.
                        ty::GenericArgKind::Lifetime(_) => false,
                        ty::GenericArgKind::Type(ty) => match ty.kind() {
                            ty::Infer(ty::TyVar(vid)) => {
                                sub_roots.push(self.delegate.sub_unification_table_root_var(vid));
                                true
                            }
                            ty::Infer(_) => true,
                            ty::Param(_) | ty::Placeholder(_) => false,
                            _ => unreachable!("unexpected orig_value: {ty:?}"),
                        },
                        ty::GenericArgKind::Const(ct) => match ct.kind() {
                            ty::ConstKind::Infer(_) => true,
                            ty::ConstKind::Param(_) | ty::ConstKind::Placeholder(_) => false,
                            _ => unreachable!("unexpected orig_value: {ct:?}"),
                        },
                    });

                    Some(GoalStalledOn {
                        num_opaques: canonical_goal
                            .canonical
                            .value
                            .predefined_opaques_in_body
                            .len(),
                        stalled_vars,
                        sub_roots,
                        stalled_certainty: certainty,
                    })
                }
            },
        };

        Ok((
            normalization_nested_goals,
            GoalEvaluation { goal, certainty, has_changed, stalled_on },
        ))
    }

    pub(super) fn compute_goal(&mut self, goal: Goal<I, I::Predicate>) -> QueryResult<I> {
        let Goal { param_env, predicate } = goal;
        let kind = predicate.kind();
        self.enter_forall(kind, |ecx, kind| match kind {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(predicate)) => {
                ecx.compute_trait_goal(Goal { param_env, predicate }).map(|(r, _via)| r)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::HostEffect(predicate)) => {
                ecx.compute_host_effect_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(predicate)) => {
                ecx.compute_projection_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(predicate)) => {
                ecx.compute_type_outlives_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(predicate)) => {
                ecx.compute_region_outlives_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
                ecx.compute_const_arg_has_type_goal(Goal { param_env, predicate: (ct, ty) })
            }
            ty::PredicateKind::Clause(ty::ClauseKind::UnstableFeature(symbol)) => {
                ecx.compute_unstable_feature_goal(param_env, symbol)
            }
            ty::PredicateKind::Subtype(predicate) => {
                ecx.compute_subtype_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::Coerce(predicate) => {
                ecx.compute_coerce_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::DynCompatible(trait_def_id) => {
                ecx.compute_dyn_compatible_goal(trait_def_id)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(term)) => {
                ecx.compute_well_formed_goal(Goal { param_env, predicate: term })
            }
            ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(ct)) => {
                ecx.compute_const_evaluatable_goal(Goal { param_env, predicate: ct })
            }
            ty::PredicateKind::ConstEquate(_, _) => {
                panic!("ConstEquate should not be emitted when `-Znext-solver` is active")
            }
            ty::PredicateKind::NormalizesTo(predicate) => {
                ecx.compute_normalizes_to_goal(Goal { param_env, predicate })
            }
            ty::PredicateKind::AliasRelate(lhs, rhs, direction) => {
                ecx.compute_alias_relate_goal(Goal { param_env, predicate: (lhs, rhs, direction) })
            }
            ty::PredicateKind::Ambiguous => {
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
            }
        })
    }

    // Recursively evaluates all the goals added to this `EvalCtxt` to completion, returning
    // the certainty of all the goals.
    #[instrument(level = "trace", skip(self))]
    pub(super) fn try_evaluate_added_goals(&mut self) -> Result<Certainty, NoSolution> {
        for _ in 0..FIXPOINT_STEP_LIMIT {
            match self.evaluate_added_goals_step() {
                Ok(None) => {}
                Ok(Some(cert)) => return Ok(cert),
                Err(NoSolution) => {
                    self.tainted = Err(NoSolution);
                    return Err(NoSolution);
                }
            }
        }

        debug!("try_evaluate_added_goals: encountered overflow");
        Ok(Certainty::overflow(false))
    }

    /// Iterate over all added goals: returning `Ok(Some(_))` in case we can stop rerunning.
    ///
    /// Goals for the next step get directly added to the nested goals of the `EvalCtxt`.
    fn evaluate_added_goals_step(&mut self) -> Result<Option<Certainty>, NoSolution> {
        let cx = self.cx();
        // If this loop did not result in any progress, what's our final certainty.
        let mut unchanged_certainty = Some(Certainty::Yes);
        for (source, goal, stalled_on) in mem::take(&mut self.nested_goals) {
            if let Some(certainty) = self.delegate.compute_goal_fast_path(goal, self.origin_span) {
                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe { .. } => {
                        self.nested_goals.push((source, goal, None));
                        unchanged_certainty = unchanged_certainty.map(|c| c.and(certainty));
                    }
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
                    GoalEvaluation { goal, certainty, stalled_on, has_changed: _ },
                ) = self.evaluate_goal_raw(source, unconstrained_goal, stalled_on)?;
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
                if pred.alias
                    != with_resolved_vars
                        .predicate
                        .as_normalizes_to()
                        .unwrap()
                        .no_bound_vars()
                        .unwrap()
                        .alias
                {
                    unchanged_certainty = None;
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe { .. } => {
                        self.nested_goals.push((source, with_resolved_vars, stalled_on));
                        unchanged_certainty = unchanged_certainty.map(|c| c.and(certainty));
                    }
                }
            } else {
                let GoalEvaluation { goal, certainty, has_changed, stalled_on } =
                    self.evaluate_goal(source, goal, stalled_on)?;
                if has_changed == HasChanged::Yes {
                    unchanged_certainty = None;
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe { .. } => {
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
    pub(super) fn next_term_infer_of_kind(&mut self, term: I::Term) -> I::Term {
        match term.kind() {
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
                        if let ty::TermKind::Ty(term) = self.term.kind()
                            && let ty::Infer(ty::TyVar(term_vid)) = term.kind()
                            && self.delegate.root_ty_var(vid) == self.delegate.root_ty_var(term_vid)
                        {
                            return ControlFlow::Break(());
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
                        if let ty::TermKind::Const(term) = self.term.kind()
                            && let ty::ConstKind::Infer(ty::InferConst::Var(term_vid)) = term.kind()
                            && self.delegate.root_const_var(vid)
                                == self.delegate.root_const_var(term_vid)
                        {
                            return ControlFlow::Break(());
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

            fn visit_predicate(&mut self, p: I::Predicate) -> Self::Result {
                if p.has_non_region_infer() || p.has_placeholders() {
                    p.super_visit_with(self)
                } else {
                    ControlFlow::Continue(())
                }
            }

            fn visit_clauses(&mut self, c: I::Clauses) -> Self::Result {
                if c.has_non_region_infer() || c.has_placeholders() {
                    c.super_visit_with(self)
                } else {
                    ControlFlow::Continue(())
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

    pub(super) fn sub_unify_ty_vids_raw(&self, a: ty::TyVid, b: ty::TyVid) {
        self.delegate.sub_unify_ty_vids_raw(a, b)
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

    pub(super) fn shallow_resolve(&self, ty: I::Ty) -> I::Ty {
        self.delegate.shallow_resolve(ty)
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
        impl_def_id: I::ImplId,
    ) -> Result<Option<I::DefId>, I::ErrorGuaranteed> {
        self.delegate.fetch_eligible_assoc_item(goal_trait_ref, trait_assoc_def_id, impl_def_id)
    }

    #[instrument(level = "debug", skip(self), ret)]
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

    pub(super) fn replace_bound_vars<T: TypeFoldable<I>>(
        &self,
        t: T,
        universes: &mut Vec<Option<ty::UniverseIndex>>,
    ) -> T {
        BoundVarReplacer::replace_bound_vars(&**self.delegate, universes, t).0
    }

    pub(super) fn may_use_unstable_feature(
        &self,
        param_env: I::ParamEnv,
        symbol: I::Symbol,
    ) -> bool {
        may_use_unstable_feature(&**self.delegate, param_env, symbol)
    }

    pub(crate) fn opaques_with_sub_unified_hidden_type(
        &self,
        self_ty: I::Ty,
    ) -> Vec<ty::AliasTy<I>> {
        if let ty::Infer(ty::TyVar(vid)) = self_ty.kind() {
            self.delegate.opaques_with_sub_unified_hidden_type(vid)
        } else {
            vec![]
        }
    }

    /// To return the constraints of a canonical query to the caller, we canonicalize:
    ///
    /// - `var_values`: a map from bound variables in the canonical goal to
    ///   the values inferred while solving the instantiated goal.
    /// - `external_constraints`: additional constraints which aren't expressible
    ///   using simple unification of inference variables.
    ///
    /// This takes the `shallow_certainty` which represents whether we're confident
    /// that the final result of the current goal only depends on the nested goals.
    ///
    /// In case this is `Certainty::Maybe`, there may still be additional nested goals
    /// or inference constraints required for this candidate to be hold. The candidate
    /// always requires all already added constraints and nested goals.
    #[instrument(level = "trace", skip(self), ret)]
    pub(in crate::solve) fn evaluate_added_goals_and_make_canonical_response(
        &mut self,
        shallow_certainty: Certainty,
    ) -> QueryResult<I> {
        self.inspect.make_canonical_response(shallow_certainty);

        let goals_certainty = self.try_evaluate_added_goals()?;
        assert_eq!(
            self.tainted,
            Ok(()),
            "EvalCtxt is tainted -- nested goals may have been dropped in a \
            previous call to `try_evaluate_added_goals!`"
        );

        // We only check for leaks from universes which were entered inside
        // of the query.
        self.delegate.leak_check(self.max_input_universe).map_err(|NoSolution| {
            trace!("failed the leak check");
            NoSolution
        })?;

        let (certainty, normalization_nested_goals) =
            match (self.current_goal_kind, shallow_certainty) {
                // When normalizing, we've replaced the expected term with an unconstrained
                // inference variable. This means that we dropped information which could
                // have been important. We handle this by instead returning the nested goals
                // to the caller, where they are then handled. We only do so if we do not
                // need to recompute the `NormalizesTo` goal afterwards to avoid repeatedly
                // uplifting its nested goals. This is the case if the `shallow_certainty` is
                // `Certainty::Yes`.
                (CurrentGoalKind::NormalizesTo, Certainty::Yes) => {
                    let goals = std::mem::take(&mut self.nested_goals);
                    // As we return all ambiguous nested goals, we can ignore the certainty
                    // returned by `self.try_evaluate_added_goals()`.
                    if goals.is_empty() {
                        assert!(matches!(goals_certainty, Certainty::Yes));
                    }
                    (
                        Certainty::Yes,
                        NestedNormalizationGoals(
                            goals.into_iter().map(|(s, g, _)| (s, g)).collect(),
                        ),
                    )
                }
                _ => {
                    let certainty = shallow_certainty.and(goals_certainty);
                    (certainty, NestedNormalizationGoals::empty())
                }
            };

        if let Certainty::Maybe {
            cause: cause @ MaybeCause::Overflow { keep_constraints: false, .. },
            opaque_types_jank,
        } = certainty
        {
            // If we have overflow, it's probable that we're substituting a type
            // into itself infinitely and any partial substitutions in the query
            // response are probably not useful anyways, so just return an empty
            // query response.
            //
            // This may prevent us from potentially useful inference, e.g.
            // 2 candidates, one ambiguous and one overflow, which both
            // have the same inference constraints.
            //
            // Changing this to retain some constraints in the future
            // won't be a breaking change, so this is good enough for now.
            return Ok(self.make_ambiguous_response_no_constraints(cause, opaque_types_jank));
        }

        let external_constraints =
            self.compute_external_query_constraints(certainty, normalization_nested_goals);
        let (var_values, mut external_constraints) =
            eager_resolve_vars(self.delegate, (self.var_values, external_constraints));

        // Remove any trivial or duplicated region constraints once we've resolved regions
        let mut unique = HashSet::default();
        external_constraints.region_constraints.retain(|outlives| {
            outlives.0.as_region().is_none_or(|re| re != outlives.1) && unique.insert(*outlives)
        });

        let canonical = canonicalize_response(
            self.delegate,
            self.max_input_universe,
            Response {
                var_values,
                certainty,
                external_constraints: self.cx().mk_external_constraints(external_constraints),
            },
        );

        // HACK: We bail with overflow if the response would have too many non-region
        // inference variables. This tends to only happen if we encounter a lot of
        // ambiguous alias types which get replaced with fresh inference variables
        // during generalization. This prevents hangs caused by an exponential blowup,
        // see tests/ui/traits/next-solver/coherence-alias-hang.rs.
        match self.current_goal_kind {
            // We don't do so for `NormalizesTo` goals as we erased the expected term and
            // bailing with overflow here would prevent us from detecting a type-mismatch,
            // causing a coherence error in diesel, see #131969. We still bail with overflow
            // when later returning from the parent AliasRelate goal.
            CurrentGoalKind::NormalizesTo => {}
            CurrentGoalKind::Misc | CurrentGoalKind::CoinductiveTrait => {
                let num_non_region_vars = canonical
                    .variables
                    .iter()
                    .filter(|c| !c.is_region() && c.is_existential())
                    .count();
                if num_non_region_vars > self.cx().recursion_limit() {
                    debug!(?num_non_region_vars, "too many inference variables -> overflow");
                    return Ok(self.make_ambiguous_response_no_constraints(
                        MaybeCause::Overflow {
                            suggest_increasing_limit: true,
                            keep_constraints: false,
                        },
                        OpaqueTypesJank::AllGood,
                    ));
                }
            }
        }

        Ok(canonical)
    }

    /// Constructs a totally unconstrained, ambiguous response to a goal.
    ///
    /// Take care when using this, since often it's useful to respond with
    /// ambiguity but return constrained variables to guide inference.
    pub(in crate::solve) fn make_ambiguous_response_no_constraints(
        &self,
        cause: MaybeCause,
        opaque_types_jank: OpaqueTypesJank,
    ) -> CanonicalResponse<I> {
        response_no_constraints_raw(
            self.cx(),
            self.max_input_universe,
            self.variables,
            Certainty::Maybe { cause, opaque_types_jank },
        )
    }

    /// Computes the region constraints and *new* opaque types registered when
    /// proving a goal.
    ///
    /// If an opaque was already constrained before proving this goal, then the
    /// external constraints do not need to record that opaque, since if it is
    /// further constrained by inference, that will be passed back in the var
    /// values.
    #[instrument(level = "trace", skip(self), ret)]
    fn compute_external_query_constraints(
        &self,
        certainty: Certainty,
        normalization_nested_goals: NestedNormalizationGoals<I>,
    ) -> ExternalConstraintsData<I> {
        // We only return region constraints once the certainty is `Yes`. This
        // is necessary as we may drop nested goals on ambiguity, which may result
        // in unconstrained inference variables in the region constraints. It also
        // prevents us from emitting duplicate region constraints, avoiding some
        // unnecessary work. This slightly weakens the leak check in case it uses
        // region constraints from an ambiguous nested goal. This is tested in both
        // `tests/ui/higher-ranked/leak-check/leak-check-in-selection-5-ambig.rs` and
        // `tests/ui/higher-ranked/leak-check/leak-check-in-selection-6-ambig-unify.rs`.
        let region_constraints = if certainty == Certainty::Yes {
            self.delegate.make_deduplicated_outlives_constraints()
        } else {
            Default::default()
        };

        // We only return *newly defined* opaque types from canonical queries.
        //
        // Constraints for any existing opaque types are already tracked by changes
        // to the `var_values`.
        let opaque_types = self
            .delegate
            .clone_opaque_types_added_since(self.initial_opaque_types_storage_num_entries);

        ExternalConstraintsData { region_constraints, opaque_types, normalization_nested_goals }
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

/// Do not call this directly, use the `tcx` query instead.
pub fn evaluate_root_goal_for_proof_tree_raw_provider<
    D: SolverDelegate<Interner = I>,
    I: Interner,
>(
    cx: I,
    canonical_goal: CanonicalInput<I>,
) -> (QueryResult<I>, I::Probe) {
    let mut inspect = inspect::ProofTreeBuilder::new();
    let canonical_result = SearchGraph::<D>::evaluate_root_goal_for_proof_tree(
        cx,
        cx.recursion_limit(),
        canonical_goal,
        &mut inspect,
    );
    let final_revision = inspect.unwrap();
    (canonical_result, cx.mk_probe(final_revision))
}

/// Evaluate a goal to build a proof tree.
///
/// This is a copy of [EvalCtxt::evaluate_goal_raw] which avoids relying on the
/// [EvalCtxt] and uses a separate cache.
pub(super) fn evaluate_root_goal_for_proof_tree<D: SolverDelegate<Interner = I>, I: Interner>(
    delegate: &D,
    goal: Goal<I, I::Predicate>,
    origin_span: I::Span,
) -> (Result<NestedNormalizationGoals<I>, NoSolution>, inspect::GoalEvaluation<I>) {
    let opaque_types = delegate.clone_opaque_types_lookup_table();
    let (goal, opaque_types) = eager_resolve_vars(delegate, (goal, opaque_types));

    let (orig_values, canonical_goal) = canonicalize_goal(delegate, goal, &opaque_types);

    let (canonical_result, final_revision) =
        delegate.cx().evaluate_root_goal_for_proof_tree_raw(canonical_goal);

    let proof_tree = inspect::GoalEvaluation {
        uncanonicalized_goal: goal,
        orig_values,
        final_revision,
        result: canonical_result,
    };

    let response = match canonical_result {
        Err(e) => return (Err(e), proof_tree),
        Ok(response) => response,
    };

    let (normalization_nested_goals, _certainty) = instantiate_and_apply_query_response(
        delegate,
        goal.param_env,
        &proof_tree.orig_values,
        response,
        origin_span,
    );

    (Ok(normalization_nested_goals), proof_tree)
}
