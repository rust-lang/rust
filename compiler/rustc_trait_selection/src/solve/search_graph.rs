#![allow(rustc::potential_query_instability)]
use super::inspect;
use super::inspect::ProofTreeBuilder;
use super::SolverMode;
use crate::solve::FIXPOINT_STEP_LIMIT;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::fx::FxHashSet;
use rustc_index::Idx;
use rustc_index::IndexVec;
use rustc_middle::dep_graph::dep_kinds;
use rustc_middle::traits::solve::CacheData;
use rustc_middle::traits::solve::{CanonicalInput, Certainty, EvaluationCache, QueryResult};
use rustc_middle::ty::TyCtxt;
use rustc_session::Limit;
use std::cmp::Ordering;
use std::collections::BTreeMap;

rustc_index::newtype_index! {
    #[orderable]
    pub struct StackDepth {}
}

bitflags::bitflags! {
    /// Whether and how this goal has been used as the root of a
    /// cycle. We track the kind of cycle as we're otherwise forced
    /// to always rerun at least once.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct HasBeenUsed: u8 {
        const INDUCTIVE_CYCLE = 1 << 0;
        const COINDUCTIVE_CYCLE = 1 << 1;
    }
}

impl Default for HasBeenUsed {
    fn default() -> Self {
        HasBeenUsed::empty()
    }
}

#[derive(Debug)]
struct StackEntry<'tcx> {
    input: CanonicalInput<'tcx>,

    available_depth: Limit,

    /// The maximum depth reached by this stack entry, only up-to date
    /// for the top of the stack and lazily updated for the rest.
    reached_depth: StackDepth,

    /// If this entry is a non-root cycle participant, we store the depth of
    /// all cycle participants and how that participant has been used.
    ///
    /// We must not move the result of non-root cycle participants to the
    /// global cache. See [SearchGraph::cycle_participants] for more details.
    /// We store the highest stack depth of a head of a cycle this goal is involved
    /// in. This necessary to soundly cache its provisional result.
    heads: BTreeMap<StackDepth, HasBeenUsed>,

    encountered_overflow: bool,

    has_been_used: HasBeenUsed,
    /// Starts out as `None` and gets set when rerunning this
    /// goal in case we encounter a cycle.
    provisional_result: Option<QueryResult<'tcx>>,
}

/// The provisional result for a goal which is not on the stack.
#[derive(Debug)]
struct DetachedEntry<'tcx> {
    /// All stack entries upon which this `result` depends. This entry
    /// is therefore outdated once the provisional result of any of these
    /// goals changes or gets removed from the stack.
    heads: BTreeMap<StackDepth, HasBeenUsed>,
    result: QueryResult<'tcx>,
}

impl<'tcx> DetachedEntry<'tcx> {
    /// The head of the smallest non-trivial cycle involving this entry.
    ///
    /// Given the following rules, when proving `A` the head for
    /// the provisional entry of `C` would be `B`.
    /// ```plain
    /// A :- B
    /// B :- C
    /// C :- A + B + C
    /// ```
    fn head(&self) -> StackDepth {
        *self.heads.last_key_value().unwrap().0
    }
}

/// Stores the stack depth of a currently evaluated goal *and* already
/// computed results for goals which depend on other goals still on the stack.
///
/// The provisional result may depend on whether the stack above it is inductive
/// or coinductive. Because of this, we store separate provisional results for
/// each case. If an provisional entry is not applicable, it may be the case
/// that we already have provisional result while computing a goal. In this case
/// we prefer the provisional result to potentially avoid fixpoint iterations.
/// See tests/ui/traits/next-solver/cycles/mixed-cycles-2.rs for an example.
///
/// The provisional cache can theoretically result in changes to the observable behavior,
/// see tests/ui/traits/next-solver/cycles/provisional-cache-impacts-behavior.rs.
#[derive(Debug, Default)]
struct ProvisionalCacheEntry<'tcx> {
    stack_depth: Option<StackDepth>,
    with_inductive_stack: Option<DetachedEntry<'tcx>>,
    with_coinductive_stack: Option<DetachedEntry<'tcx>>,
}

impl<'tcx> ProvisionalCacheEntry<'tcx> {
    fn is_empty(&self) -> bool {
        self.stack_depth.is_none()
            && self.with_inductive_stack.is_none()
            && self.with_coinductive_stack.is_none()
    }
}

#[derive(Debug)]
struct ReuseableDetachedEntry<'tcx> {
    /// See the comment on `DetachedEntry::heads`. This entry is only
    /// valid as long as all heads are on the stack and their provisional
    /// results have not changed.
    heads: BTreeMap<StackDepth, HasBeenUsed>,
    /// The provisional results of all cycle heads used
    /// when computing this goal.
    provisional_results: Vec<QueryResult<'tcx>>,
    /// Whether the path from the `head` of the smallest
    /// cycle to this goal was coinductive.
    is_coinductive: bool,

    result: QueryResult<'tcx>,
}

impl<'tcx> ReuseableDetachedEntry<'tcx> {
    /// We only want to reuse a provisional cache entry if the
    /// provisional results of all used cycle heads has not changed
    /// since computing this result.
    fn is_applicable(
        &self,
        tcx: TyCtxt<'tcx>,
        stack: &IndexVec<StackDepth, StackEntry<'tcx>>,
    ) -> bool {
        itertools::zip_eq(&self.heads, &self.provisional_results).all(|x| {
            let ((&stack_depth, &usage_kind), &result) = x;
            let actual = if let Some(result) = stack[stack_depth].provisional_result {
                result
            } else if usage_kind == HasBeenUsed::COINDUCTIVE_CYCLE {
                response_no_constraints(tcx, stack[stack_depth].input, Certainty::Yes)
            } else if usage_kind == HasBeenUsed::INDUCTIVE_CYCLE {
                response_no_constraints(tcx, stack[stack_depth].input, Certainty::overflow(false))
            } else {
                unreachable!(); // for now
            };

            actual == result
        })
    }
}

/// When rerunning a goal as we haven't yet reached a fixpoint,
/// try to reuse as much from the previous iteration as possible to
/// avoid exponential blowup.
#[derive(Debug)]
struct FixpointInstantiationCacheEntry<'tcx> {
    expected_stack: IndexVec<StackDepth, CanonicalInput<'tcx>>,
    heads: BTreeMap<StackDepth, HasBeenUsed>,
    provisional_result: QueryResult<'tcx>,
    detached_cache_entries: FxHashMap<CanonicalInput<'tcx>, ReuseableDetachedEntry<'tcx>>,
}

impl<'tcx> FixpointInstantiationCacheEntry<'tcx> {
    /// When reusing a provisional result we have to make sure the
    /// result is actually applicable for the given goal. We check this
    /// by only adding a provisional result if the stack matches the
    /// previous iteration.
    ///
    /// This may still impact behavior if there is incompleteness,
    /// but given that we only cache the root if cycles we can just
    /// ignore this.
    fn is_applicable(&self, stack: &IndexVec<StackDepth, StackEntry<'tcx>>) -> bool {
        self.expected_stack.len() == stack.len()
            && itertools::zip_eq(&self.expected_stack, stack.iter().map(|e| &e.input))
                .all(|(l, r)| l == r)
    }
}

// FIXME(@lcnr): We may want to move `StackEntry::heads` and
// `StackEntry::has_been_used` into a separate list in `CycleData`.
#[derive(Debug)]
struct CycleData<'tcx> {
    /// The lowest stack depth of all participants. The root is the only cycle
    /// participant which will get moved to the global cache.
    root: StackDepth,

    /// The provisional results for all nested cycle participants we've already computed.
    /// The next time we evaluate these nested goals we use that result in the first
    /// iteration.
    fixpoint_instantiation_cache:
        FxHashMap<CanonicalInput<'tcx>, Vec<FixpointInstantiationCacheEntry<'tcx>>>,

    /// We put only the root goal of a coinductive cycle into the global cache.
    ///
    /// If we were to use that result when later trying to prove another cycle
    /// participant, we can end up with unstable query results.
    ///
    /// See tests/ui/next-solver/coinduction/incompleteness-unstable-result.rs for
    /// an example of where this is needed.
    cycle_participants: FxHashSet<CanonicalInput<'tcx>>,
}

impl<'tcx> CycleData<'tcx> {
    fn new(root: StackDepth) -> CycleData<'tcx> {
        CycleData {
            root,
            fixpoint_instantiation_cache: Default::default(),
            cycle_participants: Default::default(),
        }
    }

    /// When encountering a solver cycle, the result of the current goal
    /// depends on goals lower on the stack.
    ///
    /// We have to therefore be careful when caching goals. Only the final result
    /// of the cycle root, i.e. the lowest goal on the stack involved in this cycle,
    /// is moved to the global cache while all others are stored in a provisional cache.
    ///
    /// We update both the head of this cycle to rerun its evaluation until
    /// we reach a fixpoint and all other cycle participants to make sure that
    /// their result does not get moved to the global cache.
    //
    // TODO
    fn tag_cycle_participants(
        &mut self,
        stack: &mut IndexVec<StackDepth, StackEntry<'tcx>>,
        heads: &BTreeMap<StackDepth, HasBeenUsed>,
    ) {
        let head = *heads.last_key_value().unwrap().0;
        self.root = self.root.min(head);

        for (&head, &usage_kind) in heads {
            stack[head].has_been_used |= usage_kind;
        }

        for stack_entry in stack.iter_mut().skip(head.index() + 1) {
            for (&head, &usage_kind) in heads {
                debug_assert_ne!(usage_kind, HasBeenUsed::empty());
                *stack_entry.heads.entry(head).or_default() |= usage_kind;
            }

            self.cycle_participants.insert(stack_entry.input);
        }
    }

    /// Add the provisional result for a given goal to the storage.
    fn add_fixpoint_instantiation_cache_entry(
        &mut self,
        stack: &IndexVec<StackDepth, StackEntry<'tcx>>,
        entry: &StackEntry<'tcx>,
        provisional_result: QueryResult<'tcx>,
        detached_cache_entries: FxHashMap<CanonicalInput<'tcx>, ReuseableDetachedEntry<'tcx>>,
    ) {
        // Store the provisional result for this goal to enable
        // us to reuse it in case of a fixpoint.
        let cache_entry = FixpointInstantiationCacheEntry {
            expected_stack: stack.iter().map(|e| e.input).collect(),
            heads: entry.heads.clone(),
            provisional_result,
            detached_cache_entries,
        };
        let cache_entries = self.fixpoint_instantiation_cache.entry(entry.input).or_default();
        if cfg!(debug_assertions) {
            // We should only use this function after computing a goal, at
            // the start of which we remove its previous fixpoint instantiation
            // cache entry.
            if let Some(r) = cache_entries.iter().find(|r| r.is_applicable(stack)) {
                bug!("existing fixpoint instantiation cache entry: {r:?}");
            }
        }

        cache_entries.push(cache_entry);
    }

    /// Tries to reuse results from the previous fixpoint iteration if they
    /// are still applicable.
    ///
    /// We are able to use this to both initialize the provisional result
    /// and add the results of nested goals to the provisional cache if they
    /// only depend on stack entries whose provisional result has not changed.
    fn try_apply_fixpoint_instantiation_cache_entry(
        &mut self,
        tcx: TyCtxt<'tcx>,
        stack: &mut IndexVec<StackDepth, StackEntry<'tcx>>,
        provisional_cache: &mut FxHashMap<CanonicalInput<'tcx>, ProvisionalCacheEntry<'tcx>>,
        available_depth: Limit,
        depth: StackDepth,
        input: CanonicalInput<'tcx>,
    ) -> bool {
        let Some(entries) = self.fixpoint_instantiation_cache.get_mut(&input) else {
            return false;
        };
        let Some(idx) = entries.iter().position(|r| r.is_applicable(stack)) else {
            return false;
        };

        let FixpointInstantiationCacheEntry {
            expected_stack: _,
            heads,
            provisional_result,
            detached_cache_entries,
        } = entries.remove(idx);

        assert_ne!(*heads.last_key_value().unwrap().0, depth);
        let stack_entry = StackEntry {
            input,
            available_depth,
            reached_depth: depth,
            heads: Default::default(),
            encountered_overflow: false,
            has_been_used: HasBeenUsed::empty(),
            provisional_result: Some(provisional_result),
        };
        assert_eq!(stack.push(stack_entry), depth);

        for (input, provisional_cache_entry) in detached_cache_entries {
            if provisional_cache_entry.is_applicable(tcx, stack) {
                let cache_entry = provisional_cache.entry(input).or_default();
                let heads = provisional_cache_entry.heads;
                for (&head, &usage_kind) in heads.iter() {
                    assert_ne!(usage_kind, HasBeenUsed::empty());
                    stack[head].has_been_used |= usage_kind;
                }

                if cfg!(debug_assertions) {
                    let (&head, &usage_kind) = heads.last_key_value().unwrap();
                    assert_eq!(head, depth);
                    assert_ne!(usage_kind, HasBeenUsed::empty());
                    assert!(stack[head].has_been_used.contains(usage_kind));
                }

                if provisional_cache_entry.is_coinductive {
                    cache_entry.with_coinductive_stack =
                        Some(DetachedEntry { heads, result: provisional_cache_entry.result });
                } else {
                    cache_entry.with_inductive_stack =
                        Some(DetachedEntry { heads, result: provisional_cache_entry.result });
                }
            }
        }

        true
    }
}

#[derive(Debug)]
pub(super) struct SearchGraph<'tcx> {
    mode: SolverMode,
    /// The stack of goals currently being computed.
    ///
    /// An element is *deeper* in the stack if its index is *lower*.
    stack: IndexVec<StackDepth, StackEntry<'tcx>>,

    /// In case we're currently in a solver cycle, we track a lot of
    /// additional data.
    cycle_data: Option<CycleData<'tcx>>,

    /// A cache for the result of nested goals which depend on goals currently on the
    /// stack. We remove cached results once we pop any goal used while computing it.
    ///
    /// This is not part of `cycle_data` as it contains all stack entries even while we're
    /// not yet in a cycle.
    provisional_cache: FxHashMap<CanonicalInput<'tcx>, ProvisionalCacheEntry<'tcx>>,
}

impl<'tcx> SearchGraph<'tcx> {
    pub(super) fn new(mode: SolverMode) -> SearchGraph<'tcx> {
        Self {
            mode,
            stack: Default::default(),
            cycle_data: None,
            provisional_cache: Default::default(),
        }
    }

    pub(super) fn solver_mode(&self) -> SolverMode {
        self.mode
    }

    /// Pops the highest goal from the stack, lazily updating the
    /// the next goal in the stack.
    ///
    /// Directly popping from the stack instead of using this method
    /// would cause us to not track overflow and recursion depth correctly.
    fn pop_stack(&mut self) -> StackEntry<'tcx> {
        let elem = self.stack.pop().unwrap();
        if let Some(last) = self.stack.raw.last_mut() {
            last.reached_depth = last.reached_depth.max(elem.reached_depth);
            last.encountered_overflow |= elem.encountered_overflow;
        }
        elem
    }

    /// The trait solver behavior is different for coherence
    /// so we use a separate cache. Alternatively we could use
    /// a single cache and share it between coherence and ordinary
    /// trait solving.
    pub(super) fn global_cache(&self, tcx: TyCtxt<'tcx>) -> &'tcx EvaluationCache<'tcx> {
        match self.mode {
            SolverMode::Normal => &tcx.new_solver_evaluation_cache,
            SolverMode::Coherence => &tcx.new_solver_coherence_evaluation_cache,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.check_invariants();
        if self.stack.is_empty() { true } else { false }
    }

    /// Returns the remaining depth allowed for nested goals.
    ///
    /// This is generally simply one less than the current depth.
    /// However, if we encountered overflow, we significantly reduce
    /// the remaining depth of all nested goals to prevent hangs
    /// in case there is exponential blowup.
    fn allowed_depth_for_nested(
        tcx: TyCtxt<'tcx>,
        stack: &IndexVec<StackDepth, StackEntry<'tcx>>,
    ) -> Option<Limit> {
        if let Some(last) = stack.raw.last() {
            if last.available_depth.0 == 0 {
                return None;
            }

            Some(if last.encountered_overflow {
                Limit(last.available_depth.0 / 4)
            } else {
                Limit(last.available_depth.0 - 1)
            })
        } else {
            Some(tcx.recursion_limit())
        }
    }

    fn stack_coinductive_from(
        tcx: TyCtxt<'tcx>,
        stack: &IndexVec<StackDepth, StackEntry<'tcx>>,
        head: StackDepth,
    ) -> bool {
        stack.raw[head.index()..]
            .iter()
            .all(|entry| entry.input.value.goal.predicate.is_coinductive(tcx))
    }

    fn clear_dependent_provisional_results(
        provisional_cache: &mut FxHashMap<CanonicalInput<'tcx>, ProvisionalCacheEntry<'tcx>>,
        head: StackDepth,
        mut f: impl FnMut(
            CanonicalInput<'tcx>,
            Option<DetachedEntry<'tcx>>,
            Option<DetachedEntry<'tcx>>,
        ),
    ) {
        let condition = |p: &mut DetachedEntry<'tcx>| match p.head().cmp(&head) {
            Ordering::Less => false,
            Ordering::Equal => true,
            Ordering::Greater => bug!("provisional entry for popped value"),
        };

        #[allow(rustc::potential_query_instability)]
        provisional_cache.retain(|input, entry| {
            let coinductive = entry.with_coinductive_stack.take_if(condition);
            let inductive = entry.with_inductive_stack.take_if(condition);
            f(*input, coinductive, inductive);
            !entry.is_empty()
        });
    }

    /// Probably the most involved method of the whole solver.
    ///
    /// Given some goal which is proven via the `prove_goal` closure, this
    /// handles caching, overflow, and coinductive cycles.
    pub(super) fn with_new_goal(
        &mut self,
        tcx: TyCtxt<'tcx>,
        input: CanonicalInput<'tcx>,
        inspect: &mut ProofTreeBuilder<'tcx>,
        mut prove_goal: impl FnMut(&mut Self, &mut ProofTreeBuilder<'tcx>) -> QueryResult<'tcx>,
    ) -> QueryResult<'tcx> {
        self.check_invariants();

        // Check for overflow.
        let Some(available_depth) = Self::allowed_depth_for_nested(tcx, &self.stack) else {
            if let Some(last) = self.stack.raw.last_mut() {
                last.encountered_overflow = true;
            }

            inspect.goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::Overflow);
            return response_no_constraints(tcx, input, Certainty::overflow(true));
        };

        if let Some(result) = self.lookup_global_cache(tcx, input, available_depth, inspect) {
            return result;
        }

        // Check whether the goal is in the provisional cache.
        // The provisional result may rely on the path to its cycle roots,
        // so we have to check the path of the current goal matches that of
        // the cache entry.
        let cache_entry = self.provisional_cache.entry(input).or_default();
        if let Some(entry) = cache_entry
            .with_coinductive_stack
            .as_ref()
            .filter(|p| Self::stack_coinductive_from(tcx, &self.stack, p.head()))
            .or_else(|| {
                cache_entry
                    .with_inductive_stack
                    .as_ref()
                    .filter(|p| !Self::stack_coinductive_from(tcx, &self.stack, p.head()))
            })
        {
            // We have a nested goal which is already in the provisional cache, use
            // its result. We do not provide any usage kind as that should have been
            // already set correctly while computing the cache entry.
            inspect
                .goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::ProvisionalCacheHit);
            self.cycle_data.as_mut().unwrap().tag_cycle_participants(&mut self.stack, &entry.heads);
            return entry.result;
        } else if let Some(stack_depth) = cache_entry.stack_depth {
            debug!("encountered cycle with depth {stack_depth:?}");
            // We have a nested goal which directly relies on a goal deeper in the stack.
            //
            // We start by tagging all cycle participants, as that's necessary for caching.
            //
            // Finally we can return either the provisional response or the initial response
            // in case we're in the first fixpoint iteration for this goal.
            inspect.goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::CycleInStack);
            let is_coinductive_cycle = Self::stack_coinductive_from(tcx, &self.stack, stack_depth);
            let usage_kind = if is_coinductive_cycle {
                HasBeenUsed::COINDUCTIVE_CYCLE
            } else {
                HasBeenUsed::INDUCTIVE_CYCLE
            };

            let cycle_data = self.cycle_data.get_or_insert_with(|| CycleData::new(stack_depth));
            let heads = [(stack_depth, usage_kind)].into_iter().collect(); // TODO
            cycle_data.tag_cycle_participants(&mut self.stack, &heads);

            // Return the provisional result or, if we're in the first iteration,
            // start with no constraints.
            return if let Some(result) = self.stack[stack_depth].provisional_result {
                result
            } else if is_coinductive_cycle {
                response_no_constraints(tcx, input, Certainty::Yes)
            } else {
                response_no_constraints(tcx, input, Certainty::overflow(false))
            };
        } else {
            // No entry, we push this goal on the stack and try to prove it.
            let depth = self.stack.next_index();
            cache_entry.stack_depth = Some(depth);
            let from_cache = if let Some(cycle_data) = &mut self.cycle_data {
                cycle_data.try_apply_fixpoint_instantiation_cache_entry(
                    tcx,
                    &mut self.stack,
                    &mut self.provisional_cache,
                    available_depth,
                    depth,
                    input,
                )
            } else {
                false
            };

            if !from_cache {
                let stack_entry = StackEntry {
                    input,
                    available_depth,
                    reached_depth: depth,
                    heads: Default::default(),
                    encountered_overflow: false,
                    has_been_used: HasBeenUsed::empty(),
                    provisional_result: None,
                };
                assert_eq!(self.stack.push(stack_entry), depth);
            }
        }

        // This is for global caching, so we properly track query dependencies.
        // Everything that affects the `result` should be performed within this
        // `with_anon_task` closure. If computing this goal depends on something
        // not tracked by the cache key and from outside of this anon task, it
        // must not be added to the global cache. Notably, this is the case for
        // trait solver cycles participants.
        let ((final_entry, result), dep_node) =
            tcx.dep_graph.with_anon_task(tcx, dep_kinds::TraitSelect, || {
                for _ in 0..FIXPOINT_STEP_LIMIT {
                    match self.fixpoint_step_in_task(tcx, input, inspect, &mut prove_goal) {
                        StepResult::Done(final_entry, result) => return (final_entry, result),
                        StepResult::HasChanged => {}
                    }
                }

                debug!("canonical cycle overflow");
                let current_entry = self.pop_stack();
                debug_assert!(current_entry.has_been_used.is_empty());
                let result = response_no_constraints(tcx, input, Certainty::overflow(false));
                (current_entry, result)
            });

        let proof_tree = inspect.finalize_evaluation(tcx);

        // We're now done with this goal. In case this goal is involved in a larger cycle
        // do not remove it from the provisional cache and update its provisional result.
        // We only add the root of cycles to the global cache.
        if final_entry.heads.is_empty() {
            let depth = self.stack.next_index();
            let provisional_cache_entry = self.provisional_cache.remove(&input);
            assert_eq!(provisional_cache_entry.unwrap().stack_depth, Some(depth));
            // When encountering a cycle, both inductive and coinductive, we only
            // move the root into the global cache. We also store all other cycle
            // participants involved.
            //
            // We must not use the global cache entry of a root goal if a cycle
            // participant is on the stack. This is necessary to prevent unstable
            // results. See the comment of `SearchGraph::cycle_participants` for
            // more details.
            let cycle_participants =
                if let Some(cycle_data) = self.cycle_data.take_if(|data| data.root == depth) {
                    cycle_data.cycle_participants
                } else {
                    Default::default()
                };

            let reached_depth = final_entry.reached_depth.as_usize() - self.stack.len();
            self.global_cache(tcx).insert(
                tcx,
                input,
                proof_tree,
                reached_depth,
                final_entry.encountered_overflow,
                cycle_participants,
                dep_node,
                result,
            )
        } else {
            let heads = final_entry.heads;
            let head = *heads.last_key_value().unwrap().0;
            debug_assert!(head < self.stack.next_index());
            debug_assert_ne!(self.stack[head].has_been_used, HasBeenUsed::empty());

            let coinductive_stack = Self::stack_coinductive_from(tcx, &self.stack, head);
            let entry = self.provisional_cache.get_mut(&input).unwrap();
            entry.stack_depth = None;
            if coinductive_stack {
                entry.with_coinductive_stack = Some(DetachedEntry { heads, result });
            } else {
                entry.with_inductive_stack = Some(DetachedEntry { heads, result });
            }
        }

        self.check_invariants();
        result
    }

    /// Try to fetch a previously computed result from the global cache,
    /// making sure to only do so if it would match the result of reevaluating
    /// this goal.
    fn lookup_global_cache(
        &mut self,
        tcx: TyCtxt<'tcx>,
        input: CanonicalInput<'tcx>,
        available_depth: Limit,
        inspect: &mut ProofTreeBuilder<'tcx>,
    ) -> Option<QueryResult<'tcx>> {
        let CacheData { result, proof_tree, additional_depth, encountered_overflow } = self
            .global_cache(tcx)
            .get(tcx, input, self.stack.iter().map(|e| e.input), available_depth)?;

        // If we're building a proof tree and the current cache entry does not
        // contain a proof tree, we do not use the entry but instead recompute
        // the goal. We simply overwrite the existing entry once we're done,
        // caching the proof tree.
        if !inspect.is_noop() {
            if let Some(revisions) = proof_tree {
                let kind = inspect::WipCanonicalGoalEvaluationKind::Interned { revisions };
                inspect.goal_evaluation_kind(kind);
            } else {
                return None;
            }
        }

        // Update the reached depth of the current goal to make sure
        // its state is the same regardless of whether we've used the
        // global cache or not.
        let reached_depth = self.stack.next_index().plus(additional_depth);
        if let Some(last) = self.stack.raw.last_mut() {
            last.reached_depth = last.reached_depth.max(reached_depth);
            last.encountered_overflow |= encountered_overflow;
        }

        Some(result)
    }
}

enum StepResult<'tcx> {
    Done(StackEntry<'tcx>, QueryResult<'tcx>),
    HasChanged,
}

impl<'tcx> SearchGraph<'tcx> {
    /// When we encounter a coinductive cycle, we have to fetch the
    /// result of that cycle while we are still computing it. Because
    /// of this we continuously recompute the cycle until the result
    /// of the previous iteration is equal to the final result, at which
    /// point we are done.
    fn fixpoint_step_in_task<F>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        input: CanonicalInput<'tcx>,
        inspect: &mut ProofTreeBuilder<'tcx>,
        prove_goal: &mut F,
    ) -> StepResult<'tcx>
    where
        F: FnMut(&mut Self, &mut ProofTreeBuilder<'tcx>) -> QueryResult<'tcx>,
    {
        self.check_invariants();
        let result = prove_goal(self, inspect);
        let stack_entry = self.pop_stack();
        debug_assert_eq!(stack_entry.input, input);

        // If the current goal is not the root of a cycle, we are done.
        if stack_entry.has_been_used.is_empty() {
            Self::clear_dependent_provisional_results(
                &mut self.provisional_cache,
                self.stack.next_index(),
                |_, co, ind| {
                    if co.is_some() || ind.is_some() {
                        bug!()
                    }
                },
            );
            return StepResult::Done(stack_entry, result);
        }

        // If it is a cycle head, we have to keep trying to prove it until
        // we reach a fixpoint. We need to do so for all cycle heads,
        // not only for the root.
        //
        // See tests/ui/traits/next-solver/cycles/fixpoint-rerun-all-cycle-heads.rs
        // for an example.

        // Check whether we reached a fixpoint, either because the final result
        // is equal to the provisional result of the previous iteration, or because
        // this was only the root of either coinductive or inductive cycles, and the
        // final result is equal to the initial response for that case.
        let reached_fixpoint = if let Some(r) = stack_entry.provisional_result {
            r == result
        } else if stack_entry.has_been_used == HasBeenUsed::COINDUCTIVE_CYCLE {
            response_no_constraints(tcx, input, Certainty::Yes) == result
        } else if stack_entry.has_been_used == HasBeenUsed::INDUCTIVE_CYCLE {
            response_no_constraints(tcx, input, Certainty::overflow(false)) == result
        } else {
            false
        };

        if reached_fixpoint {
            // If we're done, move this goal and all detached cache entries which depend on
            // it into the fixpoint instantiation cache.
            let cycle_data = self.cycle_data.as_mut().unwrap();
            let mut detached_cache_entries = FxHashMap::default();
            Self::clear_dependent_provisional_results(
                &mut self.provisional_cache,
                self.stack.next_index(),
                |input, coinductive, inductive| {
                    let (entry, is_coinductive) = match (coinductive, inductive) {
                        (Some(entry), None) => (entry, true),
                        (None, Some(entry)) => (entry, false),
                        _ => return,
                    };

                    let mut provisional_results = Vec::new();
                    for (&head, &usage_kind) in &entry.heads {
                        // We compute this after already popping the current goal from the stack.
                        let elem = &self.stack.get(head).unwrap_or(&stack_entry);
                        provisional_results.push(if let Some(result) = elem.provisional_result {
                            result
                        } else if usage_kind == HasBeenUsed::COINDUCTIVE_CYCLE {
                            response_no_constraints(tcx, elem.input, Certainty::Yes)
                        } else if usage_kind == HasBeenUsed::COINDUCTIVE_CYCLE {
                            response_no_constraints(tcx, elem.input, Certainty::overflow(false))
                        } else {
                            return; // just bail and ignore that result in this case for now
                        });
                    }
                    let entry = ReuseableDetachedEntry {
                        heads: entry.heads,
                        provisional_results,
                        is_coinductive,
                        result: entry.result,
                    };
                    assert!(detached_cache_entries.insert(input, entry).is_none());
                },
            );
            cycle_data.add_fixpoint_instantiation_cache_entry(
                &self.stack,
                &stack_entry,
                result,
                detached_cache_entries,
            );
            StepResult::Done(stack_entry, result)
        } else {
            // If not, recompute after throwing out all provisional cache
            // entries which depend on the current goal. We then update
            // the provisional result and recompute.
            Self::clear_dependent_provisional_results(
                &mut self.provisional_cache,
                self.stack.next_index(),
                |_, _, _| {},
            );
            let depth = self.stack.push(StackEntry {
                has_been_used: HasBeenUsed::empty(),
                provisional_result: Some(result),
                heads: Default::default(),
                ..stack_entry
            });
            debug_assert_eq!(self.provisional_cache[&input].stack_depth, Some(depth));
            StepResult::HasChanged
        }
    }
}

fn response_no_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: CanonicalInput<'tcx>,
    certainty: Certainty,
) -> QueryResult<'tcx> {
    Ok(super::response_no_constraints_raw(tcx, goal.max_universe, goal.variables, certainty))
}

impl<'tcx> SearchGraph<'tcx> {
    fn check_invariants(&self) {
        if !cfg!(debug_assertions) {
            return;
        }

        let SearchGraph { mode: _, stack, cycle_data, provisional_cache } = self;
        if stack.is_empty() {
            debug_assert!(cycle_data.is_none());
            debug_assert!(provisional_cache.is_empty());
        }

        for (depth, entry) in stack.iter_enumerated() {
            let StackEntry {
                input: _,
                available_depth: _,
                reached_depth: _,
                heads,
                encountered_overflow: _,
                has_been_used: _,
                provisional_result,
            } = entry;
            let cache_entry = provisional_cache.get(&entry.input).unwrap();
            assert_eq!(cache_entry.stack_depth, Some(depth));
            for (&head, &usage_kind) in heads {
                assert!(head < depth);
                assert!(cycle_data.is_some());
                assert_ne!(usage_kind, HasBeenUsed::empty());
                stack[head].has_been_used.contains(usage_kind);
            }

            if provisional_result.is_some() {
                assert!(cycle_data.is_some());
            }
        }

        for (&input, entry) in &self.provisional_cache {
            let ProvisionalCacheEntry { stack_depth, with_coinductive_stack, with_inductive_stack } =
                entry;
            assert!(
                stack_depth.is_some()
                    || with_coinductive_stack.is_some()
                    || with_inductive_stack.is_some()
            );

            if let &Some(stack_depth) = stack_depth {
                assert_eq!(stack[stack_depth].input, input);
            }

            let check_detached = |detached_entry: &DetachedEntry<'tcx>| {
                assert!(cycle_data.is_some());
                let DetachedEntry { heads, result: _ } = detached_entry;
                for (&head, &usage_kind) in heads {
                    assert_ne!(usage_kind, HasBeenUsed::empty());
                    stack[head].has_been_used.contains(usage_kind);
                }
            };

            if let Some(with_coinductive_stack) = with_coinductive_stack {
                check_detached(with_coinductive_stack);
            }

            if let Some(with_inductive_stack) = with_inductive_stack {
                check_detached(with_inductive_stack);
            }
        }
    }
}
