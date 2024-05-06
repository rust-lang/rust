use crate::solve::FIXPOINT_STEP_LIMIT;

use super::inspect;
use super::inspect::ProofTreeBuilder;
use super::SolverMode;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::fx::FxHashSet;
use rustc_index::Idx;
use rustc_index::IndexVec;
use rustc_middle::dep_graph::dep_kinds;
use rustc_middle::traits::solve::CacheData;
use rustc_middle::traits::solve::{CanonicalInput, Certainty, EvaluationCache, QueryResult};
use rustc_middle::ty::TyCtxt;
use rustc_session::Limit;
use std::mem;

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

#[derive(Debug)]
struct StackEntry<'tcx> {
    input: CanonicalInput<'tcx>,

    available_depth: Limit,

    /// The maximum depth reached by this stack entry, only up-to date
    /// for the top of the stack and lazily updated for the rest.
    reached_depth: StackDepth,

    /// Whether this entry is a non-root cycle participant.
    ///
    /// We must not move the result of non-root cycle participants to the
    /// global cache. See [SearchGraph::cycle_participants] for more details.
    /// We store the highest stack depth of a head of a cycle this goal is involved
    /// in. This necessary to soundly cache its provisional result.
    non_root_cycle_participant: Option<StackDepth>,

    encountered_overflow: bool,

    has_been_used: HasBeenUsed,
    /// Starts out as `None` and gets set when rerunning this
    /// goal in case we encounter a cycle.
    provisional_result: Option<QueryResult<'tcx>>,
}

/// The provisional result for a goal which is not on the stack.
struct DetachedEntry<'tcx> {
    /// The head of the smallest non-trivial cycle involving this entry.
    ///
    /// Given the following rules, when proving `A` the head for
    /// the provisional entry of `C` would be `B`.
    /// ```plain
    /// A :- B
    /// B :- C
    /// C :- A + B + C
    /// ```
    head: StackDepth,
    result: QueryResult<'tcx>,
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
#[derive(Default)]
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

pub(super) struct SearchGraph<'tcx> {
    mode: SolverMode,
    /// The stack of goals currently being computed.
    ///
    /// An element is *deeper* in the stack if its index is *lower*.
    stack: IndexVec<StackDepth, StackEntry<'tcx>>,
    provisional_cache: FxHashMap<CanonicalInput<'tcx>, ProvisionalCacheEntry<'tcx>>,
    /// We put only the root goal of a coinductive cycle into the global cache.
    ///
    /// If we were to use that result when later trying to prove another cycle
    /// participant, we can end up with unstable query results.
    ///
    /// See tests/ui/next-solver/coinduction/incompleteness-unstable-result.rs for
    /// an example of where this is needed.
    cycle_participants: FxHashSet<CanonicalInput<'tcx>>,
}

impl<'tcx> SearchGraph<'tcx> {
    pub(super) fn new(mode: SolverMode) -> SearchGraph<'tcx> {
        Self {
            mode,
            stack: Default::default(),
            provisional_cache: Default::default(),
            cycle_participants: Default::default(),
        }
    }

    pub(super) fn solver_mode(&self) -> SolverMode {
        self.mode
    }

    /// Update the stack and reached depths on cache hits.
    #[instrument(level = "debug", skip(self))]
    fn on_cache_hit(&mut self, additional_depth: usize, encountered_overflow: bool) {
        let reached_depth = self.stack.next_index().plus(additional_depth);
        if let Some(last) = self.stack.raw.last_mut() {
            last.reached_depth = last.reached_depth.max(reached_depth);
            last.encountered_overflow |= encountered_overflow;
        }
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
        if self.stack.is_empty() {
            debug_assert!(self.provisional_cache.is_empty());
            debug_assert!(self.cycle_participants.is_empty());
            true
        } else {
            false
        }
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

    // When encountering a solver cycle, the result of the current goal
    // depends on goals lower on the stack.
    //
    // We have to therefore be careful when caching goals. Only the final result
    // of the cycle root, i.e. the lowest goal on the stack involved in this cycle,
    // is moved to the global cache while all others are stored in a provisional cache.
    //
    // We update both the head of this cycle to rerun its evaluation until
    // we reach a fixpoint and all other cycle participants to make sure that
    // their result does not get moved to the global cache.
    fn tag_cycle_participants(
        stack: &mut IndexVec<StackDepth, StackEntry<'tcx>>,
        cycle_participants: &mut FxHashSet<CanonicalInput<'tcx>>,
        usage_kind: HasBeenUsed,
        head: StackDepth,
    ) {
        stack[head].has_been_used |= usage_kind;
        debug_assert!(!stack[head].has_been_used.is_empty());
        for entry in &mut stack.raw[head.index() + 1..] {
            entry.non_root_cycle_participant = entry.non_root_cycle_participant.max(Some(head));
            cycle_participants.insert(entry.input);
        }
    }

    fn clear_dependent_provisional_results(
        provisional_cache: &mut FxHashMap<CanonicalInput<'tcx>, ProvisionalCacheEntry<'tcx>>,
        head: StackDepth,
    ) {
        #[allow(rustc::potential_query_instability)]
        provisional_cache.retain(|_, entry| {
            entry.with_coinductive_stack.take_if(|p| p.head == head);
            entry.with_inductive_stack.take_if(|p| p.head == head);
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
        // Check for overflow.
        let Some(available_depth) = Self::allowed_depth_for_nested(tcx, &self.stack) else {
            if let Some(last) = self.stack.raw.last_mut() {
                last.encountered_overflow = true;
            }

            inspect.goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::Overflow);
            return Self::response_no_constraints(tcx, input, Certainty::overflow(true));
        };

        // Try to fetch the goal from the global cache.
        'global: {
            let Some(CacheData { result, proof_tree, reached_depth, encountered_overflow }) =
                self.global_cache(tcx).get(
                    tcx,
                    input,
                    |cycle_participants| {
                        self.stack.iter().any(|entry| cycle_participants.contains(&entry.input))
                    },
                    available_depth,
                )
            else {
                break 'global;
            };

            // If we're building a proof tree and the current cache entry does not
            // contain a proof tree, we do not use the entry but instead recompute
            // the goal. We simply overwrite the existing entry once we're done,
            // caching the proof tree.
            if !inspect.is_noop() {
                if let Some(revisions) = proof_tree {
                    inspect.goal_evaluation_kind(
                        inspect::WipCanonicalGoalEvaluationKind::Interned { revisions },
                    );
                } else {
                    break 'global;
                }
            }

            self.on_cache_hit(reached_depth, encountered_overflow);
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
            .filter(|p| Self::stack_coinductive_from(tcx, &self.stack, p.head))
            .or_else(|| {
                cache_entry
                    .with_inductive_stack
                    .as_ref()
                    .filter(|p| !Self::stack_coinductive_from(tcx, &self.stack, p.head))
            })
        {
            // We have a nested goal which is already in the provisional cache, use
            // its result. We do not provide any usage kind as that should have been
            // already set correctly while computing the cache entry.
            inspect
                .goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::ProvisionalCacheHit);
            Self::tag_cycle_participants(
                &mut self.stack,
                &mut self.cycle_participants,
                HasBeenUsed::empty(),
                entry.head,
            );
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
            Self::tag_cycle_participants(
                &mut self.stack,
                &mut self.cycle_participants,
                usage_kind,
                stack_depth,
            );

            // Return the provisional result or, if we're in the first iteration,
            // start with no constraints.
            return if let Some(result) = self.stack[stack_depth].provisional_result {
                result
            } else if is_coinductive_cycle {
                Self::response_no_constraints(tcx, input, Certainty::Yes)
            } else {
                Self::response_no_constraints(tcx, input, Certainty::overflow(false))
            };
        } else {
            // No entry, we push this goal on the stack and try to prove it.
            let depth = self.stack.next_index();
            let entry = StackEntry {
                input,
                available_depth,
                reached_depth: depth,
                non_root_cycle_participant: None,
                encountered_overflow: false,
                has_been_used: HasBeenUsed::empty(),
                provisional_result: None,
            };
            assert_eq!(self.stack.push(entry), depth);
            cache_entry.stack_depth = Some(depth);
        }

        // This is for global caching, so we properly track query dependencies.
        // Everything that affects the `result` should be performed within this
        // `with_anon_task` closure.
        let ((final_entry, result), dep_node) =
            tcx.dep_graph.with_anon_task(tcx, dep_kinds::TraitSelect, || {
                // When we encounter a coinductive cycle, we have to fetch the
                // result of that cycle while we are still computing it. Because
                // of this we continuously recompute the cycle until the result
                // of the previous iteration is equal to the final result, at which
                // point we are done.
                for _ in 0..FIXPOINT_STEP_LIMIT {
                    let result = prove_goal(self, inspect);
                    let stack_entry = self.pop_stack();
                    debug_assert_eq!(stack_entry.input, input);

                    // If the current goal is not the root of a cycle, we are done.
                    if stack_entry.has_been_used.is_empty() {
                        return (stack_entry, result);
                    }

                    // If it is a cycle head, we have to keep trying to prove it until
                    // we reach a fixpoint. We need to do so for all cycle heads,
                    // not only for the root.
                    //
                    // See tests/ui/traits/next-solver/cycles/fixpoint-rerun-all-cycle-heads.rs
                    // for an example.

                    // Start by clearing all provisional cache entries which depend on this
                    // the current goal.
                    Self::clear_dependent_provisional_results(
                        &mut self.provisional_cache,
                        self.stack.next_index(),
                    );

                    // Check whether we reached a fixpoint, either because the final result
                    // is equal to the provisional result of the previous iteration, or because
                    // this was only the root of either coinductive or inductive cycles, and the
                    // final result is equal to the initial response for that case.
                    let reached_fixpoint = if let Some(r) = stack_entry.provisional_result {
                        r == result
                    } else if stack_entry.has_been_used == HasBeenUsed::COINDUCTIVE_CYCLE {
                        Self::response_no_constraints(tcx, input, Certainty::Yes) == result
                    } else if stack_entry.has_been_used == HasBeenUsed::INDUCTIVE_CYCLE {
                        Self::response_no_constraints(tcx, input, Certainty::overflow(false))
                            == result
                    } else {
                        false
                    };

                    // If we did not reach a fixpoint, update the provisional result and reevaluate.
                    if reached_fixpoint {
                        return (stack_entry, result);
                    } else {
                        let depth = self.stack.push(StackEntry {
                            has_been_used: HasBeenUsed::empty(),
                            provisional_result: Some(result),
                            ..stack_entry
                        });
                        debug_assert_eq!(self.provisional_cache[&input].stack_depth, Some(depth));
                    }
                }

                debug!("canonical cycle overflow");
                let current_entry = self.pop_stack();
                debug_assert!(current_entry.has_been_used.is_empty());
                let result = Self::response_no_constraints(tcx, input, Certainty::overflow(false));
                (current_entry, result)
            });

        let proof_tree = inspect.finalize_evaluation(tcx);

        // We're now done with this goal. In case this goal is involved in a larger cycle
        // do not remove it from the provisional cache and update its provisional result.
        // We only add the root of cycles to the global cache.
        if let Some(head) = final_entry.non_root_cycle_participant {
            let coinductive_stack = Self::stack_coinductive_from(tcx, &self.stack, head);

            let entry = self.provisional_cache.get_mut(&input).unwrap();
            entry.stack_depth = None;
            if coinductive_stack {
                entry.with_coinductive_stack = Some(DetachedEntry { head, result });
            } else {
                entry.with_inductive_stack = Some(DetachedEntry { head, result });
            }
        } else {
            self.provisional_cache.remove(&input);
            let reached_depth = final_entry.reached_depth.as_usize() - self.stack.len();
            let cycle_participants = mem::take(&mut self.cycle_participants);
            // When encountering a cycle, both inductive and coinductive, we only
            // move the root into the global cache. We also store all other cycle
            // participants involved.
            //
            // We must not use the global cache entry of a root goal if a cycle
            // participant is on the stack. This is necessary to prevent unstable
            // results. See the comment of `SearchGraph::cycle_participants` for
            // more details.
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
        }

        result
    }

    fn response_no_constraints(
        tcx: TyCtxt<'tcx>,
        goal: CanonicalInput<'tcx>,
        certainty: Certainty,
    ) -> QueryResult<'tcx> {
        Ok(super::response_no_constraints_raw(tcx, goal.max_universe, goal.variables, certainty))
    }
}
