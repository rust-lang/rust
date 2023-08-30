mod cache;

use self::cache::ProvisionalEntry;
use super::inspect::ProofTreeBuilder;
use super::SolverMode;
use cache::ProvisionalCache;
use rustc_data_structures::fx::FxHashSet;
use rustc_index::Idx;
use rustc_index::IndexVec;
use rustc_middle::dep_graph::DepKind;
use rustc_middle::traits::solve::inspect::CacheHit;
use rustc_middle::traits::solve::CacheData;
use rustc_middle::traits::solve::{CanonicalInput, Certainty, EvaluationCache, QueryResult};
use rustc_middle::ty::TyCtxt;
use rustc_session::Limit;
use std::collections::hash_map::Entry;

rustc_index::newtype_index! {
    pub struct StackDepth {}
}

#[derive(Debug)]
struct StackEntry<'tcx> {
    input: CanonicalInput<'tcx>,
    available_depth: Limit,
    // The maximum depth reached by this stack entry, only up-to date
    // for the top of the stack and lazily updated for the rest.
    reached_depth: StackDepth,
    encountered_overflow: bool,
    has_been_used: bool,

    /// We put only the root goal of a coinductive cycle into the global cache.
    ///
    /// If we were to use that result when later trying to prove another cycle
    /// participant, we can end up with unstable query results.
    ///
    /// See tests/ui/new-solver/coinduction/incompleteness-unstable-result.rs for
    /// an example of where this is needed.
    cycle_participants: FxHashSet<CanonicalInput<'tcx>>,
}

pub(super) struct SearchGraph<'tcx> {
    mode: SolverMode,
    local_overflow_limit: usize,
    /// The stack of goals currently being computed.
    ///
    /// An element is *deeper* in the stack if its index is *lower*.
    stack: IndexVec<StackDepth, StackEntry<'tcx>>,
    provisional_cache: ProvisionalCache<'tcx>,
}

impl<'tcx> SearchGraph<'tcx> {
    pub(super) fn new(tcx: TyCtxt<'tcx>, mode: SolverMode) -> SearchGraph<'tcx> {
        Self {
            mode,
            local_overflow_limit: tcx.recursion_limit().0.checked_ilog2().unwrap_or(0) as usize,
            stack: Default::default(),
            provisional_cache: ProvisionalCache::empty(),
        }
    }

    pub(super) fn solver_mode(&self) -> SolverMode {
        self.mode
    }

    pub(super) fn local_overflow_limit(&self) -> usize {
        self.local_overflow_limit
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
        self.stack.is_empty() && self.provisional_cache.is_empty()
    }

    /// Whether we're currently in a cycle. This should only be used
    /// for debug assertions.
    pub(super) fn in_cycle(&self) -> bool {
        if let Some(stack_depth) = self.stack.last_index() {
            // Either the current goal on the stack is the root of a cycle...
            if self.stack[stack_depth].has_been_used {
                return true;
            }

            // ...or it depends on a goal with a lower depth.
            let current_goal = self.stack[stack_depth].input;
            let entry_index = self.provisional_cache.lookup_table[&current_goal];
            self.provisional_cache.entries[entry_index].depth != stack_depth
        } else {
            false
        }
    }

    /// Fetches whether the current goal encountered overflow.
    ///
    /// This should only be used for the check in `evaluate_goal`.
    pub(super) fn encountered_overflow(&self) -> bool {
        if let Some(last) = self.stack.raw.last() { last.encountered_overflow } else { false }
    }

    /// Resets `encountered_overflow` of the current goal.
    ///
    /// This should only be used for the check in `evaluate_goal`.
    pub(super) fn reset_encountered_overflow(&mut self, encountered_overflow: bool) -> bool {
        if let Some(last) = self.stack.raw.last_mut() {
            let prev = last.encountered_overflow;
            last.encountered_overflow = encountered_overflow;
            prev
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
            return Self::response_no_constraints(tcx, input, Certainty::OVERFLOW);
        };

        // Try to fetch the goal from the global cache.
        if inspect.use_global_cache() {
            if let Some(CacheData { result, reached_depth, encountered_overflow }) =
                self.global_cache(tcx).get(
                    tcx,
                    input,
                    |cycle_participants| {
                        self.stack.iter().any(|entry| cycle_participants.contains(&entry.input))
                    },
                    available_depth,
                )
            {
                self.on_cache_hit(reached_depth, encountered_overflow);
                return result;
            }
        }

        // Look at the provisional cache to detect cycles.
        let cache = &mut self.provisional_cache;
        match cache.lookup_table.entry(input) {
            // No entry, we push this goal on the stack and try to prove it.
            Entry::Vacant(v) => {
                let depth = self.stack.next_index();
                let entry = StackEntry {
                    input,
                    available_depth,
                    reached_depth: depth,
                    encountered_overflow: false,
                    has_been_used: false,
                    cycle_participants: Default::default(),
                };
                assert_eq!(self.stack.push(entry), depth);
                let entry_index =
                    cache.entries.push(ProvisionalEntry { response: None, depth, input });
                v.insert(entry_index);
            }
            // We have a nested goal which relies on a goal `root` deeper in the stack.
            //
            // We first store that we may have to reprove `root` in case the provisional
            // response is not equal to the final response. We also update the depth of all
            // goals which recursively depend on our current goal to depend on `root`
            // instead.
            //
            // Finally we can return either the provisional response for that goal if we have a
            // coinductive cycle or an ambiguous result if the cycle is inductive.
            Entry::Occupied(entry_index) => {
                inspect.cache_hit(CacheHit::Provisional);

                let entry_index = *entry_index.get();
                let stack_depth = cache.depth(entry_index);
                debug!("encountered cycle with depth {stack_depth:?}");

                cache.add_dependency_of_leaf_on(entry_index);
                let mut iter = self.stack.iter_mut();
                let root = iter.nth(stack_depth.as_usize()).unwrap();
                for e in iter {
                    root.cycle_participants.insert(e.input);
                }

                // If we're in a cycle, we have to retry proving the current goal
                // until we reach a fixpoint.
                self.stack[stack_depth].has_been_used = true;
                return if let Some(result) = cache.provisional_result(entry_index) {
                    result
                } else {
                    // If we don't have a provisional result yet, the goal has to
                    // still be on the stack.
                    let mut goal_on_stack = false;
                    let mut is_coinductive = true;
                    for entry in self.stack.raw[stack_depth.index()..]
                        .iter()
                        .skip_while(|entry| entry.input != input)
                    {
                        goal_on_stack = true;
                        is_coinductive &= entry.input.value.goal.predicate.is_coinductive(tcx);
                    }
                    debug_assert!(goal_on_stack);

                    if is_coinductive {
                        Self::response_no_constraints(tcx, input, Certainty::Yes)
                    } else {
                        Self::response_no_constraints(tcx, input, Certainty::OVERFLOW)
                    }
                };
            }
        }

        // This is for global caching, so we properly track query dependencies.
        // Everything that affects the `result` should be performed within this
        // `with_anon_task` closure.
        let ((final_entry, result), dep_node) =
            tcx.dep_graph.with_anon_task(tcx, DepKind::TraitSelect, || {
                // When we encounter a coinductive cycle, we have to fetch the
                // result of that cycle while we are still computing it. Because
                // of this we continuously recompute the cycle until the result
                // of the previous iteration is equal to the final result, at which
                // point we are done.
                for _ in 0..self.local_overflow_limit() {
                    let response = prove_goal(self, inspect);

                    // Check whether the current goal is the root of a cycle and whether
                    // we have to rerun because its provisional result differed from the
                    // final result.
                    //
                    // Also update the response for this goal stored in the provisional
                    // cache.
                    let stack_entry = self.pop_stack();
                    debug_assert_eq!(stack_entry.input, input);
                    let cache = &mut self.provisional_cache;
                    let provisional_entry_index =
                        *cache.lookup_table.get(&stack_entry.input).unwrap();
                    let provisional_entry = &mut cache.entries[provisional_entry_index];
                    if stack_entry.has_been_used
                        && provisional_entry.response.map_or(true, |r| r != response)
                    {
                        // If so, update the provisional result for this goal and remove
                        // all entries whose result depends on this goal from the provisional
                        // cache...
                        //
                        // That's not completely correct, as a nested goal can also only
                        // depend on a goal which is lower in the stack so it doesn't
                        // actually depend on the current goal. This should be fairly
                        // rare and is hopefully not relevant for performance.
                        provisional_entry.response = Some(response);
                        #[allow(rustc::potential_query_instability)]
                        cache.lookup_table.retain(|_key, index| *index <= provisional_entry_index);
                        cache.entries.truncate(provisional_entry_index.index() + 1);

                        // ...and finally push our goal back on the stack and reevaluate it.
                        self.stack.push(StackEntry { has_been_used: false, ..stack_entry });
                    } else {
                        return (stack_entry, response);
                    }
                }

                debug!("canonical cycle overflow");
                let current_entry = self.pop_stack();
                let result = Self::response_no_constraints(tcx, input, Certainty::OVERFLOW);
                (current_entry, result)
            });

        // We're now done with this goal. In case this goal is involved in a larger cycle
        // do not remove it from the provisional cache and update its provisional result.
        // We only add the root of cycles to the global cache.
        //
        // It is not possible for any nested goal to depend on something deeper on the
        // stack, as this would have also updated the depth of the current goal.
        let cache = &mut self.provisional_cache;
        let provisional_entry_index = *cache.lookup_table.get(&input).unwrap();
        let provisional_entry = &mut cache.entries[provisional_entry_index];
        let depth = provisional_entry.depth;
        if depth == self.stack.next_index() {
            for (i, entry) in cache.entries.drain_enumerated(provisional_entry_index.index()..) {
                let actual_index = cache.lookup_table.remove(&entry.input);
                debug_assert_eq!(Some(i), actual_index);
                debug_assert!(entry.depth == depth);
            }

            // When encountering a cycle, both inductive and coinductive, we only
            // move the root into the global cache. We also store all other cycle
            // participants involved.
            //
            // We disable the global cache entry of the root goal if a cycle
            // participant is on the stack. This is necessary to prevent unstable
            // results. See the comment of `StackEntry::cycle_participants` for
            // more details.
            let reached_depth = final_entry.reached_depth.as_usize() - self.stack.len();
            self.global_cache(tcx).insert(
                input,
                reached_depth,
                final_entry.encountered_overflow,
                final_entry.cycle_participants,
                dep_node,
                result,
            )
        } else {
            provisional_entry.response = Some(result);
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
