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
    // In case of a cycle, the depth of the root.
    cycle_root_depth: StackDepth,

    encountered_overflow: bool,
    has_been_used: bool,
    /// Starts out as `None` and gets set when rerunning this
    /// goal in case we encounter a cycle.
    provisional_result: Option<QueryResult<'tcx>>,

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
    stack_entries: FxHashMap<CanonicalInput<'tcx>, StackDepth>,
}

impl<'tcx> SearchGraph<'tcx> {
    pub(super) fn new(tcx: TyCtxt<'tcx>, mode: SolverMode) -> SearchGraph<'tcx> {
        Self {
            mode,
            local_overflow_limit: tcx.recursion_limit().0.checked_ilog2().unwrap_or(0) as usize,
            stack: Default::default(),
            stack_entries: Default::default(),
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
        assert!(self.stack_entries.remove(&elem.input).is_some());
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
        self.stack.is_empty()
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

            inspect.goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::Overflow);
            return Self::response_no_constraints(tcx, input, Certainty::OVERFLOW);
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

        // Check whether we're in a cycle.
        match self.stack_entries.entry(input) {
            // No entry, we push this goal on the stack and try to prove it.
            Entry::Vacant(v) => {
                let depth = self.stack.next_index();
                let entry = StackEntry {
                    input,
                    available_depth,
                    reached_depth: depth,
                    cycle_root_depth: depth,
                    encountered_overflow: false,
                    has_been_used: false,
                    provisional_result: None,
                    cycle_participants: Default::default(),
                };
                assert_eq!(self.stack.push(entry), depth);
                v.insert(depth);
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
            Entry::Occupied(entry) => {
                inspect.goal_evaluation_kind(inspect::WipCanonicalGoalEvaluationKind::CycleInStack);

                let stack_depth = *entry.get();
                debug!("encountered cycle with depth {stack_depth:?}");
                // We start by updating the root depth of all cycle participants, and
                // add all cycle participants to the root.
                let root_depth = self.stack[stack_depth].cycle_root_depth;
                let (prev, participants) = self.stack.raw.split_at_mut(stack_depth.as_usize() + 1);
                let root = &mut prev[root_depth.as_usize()];
                for entry in participants {
                    debug_assert!(entry.cycle_root_depth >= root_depth);
                    entry.cycle_root_depth = root_depth;
                    root.cycle_participants.insert(entry.input);
                    // FIXME(@lcnr): I believe that this line is needed as we could
                    // otherwise access a cache entry for the root of a cycle while
                    // computing the result for a cycle participant. This can result
                    // in unstable results due to incompleteness.
                    //
                    // However, a test for this would be an even more complex version of
                    // tests/ui/traits/new-solver/coinduction/incompleteness-unstable-result.rs.
                    // I did not bother to write such a test and we have no regression test
                    // for this. It would be good to have such a test :)
                    #[allow(rustc::potential_query_instability)]
                    root.cycle_participants.extend(entry.cycle_participants.drain());
                }

                // If we're in a cycle, we have to retry proving the cycle head
                // until we reach a fixpoint. It is not enough to simply retry the
                // `root` goal of this cycle.
                //
                // See tests/ui/traits/new-solver/cycles/fixpoint-rerun-all-cycle-heads.rs
                // for an example.
                self.stack[stack_depth].has_been_used = true;
                return if let Some(result) = self.stack[stack_depth].provisional_result {
                    result
                } else {
                    // If we don't have a provisional result yet we're in the first iteration,
                    // so we start with no constraints.
                    let is_coinductive = self.stack.raw[stack_depth.index()..]
                        .iter()
                        .all(|entry| entry.input.value.goal.predicate.is_coinductive(tcx));
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
            tcx.dep_graph.with_anon_task(tcx, dep_kinds::TraitSelect, || {
                // When we encounter a coinductive cycle, we have to fetch the
                // result of that cycle while we are still computing it. Because
                // of this we continuously recompute the cycle until the result
                // of the previous iteration is equal to the final result, at which
                // point we are done.
                for _ in 0..self.local_overflow_limit() {
                    let result = prove_goal(self, inspect);

                    // Check whether the current goal is the root of a cycle and whether
                    // we have to rerun because its provisional result differed from the
                    // final result.
                    let stack_entry = self.pop_stack();
                    debug_assert_eq!(stack_entry.input, input);
                    if stack_entry.has_been_used
                        && stack_entry.provisional_result.map_or(true, |r| r != result)
                    {
                        // If so, update its provisional result and reevaluate it.
                        let depth = self.stack.push(StackEntry {
                            has_been_used: false,
                            provisional_result: Some(result),
                            ..stack_entry
                        });
                        assert_eq!(self.stack_entries.insert(input, depth), None);
                    } else {
                        return (stack_entry, result);
                    }
                }

                debug!("canonical cycle overflow");
                let current_entry = self.pop_stack();
                let result = Self::response_no_constraints(tcx, input, Certainty::OVERFLOW);
                (current_entry, result)
            });

        let proof_tree = inspect.finalize_evaluation(tcx);

        // We're now done with this goal. In case this goal is involved in a larger cycle
        // do not remove it from the provisional cache and update its provisional result.
        // We only add the root of cycles to the global cache.
        //
        // It is not possible for any nested goal to depend on something deeper on the
        // stack, as this would have also updated the depth of the current goal.
        if final_entry.cycle_root_depth == self.stack.next_index() {
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
                tcx,
                input,
                proof_tree,
                reached_depth,
                final_entry.encountered_overflow,
                final_entry.cycle_participants,
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
