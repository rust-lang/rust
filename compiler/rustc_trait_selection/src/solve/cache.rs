//! This module both handles the global cache which stores "finished" goals,
//! and the provisional cache which contains partially computed goals.
//!
//! The provisional cache is necessary when dealing with coinductive cycles.
//!
//! For more information about the provisional cache and coinduction in general,
//! check out the relevant section of the rustc-dev-guide.
//!
//! FIXME(@lcnr): Write that section, feel free to ping me if you need help here
//! before then or if I still haven't done that before January 2023.
use super::overflow::OverflowData;
use super::CanonicalGoal;
use super::{EvalCtxt, QueryResult};

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::TyCtxt;
use std::{cmp::Ordering, collections::hash_map::Entry};

#[derive(Debug, Clone)]
struct ProvisionalEntry<'tcx> {
    // In case we have a coinductive cycle, this is the
    // the currently least restrictive result of this goal.
    response: QueryResult<'tcx>,
    // The lowest element on the stack on which this result
    // relies on. Starts out as just being the depth at which
    // we've proven this obligation, but gets lowered to the
    // depth of another goal if we rely on it in a cycle.
    depth: usize,
}

struct StackElem<'tcx> {
    goal: CanonicalGoal<'tcx>,
    has_been_used: bool,
}

/// The cache used for goals which are currently in progress or which depend
/// on in progress results.
///
/// Once we're done with a goal we can store it in the global trait solver
/// cache of the `TyCtxt`. For goals which we're currently proving, or which
/// have only been proven via a coinductive cycle using a goal still on our stack
/// we have to use this separate data structure.
///
/// The current data structure is not perfect, so there may still be room for
/// improvement here. We have the following requirements:
///
/// ## Is there is a provisional entry for the given goal:
///
/// ```ignore (for syntax highlighting)
/// self.entries.get(goal)
/// ```
///
/// ## Get all goals on the stack involved in a cycle:
///
/// ```ignore (for syntax highlighting)
/// let entry = self.entries.get(goal).unwrap();
/// let involved_goals = self.stack.iter().skip(entry.depth);
/// ```
///
/// ## Capping the depth of all entries
///
/// Needed whenever we encounter a cycle. The current implementation always
/// iterates over all entries instead of only the ones with a larger depth.
/// Changing this may result in notable performance improvements.
///
/// ```ignore (for syntax highlighting)
/// let cycle_depth = self.entries.get(goal).unwrap().depth;
/// for e in &mut self.entries {
///     e.depth = e.depth.min(cycle_depth);
/// }
/// ```
///
/// ## Checking whether we have to rerun the current goal
///
/// A goal has to be rerun if its provisional result was used in a cycle
/// and that result is different from its final result. We update
/// [StackElem::has_been_used] for the deepest stack element involved in a cycle.
///
/// ## Moving all finished goals into the global cache
///
/// If `stack_elem.has_been_used` is true, iterate over all entries, moving the ones
/// with equal depth. If not, simply move this single entry.
pub(super) struct ProvisionalCache<'tcx> {
    stack: Vec<StackElem<'tcx>>,
    entries: FxHashMap<CanonicalGoal<'tcx>, ProvisionalEntry<'tcx>>,
}

impl<'tcx> ProvisionalCache<'tcx> {
    pub(super) fn empty() -> ProvisionalCache<'tcx> {
        ProvisionalCache { stack: Vec::new(), entries: Default::default() }
    }

    pub(super) fn current_depth(&self) -> usize {
        self.stack.len()
    }
}

impl<'tcx> EvalCtxt<'tcx> {
    /// Tries putting the new goal on the stack, returning an error if it is already cached.
    ///
    /// This correctly updates the provisional cache if there is a cycle.
    pub(super) fn try_push_stack(
        &mut self,
        goal: CanonicalGoal<'tcx>,
    ) -> Result<(), QueryResult<'tcx>> {
        // FIXME: start by checking the global cache

        // Look at the provisional cache to check for cycles.
        let cache = &mut self.provisional_cache;
        match cache.entries.entry(goal) {
            // No entry, simply push this goal on the stack after dealing with overflow.
            Entry::Vacant(v) => {
                if self.overflow_data.has_overflow(cache.stack.len()) {
                    return Err(self.deal_with_overflow());
                }

                v.insert(ProvisionalEntry {
                    response: fixme_response_yes_no_constraints(),
                    depth: cache.stack.len(),
                });
                cache.stack.push(StackElem { goal, has_been_used: false });
                Ok(())
            }
            // We have a nested goal which relies on a goal `root` deeper in the stack.
            //
            // We first store that we may have to rerun `evaluate_goal` for `root` in case the
            // provisional response is not equal to the final response. We also update the depth
            // of all goals which recursively depend on our current goal to depend on `root`
            // instead.
            //
            // Finally we can return either the provisional response for that goal if we have a
            // coinductive cycle or an ambiguous result if the cycle is inductive.
            Entry::Occupied(entry) => {
                // FIXME: `ProvisionalEntry` should be `Copy`.
                let entry = entry.get().clone();
                cache.stack[entry.depth].has_been_used = true;
                for provisional_entry in cache.entries.values_mut() {
                    provisional_entry.depth = provisional_entry.depth.min(entry.depth);
                }

                // NOTE: The goals on the stack aren't the only goals involved in this cycle.
                // We can also depend on goals which aren't part of the stack but coinductively
                // depend on the stack themselves. We already checked whether all the goals
                // between these goals and their root on the stack. This means that as long as
                // each goal in a cycle is checked for coinductivity by itself simply checking
                // the stack is enough.
                if cache.stack[entry.depth..]
                    .iter()
                    .all(|g| g.goal.value.predicate.is_coinductive(self.tcx))
                {
                    Err(entry.response)
                } else {
                    Err(fixme_response_maybe_no_constraints())
                }
            }
        }
    }

    /// We cannot simply store the result of [EvalCtxt::compute_goal] as we have to deal with
    /// coinductive cycles.
    ///
    /// When we encounter a coinductive cycle, we have to prove the final result of that cycle
    /// while we are still computing that result. Because of this we continously recompute the
    /// cycle until the result of the previous iteration is equal to the final result, at which
    /// point we are done.
    ///
    /// This function returns `true` if we were able to finalize the goal and `false` if it has
    /// updated the provisional cache and we have to recompute the current goal.
    ///
    /// FIXME: Refer to the rustc-dev-guide entry once it exists.
    pub(super) fn try_finalize_goal(
        &mut self,
        actual_goal: CanonicalGoal<'tcx>,
        response: QueryResult<'tcx>,
    ) -> bool {
        let cache = &mut self.provisional_cache;
        let StackElem { goal, has_been_used } = cache.stack.pop().unwrap();
        assert_eq!(goal, actual_goal);

        let provisional_entry = cache.entries.get_mut(&goal).unwrap();
        // Check whether the current stack entry is the root of a cycle.
        //
        // If so, we either move all participants of that cycle to the global cache
        // or, in case the provisional response used in the cycle is not equal to the
        // final response, have to recompute the goal after updating the provisional
        // response to the final response of this iteration.
        if has_been_used {
            if provisional_entry.response == response {
                // We simply drop all entries according to an immutable condition, so
                // query instability is not a concern here.
                #[allow(rustc::potential_query_instability)]
                cache.entries.retain(|goal, entry| match entry.depth.cmp(&cache.stack.len()) {
                    Ordering::Less => true,
                    Ordering::Equal => {
                        Self::try_move_finished_goal_to_global_cache(
                            self.tcx,
                            &mut self.overflow_data,
                            &cache.stack,
                            // FIXME: these should be `Copy` :(
                            goal.clone(),
                            entry.response.clone(),
                        );
                        false
                    }
                    Ordering::Greater => bug!("entry with greater depth than the current leaf"),
                });

                true
            } else {
                provisional_entry.response = response;
                cache.stack.push(StackElem { goal, has_been_used: false });
                false
            }
        } else {
            Self::try_move_finished_goal_to_global_cache(
                self.tcx,
                &mut self.overflow_data,
                &cache.stack,
                goal,
                response,
            );
            cache.entries.remove(&goal);
            true
        }
    }

    fn try_move_finished_goal_to_global_cache(
        tcx: TyCtxt<'tcx>,
        overflow_data: &mut OverflowData,
        stack: &[StackElem<'tcx>],
        goal: CanonicalGoal<'tcx>,
        response: QueryResult<'tcx>,
    ) {
        // We move goals to the global cache if we either did not hit an overflow or if it's
        // the root goal as that will now always hit the same overflow limit.
        //
        // NOTE: We cannot move any non-root goals to the global cache even if their final result
        // isn't impacted by the overflow as that goal still has unstable query dependencies
        // because it didn't go its full depth.
        //
        // FIXME(@lcnr): We could still cache subtrees which are not impacted by overflow though.
        // Tracking that info correctly isn't trivial, so I haven't implemented it for now.
        let should_cache_globally = !overflow_data.did_overflow() || stack.is_empty();
        if should_cache_globally {
            // FIXME: move the provisional entry to the global cache.
            let _ = (tcx, goal, response);
        }
    }
}

fn fixme_response_yes_no_constraints<'tcx>() -> QueryResult<'tcx> {
    unimplemented!()
}

fn fixme_response_maybe_no_constraints<'tcx>() -> QueryResult<'tcx> {
    unimplemented!()
}
