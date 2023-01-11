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
use super::{CanonicalGoal, Certainty, MaybeCause, Response};
use super::{EvalCtxt, QueryResult};
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::canonical::{Canonical, CanonicalVarKind, CanonicalVarValues};
use rustc_middle::ty::{self, TyCtxt};
use std::collections::hash_map::Entry;

rustc_index::newtype_index! {
    pub struct StackDepth {}
}
rustc_index::newtype_index! {
    pub struct EntryIndex {}
}

#[derive(Debug, Clone)]
struct ProvisionalEntry<'tcx> {
    // In case we have a coinductive cycle, this is the
    // the currently least restrictive result of this goal.
    response: QueryResult<'tcx>,
    // In case of a cycle, the depth of lowest stack entry involved
    // in that cycle. This is monotonically decreasing in the stack as all
    // elements between the current stack element in the lowest stack entry
    // involved have to also be involved in that cycle.
    //
    // We can only move entries to the global cache once we're complete done
    // with the cycle. If this entry has not been involved in a cycle,
    // this is just its own depth.
    depth: StackDepth,

    // The goal for this entry. Should always be equal to the corresponding goal
    // in the lookup table.
    goal: CanonicalGoal<'tcx>,
}

struct StackElem<'tcx> {
    goal: CanonicalGoal<'tcx>,
    has_been_used: bool,
}

pub(super) struct ProvisionalCache<'tcx> {
    stack: IndexVec<StackDepth, StackElem<'tcx>>,
    entries: IndexVec<EntryIndex, ProvisionalEntry<'tcx>>,
    // FIXME: This is only used to quickly check whether a given goal
    // is in the cache. We should experiment with using something like
    // `SsoHashSet` here because in most cases there are only a few entries.
    lookup_table: FxHashMap<CanonicalGoal<'tcx>, EntryIndex>,
}

impl<'tcx> ProvisionalCache<'tcx> {
    pub(super) fn empty() -> ProvisionalCache<'tcx> {
        ProvisionalCache {
            stack: Default::default(),
            entries: Default::default(),
            lookup_table: Default::default(),
        }
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
        match cache.lookup_table.entry(goal) {
            // No entry, simply push this goal on the stack after dealing with overflow.
            Entry::Vacant(v) => {
                if self.overflow_data.has_overflow(cache.stack.len()) {
                    return Err(self.deal_with_overflow(goal));
                }

                let depth = cache.stack.push(StackElem { goal, has_been_used: false });
                let response = response_no_constraints(self.tcx, goal, Certainty::Yes);
                let entry_index = cache.entries.push(ProvisionalEntry { response, depth, goal });
                v.insert(entry_index);
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
            Entry::Occupied(entry_index) => {
                let entry_index = *entry_index.get();
                // FIXME `ProvisionalEntry` should be `Copy`.
                let entry = cache.entries.get(entry_index).unwrap().clone();
                cache.stack[entry.depth].has_been_used = true;
                for provisional_entry in cache.entries.iter_mut().skip(entry_index.index()) {
                    provisional_entry.depth = provisional_entry.depth.min(entry.depth);
                }

                // NOTE: The goals on the stack aren't the only goals involved in this cycle.
                // We can also depend on goals which aren't part of the stack but coinductively
                // depend on the stack themselves. We already checked whether all the goals
                // between these goals and their root on the stack. This means that as long as
                // each goal in a cycle is checked for coinductivity by itself, simply checking
                // the stack is enough.
                if cache.stack.raw[entry.depth.index()..]
                    .iter()
                    .all(|g| g.goal.value.predicate.is_coinductive(self.tcx))
                {
                    Err(entry.response)
                } else {
                    Err(response_no_constraints(
                        self.tcx,
                        goal,
                        Certainty::Maybe(MaybeCause::Overflow),
                    ))
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

        let provisional_entry_index = *cache.lookup_table.get(&goal).unwrap();
        let provisional_entry = &mut cache.entries[provisional_entry_index];
        // Was the current goal the root of a cycle and was the provisional response
        // different from the final one.
        if has_been_used && provisional_entry.response != response {
            // If so, update the provisional reponse for this goal...
            provisional_entry.response = response;
            // ...remove all entries whose result depends on this goal
            // from the provisional cache...
            //
            // That's not completely correct, as a nested goal can also
            // depend on a goal which is lower in the stack so it doesn't
            // actually depend on the current goal. This should be fairly
            // rare and is hopefully not relevant for performance.
            #[allow(rustc::potential_query_instability)]
            cache.lookup_table.retain(|_key, index| *index <= provisional_entry_index);
            cache.entries.truncate(provisional_entry_index.index() + 1);

            // ...and finally push our goal back on the stack and reevaluate it.
            cache.stack.push(StackElem { goal, has_been_used: false });
            false
        } else {
            // If not, we're done with this goal.
            //
            // Check whether that this goal doesn't depend on a goal deeper on the stack
            // and if so, move it and all nested goals to the global cache.
            //
            // Note that if any nested goal were to depend on something deeper on the stack,
            // this would have also updated the depth of this goal.
            if provisional_entry.depth == cache.stack.next_index() {
                for (i, entry) in cache.entries.drain_enumerated(provisional_entry_index.index()..)
                {
                    let actual_index = cache.lookup_table.remove(&entry.goal);
                    debug_assert_eq!(Some(i), actual_index);
                    Self::try_move_finished_goal_to_global_cache(
                        self.tcx,
                        &mut self.overflow_data,
                        &cache.stack,
                        entry.goal,
                        entry.response,
                    );
                }
            }
            true
        }
    }

    fn try_move_finished_goal_to_global_cache(
        tcx: TyCtxt<'tcx>,
        overflow_data: &mut OverflowData,
        stack: &IndexVec<StackDepth, StackElem<'tcx>>,
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

pub(super) fn response_no_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal: Canonical<'tcx, impl Sized>,
    certainty: Certainty,
) -> QueryResult<'tcx> {
    let var_values = goal
        .variables
        .iter()
        .enumerate()
        .map(|(i, info)| match info.kind {
            CanonicalVarKind::Ty(_) | CanonicalVarKind::PlaceholderTy(_) => {
                tcx.mk_ty(ty::Bound(ty::INNERMOST, ty::BoundVar::from_usize(i).into())).into()
            }
            CanonicalVarKind::Region(_) | CanonicalVarKind::PlaceholderRegion(_) => {
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_usize(i),
                    kind: ty::BrAnon(i as u32, None),
                };
                tcx.mk_region(ty::ReLateBound(ty::INNERMOST, br)).into()
            }
            CanonicalVarKind::Const(_, ty) | CanonicalVarKind::PlaceholderConst(_, ty) => tcx
                .mk_const(ty::ConstKind::Bound(ty::INNERMOST, ty::BoundVar::from_usize(i)), ty)
                .into(),
        })
        .collect();

    Ok(Canonical {
        max_universe: goal.max_universe,
        variables: goal.variables,
        value: Response {
            var_values: CanonicalVarValues { var_values },
            external_constraints: Default::default(),
            certainty,
        },
    })
}
