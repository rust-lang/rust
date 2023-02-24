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
use super::StackDepth;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_middle::traits::solve::{CanonicalGoal, QueryResult};

rustc_index::newtype_index! {
    pub struct EntryIndex {}
}

#[derive(Debug, Clone)]
pub(super) struct ProvisionalEntry<'tcx> {
    // In case we have a coinductive cycle, this is the
    // the currently least restrictive result of this goal.
    pub(super) response: QueryResult<'tcx>,
    // In case of a cycle, the position of deepest stack entry involved
    // in that cycle. This is monotonically decreasing in the stack as all
    // elements between the current stack element in the deepest stack entry
    // involved have to also be involved in that cycle.
    //
    // We can only move entries to the global cache once we're complete done
    // with the cycle. If this entry has not been involved in a cycle,
    // this is just its own depth.
    pub(super) depth: StackDepth,

    // The goal for this entry. Should always be equal to the corresponding goal
    // in the lookup table.
    pub(super) goal: CanonicalGoal<'tcx>,
}

pub(super) struct ProvisionalCache<'tcx> {
    pub(super) entries: IndexVec<EntryIndex, ProvisionalEntry<'tcx>>,
    // FIXME: This is only used to quickly check whether a given goal
    // is in the cache. We should experiment with using something like
    // `SsoHashSet` here because in most cases there are only a few entries.
    pub(super) lookup_table: FxHashMap<CanonicalGoal<'tcx>, EntryIndex>,
}

impl<'tcx> ProvisionalCache<'tcx> {
    pub(super) fn empty() -> ProvisionalCache<'tcx> {
        ProvisionalCache { entries: Default::default(), lookup_table: Default::default() }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.lookup_table.is_empty()
    }

    /// Adds a dependency from the current leaf to `target` in the cache
    /// to prevent us from moving any goals which depend on the current leaf
    /// to the global cache while we're still computing `target`.
    ///
    /// Its important to note that `target` may already be part of a different cycle.
    /// In this case we have to ensure that we also depend on all other goals
    /// in the existing cycle in addition to the potentially direct cycle with `target`.
    pub(super) fn add_dependency_of_leaf_on(&mut self, target: EntryIndex) {
        let depth = self.entries[target].depth;
        for provisional_entry in &mut self.entries.raw[target.index()..] {
            // The depth of `target` is the position of the deepest goal in the stack
            // on which `target` depends. That goal is the `root` of this cycle.
            //
            // Any entry which was added after `target` is either on the stack itself
            // at which point its depth is definitely at least as high as the depth of
            // `root`. If it's not on the stack itself it has to depend on a goal
            // between `root` and `leaf`. If it were to depend on a goal deeper in the
            // stack than `root`, then `root` would also depend on that goal, at which
            // point `root` wouldn't be the root anymore.
            debug_assert!(provisional_entry.depth >= depth);
            provisional_entry.depth = depth;
        }

        // We only update entries which were added after `target` as no other
        // entry should have a higher depth.
        //
        // Any entry which previously had a higher depth than target has to
        // be between `target` and `root`. Because of this we would have updated
        // its depth when calling `add_dependency_of_leaf_on(root)` for `target`.
        if cfg!(debug_assertions) {
            self.entries.iter().all(|e| e.depth <= depth);
        }
    }

    pub(super) fn depth(&self, entry_index: EntryIndex) -> StackDepth {
        self.entries[entry_index].depth
    }

    pub(super) fn provisional_result(&self, entry_index: EntryIndex) -> QueryResult<'tcx> {
        self.entries[entry_index].response
    }
}
