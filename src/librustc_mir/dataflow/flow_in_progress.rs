// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::{BasicBlock, Location};
use rustc_data_structures::indexed_set::{self, IdxSetBuf};
use rustc_data_structures::indexed_vec::Idx;

use dataflow::{BitDenotation, BlockSets, DataflowResults};
use dataflow::move_paths::{HasMoveData, MovePathIndex};

/// Data flow results are typically computed only as basic block
/// boundaries.  A `FlowInProgress` allows you to reconstruct the
/// effects at any point in the control-flow graph by starting with
/// the state at the start of the basic block (`reset_to_entry_of`)
/// and then replaying the effects of statements and terminators
/// (e.g. via `reconstruct_statement_effect` and
/// `reconstruct_terminator_effect`; don't forget to call `apply_local_effect`).
pub(crate) struct FlowInProgress<BD>
where
    BD: BitDenotation,
{
    base_results: DataflowResults<BD>,
    curr_state: IdxSetBuf<BD::Idx>,
    stmt_gen: IdxSetBuf<BD::Idx>,
    stmt_kill: IdxSetBuf<BD::Idx>,
}

impl<'tcx, T> FlowInProgress<T>
where
    T: HasMoveData<'tcx> + BitDenotation<Idx = MovePathIndex>,
{
    pub(crate) fn has_any_child_of(&self, mpi: T::Idx) -> Option<T::Idx> {
        let move_data = self.base_results.operator().move_data();

        let mut todo = vec![mpi];
        let mut push_siblings = false; // don't look at siblings of original `mpi`.
        while let Some(mpi) = todo.pop() {
            if self.curr_state.contains(&mpi) {
                return Some(mpi);
            }
            let move_path = &move_data.move_paths[mpi];
            if let Some(child) = move_path.first_child {
                todo.push(child);
            }
            if push_siblings {
                if let Some(sibling) = move_path.next_sibling {
                    todo.push(sibling);
                }
            } else {
                // after we've processed the original `mpi`, we should
                // always traverse the siblings of any of its
                // children.
                push_siblings = true;
            }
        }
        return None;
    }
}

impl<BD> FlowInProgress<BD>
where
    BD: BitDenotation,
{
    pub(crate) fn new(results: DataflowResults<BD>) -> Self {
        let bits_per_block = results.sets().bits_per_block();
        let curr_state = IdxSetBuf::new_empty(bits_per_block);
        let stmt_gen = IdxSetBuf::new_empty(bits_per_block);
        let stmt_kill = IdxSetBuf::new_empty(bits_per_block);
        FlowInProgress {
            base_results: results,
            curr_state: curr_state,
            stmt_gen: stmt_gen,
            stmt_kill: stmt_kill,
        }
    }


    /// Access the underlying operator.
    pub(crate) fn operator(&self) -> &BD {
        self.base_results.operator()
    }

    /// Access the current state.
    pub(crate) fn curr_state(&self) -> &IdxSetBuf<BD::Idx> {
        &self.curr_state
    }

    /// Iterate over each bit set in the current state.
    pub(crate) fn each_state_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.curr_state
            .each_bit(self.base_results.operator().bits_per_block(), f)
    }

    /// Iterate over each `gen` bit in the current effect (invoke
    /// `reconstruct_statement_effect` or
    /// `reconstruct_terminator_effect` first).
    pub(crate) fn each_gen_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.stmt_gen
            .each_bit(self.base_results.operator().bits_per_block(), f)
    }

    /// Resets the current state to the state at the entry to `bb`.
    pub(crate) fn reset_to_entry_of(&mut self, bb: BasicBlock) {
        (*self.curr_state).clone_from(self.base_results.sets().on_entry_set_for(bb.index()));
    }

    /// Accumulates the effects of the statement at `loc` into our local gen/kill sets.
    ///
    /// Note that invoking this method alone does not change the
    /// `curr_state` -- you must invoke `apply_local_effect`
    /// afterwards.
    pub(crate) fn reconstruct_statement_effect(&mut self, loc: Location) {
        self.stmt_gen.reset_to_empty();
        self.stmt_kill.reset_to_empty();
        let mut ignored = IdxSetBuf::new_empty(0);
        let mut sets = BlockSets {
            on_entry: &mut ignored,
            gen_set: &mut self.stmt_gen,
            kill_set: &mut self.stmt_kill,
        };
        self.base_results
            .operator()
            .statement_effect(&mut sets, loc);
    }

    /// Accumulates the effects of the terminator at `loc` into our local gen/kill sets.
    ///
    /// Note that invoking this method alone does not change the
    /// `curr_state` -- you must invoke `apply_local_effect`
    /// afterwards.
    pub(crate) fn reconstruct_terminator_effect(&mut self, loc: Location) {
        self.stmt_gen.reset_to_empty();
        self.stmt_kill.reset_to_empty();
        let mut ignored = IdxSetBuf::new_empty(0);
        let mut sets = BlockSets {
            on_entry: &mut ignored,
            gen_set: &mut self.stmt_gen,
            kill_set: &mut self.stmt_kill,
        };
        self.base_results
            .operator()
            .terminator_effect(&mut sets, loc);
    }

    /// Updates current stat with the statement/terminator effects
    /// that been reconstructed.
    pub(crate) fn apply_local_effect(&mut self) {
        self.curr_state.union(&self.stmt_gen);
        self.curr_state.subtract(&self.stmt_kill);
    }

    /// Finds the elements present in the current state.
    pub(crate) fn elems_incoming(&self) -> indexed_set::Elems<BD::Idx> {
        let univ = self.base_results.sets().bits_per_block();
        self.curr_state.elems(univ)
    }

    /// Creates a clone of the current state and applies the local
    /// effects to the clone (leaving the state of self intact).
    /// Walks through the resulting state.
    pub(crate) fn with_elems_outgoing<F>(&self, f: F)
    where
        F: FnOnce(indexed_set::Elems<BD::Idx>),
    {
        let mut curr_state = self.curr_state.clone();
        curr_state.union(&self.stmt_gen);
        curr_state.subtract(&self.stmt_kill);
        let univ = self.base_results.sets().bits_per_block();
        f(curr_state.elems(univ));
    }
}
