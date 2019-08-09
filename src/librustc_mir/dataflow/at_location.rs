//! A nice wrapper to consume dataflow results at several CFG
//! locations.

use rustc::mir::{BasicBlock, Location};
use rustc_data_structures::bit_set::{BitIter, BitSet, HybridBitSet};

use crate::dataflow::{BitDenotation, DataflowResults, GenKillSet};
use crate::dataflow::move_paths::{HasMoveData, MovePathIndex};

use std::iter;
use std::borrow::Borrow;

/// A trait for "cartesian products" of multiple FlowAtLocation.
///
/// There's probably a way to auto-impl this, but I think
/// it is cleaner to have manual visitor impls.
pub trait FlowsAtLocation {
    /// Reset the state bitvector to represent the entry to block `bb`.
    fn reset_to_entry_of(&mut self, bb: BasicBlock);

    /// Reset the state bitvector to represent the exit of the
    /// terminator of block `bb`.
    ///
    /// **Important:** In the case of a `Call` terminator, these
    /// effects do *not* include the result of storing the destination
    /// of the call, since that is edge-dependent (in other words, the
    /// effects don't apply to the unwind edge).
    fn reset_to_exit_of(&mut self, bb: BasicBlock);

    /// Builds gen and kill sets for statement at `loc`.
    ///
    /// Note that invoking this method alone does not change the
    /// `curr_state` -- you must invoke `apply_local_effect`
    /// afterwards.
    fn reconstruct_statement_effect(&mut self, loc: Location);

    /// Builds gen and kill sets for terminator for `loc`.
    ///
    /// Note that invoking this method alone does not change the
    /// `curr_state` -- you must invoke `apply_local_effect`
    /// afterwards.
    fn reconstruct_terminator_effect(&mut self, loc: Location);

    /// Apply current gen + kill sets to `flow_state`.
    ///
    /// (`loc` parameters can be ignored if desired by
    /// client. For the terminator, the `stmt_idx` will be the number
    /// of statements in the block.)
    fn apply_local_effect(&mut self, loc: Location);
}

/// Represents the state of dataflow at a particular
/// CFG location, both before and after it is
/// executed.
///
/// Data flow results are typically computed only as basic block
/// boundaries. A `FlowInProgress` allows you to reconstruct the
/// effects at any point in the control-flow graph by starting with
/// the state at the start of the basic block (`reset_to_entry_of`)
/// and then replaying the effects of statements and terminators
/// (e.g., via `reconstruct_statement_effect` and
/// `reconstruct_terminator_effect`; don't forget to call
/// `apply_local_effect`).
pub struct FlowAtLocation<'tcx, BD, DR = DataflowResults<'tcx, BD>>
where
    BD: BitDenotation<'tcx>,
    DR: Borrow<DataflowResults<'tcx, BD>>,
{
    base_results: DR,
    curr_state: BitSet<BD::Idx>,
    stmt_trans: GenKillSet<BD::Idx>,
}

impl<'tcx, BD, DR> FlowAtLocation<'tcx, BD, DR>
where
    BD: BitDenotation<'tcx>,
    DR: Borrow<DataflowResults<'tcx, BD>>,
{
    /// Iterate over each bit set in the current state.
    pub fn each_state_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.curr_state.iter().for_each(f)
    }

    /// Iterate over each `gen` bit in the current effect (invoke
    /// `reconstruct_statement_effect` or
    /// `reconstruct_terminator_effect` first).
    pub fn each_gen_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.stmt_trans.gen_set.iter().for_each(f)
    }

    pub fn new(results: DR) -> Self {
        let bits_per_block = results.borrow().sets().bits_per_block();
        let curr_state = BitSet::new_empty(bits_per_block);
        let stmt_trans = GenKillSet::from_elem(HybridBitSet::new_empty(bits_per_block));
        FlowAtLocation {
            base_results: results,
            curr_state,
            stmt_trans,
        }
    }

    /// Access the underlying operator.
    pub fn operator(&self) -> &BD {
        self.base_results.borrow().operator()
    }

    pub fn contains(&self, x: BD::Idx) -> bool {
        self.curr_state.contains(x)
    }

    /// Returns an iterator over the elements present in the current state.
    pub fn iter_incoming(&self) -> iter::Peekable<BitIter<'_, BD::Idx>> {
        self.curr_state.iter().peekable()
    }

    /// Creates a clone of the current state and applies the local
    /// effects to the clone (leaving the state of self intact).
    /// Invokes `f` with an iterator over the resulting state.
    pub fn with_iter_outgoing<F>(&self, f: F)
    where
        F: FnOnce(BitIter<'_, BD::Idx>),
    {
        let mut curr_state = self.curr_state.clone();
        self.stmt_trans.apply(&mut curr_state);
        f(curr_state.iter());
    }

    /// Returns a bitset of the elements present in the current state.
    pub fn as_dense(&self) -> &BitSet<BD::Idx> {
        &self.curr_state
    }
}

impl<'tcx, BD, DR> FlowsAtLocation for FlowAtLocation<'tcx, BD, DR>
where
    BD: BitDenotation<'tcx>,
    DR: Borrow<DataflowResults<'tcx, BD>>,
{
    fn reset_to_entry_of(&mut self, bb: BasicBlock) {
        self.curr_state.overwrite(self.base_results.borrow().sets().entry_set_for(bb.index()));
    }

    fn reset_to_exit_of(&mut self, bb: BasicBlock) {
        self.reset_to_entry_of(bb);
        let trans = self.base_results.borrow().sets().trans_for(bb.index());
        trans.apply(&mut self.curr_state)
    }

    fn reconstruct_statement_effect(&mut self, loc: Location) {
        self.stmt_trans.clear();
        self.base_results
            .borrow()
            .operator()
            .before_statement_effect(&mut self.stmt_trans, loc);
        self.stmt_trans.apply(&mut self.curr_state);

        self.base_results
            .borrow()
            .operator()
            .statement_effect(&mut self.stmt_trans, loc);
    }

    fn reconstruct_terminator_effect(&mut self, loc: Location) {
        self.stmt_trans.clear();
        self.base_results
            .borrow()
            .operator()
            .before_terminator_effect(&mut self.stmt_trans, loc);
        self.stmt_trans.apply(&mut self.curr_state);

        self.base_results
            .borrow()
            .operator()
            .terminator_effect(&mut self.stmt_trans, loc);
    }

    fn apply_local_effect(&mut self, _loc: Location) {
        self.stmt_trans.apply(&mut self.curr_state)
    }
}


impl<'tcx, T, DR> FlowAtLocation<'tcx, T, DR>
where
    T: HasMoveData<'tcx> + BitDenotation<'tcx, Idx = MovePathIndex>,
    DR: Borrow<DataflowResults<'tcx, T>>,
{
    pub fn has_any_child_of(&self, mpi: T::Idx) -> Option<T::Idx> {
        // We process `mpi` before the loop below, for two reasons:
        // - it's a little different from the loop case (we don't traverse its
        //   siblings);
        // - ~99% of the time the loop isn't reached, and this code is hot, so
        //   we don't want to allocate `todo` unnecessarily.
        if self.contains(mpi) {
            return Some(mpi);
        }
        let move_data = self.operator().move_data();
        let move_path = &move_data.move_paths[mpi];
        let mut todo = if let Some(child) = move_path.first_child {
            vec![child]
        } else {
            return None;
        };

        while let Some(mpi) = todo.pop() {
            if self.contains(mpi) {
                return Some(mpi);
            }
            let move_path = &move_data.move_paths[mpi];
            if let Some(child) = move_path.first_child {
                todo.push(child);
            }
            // After we've processed the original `mpi`, we should always
            // traverse the siblings of any of its children.
            if let Some(sibling) = move_path.next_sibling {
                todo.push(sibling);
            }
        }
        return None;
    }
}
