// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Manages the dataflow bits required for borrowck.
//!
//! FIXME: this might be better as a "generic" fixed-point combinator,
//! but is not as ugly as it is right now.

use rustc::mir::{BasicBlock, Location};
use rustc_data_structures::indexed_set::{self, IdxSetBuf};
use rustc_data_structures::indexed_vec::Idx;

use dataflow::{BitDenotation, BlockSets, DataflowResults};
use dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use dataflow::{EverInitializedLvals, MovingOutStatements};
use dataflow::Borrows;
use dataflow::move_paths::{HasMoveData, MovePathIndex};
use std::fmt;

// (forced to be `pub` due to its use as an associated type below.)
pub struct InProgress<'b, 'gcx: 'tcx, 'tcx: 'b> {
    pub borrows: FlowInProgress<Borrows<'b, 'gcx, 'tcx>>,
    pub inits: FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>,
    pub uninits: FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>,
    pub move_outs: FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>,
    pub ever_inits: FlowInProgress<EverInitializedLvals<'b, 'gcx, 'tcx>>,
}

pub struct FlowInProgress<BD>
where
    BD: BitDenotation,
{
    base_results: DataflowResults<BD>,
    curr_state: IdxSetBuf<BD::Idx>,
    stmt_gen: IdxSetBuf<BD::Idx>,
    stmt_kill: IdxSetBuf<BD::Idx>,
}

impl<'b, 'gcx, 'tcx> InProgress<'b, 'gcx, 'tcx> {
    pub fn new(
        borrows: FlowInProgress<Borrows<'b, 'gcx, 'tcx>>,
        inits: FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>,
        uninits: FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>,
        move_outs: FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>,
        ever_inits: FlowInProgress<EverInitializedLvals<'b, 'gcx, 'tcx>>,
    ) -> Self {
        InProgress {
            borrows,
            inits,
            uninits,
            move_outs,
            ever_inits,
        }
    }

    fn each_flow<XB, XI, XU, XM, XE>(
        &mut self,
        mut xform_borrows: XB,
        mut xform_inits: XI,
        mut xform_uninits: XU,
        mut xform_move_outs: XM,
        mut xform_ever_inits: XE,
    ) where
        XB: FnMut(&mut FlowInProgress<Borrows<'b, 'gcx, 'tcx>>),
        XI: FnMut(&mut FlowInProgress<MaybeInitializedLvals<'b, 'gcx, 'tcx>>),
        XU: FnMut(&mut FlowInProgress<MaybeUninitializedLvals<'b, 'gcx, 'tcx>>),
        XM: FnMut(&mut FlowInProgress<MovingOutStatements<'b, 'gcx, 'tcx>>),
        XE: FnMut(&mut FlowInProgress<EverInitializedLvals<'b, 'gcx, 'tcx>>),
    {
        xform_borrows(&mut self.borrows);
        xform_inits(&mut self.inits);
        xform_uninits(&mut self.uninits);
        xform_move_outs(&mut self.move_outs);
        xform_ever_inits(&mut self.ever_inits);
    }

    pub fn reset_to_entry_of(&mut self, bb: BasicBlock) {
        self.each_flow(
            |b| b.reset_to_entry_of(bb),
            |i| i.reset_to_entry_of(bb),
            |u| u.reset_to_entry_of(bb),
            |m| m.reset_to_entry_of(bb),
            |e| e.reset_to_entry_of(bb),
        );
    }

    pub fn reconstruct_statement_effect(
        &mut self,
        location: Location,
    ) {
        self.each_flow(
            |b| b.reconstruct_statement_effect(location),
            |i| i.reconstruct_statement_effect(location),
            |u| u.reconstruct_statement_effect(location),
            |m| m.reconstruct_statement_effect(location),
            |e| e.reconstruct_statement_effect(location),
        );
    }

    pub fn apply_local_effect(&mut self, _location: Location) {
        self.each_flow(
            |b| b.apply_local_effect(),
            |i| i.apply_local_effect(),
            |u| u.apply_local_effect(),
            |m| m.apply_local_effect(),
            |e| e.apply_local_effect(),
        );
    }

    pub fn reconstruct_terminator_effect(&mut self, location: Location) {
        self.each_flow(
            |b| b.reconstruct_terminator_effect(location),
            |i| i.reconstruct_terminator_effect(location),
            |u| u.reconstruct_terminator_effect(location),
            |m| m.reconstruct_terminator_effect(location),
            |e| e.reconstruct_terminator_effect(location),
        );
    }
}

impl<'b, 'gcx, 'tcx> fmt::Display for InProgress<'b, 'gcx, 'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        s.push_str("borrows in effect: [");
        let mut saw_one = false;
        self.borrows.each_state_bit(|borrow| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let borrow_data = &self.borrows.base_results.operator().borrows()[borrow];
            s.push_str(&format!("{}", borrow_data));
        });
        s.push_str("] ");

        s.push_str("borrows generated: [");
        let mut saw_one = false;
        self.borrows.each_gen_bit(|borrow| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let borrow_data = &self.borrows.base_results.operator().borrows()[borrow];
            s.push_str(&format!("{}", borrow_data));
        });
        s.push_str("] ");

        s.push_str("inits: [");
        let mut saw_one = false;
        self.inits.each_state_bit(|mpi_init| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_path = &self.inits.base_results.operator().move_data().move_paths[mpi_init];
            s.push_str(&format!("{}", move_path));
        });
        s.push_str("] ");

        s.push_str("uninits: [");
        let mut saw_one = false;
        self.uninits.each_state_bit(|mpi_uninit| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_path =
                &self.uninits.base_results.operator().move_data().move_paths[mpi_uninit];
            s.push_str(&format!("{}", move_path));
        });
        s.push_str("] ");

        s.push_str("move_out: [");
        let mut saw_one = false;
        self.move_outs.each_state_bit(|mpi_move_out| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_out = &self.move_outs.base_results.operator().move_data().moves[mpi_move_out];
            s.push_str(&format!("{:?}", move_out));
        });
        s.push_str("] ");

        s.push_str("ever_init: [");
        let mut saw_one = false;
        self.ever_inits.each_state_bit(|mpi_ever_init| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let ever_init =
                &self.ever_inits.base_results.operator().move_data().inits[mpi_ever_init];
            s.push_str(&format!("{:?}", ever_init));
        });
        s.push_str("]");

        fmt::Display::fmt(&s, fmt)
    }
}

impl<'tcx, T> FlowInProgress<T>
where
    T: HasMoveData<'tcx> + BitDenotation<Idx = MovePathIndex>,
{
    pub fn has_any_child_of(&self, mpi: T::Idx) -> Option<T::Idx> {
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
    pub fn each_state_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.curr_state
            .each_bit(self.base_results.operator().bits_per_block(), f)
    }

    fn each_gen_bit<F>(&self, f: F)
    where
        F: FnMut(BD::Idx),
    {
        self.stmt_gen
            .each_bit(self.base_results.operator().bits_per_block(), f)
    }

    pub fn new(results: DataflowResults<BD>) -> Self {
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

    pub fn operator(&self) -> &BD {
        self.base_results.operator()
    }

    pub fn contains(&self, x: &BD::Idx) -> bool {
        self.curr_state.contains(x)
    }

    pub fn reset_to_entry_of(&mut self, bb: BasicBlock) {
        (*self.curr_state).clone_from(self.base_results.sets().on_entry_set_for(bb.index()));
    }

    pub fn reconstruct_statement_effect(&mut self, loc: Location) {
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

    pub fn reconstruct_terminator_effect(&mut self, loc: Location) {
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

    pub fn apply_local_effect(&mut self) {
        self.curr_state.union(&self.stmt_gen);
        self.curr_state.subtract(&self.stmt_kill);
    }

    pub fn elems_incoming(&self) -> indexed_set::Elems<BD::Idx> {
        let univ = self.base_results.sets().bits_per_block();
        self.curr_state.elems(univ)
    }

    pub fn with_elems_outgoing<F>(&self, f: F)
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
