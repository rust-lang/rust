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
use rustc::ty::RegionVid;
use rustc_data_structures::bit_set::BitIter;

use borrow_check::location::LocationIndex;

use polonius_engine::Output;

use dataflow::move_paths::indexes::BorrowIndex;
use dataflow::move_paths::HasMoveData;
use dataflow::Borrows;
use dataflow::EverInitializedPlaces;
use dataflow::{FlowAtLocation, FlowsAtLocation};
use dataflow::MaybeUninitializedPlaces;
use either::Either;
use std::fmt;
use std::rc::Rc;

// (forced to be `pub` due to its use as an associated type below.)
crate struct Flows<'b, 'gcx: 'tcx, 'tcx: 'b> {
    borrows: FlowAtLocation<'tcx, Borrows<'b, 'gcx, 'tcx>>,
    pub uninits: FlowAtLocation<'tcx, MaybeUninitializedPlaces<'b, 'gcx, 'tcx>>,
    pub ever_inits: FlowAtLocation<'tcx, EverInitializedPlaces<'b, 'gcx, 'tcx>>,

    /// Polonius Output
    pub polonius_output: Option<Rc<Output<RegionVid, BorrowIndex, LocationIndex>>>,
}

impl<'b, 'gcx, 'tcx> Flows<'b, 'gcx, 'tcx> {
    crate fn new(
        borrows: FlowAtLocation<'tcx, Borrows<'b, 'gcx, 'tcx>>,
        uninits: FlowAtLocation<'tcx, MaybeUninitializedPlaces<'b, 'gcx, 'tcx>>,
        ever_inits: FlowAtLocation<'tcx, EverInitializedPlaces<'b, 'gcx, 'tcx>>,
        polonius_output: Option<Rc<Output<RegionVid, BorrowIndex, LocationIndex>>>,
    ) -> Self {
        Flows {
            borrows,
            uninits,
            ever_inits,
            polonius_output,
        }
    }

    crate fn borrows_in_scope(
        &self,
        location: LocationIndex,
    ) -> impl Iterator<Item = BorrowIndex> + '_ {
        if let Some(ref polonius) = self.polonius_output {
            Either::Left(polonius.errors_at(location).iter().cloned())
        } else {
            Either::Right(self.borrows.iter_incoming())
        }
    }

    crate fn with_outgoing_borrows(&self, op: impl FnOnce(BitIter<BorrowIndex>)) {
        self.borrows.with_iter_outgoing(op)
    }
}

macro_rules! each_flow {
    ($this:ident, $meth:ident($arg:ident)) => {
        FlowAtLocation::$meth(&mut $this.borrows, $arg);
        FlowAtLocation::$meth(&mut $this.uninits, $arg);
        FlowAtLocation::$meth(&mut $this.ever_inits, $arg);
    };
}

impl<'b, 'gcx, 'tcx> FlowsAtLocation for Flows<'b, 'gcx, 'tcx> {
    fn reset_to_entry_of(&mut self, bb: BasicBlock) {
        each_flow!(self, reset_to_entry_of(bb));
    }

    fn reset_to_exit_of(&mut self, bb: BasicBlock) {
        each_flow!(self, reset_to_exit_of(bb));
    }

    fn reconstruct_statement_effect(&mut self, location: Location) {
        each_flow!(self, reconstruct_statement_effect(location));
    }

    fn reconstruct_terminator_effect(&mut self, location: Location) {
        each_flow!(self, reconstruct_terminator_effect(location));
    }

    fn apply_local_effect(&mut self, location: Location) {
        each_flow!(self, apply_local_effect(location));
    }
}

impl<'b, 'gcx, 'tcx> fmt::Display for Flows<'b, 'gcx, 'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        s.push_str("borrows in effect: [");
        let mut saw_one = false;
        self.borrows.each_state_bit(|borrow| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let borrow_data = &self.borrows.operator().borrows()[borrow];
            s.push_str(&borrow_data.to_string());
        });
        s.push_str("] ");

        s.push_str("borrows generated: [");
        let mut saw_one = false;
        self.borrows.each_gen_bit(|borrow| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let borrow_data = &self.borrows.operator().borrows()[borrow];
            s.push_str(&borrow_data.to_string());
        });
        s.push_str("] ");

        s.push_str("uninits: [");
        let mut saw_one = false;
        self.uninits.each_state_bit(|mpi_uninit| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let move_path = &self.uninits.operator().move_data().move_paths[mpi_uninit];
            s.push_str(&move_path.to_string());
        });
        s.push_str("] ");

        s.push_str("ever_init: [");
        let mut saw_one = false;
        self.ever_inits.each_state_bit(|mpi_ever_init| {
            if saw_one {
                s.push_str(", ");
            };
            saw_one = true;
            let ever_init = &self.ever_inits.operator().move_data().inits[mpi_ever_init];
            s.push_str(&format!("{:?}", ever_init));
        });
        s.push_str("]");

        fmt::Display::fmt(&s, fmt)
    }
}
