// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::indexed_vec::Idx;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::ty;
use syntax::ast;
use analysis::borrows::MaybeBorrowed;
use analysis::dataflow::{do_dataflow, BitDenotation, BlockSets, DebugFormatted};
use analysis::eventflow::{Backward, Events, EventFlowResults, Forward, PastAndFuture};
use analysis::locations::FlatLocations;

pub struct Accesses<'a> {
    pub results: PastAndFuture<EventFlowResults<'a, Forward, Local>,
                               EventFlowResults<'a, Backward, Local>>
}

impl<'a> Accesses<'a> {
    pub fn collect(mir: &Mir, flat_locations: &'a FlatLocations) -> Self {
        let borrows = ty::tls::with(|tcx| {
            do_dataflow(tcx, mir, ast::DUMMY_NODE_ID, &[],
                        &IdxSetBuf::new_empty(mir.basic_blocks().len()),
                        MaybeBorrowed::new(mir),
                        |_, path| DebugFormatted::new(&path))
        });

        let mut collector = AccessPathCollector {
            location: Location {
                block: START_BLOCK,
                statement_index: !0
            },
            accesses: Events::new(mir, flat_locations, mir.local_decls.len()),
            maybe_borrowed: IdxSetBuf::new_empty(0)
        };

        // FIXME(eddyb) introduce a seeker for this (like in eventflow),
        // maybe reusing `dataflow::at_location(::FlowAtLocation)`.
        // That would remove the need for asserting the location.

        for (block, data) in mir.basic_blocks().iter_enumerated() {
            collector.location.block = block;
            collector.maybe_borrowed = borrows.sets().on_entry_set_for(block.index()).to_owned();

            let on_entry = &mut collector.maybe_borrowed.clone();
            let kill_set = &mut collector.maybe_borrowed.clone();
            for (i, statement) in data.statements.iter().enumerate() {
                collector.location.statement_index = i;
                borrows.operator().before_statement_effect(&mut BlockSets {
                    on_entry,
                    kill_set,
                    gen_set: &mut collector.maybe_borrowed,
                }, collector.location);
                // FIXME(eddyb) get rid of temporary with NLL/2phi.
                let location = collector.location;
                collector.visit_statement(block, statement, location);
                borrows.operator().statement_effect(&mut BlockSets {
                    on_entry,
                    kill_set,
                    gen_set: &mut collector.maybe_borrowed,
                }, collector.location);
            }

            if let Some(ref terminator) = data.terminator {
                collector.location.statement_index = data.statements.len();
                borrows.operator().before_terminator_effect(&mut BlockSets {
                    on_entry,
                    kill_set,
                    gen_set: &mut collector.maybe_borrowed,
                }, collector.location);
                // FIXME(eddyb) get rid of temporary with NLL/2phi.
                let location = collector.location;
                collector.visit_terminator(block, terminator, location);
            }
        }
        // All arguments have been accessed prior to the call to this function.
        let results = collector.accesses.flow(mir.args_iter());
        Accesses { results }
    }
}

struct AccessPathCollector<'a, 'b, 'tcx: 'a> {
    accesses: Events<'a, 'b, 'tcx, Local>,
    location: Location,
    maybe_borrowed: IdxSetBuf<Local>
}

impl<'a, 'b, 'tcx> AccessPathCollector<'a, 'b, 'tcx> {
    fn access_anything_borrowed(&mut self, location: Location) {
        assert_eq!(self.location, location);

        // FIXME(eddyb) OR `maybe_borrowed` into the accesses for performance.
        for path in self.maybe_borrowed.iter() {
            self.accesses.insert_at(path, location);
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for AccessPathCollector<'a, 'b, 'tcx> {
    fn visit_local(&mut self,
                   &local: &Local,
                   context: PlaceContext,
                   location: Location) {
        if context.is_use() {
            self.accesses.insert_at(local, location);
        }
    }

    fn visit_projection_elem(&mut self,
                             elem: &PlaceElem<'tcx>,
                             context: PlaceContext<'tcx>,
                             location: Location) {
        if let ProjectionElem::Deref = *elem {
            self.access_anything_borrowed(location);
        }
        self.super_projection_elem(elem, context, location);
    }

    fn visit_terminator_kind(&mut self,
                             block: BasicBlock,
                             kind: &TerminatorKind<'tcx>,
                             location: Location) {
        match *kind {
            TerminatorKind::Call { .. } => {
                self.access_anything_borrowed(location);
            }
            TerminatorKind::Return => {
                self.visit_local(&RETURN_PLACE, PlaceContext::Move, location);
            }
            _ => {}
        }
        self.super_terminator_kind(block, kind, location);
    }
}
