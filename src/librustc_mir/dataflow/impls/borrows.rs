// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::{self, Location, Mir};
use rustc::mir::visit::Visitor;
use rustc::ty::{Region, TyCtxt};
use rustc::ty::RegionKind::ReScope;
use rustc::util::nodemap::{FxHashMap, FxHashSet};

use rustc_data_structures::bitslice::{BitwiseOperator};
use rustc_data_structures::indexed_set::{IdxSet};
use rustc_data_structures::indexed_vec::{IndexVec};

use dataflow::{BitDenotation, BlockSets, DataflowOperator};
pub use dataflow::indexes::BorrowIndex;

use std::fmt;

// `Borrows` maps each dataflow bit to an `Rvalue::Ref`, which can be
// uniquely identified in the MIR by the `Location` of the assigment
// statement in which it appears on the right hand side.
pub struct Borrows<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    borrows: IndexVec<BorrowIndex, BorrowData<'tcx>>,
    location_map: FxHashMap<Location, BorrowIndex>,
    region_map: FxHashMap<Region<'tcx>, FxHashSet<BorrowIndex>>,
}

// temporarily allow some dead fields: `kind` and `region` will be
// needed by borrowck; `lvalue` will probably be a MovePathIndex when
// that is extended to include borrowed data paths.
#[allow(dead_code)]
#[derive(Debug)]
pub struct BorrowData<'tcx> {
    pub(crate) location: Location,
    pub(crate) kind: mir::BorrowKind,
    pub(crate) region: Region<'tcx>,
    pub(crate) lvalue: mir::Lvalue<'tcx>,
}

impl<'tcx> fmt::Display for BorrowData<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        let kind = match self.kind {
            mir::BorrowKind::Shared => "",
            mir::BorrowKind::Unique => "uniq ",
            mir::BorrowKind::Mut => "mut ",
        };
        let region = format!("{}", self.region);
        let region = if region.len() > 0 { format!("{} ", region) } else { region };
        write!(w, "&{}{}{:?}", region, kind, self.lvalue)
    }
}

impl<'a, 'tcx> Borrows<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, mir: &'a Mir<'tcx>) -> Self {
        let mut visitor = GatherBorrows { idx_vec: IndexVec::new(),
                                          location_map: FxHashMap(),
                                          region_map: FxHashMap(), };
        visitor.visit_mir(mir);
        return Borrows { tcx: tcx,
                         mir: mir,
                         borrows: visitor.idx_vec,
                         location_map: visitor.location_map,
                         region_map: visitor.region_map, };

        struct GatherBorrows<'tcx> {
            idx_vec: IndexVec<BorrowIndex, BorrowData<'tcx>>,
            location_map: FxHashMap<Location, BorrowIndex>,
            region_map: FxHashMap<Region<'tcx>, FxHashSet<BorrowIndex>>,
        }
        impl<'tcx> Visitor<'tcx> for GatherBorrows<'tcx> {
            fn visit_rvalue(&mut self,
                            rvalue: &mir::Rvalue<'tcx>,
                            location: mir::Location) {
                if let mir::Rvalue::Ref(region, kind, ref lvalue) = *rvalue {
                    let borrow = BorrowData {
                        location: location, kind: kind, region: region, lvalue: lvalue.clone(),
                    };
                    let idx = self.idx_vec.push(borrow);
                    self.location_map.insert(location, idx);
                    let borrows = self.region_map.entry(region).or_insert(FxHashSet());
                    borrows.insert(idx);
                }
            }
        }
    }

    pub fn borrows(&self) -> &IndexVec<BorrowIndex, BorrowData<'tcx>> { &self.borrows }

    pub fn location(&self, idx: BorrowIndex) -> &Location {
        &self.borrows[idx].location
    }
}

impl<'a, 'tcx> BitDenotation for Borrows<'a, 'tcx> {
    type Idx = BorrowIndex;
    fn name() -> &'static str { "borrows" }
    fn bits_per_block(&self) -> usize {
        self.borrows.len()
    }
    fn start_block_effect(&self, _sets: &mut BlockSets<BorrowIndex>)  {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect on
        // `_sets`.
    }
    fn statement_effect(&self,
                        sets: &mut BlockSets<BorrowIndex>,
                        location: Location) {
        let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });
        let stmt = block.statements.get(location.statement_index).unwrap_or_else(|| {
            panic!("could not find statement at location {:?}");
        });
        match stmt.kind {
            mir::StatementKind::EndRegion(region_scope) => {
                let borrow_indexes = self.region_map.get(&ReScope(region_scope)).unwrap_or_else(|| {
                    panic!("could not find BorrowIndexs for region scope {:?}", region_scope);
                });

                for idx in borrow_indexes { sets.kill(&idx); }
            }

            mir::StatementKind::Assign(_, ref rhs) => {
                if let mir::Rvalue::Ref(region, _, _) = *rhs {
                    let index = self.location_map.get(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {:?}", location);
                    });
                    assert!(self.region_map.get(region).unwrap_or_else(|| {
                        panic!("could not find BorrowIndexs for region {:?}", region);
                    }).contains(&index));
                    sets.gen(&index);
                }
            }

            mir::StatementKind::InlineAsm { .. } |
            mir::StatementKind::SetDiscriminant { .. } |
            mir::StatementKind::StorageLive(..) |
            mir::StatementKind::StorageDead(..) |
            mir::StatementKind::Validate(..) |
            mir::StatementKind::Nop => {}

        }
    }
    fn terminator_effect(&self,
                         _sets: &mut BlockSets<BorrowIndex>,
                         _location: Location) {
        // no terminators start nor end region scopes.
    }

    fn propagate_call_return(&self,
                             _in_out: &mut IdxSet<BorrowIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             _dest_lval: &mir::Lvalue) {
        // there are no effects on the region scopes from method calls.
    }
}

impl<'a, 'tcx> BitwiseOperator for Borrows<'a, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // union effects of preds when computing borrows
    }
}

impl<'a, 'tcx> DataflowOperator for Borrows<'a, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = no Rvalue::Refs are active by default
    }
}
