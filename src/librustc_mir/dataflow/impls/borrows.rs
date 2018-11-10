// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::{BorrowSet, BorrowData};
use borrow_check::place_ext::PlaceExt;

use rustc;
use rustc::mir::{self, Location, Place, Mir};
use rustc::ty::TyCtxt;
use rustc::ty::RegionVid;

use rustc_data_structures::bit_set::{BitSet, BitSetOperator};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

use dataflow::{BitDenotation, BlockSets, InitialFlow};
pub use dataflow::indexes::BorrowIndex;
use borrow_check::nll::region_infer::RegionInferenceContext;
use borrow_check::nll::ToRegionVid;

use std::rc::Rc;

/// `Borrows` stores the data used in the analyses that track the flow
/// of borrows.
///
/// It uniquely identifies every borrow (`Rvalue::Ref`) by a
/// `BorrowIndex`, and maps each such index to a `BorrowData`
/// describing the borrow. These indexes are used for representing the
/// borrows in compact bitvectors.
pub struct Borrows<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,

    borrow_set: Rc<BorrowSet<'tcx>>,
    borrows_out_of_scope_at_location: FxHashMap<Location, Vec<BorrowIndex>>,

    /// NLL region inference context with which NLL queries should be resolved
    _nonlexical_regioncx: Rc<RegionInferenceContext<'tcx>>,
}

struct StackEntry {
    bb: mir::BasicBlock,
    lo: usize,
    hi: usize,
    first_part_only: bool
}

fn precompute_borrows_out_of_scope<'tcx>(
    mir: &Mir<'tcx>,
    regioncx: &Rc<RegionInferenceContext<'tcx>>,
    borrows_out_of_scope_at_location: &mut FxHashMap<Location, Vec<BorrowIndex>>,
    borrow_index: BorrowIndex,
    borrow_region: RegionVid,
    location: Location,
) {
    // We visit one BB at a time. The complication is that we may start in the
    // middle of the first BB visited (the one containing `location`), in which
    // case we may have to later on process the first part of that BB if there
    // is a path back to its start.

    // For visited BBs, we record the index of the first statement processed.
    // (In fully processed BBs this index is 0.) Note also that we add BBs to
    // `visited` once they are added to `stack`, before they are actually
    // processed, because this avoids the need to look them up again on
    // completion.
    let mut visited = FxHashMap::default();
    visited.insert(location.block, location.statement_index);

    let mut stack = vec![];
    stack.push(StackEntry {
        bb: location.block,
        lo: location.statement_index,
        hi: mir[location.block].statements.len(),
        first_part_only: false,
    });

    while let Some(StackEntry { bb, lo, hi, first_part_only }) = stack.pop() {
        let mut finished_early = first_part_only;
        for i in lo ..= hi {
            let location = Location { block: bb, statement_index: i };
            // If region does not contain a point at the location, then add to list and skip
            // successor locations.
            if !regioncx.region_contains(borrow_region, location) {
                debug!("borrow {:?} gets killed at {:?}", borrow_index, location);
                borrows_out_of_scope_at_location
                    .entry(location)
                    .or_default()
                    .push(borrow_index);
                finished_early = true;
                break;
            }
        }

        if !finished_early {
            // Add successor BBs to the work list, if necessary.
            let bb_data = &mir[bb];
            assert!(hi == bb_data.statements.len());
            for &succ_bb in bb_data.terminator.as_ref().unwrap().successors() {
                visited.entry(succ_bb)
                    .and_modify(|lo| {
                        // `succ_bb` has been seen before. If it wasn't
                        // fully processed, add its first part to `stack`
                        // for processing.
                        if *lo > 0 {
                            stack.push(StackEntry {
                                bb: succ_bb,
                                lo: 0,
                                hi: *lo - 1,
                                first_part_only: true,
                            });
                        }
                        // And update this entry with 0, to represent the
                        // whole BB being processed.
                        *lo = 0;
                    })
                    .or_insert_with(|| {
                        // succ_bb hasn't been seen before. Add it to
                        // `stack` for processing.
                        stack.push(StackEntry {
                            bb: succ_bb,
                            lo: 0,
                            hi: mir[succ_bb].statements.len(),
                            first_part_only: false,
                        });
                        // Insert 0 for this BB, to represent the whole BB
                        // being processed.
                        0
                    });
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> Borrows<'a, 'gcx, 'tcx> {
    crate fn new(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        mir: &'a Mir<'tcx>,
        nonlexical_regioncx: Rc<RegionInferenceContext<'tcx>>,
        borrow_set: &Rc<BorrowSet<'tcx>>,
    ) -> Self {
        let mut borrows_out_of_scope_at_location = FxHashMap::default();
        for (borrow_index, borrow_data) in borrow_set.borrows.iter_enumerated() {
            let borrow_region = borrow_data.region.to_region_vid();
            let location = borrow_set.borrows[borrow_index].reserve_location;

            precompute_borrows_out_of_scope(mir, &nonlexical_regioncx,
                                            &mut borrows_out_of_scope_at_location,
                                            borrow_index, borrow_region, location);
        }

        Borrows {
            tcx: tcx,
            mir: mir,
            borrow_set: borrow_set.clone(),
            borrows_out_of_scope_at_location,
            _nonlexical_regioncx: nonlexical_regioncx,
        }
    }

    crate fn borrows(&self) -> &IndexVec<BorrowIndex, BorrowData<'tcx>> { &self.borrow_set.borrows }

    pub fn location(&self, idx: BorrowIndex) -> &Location {
        &self.borrow_set.borrows[idx].reserve_location
    }

    /// Add all borrows to the kill set, if those borrows are out of scope at `location`.
    /// That means they went out of a nonlexical scope
    fn kill_loans_out_of_scope_at_location(&self,
                                           sets: &mut BlockSets<BorrowIndex>,
                                           location: Location) {
        // NOTE: The state associated with a given `location`
        // reflects the dataflow on entry to the statement.
        // Iterate over each of the borrows that we've precomputed
        // to have went out of scope at this location and kill them.
        //
        // We are careful always to call this function *before* we
        // set up the gen-bits for the statement or
        // termanator. That way, if the effect of the statement or
        // terminator *does* introduce a new loan of the same
        // region, then setting that gen-bit will override any
        // potential kill introduced here.
        if let Some(indices) = self.borrows_out_of_scope_at_location.get(&location) {
            sets.kill_all(indices);
        }
    }

    fn kill_borrows_on_local(&self,
                             sets: &mut BlockSets<BorrowIndex>,
                             local: &rustc::mir::Local)
    {
        if let Some(borrow_indexes) = self.borrow_set.local_map.get(local) {
            sets.kill_all(borrow_indexes);
        }
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for Borrows<'a, 'gcx, 'tcx> {
    type Idx = BorrowIndex;
    fn name() -> &'static str { "borrows" }
    fn bits_per_block(&self) -> usize {
        self.borrow_set.borrows.len() * 2
    }

    fn start_block_effect(&self, _entry_set: &mut BitSet<BorrowIndex>) {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect on
        // `_sets`.
    }

    fn before_statement_effect(&self,
                               sets: &mut BlockSets<BorrowIndex>,
                               location: Location) {
        debug!("Borrows::before_statement_effect sets: {:?} location: {:?}", sets, location);
        self.kill_loans_out_of_scope_at_location(sets, location);
    }

    fn statement_effect(&self, sets: &mut BlockSets<BorrowIndex>, location: Location) {
        debug!("Borrows::statement_effect sets: {:?} location: {:?}", sets, location);

        let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });
        let stmt = block.statements.get(location.statement_index).unwrap_or_else(|| {
            panic!("could not find statement at location {:?}");
        });

        match stmt.kind {
            mir::StatementKind::Assign(ref lhs, ref rhs) => {
                // Make sure there are no remaining borrows for variables
                // that are assigned over.
                if let Place::Local(ref local) = *lhs {
                    // FIXME: Handle the case in which we're assigning over
                    // a projection (`foo.bar`).
                    self.kill_borrows_on_local(sets, local);
                }

                // NOTE: if/when the Assign case is revised to inspect
                // the assigned_place here, make sure to also
                // re-consider the current implementations of the
                // propagate_call_return method.

                if let mir::Rvalue::Ref(region, _, ref place) = **rhs {
                    if place.ignore_borrow(
                        self.tcx,
                        self.mir,
                        &self.borrow_set.locals_state_at_exit,
                    ) {
                        return;
                    }
                    let index = self.borrow_set.location_map.get(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {:?}", location);
                    });

                    assert!(self.borrow_set.region_map
                        .get(&region.to_region_vid())
                        .unwrap_or_else(|| {
                            panic!("could not find BorrowIndexs for RegionVid {:?}", region);
                        })
                        .contains(&index)
                    );
                    sets.gen(*index);

                    // Issue #46746: Two-phase borrows handles
                    // stmts of form `Tmp = &mut Borrow` ...
                    match lhs {
                        Place::Promoted(_) |
                        Place::Local(..) | Place::Static(..) => {} // okay
                        Place::Projection(..) => {
                            // ... can assign into projections,
                            // e.g. `box (&mut _)`. Current
                            // conservative solution: force
                            // immediate activation here.
                            sets.gen(*index);
                        }
                    }
                }
            }

            mir::StatementKind::StorageDead(local) => {
                // Make sure there are no remaining borrows for locals that
                // are gone out of scope.
                self.kill_borrows_on_local(sets, &local)
            }

            mir::StatementKind::InlineAsm { ref outputs, ref asm, .. } => {
                for (output, kind) in outputs.iter().zip(&asm.outputs) {
                    if !kind.is_indirect && !kind.is_rw {
                        // Make sure there are no remaining borrows for direct
                        // output variables.
                        if let Place::Local(ref local) = *output {
                            // FIXME: Handle the case in which we're assigning over
                            // a projection (`foo.bar`).
                            self.kill_borrows_on_local(sets, local);
                        }
                    }
                }
            }

            mir::StatementKind::FakeRead(..) |
            mir::StatementKind::SetDiscriminant { .. } |
            mir::StatementKind::StorageLive(..) |
            mir::StatementKind::Retag { .. } |
            mir::StatementKind::EscapeToRaw { .. } |
            mir::StatementKind::AscribeUserType(..) |
            mir::StatementKind::Nop => {}

        }
    }

    fn before_terminator_effect(&self,
                                sets: &mut BlockSets<BorrowIndex>,
                                location: Location) {
        debug!("Borrows::before_terminator_effect sets: {:?} location: {:?}", sets, location);
        self.kill_loans_out_of_scope_at_location(sets, location);
    }

    fn terminator_effect(&self, _: &mut BlockSets<BorrowIndex>, _: Location) {}

    fn propagate_call_return(&self,
                             _in_out: &mut BitSet<BorrowIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             _dest_place: &mir::Place) {
        // there are no effects on borrows from method call return...
        //
        // ... but if overwriting a place can affect flow state, then
        // latter is not true; see NOTE on Assign case in
        // statement_effect_on_borrows.
    }
}

impl<'a, 'gcx, 'tcx> BitSetOperator for Borrows<'a, 'gcx, 'tcx> {
    #[inline]
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool {
        inout_set.union(in_set) // "maybe" means we union effects of both preds
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for Borrows<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = nothing is reserved or activated yet
    }
}

