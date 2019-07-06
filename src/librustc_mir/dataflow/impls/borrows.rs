use crate::borrow_check::borrow_set::{BorrowSet, BorrowData};
use crate::borrow_check::place_ext::PlaceExt;

use rustc::mir::{self, Location, Place, PlaceBase, Body};
use rustc::ty::TyCtxt;
use rustc::ty::RegionVid;

use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

use crate::dataflow::{BitDenotation, BottomValue, GenKillSet};
use crate::borrow_check::nll::region_infer::RegionInferenceContext;
use crate::borrow_check::nll::ToRegionVid;
use crate::borrow_check::places_conflict;

use std::rc::Rc;

newtype_index! {
    pub struct BorrowIndex {
        DEBUG_FORMAT = "bw{}"
    }
}

/// `Borrows` stores the data used in the analyses that track the flow
/// of borrows.
///
/// It uniquely identifies every borrow (`Rvalue::Ref`) by a
/// `BorrowIndex`, and maps each such index to a `BorrowData`
/// describing the borrow. These indexes are used for representing the
/// borrows in compact bitvectors.
pub struct Borrows<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,

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
    body: &Body<'tcx>,
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
        hi: body[location.block].statements.len(),
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
            let bb_data = &body[bb];
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
                            hi: body[succ_bb].statements.len(),
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

impl<'a, 'tcx> Borrows<'a, 'tcx> {
    crate fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a Body<'tcx>,
        nonlexical_regioncx: Rc<RegionInferenceContext<'tcx>>,
        borrow_set: &Rc<BorrowSet<'tcx>>,
    ) -> Self {
        let mut borrows_out_of_scope_at_location = FxHashMap::default();
        for (borrow_index, borrow_data) in borrow_set.borrows.iter_enumerated() {
            let borrow_region = borrow_data.region.to_region_vid();
            let location = borrow_set.borrows[borrow_index].reserve_location;

            precompute_borrows_out_of_scope(body, &nonlexical_regioncx,
                                            &mut borrows_out_of_scope_at_location,
                                            borrow_index, borrow_region, location);
        }

        Borrows {
            tcx: tcx,
            body: body,
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
                                           trans: &mut GenKillSet<BorrowIndex>,
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
            trans.kill_all(indices);
        }
    }

    /// Kill any borrows that conflict with `place`.
    fn kill_borrows_on_place(
        &self,
        trans: &mut GenKillSet<BorrowIndex>,
        place: &Place<'tcx>
    ) {
        debug!("kill_borrows_on_place: place={:?}", place);

        if let Some(local) = place.base_local() {
            let other_borrows_of_local = self
                .borrow_set
                .local_map
                .get(&local)
                .into_iter()
                .flat_map(|bs| bs.into_iter());

            // If the borrowed place is a local with no projections, all other borrows of this
            // local must conflict. This is purely an optimization so we don't have to call
            // `places_conflict` for every borrow.
            if let Place::Base(PlaceBase::Local(_)) = place {
                trans.kill_all(other_borrows_of_local);
                return;
            }

            // By passing `PlaceConflictBias::NoOverlap`, we conservatively assume that any given
            // pair of array indices are unequal, so that when `places_conflict` returns true, we
            // will be assured that two places being compared definitely denotes the same sets of
            // locations.
            let definitely_conflicting_borrows = other_borrows_of_local
                .filter(|&&i| {
                    places_conflict::places_conflict(
                        self.tcx,
                        self.body,
                        &self.borrow_set.borrows[i].borrowed_place,
                        place,
                        places_conflict::PlaceConflictBias::NoOverlap)
                });

            trans.kill_all(definitely_conflicting_borrows);
        }
    }
}

impl<'a, 'tcx> BitDenotation<'tcx> for Borrows<'a, 'tcx> {
    type Idx = BorrowIndex;
    fn name() -> &'static str { "borrows" }
    fn bits_per_block(&self) -> usize {
        self.borrow_set.borrows.len() * 2
    }

    fn start_block_effect(&self, _entry_set: &mut BitSet<Self::Idx>) {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect.
    }

    fn before_statement_effect(&self,
                               trans: &mut GenKillSet<Self::Idx>,
                               location: Location) {
        debug!("Borrows::before_statement_effect trans: {:?} location: {:?}",
               trans, location);
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn statement_effect(&self,
                        trans: &mut GenKillSet<Self::Idx>,
                        location: Location) {
        debug!("Borrows::statement_effect: trans={:?} location={:?}",
               trans, location);

        let block = &self.body.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });
        let stmt = block.statements.get(location.statement_index).unwrap_or_else(|| {
            panic!("could not find statement at location {:?}");
        });

        debug!("Borrows::statement_effect: stmt={:?}", stmt);
        match stmt.kind {
            mir::StatementKind::Assign(ref lhs, ref rhs) => {
                // Make sure there are no remaining borrows for variables
                // that are assigned over.
                self.kill_borrows_on_place(trans, lhs);

                if let mir::Rvalue::Ref(_, _, ref place) = **rhs {
                    if place.ignore_borrow(
                        self.tcx,
                        self.body,
                        &self.borrow_set.locals_state_at_exit,
                    ) {
                        return;
                    }
                    let index = self.borrow_set.location_map.get(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {:?}", location);
                    });

                    trans.gen(*index);
                }
            }

            mir::StatementKind::StorageDead(local) => {
                // Make sure there are no remaining borrows for locals that
                // are gone out of scope.
                self.kill_borrows_on_place(trans, &Place::from(local));
            }

            mir::StatementKind::InlineAsm(ref asm) => {
                for (output, kind) in asm.outputs.iter().zip(&asm.asm.outputs) {
                    if !kind.is_indirect && !kind.is_rw {
                        self.kill_borrows_on_place(trans, output);
                    }
                }
            }

            mir::StatementKind::FakeRead(..) |
            mir::StatementKind::SetDiscriminant { .. } |
            mir::StatementKind::StorageLive(..) |
            mir::StatementKind::Retag { .. } |
            mir::StatementKind::AscribeUserType(..) |
            mir::StatementKind::Nop => {}

        }
    }

    fn before_terminator_effect(&self,
                                trans: &mut GenKillSet<Self::Idx>,
                                location: Location) {
        debug!("Borrows::before_terminator_effect: trans={:?} location={:?}",
               trans, location);
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn terminator_effect(&self,
                         _: &mut GenKillSet<Self::Idx>,
                         _: Location) {}

    fn propagate_call_return(
        &self,
        _in_out: &mut BitSet<BorrowIndex>,
        _call_bb: mir::BasicBlock,
        _dest_bb: mir::BasicBlock,
        _dest_place: &mir::Place<'tcx>,
    ) {
    }
}

impl<'a, 'tcx> BottomValue for Borrows<'a, 'tcx> {
    /// bottom = nothing is reserved or activated yet;
    const BOTTOM_VALUE: bool = false;
}
