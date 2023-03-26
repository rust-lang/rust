#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
use rustc_data_structures::fx::FxIndexMap;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{self, BasicBlock, Body, Location, Place};
use rustc_middle::ty::RegionVid;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::{EverInitializedPlaces, MaybeUninitializedPlaces};
use rustc_mir_dataflow::ResultsVisitable;
use rustc_mir_dataflow::{self, fmt::DebugWithContext, CallReturnPlaces, GenKill};
use rustc_mir_dataflow::{Analysis, Direction, Results};
use std::fmt;

use crate::{
    places_conflict, BorrowSet, PlaceConflictBias, PlaceExt, RegionInferenceContext, ToRegionVid,
};

/// A tuple with named fields that can hold either the results or the transient state of the
/// dataflow analyses used by the borrow checker.
#[derive(Debug)]
pub struct BorrowckAnalyses<B, U, E> {
    pub borrows: B,
    pub uninits: U,
    pub ever_inits: E,
}

/// The results of the dataflow analyses used by the borrow checker.
pub type BorrowckResults<'mir, 'tcx> = BorrowckAnalyses<
    Results<'tcx, Borrows<'mir, 'tcx>>,
    Results<'tcx, MaybeUninitializedPlaces<'mir, 'tcx>>,
    Results<'tcx, EverInitializedPlaces<'mir, 'tcx>>,
>;

/// The transient state of the dataflow analyses used by the borrow checker.
pub type BorrowckFlowState<'mir, 'tcx> =
    <BorrowckResults<'mir, 'tcx> as ResultsVisitable<'tcx>>::FlowState;

macro_rules! impl_visitable {
    ( $(
        $T:ident { $( $field:ident : $A:ident ),* $(,)? }
    )* ) => { $(
        impl<'tcx, $($A),*, D: Direction> ResultsVisitable<'tcx> for $T<$( Results<'tcx, $A> ),*>
        where
            $( $A: Analysis<'tcx, Direction = D>, )*
        {
            type Direction = D;
            type FlowState = $T<$( $A::Domain ),*>;

            fn new_flow_state(&self, body: &mir::Body<'tcx>) -> Self::FlowState {
                $T {
                    $( $field: self.$field.analysis.bottom_value(body) ),*
                }
            }

            fn reset_to_block_entry(
                &self,
                state: &mut Self::FlowState,
                block: BasicBlock,
            ) {
                $( state.$field.clone_from(&self.$field.entry_set_for_block(block)); )*
            }

            fn reconstruct_before_statement_effect(
                &self,
                state: &mut Self::FlowState,
                stmt: &mir::Statement<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_before_statement_effect(&mut state.$field, stmt, loc); )*
            }

            fn reconstruct_statement_effect(
                &self,
                state: &mut Self::FlowState,
                stmt: &mir::Statement<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_statement_effect(&mut state.$field, stmt, loc); )*
            }

            fn reconstruct_before_terminator_effect(
                &self,
                state: &mut Self::FlowState,
                term: &mir::Terminator<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_before_terminator_effect(&mut state.$field, term, loc); )*
            }

            fn reconstruct_terminator_effect(
                &self,
                state: &mut Self::FlowState,
                term: &mir::Terminator<'tcx>,
                loc: Location,
            ) {
                $( self.$field.analysis
                    .apply_terminator_effect(&mut state.$field, term, loc); )*
            }
        }
    )* }
}

impl_visitable! {
    BorrowckAnalyses { borrows: B, uninits: U, ever_inits: E }
}

rustc_index::newtype_index! {
    #[debug_format = "bw{}"]
    pub struct BorrowIndex {}
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

    borrow_set: &'a BorrowSet<'tcx>,
    borrows_out_of_scope_at_location: FxIndexMap<Location, Vec<BorrowIndex>>,
}

struct StackEntry {
    bb: mir::BasicBlock,
    lo: usize,
    hi: usize,
}

struct OutOfScopePrecomputer<'a, 'tcx> {
    visited: BitSet<mir::BasicBlock>,
    visit_stack: Vec<StackEntry>,
    body: &'a Body<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    borrows_out_of_scope_at_location: FxIndexMap<Location, Vec<BorrowIndex>>,
}

impl<'a, 'tcx> OutOfScopePrecomputer<'a, 'tcx> {
    fn new(body: &'a Body<'tcx>, regioncx: &'a RegionInferenceContext<'tcx>) -> Self {
        OutOfScopePrecomputer {
            visited: BitSet::new_empty(body.basic_blocks.len()),
            visit_stack: vec![],
            body,
            regioncx,
            borrows_out_of_scope_at_location: FxIndexMap::default(),
        }
    }
}

impl<'tcx> OutOfScopePrecomputer<'_, 'tcx> {
    fn precompute_borrows_out_of_scope(
        &mut self,
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
        self.visited.insert(location.block);

        let mut first_lo = location.statement_index;
        let first_hi = self.body[location.block].statements.len();

        self.visit_stack.push(StackEntry { bb: location.block, lo: first_lo, hi: first_hi });

        while let Some(StackEntry { bb, lo, hi }) = self.visit_stack.pop() {
            // If we process the first part of the first basic block (i.e. we encounter that block
            // for the second time), we no longer have to visit its successors again.
            let mut finished_early = bb == location.block && hi != first_hi;
            for i in lo..=hi {
                let location = Location { block: bb, statement_index: i };
                // If region does not contain a point at the location, then add to list and skip
                // successor locations.
                if !self.regioncx.region_contains(borrow_region, location) {
                    debug!("borrow {:?} gets killed at {:?}", borrow_index, location);
                    self.borrows_out_of_scope_at_location
                        .entry(location)
                        .or_default()
                        .push(borrow_index);
                    finished_early = true;
                    break;
                }
            }

            if !finished_early {
                // Add successor BBs to the work list, if necessary.
                let bb_data = &self.body[bb];
                debug_assert!(hi == bb_data.statements.len());
                for succ_bb in bb_data.terminator().successors() {
                    if !self.visited.insert(succ_bb) {
                        if succ_bb == location.block && first_lo > 0 {
                            // `succ_bb` has been seen before. If it wasn't
                            // fully processed, add its first part to `stack`
                            // for processing.
                            self.visit_stack.push(StackEntry {
                                bb: succ_bb,
                                lo: 0,
                                hi: first_lo - 1,
                            });

                            // And update this entry with 0, to represent the
                            // whole BB being processed.
                            first_lo = 0;
                        }
                    } else {
                        // succ_bb hasn't been seen before. Add it to
                        // `stack` for processing.
                        self.visit_stack.push(StackEntry {
                            bb: succ_bb,
                            lo: 0,
                            hi: self.body[succ_bb].statements.len(),
                        });
                    }
                }
            }
        }

        self.visited.clear();
    }
}

impl<'a, 'tcx> Borrows<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a Body<'tcx>,
        nonlexical_regioncx: &'a RegionInferenceContext<'tcx>,
        borrow_set: &'a BorrowSet<'tcx>,
    ) -> Self {
        let mut prec = OutOfScopePrecomputer::new(body, nonlexical_regioncx);
        for (borrow_index, borrow_data) in borrow_set.iter_enumerated() {
            let borrow_region = borrow_data.region.to_region_vid();
            let location = borrow_data.reserve_location;

            prec.precompute_borrows_out_of_scope(borrow_index, borrow_region, location);
        }

        Borrows {
            tcx,
            body,
            borrow_set,
            borrows_out_of_scope_at_location: prec.borrows_out_of_scope_at_location,
        }
    }

    pub fn location(&self, idx: BorrowIndex) -> &Location {
        &self.borrow_set[idx].reserve_location
    }

    /// Add all borrows to the kill set, if those borrows are out of scope at `location`.
    /// That means they went out of a nonlexical scope
    fn kill_loans_out_of_scope_at_location(
        &self,
        trans: &mut impl GenKill<BorrowIndex>,
        location: Location,
    ) {
        // NOTE: The state associated with a given `location`
        // reflects the dataflow on entry to the statement.
        // Iterate over each of the borrows that we've precomputed
        // to have went out of scope at this location and kill them.
        //
        // We are careful always to call this function *before* we
        // set up the gen-bits for the statement or
        // terminator. That way, if the effect of the statement or
        // terminator *does* introduce a new loan of the same
        // region, then setting that gen-bit will override any
        // potential kill introduced here.
        if let Some(indices) = self.borrows_out_of_scope_at_location.get(&location) {
            trans.kill_all(indices.iter().copied());
        }
    }

    /// Kill any borrows that conflict with `place`.
    fn kill_borrows_on_place(&self, trans: &mut impl GenKill<BorrowIndex>, place: Place<'tcx>) {
        debug!("kill_borrows_on_place: place={:?}", place);

        let other_borrows_of_local = self
            .borrow_set
            .local_map
            .get(&place.local)
            .into_iter()
            .flat_map(|bs| bs.iter())
            .copied();

        // If the borrowed place is a local with no projections, all other borrows of this
        // local must conflict. This is purely an optimization so we don't have to call
        // `places_conflict` for every borrow.
        if place.projection.is_empty() {
            if !self.body.local_decls[place.local].is_ref_to_static() {
                trans.kill_all(other_borrows_of_local);
            }
            return;
        }

        // By passing `PlaceConflictBias::NoOverlap`, we conservatively assume that any given
        // pair of array indices are not equal, so that when `places_conflict` returns true, we
        // will be assured that two places being compared definitely denotes the same sets of
        // locations.
        let definitely_conflicting_borrows = other_borrows_of_local.filter(|&i| {
            places_conflict(
                self.tcx,
                self.body,
                self.borrow_set[i].borrowed_place,
                place,
                PlaceConflictBias::NoOverlap,
            )
        });

        trans.kill_all(definitely_conflicting_borrows);
    }
}

impl<'tcx> rustc_mir_dataflow::AnalysisDomain<'tcx> for Borrows<'_, 'tcx> {
    type Domain = BitSet<BorrowIndex>;

    const NAME: &'static str = "borrows";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = nothing is reserved or activated yet;
        BitSet::new_empty(self.borrow_set.len() * 2)
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect.
    }
}

impl<'tcx> rustc_mir_dataflow::GenKillAnalysis<'tcx> for Borrows<'_, 'tcx> {
    type Idx = BorrowIndex;

    fn before_statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        stmt: &mir::Statement<'tcx>,
        location: Location,
    ) {
        match &stmt.kind {
            mir::StatementKind::Assign(box (lhs, rhs)) => {
                if let mir::Rvalue::Ref(_, _, place) = rhs {
                    if place.ignore_borrow(
                        self.tcx,
                        self.body,
                        &self.borrow_set.locals_state_at_exit,
                    ) {
                        return;
                    }
                    let index = self.borrow_set.get_index_of(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {:?}", location);
                    });

                    trans.gen(index);
                }

                // Make sure there are no remaining borrows for variables
                // that are assigned over.
                self.kill_borrows_on_place(trans, *lhs);
            }

            mir::StatementKind::StorageDead(local) => {
                // Make sure there are no remaining borrows for locals that
                // are gone out of scope.
                self.kill_borrows_on_place(trans, Place::from(*local));
            }

            mir::StatementKind::FakeRead(..)
            | mir::StatementKind::SetDiscriminant { .. }
            | mir::StatementKind::Deinit(..)
            | mir::StatementKind::StorageLive(..)
            | mir::StatementKind::Retag { .. }
            | mir::StatementKind::PlaceMention(..)
            | mir::StatementKind::AscribeUserType(..)
            | mir::StatementKind::Coverage(..)
            | mir::StatementKind::Intrinsic(..)
            | mir::StatementKind::ConstEvalCounter
            | mir::StatementKind::Nop => {}
        }
    }

    fn before_terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        _location: Location,
    ) {
        if let mir::TerminatorKind::InlineAsm { operands, .. } = &terminator.kind {
            for op in operands {
                if let mir::InlineAsmOperand::Out { place: Some(place), .. }
                | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } = *op
                {
                    self.kill_borrows_on_place(trans, place);
                }
            }
        }
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }
}

impl DebugWithContext<Borrows<'_, '_>> for BorrowIndex {
    fn fmt_with(&self, ctxt: &Borrows<'_, '_>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", ctxt.location(*self))
    }
}
