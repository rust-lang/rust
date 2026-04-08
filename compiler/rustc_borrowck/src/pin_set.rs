use rustc_abi::FieldIdx;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Body, Location, traversal};
use rustc_middle::ty::TyCtxt;

use crate::consumers::BorrowSet;
use crate::dataflow::BorrowIndex;

pub(crate) struct PinSet {
    /// Map from Pin aggregate location to the borrows it pinned.
    pub(crate) pin_location_map: FxIndexMap<Location, FxIndexSet<BorrowIndex>>,
}

impl PinSet {
    pub(crate) fn build<'tcx>(
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        borrow_set: &BorrowSet<'tcx>,
    ) -> Self {
        let mut visitor =
            GatherPins { tcx, body, borrow_set, pin_location_map: Default::default() };

        for (block, block_data) in traversal::preorder(body) {
            visitor.visit_basic_block_data(block, block_data);
        }

        PinSet { pin_location_map: visitor.pin_location_map }
    }
}

struct GatherPins<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    borrow_set: &'a BorrowSet<'tcx>,
    pin_location_map: FxIndexMap<Location, FxIndexSet<BorrowIndex>>,
}

impl<'tcx> Visitor<'tcx> for GatherPins<'_, 'tcx> {
    fn visit_assign(
        &mut self,
        pin_result_place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        // Look for patterns like:
        //   _ref = &[mut] _original;                              // predecessor statement
        //   pin_result_place = Pin::<&[mut] T> { pointer: move _ref };  // this statement
        //
        // We want to track `_original` as the pinned place.
        if let mir::Rvalue::Aggregate(box agg_kind, operands) = rvalue
            && let mir::AggregateKind::Adt(adt_did, _, args, _, _) = agg_kind
            && self.tcx.adt_def(*adt_did).is_pin()
            && args.type_at(0).is_ref()
            && let mir::Operand::Move(ref_place) = operands[FieldIdx::ZERO]
            // Look at the preceding statement to find the original borrow:
            //   _ref = &[mut] _original
            && let Some((borrow_location, _borrowed_place)) = self.find_original_borrowed_place(location, ref_place)
            && let Some(idx) = self.borrow_set.get_index_of(&borrow_location)
        {
            self.pin_location_map.entry(location).or_default().insert(idx);
        }

        self.super_assign(pin_result_place, rvalue, location)
    }
}

impl<'tcx> GatherPins<'_, 'tcx> {
    /// Given a Pin aggregate at `location` that uses `move ref_place`,
    /// look at the preceding statement to find the original borrow.
    /// i.e. find `_original` in `ref_place = &[mut] _original`.
    fn find_original_borrowed_place(
        &self,
        location: Location,
        ref_place: mir::Place<'tcx>,
    ) -> Option<(mir::Location, mir::Place<'tcx>)> {
        // The borrow statement should be immediately before the Pin aggregate
        if location.statement_index == 0 {
            return None;
        }
        let predecessor =
            Location { block: location.block, statement_index: location.statement_index - 1 };
        let stmt =
            &self.body.basic_blocks[predecessor.block].statements[predecessor.statement_index];

        if let mir::StatementKind::Assign(box (place, ref rvalue)) = stmt.kind
            && place == ref_place
            && let &mir::Rvalue::Ref(_, _, original_place) = rvalue
        {
            Some((predecessor, original_place))
        } else {
            None
        }
    }
}
