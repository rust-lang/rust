use std::ops::Index;

use rustc_abi::FieldIdx;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Body, Location, traversal};
use rustc_middle::ty::TyCtxt;

use crate::dataflow::PinIndex;

pub(crate) struct PinSet<'tcx> {
    /// The fundamental map relating bitvector indexes to the pins
    /// in the MIR. Each pin is also uniquely identified in the MIR
    /// by the `Location` of the assignment statement in which it
    /// appears on the right hand side. Thus the location is the map
    /// key, and its position in the map corresponds to `PinIndex`.
    pub(crate) location_map: FxIndexMap<Location, PinData<'tcx>>,

    /// Map from the original borrowed local to all the pins on that local.
    pub(crate) local_map: FxIndexMap<mir::Local, FxIndexSet<PinIndex>>,

    /// Map from Pin result local to all the pins it holds.
    pub(crate) pin_local_map: FxIndexMap<mir::Local, FxIndexSet<PinIndex>>,
}

#[derive(Debug, Clone)]
pub(crate) struct PinData<'tcx> {
    /// Location where the pin is created (the aggregate statement).
    #[allow(dead_code)]
    pub(crate) location: Location,
    /// The original place being borrowed/pinned (e.g. `_1` for `x` in `&pin mut x`).
    pub(crate) pinned_place: mir::Place<'tcx>,
    /// The local holding the Pin aggregate result (e.g. `_3` for `let _x = &pin mut x`).
    /// The pin is killed when this local has StorageDead.
    #[allow(dead_code)]
    pub(crate) pin_result_local: mir::Local,
}

impl<'tcx> PinSet<'tcx> {
    pub(crate) fn build(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut visitor = GatherPins {
            tcx,
            body,
            location_map: Default::default(),
            local_map: Default::default(),
            pin_local_map: Default::default(),
        };

        for (block, block_data) in traversal::preorder(body) {
            visitor.visit_basic_block_data(block, block_data);
        }

        PinSet {
            location_map: visitor.location_map,
            local_map: visitor.local_map,
            pin_local_map: visitor.pin_local_map,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.location_map.len()
    }

    #[allow(dead_code)]
    pub(crate) fn indices(&self) -> impl Iterator<Item = PinIndex> {
        PinIndex::ZERO..PinIndex::from_usize(self.len())
    }

    #[allow(dead_code)]
    pub(crate) fn iter_enumerated(&self) -> impl Iterator<Item = (PinIndex, &PinData<'tcx>)> {
        self.indices().zip(self.location_map.values())
    }

    pub(crate) fn get_index_of(&self, location: &Location) -> Option<PinIndex> {
        self.location_map.get_index_of(location).map(PinIndex::from)
    }
}

impl<'tcx> Index<PinIndex> for PinSet<'tcx> {
    type Output = PinData<'tcx>;

    fn index(&self, index: PinIndex) -> &PinData<'tcx> {
        &self.location_map[index.as_usize()]
    }
}

struct GatherPins<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    location_map: FxIndexMap<Location, PinData<'tcx>>,
    local_map: FxIndexMap<mir::Local, FxIndexSet<PinIndex>>,
    pin_local_map: FxIndexMap<mir::Local, FxIndexSet<PinIndex>>,
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
        {
            // Look at the preceding statement to find the original borrow:
            //   _ref = &[mut] _original
            if let Some(original_place) =
                self.find_original_borrowed_place(location, ref_place)
            {
                let pin_data = PinData {
                    location,
                    pinned_place: original_place,
                    pin_result_local: pin_result_place.local,
                };

                let (idx, _) = self.location_map.insert_full(location, pin_data);
                let idx = PinIndex::from(idx);

                self.local_map.entry(original_place.local).or_default().insert(idx);
                self.pin_local_map.entry(pin_result_place.local).or_default().insert(idx);
            }
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
    ) -> Option<mir::Place<'tcx>> {
        // The borrow statement should be immediately before the Pin aggregate
        if location.statement_index == 0 {
            return None;
        }
        let predecessor = Location {
            block: location.block,
            statement_index: location.statement_index - 1,
        };
        let stmt =
            &self.body.basic_blocks[predecessor.block].statements[predecessor.statement_index];

        if let mir::StatementKind::Assign(box (place, ref rvalue)) = stmt.kind
            && place == ref_place
            && let mir::Rvalue::Ref(_, _, original_place) = rvalue
        {
            Some(*original_place)
        } else {
            None
        }
    }
}
