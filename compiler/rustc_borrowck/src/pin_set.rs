use std::ops::Index;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_middle::mir::{self, Body, Location, traversal};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::ty::TyCtxt;
use rustc_abi::FieldIdx;

use crate::PinIndex;

pub(crate) struct PinSet<'tcx> {
    /// The fundamental map relating bitvector indexes to the pins
    /// in the MIR. Each pin is also uniquely identified in the MIR
    /// by the `Location` of the assignment statement in which it
    /// appears on the right hand side. Thus the location is the map
    /// key, and its position in the map corresponds to `PinIndex`.
    pub(crate) location_map: FxIndexMap<Location, PinData<'tcx>>,

    /// Map from local to all the pins on that local.
    pub(crate) local_map: FxIndexMap<mir::Local, FxIndexSet<PinIndex>>,
}

#[derive(Debug, Clone)]
pub(crate) struct PinData<'tcx> {
    /// Location where the pin is created (the aggregate statement).
    #[allow(dead_code)]
    pub(crate) location: Location,
    /// The place that is pinned (the result of the Pin aggregate).
    pub(crate) pinned_place: mir::Place<'tcx>,
    /// The borrowed place that is wrapped in Pin.
    #[allow(dead_code)]
    pub(crate) borrowed_place: mir::Place<'tcx>,
}

impl<'tcx> PinSet<'tcx> {
    pub(crate) fn build(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut visitor = GatherPins {
            tcx,
            location_map: Default::default(),
            local_map: Default::default(),
        };

        for (block, block_data) in traversal::preorder(body) {
            visitor.visit_basic_block_data(block, block_data);
        }

        PinSet {
            location_map: visitor.location_map,
            local_map: visitor.local_map,
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

struct GatherPins<'tcx> {
    tcx: TyCtxt<'tcx>,
    location_map: FxIndexMap<Location, PinData<'tcx>>,
    local_map: FxIndexMap<mir::Local, FxIndexSet<PinIndex>>,
}

impl<'tcx> Visitor<'tcx> for GatherPins<'tcx> {
    fn visit_assign(
        &mut self,
        pinned_place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        // Look for patterns like:
        // pinned_place = Pin::<&[mut] T> { pointer: move borrowed_place }
        if let mir::Rvalue::Aggregate(box agg_kind, operands) = rvalue
            && let mir::AggregateKind::Adt(adt_did, _, args, _, _) = agg_kind
            && self.tcx.adt_def(adt_did).is_pin()
            && args.type_at(0).is_ref()
            && let mir::Operand::Move(borrowed_place) = operands[FieldIdx::ZERO]
        {
            let pin_data = PinData {
                location,
                pinned_place: *pinned_place,
                borrowed_place,
            };

            let (idx, _) = self.location_map.insert_full(location, pin_data);
            let idx = PinIndex::from(idx);

            self.local_map.entry(borrowed_place.local).or_default().insert(idx);
        }

        self.super_assign(pinned_place, rvalue, location)
    }
}
