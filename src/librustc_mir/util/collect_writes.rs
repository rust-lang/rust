use rustc::mir::{Local, Location};
use rustc::mir::Mir;
use rustc::mir::visit::PlaceContext;
use rustc::mir::visit::Visitor;

crate trait FindAssignments {
    // Finds all statements that assign directly to local (i.e., X = ...)
    // and returns their locations.
    fn find_assignments(&self, local: Local) -> Vec<Location>;
}

impl<'tcx> FindAssignments for Mir<'tcx>{
    fn find_assignments(&self, local: Local) -> Vec<Location>{
            let mut visitor = FindLocalAssignmentVisitor{ needle: local, locations: vec![]};
            visitor.visit_mir(self);
            visitor.locations
    }
}

// The Visitor walks the MIR to return the assignment statements corresponding
// to a Local.
struct FindLocalAssignmentVisitor {
    needle: Local,
    locations: Vec<Location>,
}

impl<'tcx> Visitor<'tcx> for FindLocalAssignmentVisitor {
    fn visit_local(&mut self,
                   local: &Local,
                   place_context: PlaceContext<'tcx>,
                   location: Location) {
        if self.needle != *local {
            return;
        }

        if is_place_assignment(&place_context) {
            self.locations.push(location);
        }
    }
}

/// Returns true if this place context represents an assignment statement
crate fn is_place_assignment(place_context: &PlaceContext) -> bool {
    match *place_context {
        PlaceContext::Store | PlaceContext::Call | PlaceContext::AsmOutput => true,
        PlaceContext::Drop
        | PlaceContext::Inspect
        | PlaceContext::Borrow { .. }
        | PlaceContext::Projection(..)
        | PlaceContext::Copy
        | PlaceContext::Move
        | PlaceContext::StorageLive
        | PlaceContext::StorageDead
        | PlaceContext::Validate => false,
    }
}
