use std::collections::BTreeSet;

use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{Body, Local, Location};

/// Find all uses of (including assignments to) a [`Local`].
///
/// Uses `BTreeSet` so output is deterministic.
pub(super) fn find<'tcx>(body: &Body<'tcx>, local: Local) -> BTreeSet<Location> {
    let mut visitor = AllLocalUsesVisitor { for_local: local, uses: BTreeSet::default() };
    visitor.visit_body(body);
    visitor.uses
}

struct AllLocalUsesVisitor {
    for_local: Local,
    uses: BTreeSet<Location>,
}

impl<'tcx> Visitor<'tcx> for AllLocalUsesVisitor {
    fn visit_local(&mut self, local: &Local, _context: PlaceContext, location: Location) {
        if *local == self.for_local {
            self.uses.insert(location);
        }
    }
}
