use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::mir::{Local, Location, Place};

use rustc_data_structures::fx::FxHashSet;

use borrow_check::MirBorrowckCtxt;
use util::collect_writes::is_place_assignment;

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    /// Walks the MIR looking for assignments to a set of locals, as part of the unused mutable
    /// local variables lint, to update the context's `used_mut` in a single walk.
    crate fn gather_used_muts(&mut self, locals: FxHashSet<Local>) {
        let mut visitor = GatherUsedMutsVisitor {
            needles: locals,
            mbcx: self,
        };
        visitor.visit_mir(visitor.mbcx.mir);
    }
}

/// MIR visitor gathering the assignments to a set of locals, in a single walk.
/// 'visit = the duration of the MIR walk
struct GatherUsedMutsVisitor<'visit, 'cx: 'visit, 'gcx: 'tcx, 'tcx: 'cx> {
    needles: FxHashSet<Local>,
    mbcx: &'visit mut MirBorrowckCtxt<'cx, 'gcx, 'tcx>,
}

impl<'visit, 'cx, 'gcx, 'tcx> Visitor<'tcx> for GatherUsedMutsVisitor<'visit, 'cx, 'gcx, 'tcx> {
    fn visit_local(
        &mut self,
        local: &Local,
        place_context: PlaceContext<'tcx>,
        location: Location,
    ) {
        if !self.needles.contains(local) {
            return;
        }

        if is_place_assignment(&place_context) {
            // Propagate the Local assigned at this Location as a used mutable local variable
            for moi in &self.mbcx.move_data.loc_map[location] {
                let mpi = &self.mbcx.move_data.moves[*moi].path;
                let path = &self.mbcx.move_data.move_paths[*mpi];
                debug!(
                    "assignment of {:?} to {:?}, adding {:?} to used mutable set",
                    path.place, local, path.place
                );
                if let Place::Local(user_local) = path.place {
                    self.mbcx.used_mut.insert(user_local);
                }
            }
        }
    }
}
