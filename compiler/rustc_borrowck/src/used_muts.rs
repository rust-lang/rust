use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    Local, Location, Place, Statement, StatementKind, Terminator, TerminatorKind,
};

use crate::MirBorrowckCtxt;

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Walks the MIR adding to the set of `used_mut` locals that will be ignored for the purposes
    /// of the `unused_mut` lint.
    ///
    /// `temporary_used_locals` should contain locals that were found to be temporary, mutable and
    ///  used from borrow checking. This function looks for assignments into these locals from
    ///  user-declared locals and adds those user-defined locals to the `used_mut` set. This can
    ///  occur due to a rare case involving upvars in closures.
    ///
    /// `never_initialized_mut_locals` should contain the set of user-declared mutable locals
    ///  (not arguments) that have not already been marked as being used.
    ///  This function then looks for assignments from statements or the terminator into the locals
    ///  from this set and removes them from the set. This leaves only those locals that have not
    ///  been assigned to - this set is used as a proxy for locals that were not initialized due to
    ///  unreachable code. These locals are then considered "used" to silence the lint for them.
    ///  See #55344 for context.
    crate fn gather_used_muts(
        &mut self,
        temporary_used_locals: FxHashSet<Local>,
        mut never_initialized_mut_locals: FxHashSet<Local>,
    ) {
        {
            let mut visitor = GatherUsedMutsVisitor {
                temporary_used_locals,
                never_initialized_mut_locals: &mut never_initialized_mut_locals,
                mbcx: self,
            };
            visitor.visit_body(&visitor.mbcx.body);
        }

        // Take the union of the existed `used_mut` set with those variables we've found were
        // never initialized.
        debug!("gather_used_muts: never_initialized_mut_locals={:?}", never_initialized_mut_locals);
        self.used_mut = self.used_mut.union(&never_initialized_mut_locals).cloned().collect();
    }
}

/// MIR visitor for collecting used mutable variables.
/// The 'visit lifetime represents the duration of the MIR walk.
struct GatherUsedMutsVisitor<'visit, 'cx, 'tcx> {
    temporary_used_locals: FxHashSet<Local>,
    never_initialized_mut_locals: &'visit mut FxHashSet<Local>,
    mbcx: &'visit mut MirBorrowckCtxt<'cx, 'tcx>,
}

impl GatherUsedMutsVisitor<'_, '_, '_> {
    fn remove_never_initialized_mut_locals(&mut self, into: Place<'_>) {
        // Remove any locals that we found were initialized from the
        // `never_initialized_mut_locals` set. At the end, the only remaining locals will
        // be those that were never initialized - we will consider those as being used as
        // they will either have been removed by unreachable code optimizations; or linted
        // as unused variables.
        self.never_initialized_mut_locals.remove(&into.local);
    }
}

impl<'visit, 'cx, 'tcx> Visitor<'tcx> for GatherUsedMutsVisitor<'visit, 'cx, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        debug!("visit_terminator: terminator={:?}", terminator);
        match &terminator.kind {
            TerminatorKind::Call { destination: Some((into, _)), .. } => {
                self.remove_never_initialized_mut_locals(*into);
            }
            TerminatorKind::DropAndReplace { place, .. } => {
                self.remove_never_initialized_mut_locals(*place);
            }
            _ => {}
        }

        self.super_terminator(terminator, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        if let StatementKind::Assign(box (into, _)) = &statement.kind {
            debug!(
                "visit_statement: statement={:?} local={:?} \
                    never_initialized_mut_locals={:?}",
                statement, into.local, self.never_initialized_mut_locals
            );
            self.remove_never_initialized_mut_locals(*into);
        }

        self.super_statement(statement, location);
    }

    fn visit_local(&mut self, local: &Local, place_context: PlaceContext, location: Location) {
        if place_context.is_place_assignment() && self.temporary_used_locals.contains(local) {
            // Propagate the Local assigned at this Location as a used mutable local variable
            for moi in &self.mbcx.move_data.loc_map[location] {
                let mpi = &self.mbcx.move_data.moves[*moi].path;
                let path = &self.mbcx.move_data.move_paths[*mpi];
                debug!(
                    "assignment of {:?} to {:?}, adding {:?} to used mutable set",
                    path.place, local, path.place
                );
                if let Some(user_local) = path.place.as_local() {
                    self.mbcx.used_mut.insert(user_local);
                }
            }
        }
    }
}
