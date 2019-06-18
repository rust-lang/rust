use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::mir::{
    Local, Location, Place, PlaceBase, Statement, StatementKind, TerminatorKind
};

use rustc_data_structures::fx::FxHashSet;

use crate::borrow_check::MirBorrowckCtxt;

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
            visitor.visit_body(visitor.mbcx.body);
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
    fn remove_never_initialized_mut_locals(&mut self, into: &Place<'_>) {
        // Remove any locals that we found were initialized from the
        // `never_initialized_mut_locals` set. At the end, the only remaining locals will
        // be those that were never initialized - we will consider those as being used as
        // they will either have been removed by unreachable code optimizations; or linted
        // as unused variables.
        if let Some(local) = into.base_local() {
            let _ = self.never_initialized_mut_locals.remove(&local);
        }
    }
}

impl<'visit, 'cx, 'tcx> Visitor<'tcx> for GatherUsedMutsVisitor<'visit, 'cx, 'tcx> {
    fn visit_terminator_kind(
        &mut self,
        kind: &TerminatorKind<'tcx>,
        _location: Location,
    ) {
        debug!("visit_terminator_kind: kind={:?}", kind);
        match &kind {
            TerminatorKind::Call { destination: Some((into, _)), .. } => {
                self.remove_never_initialized_mut_locals(&into);
            },
            TerminatorKind::DropAndReplace { location, .. } => {
                self.remove_never_initialized_mut_locals(&location);
            },
            _ => {},
        }
    }

    fn visit_statement(
        &mut self,
        statement: &Statement<'tcx>,
        _location: Location,
    ) {
        match &statement.kind {
            StatementKind::Assign(into, _) => {
                if let Some(local) = into.base_local() {
                    debug!(
                        "visit_statement: statement={:?} local={:?} \
                         never_initialized_mut_locals={:?}",
                        statement, local, self.never_initialized_mut_locals
                    );
                }
                self.remove_never_initialized_mut_locals(into);
            },
            _ => {},
        }
    }

    fn visit_local(
        &mut self,
        local: &Local,
        place_context: PlaceContext,
        location: Location,
    ) {
        if place_context.is_place_assignment() && self.temporary_used_locals.contains(local) {
            // Propagate the Local assigned at this Location as a used mutable local variable
            for moi in &self.mbcx.move_data.loc_map[location] {
                let mpi = &self.mbcx.move_data.moves[*moi].path;
                let path = &self.mbcx.move_data.move_paths[*mpi];
                debug!(
                    "assignment of {:?} to {:?}, adding {:?} to used mutable set",
                    path.place, local, path.place
                );
                if let Place::Base(PlaceBase::Local(user_local)) = path.place {
                    self.mbcx.used_mut.insert(user_local);
                }
            }
        }
    }
}
