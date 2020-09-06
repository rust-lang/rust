use crate::borrow_check::borrow_set::LocalsStateAtExit;
use rustc_hir as hir;
use rustc_middle::mir::ProjectionElem;
use rustc_middle::mir::{Body, Mutability, Place};
use rustc_middle::ty::{self, TyCtxt};

/// Extension methods for the `Place` type.
crate trait PlaceExt<'tcx> {
    /// Returns `true` if we can safely ignore borrows of this place.
    /// This is true whenever there is no action that the user can do
    /// to the place `self` that would invalidate the borrow. This is true
    /// for borrows of raw pointer dereferents as well as shared references.
    fn ignore_borrow(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        locals_state_at_exit: &LocalsStateAtExit,
    ) -> bool;
}

impl<'tcx> PlaceExt<'tcx> for Place<'tcx> {
    fn ignore_borrow(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        locals_state_at_exit: &LocalsStateAtExit,
    ) -> bool {
        // If a local variable is immutable, then we only need to track borrows to guard
        // against two kinds of errors:
        // * The variable being dropped while still borrowed (e.g., because the fn returns
        //   a reference to a local variable)
        // * The variable being moved while still borrowed
        //
        // In particular, the variable cannot be mutated -- the "access checks" will fail --
        // so we don't have to worry about mutation while borrowed.
        if let LocalsStateAtExit::SomeAreInvalidated { has_storage_dead_or_moved } =
            locals_state_at_exit
        {
            let ignore = !has_storage_dead_or_moved.contains(self.local)
                && body.local_decls[self.local].mutability == Mutability::Not;
            debug!("ignore_borrow: local {:?} => {:?}", self.local, ignore);
            if ignore {
                return true;
            }
        }

        for (i, elem) in self.projection.iter().enumerate() {
            let proj_base = &self.projection[..i];

            if elem == ProjectionElem::Deref {
                let ty = Place::ty_from(self.local, proj_base, body, tcx).ty;
                match ty.kind() {
                    ty::Ref(_, _, hir::Mutability::Not) if i == 0 => {
                        // For references to thread-local statics, we do need
                        // to track the borrow.
                        if body.local_decls[self.local].is_ref_to_thread_local() {
                            continue;
                        }
                        return true;
                    }
                    ty::RawPtr(..) | ty::Ref(_, _, hir::Mutability::Not) => {
                        // For both derefs of raw pointers and `&T`
                        // references, the original path is `Copy` and
                        // therefore not significant.  In particular,
                        // there is nothing the user can do to the
                        // original path that would invalidate the
                        // newly created reference -- and if there
                        // were, then the user could have copied the
                        // original path into a new variable and
                        // borrowed *that* one, leaving the original
                        // path unborrowed.
                        return true;
                    }
                    _ => {}
                }
            }
        }

        false
    }
}
