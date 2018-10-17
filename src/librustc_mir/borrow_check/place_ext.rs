use rustc::hir;
use rustc::mir::ProjectionElem;
use rustc::mir::{Mir, Place, PlaceBase, Mutability};
use rustc::mir::tcx::PlaceTy;
use rustc::ty::{self, TyCtxt};
use borrow_check::borrow_set::LocalsStateAtExit;

/// Extension methods for the `Place` type.
crate trait PlaceExt<'tcx> {
    /// Returns true if we can safely ignore borrows of this place.
    /// This is true whenever there is no action that the user can do
    /// to the place `self` that would invalidate the borrow. This is true
    /// for borrows of raw pointer dereferents as well as shared references.
    fn ignore_borrow(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir: &Mir<'tcx>,
        locals_state_at_exit: &LocalsStateAtExit,
        ) -> bool;
}

impl<'tcx> PlaceExt<'tcx> for Place<'tcx> {
    fn ignore_borrow(
        &self,
        tcx: TyCtxt<'_, '_, 'tcx>,
        mir: &Mir<'tcx>,
        locals_state_at_exit: &LocalsStateAtExit,
    ) -> bool {
        let neo_place = tcx.as_new_place(self);
        let mut is_unsafe_place = match &neo_place.base {
            // If a local variable is immutable, then we only need to track borrows to guard
            // against two kinds of errors:
            // * The variable being dropped while still borrowed (e.g., because the fn returns
            //   a reference to a local variable)
            // * The variable being moved while still borrowed
            //
            // In particular, the variable cannot be mutated -- the "access checks" will fail --
            // so we don't have to worry about mutation while borrowed.
            PlaceBase::Local(index) => {
                match locals_state_at_exit {
                    LocalsStateAtExit::AllAreInvalidated => false,
                    LocalsStateAtExit::SomeAreInvalidated { has_storage_dead_or_moved } => {
                        let ignore = !has_storage_dead_or_moved.contains(*index) &&
                            mir.local_decls[*index].mutability == Mutability::Not;
                        debug!("ignore_borrow: local {:?} => {:?}", index, ignore);
                        ignore
                    }
                }
            },
            PlaceBase::Promoted(_) => false,
            PlaceBase::Static(static_) => {
                tcx.is_static(static_.def_id) == Some(hir::Mutability::MutMutable)
            }
        };

        let mut base_ty = neo_place.base.ty(mir);
        for elem in neo_place.elems.iter() {
            if let ProjectionElem::Deref = elem {
                if let ty::RawPtr(..) | ty::Ref(_, _, hir::MutImmutable) = base_ty.sty {
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
                    is_unsafe_place = true;
                }
            }
            base_ty = PlaceTy::from(base_ty).projection_ty(tcx, elem).to_ty(tcx);
        }

        is_unsafe_place
    }
}
