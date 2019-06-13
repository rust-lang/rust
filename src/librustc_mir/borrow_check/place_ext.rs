use rustc::hir;
use rustc::mir::ProjectionElem;
use rustc::mir::{Body, Place, PlaceBase, Mutability, Static, StaticKind};
use rustc::ty::{self, TyCtxt};
use crate::borrow_check::borrow_set::LocalsStateAtExit;

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
        self.iterate(|place_base, place_projection| {
            let ignore = match place_base {
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
                                body.local_decls[*index].mutability == Mutability::Not;
                            debug!("ignore_borrow: local {:?} => {:?}", index, ignore);
                            ignore
                        }
                    }
                }
                PlaceBase::Static(box Static{ kind: StaticKind::Promoted(_), .. }) =>
                    false,
                PlaceBase::Static(box Static{ kind: StaticKind::Static(def_id), .. }) => {
                    tcx.is_mutable_static(*def_id)
                }
            };

            for proj in place_projection {
                if proj.elem == ProjectionElem::Deref {
                    let ty = proj.base.ty(body, tcx).ty;
                    match ty.sty {
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
                        ty::RawPtr(..) | ty::Ref(_, _, hir::MutImmutable) => return true,
                        _ => {}
                    }
                }
            }

            ignore
        })
    }
}
