use rustc::hir;
use rustc::mir::ProjectionElem;
use rustc::mir::{Local, Mir, Place};
use rustc::ty::{self, TyCtxt};

/// Extension methods for the `Place` type.
crate trait PlaceExt<'tcx> {
    /// Returns true if we can safely ignore borrows of this place.
    /// This is true whenever there is no action that the user can do
    /// to the place `self` that would invalidate the borrow. This is true
    /// for borrows of raw pointer dereferents as well as shared references.
    fn ignore_borrow(&self, tcx: TyCtxt<'_, '_, 'tcx>, mir: &Mir<'tcx>) -> bool;

    /// If this is a place like `x.f.g`, returns the local
    /// `x`. Returns `None` if this is based in a static.
    fn root_local(&self) -> Option<Local>;
}

impl<'tcx> PlaceExt<'tcx> for Place<'tcx> {
    fn ignore_borrow(&self, tcx: TyCtxt<'_, '_, 'tcx>, mir: &Mir<'tcx>) -> bool {
        match self {
            Place::Promoted(_) |
            Place::Local(_) => false,
            Place::Static(static_) => {
                tcx.is_static(static_.def_id) == Some(hir::Mutability::MutMutable)
            }
            Place::Projection(proj) => match proj.elem {
                ProjectionElem::Field(..)
                | ProjectionElem::Downcast(..)
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Index(_) => proj.base.ignore_borrow(tcx, mir),

                ProjectionElem::Deref => {
                    let ty = proj.base.ty(mir, tcx).to_ty(tcx);
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
                        ty::RawPtr(..) | ty::Ref(_, _, hir::MutImmutable) => true,
                        _ => proj.base.ignore_borrow(tcx, mir),
                    }
                }
            },
        }
    }

    fn root_local(&self) -> Option<Local> {
        let mut p = self;
        loop {
            match p {
                Place::Projection(pi) => p = &pi.base,
                Place::Promoted(_) |
                Place::Static(_) => return None,
                Place::Local(l) => return Some(*l),
            }
        }
    }
}
