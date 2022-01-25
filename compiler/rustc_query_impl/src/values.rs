use super::QueryCtxt;
use rustc_middle::ty::{self, AdtSizedConstraint, Ty};

pub(super) trait Value<'tcx>: Sized {
    fn from_cycle_error(tcx: QueryCtxt<'tcx>) -> Self;
}

impl<'tcx, T> Value<'tcx> for T {
    default fn from_cycle_error(tcx: QueryCtxt<'tcx>) -> T {
        tcx.sess.abort_if_errors();
        bug!("Value::from_cycle_error called without errors");
    }
}

impl<'tcx> Value<'tcx> for Ty<'_> {
    fn from_cycle_error(tcx: QueryCtxt<'tcx>) -> Self {
        // SAFETY: This is never called when `Self` is not `Ty<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe { std::mem::transmute::<Ty<'tcx>, Ty<'_>>(tcx.ty_error()) }
    }
}

impl<'tcx> Value<'tcx> for ty::SymbolName<'_> {
    fn from_cycle_error(tcx: QueryCtxt<'tcx>) -> Self {
        // SAFETY: This is never called when `Self` is not `SymbolName<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe {
            std::mem::transmute::<ty::SymbolName<'tcx>, ty::SymbolName<'_>>(ty::SymbolName::new(
                *tcx, "<error>",
            ))
        }
    }
}

impl<'tcx> Value<'tcx> for AdtSizedConstraint<'_> {
    fn from_cycle_error(tcx: QueryCtxt<'tcx>) -> Self {
        // SAFETY: This is never called when `Self` is not `AdtSizedConstraint<'tcx>`.
        // FIXME: Represent the above fact in the trait system somehow.
        unsafe {
            std::mem::transmute::<AdtSizedConstraint<'tcx>, AdtSizedConstraint<'_>>(
                AdtSizedConstraint(tcx.intern_type_list(&[tcx.ty_error()])),
            )
        }
    }
}
