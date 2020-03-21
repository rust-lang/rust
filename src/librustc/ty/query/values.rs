use crate::ty::{self, AdtSizedConstraint, Ty, TyCtxt};

use rustc_errors::ErrorProof;
use rustc_span::symbol::Symbol;

pub(super) trait Value<'tcx>: Sized {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, proof: ErrorProof) -> Self;
}

impl<'tcx, T> Value<'tcx> for T {
    default fn from_cycle_error(tcx: TyCtxt<'tcx>, _: ErrorProof) -> T {
        tcx.sess.abort_if_errors();
        bug!("Value::from_cycle_error called without errors");
    }
}

impl<'tcx> Value<'tcx> for Ty<'tcx> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, proof: ErrorProof) -> Ty<'tcx> {
        tcx.err(proof)
    }
}

impl<'tcx> Value<'tcx> for ty::SymbolName {
    fn from_cycle_error(_: TyCtxt<'tcx>, _: ErrorProof) -> Self {
        ty::SymbolName { name: Symbol::intern("<error>") }
    }
}

impl<'tcx> Value<'tcx> for AdtSizedConstraint<'tcx> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>, proof: ErrorProof) -> Self {
        AdtSizedConstraint(tcx.intern_type_list(&[tcx.err(proof)]))
    }
}
