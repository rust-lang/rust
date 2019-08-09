use crate::ty::{self, Ty, TyCtxt, AdtSizedConstraint};
use crate::ty::util::NeedsDrop;

use syntax::symbol::InternedString;

pub(super) trait Value<'tcx>: Sized {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Self;
}

impl<'tcx, T> Value<'tcx> for T {
    default fn from_cycle_error(tcx: TyCtxt<'tcx>) -> T {
        tcx.sess.abort_if_errors();
        bug!("Value::from_cycle_error called without errors");
    }
}

impl<'tcx> Value<'tcx> for Ty<'tcx> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        tcx.types.err
    }
}

impl<'tcx> Value<'tcx> for ty::SymbolName {
    fn from_cycle_error(_: TyCtxt<'tcx>) -> Self {
        ty::SymbolName { name: InternedString::intern("<error>") }
    }
}

impl<'tcx> Value<'tcx> for NeedsDrop {
    fn from_cycle_error(_: TyCtxt<'tcx>) -> Self {
        NeedsDrop(false)
    }
}

impl<'tcx> Value<'tcx> for AdtSizedConstraint<'tcx> {
    fn from_cycle_error(tcx: TyCtxt<'tcx>) -> Self {
        AdtSizedConstraint(tcx.intern_type_list(&[tcx.types.err]))
    }
}
