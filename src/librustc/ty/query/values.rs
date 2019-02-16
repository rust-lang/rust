use crate::ty::{self, Ty, TyCtxt, AdtSizedConstraint};
use crate::ty::util::NeedsDrop;

use syntax::symbol::Symbol;

pub(super) trait Value<'tcx>: Sized {
    fn from_cycle_error<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self;
}

impl<'tcx, T> Value<'tcx> for T {
    default fn from_cycle_error<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> T {
        tcx.sess.abort_if_errors();
        bug!("Value::from_cycle_error called without errors");
    }
}

impl<'tcx, T: Default> Value<'tcx> for T {
    default fn from_cycle_error<'a>(_: TyCtxt<'a, 'tcx, 'tcx>) -> T {
        T::default()
    }
}

impl<'tcx> Value<'tcx> for Ty<'tcx> {
    fn from_cycle_error<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        tcx.types.err
    }
}

impl<'tcx> Value<'tcx> for ty::SymbolName {
    fn from_cycle_error<'a>(_: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        ty::SymbolName { name: Symbol::intern("<error>").as_interned_str() }
    }
}

impl<'tcx> Value<'tcx> for NeedsDrop {
    fn from_cycle_error(_: TyCtxt<'_, 'tcx, 'tcx>) -> Self {
        NeedsDrop(false)
    }
}

impl<'tcx> Value<'tcx> for AdtSizedConstraint<'tcx> {
    fn from_cycle_error(tcx: TyCtxt<'_, 'tcx, 'tcx>) -> Self {
        AdtSizedConstraint(tcx.intern_type_list(&[tcx.types.err]))
    }
}
