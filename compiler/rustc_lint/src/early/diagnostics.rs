use std::any::Any;

use rustc_data_structures::sync::DynSend;
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, Level};
use rustc_hir::lints::AttributeLintKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;

mod check_cfg;

pub struct DiagAndSess<'sess> {
    pub callback: Box<
        dyn for<'b> FnOnce(DiagCtxtHandle<'b>, Level, &dyn Any) -> Diag<'b, ()> + DynSend + 'static,
    >,
    pub sess: &'sess Session,
}

impl<'a> Diagnostic<'a, ()> for DiagAndSess<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        (self.callback)(dcx, level, self.sess)
    }
}

/// This is a diagnostic struct that will decorate a `AttributeLintKind`
/// Directly creating the lint structs is expensive, using this will only decorate the lint structs when needed.
pub struct DecorateAttrLint<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub diagnostic: &'a AttributeLintKind,
}

impl<'a> Diagnostic<'a, ()> for DecorateAttrLint<'_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        match self.diagnostic {
            &AttributeLintKind::UnexpectedCfgName(name, value) => {
                check_cfg::unexpected_cfg_name(self.tcx, name, value).into_diag(dcx, level)
            }
            &AttributeLintKind::UnexpectedCfgValue(name, value) => {
                check_cfg::unexpected_cfg_value(self.tcx, name, value).into_diag(dcx, level)
            }
        }
    }
}
