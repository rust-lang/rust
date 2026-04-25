use std::any::Any;

use rustc_data_structures::sync::DynSend;
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, Level};
use rustc_hir::lints::AttributeLintKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::{Session, SessionAndCrateName};
use rustc_span::Symbol;
use rustc_span::def_id::CrateNum;

pub struct DiagAndSess<'sess, 'tcx> {
    pub callback: Box<
        dyn for<'b> FnOnce(DiagCtxtHandle<'b>, Level, &dyn Any) -> Diag<'b, ()> + DynSend + 'static,
    >,
    pub sess: &'sess Session,
    pub tcx: Option<TyCtxt<'tcx>>,
}

impl<'a> Diagnostic<'a, ()> for DiagAndSess<'_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        let crate_name: Option<&dyn Fn(CrateNum) -> Symbol> = match self.tcx {
            Some(tcx) => Some(&move |crate_num| tcx.crate_name(crate_num)),
            None => None,
        };
        let sess = SessionAndCrateName { sess: self.sess, crate_name };
        // FIXME: remove this transmute call once lifetime coercion issue is fixed.
        let sess: SessionAndCrateName<'static> = unsafe { std::mem::transmute(sess) };
        (self.callback)(dcx, level, &sess)
    }
}

/// This is a diagnostic struct that will decorate a `AttributeLintKind`
/// Directly creating the lint structs is expensive, using this will only decorate the lint structs when needed.
pub struct DecorateAttrLint<'a, 'sess, 'tcx> {
    pub sess: &'sess Session,
    pub tcx: Option<TyCtxt<'tcx>>,
    pub diagnostic: &'a AttributeLintKind,
}

impl<'a> Diagnostic<'a, ()> for DecorateAttrLint<'_, '_, '_> {
    fn into_diag(self, _dcx: DiagCtxtHandle<'a>, _level: Level) -> Diag<'a, ()> {
        panic!("should never be called")
    }
}
