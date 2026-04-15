use std::any::Any;

use rustc_data_structures::sync::DynSend;
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, Level};
use rustc_hir::lints::AttributeLintKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;

use crate::lints;

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
pub struct DecorateAttrLint<'a, 'sess, 'tcx> {
    pub sess: &'sess Session,
    pub tcx: Option<TyCtxt<'tcx>>,
    pub diagnostic: &'a AttributeLintKind,
}

impl<'a> Diagnostic<'a, ()> for DecorateAttrLint<'_, '_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, ()> {
        match self.diagnostic {
            &AttributeLintKind::UnexpectedCfgName(name, value) => {
                check_cfg::unexpected_cfg_name(self.sess, self.tcx, name, value)
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::UnexpectedCfgValue(name, value) => {
                check_cfg::unexpected_cfg_value(self.sess, self.tcx, name, value)
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::RenamedLint { name, replace, suggestion } => lints::RenamedLint {
                name,
                replace,
                suggestion: lints::RenamedLintSuggestion::WithSpan { suggestion, replace },
            }
            .into_diag(dcx, level),
            &AttributeLintKind::DeprecatedLintName { name, suggestion, replace } => {
                lints::DeprecatedLintName { name, suggestion, replace }.into_diag(dcx, level)
            }
            &AttributeLintKind::RemovedLint { name, ref reason } => {
                lints::RemovedLint { name, reason }.into_diag(dcx, level)
            }
            &AttributeLintKind::UnknownLint { name, span, suggestion } => lints::UnknownLint {
                name,
                suggestion: suggestion.map(|(replace, from_rustc)| {
                    lints::UnknownLintSuggestion::WithSpan { suggestion: span, replace, from_rustc }
                }),
            }
            .into_diag(dcx, level),
            &AttributeLintKind::IgnoredUnlessCrateSpecified { level: attr_level, name } => {
                lints::IgnoredUnlessCrateSpecified { level: attr_level, name }.into_diag(dcx, level)
            }
        }
    }
}
