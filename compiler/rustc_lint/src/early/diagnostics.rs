use std::any::Any;

use rustc_data_structures::sync::DynSend;
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, Level};
use rustc_hir::lints::{AttributeLintKind, FormatWarning};
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

            &AttributeLintKind::MalformedDoc => lints::MalformedDoc.into_diag(dcx, level),

            &AttributeLintKind::ExpectedNoArgs => lints::ExpectedNoArgs.into_diag(dcx, level),

            &AttributeLintKind::ExpectedNameValue => lints::ExpectedNameValue.into_diag(dcx, level),
            &AttributeLintKind::MalFormedDiagnosticAttribute { attribute, options, span } => {
                lints::MalFormedDiagnosticAttributeLint { attribute, options, span }
                    .into_diag(dcx, level)
            }
            AttributeLintKind::MalformedDiagnosticFormat { warning } => match warning {
                FormatWarning::PositionalArgument { .. } => {
                    lints::DisallowedPositionalArgument.into_diag(dcx, level)
                }
                FormatWarning::InvalidSpecifier { .. } => {
                    lints::InvalidFormatSpecifier.into_diag(dcx, level)
                }
                FormatWarning::DisallowedPlaceholder { .. } => {
                    lints::DisallowedPlaceholder.into_diag(dcx, level)
                }
            },
            AttributeLintKind::DiagnosticWrappedParserError { description, label, span } => {
                lints::WrappedParserError { description, label, span: *span }.into_diag(dcx, level)
            }
            &AttributeLintKind::IgnoredDiagnosticOption { option_name, first_span, later_span } => {
                lints::IgnoredDiagnosticOption { option_name, first_span, later_span }
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::MissingOptionsForDiagnosticAttribute { attribute, options } => {
                lints::MissingOptionsForDiagnosticAttribute { attribute, options }
                    .into_diag(dcx, level)
            }
            &AttributeLintKind::NonMetaItemDiagnosticAttribute => {
                lints::NonMetaItemDiagnosticAttribute.into_diag(dcx, level)
            }
        }
    }
}
