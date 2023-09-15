use rustc_errors::{
    DiagnosticArgValue, DiagnosticBuilder, ErrorGuaranteed, Handler, IntoDiagnostic, IntoDiagnosticArg,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;
use std::borrow::Cow;

use crate::fluent_generated as fluent;

#[derive(Diagnostic)]
#[diag(codegen_gcc_unknown_ctarget_feature_prefix)]
#[note]
pub(crate) struct UnknownCTargetFeaturePrefix<'a> {
    pub feature: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_unknown_ctarget_feature)]
#[note]
pub(crate) struct UnknownCTargetFeature<'a> {
    pub feature: &'a str,
    #[subdiagnostic]
    pub rust_feature: PossibleFeature<'a>,
}

#[derive(Subdiagnostic)]
pub(crate) enum PossibleFeature<'a> {
    #[help(codegen_gcc_possible_feature)]
    Some { rust_feature: &'a str },
    #[help(codegen_gcc_consider_filing_feature_request)]
    None,
}

struct ExitCode(Option<i32>);

impl IntoDiagnosticArg for ExitCode {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        let ExitCode(exit_code) = self;
        match exit_code {
            Some(t) => t.into_diagnostic_arg(),
            None => DiagnosticArgValue::Str(Cow::Borrowed("<signal>")),
        }
    }
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_not_supported)]
pub(crate) struct LTONotSupported;

#[derive(Diagnostic)]
#[diag(codegen_gcc_unwinding_inline_asm)]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_minimum_alignment)]
pub(crate) struct InvalidMinimumAlignment {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_tied_target_features)]
#[help]
pub(crate) struct TiedTargetFeatures {
    #[primary_span]
    pub span: Span,
    pub features: String,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_copy_bitcode)]
pub(crate) struct CopyBitcode {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_dynamic_linking_with_lto)]
#[note]
pub(crate) struct DynamicLinkingWithLTO;

#[derive(Diagnostic)]
#[diag(codegen_gcc_load_bitcode)]
pub(crate) struct LoadBitcode {
    name: String,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_disallowed)]
pub(crate) struct LtoDisallowed;

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_dylib)]
pub(crate) struct LtoDylib;

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_bitcode_from_rlib)]
pub(crate) struct LtoBitcodeFromRlib {
    pub gcc_err: String,
}

pub(crate) struct TargetFeatureDisableOrEnable<'a> {
    pub features: &'a [&'a str],
    pub span: Option<Span>,
    pub missing_features: Option<MissingFeatures>,
}

#[derive(Subdiagnostic)]
#[help(codegen_gcc_missing_features)]
pub(crate) struct MissingFeatures;

impl IntoDiagnostic<'_, ErrorGuaranteed> for TargetFeatureDisableOrEnable<'_> {
    fn into_diagnostic(self, sess: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(fluent::codegen_gcc_target_feature_disable_or_enable);
        if let Some(span) = self.span {
            diag.set_span(span);
        };
        if let Some(missing_features) = self.missing_features {
            diag.subdiagnostic(missing_features);
        }
        diag.set_arg("features", self.features.join(", "));
        diag
    }
}
