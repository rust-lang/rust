use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;

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

#[derive(Diagnostic)]
#[diag(codegen_gcc_unstable_ctarget_feature)]
#[note]
pub(crate) struct UnstableCTargetFeature<'a> {
    pub feature: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_forbidden_ctarget_feature)]
pub(crate) struct ForbiddenCTargetFeature<'a> {
    pub feature: &'a str,
    pub enabled: &'a str,
    pub reason: &'a str,
}

#[derive(Subdiagnostic)]
pub(crate) enum PossibleFeature<'a> {
    #[help(codegen_gcc_possible_feature)]
    Some { rust_feature: &'a str },
    #[help(codegen_gcc_consider_filing_feature_request)]
    None,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_unwinding_inline_asm)]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span,
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
