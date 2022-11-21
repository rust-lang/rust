use std::borrow::Cow;

use rustc_errors::fluent;
use rustc_errors::DiagnosticBuilder;
use rustc_errors::ErrorGuaranteed;
use rustc_errors::Handler;
use rustc_errors::IntoDiagnostic;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(codegen_llvm_unknown_ctarget_feature_prefix)]
#[note]
pub(crate) struct UnknownCTargetFeaturePrefix<'a> {
    pub feature: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_unknown_ctarget_feature)]
#[note]
pub(crate) struct UnknownCTargetFeature<'a> {
    pub feature: &'a str,
    #[subdiagnostic]
    pub rust_feature: PossibleFeature<'a>,
}

#[derive(Subdiagnostic)]
pub(crate) enum PossibleFeature<'a> {
    #[help(possible_feature)]
    Some { rust_feature: &'a str },
    #[help(consider_filing_feature_request)]
    None,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_error_creating_import_library)]
pub(crate) struct ErrorCreatingImportLibrary<'a> {
    pub lib_name: &'a str,
    pub error: String,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_instrument_coverage_requires_llvm_12)]
pub(crate) struct InstrumentCoverageRequiresLLVM12;

#[derive(Diagnostic)]
#[diag(codegen_llvm_symbol_already_defined)]
pub(crate) struct SymbolAlreadyDefined<'a> {
    #[primary_span]
    pub span: Span,
    pub symbol_name: &'a str,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_invalid_minimum_alignment)]
pub(crate) struct InvalidMinimumAlignment {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_linkage_const_or_mut_type)]
pub(crate) struct LinkageConstOrMutType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_sanitizer_memtag_requires_mte)]
pub(crate) struct SanitizerMemtagRequiresMte;

#[derive(Diagnostic)]
#[diag(codegen_llvm_error_writing_def_file)]
pub(crate) struct ErrorWritingDEFFile {
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_error_calling_dlltool)]
pub(crate) struct ErrorCallingDllTool {
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_dlltool_fail_import_library)]
pub(crate) struct DlltoolFailImportLibrary<'a> {
    pub stdout: Cow<'a, str>,
    pub stderr: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag(codegen_llvm_dynamic_linking_with_lto)]
#[note]
pub(crate) struct DynamicLinkingWithLTO;

#[derive(Diagnostic)]
#[diag(codegen_llvm_fail_parsing_target_machine_config_to_target_machine)]
pub(crate) struct FailParsingTargetMachineConfigToTargetMachine {
    pub error: String,
}

pub(crate) struct TargetFeatureDisableOrEnable<'a> {
    pub features: &'a [&'a str],
    pub span: Option<Span>,
    pub missing_features: Option<MissingFeatures>,
}

#[derive(Subdiagnostic)]
#[help(codegen_llvm_missing_features)]
pub(crate) struct MissingFeatures;

impl IntoDiagnostic<'_, ErrorGuaranteed> for TargetFeatureDisableOrEnable<'_> {
    fn into_diagnostic(self, sess: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = sess.struct_err(fluent::codegen_llvm_target_feature_disable_or_enable);
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
