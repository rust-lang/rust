use std::borrow::Cow;

use rustc_errors::fluent;
use rustc_errors::DiagnosticBuilder;
use rustc_errors::ErrorGuaranteed;
use rustc_errors::Handler;
use rustc_errors::IntoDiagnostic;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;

pub(crate) enum UnknownCTargetFeature<'a> {
    UnknownFeaturePrefix { feature: &'a str },
    UnknownFeature { feature: &'a str, rust_feature: Option<&'a str> },
}

impl IntoDiagnostic<'_, ()> for UnknownCTargetFeature<'_> {
    fn into_diagnostic(self, sess: &'_ Handler) -> DiagnosticBuilder<'_, ()> {
        match self {
            UnknownCTargetFeature::UnknownFeaturePrefix { feature } => {
                let mut diag = sess.struct_warn(fluent::codegen_llvm_unknown_ctarget_feature);
                diag.set_arg("feature", feature);
                diag.note(fluent::codegen_llvm_unknown_feature_prefix);
                diag
            }
            UnknownCTargetFeature::UnknownFeature { feature, rust_feature } => {
                let mut diag = sess.struct_warn(fluent::codegen_llvm_unknown_ctarget_feature);
                diag.set_arg("feature", feature);
                diag.note(fluent::codegen_llvm_unknown_feature);
                if let Some(rust_feature) = rust_feature {
                    diag.help(fluent::codegen_llvm_rust_feature);
                    diag.set_arg("rust_feature", rust_feature);
                } else {
                    diag.note(fluent::codegen_llvm_unknown_feature_fill_request);
                }
                diag
            }
        }
    }
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
#[diag(codegen_llvm_branch_protection_requires_aarch64)]
pub(crate) struct BranchProtectionRequiresAArch64;

#[derive(Diagnostic)]
#[diag(codegen_llvm_layout_size_overflow)]
pub(crate) struct LayoutSizeOverflow {
    #[primary_span]
    pub span: Span,
    pub error: String,
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
#[diag(codegen_llvm_archive_build_failure)]
pub(crate) struct ArchiveBuildFailure {
    pub error: std::io::Error,
}

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
#[diag(codegen_llvm_unknown_archive_kind)]
pub(crate) struct UnknownArchiveKind<'a> {
    pub kind: &'a str,
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
}

#[derive(Subdiagnostic)]
#[help(codegen_llvm_missing_features)]
pub(crate) struct MissingFeatures;

impl IntoDiagnostic<'_, ErrorGuaranteed> for TargetFeatureDisableOrEnable<'_> {
    fn into_diagnostic(self, sess: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = if let Some(span) = self.span {
            let mut diag = sess.struct_err(fluent::codegen_llvm_target_feature_disable_or_enable);
            diag.set_span(span);
            diag
        } else {
            sess.struct_err(fluent::codegen_llvm_target_feature_disable_or_enable)
        };
        diag.set_arg("features", self.features.join(", "));
        diag
    }
}
