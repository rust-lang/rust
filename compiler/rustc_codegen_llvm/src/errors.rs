use std::borrow::Cow;

use rustc_errors::fluent;
use rustc_errors::DiagnosticBuilder;
use rustc_errors::ErrorGuaranteed;
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_session::SessionDiagnostic;
use rustc_span::Span;

pub(crate) enum UnknownCTargetFeature<'a> {
    UnknownFeaturePrefix { feature: &'a str },
    UnknownFeature { feature: &'a str, rust_feature: Option<&'a str> },
}

impl SessionDiagnostic<'_, ()> for UnknownCTargetFeature<'_> {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> DiagnosticBuilder<'_, ()> {
        match self {
            UnknownCTargetFeature::UnknownFeaturePrefix { feature } => {
                let mut diag = sess.struct_warn(fluent::codegen_llvm::unknown_ctarget_feature);
                diag.set_arg("feature", feature);
                diag.note(fluent::codegen_llvm::unknown_feature_prefix);
                diag
            }
            UnknownCTargetFeature::UnknownFeature { feature, rust_feature } => {
                let mut diag = sess.struct_warn(fluent::codegen_llvm::unknown_ctarget_feature);
                diag.set_arg("feature", feature);
                diag.note(fluent::codegen_llvm::unknown_feature);
                if let Some(rust_feature) = rust_feature {
                    diag.help(fluent::codegen_llvm::rust_feature);
                    diag.set_arg("rust_feature", rust_feature);
                } else {
                    diag.note(fluent::codegen_llvm::unknown_feature_fill_request);
                }
                diag
            }
        }
    }
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::error_creating_import_library)]
pub(crate) struct ErrorCreatingImportLibrary<'a> {
    pub lib_name: &'a str,
    pub error: String,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::instrument_coverage_requires_llvm_12)]
pub(crate) struct InstrumentCoverageRequiresLLVM12;

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::symbol_already_defined)]
pub(crate) struct SymbolAlreadyDefined<'a> {
    #[primary_span]
    pub span: Span,
    pub symbol_name: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::branch_protection_requires_aarch64)]
pub(crate) struct BranchProtectionRequiresAArch64;

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::layout_size_overflow)]
pub(crate) struct LayoutSizeOverflow {
    #[primary_span]
    pub span: Span,
    pub error: String,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::invalid_minimum_alignment)]
pub(crate) struct InvalidMinimumAlignment {
    pub err: String,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::linkage_const_or_mut_type)]
pub(crate) struct LinkageConstOrMutType {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::sanitizer_memtag_requires_mte)]
pub(crate) struct SanitizerMemtagRequiresMte;

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::archive_build_failure)]
pub(crate) struct ArchiveBuildFailure {
    pub error: std::io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::error_writing_def_file)]
pub(crate) struct ErrorWritingDEFFile {
    pub error: std::io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::error_calling_dlltool)]
pub(crate) struct ErrorCallingDllTool {
    pub error: std::io::Error,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::dlltool_fail_import_library)]
pub(crate) struct DlltoolFailImportLibrary<'a> {
    pub stdout: Cow<'a, str>,
    pub stderr: Cow<'a, str>,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_llvm::unknown_archive_kind)]
pub(crate) struct UnknownArchiveKind<'a> {
    pub kind: &'a str,
}

pub(crate) struct TargetFeatureDisableOrEnable<'a> {
    pub features: &'a [&'a str],
    pub span: Option<Span>,
}

#[derive(SessionSubdiagnostic)]
#[help(codegen_llvm::missing_features)]
pub(crate) struct MissingFeatures;

impl SessionDiagnostic<'_, ErrorGuaranteed> for TargetFeatureDisableOrEnable<'_> {
    fn into_diagnostic(
        self,
        sess: &'_ rustc_session::parse::ParseSess,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = if let Some(span) = self.span {
            let mut diag = sess.struct_err(fluent::codegen_llvm::target_feature_disable_or_enable);
            diag.set_span(span);
            diag
        } else {
            sess.struct_err(fluent::codegen_llvm::target_feature_disable_or_enable)
        };
        diag.set_arg("features", self.features.join(", "));
        diag
    }
}
