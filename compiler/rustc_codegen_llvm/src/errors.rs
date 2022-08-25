use rustc_errors::fluent;
use rustc_errors::DiagnosticBuilder;
use rustc_macros::SessionDiagnostic;
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
