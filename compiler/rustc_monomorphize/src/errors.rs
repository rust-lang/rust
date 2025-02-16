use std::path::PathBuf;

use rustc_macros::{Diagnostic, LintDiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(monomorphize_recursion_limit)]
pub(crate) struct RecursionLimit {
    #[primary_span]
    pub span: Span,
    pub shrunk: String,
    #[note]
    pub def_span: Span,
    pub def_path_str: String,
    #[note(monomorphize_written_to_path)]
    pub was_written: bool,
    pub path: PathBuf,
}

#[derive(Diagnostic)]
#[diag(monomorphize_no_optimized_mir)]
pub(crate) struct NoOptimizedMir {
    #[note]
    pub span: Span,
    pub crate_name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(monomorphize_large_assignments)]
#[note]
pub(crate) struct LargeAssignmentsLint {
    #[label]
    pub span: Span,
    pub size: u64,
    pub limit: u64,
}

#[derive(Diagnostic)]
#[diag(monomorphize_symbol_already_defined)]
pub(crate) struct SymbolAlreadyDefined {
    #[primary_span]
    pub span: Option<Span>,
    pub symbol: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_couldnt_dump_mono_stats)]
pub(crate) struct CouldntDumpMonoStats {
    pub error: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_encountered_error_while_instantiating)]
pub(crate) struct EncounteredErrorWhileInstantiating {
    #[primary_span]
    pub span: Span,
    pub formatted_item: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_start_not_found)]
#[help]
pub(crate) struct StartNotFound;

#[derive(Diagnostic)]
#[diag(monomorphize_unknown_cgu_collection_mode)]
pub(crate) struct UnknownCguCollectionMode<'a> {
    pub mode: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(monomorphize_abi_error_disabled_vector_type_def)]
#[help]
pub(crate) struct AbiErrorDisabledVectorTypeDef<'a> {
    #[label]
    pub span: Span,
    pub required_feature: &'a str,
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(monomorphize_abi_error_disabled_vector_type_call)]
#[help]
pub(crate) struct AbiErrorDisabledVectorTypeCall<'a> {
    #[label]
    pub span: Span,
    pub required_feature: &'a str,
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(monomorphize_abi_error_unsupported_vector_type_def)]
pub(crate) struct AbiErrorUnsupportedVectorTypeDef<'a> {
    #[label]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(LintDiagnostic)]
#[diag(monomorphize_abi_error_unsupported_vector_type_call)]
pub(crate) struct AbiErrorUnsupportedVectorTypeCall<'a> {
    #[label]
    pub span: Span,
    pub ty: Ty<'a>,
}
