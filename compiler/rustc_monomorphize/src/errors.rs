use rustc_macros::{Diagnostic, LintDiagnostic};
use rustc_middle::ty::{Instance, Ty};
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(monomorphize_recursion_limit)]
pub(crate) struct RecursionLimit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub instance: Instance<'tcx>,
    #[note]
    pub def_span: Span,
    pub def_path_str: String,
}

#[derive(Diagnostic)]
#[diag(monomorphize_no_optimized_mir)]
pub(crate) struct NoOptimizedMir {
    #[note]
    pub span: Span,
    pub crate_name: Symbol,
    pub instance: String,
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
pub(crate) struct EncounteredErrorWhileInstantiating<'tcx> {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
    pub instance: Instance<'tcx>,
}

#[derive(Diagnostic)]
#[diag(monomorphize_encountered_error_while_instantiating_global_asm)]
pub(crate) struct EncounteredErrorWhileInstantiatingGlobalAsm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(monomorphize_start_not_found)]
#[help]
pub(crate) struct StartNotFound;

#[derive(Diagnostic)]
#[diag(monomorphize_abi_error_disabled_vector_type)]
#[help]
pub(crate) struct AbiErrorDisabledVectorType<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub required_feature: &'a str,
    pub ty: Ty<'a>,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
}

#[derive(Diagnostic)]
#[diag(monomorphize_abi_error_unsupported_vector_type)]
pub(crate) struct AbiErrorUnsupportedVectorType<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub ty: Ty<'a>,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
}

#[derive(Diagnostic)]
#[diag(monomorphize_abi_required_target_feature)]
#[help]
pub(crate) struct AbiRequiredTargetFeature<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub required_feature: &'a str,
    pub abi: &'a str,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
}
