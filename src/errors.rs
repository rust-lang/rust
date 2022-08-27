use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::ranlib_failure)]
pub(crate) struct RanlibFailure {
    pub exit_code: Option<i32>
}

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::linkage_const_or_mut_type)]
pub(crate) struct LinkageConstOrMutType {
    #[primary_span]
    pub span: Span
}
