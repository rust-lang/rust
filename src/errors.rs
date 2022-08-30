use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::ranlib_failure)]
pub(crate) struct RanlibFailure {
    pub exit_code: String,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::layout_size_overflow)]
pub(crate) struct LayoutSizeOverflow {
    #[primary_span]
    pub span: Span,
    pub error: String,
}

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::linkage_const_or_mut_type)]
pub(crate) struct LinkageConstOrMutType {
    #[primary_span]
    pub span: Span
}

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::lto_not_supported)]
pub(crate) struct LTONotSupported {}

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::unwinding_inline_asm)]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span
}
