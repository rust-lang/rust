use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;
use std::borrow::Cow;

struct ExitCode {
    pub exit_code: Option<i32>,
}

impl IntoDiagnosticArg for ExitCode {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        match self.exit_code {
            Some(t) => t.into_diagnostic_arg(),
            None => DiagnosticArgValue::Str(Cow::Borrowed("None")),
        }
    }
}

#[derive(SessionDiagnostic)]
#[diag(codegen_gcc::ranlib_failure)]
pub(crate) struct RanlibFailure {
    exit_code: ExitCode,
}

impl RanlibFailure {
    pub fn new(exit_code: Option<i32>) -> Self {
        let exit_code = ExitCode{ exit_code };
        RanlibFailure { exit_code }
    }
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
