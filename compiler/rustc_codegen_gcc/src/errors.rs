use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::Diagnostic;
use rustc_middle::ty::Ty;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

struct ExitCode(Option<i32>);

impl IntoDiagnosticArg for ExitCode {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        let ExitCode(exit_code) = self;
        match exit_code {
            Some(t) => t.into_diagnostic_arg(),
            None => DiagnosticArgValue::Str(Cow::Borrowed("<signal>")),
        }
    }
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_monomorphization_expected_simd, code = "E0511")]
pub(crate) struct InvalidMonomorphizationExpectedSimd<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub position: &'a str,
    pub found_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_not_supported)]
pub(crate) struct LTONotSupported;

#[derive(Diagnostic)]
#[diag(codegen_gcc_unwinding_inline_asm)]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_invalid_minimum_alignment)]
pub(crate) struct InvalidMinimumAlignment {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_tied_target_features)]
#[help]
pub(crate) struct TiedTargetFeatures {
    #[primary_span]
    pub span: Span,
    pub features: String,
}
