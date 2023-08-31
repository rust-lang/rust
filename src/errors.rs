use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_macros::Diagnostic;
use rustc_span::Span;
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

#[derive(Diagnostic)]
#[diag(codegen_gcc_copy_bitcode)]
pub(crate) struct CopyBitcode {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_dynamic_linking_with_lto)]
#[note]
pub(crate) struct DynamicLinkingWithLTO;

#[derive(Diagnostic)]
#[diag(codegen_gcc_load_bitcode)]
pub(crate) struct LoadBitcode {
    name: String,
}

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_disallowed)]
pub(crate) struct LtoDisallowed;

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_dylib)]
pub(crate) struct LtoDylib;

#[derive(Diagnostic)]
#[diag(codegen_gcc_lto_bitcode_from_rlib)]
pub(crate) struct LtoBitcodeFromRlib {
    pub gcc_err: String,
}
