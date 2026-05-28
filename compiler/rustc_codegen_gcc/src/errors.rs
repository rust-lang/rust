use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag("GCC backend does not support unwinding from inline asm")]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("failed to copy bitcode to object file: {$err}")]
pub(crate) struct CopyBitcode {
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to get bitcode from object file for LTO ({$gcc_err})")]
pub(crate) struct LtoBitcodeFromRlib {
    pub gcc_err: String,
}

#[derive(Diagnostic)]
#[diag("explicit tail calls with the 'become' keyword are not implemented in the GCC backend")]
pub(crate) struct ExplicitTailCallsUnsupported;

#[derive(Diagnostic)]
#[diag("asm contains a NUL byte")]
pub(crate) struct NulBytesInAsm {
    #[primary_span]
    pub span: Span,
}
