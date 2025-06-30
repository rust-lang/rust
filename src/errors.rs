use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(codegen_gcc_unwinding_inline_asm)]
pub(crate) struct UnwindingInlineAsm {
    #[primary_span]
    pub span: Span,
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
