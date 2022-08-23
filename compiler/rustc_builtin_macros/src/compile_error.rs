// The compiler code necessary to support the compile_error! extension.

use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{self, *};
use rustc_span::Span;

use crate::errors::CompileError;

pub fn expand_compile_error<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    let Some(var) = get_single_str_from_tts(cx, sp, tts, "compile_error!") else {
        return DummyResult::any(sp);
    };

    cx.emit_err(CompileError { span: sp, msg: var.as_str().to_owned() });

    DummyResult::any(sp)
}
