// The compiler code necessary to support the compile_error! extension.

use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{self, *};
use rustc_span::Span;

pub fn expand_compile_error<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    let Some(var) = get_single_str_from_tts(cx, sp, tts, "compile_error!") else {
        return DummyResult::any(sp);
    };

    cx.span_err(sp, &var);

    DummyResult::any(sp)
}
