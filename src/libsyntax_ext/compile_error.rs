// The compiler code necessary to support the compile_error! extension.

use syntax_expand::base::{self, *};
use syntax_pos::Span;
use syntax::tokenstream::TokenStream;

pub fn expand_compile_error<'cx>(cx: &'cx mut ExtCtxt<'_>,
                              sp: Span,
                              tts: TokenStream)
                              -> Box<dyn base::MacResult + 'cx> {
    let var = match get_single_str_from_tts(cx, sp, tts, "compile_error!") {
        None => return DummyResult::any(sp),
        Some(v) => v,
    };

    cx.span_err(sp, &var);

    DummyResult::any(sp)
}
