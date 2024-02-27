// The compiler code necessary to support the compile_error! extension.

use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{get_single_str_from_tts, DummyResult, ExtCtxt, MacResult};
use rustc_span::Span;

pub fn expand_compile_error<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'cx> {
    let var = match get_single_str_from_tts(cx, sp, tts, "compile_error!") {
        Ok(var) => var,
        Err(guar) => return DummyResult::any(sp, guar),
    };

    #[expect(rustc::diagnostic_outside_of_impl, reason = "diagnostic message is specified by user")]
    #[expect(rustc::untranslatable_diagnostic, reason = "diagnostic message is specified by user")]
    let guar = cx.dcx().span_err(sp, var.to_string());

    DummyResult::any(sp, guar)
}
