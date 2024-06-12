// The compiler code necessary to support the compile_error! extension.

use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::Span;

use crate::util::get_single_str_from_tts;

pub(crate) fn expand_compile_error<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let ExpandResult::Ready(mac) = get_single_str_from_tts(cx, sp, tts, "compile_error!") else {
        return ExpandResult::Retry(());
    };
    let var = match mac {
        Ok(var) => var,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };

    #[expect(rustc::diagnostic_outside_of_impl, reason = "diagnostic message is specified by user")]
    #[expect(rustc::untranslatable_diagnostic, reason = "diagnostic message is specified by user")]
    let guar = cx.dcx().span_err(sp, var.to_string());

    ExpandResult::Ready(DummyResult::any(sp, guar))
}
