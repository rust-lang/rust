use rustc_ast::tokenstream::TokenStream;
use rustc_ast_pretty::pprust;
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};

pub(crate) fn expand_log_syntax<'cx>(
    _cx: &'cx mut ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    println!("{}", pprust::tts_to_string(&tts));

    // any so that `log_syntax` can be invoked as an expression and item.
    ExpandResult::Ready(DummyResult::any_valid(sp))
}
