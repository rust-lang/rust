use rustc_ast::tokenstream::TokenStream;
use rustc_ast_pretty::pprust;
use rustc_expand::base;

pub fn expand_log_syntax<'cx>(
    _cx: &'cx mut base::ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    println!("{}", pprust::tts_to_string(tts));

    // any so that `log_syntax` can be invoked as an expression and item.
    base::DummyResult::any_valid(sp)
}
