use rustc_expand::base;
use rustc_span;
use syntax::print;
use syntax::tokenstream::TokenStream;

pub fn expand_log_syntax<'cx>(
    _cx: &'cx mut base::ExtCtxt<'_>,
    sp: rustc_span::Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    println!("{}", print::pprust::tts_to_string(tts));

    // any so that `log_syntax` can be invoked as an expression and item.
    base::DummyResult::any_valid(sp)
}
