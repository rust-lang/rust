use syntax::ext::base;
use syntax::print;
use syntax::tokenstream;
use syntax_pos;

pub fn expand_syntax_ext<'cx>(_cx: &'cx mut base::ExtCtxt<'_>,
                              sp: syntax_pos::Span,
                              tts: &[tokenstream::TokenTree])
                              -> Box<dyn base::MacResult + 'cx> {
    println!("{}", print::pprust::tts_to_string(tts));

    // any so that `log_syntax` can be invoked as an expression and item.
    base::DummyResult::any_valid(sp)
}
