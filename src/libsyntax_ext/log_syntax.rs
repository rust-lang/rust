use syntax::ext::base;
use syntax::feature_gate;
use syntax::print;
use syntax::tokenstream;
use syntax::symbol::sym;
use syntax_pos;

pub fn expand_syntax_ext<'cx>(cx: &'cx mut base::ExtCtxt<'_>,
                              sp: syntax_pos::Span,
                              tts: &[tokenstream::TokenTree])
                              -> Box<dyn base::MacResult + 'cx> {
    if !cx.ecfg.enable_log_syntax() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       sym::log_syntax,
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_LOG_SYNTAX);
    }

    println!("{}", print::pprust::tts_to_string(tts));

    // any so that `log_syntax` can be invoked as an expression and item.
    base::DummyResult::any_valid(sp)
}
