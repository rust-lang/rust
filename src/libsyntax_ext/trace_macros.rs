use syntax::ext::base::{self, ExtCtxt};
use syntax::symbol::kw;
use syntax_pos::Span;
use syntax::tokenstream::TokenTree;

pub fn expand_trace_macros(cx: &mut ExtCtxt<'_>,
                           sp: Span,
                           tt: &[TokenTree])
                           -> Box<dyn base::MacResult + 'static> {
    match tt {
        [TokenTree::Token(token)] if token.is_keyword(kw::True) => {
            cx.set_trace_macros(true);
        }
        [TokenTree::Token(token)] if token.is_keyword(kw::False) => {
            cx.set_trace_macros(false);
        }
        _ => cx.span_err(sp, "trace_macros! accepts only `true` or `false`"),
    }

    base::DummyResult::any_valid(sp)
}
