use syntax::ext::base::{self, ExtCtxt};
use syntax::symbol::kw;
use syntax_pos::Span;
use syntax::tokenstream::{TokenTree, TokenStream};

pub fn expand_trace_macros(cx: &mut ExtCtxt<'_>,
                           sp: Span,
                           tt: TokenStream)
                           -> Box<dyn base::MacResult + 'static> {
    let mut cursor = tt.into_trees();
    let mut err = false;
    let value = match &cursor.next() {
        Some(TokenTree::Token(token)) if token.is_keyword(kw::True) => true,
        Some(TokenTree::Token(token)) if token.is_keyword(kw::False) => false,
        _ => {
            err = true;
            false
        },
    };
    err |= cursor.next().is_some();
    if err {
        cx.span_err(sp, "trace_macros! accepts only `true` or `false`")
    } else {
        cx.set_trace_macros(value);
    }

    base::DummyResult::any_valid(sp)
}
