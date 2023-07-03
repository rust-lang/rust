use crate::errors;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_expand::base::{self, ExtCtxt};
use rustc_span::symbol::kw;
use rustc_span::Span;

pub fn expand_trace_macros(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tt: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let mut cursor = tt.trees();
    let mut err = false;
    let value = match &cursor.next() {
        Some(TokenTree::Token(token, _)) if token.is_keyword(kw::True) => true,
        Some(TokenTree::Token(token, _)) if token.is_keyword(kw::False) => false,
        _ => {
            err = true;
            false
        }
    };
    err |= cursor.next().is_some();
    if err {
        cx.emit_err(errors::TraceMacros { span: sp });
    } else {
        cx.set_trace_macros(value);
    }

    base::DummyResult::any_valid(sp)
}
