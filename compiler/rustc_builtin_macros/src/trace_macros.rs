use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::{Span, kw};

use crate::errors;

pub(crate) fn expand_trace_macros(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tt: TokenStream,
) -> MacroExpanderResult<'static> {
    let mut iter = tt.iter();
    let mut err = false;
    let value = match iter.next() {
        Some(TokenTree::Token(token, _)) if token.is_keyword(kw::True) => true,
        Some(TokenTree::Token(token, _)) if token.is_keyword(kw::False) => false,
        _ => {
            err = true;
            false
        }
    };
    err |= iter.next().is_some();
    if err {
        cx.dcx().emit_err(errors::TraceMacros { span: sp });
    } else {
        cx.set_trace_macros(value);
    }

    ExpandResult::Ready(DummyResult::any_valid(sp))
}
