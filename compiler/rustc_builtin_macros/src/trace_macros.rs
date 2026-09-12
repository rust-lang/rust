use rustc_ast::tokenarena::{ArenaTokenStream, ArenaTokenTree};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::{Span, kw};

use crate::diagnostics;

pub(crate) fn expand_trace_macros(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tt: ArenaTokenStream,
) -> MacroExpanderResult<'static> {
    let mut iter = tt.iter_top_level_trees();
    let mut err = false;
    let value = match iter.next() {
        Some(ArenaTokenTree::Token(token, _)) if token.is_keyword(kw::True) => true,
        Some(ArenaTokenTree::Token(token, _)) if token.is_keyword(kw::False) => false,
        _ => {
            err = true;
            false
        }
    };
    err |= iter.next().is_some();
    if err {
        cx.dcx().emit_err(diagnostics::TraceMacros { span: sp });
    } else {
        cx.set_trace_macros(value);
    }

    ExpandResult::Ready(DummyResult::any_valid(sp))
}
