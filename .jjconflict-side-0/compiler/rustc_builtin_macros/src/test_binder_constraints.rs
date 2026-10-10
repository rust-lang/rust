use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrVec, ast, token};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_span::Span;
use smallvec::SmallVec;

use crate::diagnostics;

pub(crate) fn expand<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let name = "test_binder_constraints!";
    let mut p = cx.new_parser_from_tts(tts);
    if p.token == token::Eof {
        cx.dcx().emit_err(diagnostics::OnlyOneArgument { span, name });
    };
    let item = match p.parse_test_binder_constraints() {
        Ok(expr) => expr,
        Err(diag) => {
            let guar = diag.emit_err();
            return ExpandResult::Ready(DummyResult::any(span, guar));
        }
    };
    if p.token != token::Eof {
        cx.dcx().emit_err(diagnostics::OnlyOneArgument { span: p.token.span, name });
    }
    let item = cx.item(span, AttrVec::default(), ast::ItemKind::TestBinderConstraints(item));
    ExpandResult::Ready(Box::new(MacEager {
        expr: None,
        items: Some(SmallVec::from_buf([item])),
        ty: None,
    }))
}
