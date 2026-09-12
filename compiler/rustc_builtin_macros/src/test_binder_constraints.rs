use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrVec, VisibilityKind, ast, token};
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
            let guar = diag.emit();
            return ExpandResult::Ready(DummyResult::any(span, guar));
        }
    };
    if p.token != token::Eof {
        cx.dcx().emit_err(diagnostics::OnlyOneArgument { span: p.token.span, name });
    }
    let item = Box::new(ast::Item {
        attrs: AttrVec::default(),
        id: ast::DUMMY_NODE_ID,
        span,
        vis: ast::Visibility { kind: VisibilityKind::Inherited, span: span.shrink_to_lo() },
        kind: ast::ItemKind::TestBinderConstraints(item),
        tokens: None,
    });
    rustc_expand::base::ExpandResult::Ready(Box::new(MacEager {
        expr: None,
        items: Some(SmallVec::from_buf([item])),
        ty: None,
    }))
}
