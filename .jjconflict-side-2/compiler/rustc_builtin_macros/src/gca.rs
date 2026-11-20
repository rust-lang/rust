use rustc_ast::ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::Span;

use crate::util::get_single_expr_from_tts;

pub(crate) fn expand<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let ExpandResult::Ready(expr) = get_single_expr_from_tts(cx, span, tts, "gca!") else {
        return ExpandResult::Retry(());
    };
    let expr = match expr {
        Ok(expr) => expr,
        Err(err) => return ExpandResult::Ready(DummyResult::any(span, err)),
    };

    ExpandResult::Ready(Box::new(base::MacEager {
        expr: Some(cx.expr(span, ast::ExprKind::GcaMacro(expr.clone()))),
        ty: Some(cx.ty(span, ast::TyKind::GcaMacro(expr))),
        ..Default::default()
    }))
}
