use crate::{LateContext, LateLintPass};
use rustc_hir::{BinOp, BinOpKind, Expr, ExprKind};
use rustc_span::sym;

declare_lint! {
    pub SPAN_USE_EQ_CTXT,
    Warn, // is this the right level?
    "Use of `==` with `Span::ctxt` rather than `Span::eq_ctxt`"
}

declare_lint_pass!(SpanUseEqCtxt => [SPAN_USE_EQ_CTXT]);

impl<'tcx> LateLintPass<'tcx> for SpanUseEqCtxt {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if let ExprKind::Binary(BinOp { node: BinOpKind::Eq, .. }, lhs, rhs) = expr.kind {
            if is_span_ctxt_call(cx, lhs) && is_span_ctxt_call(cx, rhs) {
                todo!(); // emit lint
            }
        }
    }
}

fn is_span_ctxt_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match &expr.kind {
        ExprKind::MethodCall(..) => {
            // i gave a method a diagnostic item -- FIXME: switch to a diagnostic
            // item for the Span type and check:
            //   * method call path == "ctxt"
            //   * receiver type matches Span diag item
            // also FIXME(todo) remove old SpanCtxt diagnostic item
            cx.typeck_results()
                .type_dependent_def_id(expr.hir_id)
                .is_some_and(|did| cx.tcx.is_diagnostic_item(sym::SpanCtxt, did))
        }

        _ => false,
    }
}
