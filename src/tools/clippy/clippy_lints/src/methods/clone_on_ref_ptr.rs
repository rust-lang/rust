use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::{sym, Symbol};

use super::CLONE_ON_REF_PTR;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    method_name: Symbol,
    receiver: &hir::Expr<'_>,
    args: &[hir::Expr<'_>],
) {
    if !(args.is_empty() && method_name == sym::clone) {
        return;
    }
    let obj_ty = cx.typeck_results().expr_ty(receiver).peel_refs();

    if let ty::Adt(adt, subst) = obj_ty.kind()
        && let Some(name) = cx.tcx.get_diagnostic_name(adt.did())
    {
        let caller_type = match name {
            sym::Rc => "Rc",
            sym::Arc => "Arc",
            sym::RcWeak | sym::ArcWeak => "Weak",
            _ => return,
        };

        // Sometimes unnecessary ::<_> after Rc/Arc/Weak
        let mut app = Applicability::Unspecified;
        let snippet = snippet_with_context(cx, receiver.span, expr.span.ctxt(), "..", &mut app).0;

        span_lint_and_sugg(
            cx,
            CLONE_ON_REF_PTR,
            expr.span,
            "using `.clone()` on a ref-counted pointer",
            "try",
            format!("{caller_type}::<{}>::clone(&{snippet})", subst.type_at(0)),
            app,
        );
    }
}
