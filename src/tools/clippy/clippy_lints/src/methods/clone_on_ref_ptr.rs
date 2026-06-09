use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::peel_and_count_ty_refs;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, IsSuggestable};
use rustc_span::symbol::sym;

use super::CLONE_ON_REF_PTR;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>) {
    let receiver_ty = cx.typeck_results().expr_ty(receiver);
    let (receiver_ty_peeled, n_refs, _) = peel_and_count_ty_refs(receiver_ty);

    if let ty::Adt(adt, subst) = receiver_ty_peeled.kind()
        && let Some(name) = cx.tcx.get_diagnostic_name(adt.did())
    {
        let caller_type = match name {
            sym::Rc => "std::rc::Rc",
            sym::Arc => "std::sync::Arc",
            sym::RcWeak => "std::rc::Weak",
            sym::ArcWeak => "std::sync::Weak",
            _ => return,
        };
        span_lint_and_then(
            cx,
            CLONE_ON_REF_PTR,
            expr.span,
            "using `.clone()` on a ref-counted pointer",
            |diag| {
                // Sometimes unnecessary ::<_> after Rc/Arc/Weak
                let mut app = Applicability::Unspecified;
                let mut sugg = Sugg::hir_with_context(cx, receiver, expr.span.ctxt(), "..", &mut app);
                if n_refs == 0 {
                    sugg = sugg.addr();
                }
                // References on argument position don't need to preserve parentheses
                // even if they were present in the original expression.
                sugg = sugg.strip_paren();
                let generic = subst.type_at(0);
                if generic.is_suggestable(cx.tcx, true) {
                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!("{caller_type}::<{generic}>::clone({sugg})"),
                        app,
                    );
                } else {
                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!("{caller_type}::</* generic */>::clone({sugg})"),
                        Applicability::HasPlaceholders,
                    );
                }
            },
        );
    }
}
