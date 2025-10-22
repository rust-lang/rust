use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, IsSuggestable};
use rustc_span::symbol::sym;

use super::CLONE_ON_REF_PTR;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>) {
    let obj_ty = cx.typeck_results().expr_ty(receiver).peel_refs();

    if let ty::Adt(adt, subst) = obj_ty.kind()
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
                let snippet = snippet_with_context(cx, receiver.span, expr.span.ctxt(), "..", &mut app).0;
                let generic = subst.type_at(0);
                if generic.is_suggestable(cx.tcx, true) {
                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!("{caller_type}::<{generic}>::clone(&{snippet})"),
                        app,
                    );
                } else {
                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!("{caller_type}::</* generic */>::clone(&{snippet})"),
                        Applicability::HasPlaceholders,
                    );
                }
            },
        );
    }
}
