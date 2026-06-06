use crate::methods::utils::derefs_to_slice;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef;
use clippy_utils::ty::get_iterator_item_ty;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Symbol, sym};

use super::ITER_CLONED_COLLECT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, method_name: Symbol, expr: &Expr<'_>, recv: &'tcx Expr<'_>) {
    let expr_ty = cx.typeck_results().expr_ty(expr);
    if let ty::Adt(def, args) = expr_ty.kind()
        && def.is_diag_item(cx, sym::Vec)
        && let recv_ty = cx.typeck_results().expr_ty(recv)
        && let Some(slice) = derefs_to_slice(cx, recv, recv_ty)
        && let Some(iter_item_ty) = get_iterator_item_ty(cx, recv_ty)
        && let ty::Ref(_, iter_item_ty, _) = iter_item_ty.kind()
        && *iter_item_ty == args.type_at(0)
        && let Some(to_replace) = expr.span.trim_start(slice.span.source_callsite())
    {
        span_lint_and_sugg(
            cx,
            ITER_CLONED_COLLECT,
            to_replace,
            format!(
                "called `iter().{method_name}().collect()` on a slice to create a `Vec`. Calling `to_vec()` is both faster and \
            more readable"
            ),
            "try",
            ".to_vec()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
