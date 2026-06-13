use crate::methods::DRAIN_COLLECT;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet;
use clippy_utils::{is_range_full, std_or_core, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty;

pub(super) fn check(cx: &LateContext<'_>, arg: Option<&Expr<'_>>, expr: &Expr<'_>, recv: &Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(recv);
    let (is_ref, ty) = match *ty.kind() {
        ty::Ref(_, ty, _) => (true, ty),
        _ => (false, ty),
    };
    if cx.typeck_results().expr_ty(expr) == ty
        && let Some(did) = ty.opt_def_id()
        && (cx.tcx.lang_items().string() == Some(did)
            || matches!(
                ty.opt_diag_name(cx),
                Some(sym::HashMap | sym::HashSet | sym::BinaryHeap | sym::Vec | sym::VecDeque)
            ))
        && arg.is_none_or(|arg| {
            if let ExprKind::Path(QPath::Resolved(None, path)) = recv.kind {
                is_range_full(cx, arg, Some(path))
            } else {
                false
            }
        })
        && let Some(exec_context) = std_or_core(cx)
    {
        let recv = snippet(cx, recv.span, "<expr>");
        let sugg = if is_ref {
            format!("{exec_context}::mem::take({recv})")
        } else {
            format!("{exec_context}::mem::take(&mut {recv})")
        };

        span_lint_and_sugg(
            cx,
            DRAIN_COLLECT,
            expr.span,
            "draining all elements of a collection into a new collection of the same type",
            "use `mem::take` to avoid creating a new allocation",
            sugg,
            Applicability::MachineApplicable,
        );
    }
}
