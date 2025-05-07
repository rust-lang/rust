use crate::methods::DRAIN_COLLECT;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_lang_item;
use clippy_utils::{is_range_full, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, Path, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::Ty;
use rustc_span::{Symbol, sym};

/// Checks if both types match the given diagnostic item, e.g.:
///
/// `vec![1,2].drain(..).collect::<Vec<_>>()`
///  ^^^^^^^^^                     ^^^^^^   true
/// `vec![1,2].drain(..).collect::<HashSet<_>>()`
///  ^^^^^^^^^                     ^^^^^^^^^^  false
fn types_match_diagnostic_item(cx: &LateContext<'_>, expr: Ty<'_>, recv: Ty<'_>, sym: Symbol) -> bool {
    if let Some(expr_adt) = expr.ty_adt_def()
        && let Some(recv_adt) = recv.ty_adt_def()
    {
        cx.tcx.is_diagnostic_item(sym, expr_adt.did()) && cx.tcx.is_diagnostic_item(sym, recv_adt.did())
    } else {
        false
    }
}

/// Checks `std::{vec::Vec, collections::VecDeque}`.
fn check_vec(cx: &LateContext<'_>, args: &[Expr<'_>], expr: Ty<'_>, recv: Ty<'_>, recv_path: &Path<'_>) -> bool {
    (types_match_diagnostic_item(cx, expr, recv, sym::Vec)
        || types_match_diagnostic_item(cx, expr, recv, sym::VecDeque))
        && matches!(args, [arg] if is_range_full(cx, arg, Some(recv_path)))
}

/// Checks `std::string::String`
fn check_string(cx: &LateContext<'_>, args: &[Expr<'_>], expr: Ty<'_>, recv: Ty<'_>, recv_path: &Path<'_>) -> bool {
    is_type_lang_item(cx, expr, LangItem::String)
        && is_type_lang_item(cx, recv, LangItem::String)
        && matches!(args, [arg] if is_range_full(cx, arg, Some(recv_path)))
}

/// Checks `std::collections::{HashSet, HashMap, BinaryHeap}`.
fn check_collections(cx: &LateContext<'_>, expr: Ty<'_>, recv: Ty<'_>) -> Option<&'static str> {
    types_match_diagnostic_item(cx, expr, recv, sym::HashSet)
        .then_some("HashSet")
        .or_else(|| types_match_diagnostic_item(cx, expr, recv, sym::HashMap).then_some("HashMap"))
        .or_else(|| types_match_diagnostic_item(cx, expr, recv, sym::BinaryHeap).then_some("BinaryHeap"))
}

pub(super) fn check(cx: &LateContext<'_>, args: &[Expr<'_>], expr: &Expr<'_>, recv: &Expr<'_>) {
    let expr_ty = cx.typeck_results().expr_ty(expr);
    let recv_ty = cx.typeck_results().expr_ty(recv);
    let recv_ty_no_refs = recv_ty.peel_refs();

    if let ExprKind::Path(QPath::Resolved(_, recv_path)) = recv.kind
        && let Some(typename) = check_vec(cx, args, expr_ty, recv_ty_no_refs, recv_path)
            .then_some("Vec")
            .or_else(|| check_string(cx, args, expr_ty, recv_ty_no_refs, recv_path).then_some("String"))
            .or_else(|| check_collections(cx, expr_ty, recv_ty_no_refs))
        && let Some(exec_context) = std_or_core(cx)
    {
        let recv = snippet(cx, recv.span, "<expr>");
        let sugg = if let ty::Ref(..) = recv_ty.kind() {
            format!("{exec_context}::mem::take({recv})")
        } else {
            format!("{exec_context}::mem::take(&mut {recv})")
        };

        span_lint_and_sugg(
            cx,
            DRAIN_COLLECT,
            expr.span,
            format!("you seem to be trying to move all elements into a new `{typename}`"),
            "consider using `mem::take`",
            sugg,
            Applicability::MachineApplicable,
        );
    }
}
