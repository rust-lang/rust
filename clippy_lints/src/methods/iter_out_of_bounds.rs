use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::higher::VecArgs;
use clippy_utils::{expr_or_init, is_trait_method};
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self};
use rustc_span::sym;

use super::ITER_OUT_OF_BOUNDS;

fn expr_as_u128(cx: &LateContext<'_>, e: &Expr<'_>) -> Option<u128> {
    if let ExprKind::Lit(lit) = expr_or_init(cx, e).kind
        && let LitKind::Int(n, _) = lit.node
    {
        Some(n.get())
    } else {
        None
    }
}

/// Attempts to extract the length out of an iterator expression.
fn get_iterator_length<'tcx>(cx: &LateContext<'tcx>, iter: &'tcx Expr<'tcx>) -> Option<u128> {
    let ty::Adt(adt, substs) = cx.typeck_results().expr_ty(iter).kind() else {
        return None;
    };

    match cx.tcx.get_diagnostic_name(adt.did()) {
        Some(sym::ArrayIntoIter) => {
            // For array::IntoIter<T, const N: usize>, the length is the second generic
            // parameter.
            substs.const_at(1).try_to_target_usize(cx.tcx).map(u128::from)
        },
        Some(sym::SliceIter) if let ExprKind::MethodCall(_, recv, ..) = iter.kind => {
            if let ty::Array(_, len) = cx.typeck_results().expr_ty(recv).peel_refs().kind() {
                // For slice::Iter<'_, T>, the receiver might be an array literal: [1,2,3].iter().skip(..)
                len.try_to_target_usize(cx.tcx).map(u128::from)
            } else if let Some(args) = VecArgs::hir(cx, expr_or_init(cx, recv)) {
                match args {
                    VecArgs::Vec(vec) => vec.len().try_into().ok(),
                    VecArgs::Repeat(_, len) => expr_as_u128(cx, len),
                }
            } else {
                None
            }
        },
        Some(sym::IterEmpty) => Some(0),
        Some(sym::IterOnce) => Some(1),
        _ => None,
    }
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    arg: &'tcx Expr<'tcx>,
    message: &'static str,
    note: &'static str,
) {
    if is_trait_method(cx, expr, sym::Iterator)
        && let Some(len) = get_iterator_length(cx, recv)
        && let Some(skipped) = expr_as_u128(cx, arg)
        && skipped > len
    {
        span_lint_and_note(cx, ITER_OUT_OF_BOUNDS, expr.span, message, None, note);
    }
}

pub(super) fn check_skip<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    arg: &'tcx Expr<'tcx>,
) {
    check(
        cx,
        expr,
        recv,
        arg,
        "this `.skip()` call skips more items than the iterator will produce",
        "this operation is useless and will create an empty iterator",
    );
}

pub(super) fn check_take<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    arg: &'tcx Expr<'tcx>,
) {
    check(
        cx,
        expr,
        recv,
        arg,
        "this `.take()` call takes more items than the iterator will produce",
        "this operation is useless and the returned iterator will simply yield the same items",
    );
}
