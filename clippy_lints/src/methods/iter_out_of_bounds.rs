use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::{is_trait_method, match_def_path, paths};
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self};
use rustc_span::sym;

use super::ITER_OUT_OF_BOUNDS;

/// Attempts to extract the length out of an iterator expression.
fn get_iterator_length<'tcx>(cx: &LateContext<'tcx>, iter: &'tcx Expr<'tcx>) -> Option<u128> {
    let iter_ty = cx.typeck_results().expr_ty(iter);

    if let ty::Adt(adt, substs) = iter_ty.kind() {
        let did = adt.did();

        if match_def_path(cx, did, &paths::ARRAY_INTO_ITER) {
            // For array::IntoIter<T, const N: usize>, the length is the second generic
            // parameter.
            substs
                .const_at(1)
                .try_eval_target_usize(cx.tcx, cx.param_env)
                .map(u128::from)
        } else if match_def_path(cx, did, &paths::SLICE_ITER)
            && let ExprKind::MethodCall(_, recv, ..) = iter.kind
            && let ExprKind::Array(array) = recv.peel_borrows().kind
        {
            // For slice::Iter<'_, T>, the receiver might be an array literal: [1,2,3].iter().skip(..)
            array.len().try_into().ok()
        } else if match_def_path(cx, did, &paths::ITER_EMPTY) {
            Some(0)
        } else if match_def_path(cx, did, &paths::ITER_ONCE) {
            Some(1)
        } else {
            None
        }
    } else {
        None
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
        && let ExprKind::Lit(lit) = arg.kind
        && let LitKind::Int(skip, _) = lit.node
        && skip > len
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
