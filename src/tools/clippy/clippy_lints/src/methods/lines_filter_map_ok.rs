use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef, MaybeResPath, MaybeTypeckRes};
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Body, Closure, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::LINES_FILTER_MAP_OK;

pub(super) fn check_flatten(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, call_span: Span, msrv: Msrv) {
    if cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator)
        && cx
            .typeck_results()
            .expr_ty_adjusted(recv)
            .is_diag_item(cx, sym::IoLines)
        && msrv.meets(cx, msrvs::MAP_WHILE)
    {
        emit(cx, recv, "flatten", call_span);
    }
}

pub(super) fn check_filter_or_flat_map(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    method_name: &'static str,
    method_arg: &Expr<'_>,
    call_span: Span,
    msrv: Msrv,
) {
    if cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator)
        && cx
            .typeck_results()
            .expr_ty_adjusted(recv)
            .is_diag_item(cx, sym::IoLines)
        && match method_arg.kind {
            // Detect `Result::ok`
            ExprKind::Path(ref qpath) => cx
                .qpath_res(qpath, method_arg.hir_id)
                .is_diag_item(cx, sym::result_ok_method),
            // Detect `|x| x.ok()`
            ExprKind::Closure(&Closure { body, .. }) => {
                if let Body {
                    params: [param], value, ..
                } = cx.tcx.hir_body(body)
                    && let ExprKind::MethodCall(method, receiver, [], _) = value.kind
                {
                    method.ident.name == sym::ok
                        && receiver.res_local_id() == Some(param.pat.hir_id)
                        && cx.ty_based_def(*value).is_diag_item(cx, sym::result_ok_method)
                } else {
                    false
                }
            },
            _ => false,
        }
        && msrv.meets(cx, msrvs::MAP_WHILE)
    {
        emit(cx, recv, method_name, call_span);
    }
}

fn emit(cx: &LateContext<'_>, recv: &Expr<'_>, method_name: &'static str, call_span: Span) {
    span_lint_and_then(
        cx,
        LINES_FILTER_MAP_OK,
        call_span,
        format!("`{method_name}()` will run forever if the iterator repeatedly produces an `Err`"),
        |diag| {
            diag.span_note(
                recv.span,
                "this expression returning a `std::io::Lines` may produce \
                        an infinite number of `Err` in case of a read error",
            );
            diag.span_suggestion(
                call_span,
                "replace with",
                "map_while(Result::ok)",
                Applicability::MaybeIncorrect,
            );
        },
    );
}
