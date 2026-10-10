use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef as _, MaybeTypeckRes as _};
use clippy_utils::source::snippet_with_context;
use clippy_utils::{is_none_expr, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

use super::OPTION_ZIP_NONE;

fn emit_lint(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) {
    let recv_is_none = is_none_expr(cx, recv);
    let arg_is_none = is_none_expr(cx, arg);

    if !recv_is_none && !arg_is_none {
        return;
    }

    span_lint_and_then(
        cx,
        OPTION_ZIP_NONE,
        expr.span,
        "calling `.zip()` on an `Option` where one side is `None` always returns `None`",
        |diag| {
            let mut app = Applicability::MaybeIncorrect;
            let ctxt = expr.span.ctxt();
            let none_snippet = if recv_is_none {
                snippet_with_context(cx, recv.span, ctxt, "_", &mut app).0
            } else {
                snippet_with_context(cx, arg.span, ctxt, "_", &mut app).0
            };

            if let ExprKind::MethodCall(_, _, _, call_span) = expr.kind {
                if recv_is_none && !arg_is_none {
                    let arg_snip = snippet_with_context(cx, arg.span, ctxt, "_", &mut app).0;
                    diag.span_suggestion(
                        expr.span,
                        "if you meant to zip the contents of the `Option` with `None`, use `Option::map`",
                        format!("{arg_snip}.map(|n| ({none_snippet}, n))"),
                        app,
                    );
                } else if !recv_is_none && arg_is_none {
                    diag.span_suggestion(
                        call_span,
                        "if you meant to zip the contents of the `Option` with `None`, use `Option::map`",
                        format!("map(|n| (n, {none_snippet}))"),
                        app,
                    );
                }
            }
        },
    );
}

pub(super) fn check_call(cx: &LateContext<'_>, expr: &Expr<'_>, func: &Expr<'_>, args: &[Expr<'_>]) {
    if let [left, right] = args
        && let ExprKind::Path(ref qpath) = func.kind
        && let Some(def_id) = cx.qpath_res(qpath, func.hir_id).opt_def_id()
        && cx.tcx.item_name(def_id) == sym::zip
        && def_id.opt_parent(cx).opt_impl_ty(cx).is_some_and(|impl_ty| {
            impl_ty
                .instantiate_identity()
                .skip_norm_wip()
                .ty_adt_def()
                .is_some_and(|adt| cx.tcx.is_diagnostic_item(sym::Option, adt.did()))
        })
    {
        emit_lint(cx, expr, left, right);
    }
}

pub(super) fn check_method(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) {
    if cx
        .ty_based_def(expr)
        .opt_parent(cx)
        .opt_impl_ty(cx)
        .is_some_and(|impl_ty| {
            impl_ty
                .instantiate_identity()
                .skip_norm_wip()
                .ty_adt_def()
                .is_some_and(|adt| cx.tcx.is_diagnostic_item(sym::Option, adt.did()))
        })
    {
        emit_lint(cx, expr, recv, arg);
    }
}
