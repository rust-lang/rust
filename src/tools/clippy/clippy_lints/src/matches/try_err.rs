use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::option_arg_ty;
use clippy_utils::{get_parent_expr, is_res_lang_ctor, path_res};
use rustc_errors::Applicability;
use rustc_hir::LangItem::ResultErr;
use rustc_hir::{Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::{hygiene, sym};

use super::TRY_ERR;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, scrutinee: &'tcx Expr<'_>) {
    // Looks for a structure like this:
    // match ::std::ops::Try::into_result(Err(5)) {
    //     ::std::result::Result::Err(err) =>
    //         #[allow(unreachable_code)]
    //         return ::std::ops::Try::from_error(::std::convert::From::from(err)),
    //     ::std::result::Result::Ok(val) =>
    //         #[allow(unreachable_code)]
    //         val,
    // };
    if let ExprKind::Call(match_fun, [try_arg]) = scrutinee.kind
        && let ExprKind::Path(ref match_fun_path) = match_fun.kind
        && matches!(match_fun_path, QPath::LangItem(LangItem::TryTraitBranch, ..))
        && let ExprKind::Call(err_fun, [err_arg]) = try_arg.kind
        && is_res_lang_ctor(cx, path_res(cx, err_fun), ResultErr)
        && let Some(return_ty) = find_return_type(cx, &expr.kind)
    {
        let (prefix, suffix, err_ty) = if let Some(ty) = result_error_type(cx, return_ty) {
            ("Err(", ")", ty)
        } else if let Some(ty) = poll_result_error_type(cx, return_ty) {
            ("Poll::Ready(Err(", "))", ty)
        } else if let Some(ty) = poll_option_result_error_type(cx, return_ty) {
            ("Poll::Ready(Some(Err(", ")))", ty)
        } else {
            return;
        };

        span_lint_and_then(
            cx,
            TRY_ERR,
            expr.span,
            "returning an `Err(_)` with the `?` operator",
            |diag| {
                let expr_err_ty = cx.typeck_results().expr_ty(err_arg);
                let span = hygiene::walk_chain(err_arg.span, try_arg.span.ctxt());
                let mut applicability = Applicability::MachineApplicable;
                let origin_snippet = snippet_with_applicability(cx, span, "_", &mut applicability);
                let ret_prefix = if get_parent_expr(cx, expr).is_some_and(|e| matches!(e.kind, ExprKind::Ret(_))) {
                    "" // already returns
                } else {
                    "return "
                };
                let suggestion = if err_ty == expr_err_ty {
                    format!("{ret_prefix}{prefix}{origin_snippet}{suffix}")
                } else {
                    format!("{ret_prefix}{prefix}{origin_snippet}.into(){suffix}")
                };
                diag.span_suggestion(expr.span, "try", suggestion, applicability);
            },
        );
    }
}

/// Finds function return type by examining return expressions in match arms.
fn find_return_type<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx ExprKind<'_>) -> Option<Ty<'tcx>> {
    if let ExprKind::Match(_, arms, MatchSource::TryDesugar(_)) = expr {
        for arm in *arms {
            if let ExprKind::Ret(Some(ret)) = arm.body.kind {
                return Some(cx.typeck_results().expr_ty(ret));
            }
        }
    }
    None
}

/// Extracts the error type from Result<T, E>.
fn result_error_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if let ty::Adt(def, subst) = ty.kind()
        && cx.tcx.is_diagnostic_item(sym::Result, def.did())
    {
        Some(subst.type_at(1))
    } else {
        None
    }
}

/// Extracts the error type from Poll<Result<T, E>>.
fn poll_result_error_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if let ty::Adt(def, subst) = ty.kind()
        && cx.tcx.lang_items().get(LangItem::Poll) == Some(def.did())
    {
        let ready_ty = subst.type_at(0);
        result_error_type(cx, ready_ty)
    } else {
        None
    }
}

/// Extracts the error type from Poll<Option<Result<T, E>>>.
fn poll_option_result_error_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if let ty::Adt(def, subst) = ty.kind()
        && cx.tcx.lang_items().get(LangItem::Poll) == Some(def.did())
        && let ready_ty = subst.type_at(0)
        && let Some(some_ty) = option_arg_ty(cx, ready_ty)
    {
        result_error_type(cx, some_ty)
    } else {
        None
    }
}
