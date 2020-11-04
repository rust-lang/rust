use crate::utils::{
    is_type_diagnostic_item, match_def_path, match_qpath, paths, snippet, snippet_with_macro_callsite,
    span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:** Checks for usages of `Err(x)?`.
    ///
    /// **Why is this bad?** The `?` operator is designed to allow calls that
    /// can fail to be easily chained. For example, `foo()?.bar()` or
    /// `foo(bar()?)`. Because `Err(x)?` can't be used that way (it will
    /// always return), it is more clear to write `return Err(x)`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo(fail: bool) -> Result<i32, String> {
    ///     if fail {
    ///       Err("failed")?;
    ///     }
    ///     Ok(0)
    /// }
    /// ```
    /// Could be written:
    ///
    /// ```rust
    /// fn foo(fail: bool) -> Result<i32, String> {
    ///     if fail {
    ///       return Err("failed".into());
    ///     }
    ///     Ok(0)
    /// }
    /// ```
    pub TRY_ERR,
    style,
    "return errors explicitly rather than hiding them behind a `?`"
}

declare_lint_pass!(TryErr => [TRY_ERR]);

impl<'tcx> LateLintPass<'tcx> for TryErr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Looks for a structure like this:
        // match ::std::ops::Try::into_result(Err(5)) {
        //     ::std::result::Result::Err(err) =>
        //         #[allow(unreachable_code)]
        //         return ::std::ops::Try::from_error(::std::convert::From::from(err)),
        //     ::std::result::Result::Ok(val) =>
        //         #[allow(unreachable_code)]
        //         val,
        // };
        if_chain! {
            if !in_external_macro(cx.tcx.sess, expr.span);
            if let ExprKind::Match(ref match_arg, _, MatchSource::TryDesugar) = expr.kind;
            if let ExprKind::Call(ref match_fun, ref try_args) = match_arg.kind;
            if let ExprKind::Path(ref match_fun_path) = match_fun.kind;
            if matches!(match_fun_path, QPath::LangItem(LangItem::TryIntoResult, _));
            if let Some(ref try_arg) = try_args.get(0);
            if let ExprKind::Call(ref err_fun, ref err_args) = try_arg.kind;
            if let Some(ref err_arg) = err_args.get(0);
            if let ExprKind::Path(ref err_fun_path) = err_fun.kind;
            if match_qpath(err_fun_path, &paths::RESULT_ERR);
            if let Some(return_ty) = find_return_type(cx, &expr.kind);
            then {
                let prefix;
                let suffix;
                let err_ty;

                if let Some(ty) = result_error_type(cx, return_ty) {
                    prefix = "Err(";
                    suffix = ")";
                    err_ty = ty;
                } else if let Some(ty) = poll_result_error_type(cx, return_ty) {
                    prefix = "Poll::Ready(Err(";
                    suffix = "))";
                    err_ty = ty;
                } else if let Some(ty) = poll_option_result_error_type(cx, return_ty) {
                    prefix = "Poll::Ready(Some(Err(";
                    suffix = ")))";
                    err_ty = ty;
                } else {
                    return;
                };

                let expr_err_ty = cx.typeck_results().expr_ty(err_arg);

                let origin_snippet = if err_arg.span.from_expansion() {
                    snippet_with_macro_callsite(cx, err_arg.span, "_")
                } else {
                    snippet(cx, err_arg.span, "_")
                };
                let suggestion = if err_ty == expr_err_ty {
                    format!("return {}{}{}", prefix, origin_snippet, suffix)
                } else {
                    format!("return {}{}.into(){}", prefix, origin_snippet, suffix)
                };

                span_lint_and_sugg(
                    cx,
                    TRY_ERR,
                    expr.span,
                    "returning an `Err(_)` with the `?` operator",
                    "try this",
                    suggestion,
                    Applicability::MachineApplicable
                );
            }
        }
    }
}

/// Finds function return type by examining return expressions in match arms.
fn find_return_type<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx ExprKind<'_>) -> Option<Ty<'tcx>> {
    if let ExprKind::Match(_, ref arms, MatchSource::TryDesugar) = expr {
        for arm in arms.iter() {
            if let ExprKind::Ret(Some(ref ret)) = arm.body.kind {
                return Some(cx.typeck_results().expr_ty(ret));
            }
        }
    }
    None
}

/// Extracts the error type from Result<T, E>.
fn result_error_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if_chain! {
        if let ty::Adt(_, subst) = ty.kind();
        if is_type_diagnostic_item(cx, ty, sym::result_type);
        let err_ty = subst.type_at(1);
        then {
            Some(err_ty)
        } else {
            None
        }
    }
}

/// Extracts the error type from Poll<Result<T, E>>.
fn poll_result_error_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if_chain! {
        if let ty::Adt(def, subst) = ty.kind();
        if match_def_path(cx, def.did, &paths::POLL);
        let ready_ty = subst.type_at(0);

        if let ty::Adt(ready_def, ready_subst) = ready_ty.kind();
        if cx.tcx.is_diagnostic_item(sym::result_type, ready_def.did);
        let err_ty = ready_subst.type_at(1);

        then {
            Some(err_ty)
        } else {
            None
        }
    }
}

/// Extracts the error type from Poll<Option<Result<T, E>>>.
fn poll_option_result_error_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if_chain! {
        if let ty::Adt(def, subst) = ty.kind();
        if match_def_path(cx, def.did, &paths::POLL);
        let ready_ty = subst.type_at(0);

        if let ty::Adt(ready_def, ready_subst) = ready_ty.kind();
        if cx.tcx.is_diagnostic_item(sym::option_type, ready_def.did);
        let some_ty = ready_subst.type_at(0);

        if let ty::Adt(some_def, some_subst) = some_ty.kind();
        if cx.tcx.is_diagnostic_item(sym::result_type, some_def.did);
        let err_ty = some_subst.type_at(1);

        then {
            Some(err_ty)
        } else {
            None
        }
    }
}
