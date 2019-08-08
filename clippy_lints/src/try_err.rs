use crate::utils::{in_macro_or_desugar, match_qpath, paths, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty::Ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::source_map::Span;

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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TryErr {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
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
            if let ExprKind::Match(ref match_arg, _, MatchSource::TryDesugar) = expr.node;
            if let ExprKind::Call(ref match_fun, ref try_args) = match_arg.node;
            if let ExprKind::Path(ref match_fun_path) = match_fun.node;
            if match_qpath(match_fun_path, &paths::TRY_INTO_RESULT);
            if let Some(ref try_arg) = try_args.get(0);
            if let ExprKind::Call(ref err_fun, ref err_args) = try_arg.node;
            if let Some(ref err_arg) = err_args.get(0);
            if let ExprKind::Path(ref err_fun_path) = err_fun.node;
            if match_qpath(err_fun_path, &paths::RESULT_ERR);
            if let Some(return_type) = find_err_return_type(cx, &expr.node);

            then {
                let err_type = cx.tables.expr_ty(err_arg);
                let span = if in_macro_or_desugar(err_arg.span) {
                    span_to_outer_expn(err_arg.span)
                } else {
                    err_arg.span
                };
                let suggestion = if err_type == return_type {
                    format!("return Err({})", snippet(cx, span, "_"))
                } else {
                    format!("return Err({}.into())", snippet(cx, span, "_"))
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

fn span_to_outer_expn(span: Span) -> Span {
    let mut span = span;
    while let Some(expr) = span.ctxt().outer_expn_info() {
        span = expr.call_site;
    }
    span
}

// In order to determine whether to suggest `.into()` or not, we need to find the error type the
// function returns. To do that, we look for the From::from call (see tree above), and capture
// its output type.
fn find_err_return_type<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx ExprKind) -> Option<Ty<'tcx>> {
    if let ExprKind::Match(_, ref arms, MatchSource::TryDesugar) = expr {
        arms.iter().find_map(|ty| find_err_return_type_arm(cx, ty))
    } else {
        None
    }
}

// Check for From::from in one of the match arms.
fn find_err_return_type_arm<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, arm: &'tcx Arm) -> Option<Ty<'tcx>> {
    if_chain! {
        if let ExprKind::Ret(Some(ref err_ret)) = arm.body.node;
        if let ExprKind::Call(ref from_error_path, ref from_error_args) = err_ret.node;
        if let ExprKind::Path(ref from_error_fn) = from_error_path.node;
        if match_qpath(from_error_fn, &paths::TRY_FROM_ERROR);
        if let Some(from_error_arg) = from_error_args.get(0);
        then {
            Some(cx.tables.expr_ty(from_error_arg))
        } else {
            None
        }
    }
}
