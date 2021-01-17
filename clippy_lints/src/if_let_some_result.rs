use crate::utils::{is_type_diagnostic_item, method_chain_args, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, MatchSource, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:*** Checks for unnecessary `ok()` in if let.
    ///
    /// **Why is this bad?** Calling `ok()` in if let is unnecessary, instead match
    /// on `Ok(pat)`
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// for i in iter {
    ///     if let Some(value) = i.parse().ok() {
    ///         vec.push(value)
    ///     }
    /// }
    /// ```
    /// Could be written:
    ///
    /// ```ignore
    /// for i in iter {
    ///     if let Ok(value) = i.parse() {
    ///         vec.push(value)
    ///     }
    /// }
    /// ```
    pub IF_LET_SOME_RESULT,
    style,
    "usage of `ok()` in `if let Some(pat)` statements is unnecessary, match on `Ok(pat)` instead"
}

declare_lint_pass!(OkIfLet => [IF_LET_SOME_RESULT]);

impl<'tcx> LateLintPass<'tcx> for OkIfLet {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! { //begin checking variables
            if let ExprKind::Match(ref op, ref body, MatchSource::IfLetDesugar { .. }) = expr.kind; //test if expr is if let
            if let ExprKind::MethodCall(_, ok_span, ref result_types, _) = op.kind; //check is expr.ok() has type Result<T,E>.ok(, _)
            if let PatKind::TupleStruct(QPath::Resolved(_, ref x), ref y, _)  = body[0].pat.kind; //get operation
            if method_chain_args(op, &["ok"]).is_some(); //test to see if using ok() methoduse std::marker::Sized;
            if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&result_types[0]), sym::result_type);
            if rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_path(x, false)) == "Some";

            then {
                let mut applicability = Applicability::MachineApplicable;
                let some_expr_string = snippet_with_applicability(cx, y[0].span, "", &mut applicability);
                let trimmed_ok = snippet_with_applicability(cx, op.span.until(ok_span), "", &mut applicability);
                let sugg = format!(
                    "if let Ok({}) = {}",
                    some_expr_string,
                    trimmed_ok.trim().trim_end_matches('.'),
                );
                span_lint_and_sugg(
                    cx,
                    IF_LET_SOME_RESULT,
                    expr.span.with_hi(op.span.hi()),
                    "matching on `Some` with `ok()` is redundant",
                    &format!("consider matching on `Ok({})` and removing the call to `ok` instead", some_expr_string),
                    sugg,
                    applicability,
                );
            }
        }
    }
}
