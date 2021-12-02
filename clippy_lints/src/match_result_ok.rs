use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::method_chain_args;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary `ok()` in `while let`.
    ///
    /// ### Why is this bad?
    /// Calling `ok()` in `while let` is unnecessary, instead match
    /// on `Ok(pat)`
    ///
    /// ### Example
    /// ```ignore
    /// while let Some(value) = iter.next().ok() {
    ///     vec.push(value)
    /// }
    ///
    /// if let Some(valie) = iter.next().ok() {
    ///     vec.push(value)
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// while let Ok(value) = iter.next() {
    ///     vec.push(value)
    /// }
    ///
    /// if let Ok(value) = iter.next() {
    ///        vec.push(value)
    /// }
    /// ```
    #[clippy::version = "1.57.0"]
    pub MATCH_RESULT_OK,
    style,
    "usage of `ok()` in `let Some(pat)` statements is unnecessary, match on `Ok(pat)` instead"
}

declare_lint_pass!(MatchResultOk => [MATCH_RESULT_OK]);

impl<'tcx> LateLintPass<'tcx> for MatchResultOk {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (let_pat, let_expr, ifwhile) =
            if let Some(higher::IfLet { let_pat, let_expr, .. }) = higher::IfLet::hir(cx, expr) {
                (let_pat, let_expr, "if")
            } else if let Some(higher::WhileLet { let_pat, let_expr, .. }) = higher::WhileLet::hir(expr) {
                (let_pat, let_expr, "while")
            } else {
                return;
            };

        if_chain! {
            if let ExprKind::MethodCall(_, ok_span, [ref result_types_0, ..], _) = let_expr.kind; //check is expr.ok() has type Result<T,E>.ok(, _)
            if let PatKind::TupleStruct(QPath::Resolved(_, x), y, _)  = let_pat.kind; //get operation
            if method_chain_args(let_expr, &["ok"]).is_some(); //test to see if using ok() methoduse std::marker::Sized;
            if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(result_types_0), sym::Result);
            if rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_path(x, false)) == "Some";

            then {

                let mut applicability = Applicability::MachineApplicable;
                let some_expr_string = snippet_with_applicability(cx, y[0].span, "", &mut applicability);
                let trimmed_ok = snippet_with_applicability(cx, let_expr.span.until(ok_span), "", &mut applicability);
                let sugg = format!(
                    "{} let Ok({}) = {}",
                    ifwhile,
                    some_expr_string,
                    trimmed_ok.trim().trim_end_matches('.'),
                );
                span_lint_and_sugg(
                    cx,
                    MATCH_RESULT_OK,
                    expr.span.with_hi(let_expr.span.hi()),
                    "matching on `Some` with `ok()` is redundant",
                    &format!("consider matching on `Ok({})` and removing the call to `ok` instead", some_expr_string),
                    sugg,
                    applicability,
                );
            }
        }
    }
}
