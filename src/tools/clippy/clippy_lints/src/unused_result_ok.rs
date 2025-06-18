use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::sym;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `Result::ok()` without using the returned `Option`.
    ///
    /// ### Why is this bad?
    /// Using `Result::ok()` may look like the result is checked like `unwrap` or `expect` would do
    /// but it only silences the warning caused by `#[must_use]` on the `Result`.
    ///
    /// ### Example
    /// ```no_run
    /// # fn some_function() -> Result<(), ()> { Ok(()) }
    /// some_function().ok();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # fn some_function() -> Result<(), ()> { Ok(()) }
    /// let _ = some_function();
    /// ```
    #[clippy::version = "1.82.0"]
    pub UNUSED_RESULT_OK,
    restriction,
    "Use of `.ok()` to silence `Result`'s `#[must_use]` is misleading. Use `let _ =` instead."
}
declare_lint_pass!(UnusedResultOk => [UNUSED_RESULT_OK]);

impl LateLintPass<'_> for UnusedResultOk {
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        if let StmtKind::Semi(expr) = stmt.kind
            && let ExprKind::MethodCall(ok_path, recv, [], ..) = expr.kind //check is expr.ok() has type Result<T,E>.ok(, _)
            && ok_path.ident.name == sym::ok
            && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result)
            && !stmt.span.in_external_macro(cx.sess().source_map())
        {
            let ctxt = expr.span.ctxt();
            let mut applicability = Applicability::MaybeIncorrect;
            let snippet = snippet_with_context(cx, recv.span, ctxt, "", &mut applicability).0;
            let sugg = format!("let _ = {snippet}");
            span_lint_and_sugg(
                cx,
                UNUSED_RESULT_OK,
                expr.span,
                "ignoring a result with `.ok()` is misleading",
                "consider using `let _ =` and removing the call to `.ok()` instead",
                sugg,
                applicability,
            );
        }
    }
}
