use clippy_utils::diagnostics::span_lint_and_note;
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// It checks for `str::bytes().count()` and suggests replacing it with
    /// `str::len()`.
    ///
    /// ### Why is this bad?
    /// `str::bytes().count()` is longer and may not be as performant as using
    /// `str::len()`.
    ///
    /// ### Example
    /// ```rust
    /// "hello".bytes().count();
    /// ```
    /// Use instead:
    /// ```rust
    /// "hello".len();
    /// ```
    #[clippy::version = "1.60.0"]
    pub BYTES_COUNT_TO_LEN,
    complexity,
    "Using bytest().count() when len() performs the same functionality"
}

declare_lint_pass!(BytesCountToLen => [BYTES_COUNT_TO_LEN]);

impl<'tcx> LateLintPass<'tcx> for BytesCountToLen {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            //check for method call called "count"
            if let hir::ExprKind::MethodCall(count_path, count_args, _) = &expr.kind;
            if count_path.ident.name == rustc_span::sym::count;
            if let [bytes_expr] = &**count_args;
            //check for method call called "bytes" that was linked to "count"
            if let hir::ExprKind::MethodCall(bytes_path, _, _) = &bytes_expr.kind;
            if bytes_path.ident.name.as_str() == "bytes";
            then {
                span_lint_and_note(
                    cx,
                    BYTES_COUNT_TO_LEN,
                    expr.span,
                    "using long and hard to read `.bytes().count()`",
                    None,
                    "`.len()` achieves same functionality"
                );
            }
        };
    }
}
