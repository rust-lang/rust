use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{BindingMode, Pat, PatKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of the `ref` keyword.
    ///
    /// ### Why restrict this?
    /// The `ref` keyword can be confusing for people unfamiliar with it, and often
    /// it is more concise to use `&` instead.
    ///
    /// ### Example
    /// ```no_run
    /// let opt = Some(5);
    /// if let Some(ref foo) = opt {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// let opt = Some(5);
    /// if let Some(foo) = &opt {}
    /// ```
    #[clippy::version = "1.71.0"]
    pub REF_PATTERNS,
    restriction,
    "use of a ref pattern, e.g. Some(ref value)"
}
declare_lint_pass!(RefPatterns => [REF_PATTERNS]);

impl EarlyLintPass for RefPatterns {
    fn check_pat(&mut self, cx: &EarlyContext<'_>, pat: &Pat) {
        if let PatKind::Ident(BindingMode::REF, _, _) = pat.kind
            && !pat.span.from_expansion()
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(cx, REF_PATTERNS, pat.span, "usage of ref pattern", |diag| {
                diag.help("consider using `&` for clarity instead");
            });
        }
    }
}
