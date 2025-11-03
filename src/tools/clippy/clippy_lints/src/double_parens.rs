use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{HasSession, SpanRangeExt, snippet_with_applicability, snippet_with_context};
use rustc_ast::ast::{Expr, ExprKind, MethodCall};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary double parentheses.
    ///
    /// ### Why is this bad?
    /// This makes code harder to read and might indicate a
    /// mistake.
    ///
    /// ### Example
    /// ```no_run
    /// fn simple_double_parens() -> i32 {
    ///     ((0))
    /// }
    ///
    /// # fn foo(bar: usize) {}
    /// foo((0));
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn simple_no_parens() -> i32 {
    ///     (0)
    /// }
    ///
    /// # fn foo(bar: usize) {}
    /// foo(0);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DOUBLE_PARENS,
    complexity,
    "Warn on unnecessary double parentheses"
}

declare_lint_pass!(DoubleParens => [DOUBLE_PARENS]);

impl EarlyLintPass for DoubleParens {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        match &expr.kind {
            // ((..))
            // ^^^^^^ expr
            //  ^^^^  inner
            ExprKind::Paren(inner) if matches!(inner.kind, ExprKind::Paren(_) | ExprKind::Tup(_)) => {
                if expr.span.eq_ctxt(inner.span)
                    && !expr.span.in_external_macro(cx.sess().source_map())
                    && check_source(cx, inner)
                {
                    // suggest removing the outer parens

                    let mut applicability = Applicability::MachineApplicable;
                    // We don't need to use `snippet_with_context` here, because:
                    // - if `inner`'s `ctxt` is from macro, we don't lint in the first place (see the check above)
                    // - otherwise, calling `snippet_with_applicability` on a not-from-macro span is fine
                    let sugg = snippet_with_applicability(cx.sess(), inner.span, "_", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        DOUBLE_PARENS,
                        expr.span,
                        "unnecessary parentheses",
                        "remove them",
                        sugg.to_string(),
                        applicability,
                    );
                }
            },

            // func((n))
            // ^^^^^^^^^ expr
            //      ^^^  arg
            //       ^   inner
            ExprKind::Call(_, args) | ExprKind::MethodCall(box MethodCall { args, .. })
                if let [arg] = &**args
                    && let ExprKind::Paren(inner) = &arg.kind =>
            {
                if expr.span.eq_ctxt(arg.span)
                    && !arg.span.in_external_macro(cx.sess().source_map())
                    && check_source(cx, arg)
                {
                    // suggest removing the inner parens

                    let mut applicability = Applicability::MachineApplicable;
                    let sugg = snippet_with_context(cx.sess(), inner.span, arg.span.ctxt(), "_", &mut applicability).0;
                    span_lint_and_sugg(
                        cx,
                        DOUBLE_PARENS,
                        arg.span,
                        "unnecessary parentheses",
                        "remove them",
                        sugg.to_string(),
                        applicability,
                    );
                }
            },
            _ => {},
        }
    }
}

/// Check that the span does indeed look like `(  (..)  )`
fn check_source(cx: &EarlyContext<'_>, inner: &Expr) -> bool {
    if let Some(sfr) = inner.span.get_source_range(cx)
        // this is the same as `SourceFileRange::as_str`, but doesn't apply the range right away, because
        // we're interested in the source code outside it
        && let Some(src) = sfr.sf.src.as_ref().map(|src| src.as_str())
        && let Some((start, outer_after_inner)) = src.split_at_checked(sfr.range.end)
        && let Some((outer_before_inner, inner)) = start.split_at_checked(sfr.range.start)
        && outer_before_inner.trim_end().ends_with('(')
        && inner.starts_with('(')
        && inner.ends_with(')')
        && outer_after_inner.trim_start().starts_with(')')
    {
        true
    } else {
        false
    }
}
