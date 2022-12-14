use clippy_utils::diagnostics::span_lint;
use clippy_utils::{match_function_call, paths};
use rustc_ast::{BorrowKind, LitKind};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `std::str::from_utf8_unchecked` with an invalid UTF-8 literal
    ///
    /// ### Why is this bad?
    /// Creating such a `str` would result in undefined behavior
    ///
    /// ### Example
    /// ```rust
    /// # #[allow(unused)]
    /// unsafe {
    ///     std::str::from_utf8_unchecked(b"cl\x82ippy");
    /// }
    /// ```
    #[clippy::version = "1.64.0"]
    pub INVALID_UTF8_IN_UNCHECKED,
    correctness,
    "using a non UTF-8 literal in `std::std::from_utf8_unchecked`"
}
declare_lint_pass!(InvalidUtf8InUnchecked => [INVALID_UTF8_IN_UNCHECKED]);

impl<'tcx> LateLintPass<'tcx> for InvalidUtf8InUnchecked {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some([arg]) = match_function_call(cx, expr, &paths::STR_FROM_UTF8_UNCHECKED) {
            match &arg.kind {
                ExprKind::Lit(Spanned { node: lit, .. }) => {
                    if let LitKind::ByteStr(bytes, _) = &lit
                        && std::str::from_utf8(bytes).is_err()
                    {
                        lint(cx, expr.span);
                    }
                },
                ExprKind::AddrOf(BorrowKind::Ref, _, Expr { kind: ExprKind::Array(args), .. }) => {
                    let elements = args.iter().map(|e|{
                        match &e.kind {
                            ExprKind::Lit(Spanned { node: lit, .. }) => match lit {
                                LitKind::Byte(b) => Some(*b),
                                #[allow(clippy::cast_possible_truncation)]
                                LitKind::Int(b, _) => Some(*b as u8),
                                _ => None
                            }
                            _ => None
                        }
                    }).collect::<Option<Vec<_>>>();

                    if let Some(elements) = elements
                        && std::str::from_utf8(&elements).is_err()
                    {
                        lint(cx, expr.span);
                    }
                }
                _ => {}
            }
        }
    }
}

fn lint(cx: &LateContext<'_>, span: Span) {
    span_lint(
        cx,
        INVALID_UTF8_IN_UNCHECKED,
        span,
        "non UTF-8 literal in `std::str::from_utf8_unchecked`",
    );
}
