use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::{BorrowKind, Expr, ExprKind, Mutability};
use rustc_ast::token::{Lit, LitKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for hard to read slices of byte characters, that could be more easily expressed as a
    /// byte string.
    ///
    /// ### Why is this bad?
    ///
    /// Potentially makes the string harder to read.
    ///
    /// ### Example
    /// ```ignore
    /// &[b'H', b'e', b'l', b'l', b'o'];
    /// ```
    /// Use instead:
    /// ```ignore
    /// b"Hello"
    /// ```
    #[clippy::version = "1.81.0"]
    pub BYTE_CHAR_SLICES,
    style,
    "hard to read byte char slice"
}
declare_lint_pass!(ByteCharSlice => [BYTE_CHAR_SLICES]);

impl EarlyLintPass for ByteCharSlice {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let Some(slice) = is_byte_char_slices(expr)
            && !expr.span.from_expansion()
        {
            span_lint_and_sugg(
                cx,
                BYTE_CHAR_SLICES,
                expr.span,
                "can be more succinctly written as a byte str",
                "try",
                format!("b\"{slice}\""),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn is_byte_char_slices(expr: &Expr) -> Option<String> {
    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, expr) = &expr.kind {
        match &expr.kind {
            ExprKind::Array(members) => {
                if members.is_empty() {
                    return None;
                }

                members
                    .iter()
                    .map(|member| match &member.kind {
                        ExprKind::Lit(Lit {
                            kind: LitKind::Byte,
                            symbol,
                            ..
                        }) => Some(symbol.as_str()),
                        _ => None,
                    })
                    .map(|maybe_quote| match maybe_quote {
                        Some("\"") => Some("\\\""),
                        Some("\\'") => Some("'"),
                        other => other,
                    })
                    .collect::<Option<String>>()
            },
            _ => None,
        }
    } else {
        None
    }
}
