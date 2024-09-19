use rustc_ast::ast::{Expr, ExprKind};
use rustc_ast::token::LitKind;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_parse_format::{ParseMode, Parser, Piece};
use rustc_session::declare_lint_pass;
use rustc_span::BytePos;

use clippy_utils::diagnostics::span_lint;

declare_clippy_lint! {
    /// ### What it does
    /// Checks if string literals have formatting arguments outside of macros
    /// using them (like `format!`).
    ///
    /// ### Why is this bad?
    /// It will likely not generate the expected content.
    ///
    /// ### Example
    /// ```no_run
    /// let x: Option<usize> = None;
    /// let y = "hello";
    /// x.expect("{y:?}");
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: Option<usize> = None;
    /// let y = "hello";
    /// x.expect(&format!("{y:?}"));
    /// ```
    #[clippy::version = "1.83.0"]
    pub LITERAL_STRING_WITH_FORMATTING_ARG,
    suspicious,
    "Checks if string literals have formatting arguments"
}

declare_lint_pass!(LiteralStringWithFormattingArg => [LITERAL_STRING_WITH_FORMATTING_ARG]);

impl EarlyLintPass for LiteralStringWithFormattingArg {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Lit(lit) = expr.kind {
            let add = match lit.kind {
                LitKind::Str => 1,
                LitKind::StrRaw(nb) => nb as usize + 2,
                _ => return,
            };
            let fmt_str = lit.symbol.as_str();
            let lo = expr.span.lo();
            let mut current = fmt_str;
            let mut diff_len = 0;

            let mut parser = Parser::new(current, None, None, false, ParseMode::Format);
            let mut spans = Vec::new();
            while let Some(piece) = parser.next() {
                if let Some(error) = parser.errors.last() {
                    // We simply ignore the errors and move after them.
                    if error.span.end >= current.len() {
                        break;
                    }
                    current = &current[error.span.end + 1..];
                    diff_len = fmt_str.len() - current.len();
                    parser = Parser::new(current, None, None, false, ParseMode::Format);
                } else if let Piece::NextArgument(arg) = piece {
                    let mut pos = arg.position_span;
                    pos.start += diff_len;
                    pos.end += diff_len;

                    let start = fmt_str[..pos.start].rfind('{').unwrap_or(pos.start);
                    // If this is a unicode character escape, we don't want to lint.
                    if start > 1 && fmt_str[..start].ends_with("\\u") {
                        continue;
                    }

                    if fmt_str[start + 1..].trim_start().starts_with('}') {
                        // For now, we ignore `{}`.
                        continue;
                    }

                    let end = fmt_str[start + 1..]
                        .find('}')
                        .map_or(pos.end, |found| start + 1 + found)
                        + 1;
                    spans.push(
                        expr.span
                            .with_hi(lo + BytePos((start + add).try_into().unwrap()))
                            .with_lo(lo + BytePos((end + add).try_into().unwrap())),
                    );
                }
            }
            match spans.len() {
                0 => {},
                1 => {
                    span_lint(
                        cx,
                        LITERAL_STRING_WITH_FORMATTING_ARG,
                        spans,
                        "this looks like a formatting argument but it is not part of a formatting macro",
                    );
                },
                _ => {
                    span_lint(
                        cx,
                        LITERAL_STRING_WITH_FORMATTING_ARG,
                        spans,
                        "these look like formatting arguments but are not part of a formatting macro",
                    );
                },
            }
        }
    }
}
