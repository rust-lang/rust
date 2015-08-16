use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::{BytePos, Span};

use utils::span_lint;

declare_lint!{ pub ZERO_WIDTH_SPACE, Deny,
               "using a zero-width space in a string literal, which is confusing" }
declare_lint!{ pub NON_ASCII_LITERAL, Allow,
               "using any literal non-ASCII chars in a string literal; suggests \
                using the \\u escape instead" }

#[derive(Copy, Clone)]
pub struct Unicode;

impl LintPass for Unicode {
    fn get_lints(&self) -> LintArray {
        lint_array!(ZERO_WIDTH_SPACE, NON_ASCII_LITERAL)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprLit(ref lit) = expr.node {
            if let LitStr(ref string, _) = lit.node {
                check_str(cx, string, lit.span)
            }
        }
    }
}

fn check_str(cx: &Context, string: &str, span: Span) {
    for (i, c) in string.char_indices() {
        if c == '\u{200B}' {
            str_pos_lint(cx, ZERO_WIDTH_SPACE, span, i,
                         "zero-width space detected. Consider using `\\u{200B}`");
        }
        if c as u32 > 0x7F {
            str_pos_lint(cx, NON_ASCII_LITERAL, span, i, &format!(
                "literal non-ASCII character detected. Consider using `\\u{{{:X}}}`", c as u32));
        }
    }
}

fn str_pos_lint(cx: &Context, lint: &'static Lint, span: Span, index: usize, msg: &str) {
    span_lint(cx, lint, Span { lo: span.lo + BytePos((1 + index) as u32),
                               hi: span.lo + BytePos((1 + index) as u32),
                               expn_id: span.expn_id }, msg);

}
