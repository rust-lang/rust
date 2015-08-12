use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::{BytePos, Span};
use utils::span_lint;

declare_lint!{ pub ZERO_WIDTH_SPACE, Deny, "Zero-width space is confusing" }

#[derive(Copy, Clone)]
pub struct Unicode;

impl LintPass for Unicode {
    fn get_lints(&self) -> LintArray {
        lint_array!(ZERO_WIDTH_SPACE)
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
    let mut start: Option<usize> = None;
    for (i, c) in string.char_indices() {
        if c == '\u{200B}' {
            if start.is_none() { start = Some(i); }
        } else {
            lint_zero_width(cx, span, start);
            start = None;
        }
    }
    lint_zero_width(cx, span, start);
}

fn lint_zero_width(cx: &Context, span: Span, start: Option<usize>) {
    start.map(|index| {
        span_lint(cx, ZERO_WIDTH_SPACE, Span {
            lo: span.lo + BytePos(index as u32),
            hi: span.lo + BytePos(index as u32),
            expn_id: span.expn_id,
        }, "zero-width space detected. Consider using `\\u{200B}`.")
    });
}
