use ast::util::unicode::{TEXT_FLOW_CONTROL_CHARS, contains_text_flow_control_chars};
use rustc_ast as ast;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{BytePos, Span, Symbol};

use crate::lints::{
    HiddenUnicodeCodepointsDiag, HiddenUnicodeCodepointsDiagLabels, HiddenUnicodeCodepointsDiagSub,
};
use crate::{EarlyContext, EarlyLintPass, LintContext};

declare_lint! {
    #[allow(text_direction_codepoint_in_literal)]
    /// The `text_direction_codepoint_in_literal` lint detects Unicode codepoints that change the
    /// visual representation of text on screen in a way that does not correspond to their on
    /// memory representation.
    ///
    /// ### Explanation
    ///
    /// The unicode characters `\u{202A}`, `\u{202B}`, `\u{202D}`, `\u{202E}`, `\u{2066}`,
    /// `\u{2067}`, `\u{2068}`, `\u{202C}` and `\u{2069}` make the flow of text on screen change
    /// its direction on software that supports these codepoints. This makes the text "abc" display
    /// as "cba" on screen. By leveraging software that supports these, people can write specially
    /// crafted literals that make the surrounding code seem like it's performing one action, when
    /// in reality it is performing another. Because of this, we proactively lint against their
    /// presence to avoid surprises.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(text_direction_codepoint_in_literal)]
    /// fn main() {
    ///     println!("{:?}", '‮');
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    pub TEXT_DIRECTION_CODEPOINT_IN_LITERAL,
    Deny,
    "detect special Unicode codepoints that affect the visual representation of text on screen, \
     changing the direction in which text flows",
}

declare_lint_pass!(HiddenUnicodeCodepoints => [TEXT_DIRECTION_CODEPOINT_IN_LITERAL]);

impl HiddenUnicodeCodepoints {
    fn lint_text_direction_codepoint(
        &self,
        cx: &EarlyContext<'_>,
        text: Symbol,
        span: Span,
        padding: u32,
        point_at_inner_spans: bool,
        label: &str,
    ) {
        // Obtain the `Span`s for each of the forbidden chars.
        let spans: Vec<_> = text
            .as_str()
            .char_indices()
            .filter_map(|(i, c)| {
                TEXT_FLOW_CONTROL_CHARS.contains(&c).then(|| {
                    let lo = span.lo() + BytePos(i as u32 + padding);
                    (c, span.with_lo(lo).with_hi(lo + BytePos(c.len_utf8() as u32)))
                })
            })
            .collect();

        let count = spans.len();
        let labels = point_at_inner_spans
            .then_some(HiddenUnicodeCodepointsDiagLabels { spans: spans.clone() });
        let sub = if point_at_inner_spans && !spans.is_empty() {
            HiddenUnicodeCodepointsDiagSub::Escape { spans }
        } else {
            HiddenUnicodeCodepointsDiagSub::NoEscape { spans }
        };

        cx.emit_span_lint(
            TEXT_DIRECTION_CODEPOINT_IN_LITERAL,
            span,
            HiddenUnicodeCodepointsDiag { label, count, span_label: span, labels, sub },
        );
    }

    fn check_literal(
        &mut self,
        cx: &EarlyContext<'_>,
        text: Symbol,
        lit_kind: ast::token::LitKind,
        span: Span,
        label: &'static str,
    ) {
        if !contains_text_flow_control_chars(text.as_str()) {
            return;
        }
        let (padding, point_at_inner_spans) = match lit_kind {
            // account for `"` or `'`
            ast::token::LitKind::Str | ast::token::LitKind::Char => (1, true),
            // account for `c"`
            ast::token::LitKind::CStr => (2, true),
            // account for `r###"`
            ast::token::LitKind::StrRaw(n) => (n as u32 + 2, true),
            // account for `cr###"`
            ast::token::LitKind::CStrRaw(n) => (n as u32 + 3, true),
            // suppress bad literals.
            ast::token::LitKind::Err(_) => return,
            // Be conservative just in case new literals do support these.
            _ => (0, false),
        };
        self.lint_text_direction_codepoint(cx, text, span, padding, point_at_inner_spans, label);
    }
}

impl EarlyLintPass for HiddenUnicodeCodepoints {
    fn check_attribute(&mut self, cx: &EarlyContext<'_>, attr: &ast::Attribute) {
        if let ast::AttrKind::DocComment(_, comment) = attr.kind {
            if contains_text_flow_control_chars(comment.as_str()) {
                self.lint_text_direction_codepoint(cx, comment, attr.span, 0, false, "doc comment");
            }
        }
    }

    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        // byte strings are already handled well enough by `EscapeError::NonAsciiCharInByteString`
        match &expr.kind {
            ast::ExprKind::Lit(token_lit) => {
                self.check_literal(cx, token_lit.symbol, token_lit.kind, expr.span, "literal");
            }
            ast::ExprKind::FormatArgs(args) => {
                let (lit_kind, text) = args.uncooked_fmt_str;
                self.check_literal(cx, text, lit_kind, args.span, "format string");
            }
            _ => {}
        };
    }
}
