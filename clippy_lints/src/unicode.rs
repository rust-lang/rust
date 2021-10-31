use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_lint_allowed;
use clippy_utils::source::snippet;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, HirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use unicode_normalization::UnicodeNormalization;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for invisible Unicode characters in the code.
    ///
    /// ### Why is this bad?
    /// Having an invisible character in the code makes for all
    /// sorts of April fools, but otherwise is very much frowned upon.
    ///
    /// ### Example
    /// You don't see it, but there may be a zero-width space or soft hyphen
    /// some­where in this text.
    pub INVISIBLE_CHARACTERS,
    correctness,
    "using an invisible character in a string literal, which is confusing"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for non-ASCII characters in string literals.
    ///
    /// ### Why is this bad?
    /// Yeah, we know, the 90's called and wanted their charset
    /// back. Even so, there still are editors and other programs out there that
    /// don't work well with Unicode. So if the code is meant to be used
    /// internationally, on multiple operating systems, or has other portability
    /// requirements, activating this lint could be useful.
    ///
    /// ### Example
    /// ```rust
    /// let x = String::from("€");
    /// ```
    /// Could be written as:
    /// ```rust
    /// let x = String::from("\u{20ac}");
    /// ```
    pub NON_ASCII_LITERAL,
    restriction,
    "using any literal non-ASCII chars in a string literal instead of using the `\\u` escape"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for string literals that contain Unicode in a form
    /// that is not equal to its
    /// [NFC-recomposition](http://www.unicode.org/reports/tr15/#Norm_Forms).
    ///
    /// ### Why is this bad?
    /// If such a string is compared to another, the results
    /// may be surprising.
    ///
    /// ### Example
    /// You may not see it, but "à"" and "à"" aren't the same string. The
    /// former when escaped is actually `"a\u{300}"` while the latter is `"\u{e0}"`.
    pub UNICODE_NOT_NFC,
    pedantic,
    "using a Unicode literal not in NFC normal form (see [Unicode tr15](http://www.unicode.org/reports/tr15/) for further information)"
}

declare_lint_pass!(Unicode => [INVISIBLE_CHARACTERS, NON_ASCII_LITERAL, UNICODE_NOT_NFC]);

impl LateLintPass<'_> for Unicode {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Lit(ref lit) = expr.kind {
            if let LitKind::Str(_, _) = lit.node {
                check_str(cx, lit.span, expr.hir_id);
            }
        }
    }
}

fn escape<T: Iterator<Item = char>>(s: T) -> String {
    let mut result = String::new();
    for c in s {
        if c as u32 > 0x7F {
            for d in c.escape_unicode() {
                result.push(d);
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn check_str(cx: &LateContext<'_>, span: Span, id: HirId) {
    let string = snippet(cx, span, "");
    if string.chars().any(|c| ['\u{200B}', '\u{ad}', '\u{2060}'].contains(&c)) {
        span_lint_and_sugg(
            cx,
            INVISIBLE_CHARACTERS,
            span,
            "invisible character detected",
            "consider replacing the string with",
            string
                .replace("\u{200B}", "\\u{200B}")
                .replace("\u{ad}", "\\u{AD}")
                .replace("\u{2060}", "\\u{2060}"),
            Applicability::MachineApplicable,
        );
    }
    if string.chars().any(|c| c as u32 > 0x7F) {
        span_lint_and_sugg(
            cx,
            NON_ASCII_LITERAL,
            span,
            "literal non-ASCII character detected",
            "consider replacing the string with",
            if is_lint_allowed(cx, UNICODE_NOT_NFC, id) {
                escape(string.chars())
            } else {
                escape(string.nfc())
            },
            Applicability::MachineApplicable,
        );
    }
    if is_lint_allowed(cx, NON_ASCII_LITERAL, id) && string.chars().zip(string.nfc()).any(|(a, b)| a != b) {
        span_lint_and_sugg(
            cx,
            UNICODE_NOT_NFC,
            span,
            "non-NFC Unicode sequence detected",
            "consider replacing the string with",
            string.nfc().collect::<String>(),
            Applicability::MachineApplicable,
        );
    }
}
