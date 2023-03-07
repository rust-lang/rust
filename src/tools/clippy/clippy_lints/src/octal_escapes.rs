use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_ast::token::{Lit, LitKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;
use std::fmt::Write;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `\0` escapes in string and byte literals that look like octal
    /// character escapes in C.
    ///
    /// ### Why is this bad?
    ///
    /// C and other languages support octal character escapes in strings, where
    /// a backslash is followed by up to three octal digits. For example, `\033`
    /// stands for the ASCII character 27 (ESC). Rust does not support this
    /// notation, but has the escape code `\0` which stands for a null
    /// byte/character, and any following digits do not form part of the escape
    /// sequence. Therefore, `\033` is not a compiler error but the result may
    /// be surprising.
    ///
    /// ### Known problems
    /// The actual meaning can be the intended one. `\x00` can be used in these
    /// cases to be unambiguous.
    ///
    /// The lint does not trigger for format strings in `print!()`, `write!()`
    /// and friends since the string is already preprocessed when Clippy lints
    /// can see it.
    ///
    /// ### Example
    /// ```rust
    /// let one = "\033[1m Bold? \033[0m";  // \033 intended as escape
    /// let two = "\033\0";                 // \033 intended as null-3-3
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let one = "\x1b[1mWill this be bold?\x1b[0m";
    /// let two = "\x0033\x00";
    /// ```
    #[clippy::version = "1.59.0"]
    pub OCTAL_ESCAPES,
    suspicious,
    "string escape sequences looking like octal characters"
}

declare_lint_pass!(OctalEscapes => [OCTAL_ESCAPES]);

impl EarlyLintPass for OctalEscapes {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Lit(token_lit) = &expr.kind {
            if matches!(token_lit.kind, LitKind::Str) {
                check_lit(cx, token_lit, expr.span, true);
            } else if matches!(token_lit.kind, LitKind::ByteStr) {
                check_lit(cx, token_lit, expr.span, false);
            }
        }
    }
}

fn check_lit(cx: &EarlyContext<'_>, lit: &Lit, span: Span, is_string: bool) {
    let contents = lit.symbol.as_str();
    let mut iter = contents.char_indices().peekable();
    let mut found = vec![];

    // go through the string, looking for \0[0-7][0-7]?
    while let Some((from, ch)) = iter.next() {
        if ch == '\\' {
            if let Some((_, '0')) = iter.next() {
                // collect up to two further octal digits
                if let Some((mut to, '0'..='7')) = iter.next() {
                    if let Some((_, '0'..='7')) = iter.peek() {
                        to += 1;
                    }
                    found.push((from, to + 1));
                }
            }
        }
    }

    if found.is_empty() {
        return;
    }

    // construct two suggestion strings, one with \x escapes with octal meaning
    // as in C, and one with \x00 for null bytes.
    let mut suggest_1 = if is_string { "\"" } else { "b\"" }.to_string();
    let mut suggest_2 = suggest_1.clone();
    let mut index = 0;
    for (from, to) in found {
        suggest_1.push_str(&contents[index..from]);
        suggest_2.push_str(&contents[index..from]);

        // construct a replacement escape
        // the maximum value is \077, or \x3f, so u8 is sufficient here
        if let Ok(n) = u8::from_str_radix(&contents[from + 1..to], 8) {
            write!(suggest_1, "\\x{n:02x}").unwrap();
        }

        // append the null byte as \x00 and the following digits literally
        suggest_2.push_str("\\x00");
        suggest_2.push_str(&contents[from + 2..to]);

        index = to;
    }
    suggest_1.push_str(&contents[index..]);
    suggest_1.push('"');
    suggest_2.push_str(&contents[index..]);
    suggest_2.push('"');

    span_lint_and_then(
        cx,
        OCTAL_ESCAPES,
        span,
        &format!(
            "octal-looking escape in {} literal",
            if is_string { "string" } else { "byte string" }
        ),
        |diag| {
            diag.help(format!(
                "octal escapes are not supported, `\\0` is always a null {}",
                if is_string { "character" } else { "byte" }
            ));
            // suggestion 1: equivalent hex escape
            diag.span_suggestion(
                span,
                "if an octal escape was intended, use the hexadecimal representation instead",
                suggest_1,
                Applicability::MaybeIncorrect,
            );
            // suggestion 2: unambiguous null byte
            diag.span_suggestion(
                span,
                format!(
                    "if the null {} is intended, disambiguate using",
                    if is_string { "character" } else { "byte" }
                ),
                suggest_2,
                Applicability::MaybeIncorrect,
            );
        },
    );
}
