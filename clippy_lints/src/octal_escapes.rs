use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_ast::token::{Lit, LitKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `\0` escapes in string and byte literals that look like octal character
    /// escapes in C
    ///
    /// ### Why is this bad?
    /// Rust does not support octal notation for character escapes. `\0` is always a
    /// null byte/character, and any following digits do not form part of the escape
    /// sequence.
    ///
    /// ### Known problems
    /// The actual meaning can be the intended one. `\x00` can be used in these
    /// cases to be unambigious.
    ///
    /// # Example
    /// ```rust
    /// // Bad
    /// let one = "\033[1m Bold? \033[0m";  // \033 intended as escape
    /// let two = "\033\0";                 // \033 intended as null-3-3
    ///
    /// // Good
    /// let one = "\x1b[1mWill this be bold?\x1b[0m";
    /// let two = "\x0033\x00";
    /// ```
    #[clippy::version = "1.58.0"]
    pub OCTAL_ESCAPES,
    suspicious,
    "string escape sequences looking like octal characters"
}

declare_lint_pass!(OctalEscapes => [OCTAL_ESCAPES]);

impl EarlyLintPass for OctalEscapes {
    fn check_expr(&mut self, cx: &EarlyContext<'tcx>, expr: &Expr) {
        if in_external_macro(cx.sess, expr.span) {
            return;
        }

        if let ExprKind::Lit(lit) = &expr.kind {
            if matches!(lit.token.kind, LitKind::Str) {
                check_lit(cx, &lit.token, lit.span, true);
            } else if matches!(lit.token.kind, LitKind::ByteStr) {
                check_lit(cx, &lit.token, lit.span, false);
            }
        }
    }
}

fn check_lit(cx: &EarlyContext<'tcx>, lit: &Lit, span: Span, is_string: bool) {
    let contents = lit.symbol.as_str();
    let mut iter = contents.char_indices();

    // go through the string, looking for \0[0-7]
    while let Some((from, ch)) = iter.next() {
        if ch == '\\' {
            if let Some((mut to, '0')) = iter.next() {
                // collect all further potentially octal digits
                while let Some((j, '0'..='7')) = iter.next() {
                    to = j + 1;
                }
                // if it's more than just `\0` we have a match
                if to > from + 2 {
                    emit(cx, &contents, from, to, span, is_string);
                    return;
                }
            }
        }
    }
}

fn emit(cx: &EarlyContext<'tcx>, contents: &str, from: usize, to: usize, span: Span, is_string: bool) {
    // construct a replacement escape for that case that octal was intended
    let escape = &contents[from + 1..to];
    let literal_suggestion = if is_string {
        u32::from_str_radix(escape, 8).ok().and_then(|n| {
            if n < 256 {
                Some(format!("\\x{:02x}", n))
            } else if n <= std::char::MAX as u32 {
                Some(format!("\\u{{{:x}}}", n))
            } else {
                None
            }
        })
    } else {
        u8::from_str_radix(escape, 8).ok().map(|n| format!("\\x{:02x}", n))
    };

    span_lint_and_then(
        cx,
        OCTAL_ESCAPES,
        span,
        &format!(
            "octal-looking escape in {} literal",
            if is_string { "string" } else { "byte string" }
        ),
        |diag| {
            diag.help(&format!(
                "octal escapes are not supported, `\\0` is always a null {}",
                if is_string { "character" } else { "byte" }
            ));
            // suggestion 1: equivalent hex escape
            if let Some(sugg) = literal_suggestion {
                diag.span_suggestion(
                    span,
                    "if an octal escape is intended, use",
                    format!("\"{}{}{}\"", &contents[..from], sugg, &contents[to..]),
                    Applicability::MaybeIncorrect,
                );
            }
            // suggestion 2: unambiguous null byte
            diag.span_suggestion(
                span,
                &format!(
                    "if the null {} is intended, disambiguate using",
                    if is_string { "character" } else { "byte" }
                ),
                format!("\"{}\\x00{}\"", &contents[..from], &contents[from + 2..]),
                Applicability::MaybeIncorrect,
            );
        },
    );
}
