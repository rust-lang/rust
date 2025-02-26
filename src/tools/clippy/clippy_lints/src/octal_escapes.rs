use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt;
use rustc_ast::token::LitKind;
use rustc_ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::{BytePos, Pos, SpanData};

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
    /// ```no_run
    /// let one = "\033[1m Bold? \033[0m";  // \033 intended as escape
    /// let two = "\033\0";                 // \033 intended as null-3-3
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
        if let ExprKind::Lit(lit) = &expr.kind
            // The number of bytes from the start of the token to the start of literal's text.
            && let start_offset = BytePos::from_u32(match lit.kind {
                LitKind::Str => 1,
                LitKind::ByteStr | LitKind::CStr => 2,
                _ => return,
            })
            && !expr.span.in_external_macro(cx.sess().source_map())
        {
            let s = lit.symbol.as_str();
            let mut iter = s.as_bytes().iter();
            while let Some(&c) = iter.next() {
                if c == b'\\'
                    // Always move the iterator to read the escape char.
                    && let Some(b'0') = iter.next()
                {
                    // C-style octal escapes read from one to three characters.
                    // The first character (`0`) has already been read.
                    let (tail, len, c_hi, c_lo) = match *iter.as_slice() {
                        [c_hi @ b'0'..=b'7', c_lo @ b'0'..=b'7', ref tail @ ..] => (tail, 4, c_hi, c_lo),
                        [c_lo @ b'0'..=b'7', ref tail @ ..] => (tail, 3, b'0', c_lo),
                        _ => continue,
                    };
                    iter = tail.iter();
                    let offset = start_offset + BytePos::from_usize(s.len() - tail.len());
                    let data = expr.span.data();
                    let span = SpanData {
                        lo: data.lo + offset - BytePos::from_u32(len),
                        hi: data.lo + offset,
                        ..data
                    }
                    .span();

                    // Last check to make sure the source text matches what we read from the string.
                    // Macros are involved somehow if this doesn't match.
                    if span.check_source_text(cx, |src| match *src.as_bytes() {
                        [b'\\', b'0', lo] => lo == c_lo,
                        [b'\\', b'0', hi, lo] => hi == c_hi && lo == c_lo,
                        _ => false,
                    }) {
                        span_lint_and_then(cx, OCTAL_ESCAPES, span, "octal-looking escape in a literal", |diag| {
                            diag.help_once("octal escapes are not supported, `\\0` is always null")
                                .span_suggestion(
                                    span,
                                    "if an octal escape is intended, use a hex escape instead",
                                    format!("\\x{:02x}", (((c_hi - b'0') << 3) | (c_lo - b'0'))),
                                    Applicability::MaybeIncorrect,
                                )
                                .span_suggestion(
                                    span,
                                    "if a null escape is intended, disambiguate using",
                                    format!("\\x00{}{}", c_hi as char, c_lo as char),
                                    Applicability::MaybeIncorrect,
                                );
                        });
                    } else {
                        break;
                    }
                }
            }
        }
    }
}
