use rustc::lint::*;
use rustc_front::hir::*;
use syntax::codemap::Span;

use syntax::ast::Lit_::*;

use unicode_normalization::UnicodeNormalization;

use utils::{snippet, span_help_and_lint};

/// **What it does:** This lint checks for the unicode zero-width space in the code. It is `Warn` by default.
///
/// **Why is this bad?** Having an invisible character in the code makes for all sorts of April fools, but otherwise is very much frowned upon.
///
/// **Known problems:** None
///
/// **Example:** You don't see it, but there may be a zero-width space somewhere in this text.
declare_lint!{ pub ZERO_WIDTH_SPACE, Deny,
               "using a zero-width space in a string literal, which is confusing" }
/// **What it does:** This lint checks for non-ascii characters in string literals. It is `Allow` by default.
///
/// **Why is this bad?** Yeah, we know, the 90's called and wanted their charset back. Even so, there still are editors and other programs out there that don't work well with unicode. So if the code is meant to be used internationally, on multiple operating systems, or has other portability requirements, activating this lint could be useful.
///
/// **Known problems:** None
///
/// **Example:** `let x = "Hä?"`
declare_lint!{ pub NON_ASCII_LITERAL, Allow,
               "using any literal non-ASCII chars in a string literal; suggests \
                using the \\u escape instead" }
/// **What it does:** This lint checks for string literals that contain unicode in a form that is not equal to its [NFC-recomposition](http://www.unicode.org/reports/tr15/#Norm_Forms). This lint is `Allow` by default.
///
/// **Why is this bad?** If such a string is compared to another, the results may be surprising.
///
/// **Known problems** None
///
/// **Example:** You may not see it, but "à" and "à" aren't the same string. The former when escaped is actually "a\u{300}" while the latter is "\u{e0}".
declare_lint!{ pub UNICODE_NOT_NFC, Allow,
               "using a unicode literal not in NFC normal form (see \
               http://www.unicode.org/reports/tr15/ for further information)" }


#[derive(Copy, Clone)]
pub struct Unicode;

impl LintPass for Unicode {
    fn get_lints(&self) -> LintArray {
        lint_array!(ZERO_WIDTH_SPACE, NON_ASCII_LITERAL, UNICODE_NOT_NFC)
    }
}

impl LateLintPass for Unicode {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprLit(ref lit) = expr.node {
            if let LitStr(_, _) = lit.node {
                check_str(cx, lit.span)
            }
        }
    }
}

fn escape<T: Iterator<Item = char>>(s: T) -> String {
    let mut result = String::new();
    for c in s {
        if c as u32 > 0x7F {
            for d in c.escape_unicode() {
                result.push(d)
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn check_str(cx: &LateContext, span: Span) {
    let string = snippet(cx, span, "");
    if string.contains('\u{200B}') {
        span_help_and_lint(cx,
                           ZERO_WIDTH_SPACE,
                           span,
                           "zero-width space detected",
                           &format!("Consider replacing the string with:\n\"{}\"",
                                    string.replace("\u{200B}", "\\u{200B}")));
    }
    if string.chars().any(|c| c as u32 > 0x7F) {
        span_help_and_lint(cx,
                           NON_ASCII_LITERAL,
                           span,
                           "literal non-ASCII character detected",
                           &format!("Consider replacing the string with:\n\"{}\"",
                                    if cx.current_level(UNICODE_NOT_NFC) == Level::Allow {
                                        escape(string.chars())
                                    } else {
                                        escape(string.nfc())
                                    }));
    }
    if cx.current_level(NON_ASCII_LITERAL) == Level::Allow && string.chars().zip(string.nfc()).any(|(a, b)| a != b) {
        span_help_and_lint(cx,
                           UNICODE_NOT_NFC,
                           span,
                           "non-nfc unicode sequence detected",
                           &format!("Consider replacing the string with:\n\"{}\"", string.nfc().collect::<String>()));
    }
}
