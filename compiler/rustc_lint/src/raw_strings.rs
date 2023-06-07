use crate::lints::{UnusedRawStringHashDiag, UnusedRawStringDiag};
use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast::{Expr, ExprKind, token::LitKind::{StrRaw, ByteStrRaw, CStrRaw}};

// Examples / Intuition:
// Must be raw, but hashes are just right   r#" " "#  (neither warning)
// Must be raw, but has too many hashes     r#" \ "#
// Non-raw and has too many hashes          r#" ! "#  (both warnings)
// Non-raw and hashes are just right         r" ! "

declare_lint! {
    /// The `unused_raw_string_hash` lint checks whether raw strings
    /// use more hashes than they need.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_raw_string_hash)]
    /// fn main() {
    ///     let x = r####"Use the r#"..."# notation for raw strings"####;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The hashes are not needed and should be removed.
    pub UNUSED_RAW_STRING_HASH,
    Warn,
    "Raw string literal has unneeded hashes"
}

declare_lint_pass!(UnusedRawStringHash => [UNUSED_RAW_STRING_HASH]);

impl EarlyLintPass for UnusedRawStringHash {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Lit(lit) = expr.kind {
            // Check all raw string variants with one or more hashes
            if let StrRaw(hc @1..) | ByteStrRaw(hc @1..) | CStrRaw(hc @1..) = lit.kind {
                // Now check if `hash_count` hashes are actually required
                let hash_count = hc as usize;
                let contents = lit.symbol.as_str();
                let hash_req = Self::required_hashes(contents);
                if hash_req < hash_count {
                    cx.emit_spanned_lint(
                        UNUSED_RAW_STRING_HASH,
                        expr.span,
                        UnusedRawStringHashDiag {
                            span: expr.span,
                            hash_count,
                            hash_req,
                        },
                    );
                }
            }
        }
    }
}

impl UnusedRawStringHash {
    fn required_hashes(contents: &str) -> usize {
        // How many hashes are needed to wrap the input string?
        // aka length of longest "#* sequence or zero if none exists

        // TODO potential speedup: short-circuit max() if `hash_count` found

        contents.as_bytes()
            .split(|&b| b == b'"')
            .skip(1)  // first element is the only one not starting with "
            .map(|bs| 1 + bs.iter().take_while(|&&b| b == b'#').count())
            .max()
            .unwrap_or(0)
    }
}

declare_lint! {
    /// The `unused_raw_string` lint checks whether raw strings need
    /// to be raw.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_raw_string)]
    /// fn main() {
    ///     let x = r"  totally normal string  ";
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// If a string contains no escapes and no double quotes, it does
    /// not need to be raw.
    pub UNUSED_RAW_STRING,
    Warn,
    "String literal does not need to be raw"
}

declare_lint_pass!(UnusedRawString => [UNUSED_RAW_STRING]);

impl EarlyLintPass for UnusedRawString {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Lit(lit) = expr.kind {
            // Check all raw string variants
            if let StrRaw(hc) | ByteStrRaw(hc) | CStrRaw(hc) = lit.kind {
                // Now check if string needs to be raw
                let contents = lit.symbol.as_str();
                let contains_hashes = hc > 0;

                if !contents.bytes().any(|b| matches!(b, b'\\' | b'"')) {
                    cx.emit_spanned_lint(
                        UNUSED_RAW_STRING,
                        expr.span,
                        UnusedRawStringDiag {
                            span: expr.span,
                            contains_hashes,
                        },
                    );
                }
            }
        }
    }
}

