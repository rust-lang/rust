//! Migration code for the `expr_fragment_specifier_2024`
//! rule.
use tracing::debug;

use rustc_ast::token::Token;
use rustc_ast::token::TokenKind;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::tokenstream::TokenTree;
use rustc_session::declare_lint;
use rustc_session::declare_lint_pass;
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::edition::Edition;
use rustc_span::sym;

use crate::lints::MacroExprFragment2024;
use crate::EarlyLintPass;

declare_lint! {
    /// The `edition_2024_expr_fragment_specifier` lint detects the use of `expr` fragments
    /// during migration to the 2024 edition.
    ///
    /// The `expr` fragment specifier will accept more expressions in the 2024 edition.
    /// To maintain the current behavior, use the `expr_2021` fragment specifier.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021,compile_fail
    /// #![deny(edition_2024_expr_fragment_specifier)]
    /// macro_rules! m {
    ///   ($e:expr) => {
    ///       $e
    ///   }
    /// }
    ///
    /// fn main() {
    ///    m!(1);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Rust [editions] allow the language to evolve without breaking
    /// backwards compatibility. This lint catches code that uses new keywords
    /// that are added to the language that are used as identifiers (such as a
    /// variable name, function name, etc.). If you switch the compiler to a
    /// new edition without updating the code, then it will fail to compile if
    /// you are using a new keyword as an identifier.
    ///
    /// This lint solves the problem automatically. It is "allow" by default
    /// because the code is perfectly valid in older editions. The [`cargo
    /// fix`] tool with the `--edition` flag will switch this lint to "warn"
    /// and automatically apply the suggested fix from the compiler (which is
    /// to use a raw identifier). This provides a completely automated way to
    /// update old code for a new edition.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    /// [raw identifier]: https://doc.rust-lang.org/reference/identifiers.html
    /// [`cargo fix`]: https://doc.rust-lang.org/cargo/commands/cargo-fix.html
    pub EDITION_2024_EXPR_FRAGMENT_SPECIFIER,
    Allow,
    "The `expr` fragment specifier will accept more expressions in the 2024 edition. \
    To keep the existing behavior, use the `expr_2021` fragment specifier.",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "Migration Guide <https://doc.rust-lang.org/nightly/edition-guide/rust-2024/macro-fragment-specifiers.html>",
    };
}

declare_lint_pass!(Expr2024 => [EDITION_2024_EXPR_FRAGMENT_SPECIFIER,]);

impl Expr2024 {
    fn check_tokens(&mut self, cx: &crate::EarlyContext<'_>, tokens: &TokenStream) {
        let mut prev_dollar = false;
        for tt in tokens.trees() {
            match tt {
                TokenTree::Token(token, _) => {
                    if token.kind == TokenKind::Dollar {
                        prev_dollar = true;
                        continue;
                    } else {
                        if !prev_dollar {
                            self.check_ident_token(cx, token);
                        }
                    }
                }
                TokenTree::Delimited(.., tts) => self.check_tokens(cx, tts),
            }
            prev_dollar = false;
        }
    }

    fn check_ident_token(&mut self, cx: &crate::EarlyContext<'_>, token: &Token) {
        debug!("check_ident_token: {:?}", token);
        let (sym, edition) = match token.kind {
            TokenKind::Ident(sym, _) => (sym, Edition::Edition2024),
            _ => return,
        };

        debug!("token.span.edition(): {:?}", token.span.edition());
        if token.span.edition() >= edition {
            return;
        }

        if sym != sym::expr {
            return;
        }

        debug!("emitting lint");
        cx.builder.emit_span_lint(
            &EDITION_2024_EXPR_FRAGMENT_SPECIFIER,
            token.span.into(),
            MacroExprFragment2024 { suggestion: token.span },
        );
    }
}

impl EarlyLintPass for Expr2024 {
    fn check_mac_def(&mut self, cx: &crate::EarlyContext<'_>, mc: &rustc_ast::MacroDef) {
        self.check_tokens(cx, &mc.body.tokens);
    }
}
