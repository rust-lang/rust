//! Migration code for the `expr_fragment_specifier_2024` rule.

use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::sym;
use tracing::debug;

use crate::EarlyLintPass;
use crate::lints::MacroExprFragment2024;

declare_lint! {
    /// The `edition_2024_expr_fragment_specifier` lint detects the use of
    /// `expr` fragments in macros during migration to the 2024 edition.
    ///
    /// The `expr` fragment specifier will accept more expressions in the 2024
    /// edition. To maintain the behavior from the 2021 edition and earlier, use
    /// the `expr_2021` fragment specifier.
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
    /// Rust [editions] allow the language to evolve without breaking backwards
    /// compatibility. This lint catches code that uses [macro matcher fragment
    /// specifiers] that have changed meaning in the 2024 edition. If you switch
    /// to the new edition without updating the code, your macros may behave
    /// differently.
    ///
    /// In the 2024 edition, the `expr` fragment specifier `expr` will also
    /// match `const { ... }` blocks. This means if a macro had a pattern that
    /// matched `$e:expr` and another that matches `const { $e: expr }`, for
    /// example, that under the 2024 edition the first pattern would match while
    /// in the 2021 and earlier editions the second pattern would match. To keep
    /// the old behavior, use the `expr_2021` fragment specifier.
    ///
    /// This lint detects macros whose behavior might change due to the changing
    /// meaning of the `expr` fragment specifier. It is "allow" by default
    /// because the code is perfectly valid in older editions. The [`cargo fix`]
    /// tool with the `--edition` flag will switch this lint to "warn" and
    /// automatically apply the suggested fix from the compiler. This provides a
    /// completely automated way to update old code for a new edition.
    ///
    /// Using `cargo fix --edition` with this lint will ensure that your code
    /// retains the same behavior. This may not be the desired, as macro authors
    /// often will want their macros to use the latest grammar for matching
    /// expressions. Be sure to carefully review changes introduced by this lint
    /// to ensure the macros implement the desired behavior.
    ///
    /// [editions]: https://doc.rust-lang.org/edition-guide/
    /// [macro matcher fragment specifiers]: https://doc.rust-lang.org/edition-guide/rust-2024/macro-fragment-specifiers.html
    /// [`cargo fix`]: https://doc.rust-lang.org/cargo/commands/cargo-fix.html
    pub EDITION_2024_EXPR_FRAGMENT_SPECIFIER,
    Allow,
    "The `expr` fragment specifier will accept more expressions in the 2024 edition. \
    To keep the existing behavior, use the `expr_2021` fragment specifier.",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "Migration Guide <https://doc.rust-lang.org/edition-guide/rust-2024/macro-fragment-specifiers.html>",
    };
}

declare_lint_pass!(Expr2024 => [EDITION_2024_EXPR_FRAGMENT_SPECIFIER,]);

impl Expr2024 {
    fn check_tokens(&mut self, cx: &crate::EarlyContext<'_>, tokens: &TokenStream) {
        let mut prev_colon = false;
        let mut prev_identifier = false;
        let mut prev_dollar = false;
        for tt in tokens.iter() {
            debug!(
                "check_tokens: {:?} - colon {prev_dollar} - ident {prev_identifier} - colon {prev_colon}",
                tt
            );
            match tt {
                TokenTree::Token(token, _) => match token.kind {
                    TokenKind::Dollar => {
                        prev_dollar = true;
                        continue;
                    }
                    TokenKind::Ident(..) | TokenKind::NtIdent(..) => {
                        if prev_colon && prev_identifier && prev_dollar {
                            self.check_ident_token(cx, token);
                        } else if prev_dollar {
                            prev_identifier = true;
                            continue;
                        }
                    }
                    TokenKind::Colon => {
                        if prev_dollar && prev_identifier {
                            prev_colon = true;
                            continue;
                        }
                    }
                    _ => {}
                },
                TokenTree::Delimited(.., tts) => self.check_tokens(cx, tts),
            }
            prev_colon = false;
            prev_identifier = false;
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
