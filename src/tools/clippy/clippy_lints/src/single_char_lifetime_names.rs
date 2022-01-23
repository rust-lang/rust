use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast::{GenericParam, GenericParamKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for lifetimes with names which are one character
    /// long.
    ///
    /// ### Why is this bad?
    /// A single character is likely not enough to express the
    /// purpose of a lifetime. Using a longer name can make code
    /// easier to understand, especially for those who are new to
    /// Rust.
    ///
    /// ### Known problems
    /// Rust programmers and learning resources tend to use single
    /// character lifetimes, so this lint is at odds with the
    /// ecosystem at large. In addition, the lifetime's purpose may
    /// be obvious or, rarely, expressible in one character.
    ///
    /// ### Example
    /// ```rust
    /// struct DiagnosticCtx<'a> {
    ///     source: &'a str,
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// struct DiagnosticCtx<'src> {
    ///     source: &'src str,
    /// }
    /// ```
    #[clippy::version = "1.59.0"]
    pub SINGLE_CHAR_LIFETIME_NAMES,
    restriction,
    "warns against single-character lifetime names"
}

declare_lint_pass!(SingleCharLifetimeNames => [SINGLE_CHAR_LIFETIME_NAMES]);

impl EarlyLintPass for SingleCharLifetimeNames {
    fn check_generic_param(&mut self, ctx: &EarlyContext<'_>, param: &GenericParam) {
        if in_external_macro(ctx.sess(), param.ident.span) {
            return;
        }

        if let GenericParamKind::Lifetime = param.kind {
            if !param.is_placeholder && param.ident.as_str().len() <= 2 {
                span_lint_and_help(
                    ctx,
                    SINGLE_CHAR_LIFETIME_NAMES,
                    param.ident.span,
                    "single-character lifetime names are likely uninformative",
                    None,
                    "use a more informative name",
                );
            }
        }
    }
}
