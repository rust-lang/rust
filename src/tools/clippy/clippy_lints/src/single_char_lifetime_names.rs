use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast::{GenericParam, GenericParamKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for lifetimes with names which are one character
    /// long.
    ///
    /// ### Why restrict this?
    /// A single character is likely not enough to express the
    /// purpose of a lifetime. Using a longer name can make code
    /// easier to understand.
    ///
    /// ### Known problems
    /// Rust programmers and learning resources tend to use single
    /// character lifetimes, so this lint is at odds with the
    /// ecosystem at large. In addition, the lifetime's purpose may
    /// be obvious or, rarely, expressible in one character.
    ///
    /// ### Example
    /// ```no_run
    /// struct DiagnosticCtx<'a> {
    ///     source: &'a str,
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct DiagnosticCtx<'src> {
    ///     source: &'src str,
    /// }
    /// ```
    #[clippy::version = "1.60.0"]
    pub SINGLE_CHAR_LIFETIME_NAMES,
    restriction,
    "warns against single-character lifetime names"
}

declare_lint_pass!(SingleCharLifetimeNames => [SINGLE_CHAR_LIFETIME_NAMES]);

impl EarlyLintPass for SingleCharLifetimeNames {
    fn check_generic_param(&mut self, ctx: &EarlyContext<'_>, param: &GenericParam) {
        if param.ident.span.in_external_macro(ctx.sess().source_map()) {
            return;
        }

        if let GenericParamKind::Lifetime = param.kind
            && !param.is_placeholder
            && param.ident.as_str().len() <= 2
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                ctx,
                SINGLE_CHAR_LIFETIME_NAMES,
                param.ident.span,
                "single-character lifetime names are likely uninformative",
                |diag| {
                    diag.help("use a more informative name");
                },
            );
        }
    }
}
