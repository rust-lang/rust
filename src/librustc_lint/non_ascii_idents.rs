use crate::lint::{EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use syntax::ast;

declare_lint! {
    pub NON_ASCII_IDENTS,
    Allow,
    "detects non-ASCII identifiers"
}

declare_lint_pass!(NonAsciiIdents => [NON_ASCII_IDENTS]);

impl EarlyLintPass for NonAsciiIdents {
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: ast::Ident) {
        if !ident.name.as_str().is_ascii() {
            cx.struct_span_lint(
                NON_ASCII_IDENTS,
                ident.span,
                "identifier contains non-ASCII characters",
            ).emit();
        }
    }
}
