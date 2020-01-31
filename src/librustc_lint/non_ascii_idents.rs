use crate::{EarlyContext, EarlyLintPass, LintContext};
use syntax::ast;

declare_lint! {
    pub NON_ASCII_IDENTS,
    Allow,
    "detects non-ASCII identifiers"
}

declare_lint! {
    pub UNCOMMON_CODEPOINTS,
    Warn,
    "detects uncommon Unicode codepoints in identifiers"
}

declare_lint_pass!(NonAsciiIdents => [NON_ASCII_IDENTS, UNCOMMON_CODEPOINTS]);

impl EarlyLintPass for NonAsciiIdents {
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: ast::Ident) {
        use unicode_security::GeneralSecurityProfile;
        let name_str = ident.name.as_str();
        if name_str.is_ascii() {
            return;
        }
        cx.struct_span_lint(
            NON_ASCII_IDENTS,
            ident.span,
            |lint| lint.build("identifier contains non-ASCII characters").emit(),
        );
        if !name_str.chars().all(GeneralSecurityProfile::identifier_allowed) {
            cx.struct_span_lint(
                UNCOMMON_CODEPOINTS,
                ident.span,
                |lint| lint.build("identifier contains uncommon Unicode codepoints").emit(),
            )
        }
    }
}
