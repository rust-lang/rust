use rustc::lint::*;
use utils::span_lint;
use syntax::parse::token::InternedString;
use syntax::ast::*;

/// **What it does:** This lint checks for various things we like to keep tidy in clippy
///
/// **Why is this bad?** ???
///
/// **Known problems:** None.
///
/// **Example:** wrong ordering of the util::paths constants
declare_lint! {
    pub CLIPPY_LINTS_INTERNAL, Allow,
    "Various things that will negatively affect your clippy experience"
}


#[derive(Copy, Clone)]
pub struct Clippy;

impl LintPass for Clippy {
    fn get_lints(&self) -> LintArray {
        lint_array!(CLIPPY_LINTS_INTERNAL)
    }
}

impl EarlyLintPass for Clippy {
    fn check_crate(&mut self, cx: &EarlyContext, krate: &Crate) {
        if let Some(utils) = krate.module.items.iter().find(|item| item.ident.name.as_str() == "utils") {
            if let ItemKind::Mod(ref utils_mod) = utils.node {
                if let Some(paths) = utils_mod.items.iter().find(|item| item.ident.name.as_str() == "paths") {
                    if let ItemKind::Mod(ref paths_mod) = paths.node {
                        let mut last_name: Option<InternedString> = None;
                        for item in &paths_mod.items {
                            let name = item.ident.name.as_str();
                            if let Some(ref last_name) = last_name {
                                if **last_name > *name {
                                    span_lint(cx,
                                              CLIPPY_LINTS_INTERNAL,
                                              item.span,
                                              "this constant should be before the previous constant due to lexical ordering",
                                    );
                                }
                            }
                            last_name = Some(name);
                        }
                    }
                }
            }
        }
    }
}
