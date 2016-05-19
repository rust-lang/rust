use rustc::lint::*;
use rustc::hir::*;
use utils::span_lint;

/// **What it does:** This lints about usage of blacklisted names.
///
/// **Why is this bad?** These names are usually placeholder names and should be avoided.
///
/// **Known problems:** None.
///
/// **Example:** `let foo = 3.14;`
declare_lint! {
    pub BLACKLISTED_NAME,
    Warn,
    "usage of a blacklisted/placeholder name"
}

#[derive(Clone, Debug)]
pub struct BlackListedName {
    blacklist: Vec<String>,
}

impl BlackListedName {
    pub fn new(blacklist: Vec<String>) -> BlackListedName {
        BlackListedName { blacklist: blacklist }
    }
}

impl LintPass for BlackListedName {
    fn get_lints(&self) -> LintArray {
        lint_array!(BLACKLISTED_NAME)
    }
}

impl LateLintPass for BlackListedName {
    fn check_pat(&mut self, cx: &LateContext, pat: &Pat) {
        if let PatKind::Ident(_, ref ident, _) = pat.node {
            if self.blacklist.iter().any(|s| s == &*ident.node.as_str()) {
                span_lint(cx,
                          BLACKLISTED_NAME,
                          pat.span,
                          &format!("use of a blacklisted/placeholder name `{}`", ident.node));
            }
        }
    }
}
