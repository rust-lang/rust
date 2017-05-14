use rustc::lint::*;
use rustc::hir::*;
use utils::span_lint;

/// **What it does:** Checks for usage of blacklisted names for variables, such
/// as `foo`.
///
/// **Why is this bad?** These names are usually placeholder names and should be
/// avoided.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let foo = 3.14;
/// ```
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BlackListedName {
    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat) {
        if let PatKind::Binding(_, _, ref ident, _) = pat.node {
            if self.blacklist.iter().any(|s| ident.node == *s) {
                span_lint(cx,
                          BLACKLISTED_NAME,
                          ident.span,
                          &format!("use of a blacklisted/placeholder name `{}`", ident.node));
            }
        }
    }
}
