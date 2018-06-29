use rustc::lint::*;
use rustc::hir::*;
use crate::utils::span_lint;

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
declare_clippy_lint! {
    pub BLACKLISTED_NAME,
    style,
    "usage of a blacklisted/placeholder name"
}

#[derive(Clone, Debug)]
pub struct BlackListedName {
    blacklist: Vec<String>,
}

impl BlackListedName {
    pub fn new(blacklist: Vec<String>) -> Self {
        Self {
            blacklist,
        }
    }
}

impl LintPass for BlackListedName {
    fn get_lints(&self) -> LintArray {
        lint_array!(BLACKLISTED_NAME)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BlackListedName {
    fn check_pat(&mut self, cx: &LateContext<'a, 'tcx>, pat: &'tcx Pat) {
        if let PatKind::Binding(_, _, ident, _) = pat.node {
            if self.blacklist.iter().any(|s| ident.name == *s) {
                span_lint(
                    cx,
                    BLACKLISTED_NAME,
                    ident.span,
                    &format!("use of a blacklisted/placeholder name `{}`", ident.name),
                );
            }
        }
    }
}
