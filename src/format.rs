use rustc::lint::*;
use rustc_front::hir::*;
use utils::{is_expn_of, span_lint};

/// **What it does:** This lints about use of `format!("string literal with no argument")`.
///
/// **Why is this bad?** There is no point of doing that. If you want a `String` you can use
/// `to_owned` on the string literal. The even worst `&format!("foo")` is often encountered in the
/// wild.
///
/// **Known problems:** None.
///
/// **Example:** `format!("foo")`
declare_lint! {
    pub USELESS_FORMAT,
    Warn,
    "useless use of `format!`"
}

#[derive(Copy, Clone, Debug)]
pub struct FormatMacLint;

impl LintPass for FormatMacLint {
    fn get_lints(&self) -> LintArray {
        lint_array![USELESS_FORMAT]
    }
}

impl LateLintPass for FormatMacLint {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        // `format!("foo")` expansion contains `match () { () => [], }`
        if let ExprMatch(ref matchee, _, _) = expr.node {
            if let ExprTup(ref tup) = matchee.node {
                if tup.is_empty() && is_expn_of(cx, expr.span, "format").is_some() {
                    span_lint(cx, USELESS_FORMAT, expr.span, &"useless use of `format!`");
                }
            }
        }
    }
}
