use rustc::hir::*;
use rustc::lint::*;
use utils::{span_lint_and_then, snippet};

/// **What it does:** Checks for usage of `*&` and `*&mut` in expressions.
///
/// **Why is this bad?** Immediately dereferencing a reference is no-op and
/// makes the code less clear.
///
/// **Known problems:** Multiple dereference/addrof pairs are not handled so
/// the suggested fix for `x = **&&y` is `x = *&y`, which is still incorrect.
///
/// **Example:**
/// ```rust
/// let a = f(*&mut b);
/// let c = *&d;
/// ```
declare_lint! {
    pub DEREF_ADDROF,
    Warn,
    "use of `*&` or `*&mut` in an expression"
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DEREF_ADDROF)
    }
}

impl LateLintPass for Pass {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprUnary(UnDeref, ref deref_target) = e.node {
            if let ExprAddrOf(_, ref addrof_target) = deref_target.node {
                span_lint_and_then(
                    cx,
                    DEREF_ADDROF,
                    e.span,
                    "immediately dereferencing a reference",
                    |db| {
                        db.span_suggestion(e.span, "try this",
                                             format!("{}", snippet(cx, addrof_target.span, "_")));
                    });
            }
        }
    }
}
