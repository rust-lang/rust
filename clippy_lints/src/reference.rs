use syntax::ast::{Expr, ExprKind, UnOp};
use rustc::lint::*;
use utils::{snippet, span_lint_and_sugg};

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

fn without_parens(mut e: &Expr) -> &Expr {
    while let ExprKind::Paren(ref child_e) = e.node {
        e = child_e;
    }
    e
}

impl EarlyLintPass for Pass {
    fn check_expr(&mut self, cx: &EarlyContext, e: &Expr) {
        if let ExprKind::Unary(UnOp::Deref, ref deref_target) = e.node {
            if let ExprKind::AddrOf(_, ref addrof_target) = without_parens(deref_target).node {
                span_lint_and_sugg(
                    cx,
                    DEREF_ADDROF,
                    e.span,
                    "immediately dereferencing a reference",
                    "try this",
                    format!("{}", snippet(cx, addrof_target.span, "_")),
                );
            }
        }
    }
}
