use crate::utils::{in_macro, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::ast::{Expr, ExprKind, UnOp};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `*&` and `*&mut` in expressions.
    ///
    /// **Why is this bad?** Immediately dereferencing a reference is no-op and
    /// makes the code less clear.
    ///
    /// **Known problems:** Multiple dereference/addrof pairs are not handled so
    /// the suggested fix for `x = **&&y` is `x = *&y`, which is still incorrect.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// let a = f(*&mut b);
    /// let c = *&d;
    /// ```
    pub DEREF_ADDROF,
    complexity,
    "use of `*&` or `*&mut` in an expression"
}

declare_lint_pass!(DerefAddrOf => [DEREF_ADDROF]);

fn without_parens(mut e: &Expr) -> &Expr {
    while let ExprKind::Paren(ref child_e) = e.kind {
        e = child_e;
    }
    e
}

impl EarlyLintPass for DerefAddrOf {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &Expr) {
        if_chain! {
            if let ExprKind::Unary(UnOp::Deref, ref deref_target) = e.kind;
            if let ExprKind::AddrOf(_, _, ref addrof_target) = without_parens(deref_target).kind;
            if !in_macro(addrof_target.span);
            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    DEREF_ADDROF,
                    e.span,
                    "immediately dereferencing a reference",
                    "try this",
                    format!("{}", snippet_with_applicability(cx, addrof_target.span, "_", &mut applicability)),
                    applicability,
                );
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for references in expressions that use
    /// auto dereference.
    ///
    /// **Why is this bad?** The reference is a no-op and is automatically
    /// dereferenced by the compiler and makes the code less clear.
    ///
    /// **Example:**
    /// ```rust
    /// struct Point(u32, u32);
    /// let point = Point(30, 20);
    /// let x = (&point).0;
    /// ```
    pub REF_IN_DEREF,
    complexity,
    "Use of reference in auto dereference expression."
}

declare_lint_pass!(RefInDeref => [REF_IN_DEREF]);

impl EarlyLintPass for RefInDeref {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &Expr) {
        if_chain! {
            if let ExprKind::Field(ref object, _) = e.kind;
            if let ExprKind::Paren(ref parened) = object.kind;
            if let ExprKind::AddrOf(_, _, ref inner) = parened.kind;
            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    REF_IN_DEREF,
                    object.span,
                    "Creating a reference that is immediately dereferenced.",
                    "try this",
                    snippet_with_applicability(cx, inner.span, "_", &mut applicability).to_string(),
                    applicability,
                );
            }
        }
    }
}
