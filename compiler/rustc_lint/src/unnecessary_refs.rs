use rustc_ast::{BorrowKind, UnOp};
use rustc_hir::{Expr, ExprKind, TyKind};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::UnnecessaryRefs;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `unnecessary_refs` lint checks for unnecessary references.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn via_ref(x: *const (i32, i32)) -> *const i32 {
    ///     unsafe { &(*x).0 as *const i32 }
    /// }
    ///
    /// fn main() {
    ///     let x = 0;
    ///     let _r = via_ref(&x);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating unnecessary references is almost always a mistake.
    pub UNNECESSARY_REFS,
    Warn,
    "creating unecessary reference is discouraged"
}

declare_lint_pass!(UnecessaryRefs => [UNNECESSARY_REFS]);

impl<'tcx> LateLintPass<'tcx> for UnecessaryRefs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if let ExprKind::Cast(exp, ty) = expr.kind
            && let ExprKind::AddrOf(bk, _, exp) = exp.kind
            && matches!(bk, BorrowKind::Ref)
            && let ExprKind::Field(exp, field) = exp.kind
            && let ExprKind::Unary(uo, exp) = exp.kind
            && matches!(uo, UnOp::Deref)
            && let TyKind::Ptr(_) = ty.kind
            && let ExprKind::Path(qpath) = exp.kind
        {
            cx.emit_span_lint(
                UNNECESSARY_REFS,
                expr.span,
                UnnecessaryRefs {
                    span: expr.span,
                    replace: format!(
                        "&raw const (*{}).{}",
                        rustc_hir_pretty::qpath_to_string(&cx.tcx, &qpath),
                        field
                    ),
                },
            );
        }
    }
}
