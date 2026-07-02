use rustc_ast::BorrowKind;
use rustc_hir::{Expr, ExprKind, TyKind};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{UnnecessaryRef, UnnecessaryRefSuggestion};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `unnecessary_refs` lint checks for unnecessary references.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![warn(unnecessary_refs)]
    ///
    /// fn via_ref(x: *const (i32, i32)) -> *const i32 {
    ///     unsafe { &(*x).0 as *const i32 }
    /// }
    ///
    /// fn main() {
    ///     let x = (0, 1);
    ///     let _r = via_ref(&x);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating unnecessary references is discouraged because it makes code
    /// less explicit and can lead to undefined behavior. Creating a reference
    /// induces aliasing assumptions that the compiler relies on, so an
    /// otherwise-pointless reference can cause undefined behavior; it is also
    /// UB if the reference is unaligned. Avoiding them keeps the code more
    /// explicit and easier to reason about.
    ///
    /// This lint is "allow" by default because promoting it would require a
    /// large amount of churn across the standard library to remove the many
    /// unnecessary references it currently creates.
    pub UNNECESSARY_REFS,
    // FIXME: This should eventually be `Warn`, see the "Explanation" above.
    Allow,
    "creating unnecessary reference is discouraged"
}

declare_lint_pass!(UnnecessaryRefs => [UNNECESSARY_REFS]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryRefs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if let ExprKind::Cast(exp, ty) = expr.kind
            && let ExprKind::AddrOf(BorrowKind::Ref, mutbl, addr_of_exp) = exp.kind
            && let TyKind::Ptr(_) = ty.kind
            && addr_of_exp.is_syntactic_place_expr()
        {
            cx.emit_span_lint(
                UNNECESSARY_REFS,
                expr.span,
                UnnecessaryRef {
                    suggestion: UnnecessaryRefSuggestion {
                        left: expr.span.until(addr_of_exp.span),
                        right: addr_of_exp.span.shrink_to_hi().until(ty.span.shrink_to_hi()),
                        mutbl: mutbl.ptr_str(),
                    },
                },
            );
        }
    }
}
