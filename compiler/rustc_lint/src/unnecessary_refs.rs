use rustc_ast::BorrowKind;
use rustc_hir::{Expr, ExprKind, TyKind};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{UnnecessaryRefs, UnnecessaryRefsSuggestion};
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
    ///     let x = (0, 1);
    ///     let _r = via_ref(&x);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating unnecessary references is discouraged because it can reduce
    /// readability, introduce performance overhead, and lead to undefined
    /// behavior if the reference is unaligned or uninitialized. Avoiding them
    /// ensures safer and more efficient code.
    pub UNNECESSARY_REFS,
    Warn,
    "creating unecessary reference is discouraged"
}

declare_lint_pass!(UnecessaryRefs => [UNNECESSARY_REFS]);

impl<'tcx> LateLintPass<'tcx> for UnecessaryRefs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if let ExprKind::Cast(exp, ty) = expr.kind
            && let ExprKind::AddrOf(BorrowKind::Ref, mutbl, addr_of_exp) = exp.kind
            && let TyKind::Ptr(_) = ty.kind
        {
            cx.emit_span_lint(
                UNNECESSARY_REFS,
                expr.span,
                UnnecessaryRefs {
                    suggestion: UnnecessaryRefsSuggestion {
                        left: expr.span.until(addr_of_exp.span),
                        right: addr_of_exp.span.shrink_to_hi().until(ty.span.shrink_to_hi()),
                        mutbl: mutbl.ptr_str(),
                    },
                },
            );
        }
    }
}
