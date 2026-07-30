use rustc_ast::BorrowKind;
use rustc_hir::{Expr, ExprKind, TyKind};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{RawBorrowViaReference, RawBorrowViaReferenceSuggestion};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `raw_borrows_via_references` lint checks for references that decay immediately into raw borrows.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![warn(raw_borrows_via_references)]
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
    /// otherwise-pointless reference can cause undefined behavior even when the
    /// reference is never read through. Avoiding them keeps the code more
    /// explicit and easier to reason about.
    ///
    /// See the [Reference] for the full set of validity requirements that
    /// references must uphold.
    ///
    /// [Reference]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// This lint is "allow" by default because it will trigger for a large
    /// amount of existing Rust code.
    /// Eventually it is desired for this to become warn-by-default.
    pub RAW_BORROWS_VIA_REFERENCES,
    // FIXME: This should eventually be `Warn`, see the "Explanation" above.
    Allow,
    "creating raw borrows via references is discouraged"
}

declare_lint_pass!(RawBorrowsViaReferences => [RAW_BORROWS_VIA_REFERENCES]);

impl<'tcx> LateLintPass<'tcx> for RawBorrowsViaReferences {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if let ExprKind::Cast(exp, ty) = expr.kind
            && let ExprKind::AddrOf(BorrowKind::Ref, mutbl, addr_of_exp) = exp.kind
            && let TyKind::Ptr(_) = ty.kind
            && addr_of_exp.is_syntactic_place_expr()
        {
            let suggestion = if let Some(addr_of_span) =
                addr_of_exp.span.find_ancestor_in_same_ctxt(expr.span)
                && let Some(ty_span) = ty.span.find_ancestor_in_same_ctxt(expr.span)
                && expr.span.can_be_used_for_suggestions()
                && addr_of_span.can_be_used_for_suggestions()
                && ty_span.can_be_used_for_suggestions()
            {
                RawBorrowViaReferenceSuggestion::Spanful {
                    left: expr.span.until(addr_of_span),
                    right: addr_of_span.shrink_to_hi().until(ty_span.shrink_to_hi()),
                    mutbl: mutbl.ptr_str(),
                }
            } else {
                RawBorrowViaReferenceSuggestion::Spanless { mutbl: mutbl.ptr_str() }
            };

            cx.emit_span_lint(
                RAW_BORROWS_VIA_REFERENCES,
                expr.span,
                RawBorrowViaReference { suggestion },
            );
        }
    }
}
