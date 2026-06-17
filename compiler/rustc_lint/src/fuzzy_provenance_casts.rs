use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{LossyProvenanceInt2Ptr, LossyProvenanceInt2PtrSuggestion};
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `fuzzy_provenance_casts` lint detects an `as` cast between an integer
    /// and a pointer.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(strict_provenance_lints)]
    /// #![warn(fuzzy_provenance_casts)]
    ///
    /// fn main() {
    ///     let _dangling = 16_usize as *const u8;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is part of the strict provenance effort, see [issue #95228].
    /// Casting an integer to a pointer is considered bad style, as a pointer
    /// contains, besides the *address* also a *provenance*, indicating what
    /// memory the pointer is allowed to read/write. Casting an integer, which
    /// doesn't have provenance, to a pointer requires the compiler to assign
    /// (guess) provenance. The compiler assigns "all exposed valid" (see the
    /// docs of [`ptr::with_exposed_provenance`] for more information about this
    /// "exposing"). This penalizes the optimiser and is not well suited for
    /// dynamic analysis/dynamic program verification (e.g. Miri or CHERI
    /// platforms).
    ///
    /// It is much better to use [`ptr::with_addr`] instead to specify the
    /// provenance you want. If using this function is not possible because the
    /// code relies on exposed provenance then there is as an escape hatch
    /// [`ptr::with_exposed_provenance`].
    ///
    /// [issue #95228]: https://github.com/rust-lang/rust/issues/95228
    /// [`ptr::with_addr`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.with_addr
    /// [`ptr::with_exposed_provenance`]: https://doc.rust-lang.org/core/ptr/fn.with_exposed_provenance.html
    pub FUZZY_PROVENANCE_CASTS,
    Allow,
    "a fuzzy integer to pointer cast is used",
    @feature_gate = strict_provenance_lints;
}

declare_lint_pass!(
    /// Lint for `as` casts between an integer and a pointer.
    FuzzyProvenanceCasts => [FUZZY_PROVENANCE_CASTS]
);

impl<'tcx> LateLintPass<'tcx> for FuzzyProvenanceCasts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Cast(cast_from_expr, cast_to_hir) = expr.kind else { return };

        let typeck_results = cx.typeck_results;
        // Only lint casts from integer to pointer
        let cast_from_ty = typeck_results.expr_ty(cast_from_expr);
        if !cast_from_ty.is_integral() {
            return;
        }
        let cast_to_ty = typeck_results.expr_ty(expr);
        if !cast_to_ty.is_raw_ptr() {
            return;
        }

        let sugg =
            expr.span.can_be_used_for_suggestions().then(|| LossyProvenanceInt2PtrSuggestion {
                lo: cast_from_expr.span.shrink_to_lo(),
                hi: cast_from_expr.span.shrink_to_hi().to(cast_to_hir.span),
            });
        let lint = LossyProvenanceInt2Ptr { expr_ty: cast_from_ty, cast_ty: cast_to_ty, sugg };
        cx.tcx.emit_node_span_lint(FUZZY_PROVENANCE_CASTS, expr.hir_id, expr.span, lint)
    }
}
