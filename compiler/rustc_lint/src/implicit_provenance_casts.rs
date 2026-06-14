use rustc_ast::util::parser::ExprPrecedence;
use rustc_hir as hir;
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{
    ImplicitProvenanceCastsInt2Ptr, ImplicitProvenanceCastsPtr2Int, Int2PtrSuggestion,
    Ptr2IntSuggestion,
};
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `implicit_provenance_casts` lint detects integer-to-pointer and pointer-to-integer casts
    /// that rely on [*Exposed Provenance*][exposed-provenance].
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(strict_provenance_lints)]
    /// #![warn(implicit_provenance_casts)]
    ///
    /// fn main() {
    ///     let x: u8 = 37;
    ///     let addr: usize = &x as *const u8 as usize;
    ///     let _ptr = addr as *const u8;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint exists to help migrate code to [*Strict Provenance* APIs][strict-provenance] where
    /// possible, and make remaining uses of [*Exposed Provenance*][exposed-provenance] explicit.
    /// For more information on pointer provenance, see the [`std::ptr` documentation][provenance].
    ///
    /// Earlier versions of Rust did not have a clear answer how integer-to-pointer and
    /// pointer-to-integer casts interact with provenance. Such casts are now defined to use the
    /// exposed provenance model, but in many cases the code can be updated to strict provenance
    /// APIs, which is preferable as it enables more precise reasoning about unsafe code, both by
    /// humans and by tools like [Miri].
    ///
    /// However, there are situations where exposed provenance is required or following the strict
    /// provenance model requires major refactorings. In those cases, it's still useful to replace
    /// the `as` casts with explicit use of exposed provenance APIs and a comment explaining why
    /// they are needed.
    ///
    /// [provenance]: https://doc.rust-lang.org/core/ptr/index.html#provenance
    /// [strict-provenance]: https://doc.rust-lang.org/core/ptr/index.html#strict-provenance
    /// [exposed-provenance]: https://doc.rust-lang.org/core/ptr/index.html#exposed-provenance
    /// [Miri]: https://github.com/rust-lang/miri
    pub IMPLICIT_PROVENANCE_CASTS,
    Allow,
    "an `as` cast relying on exposed provenance is used",
    @feature_gate = strict_provenance_lints;
}

declare_lint_pass!(
    /// Lint for int2ptr and ptr2int `as` casts.
    ImplicitProvenanceCasts => [IMPLICIT_PROVENANCE_CASTS]
);

impl<'tcx> LateLintPass<'tcx> for ImplicitProvenanceCasts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Cast(cast_from_expr, cast_to_hir) = expr.kind else { return };

        let typeck_results = cx.typeck_results();
        let cast_from_ty = typeck_results.expr_ty(cast_from_expr);
        if cast_from_ty.is_raw_ptr() {
            let cast_to_ty = typeck_results.expr_ty(expr);
            if cast_to_ty.is_integral() {
                lint_ptr2int(cx, expr, cast_from_expr, cast_from_ty, cast_to_hir, cast_to_ty)
            }
        } else if cast_from_ty.is_integral() {
            let cast_to_ty = typeck_results.expr_ty(expr);
            if cast_to_ty.is_raw_ptr() {
                lint_int2ptr(cx, expr, cast_from_expr, cast_from_ty, cast_to_hir, cast_to_ty)
            }
        }
    }
}

fn lint_ptr2int<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    cast_from_expr: &'tcx hir::Expr<'tcx>,
    cast_from_ty: Ty<'tcx>,
    cast_to_hir: &'tcx hir::Ty<'tcx>,
    cast_to_ty: Ty<'tcx>,
) {
    let sugg = expr.span.can_be_used_for_suggestions().then(|| {
        let needs_parens = cx.precedence(cast_from_expr) < ExprPrecedence::Unambiguous;
        let needs_cast = !cast_to_ty.is_usize();
        let cast_span = cast_from_expr.span.shrink_to_hi().to(cast_to_hir.span);
        let expr_span = cast_from_expr.span.shrink_to_lo();
        match (needs_parens, needs_cast) {
            (true, true) => Ptr2IntSuggestion::NeedsParensCast { expr_span, cast_span, cast_to_ty },
            (true, false) => Ptr2IntSuggestion::NeedsParens { expr_span, cast_span },
            (false, true) => Ptr2IntSuggestion::NeedsCast { cast_span, cast_to_ty },
            (false, false) => Ptr2IntSuggestion::Other { cast_span },
        }
    });

    let lint = ImplicitProvenanceCastsPtr2Int { cast_from_ty, cast_to_ty, sugg };
    cx.tcx.emit_node_span_lint(IMPLICIT_PROVENANCE_CASTS, expr.hir_id, expr.span, lint);
}

fn lint_int2ptr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    cast_from_expr: &'tcx hir::Expr<'tcx>,
    cast_from_ty: Ty<'tcx>,
    cast_to_hir: &'tcx hir::Ty<'tcx>,
    cast_to_ty: Ty<'tcx>,
) {
    let sugg = expr.span.can_be_used_for_suggestions().then(|| Int2PtrSuggestion {
        lo: cast_from_expr.span.shrink_to_lo(),
        hi: cast_from_expr.span.shrink_to_hi().to(cast_to_hir.span),
    });
    let lint = ImplicitProvenanceCastsInt2Ptr { expr_ty: cast_from_ty, cast_ty: cast_to_ty, sugg };
    cx.tcx.emit_node_span_lint(IMPLICIT_PROVENANCE_CASTS, expr.hir_id, expr.span, lint)
}
