use rustc_ast::util::parser::ExprPrecedence;
use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::{LossyProvenancePtr2Int, LossyProvenancePtr2IntSuggestion};
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `lossy_provenance_casts` lint detects an `as` cast between a pointer
    /// and an integer.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(strict_provenance_lints)]
    /// #![warn(lossy_provenance_casts)]
    ///
    /// fn main() {
    ///     let x: u8 = 37;
    ///     let _addr: usize = &x as *const u8 as usize;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// This lint is part of the strict provenance effort, see [issue #95228].
    /// Casting a pointer to an integer is a lossy operation, because beyond
    /// just an *address* a pointer may be associated with a particular
    /// *provenance*. This information is used by the optimiser and for dynamic
    /// analysis/dynamic program verification (e.g. Miri or CHERI platforms).
    ///
    /// Since this cast is lossy, it is considered good style to use the
    /// [`ptr::addr`] method instead, which has a similar effect, but doesn't
    /// "expose" the pointer provenance. This improves optimisation potential.
    /// See the docs of [`ptr::addr`] and [`ptr::expose_provenance`] for more information
    /// about exposing pointer provenance.
    ///
    /// If your code can't comply with strict provenance and needs to expose
    /// the provenance, then there is [`ptr::expose_provenance`] as an escape hatch,
    /// which preserves the behaviour of `as usize` casts while being explicit
    /// about the semantics.
    ///
    /// [issue #95228]: https://github.com/rust-lang/rust/issues/95228
    /// [`ptr::addr`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.addr
    /// [`ptr::expose_provenance`]: https://doc.rust-lang.org/core/primitive.pointer.html#method.expose_provenance
    pub LOSSY_PROVENANCE_CASTS,
    Allow,
    "a lossy pointer to integer cast is used",
    @feature_gate = strict_provenance_lints;
}

declare_lint_pass!(
    /// Lint for `as` casts between a pointer and an integer.
    LossyProvenanceCasts => [LOSSY_PROVENANCE_CASTS]
);

impl<'tcx> LateLintPass<'tcx> for LossyProvenanceCasts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Cast(cast_from_expr, cast_to_hir) = expr.kind else { return };

        let typeck_results = cx.typeck_results;
        // Only lint casts from pointer to integer
        let cast_from_ty = typeck_results.expr_ty(cast_from_expr);
        if !cast_from_ty.is_raw_ptr() {
            return;
        }
        let cast_to_ty = typeck_results.expr_ty(expr);
        if !cast_to_ty.is_integral() {
            return;
        }

        let sugg = expr.span.can_be_used_for_suggestions().then(|| {
            let needs_parens = cx.precedence(cast_from_expr) < ExprPrecedence::Unambiguous;
            let needs_cast = !cast_to_ty.is_usize();
            let cast_span = cast_from_expr.span.shrink_to_hi().to(cast_to_hir.span);
            let expr_span = cast_from_expr.span.shrink_to_lo();
            match (needs_parens, needs_cast) {
                (true, true) => LossyProvenancePtr2IntSuggestion::NeedsParensCast {
                    expr_span,
                    cast_span,
                    cast_to_ty,
                },
                (true, false) => {
                    LossyProvenancePtr2IntSuggestion::NeedsParens { expr_span, cast_span }
                }
                (false, true) => {
                    LossyProvenancePtr2IntSuggestion::NeedsCast { cast_span, cast_to_ty }
                }
                (false, false) => LossyProvenancePtr2IntSuggestion::Other { cast_span },
            }
        });

        let lint = LossyProvenancePtr2Int { cast_from_ty, cast_to_ty, sugg };
        cx.tcx.emit_node_span_lint(LOSSY_PROVENANCE_CASTS, expr.hir_id, expr.span, lint);
    }
}
