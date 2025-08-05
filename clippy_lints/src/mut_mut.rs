use clippy_utils::diagnostics::{span_lint, span_lint_hir_and_then};
use clippy_utils::higher;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, AmbigArg, intravisit};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for instances of `mut mut` references.
    ///
    /// ### Why is this bad?
    /// Multiple `mut`s don't add anything meaningful to the
    /// source. This is either a copy'n'paste error, or it shows a fundamental
    /// misunderstanding of references.
    ///
    /// ### Example
    /// ```no_run
    /// # let mut y = 1;
    /// let x = &mut &mut y;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MUT_MUT,
    pedantic,
    "usage of double-mut refs, e.g., `&mut &mut ...`"
}

declare_lint_pass!(MutMut => [MUT_MUT]);

impl<'tcx> LateLintPass<'tcx> for MutMut {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        intravisit::walk_block(&mut MutVisitor { cx }, block);
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'_, AmbigArg>) {
        if let hir::TyKind::Ref(_, mty) = ty.kind
            && mty.mutbl == hir::Mutability::Mut
            && let hir::TyKind::Ref(_, mty) = mty.ty.kind
            && mty.mutbl == hir::Mutability::Mut
            && !ty.span.in_external_macro(cx.sess().source_map())
        {
            span_lint(
                cx,
                MUT_MUT,
                ty.span,
                "generally you want to avoid `&mut &mut _` if possible",
            );
        }
    }
}

pub struct MutVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> intravisit::Visitor<'tcx> for MutVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if expr.span.in_external_macro(self.cx.sess().source_map()) {
            return;
        }

        if let Some(higher::ForLoop { arg, body, .. }) = higher::ForLoop::hir(expr) {
            // A `for` loop lowers to:
            // ```rust
            // match ::std::iter::Iterator::next(&mut iter) {
            // //                                ^^^^
            // ```
            // Let's ignore the generated code.
            intravisit::walk_expr(self, arg);
            intravisit::walk_expr(self, body);
        } else if let hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Mut, e) = expr.kind {
            if let hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Mut, _) = e.kind {
                let mut applicability = Applicability::MaybeIncorrect;
                let sugg = Sugg::hir_with_applicability(self.cx, e, "..", &mut applicability);
                span_lint_hir_and_then(
                    self.cx,
                    MUT_MUT,
                    expr.hir_id,
                    expr.span,
                    "an expression of form `&mut &mut _`",
                    |diag| {
                        diag.span_suggestion(expr.span, "remove the extra `&mut`", sugg, applicability);
                    },
                );
            } else if let ty::Ref(_, ty, hir::Mutability::Mut) = self.cx.typeck_results().expr_ty(e).kind()
                && ty.peel_refs().is_sized(self.cx.tcx, self.cx.typing_env())
            {
                let mut applicability = Applicability::MaybeIncorrect;
                let sugg = Sugg::hir_with_applicability(self.cx, e, "..", &mut applicability).mut_addr_deref();
                span_lint_hir_and_then(
                    self.cx,
                    MUT_MUT,
                    expr.hir_id,
                    expr.span,
                    "this expression mutably borrows a mutable reference",
                    |diag| {
                        diag.span_suggestion(expr.span, "reborrow instead", sugg, applicability);
                    },
                );
            }
        }
    }
}
