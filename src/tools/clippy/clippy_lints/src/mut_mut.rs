use clippy_utils::diagnostics::span_lint;
use clippy_utils::higher;
use rustc_hir as hir;
use rustc_hir::intravisit;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
    /// ```rust
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

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'_>) {
        use rustc_hir::intravisit::Visitor;

        MutVisitor { cx }.visit_ty(ty);
    }
}

pub struct MutVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for MutVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if in_external_macro(self.cx.sess(), expr.span) {
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
                span_lint(
                    self.cx,
                    MUT_MUT,
                    expr.span,
                    "generally you want to avoid `&mut &mut _` if possible",
                );
            } else if let ty::Ref(_, _, hir::Mutability::Mut) = self.cx.typeck_results().expr_ty(e).kind() {
                span_lint(
                    self.cx,
                    MUT_MUT,
                    expr.span,
                    "this expression mutably borrows a mutable reference. Consider reborrowing",
                );
            }
        }
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'_>) {
        if in_external_macro(self.cx.sess(), ty.span) {
            return;
        }

        if let hir::TyKind::Rptr(
            _,
            hir::MutTy {
                ty: pty,
                mutbl: hir::Mutability::Mut,
            },
        ) = ty.kind
        {
            if let hir::TyKind::Rptr(
                _,
                hir::MutTy {
                    mutbl: hir::Mutability::Mut,
                    ..
                },
            ) = pty.kind
            {
                span_lint(
                    self.cx,
                    MUT_MUT,
                    ty.span,
                    "generally you want to avoid `&mut &mut _` if possible",
                );
            }
        }

        intravisit::walk_ty(self, ty);
    }
}
