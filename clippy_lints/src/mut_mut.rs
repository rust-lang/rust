use crate::utils::{higher, span_lint};
use rustc::hir;
use rustc::hir::intravisit;
use rustc::lint::{in_external_macro, LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for instances of `mut mut` references.
    ///
    /// **Why is this bad?** Multiple `mut`s don't add anything meaningful to the
    /// source. This is either a copy'n'paste error, or it shows a fundamental
    /// misunderstanding of references.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = &mut &mut y;
    /// ```
    pub MUT_MUT,
    pedantic,
    "usage of double-mut refs, e.g., `&mut &mut ...`"
}

declare_lint_pass!(MutMut => [MUT_MUT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MutMut {
    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx hir::Block) {
        intravisit::walk_block(&mut MutVisitor { cx }, block);
    }

    fn check_ty(&mut self, cx: &LateContext<'a, 'tcx>, ty: &'tcx hir::Ty) {
        use rustc::hir::intravisit::Visitor;

        MutVisitor { cx }.visit_ty(ty);
    }
}

pub struct MutVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for MutVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if in_external_macro(self.cx.sess(), expr.span) {
            return;
        }

        if let Some((_, arg, body)) = higher::for_loop(expr) {
            // A `for` loop lowers to:
            // ```rust
            // match ::std::iter::Iterator::next(&mut iter) {
            // //                                ^^^^
            // ```
            // Let's ignore the generated code.
            intravisit::walk_expr(self, arg);
            intravisit::walk_expr(self, body);
        } else if let hir::ExprKind::AddrOf(hir::MutMutable, ref e) = expr.node {
            if let hir::ExprKind::AddrOf(hir::MutMutable, _) = e.node {
                span_lint(
                    self.cx,
                    MUT_MUT,
                    expr.span,
                    "generally you want to avoid `&mut &mut _` if possible",
                );
            } else if let ty::Ref(_, _, hir::MutMutable) = self.cx.tables.expr_ty(e).sty {
                span_lint(
                    self.cx,
                    MUT_MUT,
                    expr.span,
                    "this expression mutably borrows a mutable reference. Consider reborrowing",
                );
            }
        }
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        if let hir::TyKind::Rptr(
            _,
            hir::MutTy {
                ty: ref pty,
                mutbl: hir::MutMutable,
            },
        ) = ty.node
        {
            if let hir::TyKind::Rptr(
                _,
                hir::MutTy {
                    mutbl: hir::MutMutable, ..
                },
            ) = pty.node
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
    fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
        intravisit::NestedVisitorMap::None
    }
}
