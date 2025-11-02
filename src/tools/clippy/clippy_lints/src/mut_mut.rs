use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, AmbigArg, BorrowKind, Expr, ExprKind, HirId, Mutability, TyKind, intravisit};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for instances of `mut mut` references.
    ///
    /// ### Why is this bad?
    /// This is usually just a typo or a misunderstanding of how references work.
    ///
    /// ### Example
    /// ```no_run
    /// let x = &mut &mut 1;
    ///
    /// let mut x = &mut 1;
    /// let y = &mut x;
    ///
    /// fn foo(x: &mut &mut u32) {}
    /// ```
    /// Use instead
    /// ```no_run
    /// let x = &mut 1;
    ///
    /// let mut x = &mut 1;
    /// let y = &mut *x; // reborrow
    ///
    /// fn foo(x: &mut u32) {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MUT_MUT,
    pedantic,
    "usage of double mut-refs, e.g., `&mut &mut ...`"
}

impl_lint_pass!(MutMut => [MUT_MUT]);

#[derive(Default)]
pub(crate) struct MutMut {
    seen_tys: FxHashSet<HirId>,
}

impl<'tcx> LateLintPass<'tcx> for MutMut {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        intravisit::walk_block(&mut MutVisitor { cx }, block);
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'_, AmbigArg>) {
        if let TyKind::Ref(_, mty) = ty.kind
            && mty.mutbl == Mutability::Mut
            && let TyKind::Ref(_, mty2) = mty.ty.kind
            && mty2.mutbl == Mutability::Mut
            && !ty.span.in_external_macro(cx.sess().source_map())
        {
            if self.seen_tys.contains(&ty.hir_id) {
                // we have 2+ `&mut`s, e.g., `&mut &mut &mut x`
                // and we have already flagged on the outermost `&mut &mut (&mut x)`,
                // so don't flag the inner `&mut &mut (x)`
                return;
            }

            // if there is an even longer chain, like `&mut &mut &mut x`, suggest peeling off
            // all extra ones at once
            let (mut t, mut t2) = (mty.ty, mty2.ty);
            let mut many_muts = false;
            loop {
                // this should allow us to remember all the nested types, so that the `contains`
                // above fails faster
                self.seen_tys.insert(t.hir_id);
                if let TyKind::Ref(_, next) = t2.kind
                    && next.mutbl == Mutability::Mut
                {
                    (t, t2) = (t2, next.ty);
                    many_muts = true;
                } else {
                    break;
                }
            }

            let mut applicability = Applicability::MaybeIncorrect;
            let sugg = snippet_with_applicability(cx.sess(), t.span, "..", &mut applicability);
            let suffix = if many_muts { "s" } else { "" };
            span_lint_and_sugg(
                cx,
                MUT_MUT,
                ty.span,
                "a type of form `&mut &mut _`",
                format!("remove the extra `&mut`{suffix}"),
                sugg.to_string(),
                applicability,
            );
        }
    }
}

pub struct MutVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> intravisit::Visitor<'tcx> for MutVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
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
        } else if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, e) = expr.kind {
            if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, e2) = e.kind {
                if !expr.span.eq_ctxt(e.span) {
                    return;
                }

                // if there is an even longer chain, like `&mut &mut &mut x`, suggest peeling off
                // all extra ones at once
                let (mut e, mut e2) = (e, e2);
                let mut many_muts = false;
                loop {
                    if !e.span.eq_ctxt(e2.span) {
                        return;
                    }
                    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, next) = e2.kind {
                        (e, e2) = (e2, next);
                        many_muts = true;
                    } else {
                        break;
                    }
                }

                let mut applicability = Applicability::MaybeIncorrect;
                let sugg = Sugg::hir_with_applicability(self.cx, e, "..", &mut applicability);
                let suffix = if many_muts { "s" } else { "" };
                span_lint_hir_and_then(
                    self.cx,
                    MUT_MUT,
                    expr.hir_id,
                    expr.span,
                    "an expression of form `&mut &mut _`",
                    |diag| {
                        diag.span_suggestion(
                            expr.span,
                            format!("remove the extra `&mut`{suffix}"),
                            sugg,
                            applicability,
                        );
                    },
                );
            } else if let ty::Ref(_, ty, Mutability::Mut) = self.cx.typeck_results().expr_ty(e).kind()
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
