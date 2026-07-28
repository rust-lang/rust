use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::walk_span_to_context;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, AmbigArg, BorrowKind, Expr, ExprKind, HirId, Mutability, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::ExpnKind;

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
    skip_id: Option<HirId>,
}

impl<'tcx> LateLintPass<'tcx> for MutMut {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, base) = e.kind
            && let ctxt = e.span.ctxt()
            && ctxt == base.span.ctxt()
        {
            if self.skip_id.replace(base.hir_id) == Some(e.hir_id) {
                return;
            }

            if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, mut base2) = base.kind {
                while let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, next) = base2.kind
                    && ctxt == base2.span.ctxt()
                {
                    base2 = next;
                }
                if !ctxt.in_external_macro(cx.tcx.sess.source_map())
                    && let Some(sp) = walk_span_to_context(base2.span, ctxt)
                {
                    span_lint_and_then(
                        cx,
                        MUT_MUT,
                        e.span.until(sp),
                        "multiple successive mutable borrows",
                        |diag| {
                            diag.span_suggestion_verbose(
                                base.span.until(sp),
                                "make only a single borrow",
                                "",
                                Applicability::MaybeIncorrect,
                            );
                        },
                    );
                }
            } else if let ty::Ref(_, ty, Mutability::Mut) = *cx.typeck_results.expr_ty(base).kind()
                && ty.peel_refs().is_sized(cx.tcx, cx.typing_env())
                && !ctxt.in_external_macro(cx.tcx.sess.source_map())
                // Don't lint on the explicit borrow in for-loop desugarings.
                && !matches!(ctxt.outer_expn_data().kind, ExpnKind::Desugaring(_))
                && let Some(sp) = walk_span_to_context(base.span, ctxt)
            {
                span_lint_and_then(cx, MUT_MUT, e.span.until(sp), "borrow of a mutable reference", |diag| {
                    diag.span_suggestion_verbose(
                        sp.shrink_to_lo(),
                        "reborrow instead",
                        "*",
                        Applicability::MaybeIncorrect,
                    );
                });
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'_, AmbigArg>) {
        if let TyKind::Ref(_, base) = ty.kind
            && base.mutbl.is_mut()
            && let ctxt = ty.span.ctxt()
            && ctxt == base.ty.span.ctxt()
        {
            if self.skip_id.replace(base.ty.hir_id) == Some(ty.hir_id) {
                return;
            }

            if let TyKind::Ref(_, mut base2) = base.ty.kind
                && base2.mutbl.is_mut()
            {
                while let TyKind::Ref(_, next) = base2.ty.kind
                    && next.mutbl.is_mut()
                    && ctxt == base2.ty.span.ctxt()
                {
                    base2 = next;
                }
                if !ctxt.in_external_macro(cx.tcx.sess.source_map())
                    && let Some(sp) = walk_span_to_context(base2.ty.span, ctxt)
                {
                    span_lint_and_then(
                        cx,
                        MUT_MUT,
                        ty.span.until(sp),
                        "multiple successive mutable references",
                        |diag| {
                            diag.span_suggestion_verbose(
                                base.ty.span.until(sp),
                                "use only a single mutable reference",
                                "",
                                Applicability::MaybeIncorrect,
                            );
                        },
                    );
                }
            }
        }
    }
}
