use crate::reference::DEREF_ADDROF;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{get_parent_expr, is_from_proc_macro, is_lint_allowed, is_mutable};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::Mutability;
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `&*(&T)`.
    ///
    /// ### Why is this bad?
    /// Dereferencing and then borrowing a reference value has no effect in most cases.
    ///
    /// ### Known problems
    /// False negative on such code:
    /// ```no_run
    /// let x = &12;
    /// let addr_x = &x as *const _ as usize;
    /// let addr_y = &&*x as *const _ as usize; // assert ok now, and lint triggered.
    ///                                         // But if we fix it, assert will fail.
    /// assert_ne!(addr_x, addr_y);
    /// ```
    ///
    /// ### Example
    /// ```no_run
    /// let s = &String::new();
    ///
    /// let a: &String = &* s;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let s = &String::new();
    /// let a: &String = s;
    /// ```
    #[clippy::version = "1.63.0"]
    pub BORROW_DEREF_REF,
    complexity,
    "deref on an immutable reference returns the same type as itself"
}

declare_lint_pass!(BorrowDerefRef => [BORROW_DEREF_REF]);

impl<'tcx> LateLintPass<'tcx> for BorrowDerefRef {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &rustc_hir::Expr<'tcx>) {
        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, addrof_target) = e.kind
            && let ExprKind::Unary(UnOp::Deref, deref_target) = addrof_target.kind
            && !matches!(deref_target.kind, ExprKind::Unary(UnOp::Deref, ..))
            && !e.span.from_expansion()
            && !deref_target.span.from_expansion()
            && !addrof_target.span.from_expansion()
            && let ref_ty = cx.typeck_results().expr_ty(deref_target)
            && let ty::Ref(_, inner_ty, Mutability::Not) = ref_ty.kind()
            && get_parent_expr(cx, e).is_none_or(|parent| {
                match parent.kind {
                    // `*&*foo` should lint `deref_addrof` instead.
                    ExprKind::Unary(UnOp::Deref, _) => is_lint_allowed(cx, DEREF_ADDROF, parent.hir_id),
                    // `&*foo` creates a distinct temporary from `foo`
                    ExprKind::AddrOf(_, Mutability::Mut, _) => !matches!(
                        deref_target.kind,
                        ExprKind::Path(..)
                            | ExprKind::Field(..)
                            | ExprKind::Index(..)
                            | ExprKind::Unary(UnOp::Deref, ..)
                    ),
                    _ => true,
                }
            })
            && !is_from_proc_macro(cx, e)
            && let e_ty = cx.typeck_results().expr_ty_adjusted(e)
            // check if the reference is coercing to a mutable reference
            && (!matches!(e_ty.kind(), ty::Ref(_, _, Mutability::Mut)) || is_mutable(cx, deref_target))
            && let Some(deref_text) = deref_target.span.get_source_text(cx)
        {
            span_lint_and_then(
                cx,
                BORROW_DEREF_REF,
                e.span,
                "deref on an immutable reference",
                |diag| {
                    diag.span_suggestion(
                        e.span,
                        "if you would like to reborrow, try removing `&*`",
                        deref_text.as_str(),
                        Applicability::MachineApplicable,
                    );

                    // has deref trait -> give 2 help
                    // doesn't have deref trait -> give 1 help
                    if let Some(deref_trait_id) = cx.tcx.lang_items().deref_trait()
                        && !implements_trait(cx, *inner_ty, deref_trait_id, &[])
                    {
                        return;
                    }

                    diag.span_suggestion(
                        e.span,
                        "if you would like to deref, try using `&**`",
                        format!("&**{deref_text}"),
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}
