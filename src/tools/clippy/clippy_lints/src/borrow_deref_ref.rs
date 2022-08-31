use crate::reference::DEREF_ADDROF;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{get_parent_expr, is_lint_allowed};
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::Mutability;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `&*(&T)`.
    ///
    /// ### Why is this bad?
    /// Dereferencing and then borrowing a reference value has no effect in most cases.
    ///
    /// ### Known problems
    /// False negative on such code:
    /// ```
    /// let x = &12;
    /// let addr_x = &x as *const _ as usize;
    /// let addr_y = &&*x as *const _ as usize; // assert ok now, and lint triggered.
    ///                                         // But if we fix it, assert will fail.
    /// assert_ne!(addr_x, addr_y);
    /// ```
    ///
    /// ### Example
    /// ```rust
    /// let s = &String::new();
    ///
    /// let a: &String = &* s;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # let s = &String::new();
    /// let a: &String = s;
    /// ```
    #[clippy::version = "1.63.0"]
    pub BORROW_DEREF_REF,
    complexity,
    "deref on an immutable reference returns the same type as itself"
}

declare_lint_pass!(BorrowDerefRef => [BORROW_DEREF_REF]);

impl LateLintPass<'_> for BorrowDerefRef {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &rustc_hir::Expr<'_>) {
        if_chain! {
            if !e.span.from_expansion();
            if let ExprKind::AddrOf(_, Mutability::Not, addrof_target) = e.kind;
            if !addrof_target.span.from_expansion();
            if let ExprKind::Unary(UnOp::Deref, deref_target) = addrof_target.kind;
            if !deref_target.span.from_expansion();
            if !matches!(deref_target.kind, ExprKind::Unary(UnOp::Deref, ..) );
            let ref_ty = cx.typeck_results().expr_ty(deref_target);
            if let ty::Ref(_, inner_ty, Mutability::Not) = ref_ty.kind();
            then{

                if let Some(parent_expr) = get_parent_expr(cx, e){
                    if matches!(parent_expr.kind, ExprKind::Unary(UnOp::Deref, ..)) &&
                       !is_lint_allowed(cx, DEREF_ADDROF, parent_expr.hir_id) {
                        return;
                    }

                    // modification to `&mut &*x` is different from `&mut x`
                    if matches!(deref_target.kind, ExprKind::Path(..)
                                             | ExprKind::Field(..)
                                             | ExprKind::Index(..)
                                             | ExprKind::Unary(UnOp::Deref, ..))
                     && matches!(parent_expr.kind, ExprKind::AddrOf(_, Mutability::Mut, _)) {
                       return;
                    }
                }

                span_lint_and_then(
                    cx,
                    BORROW_DEREF_REF,
                    e.span,
                    "deref on an immutable reference",
                    |diag| {
                        diag.span_suggestion(
                            e.span,
                            "if you would like to reborrow, try removing `&*`",
                            snippet_opt(cx, deref_target.span).unwrap(),
                            Applicability::MachineApplicable
                        );

                        // has deref trait -> give 2 help
                        // doesn't have deref trait -> give 1 help
                        if let Some(deref_trait_id) = cx.tcx.lang_items().deref_trait(){
                            if !implements_trait(cx, *inner_ty, deref_trait_id, &[]) {
                                return;
                            }
                        }

                        diag.span_suggestion(
                            e.span,
                            "if you would like to deref, try using `&**`",
                            format!(
                                "&**{}",
                                &snippet_opt(cx, deref_target.span).unwrap(),
                             ),
                            Applicability::MaybeIncorrect
                        );

                    }
                );

            }
        }
    }
}
