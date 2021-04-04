use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_lang_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::{lint::in_external_macro, ty::TyS};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for redundant slicing expressions which use the full range, and
    /// do not change the type.
    ///
    /// **Why is this bad?** It unnecessarily adds complexity to the expression.
    ///
    /// **Known problems:** If the type being sliced has an implementation of `Index<RangeFull>`
    /// that actually changes anything then it can't be removed. However, this would be surprising
    /// to people reading the code and should have a note with it.
    ///
    /// **Example:**
    ///
    /// ```ignore
    /// fn get_slice(x: &[u32]) -> &[u32] {
    ///     &x[..]
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// fn get_slice(x: &[u32]) -> &[u32] {
    ///     x
    /// }
    /// ```
    pub REDUNDANT_SLICING,
    complexity,
    "redundant slicing of the whole range of a type"
}

declare_lint_pass!(RedundantSlicing => [REDUNDANT_SLICING]);

impl LateLintPass<'_> for RedundantSlicing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if_chain! {
            if let ExprKind::AddrOf(_, _, addressee) = expr.kind;
            if let ExprKind::Index(indexed, range) = addressee.kind;
            if is_type_lang_item(cx, cx.typeck_results().expr_ty_adjusted(range), LangItem::RangeFull);
            if TyS::same_type(cx.typeck_results().expr_ty(expr), cx.typeck_results().expr_ty(indexed));
            then {
                let mut app = Applicability::MachineApplicable;
                let hint = snippet_with_applicability(cx, indexed.span, "..", &mut app).into_owned();

                span_lint_and_sugg(
                    cx,
                    REDUNDANT_SLICING,
                    expr.span,
                    "redundant slicing of the whole range",
                    "use the original slice instead",
                    hint,
                    app,
                );
            }
        }
    }
}
