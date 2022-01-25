use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::is_type_lang_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, LangItem, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for redundant slicing expressions which use the full range, and
    /// do not change the type.
    ///
    /// ### Why is this bad?
    /// It unnecessarily adds complexity to the expression.
    ///
    /// ### Known problems
    /// If the type being sliced has an implementation of `Index<RangeFull>`
    /// that actually changes anything then it can't be removed. However, this would be surprising
    /// to people reading the code and should have a note with it.
    ///
    /// ### Example
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
    #[clippy::version = "1.51.0"]
    pub REDUNDANT_SLICING,
    complexity,
    "redundant slicing of the whole range of a type"
}

declare_lint_pass!(RedundantSlicing => [REDUNDANT_SLICING]);

impl<'tcx> LateLintPass<'tcx> for RedundantSlicing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        let ctxt = expr.span.ctxt();
        if_chain! {
            if let ExprKind::AddrOf(BorrowKind::Ref, mutability, addressee) = expr.kind;
            if addressee.span.ctxt() == ctxt;
            if let ExprKind::Index(indexed, range) = addressee.kind;
            if is_type_lang_item(cx, cx.typeck_results().expr_ty_adjusted(range), LangItem::RangeFull);
            if cx.typeck_results().expr_ty(expr) == cx.typeck_results().expr_ty(indexed);
            then {
                let mut app = Applicability::MachineApplicable;
                let snip = snippet_with_context(cx, indexed.span, ctxt, "..", &mut app).0;

                let (reborrow_str, help_str) = if mutability == Mutability::Mut {
                    // The slice was used to reborrow the mutable reference.
                    ("&mut *", "reborrow the original value instead")
                } else if matches!(
                    get_parent_expr(cx, expr),
                    Some(Expr {
                        kind: ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, _),
                        ..
                    })
                ) {
                    // The slice was used to make a temporary reference.
                    ("&*", "reborrow the original value instead")
                } else {
                    ("", "use the original value instead")
                };

                span_lint_and_sugg(
                    cx,
                    REDUNDANT_SLICING,
                    expr.span,
                    "redundant slicing of the whole range",
                    help_str,
                    format!("{}{}", reborrow_str, snip),
                    app,
                );
            }
        }
    }
}
