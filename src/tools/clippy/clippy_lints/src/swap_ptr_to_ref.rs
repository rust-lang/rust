use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::path_def_id;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{Span, SyntaxContext, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `core::mem::swap` where either parameter is derived from a pointer
    ///
    /// ### Why is this bad?
    /// When at least one parameter to `swap` is derived from a pointer it may overlap with the
    /// other. This would then lead to undefined behavior.
    ///
    /// ### Example
    /// ```no_run
    /// unsafe fn swap(x: &[*mut u32], y: &[*mut u32]) {
    ///     for (&x, &y) in x.iter().zip(y) {
    ///         core::mem::swap(&mut *x, &mut *y);
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// unsafe fn swap(x: &[*mut u32], y: &[*mut u32]) {
    ///     for (&x, &y) in x.iter().zip(y) {
    ///         core::ptr::swap(x, y);
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.63.0"]
    pub SWAP_PTR_TO_REF,
    suspicious,
    "call to `mem::swap` using pointer derived references"
}
declare_lint_pass!(SwapPtrToRef => [SWAP_PTR_TO_REF]);

impl LateLintPass<'_> for SwapPtrToRef {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &Expr<'_>) {
        if let ExprKind::Call(fn_expr, [arg1, arg2]) = e.kind
            && let Some(fn_id) = path_def_id(cx, fn_expr)
            && cx.tcx.is_diagnostic_item(sym::mem_swap, fn_id)
            && let ctxt = e.span.ctxt()
            && let (from_ptr1, arg1_span) = is_ptr_to_ref(cx, arg1, ctxt)
            && let (from_ptr2, arg2_span) = is_ptr_to_ref(cx, arg2, ctxt)
            && (from_ptr1 || from_ptr2)
        {
            span_lint_and_then(
                cx,
                SWAP_PTR_TO_REF,
                e.span,
                "call to `core::mem::swap` with a parameter derived from a raw pointer",
                |diag| {
                    if !((from_ptr1 && arg1_span.is_none()) || (from_ptr2 && arg2_span.is_none())) {
                        let mut app = Applicability::MachineApplicable;
                        let snip1 = snippet_with_context(cx, arg1_span.unwrap_or(arg1.span), ctxt, "..", &mut app).0;
                        let snip2 = snippet_with_context(cx, arg2_span.unwrap_or(arg2.span), ctxt, "..", &mut app).0;
                        diag.span_suggestion(
                            e.span,
                            "use ptr::swap",
                            format!("core::ptr::swap({snip1}, {snip2})"),
                            app,
                        );
                    }
                },
            );
        }
    }
}

/// Checks if the expression converts a mutable pointer to a mutable reference. If it is, also
/// returns the span of the pointer expression if it's suitable for making a suggestion.
fn is_ptr_to_ref(cx: &LateContext<'_>, e: &Expr<'_>, ctxt: SyntaxContext) -> (bool, Option<Span>) {
    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, borrowed_expr) = e.kind
        && let ExprKind::Unary(UnOp::Deref, derefed_expr) = borrowed_expr.kind
        && cx.typeck_results().expr_ty(derefed_expr).is_raw_ptr()
    {
        (
            true,
            (borrowed_expr.span.ctxt() == ctxt || derefed_expr.span.ctxt() == ctxt).then_some(derefed_expr.span),
        )
    } else {
        (false, None)
    }
}
