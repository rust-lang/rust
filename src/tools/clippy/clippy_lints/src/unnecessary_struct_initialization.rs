use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_copy;
use clippy_utils::{get_parent_expr, path_to_local};
use rustc_hir::{BindingAnnotation, Expr, ExprKind, Node, PatKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for initialization of a `struct` by copying a base without setting
    /// any field.
    ///
    /// ### Why is this bad?
    /// Readability suffers from unnecessary struct building.
    ///
    /// ### Example
    /// ```rust
    /// struct S { s: String }
    ///
    /// let a = S { s: String::from("Hello, world!") };
    /// let b = S { ..a };
    /// ```
    /// Use instead:
    /// ```rust
    /// struct S { s: String }
    ///
    /// let a = S { s: String::from("Hello, world!") };
    /// let b = a;
    /// ```
    ///
    /// ### Known Problems
    /// Has false positives when the base is a place expression that cannot be
    /// moved out of, see [#10547](https://github.com/rust-lang/rust-clippy/issues/10547).
    #[clippy::version = "1.70.0"]
    pub UNNECESSARY_STRUCT_INITIALIZATION,
    nursery,
    "struct built from a base that can be written mode concisely"
}
declare_lint_pass!(UnnecessaryStruct => [UNNECESSARY_STRUCT_INITIALIZATION]);

impl LateLintPass<'_> for UnnecessaryStruct {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Struct(_, &[], Some(base)) = expr.kind {
            if let Some(parent) = get_parent_expr(cx, expr) &&
                let parent_ty = cx.typeck_results().expr_ty_adjusted(parent) &&
                parent_ty.is_any_ptr()
            {
                if is_copy(cx, cx.typeck_results().expr_ty(expr)) && path_to_local(base).is_some() {
                    // When the type implements `Copy`, a reference to the new struct works on the
                    // copy. Using the original would borrow it.
                    return;
                }

                if parent_ty.is_mutable_ptr() && !is_mutable(cx, base) {
                    // The original can be used in a mutable reference context only if it is mutable.
                    return;
                }
            }

            // TODO: do not propose to replace *XX if XX is not Copy
            if let ExprKind::Unary(UnOp::Deref, target) = base.kind &&
                matches!(target.kind, ExprKind::Path(..)) &&
                !is_copy(cx, cx.typeck_results().expr_ty(expr))
            {
                // `*base` cannot be used instead of the struct in the general case if it is not Copy.
                return;
            }

            span_lint_and_sugg(
                cx,
                UNNECESSARY_STRUCT_INITIALIZATION,
                expr.span,
                "unnecessary struct building",
                "replace with",
                snippet(cx, base.span, "..").into_owned(),
                rustc_errors::Applicability::MachineApplicable,
            );
        }
    }
}

fn is_mutable(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(hir_id) = path_to_local(expr) &&
        let Node::Pat(pat) = cx.tcx.hir().get(hir_id)
    {
        matches!(pat.kind, PatKind::Binding(BindingAnnotation::MUT, ..))
    } else {
        true
    }
}
