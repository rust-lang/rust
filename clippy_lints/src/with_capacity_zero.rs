use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use clippy_utils::{fn_def_id, is_integer_const, last_path_segment, span_contains_comment, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `with_capacity(0)` on standard library collections and types.
    ///
    /// ### Why is this bad?
    /// Calling `with_capacity(0)` does not allocate any initial capacity and behaves identically
    /// to `new()`. Using `new()` is more idiomatic, concise, and is a `const fn` for these types.
    ///
    /// ### Example
    /// ```rust
    /// let v: Vec<i32> = Vec::with_capacity(0);
    /// let s = String::with_capacity(0);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let v: Vec<i32> = Vec::new();
    /// let s = String::new();
    /// ```
    #[clippy::version = "1.98.0"]
    pub WITH_CAPACITY_ZERO,
    pedantic,
    "calling `with_capacity(0)` which is equivalent to `new()`"
}

declare_lint_pass!(WithCapacityZero => [WITH_CAPACITY_ZERO]);

fn is_target_type(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    let ty = ty.peel_refs();
    ty.is_lang_item(cx, LangItem::String)
        || matches!(
            ty.opt_diag_name(cx),
            Some(
                sym::Vec | sym::HashMap | sym::HashSet | sym::VecDeque | sym::BinaryHeap | sym::PathBuf | sym::OsString
            )
        )
}

impl<'tcx> LateLintPass<'tcx> for WithCapacityZero {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion()
            && let ExprKind::Call(func, [arg]) = expr.kind
            && let Some(def_id) = fn_def_id(cx, expr)
            && cx.tcx.item_name(def_id) == sym::with_capacity
            && is_integer_const(cx, arg, 0)
            && let ExprKind::Path(ref qpath) = func.kind
            && let ty = cx.typeck_results().expr_ty(expr)
            && is_target_type(cx, ty)
        {
            let last_seg = last_path_segment(qpath);
            let sugg_span = last_seg.ident.span.with_hi(expr.span.hi());
            let app = if span_contains_comment(cx, expr.span) {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };

            span_lint_and_then(
                cx,
                WITH_CAPACITY_ZERO,
                expr.span,
                "calling `with_capacity(0)` is equivalent to `new()`",
                |diag| {
                    diag.span_suggestion_verbose(sugg_span, "use `new()` instead", "new()", app);
                },
            );
        }
    }
}
