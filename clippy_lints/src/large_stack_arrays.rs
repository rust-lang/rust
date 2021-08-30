use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::snippet;
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, ConstKind};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for local arrays that may be too large.
    ///
    /// ### Why is this bad?
    /// Large local arrays may cause stack overflow.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a = [0u32; 1_000_000];
    /// ```
    pub LARGE_STACK_ARRAYS,
    pedantic,
    "allocating large arrays on stack may cause stack overflow"
}

pub struct LargeStackArrays {
    maximum_allowed_size: u64,
}

impl LargeStackArrays {
    #[must_use]
    pub fn new(maximum_allowed_size: u64) -> Self {
        Self { maximum_allowed_size }
    }
}

impl_lint_pass!(LargeStackArrays => [LARGE_STACK_ARRAYS]);

impl<'tcx> LateLintPass<'tcx> for LargeStackArrays {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::Repeat(_, _) = expr.kind;
            if let ty::Array(element_type, cst) = cx.typeck_results().expr_ty(expr).kind();
            if let ConstKind::Value(ConstValue::Scalar(element_count)) = cst.val;
            if let Ok(element_count) = element_count.to_machine_usize(&cx.tcx);
            if let Ok(element_size) = cx.layout_of(element_type).map(|l| l.size.bytes());
            if self.maximum_allowed_size < element_count * element_size;
            then {
                span_lint_and_help(
                    cx,
                    LARGE_STACK_ARRAYS,
                    expr.span,
                    &format!(
                        "allocating a local array larger than {} bytes",
                        self.maximum_allowed_size
                    ),
                    None,
                    &format!(
                        "consider allocating on the heap with `vec!{}.into_boxed_slice()`",
                        snippet(cx, expr.span, "[...]")
                    ),
                );
            }
        }
    }
}
