use rustc_hir as hir;
use rustc_middle::ty;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::BytePos;

use crate::lints::{FunctionCastsAsIntegerDiag, FunctionCastsAsIntegerSugg};
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `function_casts_as_integer` lint detects cases where a function item is cast
    /// to an integer.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo() {}
    /// let x = foo as usize;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When casting a function item to an integer, it implicitly creates a
    /// function pointer that will in turn be cast to an integer. By making
    /// it explicit, it improves readability of the code and prevents bugs.
    pub FUNCTION_CASTS_AS_INTEGER,
    Warn,
    "casting a function into an integer",
}

declare_lint_pass!(
    /// Lint for casts of functions into integers.
    FunctionCastsAsInteger => [FUNCTION_CASTS_AS_INTEGER]
);

impl<'tcx> LateLintPass<'tcx> for FunctionCastsAsInteger {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Cast(cast_from_expr, cast_to_expr) = expr.kind else { return };
        let cast_to_ty = cx.typeck_results().expr_ty(expr);
        // Casting to a function (pointer?), so all good.
        //
        // Normally, only casts to integers is possible, but if it ever changed, this condition
        // will likely need to be updated.
        if matches!(cast_to_ty.kind(), ty::FnDef(..) | ty::FnPtr(..) | ty::RawPtr(..)) {
            return;
        }
        let cast_from_ty = cx.typeck_results().expr_ty(cast_from_expr);
        if matches!(cast_from_ty.kind(), ty::FnDef(..)) {
            cx.tcx.emit_node_span_lint(
                FUNCTION_CASTS_AS_INTEGER,
                expr.hir_id,
                cast_to_expr.span.with_lo(cast_from_expr.span.hi() + BytePos(1)),
                FunctionCastsAsIntegerDiag {
                    sugg: FunctionCastsAsIntegerSugg {
                        suggestion: cast_from_expr.span.shrink_to_hi(),
                        cast_to_ty,
                    },
                },
            );
        }
    }
}
