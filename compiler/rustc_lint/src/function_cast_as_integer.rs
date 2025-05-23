use rustc_hir as hir;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{BytePos, Span};

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `function_casts_as_integer` lint detects cases where users cast a function into an
    /// integer.
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
    /// You should never cast a function directly into an integer but go through
    /// a cast as `fn` first to make it obvious what's going on. It also allows
    /// to prevent confusion with (associated) constants.
    pub FUNCTION_CASTS_AS_INTEGER,
    Warn,
    "Casting a function into an integer",
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
        if matches!(cast_to_ty.kind(), ty::FnDef(..) | ty::FnPtr(..)) {
            return;
        }
        let cast_from_ty = cx.typeck_results().expr_ty(cast_from_expr);
        if matches!(cast_from_ty.kind(), ty::FnDef(..)) {
            cx.tcx.emit_node_span_lint(
                FUNCTION_CASTS_AS_INTEGER,
                expr.hir_id,
                cast_to_expr.span.with_lo(cast_from_expr.span.hi() + BytePos(1)),
                FunctionCastsAsIntegerMsg {
                    sugg: FunctionCastsAsIntegerSugg {
                        suggestion: cast_from_expr.span.shrink_to_hi(),
                        // We get the function pointer to have a nice display.
                        cast_from_ty: cx.typeck_results().expr_ty_adjusted(cast_from_expr),
                        cast_to_ty,
                    },
                },
            );
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_function_casts_as_integer)]
struct FunctionCastsAsIntegerMsg<'tcx> {
    #[subdiagnostic]
    sugg: FunctionCastsAsIntegerSugg<'tcx>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    lint_cast_as_fn,
    code = " as {cast_from_ty}",
    applicability = "machine-applicable",
    style = "verbose"
)]
struct FunctionCastsAsIntegerSugg<'tcx> {
    #[primary_span]
    pub suggestion: Span,
    pub cast_from_ty: Ty<'tcx>,
    pub cast_to_ty: Ty<'tcx>,
}
