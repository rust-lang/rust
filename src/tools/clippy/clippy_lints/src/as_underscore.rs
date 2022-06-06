use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Check for the usage of `as _` conversion using inferred type.
    ///
    /// ### Why is this bad?
    /// The conversion might include lossy conversion and dangerous cast that might go
    /// undetected du to the type being inferred.
    ///
    /// The lint is allowed by default as using `_` is less wordy than always specifying the type.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(n: usize) {}
    /// let n: u16 = 256;
    /// foo(n as _);
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo(n: usize) {}
    /// let n: u16 = 256;
    /// foo(n as usize);
    /// ```
    #[clippy::version = "1.63.0"]
    pub AS_UNDERSCORE,
    restriction,
    "detects `as _` conversion"
}
declare_lint_pass!(AsUnderscore => [AS_UNDERSCORE]);

impl<'tcx> LateLintPass<'tcx> for AsUnderscore {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Cast(_, ty) = expr.kind && let TyKind::Infer = ty.kind {

            let ty_resolved = cx.typeck_results().expr_ty(expr);
            if let ty::Error(_) = ty_resolved.kind() {
                span_lint_and_help(
                    cx,
                AS_UNDERSCORE,
                expr.span,
                "using `as _` conversion",
                None,
                "consider giving the type explicitly",
                );
            } else {
            span_lint_and_then(
                cx,
                AS_UNDERSCORE,
                expr.span,
                "using `as _` conversion",
                |diag| {
                    diag.span_suggestion(
                        ty.span,
                        "consider giving the type explicitly",
                        format!("{}", ty_resolved),
                        Applicability::MachineApplicable,
                    );
            }
            );
        }
        }
    }
}
