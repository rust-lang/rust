use clippy_utils::{
    consts::constant, diagnostics::span_lint_and_sugg, is_from_proc_macro, path_to_local, source::snippet_opt,
};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `x == <float>::INFINITY || x == <float>::NEG_INFINITY`.
    ///
    /// ### Why is this bad?
    /// This should use the dedicated method instead, `is_infinite`.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1.0f32;
    /// if x == f32::INFINITY || x == f32::NEG_INFINITY {}
    /// ```
    /// Use instead:
    /// ```rust
    /// # let x = 1.0f32;
    /// if x.is_infinite() {}
    /// ```
    #[clippy::version = "1.72.0"]
    pub MANUAL_IS_INFINITE,
    style,
    "use dedicated method to check if a float is infinite"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for `x != <float>::INFINITY && x != <float>::NEG_INFINITY`.
    ///
    /// ### Why is this bad?
    /// This should use the dedicated method instead, `is_finite`.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1.0f32;
    /// if x != f32::INFINITY && x != f32::NEG_INFINITY {}
    /// ```
    /// Use instead:
    /// ```rust
    /// # let x = 1.0f32;
    /// if x.is_finite() {}
    /// ```
    #[clippy::version = "1.72.0"]
    pub MANUAL_IS_FINITE,
    style,
    "use dedicated method to check if a float is finite"
}
declare_lint_pass!(ManualFloatMethods => [MANUAL_IS_INFINITE, MANUAL_IS_FINITE]);

impl<'tcx> LateLintPass<'tcx> for ManualFloatMethods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !in_external_macro(cx.sess(), expr.span)
            && let ExprKind::Binary(kind, lhs, rhs) = expr.kind
            && let ExprKind::Binary(lhs_kind, lhs_lhs, lhs_rhs) = lhs.kind
            && let ExprKind::Binary(rhs_kind, rhs_lhs, rhs_rhs) = rhs.kind
            && let (operands, consts) = [lhs_lhs, lhs_rhs, rhs_lhs, rhs_rhs]
                .into_iter()
                .partition::<Vec<&Expr<'_>>, _>(|i| path_to_local(i).is_some())
            && let [first, second] = &*operands
            && let Some([const_1, const_2]) = consts
                .into_iter()
                .map(|i| constant(cx, cx.typeck_results(), i).and_then(|c| c.to_bits()))
                .collect::<Option<Vec<_>>>()
                .as_deref()
            && path_to_local(first).is_some_and(|f| path_to_local(second).is_some_and(|s| f == s))
            && (is_infinity(*const_1) && is_neg_infinity(*const_2)
                || is_neg_infinity(*const_1) && is_infinity(*const_2))
            && let Some(local_snippet) = snippet_opt(cx, first.span)
            && !is_from_proc_macro(cx, expr)
        {
            let (msg, lint, sugg_fn) = match (kind.node, lhs_kind.node, rhs_kind.node) {
                (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Eq) => {
                    ("manually checking if a float is infinite", MANUAL_IS_INFINITE, "is_infinite")
                },
                (BinOpKind::And, BinOpKind::Ne, BinOpKind::Ne) => {
                    ("manually checking if a float is finite", MANUAL_IS_FINITE, "is_finite")
                },
                _ => return,
            };

            span_lint_and_sugg(
                cx,
                lint,
                expr.span,
                msg,
                "try",
                format!("{local_snippet}.{sugg_fn}()"),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn is_infinity(bits: u128) -> bool {
    bits == 0x7f80_0000 || bits == 0x7ff0_0000_0000_0000
}

fn is_neg_infinity(bits: u128) -> bool {
    bits == 0xff80_0000 || bits == 0xfff0_0000_0000_0000
}
