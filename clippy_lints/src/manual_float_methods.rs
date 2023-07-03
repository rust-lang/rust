use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::{is_from_proc_macro, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, Lint, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual `is_infinite` reimplementations
    /// (i.e., `x == <float>::INFINITY || x == <float>::NEG_INFINITY`).
    ///
    /// ### Why is this bad?
    /// The method `is_infinite` is shorter and more readable.
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
    /// Checks for manual `is_finite` reimplementations
    /// (i.e., `x != <float>::INFINITY && x != <float>::NEG_INFINITY`).
    ///
    /// ### Why is this bad?
    /// The method `is_finite` is shorter and more readable.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1.0f32;
    /// if x != f32::INFINITY && x != f32::NEG_INFINITY {}
    /// if x.abs() < f32::INFINITY {}
    /// ```
    /// Use instead:
    /// ```rust
    /// # let x = 1.0f32;
    /// if x.is_finite() {}
    /// if x.is_finite() {}
    /// ```
    #[clippy::version = "1.72.0"]
    pub MANUAL_IS_FINITE,
    style,
    "use dedicated method to check if a float is finite"
}
declare_lint_pass!(ManualFloatMethods => [MANUAL_IS_INFINITE, MANUAL_IS_FINITE]);

#[derive(Clone, Copy)]
enum Variant {
    ManualIsInfinite,
    ManualIsFinite,
}

impl Variant {
    pub fn lint(self) -> &'static Lint {
        match self {
            Self::ManualIsInfinite => MANUAL_IS_INFINITE,
            Self::ManualIsFinite => MANUAL_IS_FINITE,
        }
    }

    pub fn msg(self) -> &'static str {
        match self {
            Self::ManualIsInfinite => "manually checking if a float is infinite",
            Self::ManualIsFinite => "manually checking if a float is finite",
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualFloatMethods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !in_external_macro(cx.sess(), expr.span)
            && (!cx.param_env.is_const() || cx.tcx.features().active(sym!(const_float_classify)))
            && let ExprKind::Binary(kind, lhs, rhs) = expr.kind
            && let ExprKind::Binary(lhs_kind, lhs_lhs, lhs_rhs) = lhs.kind
            && let ExprKind::Binary(rhs_kind, rhs_lhs, rhs_rhs) = rhs.kind
            // Checking all possible scenarios using a function would be a hopeless task, as we have
            // 16 possible alignments of constants/operands. For now, let's use `partition`.
            && let (operands, constants) = [lhs_lhs, lhs_rhs, rhs_lhs, rhs_rhs]
                .into_iter()
                .partition::<Vec<&Expr<'_>>, _>(|i| path_to_local(i).is_some())
            && let [first, second] = &*operands
            && let Some([const_1, const_2]) = constants
                .into_iter()
                .map(|i| constant(cx, cx.typeck_results(), i))
                .collect::<Option<Vec<_>>>()
                .as_deref()
            && path_to_local(first).is_some_and(|f| path_to_local(second).is_some_and(|s| f == s))
            // The actual infinity check, we also allow `NEG_INFINITY` before` INFINITY` just in
            // case somebody does that for some reason
            && (is_infinity(const_1) && is_neg_infinity(const_2)
                || is_neg_infinity(const_1) && is_infinity(const_2))
            && !is_from_proc_macro(cx, expr)
            && let Some(local_snippet) = snippet_opt(cx, first.span)
        {
            let variant = match (kind.node, lhs_kind.node, rhs_kind.node) {
                (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Eq) => Variant::ManualIsInfinite,
                (BinOpKind::And, BinOpKind::Ne, BinOpKind::Ne) => Variant::ManualIsFinite,
                _ => return,
            };

            span_lint_and_then(
                cx,
                variant.lint(),
                expr.span,
                variant.msg(),
                |diag| {
                    match variant {
                        Variant::ManualIsInfinite => {
                            diag.span_suggestion(
                                expr.span,
                                "use the dedicated method instead",
                                format!("{local_snippet}.is_infinite()"),
                                Applicability::MachineApplicable,
                            );
                        },
                        Variant::ManualIsFinite => {
                            // TODO: There's probably some better way to do this, i.e., create
                            // multiple suggestions with notes between each of them
                            diag.span_suggestion_verbose(
                                expr.span,
                                "use the dedicated method instead",
                                format!("{local_snippet}.is_finite()"),
                                Applicability::MaybeIncorrect,
                            )
                            .span_suggestion_verbose(
                                expr.span,
                                "this will alter how it handles NaN; if that is a problem, use instead",
                                format!("{local_snippet}.is_finite() || {local_snippet}.is_nan()"),
                                Applicability::MaybeIncorrect,
                            )
                            .span_suggestion_verbose(
                                expr.span,
                                "or, for conciseness",
                                format!("!{local_snippet}.is_infinite()"),
                                Applicability::MaybeIncorrect,
                            );
                        },
                    }
                },
            );
        }
    }
}

fn is_infinity(constant: &Constant<'_>) -> bool {
    match constant {
        Constant::F32(float) => *float == f32::INFINITY,
        Constant::F64(float) => *float == f64::INFINITY,
        _ => false,
    }
}

fn is_neg_infinity(constant: &Constant<'_>) -> bool {
    match constant {
        Constant::F32(float) => *float == f32::NEG_INFINITY,
        Constant::F64(float) => *float == f64::NEG_INFINITY,
        _ => false,
    }
}
