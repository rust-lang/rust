use clippy_utils::{diagnostics, meets_msrv, msrvs, source, ty};
use rustc_errors::Applicability;
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Finds patterns of unchecked subtraction of [`Duration`] from [`Instant::now()`].
    ///
    /// ### Why is this bad?
    /// Unchecked subtraction could cause underflow on certain platforms, leading to bugs and/or
    /// unintentional panics.
    ///
    /// ### Example
    /// ```rust
    /// let time_passed = Instant::now() - Duration::from_secs(5);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let time_passed = Instant::now().checked_sub(Duration::from_secs(5));
    /// ```
    ///
    /// [`Duration`]: std::time::Duration
    /// [`Instant::now()`]: std::time::Instant::now;
    #[clippy::version = "1.65.0"]
    pub UNCHECKED_DURATION_SUBTRACTION,
    suspicious,
    "finds unchecked subtraction of a 'Duration' from an 'Instant'"
}

pub struct UncheckedDurationSubtraction {
    msrv: Option<RustcVersion>,
}

impl UncheckedDurationSubtraction {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(UncheckedDurationSubtraction => [UNCHECKED_DURATION_SUBTRACTION]);

impl<'tcx> LateLintPass<'tcx> for UncheckedDurationSubtraction {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() || !meets_msrv(self.msrv, msrvs::TRY_FROM) {
            return;
        }

        if_chain! {
            if let ExprKind::Binary(op, lhs, rhs) = expr.kind;

            if let BinOpKind::Sub = op.node;

            // get types of left and right side
            if is_an_instant(cx, lhs);
            if is_a_duration(cx, rhs);

            then {
                print_lint_and_sugg(cx, lhs, rhs, expr)
            }
        }
    }
}

fn is_an_instant(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr);

    match expr_ty.kind() {
        rustc_middle::ty::Adt(def, _) => clippy_utils::match_def_path(cx, def.did(), &clippy_utils::paths::INSTANT),
        _ => false,
    }
}

fn is_a_duration(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr);
    ty::is_type_diagnostic_item(cx, expr_ty, sym::Duration)
}

fn print_lint_and_sugg(cx: &LateContext<'_>, left_expr: &Expr<'_>, right_expr: &Expr<'_>, expr: &Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;

    let left_expr =
        source::snippet_with_applicability(cx, left_expr.span, "std::time::Instant::now()", &mut applicability);
    let right_expr = source::snippet_with_applicability(
        cx,
        right_expr.span,
        "std::time::Duration::from_secs(1)",
        &mut applicability,
    );

    diagnostics::span_lint_and_sugg(
        cx,
        UNCHECKED_DURATION_SUBTRACTION,
        expr.span,
        "unchecked subtraction of a 'Duration' from an 'Instant'",
        "try",
        format!("{}.checked_sub({}).unwrap();", left_expr, right_expr),
        applicability,
    );
}
