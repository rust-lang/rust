use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_from_proc_macro, sym};
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions like `N - x.leading_zeros()` (where `N` is one less than bit width
    /// of `x`) or `x.ilog(2)`, which are manual reimplementations of `x.ilog2()`
    ///
    /// ### Why is this bad?
    /// Manual reimplementations of `ilog2` increase code complexity for little benefit.
    ///
    /// ### Example
    /// ```no_run
    /// let x: u32 = 5;
    /// let log = 31 - x.leading_zeros();
    /// let log = x.ilog(2);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: u32 = 5;
    /// let log = x.ilog2();
    /// let log = x.ilog2();
    /// ```
    #[clippy::version = "1.93.0"]
    pub MANUAL_ILOG2,
    pedantic,
    "manually reimplementing `ilog2`"
}

pub struct ManualIlog2 {
    msrv: Msrv,
}

impl ManualIlog2 {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(ManualIlog2 => [MANUAL_ILOG2]);

impl LateLintPass<'_> for ManualIlog2 {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        match expr.kind {
            // `BIT_WIDTH - 1 - n.leading_zeros()`
            ExprKind::Binary(op, left, right)
                if left.span.eq_ctxt(right.span)
                    && op.node == BinOpKind::Sub
                    && let ExprKind::Lit(lit) = left.kind
                    && let LitKind::Int(Pu128(val), _) = lit.node
                    && let ExprKind::MethodCall(leading_zeros, recv, [], _) = right.kind
                    && leading_zeros.ident.name == sym::leading_zeros
                    && let ty = cx.typeck_results().expr_ty(recv)
                    && let Some(bit_width) = match ty.kind() {
                        ty::Uint(uint_ty) => uint_ty.bit_width(),
                        ty::Int(_) => {
                            // On non-positive integers, `ilog2` would panic, which might be a sign that the author does
                            // in fact want to calculate something different, so stay on the safer side and don't
                            // suggest anything.
                            return;
                        },
                        _ => return,
                    }
                    && val == u128::from(bit_width) - 1
                    && self.msrv.meets(cx, msrvs::ILOG2)
                    && !is_from_proc_macro(cx, expr) =>
            {
                emit(cx, recv, expr);
            },

            // `n.ilog(2)`
            ExprKind::MethodCall(ilog, recv, [two], _)
                if expr.span.eq_ctxt(two.span)
                    && ilog.ident.name == sym::ilog
                    && let ExprKind::Lit(lit) = two.kind
                    && let LitKind::Int(Pu128(2), _) = lit.node
                    && cx.typeck_results().expr_ty_adjusted(recv).is_integral()
                    /* no need to check MSRV here, as `ilog` and `ilog2` were introduced simultaneously */
                    && !is_from_proc_macro(cx, expr) =>
            {
                emit(cx, recv, expr);
            },

            _ => {},
        }
    }
}

fn emit(cx: &LateContext<'_>, recv: &Expr<'_>, full_expr: &Expr<'_>) {
    let mut app = Applicability::MachineApplicable;
    let recv = snippet_with_applicability(cx, recv.span, "_", &mut app);
    span_lint_and_sugg(
        cx,
        MANUAL_ILOG2,
        full_expr.span,
        "manually reimplementing `ilog2`",
        "try",
        format!("{recv}.ilog2()"),
        app,
    );
}
