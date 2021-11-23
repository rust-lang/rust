use clippy_utils::consts::{constant_context, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_expr_diagnostic_item;
use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmute calls which would receive a null pointer.
    ///
    /// ### Why is this bad?
    /// Transmuting a null pointer is undefined behavior.
    ///
    /// ### Known problems
    /// Not all cases can be detected at the moment of this writing.
    /// For example, variables which hold a null pointer and are then fed to a `transmute`
    /// call, aren't detectable yet.
    ///
    /// ### Example
    /// ```rust
    /// let null_ref: &u64 = unsafe { std::mem::transmute(0 as *const u64) };
    /// ```
    #[clippy::version = "1.35.0"]
    pub TRANSMUTING_NULL,
    correctness,
    "transmutes from a null pointer to a reference, which is undefined behavior"
}

declare_lint_pass!(TransmutingNull => [TRANSMUTING_NULL]);

const LINT_MSG: &str = "transmuting a known null pointer into a reference";

impl<'tcx> LateLintPass<'tcx> for TransmutingNull {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if_chain! {
            if let ExprKind::Call(func, [arg]) = expr.kind;
            if is_expr_diagnostic_item(cx, func, sym::transmute);

            then {
                // Catching transmute over constants that resolve to `null`.
                let mut const_eval_context = constant_context(cx, cx.typeck_results());
                if_chain! {
                    if let ExprKind::Path(ref _qpath) = arg.kind;
                    if let Some(Constant::RawPtr(x)) = const_eval_context.expr(arg);
                    if x == 0;
                    then {
                        span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG)
                    }
                }

                // Catching:
                // `std::mem::transmute(0 as *const i32)`
                if_chain! {
                    if let ExprKind::Cast(inner_expr, _cast_ty) = arg.kind;
                    if let ExprKind::Lit(ref lit) = inner_expr.kind;
                    if let LitKind::Int(0, _) = lit.node;
                    then {
                        span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG)
                    }
                }

                // Catching:
                // `std::mem::transmute(std::ptr::null::<i32>())`
                if_chain! {
                    if let ExprKind::Call(func1, []) = arg.kind;
                    if is_expr_diagnostic_item(cx, func1, sym::ptr_null);
                    then {
                        span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG)
                    }
                }

                // FIXME:
                // Also catch transmutations of variables which are known nulls.
                // To do this, MIR const propagation seems to be the better tool.
                // Whenever MIR const prop routines are more developed, this will
                // become available. As of this writing (25/03/19) it is not yet.
            }
        }
    }
}
