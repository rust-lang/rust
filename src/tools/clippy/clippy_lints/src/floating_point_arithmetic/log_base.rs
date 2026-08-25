use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::{F32, F64};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::SyntaxContext;
use std::f32::consts as f32_consts;
use std::f64::consts as f64_consts;

use super::SUBOPTIMAL_FLOPS;

// Returns the specialized log method for a given base if base is constant
// and is one of 2, 10 and e
fn get_specialized_log_method(cx: &LateContext<'_>, base: &Expr<'_>, ctxt: SyntaxContext) -> Option<&'static str> {
    if let Some(value) = ConstEvalCtxt::new(cx).eval_local(base, ctxt) {
        if F32(2.0) == value || F64(2.0) == value {
            return Some("log2");
        } else if F32(10.0) == value || F64(10.0) == value {
            return Some("log10");
        } else if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            return Some("ln");
        }
    }

    None
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    if let Some(method) = get_specialized_log_method(cx, &args[0], expr.span.ctxt()) {
        span_lint_and_then(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "logarithm for bases 2, 10 and e can be computed more accurately",
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let recv = Sugg::hir_with_applicability(cx, receiver, "_", &mut app).maybe_paren();
                diag.span_suggestion(expr.span, "consider using", format!("{recv}.{method}()"), app);
            },
        );
    }
}
