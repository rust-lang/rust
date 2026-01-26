use clippy_utils::consts::Constant::{F32, F64};
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::numeric_literal;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use std::f32::consts as f32_consts;
use std::f64::consts as f64_consts;

use super::{IMPRECISE_FLOPS, SUBOPTIMAL_FLOPS};

// Returns an integer if the float constant is a whole number and it can be
// converted to an integer without loss of precision. For now we only check
// ranges [-16777215, 16777216) for type f32 as whole number floats outside
// this range are lossy and ambiguous.
#[expect(clippy::cast_possible_truncation)]
fn get_integer_from_float_constant(value: &Constant) -> Option<i32> {
    match value {
        F32(num) if num.fract() == 0.0 => {
            if (-16_777_215.0..16_777_216.0).contains(num) {
                Some(num.round() as i32)
            } else {
                None
            }
        },
        F64(num) if num.fract() == 0.0 => {
            if (-2_147_483_648.0..2_147_483_648.0).contains(num) {
                Some(num.round() as i32)
            } else {
                None
            }
        },
        _ => None,
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    // Check receiver
    if let Some(value) = ConstEvalCtxt::new(cx).eval(receiver)
        && let Some(method) = if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            Some("exp")
        } else if F32(2.0) == value || F64(2.0) == value {
            Some("exp2")
        } else {
            None
        }
    {
        span_lint_and_then(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "exponent for bases 2 and e can be computed more accurately",
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let recv = super::lib::prepare_receiver_sugg(cx, &args[0], &mut app);
                diag.span_suggestion(expr.span, "consider using", format!("{recv}.{method}()"), app);
            },
        );
    }

    // Check argument
    if let Some(value) = ConstEvalCtxt::new(cx).eval(&args[0]) {
        let mut app = Applicability::MachineApplicable;
        let recv = Sugg::hir_with_applicability(cx, receiver, "_", &mut app).maybe_paren();
        let (lint, help, suggestion) = if F32(1.0 / 2.0) == value || F64(1.0 / 2.0) == value {
            (
                SUBOPTIMAL_FLOPS,
                "square-root of a number can be computed more efficiently and accurately",
                format!("{recv}.sqrt()"),
            )
        } else if F32(1.0 / 3.0) == value || F64(1.0 / 3.0) == value {
            (
                IMPRECISE_FLOPS,
                "cube-root of a number can be computed more accurately",
                format!("{recv}.cbrt()"),
            )
        } else if let Some(exponent) = get_integer_from_float_constant(&value) {
            (
                SUBOPTIMAL_FLOPS,
                "exponentiation with integer powers can be computed more efficiently",
                format!(
                    "{recv}.powi({})",
                    numeric_literal::format(&exponent.to_string(), None, false)
                ),
            )
        } else {
            return;
        };

        span_lint_and_sugg(cx, lint, expr.span, help, "consider using", suggestion, app);
    }
}
