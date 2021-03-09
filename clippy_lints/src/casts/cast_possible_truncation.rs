use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, FloatTy, Ty};

use crate::utils::{is_isize_or_usize, span_lint};

use super::{utils, CAST_POSSIBLE_TRUNCATION};

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let msg = match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, true) => {
            let from_nbits = utils::int_ty_to_nbits(cast_from, cx.tcx);
            let to_nbits = utils::int_ty_to_nbits(cast_to, cx.tcx);

            let (should_lint, suffix) = match (is_isize_or_usize(cast_from), is_isize_or_usize(cast_to)) {
                (true, true) | (false, false) => (to_nbits < from_nbits, ""),
                (true, false) => (
                    to_nbits <= 32,
                    if to_nbits == 32 {
                        " on targets with 64-bit wide pointers"
                    } else {
                        ""
                    },
                ),
                (false, true) => (from_nbits == 64, " on targets with 32-bit wide pointers"),
            };

            if !should_lint {
                return;
            }

            format!(
                "casting `{}` to `{}` may truncate the value{}",
                cast_from, cast_to, suffix,
            )
        },

        (false, true) => {
            format!("casting `{}` to `{}` may truncate the value", cast_from, cast_to)
        },

        (_, _) => {
            if matches!(cast_from.kind(), &ty::Float(FloatTy::F64))
                && matches!(cast_to.kind(), &ty::Float(FloatTy::F32))
            {
                "casting `f64` to `f32` may truncate the value".to_string()
            } else {
                return;
            }
        },
    };

    span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span, &msg);
}
