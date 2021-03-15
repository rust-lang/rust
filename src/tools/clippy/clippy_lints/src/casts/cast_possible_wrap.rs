use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

use crate::utils::{is_isize_or_usize, span_lint};

use super::{utils, CAST_POSSIBLE_WRAP};

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    if !(cast_from.is_integral() && cast_to.is_integral()) {
        return;
    }

    let arch_64_suffix = " on targets with 64-bit wide pointers";
    let arch_32_suffix = " on targets with 32-bit wide pointers";
    let cast_unsigned_to_signed = !cast_from.is_signed() && cast_to.is_signed();
    let from_nbits = utils::int_ty_to_nbits(cast_from, cx.tcx);
    let to_nbits = utils::int_ty_to_nbits(cast_to, cx.tcx);

    let (should_lint, suffix) = match (is_isize_or_usize(cast_from), is_isize_or_usize(cast_to)) {
        (true, true) | (false, false) => (to_nbits == from_nbits && cast_unsigned_to_signed, ""),
        (true, false) => (to_nbits <= 32 && cast_unsigned_to_signed, arch_32_suffix),
        (false, true) => (
            cast_unsigned_to_signed,
            if from_nbits == 64 {
                arch_64_suffix
            } else {
                arch_32_suffix
            },
        ),
    };

    if should_lint {
        span_lint(
            cx,
            CAST_POSSIBLE_WRAP,
            expr.span,
            &format!(
                "casting `{}` to `{}` may wrap around the value{}",
                cast_from, cast_to, suffix,
            ),
        );
    }
}
