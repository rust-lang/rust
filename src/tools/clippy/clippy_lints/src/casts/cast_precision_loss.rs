use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_isize_or_usize;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, FloatTy, Ty};

use super::{CAST_PRECISION_LOSS, utils};

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let Some(from_nbits) = utils::int_ty_to_nbits(cx.tcx, cast_from) else {
        return;
    };

    // FIXME: handle `f16` and `f128`
    let to_nbits = match cast_to.kind() {
        ty::Float(f @ (FloatTy::F32 | FloatTy::F64)) => f.bit_width(),
        _ => return,
    };

    if !(is_isize_or_usize(cast_from) || from_nbits >= to_nbits) {
        return;
    }

    let cast_to_f64 = to_nbits == 64;
    let mantissa_nbits = if cast_to_f64 { 52 } else { 23 };
    let arch_dependent = is_isize_or_usize(cast_from) && cast_to_f64;
    let arch_dependent_str = "on targets with 64-bit wide pointers ";
    let from_nbits_str = if arch_dependent {
        "64".to_owned()
    } else if is_isize_or_usize(cast_from) {
        // FIXME: handle 16 bits `usize` type
        "32 or 64".to_owned()
    } else {
        from_nbits.to_string()
    };

    span_lint(
        cx,
        CAST_PRECISION_LOSS,
        expr.span,
        format!(
            "casting `{0}` to `{1}` causes a loss of precision {2}(`{0}` is {3} bits wide, \
             but `{1}`'s mantissa is only {4} bits wide)",
            cast_from,
            if cast_to_f64 { "f64" } else { "f32" },
            if arch_dependent { arch_dependent_str } else { "" },
            from_nbits_str,
            mantissa_nbits
        ),
    );
}
