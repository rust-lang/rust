use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::expr_or_init;
use clippy_utils::ty::is_isize_or_usize;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, FloatTy, Ty};

use super::{utils, CAST_POSSIBLE_TRUNCATION};

fn constant_int(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<u128> {
    if let Some((Constant::Int(c), _)) = constant(cx, cx.typeck_results(), expr) {
        Some(c)
    } else {
        None
    }
}

fn get_constant_bits(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<u64> {
    constant_int(cx, expr).map(|c| u64::from(128 - c.leading_zeros()))
}

fn apply_reductions(cx: &LateContext<'_>, nbits: u64, expr: &Expr<'_>, signed: bool) -> u64 {
    match expr_or_init(cx, expr).kind {
        ExprKind::Cast(inner, _) => apply_reductions(cx, nbits, inner, signed),
        ExprKind::Block(block, _) => block.expr.map_or(nbits, |e| apply_reductions(cx, nbits, e, signed)),
        ExprKind::Binary(op, left, right) => match op.node {
            BinOpKind::Div => {
                apply_reductions(cx, nbits, left, signed)
                    - (if signed {
                        0 // let's be conservative here
                    } else {
                        // by dividing by 1, we remove 0 bits, etc.
                        get_constant_bits(cx, right).map_or(0, |b| b.saturating_sub(1))
                    })
            },
            BinOpKind::Rem | BinOpKind::BitAnd => get_constant_bits(cx, right)
                .unwrap_or(u64::max_value())
                .min(apply_reductions(cx, nbits, left, signed)),
            BinOpKind::Shr => {
                apply_reductions(cx, nbits, left, signed)
                    - constant_int(cx, right).map_or(0, |s| u64::try_from(s).expect("shift too high"))
            },
            _ => nbits,
        },
        ExprKind::MethodCall(method, _, [left, right], _) => {
            if signed {
                return nbits;
            }
            let max_bits = if method.ident.as_str() == "min" {
                get_constant_bits(cx, right)
            } else {
                None
            };
            apply_reductions(cx, nbits, left, signed).min(max_bits.unwrap_or(u64::max_value()))
        },
        ExprKind::MethodCall(method, _, [_, lo, hi], _) => {
            if method.ident.as_str() == "clamp" {
                //FIXME: make this a diagnostic item
                if let (Some(lo_bits), Some(hi_bits)) = (get_constant_bits(cx, lo), get_constant_bits(cx, hi)) {
                    return lo_bits.max(hi_bits);
                }
            }
            nbits
        },
        ExprKind::MethodCall(method, _, [_value], _) => {
            if method.ident.name.as_str() == "signum" {
                0 // do not lint if cast comes from a `signum` function
            } else {
                nbits
            }
        },
        _ => nbits,
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let msg = match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, true) => {
            let from_nbits = apply_reductions(
                cx,
                utils::int_ty_to_nbits(cast_from, cx.tcx),
                cast_expr,
                cast_from.is_signed(),
            );
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
