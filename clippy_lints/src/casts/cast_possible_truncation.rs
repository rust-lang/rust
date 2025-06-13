use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{get_discriminant_value, is_isize_or_usize};
use clippy_utils::{expr_or_init, sym};
use rustc_abi::IntegerType;
use rustc_errors::{Applicability, Diag};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, FloatTy, Ty};
use rustc_span::Span;

use super::{CAST_ENUM_TRUNCATION, CAST_POSSIBLE_TRUNCATION, utils};

fn constant_int(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<u128> {
    if let Some(Constant::Int(c)) = ConstEvalCtxt::new(cx).eval(expr) {
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
                apply_reductions(cx, nbits, left, signed).saturating_sub(if signed {
                    // let's be conservative here
                    0
                } else {
                    // by dividing by 1, we remove 0 bits, etc.
                    get_constant_bits(cx, right).map_or(0, |b| b.saturating_sub(1))
                })
            },
            BinOpKind::Rem => get_constant_bits(cx, right)
                .unwrap_or(u64::MAX)
                .min(apply_reductions(cx, nbits, left, signed)),
            BinOpKind::BitAnd => get_constant_bits(cx, right)
                .unwrap_or(u64::MAX)
                .min(get_constant_bits(cx, left).unwrap_or(u64::MAX))
                .min(apply_reductions(cx, nbits, right, signed))
                .min(apply_reductions(cx, nbits, left, signed)),
            BinOpKind::Shr => apply_reductions(cx, nbits, left, signed)
                .saturating_sub(constant_int(cx, right).map_or(0, |s| u64::try_from(s).unwrap_or_default())),
            _ => nbits,
        },
        ExprKind::MethodCall(method, left, [right], _) => {
            if signed {
                return nbits;
            }
            let max_bits = if method.ident.name == sym::min {
                get_constant_bits(cx, right)
            } else {
                None
            };
            apply_reductions(cx, nbits, left, signed).min(max_bits.unwrap_or(u64::MAX))
        },
        ExprKind::MethodCall(method, _, [lo, hi], _) => {
            if method.ident.name == sym::clamp
                //FIXME: make this a diagnostic item
                && let (Some(lo_bits), Some(hi_bits)) = (get_constant_bits(cx, lo), get_constant_bits(cx, hi))
            {
                return lo_bits.max(hi_bits);
            }
            nbits
        },
        ExprKind::MethodCall(method, _value, [], _) => {
            if method.ident.name == sym::signum {
                0 // do not lint if cast comes from a `signum` function
            } else {
                nbits
            }
        },
        _ => nbits,
    }
}

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
    cast_to_span: Span,
) {
    let msg = match (cast_from.kind(), utils::int_ty_to_nbits(cx.tcx, cast_to)) {
        (ty::Int(_) | ty::Uint(_), Some(to_nbits)) => {
            let from_nbits = apply_reductions(
                cx,
                utils::int_ty_to_nbits(cx.tcx, cast_from).unwrap(),
                cast_expr,
                cast_from.is_signed(),
            );

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

            format!("casting `{cast_from}` to `{cast_to}` may truncate the value{suffix}",)
        },

        (ty::Adt(def, _), Some(to_nbits)) if def.is_enum() => {
            let (from_nbits, variant) = if let ExprKind::Path(p) = &cast_expr.kind
                && let Res::Def(DefKind::Ctor(..), id) = cx.qpath_res(p, cast_expr.hir_id)
            {
                let i = def.variant_index_with_ctor_id(id);
                let variant = def.variant(i);
                let nbits = utils::enum_value_nbits(get_discriminant_value(cx.tcx, *def, i));
                (nbits, Some(variant))
            } else {
                (utils::enum_ty_to_nbits(*def, cx.tcx), None)
            };

            let cast_from_ptr_size = def.repr().int.is_none_or(|ty| matches!(ty, IntegerType::Pointer(_),));
            let suffix = match (cast_from_ptr_size, is_isize_or_usize(cast_to)) {
                (_, false) if from_nbits > to_nbits => "",
                (false, true) if from_nbits > 64 => "",
                (false, true) if from_nbits > 32 => " on targets with 32-bit wide pointers",
                _ => return,
            };

            if let Some(variant) = variant {
                span_lint(
                    cx,
                    CAST_ENUM_TRUNCATION,
                    expr.span,
                    format!(
                        "casting `{cast_from}::{}` to `{cast_to}` will truncate the value{suffix}",
                        variant.name,
                    ),
                );
                return;
            }
            format!("casting `{cast_from}` to `{cast_to}` may truncate the value{suffix}")
        },

        (ty::Float(_), Some(_)) => {
            format!("casting `{cast_from}` to `{cast_to}` may truncate the value")
        },

        (ty::Float(FloatTy::F64), None) if matches!(cast_to.kind(), &ty::Float(FloatTy::F32)) => {
            "casting `f64` to `f32` may truncate the value".to_string()
        },

        _ => return,
    };

    span_lint_and_then(cx, CAST_POSSIBLE_TRUNCATION, expr.span, msg, |diag| {
        diag.help("if this is intentional allow the lint with `#[allow(clippy::cast_possible_truncation)]` ...");
        if !cast_from.is_floating_point() {
            offer_suggestion(cx, expr, cast_expr, cast_to_span, diag);
        }
    });
}

fn offer_suggestion(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_to_span: Span,
    diag: &mut Diag<'_, ()>,
) {
    let cast_to_snip = snippet(cx, cast_to_span, "..");
    let suggestion = if cast_to_snip == "_" {
        format!("{}.try_into()", Sugg::hir(cx, cast_expr, "..").maybe_paren())
    } else {
        format!("{cast_to_snip}::try_from({})", Sugg::hir(cx, cast_expr, ".."))
    };

    diag.span_suggestion_verbose(
        expr.span,
        "... or use `try_from` and handle the error accordingly",
        suggestion,
        Applicability::Unspecified,
    );
}
