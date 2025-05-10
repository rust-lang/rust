use hir::{ExprKind, Node, is_range_literal};
use rustc_abi::{Integer, Size};
use rustc_hir::HirId;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::{bug, ty};
use rustc_span::Span;
use {rustc_ast as ast, rustc_attr_data_structures as attrs, rustc_hir as hir};

use crate::LateContext;
use crate::context::LintContext;
use crate::lints::{
    OnlyCastu8ToChar, OverflowingBinHex, OverflowingBinHexSign, OverflowingBinHexSignBitSub,
    OverflowingBinHexSub, OverflowingInt, OverflowingIntHelp, OverflowingLiteral, OverflowingUInt,
    RangeEndpointOutOfRange, UseInclusiveRange,
};
use crate::types::{OVERFLOWING_LITERALS, TypeLimits};

/// Attempts to special-case the overflowing literal lint when it occurs as a range endpoint (`expr..MAX+1`).
/// Returns `true` iff the lint was emitted.
fn lint_overflowing_range_endpoint<'tcx>(
    cx: &LateContext<'tcx>,
    lit: &hir::Lit,
    lit_val: u128,
    max: u128,
    hir_id: HirId,
    lit_span: Span,
    ty: &str,
) -> bool {
    // Look past casts to support cases like `0..256 as u8`
    let (hir_id, span) = if let Node::Expr(par_expr) = cx.tcx.parent_hir_node(hir_id)
        && let ExprKind::Cast(_, _) = par_expr.kind
    {
        (par_expr.hir_id, par_expr.span)
    } else {
        (hir_id, lit_span)
    };

    // We only want to handle exclusive (`..`) ranges,
    // which are represented as `ExprKind::Struct`.
    let Node::ExprField(field) = cx.tcx.parent_hir_node(hir_id) else { return false };
    let Node::Expr(struct_expr) = cx.tcx.parent_hir_node(field.hir_id) else { return false };
    if !is_range_literal(struct_expr) {
        return false;
    };
    let ExprKind::Struct(_, [start, end], _) = &struct_expr.kind else { return false };

    // We can suggest using an inclusive range
    // (`..=`) instead only if it is the `end` that is
    // overflowing and only by 1.
    if !(end.expr.hir_id == hir_id && lit_val - 1 == max) {
        return false;
    };

    use rustc_ast::{LitIntType, LitKind};
    let suffix = match lit.node {
        LitKind::Int(_, LitIntType::Signed(s)) => s.name_str(),
        LitKind::Int(_, LitIntType::Unsigned(s)) => s.name_str(),
        LitKind::Int(_, LitIntType::Unsuffixed) => "",
        _ => bug!(),
    };

    let sub_sugg = if span.lo() == lit_span.lo() {
        let Ok(start) = cx.sess().source_map().span_to_snippet(start.span) else { return false };
        UseInclusiveRange::WithoutParen {
            sugg: struct_expr.span.shrink_to_lo().to(lit_span.shrink_to_hi()),
            start,
            literal: lit_val - 1,
            suffix,
        }
    } else {
        UseInclusiveRange::WithParen {
            eq_sugg: span.shrink_to_lo(),
            lit_sugg: lit_span,
            literal: lit_val - 1,
            suffix,
        }
    };

    cx.emit_span_lint(
        OVERFLOWING_LITERALS,
        struct_expr.span,
        RangeEndpointOutOfRange { ty, sub: sub_sugg },
    );

    // We've just emitted a lint, special cased for `(...)..MAX+1` ranges,
    // return `true` so the callers don't also emit a lint
    true
}

// For `isize` & `usize`, be conservative with the warnings, so that the
// warnings are consistent between 32- and 64-bit platforms.
pub(crate) fn int_ty_range(int_ty: ty::IntTy) -> (i128, i128) {
    match int_ty {
        ty::IntTy::Isize => (i64::MIN.into(), i64::MAX.into()),
        ty::IntTy::I8 => (i8::MIN.into(), i8::MAX.into()),
        ty::IntTy::I16 => (i16::MIN.into(), i16::MAX.into()),
        ty::IntTy::I32 => (i32::MIN.into(), i32::MAX.into()),
        ty::IntTy::I64 => (i64::MIN.into(), i64::MAX.into()),
        ty::IntTy::I128 => (i128::MIN, i128::MAX),
    }
}

pub(crate) fn uint_ty_range(uint_ty: ty::UintTy) -> (u128, u128) {
    let max = match uint_ty {
        ty::UintTy::Usize => u64::MAX.into(),
        ty::UintTy::U8 => u8::MAX.into(),
        ty::UintTy::U16 => u16::MAX.into(),
        ty::UintTy::U32 => u32::MAX.into(),
        ty::UintTy::U64 => u64::MAX.into(),
        ty::UintTy::U128 => u128::MAX,
    };
    (0, max)
}

fn get_bin_hex_repr(cx: &LateContext<'_>, lit: &hir::Lit) -> Option<String> {
    let src = cx.sess().source_map().span_to_snippet(lit.span).ok()?;
    let firstch = src.chars().next()?;

    if firstch == '0' {
        match src.chars().nth(1) {
            Some('x' | 'b') => return Some(src),
            _ => return None,
        }
    }

    None
}

fn report_bin_hex_error(
    cx: &LateContext<'_>,
    hir_id: HirId,
    span: Span,
    ty: attrs::IntType,
    size: Size,
    repr_str: String,
    val: u128,
    negative: bool,
) {
    let (t, actually) = match ty {
        attrs::IntType::SignedInt(t) => {
            let actually = if negative { -(size.sign_extend(val)) } else { size.sign_extend(val) };
            (t.name_str(), actually.to_string())
        }
        attrs::IntType::UnsignedInt(t) => {
            let actually = size.truncate(val);
            (t.name_str(), actually.to_string())
        }
    };
    let sign =
        if negative { OverflowingBinHexSign::Negative } else { OverflowingBinHexSign::Positive };
    let sub = get_type_suggestion(cx.typeck_results().node_type(hir_id), val, negative).map(
        |suggestion_ty| {
            if let Some(pos) = repr_str.chars().position(|c| c == 'i' || c == 'u') {
                let (sans_suffix, _) = repr_str.split_at(pos);
                OverflowingBinHexSub::Suggestion { span, suggestion_ty, sans_suffix }
            } else {
                OverflowingBinHexSub::Help { suggestion_ty }
            }
        },
    );
    let sign_bit_sub = (!negative)
        .then(|| {
            let ty::Int(int_ty) = cx.typeck_results().node_type(hir_id).kind() else {
                return None;
            };

            let Some(bit_width) = int_ty.bit_width() else {
                return None; // isize case
            };

            // Skip if sign bit is not set
            if (val & (1 << (bit_width - 1))) == 0 {
                return None;
            }

            let lit_no_suffix =
                if let Some(pos) = repr_str.chars().position(|c| c == 'i' || c == 'u') {
                    repr_str.split_at(pos).0
                } else {
                    &repr_str
                };

            Some(OverflowingBinHexSignBitSub {
                span,
                lit_no_suffix,
                negative_val: actually.clone(),
                int_ty: int_ty.name_str(),
                uint_ty: Integer::fit_unsigned(val).uint_ty_str(),
            })
        })
        .flatten();

    cx.emit_span_lint(
        OVERFLOWING_LITERALS,
        span,
        OverflowingBinHex {
            ty: t,
            lit: repr_str.clone(),
            dec: val,
            actually,
            sign,
            sub,
            sign_bit_sub,
        },
    )
}

// Find the "next" fitting integer and return a suggestion string
//
// No suggestion is offered for `{i,u}size`. Otherwise, we try to suggest an equal-sized type.
fn get_type_suggestion(t: Ty<'_>, val: u128, negative: bool) -> Option<&'static str> {
    match t.kind() {
        ty::Uint(ty::UintTy::Usize) | ty::Int(ty::IntTy::Isize) => None,
        ty::Uint(_) => Some(Integer::fit_unsigned(val).uint_ty_str()),
        ty::Int(_) => {
            let signed = literal_to_i128(val, negative).map(Integer::fit_signed);
            if negative {
                signed.map(Integer::int_ty_str)
            } else {
                let unsigned = Integer::fit_unsigned(val);
                Some(if let Some(signed) = signed {
                    if unsigned.size() < signed.size() {
                        unsigned.uint_ty_str()
                    } else {
                        signed.int_ty_str()
                    }
                } else {
                    unsigned.uint_ty_str()
                })
            }
        }
        _ => None,
    }
}

fn literal_to_i128(val: u128, negative: bool) -> Option<i128> {
    if negative {
        (val <= i128::MAX as u128 + 1).then(|| val.wrapping_neg() as i128)
    } else {
        val.try_into().ok()
    }
}

fn lint_int_literal<'tcx>(
    cx: &LateContext<'tcx>,
    type_limits: &TypeLimits,
    hir_id: HirId,
    span: Span,
    lit: &hir::Lit,
    t: ty::IntTy,
    v: u128,
) {
    let int_type = t.normalize(cx.sess().target.pointer_width);
    let (min, max) = int_ty_range(int_type);
    let max = max as u128;
    let negative = type_limits.negated_expr_id == Some(hir_id);

    // Detect literal value out of range [min, max] inclusive
    // avoiding use of -min to prevent overflow/panic
    if (negative && v > max + 1) || (!negative && v > max) {
        if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
            report_bin_hex_error(
                cx,
                hir_id,
                span,
                attrs::IntType::SignedInt(ty::ast_int_ty(t)),
                Integer::from_int_ty(cx, t).size(),
                repr_str,
                v,
                negative,
            );
            return;
        }

        if lint_overflowing_range_endpoint(cx, lit, v, max, hir_id, span, t.name_str()) {
            // The overflowing literal lint was emitted by `lint_overflowing_range_endpoint`.
            return;
        }

        let span = if negative { type_limits.negated_expr_span.unwrap() } else { span };
        let lit = cx
            .sess()
            .source_map()
            .span_to_snippet(span)
            .unwrap_or_else(|_| if negative { format!("-{v}") } else { v.to_string() });
        let help = get_type_suggestion(cx.typeck_results().node_type(hir_id), v, negative)
            .map(|suggestion_ty| OverflowingIntHelp { suggestion_ty });

        cx.emit_span_lint(
            OVERFLOWING_LITERALS,
            span,
            OverflowingInt { ty: t.name_str(), lit, min, max, help },
        );
    }
}

fn lint_uint_literal<'tcx>(
    cx: &LateContext<'tcx>,
    hir_id: HirId,
    span: Span,
    lit: &hir::Lit,
    t: ty::UintTy,
) {
    let uint_type = t.normalize(cx.sess().target.pointer_width);
    let (min, max) = uint_ty_range(uint_type);
    let lit_val: u128 = match lit.node {
        // _v is u8, within range by definition
        ast::LitKind::Byte(_v) => return,
        ast::LitKind::Int(v, _) => v.get(),
        _ => bug!(),
    };

    if lit_val < min || lit_val > max {
        if let Node::Expr(par_e) = cx.tcx.parent_hir_node(hir_id) {
            match par_e.kind {
                hir::ExprKind::Cast(..) => {
                    if let ty::Char = cx.typeck_results().expr_ty(par_e).kind() {
                        cx.emit_span_lint(
                            OVERFLOWING_LITERALS,
                            par_e.span,
                            OnlyCastu8ToChar { span: par_e.span, literal: lit_val },
                        );
                        return;
                    }
                }
                _ => {}
            }
        }
        if lint_overflowing_range_endpoint(cx, lit, lit_val, max, hir_id, span, t.name_str()) {
            // The overflowing literal lint was emitted by `lint_overflowing_range_endpoint`.
            return;
        }
        if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
            report_bin_hex_error(
                cx,
                hir_id,
                span,
                attrs::IntType::UnsignedInt(ty::ast_uint_ty(t)),
                Integer::from_uint_ty(cx, t).size(),
                repr_str,
                lit_val,
                false,
            );
            return;
        }
        cx.emit_span_lint(
            OVERFLOWING_LITERALS,
            span,
            OverflowingUInt {
                ty: t.name_str(),
                lit: cx
                    .sess()
                    .source_map()
                    .span_to_snippet(lit.span)
                    .unwrap_or_else(|_| lit_val.to_string()),
                min,
                max,
            },
        );
    }
}

pub(crate) fn lint_literal<'tcx>(
    cx: &LateContext<'tcx>,
    type_limits: &TypeLimits,
    hir_id: HirId,
    span: Span,
    lit: &hir::Lit,
    negated: bool,
) {
    match *cx.typeck_results().node_type(hir_id).kind() {
        ty::Int(t) => {
            match lit.node {
                ast::LitKind::Int(v, ast::LitIntType::Signed(_) | ast::LitIntType::Unsuffixed) => {
                    lint_int_literal(cx, type_limits, hir_id, span, lit, t, v.get())
                }
                _ => bug!(),
            };
        }
        ty::Uint(t) => {
            assert!(!negated);
            lint_uint_literal(cx, hir_id, span, lit, t)
        }
        ty::Float(t) => {
            let (is_infinite, sym) = match lit.node {
                ast::LitKind::Float(v, _) => match t {
                    // FIXME(f16_f128): add this check once `is_infinite` is reliable (ABI
                    // issues resolved).
                    ty::FloatTy::F16 => (Ok(false), v),
                    ty::FloatTy::F32 => (v.as_str().parse().map(f32::is_infinite), v),
                    ty::FloatTy::F64 => (v.as_str().parse().map(f64::is_infinite), v),
                    ty::FloatTy::F128 => (Ok(false), v),
                },
                _ => bug!(),
            };
            if is_infinite == Ok(true) {
                cx.emit_span_lint(
                    OVERFLOWING_LITERALS,
                    span,
                    OverflowingLiteral {
                        ty: t.name_str(),
                        lit: cx
                            .sess()
                            .source_map()
                            .span_to_snippet(lit.span)
                            .unwrap_or_else(|_| sym.to_string()),
                    },
                );
            }
        }
        _ => {}
    }
}
