use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::{BAD_BIT_MASK, INEFFECTIVE_BIT_MASK};

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if op.is_comparison() {
        if let Some(cmp_opt) = fetch_int_literal(cx, right) {
            check_compare(cx, left, op, cmp_opt, e.span);
        } else if let Some(cmp_val) = fetch_int_literal(cx, left) {
            check_compare(cx, right, invert_cmp(op), cmp_val, e.span);
        }
    }
}

#[must_use]
fn invert_cmp(cmp: BinOpKind) -> BinOpKind {
    match cmp {
        BinOpKind::Eq => BinOpKind::Eq,
        BinOpKind::Ne => BinOpKind::Ne,
        BinOpKind::Lt => BinOpKind::Gt,
        BinOpKind::Gt => BinOpKind::Lt,
        BinOpKind::Le => BinOpKind::Ge,
        BinOpKind::Ge => BinOpKind::Le,
        _ => BinOpKind::Or, // Dummy
    }
}

fn check_compare<'a>(cx: &LateContext<'a>, bit_op: &Expr<'a>, cmp_op: BinOpKind, cmp_value: u128, span: Span) {
    if let ExprKind::Binary(op, left, right) = &bit_op.kind {
        if op.node != BinOpKind::BitAnd && op.node != BinOpKind::BitOr || is_from_proc_macro(cx, bit_op) {
            return;
        }
        if let Some(mask) = fetch_int_literal(cx, right).or_else(|| fetch_int_literal(cx, left)) {
            check_bit_mask(cx, op.node, cmp_op, mask, cmp_value, span);
        }
    }
}

#[allow(clippy::too_many_lines)]
fn check_bit_mask(
    cx: &LateContext<'_>,
    bit_op: BinOpKind,
    cmp_op: BinOpKind,
    mask_value: u128,
    cmp_value: u128,
    span: Span,
) {
    match cmp_op {
        BinOpKind::Eq | BinOpKind::Ne => match bit_op {
            BinOpKind::BitAnd => {
                if mask_value & cmp_value != cmp_value {
                    if cmp_value != 0 {
                        span_lint(
                            cx,
                            BAD_BIT_MASK,
                            span,
                            format!("incompatible bit mask: `_ & {mask_value}` can never be equal to `{cmp_value}`"),
                        );
                    }
                } else if mask_value == 0 {
                    span_lint(cx, BAD_BIT_MASK, span, "&-masking with zero");
                }
            },
            BinOpKind::BitOr => {
                if mask_value | cmp_value != cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        format!("incompatible bit mask: `_ | {mask_value}` can never be equal to `{cmp_value}`"),
                    );
                }
            },
            _ => (),
        },
        BinOpKind::Lt | BinOpKind::Ge => match bit_op {
            BinOpKind::BitAnd => {
                if mask_value < cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        format!("incompatible bit mask: `_ & {mask_value}` will always be lower than `{cmp_value}`"),
                    );
                } else if mask_value == 0 {
                    span_lint(cx, BAD_BIT_MASK, span, "&-masking with zero");
                }
            },
            BinOpKind::BitOr => {
                if mask_value >= cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        format!("incompatible bit mask: `_ | {mask_value}` will never be lower than `{cmp_value}`"),
                    );
                } else {
                    check_ineffective_lt(cx, span, mask_value, cmp_value, "|");
                }
            },
            BinOpKind::BitXor => check_ineffective_lt(cx, span, mask_value, cmp_value, "^"),
            _ => (),
        },
        BinOpKind::Le | BinOpKind::Gt => match bit_op {
            BinOpKind::BitAnd => {
                if mask_value <= cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        format!("incompatible bit mask: `_ & {mask_value}` will never be higher than `{cmp_value}`"),
                    );
                } else if mask_value == 0 {
                    span_lint(cx, BAD_BIT_MASK, span, "&-masking with zero");
                }
            },
            BinOpKind::BitOr => {
                if mask_value > cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        format!("incompatible bit mask: `_ | {mask_value}` will always be higher than `{cmp_value}`"),
                    );
                } else {
                    check_ineffective_gt(cx, span, mask_value, cmp_value, "|");
                }
            },
            BinOpKind::BitXor => check_ineffective_gt(cx, span, mask_value, cmp_value, "^"),
            _ => (),
        },
        _ => (),
    }
}

fn check_ineffective_lt(cx: &LateContext<'_>, span: Span, m: u128, c: u128, op: &str) {
    if c.is_power_of_two() && m < c {
        span_lint(
            cx,
            INEFFECTIVE_BIT_MASK,
            span,
            format!("ineffective bit mask: `x {op} {m}` compared to `{c}`, is the same as x compared directly"),
        );
    }
}

fn check_ineffective_gt(cx: &LateContext<'_>, span: Span, m: u128, c: u128, op: &str) {
    if (c + 1).is_power_of_two() && m <= c {
        span_lint(
            cx,
            INEFFECTIVE_BIT_MASK,
            span,
            format!("ineffective bit mask: `x {op} {m}` compared to `{c}`, is the same as x compared directly"),
        );
    }
}

fn fetch_int_literal(cx: &LateContext<'_>, lit: &Expr<'_>) -> Option<u128> {
    match ConstEvalCtxt::new(cx).eval(lit)? {
        Constant::Int(n) => Some(n),
        _ => None,
    }
}
