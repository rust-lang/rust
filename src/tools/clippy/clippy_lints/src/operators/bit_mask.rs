use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use clippy_utils::source::walk_span_to_context;
use core::cmp::Ordering;
use core::convert::identity;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::{BAD_BIT_MASK, INEFFECTIVE_BIT_MASK};

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    cmp_op: BinOpKind,
    lhs: &'tcx Expr<'_>,
    rhs: &'tcx Expr<'_>,
) {
    let Some(cmp_op) = CmpOp::from_bin_op(cmp_op) else {
        return;
    };

    // Check for a bitwise op compared to a constant.
    let typeck = cx.typeck_results;
    let ecx = ConstEvalCtxt::new(cx);
    let ctxt = e.span.ctxt();
    let Some((cmp_op, bit_op, bit_lhs, bit_rhs, cmp_val)) = try_maybe_swap(
        lhs,
        rhs,
        |lhs, rhs| {
            if let ExprKind::Binary(bit_op, bit_lhs, bit_rhs) = lhs.kind
                && let bit_sp = bit_op.span
                && let Some(bit_op) = BitOp::from_bin_op(bit_op.node)
                && matches!(typeck.expr_ty(rhs).peel_refs().kind(), ty::Uint(_) | ty::Int(_))
                && bit_sp.ctxt() == ctxt
                && let Some(Constant::Int(cmp_val)) = ecx.eval(rhs)
                && walk_span_to_context(rhs.span, ctxt).is_some()
            {
                Some((cmp_op, bit_op, bit_lhs, bit_rhs, cmp_val))
            } else {
                None
            }
        },
        |r| (r.0.swap_operands(), r.1, r.2, r.3, r.4),
    ) else {
        return;
    };

    if !ctxt.in_external_macro(cx.tcx.sess.source_map())
        && let ty = typeck.expr_ty(bit_lhs).peel_refs()
        && matches!(ty.kind(), ty::Uint(_) | ty::Int(_))
        && matches!(typeck.expr_ty(bit_rhs).peel_refs().kind(), ty::Uint(_) | ty::Int(_))
        && let Some(Constant::Int(op_val)) = try_maybe_swap(
            bit_lhs,
            bit_rhs,
            |_, e| ecx.eval(e).filter(|_| walk_span_to_context(e.span, ctxt).is_some()),
            identity,
        )
    {
        let (lint, msg, note) = if matches!(bit_op, BitOp::And) && op_val == 0 {
            let is_eq = cmp_op.matches_ordering(0.cmp(&cmp_val));
            (BAD_BIT_MASK, LintMsg::from_cmp_result(is_eq), NoteMsg::AndZero)
        } else {
            match cmp_op.kind {
                CmpOpKind::Eq => {
                    let help_msg = match bit_op {
                        BitOp::And if op_val & cmp_val != cmp_val => NoteMsg::OpMissingBits,
                        BitOp::Or if op_val | cmp_val != cmp_val => NoteMsg::OpExtraBits,
                        _ => return,
                    };
                    (BAD_BIT_MASK, LintMsg::from_cmp_result(cmp_op.negate), help_msg)
                },
                CmpOpKind::Lt if matches!(ty.kind(), ty::Uint(_)) => match bit_op {
                    BitOp::And if op_val < cmp_val => {
                        (BAD_BIT_MASK, LintMsg::from_cmp_result(!cmp_op.negate), NoteMsg::OpLt)
                    },
                    BitOp::Or if op_val >= cmp_val => {
                        (BAD_BIT_MASK, LintMsg::from_cmp_result(cmp_op.negate), NoteMsg::OpGe)
                    },
                    BitOp::Xor if op_val == 0 && cmp_val == 0 => (
                        BAD_BIT_MASK,
                        LintMsg::from_cmp_result(cmp_op.negate),
                        NoteMsg::XorCmpZero,
                    ),
                    BitOp::Xor | BitOp::Or if u128::MAX.wrapping_shl(cmp_val.trailing_zeros()) & op_val == 0 => (
                        INEFFECTIVE_BIT_MASK,
                        LintMsg::Ineffective,
                        NoteMsg::OpBitsInTrailingZeros,
                    ),
                    _ => return,
                },
                CmpOpKind::Gt if matches!(ty.kind(), ty::Uint(_)) => match bit_op {
                    BitOp::And if op_val <= cmp_val => {
                        (BAD_BIT_MASK, LintMsg::from_cmp_result(!cmp_op.negate), NoteMsg::OpLe)
                    },
                    BitOp::Or if op_val > cmp_val => {
                        (BAD_BIT_MASK, LintMsg::from_cmp_result(cmp_op.negate), NoteMsg::OpGt)
                    },
                    BitOp::Xor | BitOp::Or if u128::MAX.unbounded_shl(cmp_val.trailing_ones()) & op_val == 0 => (
                        INEFFECTIVE_BIT_MASK,
                        LintMsg::Ineffective,
                        NoteMsg::OpBitsInTrailingOnes,
                    ),
                    _ => return,
                },
                CmpOpKind::Lt | CmpOpKind::Gt => return,
            }
        };

        if !is_from_proc_macro(cx, e) {
            #[expect(clippy::collapsible_span_lint_calls)]
            span_lint_and_then(cx, lint, e.span, msg.to_str(), |diag| {
                diag.note(note.to_string(op_val, cmp_val));
            });
        }
    }
}

fn try_maybe_swap<T, R>(
    lhs: &T,
    rhs: &T,
    mut f: impl FnMut(&T, &T) -> Option<R>,
    inv: impl FnOnce(R) -> R,
) -> Option<R> {
    if let Some(x) = f(lhs, rhs) {
        Some(x)
    } else {
        f(rhs, lhs).map(inv)
    }
}

#[derive(Clone, Copy)]
struct CmpOp {
    pub kind: CmpOpKind,
    pub negate: bool,
}
impl CmpOp {
    fn from_bin_op(op: BinOpKind) -> Option<Self> {
        match op {
            BinOpKind::Eq => Some(Self {
                kind: CmpOpKind::Eq,
                negate: false,
            }),
            BinOpKind::Lt => Some(Self {
                kind: CmpOpKind::Lt,
                negate: false,
            }),
            BinOpKind::Le => Some(Self {
                kind: CmpOpKind::Gt,
                negate: true,
            }),
            BinOpKind::Ne => Some(Self {
                kind: CmpOpKind::Eq,
                negate: true,
            }),
            BinOpKind::Ge => Some(Self {
                kind: CmpOpKind::Lt,
                negate: true,
            }),
            BinOpKind::Gt => Some(Self {
                kind: CmpOpKind::Gt,
                negate: false,
            }),
            _ => None,
        }
    }

    fn swap_operands(self) -> Self {
        Self {
            kind: self.kind.swap_operands(),
            negate: self.negate,
        }
    }

    fn matches_ordering(self, ord: Ordering) -> bool {
        let res = matches!(
            (self.kind, ord),
            (CmpOpKind::Lt, Ordering::Less) | (CmpOpKind::Eq, Ordering::Equal) | (CmpOpKind::Gt, Ordering::Greater)
        );
        if self.negate { !res } else { res }
    }
}

#[derive(Clone, Copy)]
enum CmpOpKind {
    Eq,
    Lt,
    Gt,
}
impl CmpOpKind {
    fn swap_operands(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Lt => Self::Gt,
            Self::Gt => Self::Lt,
        }
    }
}

#[derive(Clone, Copy)]
enum BitOp {
    Xor,
    And,
    Or,
}
impl BitOp {
    fn from_bin_op(op: BinOpKind) -> Option<Self> {
        match op {
            BinOpKind::BitXor => Some(Self::Xor),
            BinOpKind::BitAnd => Some(Self::And),
            BinOpKind::BitOr => Some(Self::Or),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
enum NoteMsg {
    AndZero,
    XorCmpZero,
    OpMissingBits,
    OpExtraBits,
    OpLt,
    OpLe,
    OpGt,
    OpGe,
    OpBitsInTrailingZeros,
    OpBitsInTrailingOnes,
}
impl NoteMsg {
    pub fn to_string(self, op_val: u128, cmp_val: u128) -> String {
        match self {
            Self::AndZero => "`_ & 0` is always equal to zero".to_owned(),
            Self::XorCmpZero => "with constants resolved this is `_ ^ 0 < 0`".to_owned(),
            Self::OpMissingBits => {
                format!("`0x{op_val:x}` is missing bits contained in the compared constant `0x{cmp_val:x}`")
            },
            Self::OpExtraBits => {
                format!("`0x{op_val:x}` has bits not contained in the compared constant `0x{cmp_val:x}`")
            },
            Self::OpLt => format!("`0x{op_val:x}` is less than the compared constant `0x{cmp_val:x}`"),
            Self::OpLe => {
                format!("`0x{op_val:x}` is less than or equal to the compared constant `0x{cmp_val:x}`")
            },
            Self::OpGt => format!("`0x{op_val:x}` is greater than the compared constant `0x{cmp_val:x}`"),
            Self::OpGe => {
                format!("`0x{op_val:x}` is greater than or equal to the compared constant `0x{cmp_val:x}`")
            },
            Self::OpBitsInTrailingZeros => format!(
                "`0x{op_val:x}` contains only bits in the trailing zeros of the compared constant `0x{cmp_val:x}`"
            ),
            Self::OpBitsInTrailingOnes => format!(
                "`0x{op_val:x}` contains only bits in the trailing ones of the compared constant `0x{cmp_val:x}`"
            ),
        }
    }
}

#[derive(Clone, Copy)]
enum LintMsg {
    AlwaysTrue,
    AlwaysFalse,
    Ineffective,
}
impl LintMsg {
    fn from_cmp_result(res: bool) -> Self {
        if res { Self::AlwaysTrue } else { Self::AlwaysFalse }
    }

    fn to_str(self) -> &'static str {
        match self {
            Self::AlwaysTrue => "this comparison is always true",
            Self::AlwaysFalse => "this comparison is always false",
            Self::Ineffective => "this comparison's result is unaffected by the bitwise operation",
        }
    }
}
