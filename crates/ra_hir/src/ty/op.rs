use crate::expr::BinaryOp;
use super::{Ty, InferTy};

pub(super) fn binary_op_return_ty(op: BinaryOp, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::BooleanOr
        | BinaryOp::BooleanAnd
        | BinaryOp::EqualityTest
        | BinaryOp::NegatedEqualityTest
        | BinaryOp::LesserEqualTest
        | BinaryOp::GreaterEqualTest
        | BinaryOp::LesserTest
        | BinaryOp::GreaterTest => Ty::Bool,
        BinaryOp::Assignment
        | BinaryOp::AddAssign
        | BinaryOp::SubAssign
        | BinaryOp::DivAssign
        | BinaryOp::MulAssign
        | BinaryOp::RemAssign
        | BinaryOp::ShrAssign
        | BinaryOp::ShlAssign
        | BinaryOp::BitAndAssign
        | BinaryOp::BitOrAssign
        | BinaryOp::BitXorAssign => Ty::unit(),
        BinaryOp::Addition
        | BinaryOp::Subtraction
        | BinaryOp::Multiplication
        | BinaryOp::Division
        | BinaryOp::Remainder
        | BinaryOp::LeftShift
        | BinaryOp::RightShift
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor => match rhs_ty {
            Ty::Int(..)
            | Ty::Float(..)
            | Ty::Infer(InferTy::IntVar(..))
            | Ty::Infer(InferTy::FloatVar(..)) => rhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::RangeRightOpen | BinaryOp::RangeRightClosed => Ty::Unknown,
    }
}

pub(super) fn binary_op_rhs_expectation(op: BinaryOp, lhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::BooleanAnd | BinaryOp::BooleanOr => Ty::Bool,
        BinaryOp::Assignment | BinaryOp::EqualityTest => match lhs_ty {
            Ty::Int(..) | Ty::Float(..) | Ty::Str | Ty::Char | Ty::Bool => lhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::LesserEqualTest
        | BinaryOp::GreaterEqualTest
        | BinaryOp::LesserTest
        | BinaryOp::GreaterTest
        | BinaryOp::AddAssign
        | BinaryOp::SubAssign
        | BinaryOp::DivAssign
        | BinaryOp::MulAssign
        | BinaryOp::RemAssign
        | BinaryOp::ShrAssign
        | BinaryOp::ShlAssign
        | BinaryOp::BitAndAssign
        | BinaryOp::BitOrAssign
        | BinaryOp::BitXorAssign
        | BinaryOp::Addition
        | BinaryOp::Subtraction
        | BinaryOp::Multiplication
        | BinaryOp::Division
        | BinaryOp::Remainder
        | BinaryOp::LeftShift
        | BinaryOp::RightShift
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor => match lhs_ty {
            Ty::Int(..) | Ty::Float(..) => lhs_ty,
            _ => Ty::Unknown,
        },
        _ => Ty::Unknown,
    }
}
