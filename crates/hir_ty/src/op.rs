//! Helper functions for binary operator type inference.
use chalk_ir::TyVariableKind;
use hir_def::expr::{ArithOp, BinaryOp, CmpOp};

use crate::{Scalar, Ty};

pub(super) fn binary_op_return_ty(op: BinaryOp, lhs_ty: Ty, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(_) | BinaryOp::CmpOp(_) => Ty::Scalar(Scalar::Bool),
        BinaryOp::Assignment { .. } => Ty::unit(),
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => match lhs_ty {
            Ty::Scalar(Scalar::Int(_))
            | Ty::Scalar(Scalar::Uint(_))
            | Ty::Scalar(Scalar::Float(_)) => lhs_ty,
            Ty::InferenceVar(_, TyVariableKind::Integer)
            | Ty::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::ArithOp(_) => match rhs_ty {
            Ty::Scalar(Scalar::Int(_))
            | Ty::Scalar(Scalar::Uint(_))
            | Ty::Scalar(Scalar::Float(_)) => rhs_ty,
            Ty::InferenceVar(_, TyVariableKind::Integer)
            | Ty::InferenceVar(_, TyVariableKind::Float) => rhs_ty,
            _ => Ty::Unknown,
        },
    }
}

pub(super) fn binary_op_rhs_expectation(op: BinaryOp, lhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(..) => Ty::Scalar(Scalar::Bool),
        BinaryOp::Assignment { op: None } => lhs_ty,
        BinaryOp::CmpOp(CmpOp::Eq { .. }) => match lhs_ty {
            Ty::Scalar(_) | Ty::Str => lhs_ty,
            Ty::InferenceVar(_, TyVariableKind::Integer)
            | Ty::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => Ty::Unknown,
        BinaryOp::CmpOp(CmpOp::Ord { .. })
        | BinaryOp::Assignment { op: Some(_) }
        | BinaryOp::ArithOp(_) => match lhs_ty {
            Ty::Scalar(Scalar::Int(_))
            | Ty::Scalar(Scalar::Uint(_))
            | Ty::Scalar(Scalar::Float(_)) => lhs_ty,
            Ty::InferenceVar(_, TyVariableKind::Integer)
            | Ty::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => Ty::Unknown,
        },
    }
}
