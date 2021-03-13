//! Helper functions for binary operator type inference.
use chalk_ir::TyVariableKind;
use hir_def::expr::{ArithOp, BinaryOp, CmpOp};

use crate::{Interner, Scalar, Ty, TyKind};

pub(super) fn binary_op_return_ty(op: BinaryOp, lhs_ty: Ty, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(_) | BinaryOp::CmpOp(_) => TyKind::Scalar(Scalar::Bool).intern(&Interner),
        BinaryOp::Assignment { .. } => Ty::unit(),
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => {
            match lhs_ty.interned(&Interner) {
                TyKind::Scalar(Scalar::Int(_))
                | TyKind::Scalar(Scalar::Uint(_))
                | TyKind::Scalar(Scalar::Float(_)) => lhs_ty,
                TyKind::InferenceVar(_, TyVariableKind::Integer)
                | TyKind::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
                _ => TyKind::Unknown.intern(&Interner),
            }
        }
        BinaryOp::ArithOp(_) => match rhs_ty.interned(&Interner) {
            TyKind::Scalar(Scalar::Int(_))
            | TyKind::Scalar(Scalar::Uint(_))
            | TyKind::Scalar(Scalar::Float(_)) => rhs_ty,
            TyKind::InferenceVar(_, TyVariableKind::Integer)
            | TyKind::InferenceVar(_, TyVariableKind::Float) => rhs_ty,
            _ => TyKind::Unknown.intern(&Interner),
        },
    }
}

pub(super) fn binary_op_rhs_expectation(op: BinaryOp, lhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(..) => TyKind::Scalar(Scalar::Bool).intern(&Interner),
        BinaryOp::Assignment { op: None } => lhs_ty,
        BinaryOp::CmpOp(CmpOp::Eq { .. }) => match lhs_ty.interned(&Interner) {
            TyKind::Scalar(_) | TyKind::Str => lhs_ty,
            TyKind::InferenceVar(_, TyVariableKind::Integer)
            | TyKind::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => TyKind::Unknown.intern(&Interner),
        },
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => {
            TyKind::Unknown.intern(&Interner)
        }
        BinaryOp::CmpOp(CmpOp::Ord { .. })
        | BinaryOp::Assignment { op: Some(_) }
        | BinaryOp::ArithOp(_) => match lhs_ty.interned(&Interner) {
            TyKind::Scalar(Scalar::Int(_))
            | TyKind::Scalar(Scalar::Uint(_))
            | TyKind::Scalar(Scalar::Float(_)) => lhs_ty,
            TyKind::InferenceVar(_, TyVariableKind::Integer)
            | TyKind::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => TyKind::Unknown.intern(&Interner),
        },
    }
}
