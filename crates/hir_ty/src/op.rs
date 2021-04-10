//! Helper functions for binary operator type inference.
use chalk_ir::TyVariableKind;
use hir_def::expr::{ArithOp, BinaryOp, CmpOp};

use crate::{Interner, Scalar, Ty, TyBuilder, TyKind};

pub(super) fn binary_op_return_ty(op: BinaryOp, lhs_ty: Ty, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(_) | BinaryOp::CmpOp(_) => TyKind::Scalar(Scalar::Bool).intern(&Interner),
        BinaryOp::Assignment { .. } => TyBuilder::unit(),
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => {
            // all integer combinations are valid here
            if matches!(
                lhs_ty.kind(&Interner),
                TyKind::Scalar(Scalar::Int(_))
                    | TyKind::Scalar(Scalar::Uint(_))
                    | TyKind::InferenceVar(_, TyVariableKind::Integer)
            ) && matches!(
                rhs_ty.kind(&Interner),
                TyKind::Scalar(Scalar::Int(_))
                    | TyKind::Scalar(Scalar::Uint(_))
                    | TyKind::InferenceVar(_, TyVariableKind::Integer)
            ) {
                lhs_ty
            } else {
                TyKind::Error.intern(&Interner)
            }
        }
        BinaryOp::ArithOp(_) => match (lhs_ty.kind(&Interner), rhs_ty.kind(&Interner)) {
            // (int, int) | (uint, uint) | (float, float)
            (TyKind::Scalar(Scalar::Int(_)), TyKind::Scalar(Scalar::Int(_)))
            | (TyKind::Scalar(Scalar::Uint(_)), TyKind::Scalar(Scalar::Uint(_)))
            | (TyKind::Scalar(Scalar::Float(_)), TyKind::Scalar(Scalar::Float(_))) => rhs_ty,
            // ({int}, int) | ({int}, uint)
            (TyKind::InferenceVar(_, TyVariableKind::Integer), TyKind::Scalar(Scalar::Int(_)))
            | (TyKind::InferenceVar(_, TyVariableKind::Integer), TyKind::Scalar(Scalar::Uint(_))) => {
                rhs_ty
            }
            // (int, {int}) | (uint, {int})
            (TyKind::Scalar(Scalar::Int(_)), TyKind::InferenceVar(_, TyVariableKind::Integer))
            | (TyKind::Scalar(Scalar::Uint(_)), TyKind::InferenceVar(_, TyVariableKind::Integer)) => {
                lhs_ty
            }
            // ({float} | float)
            (TyKind::InferenceVar(_, TyVariableKind::Float), TyKind::Scalar(Scalar::Float(_))) => {
                rhs_ty
            }
            // (float, {float})
            (TyKind::Scalar(Scalar::Float(_)), TyKind::InferenceVar(_, TyVariableKind::Float)) => {
                lhs_ty
            }
            // ({int}, {int}) | ({float}, {float})
            (
                TyKind::InferenceVar(_, TyVariableKind::Integer),
                TyKind::InferenceVar(_, TyVariableKind::Integer),
            )
            | (
                TyKind::InferenceVar(_, TyVariableKind::Float),
                TyKind::InferenceVar(_, TyVariableKind::Float),
            ) => rhs_ty,
            _ => TyKind::Error.intern(&Interner),
        },
    }
}

pub(super) fn binary_op_rhs_expectation(op: BinaryOp, lhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(..) => TyKind::Scalar(Scalar::Bool).intern(&Interner),
        BinaryOp::Assignment { op: None } => lhs_ty,
        BinaryOp::CmpOp(CmpOp::Eq { .. }) => match lhs_ty.kind(&Interner) {
            TyKind::Scalar(_) | TyKind::Str => lhs_ty,
            TyKind::InferenceVar(_, TyVariableKind::Integer)
            | TyKind::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => TyKind::Error.intern(&Interner),
        },
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => {
            TyKind::Error.intern(&Interner)
        }
        BinaryOp::CmpOp(CmpOp::Ord { .. })
        | BinaryOp::Assignment { op: Some(_) }
        | BinaryOp::ArithOp(_) => match lhs_ty.kind(&Interner) {
            TyKind::Scalar(Scalar::Int(_))
            | TyKind::Scalar(Scalar::Uint(_))
            | TyKind::Scalar(Scalar::Float(_)) => lhs_ty,
            TyKind::InferenceVar(_, TyVariableKind::Integer)
            | TyKind::InferenceVar(_, TyVariableKind::Float) => lhs_ty,
            _ => TyKind::Error.intern(&Interner),
        },
    }
}
