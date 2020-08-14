//! Helper functions for binary operator type inference.
use hir_def::expr::{ArithOp, BinaryOp, CmpOp};

use super::{InferTy, Ty, TypeCtor};
use crate::ApplicationTy;

pub(super) fn binary_op_return_ty(op: BinaryOp, lhs_ty: Ty, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(_) | BinaryOp::CmpOp(_) => Ty::simple(TypeCtor::Bool),
        BinaryOp::Assignment { .. } => Ty::unit(),
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => match lhs_ty {
            Ty::Apply(ApplicationTy { ctor, .. }) => match ctor {
                TypeCtor::Int(..) | TypeCtor::Float(..) => lhs_ty,
                _ => Ty::Unknown,
            },
            Ty::Infer(InferTy::IntVar(..)) | Ty::Infer(InferTy::FloatVar(..)) => lhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::ArithOp(_) => match rhs_ty {
            Ty::Apply(ApplicationTy { ctor, .. }) => match ctor {
                TypeCtor::Int(..) | TypeCtor::Float(..) => rhs_ty,
                _ => Ty::Unknown,
            },
            Ty::Infer(InferTy::IntVar(..)) | Ty::Infer(InferTy::FloatVar(..)) => rhs_ty,
            _ => Ty::Unknown,
        },
    }
}

pub(super) fn binary_op_rhs_expectation(op: BinaryOp, lhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(..) => Ty::simple(TypeCtor::Bool),
        BinaryOp::Assignment { op: None } => lhs_ty,
        BinaryOp::CmpOp(CmpOp::Eq { .. }) => match lhs_ty {
            Ty::Apply(ApplicationTy { ctor, .. }) => match ctor {
                TypeCtor::Int(..)
                | TypeCtor::Float(..)
                | TypeCtor::Str
                | TypeCtor::Char
                | TypeCtor::Bool => lhs_ty,
                _ => Ty::Unknown,
            },
            Ty::Infer(InferTy::IntVar(..)) | Ty::Infer(InferTy::FloatVar(..)) => lhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::ArithOp(ArithOp::Shl) | BinaryOp::ArithOp(ArithOp::Shr) => Ty::Unknown,
        BinaryOp::CmpOp(CmpOp::Ord { .. })
        | BinaryOp::Assignment { op: Some(_) }
        | BinaryOp::ArithOp(_) => match lhs_ty {
            Ty::Apply(ApplicationTy { ctor, .. }) => match ctor {
                TypeCtor::Int(..) | TypeCtor::Float(..) => lhs_ty,
                _ => Ty::Unknown,
            },
            Ty::Infer(InferTy::IntVar(..)) | Ty::Infer(InferTy::FloatVar(..)) => lhs_ty,
            _ => Ty::Unknown,
        },
    }
}
