use super::{InferTy, Ty, TypeCtor};
use crate::{
    expr::{BinaryOp, CmpOp},
    ty::ApplicationTy,
};

pub(super) fn binary_op_return_ty(op: BinaryOp, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::LogicOp(_) | BinaryOp::CmpOp(_) => Ty::simple(TypeCtor::Bool),
        BinaryOp::Assignment { .. } => Ty::unit(),
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
        BinaryOp::Assignment { op: None } | BinaryOp::CmpOp(CmpOp::Eq { negated: _ }) => {
            match lhs_ty {
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
            }
        }
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
