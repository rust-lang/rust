// Helpers for handling cast expressions.

use rustc_span::bug;

use crate::mir;
use crate::ty::{self, Ty};

/// Valid types for the result of a non-coercion cast
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CastTy<'tcx> {
    /// `iN`, `uN`, and integer inference variables.
    Int,
    /// C-like (fieldless) enums.
    CEnum,
    Bool,
    Char,
    /// `fN` and float inference variables.
    Float,
    /// Function pointers.
    FnPtr,
    /// Raw pointers.
    Ptr(ty::TypeAndMut<'tcx>),
}

impl<'tcx> CastTy<'tcx> {
    /// Returns `Some` for integral/pointer casts.
    /// Casts like unsizing casts will return `None`.
    pub fn from_ty(t: Ty<'tcx>) -> Option<CastTy<'tcx>> {
        match *t.kind() {
            ty::Bool => Some(CastTy::Bool),
            ty::Char => Some(CastTy::Char),
            ty::Int(_) | ty::Uint(_) | ty::Infer(ty::InferTy::IntVar(_)) => Some(CastTy::Int),
            ty::Float(_) | ty::Infer(ty::InferTy::FloatVar(_)) => Some(CastTy::Float),
            ty::Adt(d, _) if d.is_enum() && d.is_payloadfree() => Some(CastTy::CEnum),
            ty::RawPtr(ty, mutbl) => Some(CastTy::Ptr(ty::TypeAndMut { ty, mutbl })),
            ty::FnPtr(..) => Some(CastTy::FnPtr),
            _ => None,
        }
    }

    pub fn is_int_like(self) -> bool {
        match self {
            CastTy::Int | CastTy::CEnum | CastTy::Bool | CastTy::Char => true,
            CastTy::Float | CastTy::FnPtr | CastTy::Ptr(_) => false,
        }
    }
}

/// Returns `mir::CastKind` from the given parameters.
pub fn mir_cast_kind<'tcx>(from_ty: Ty<'tcx>, cast_ty: Ty<'tcx>) -> mir::CastKind {
    let from = CastTy::from_ty(from_ty);
    let cast = CastTy::from_ty(cast_ty);
    let cast_kind = match (from, cast) {
        (Some(from), Some(cast)) if from.is_int_like() && cast.is_int_like() => {
            mir::CastKind::IntToInt
        }
        (Some(CastTy::Ptr(_) | CastTy::FnPtr), Some(CastTy::Int)) => {
            mir::CastKind::PointerExposeProvenance
        }
        (Some(CastTy::Int), Some(CastTy::Ptr(_))) => mir::CastKind::PointerWithExposedProvenance,
        (Some(CastTy::FnPtr), Some(CastTy::Ptr(_))) => mir::CastKind::FnPtrToPtr,

        (Some(CastTy::Float), Some(CastTy::Int)) => mir::CastKind::FloatToInt,
        (Some(CastTy::Int), Some(CastTy::Float)) => mir::CastKind::IntToFloat,
        (Some(CastTy::Float), Some(CastTy::Float)) => mir::CastKind::FloatToFloat,
        (Some(CastTy::Ptr(_)), Some(CastTy::Ptr(_))) => mir::CastKind::PtrToPtr,

        (_, _) => {
            bug!("Attempting to cast non-castable types {:?} and {:?}", from_ty, cast_ty)
        }
    };
    cast_kind
}
