// Helpers for handling cast expressions, used in both
// typeck and codegen.

use rustc_macros::{HashStable, TyDecodable, TyEncodable};

use crate::mir;
use crate::ty::{self, Ty};

/// Types that are represented as ints.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntTy {
    U(ty::UintTy),
    I,
    CEnum,
    Bool,
    Char,
}

impl IntTy {
    pub fn is_signed(self) -> bool {
        matches!(self, Self::I)
    }
}

// Valid types for the result of a non-coercion cast
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CastTy<'tcx> {
    /// Various types that are represented as ints and handled mostly
    /// in the same way, merged for easier matching.
    Int(IntTy),
    /// Floating-point types.
    Float,
    /// Function pointers.
    FnPtr,
    /// Raw pointers.
    Ptr(ty::TypeAndMut<'tcx>),
}

/// Cast Kind. See [RFC 401](https://rust-lang.github.io/rfcs/0401-coercions.html)
/// (or rustc_hir_analysis/check/cast.rs).
#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum CastKind {
    PtrPtrCast,
    PtrAddrCast,
    AddrPtrCast,
    NumericCast,
    EnumCast,
    PrimIntCast,
    U8CharCast,
    ArrayPtrCast,
    FnPtrPtrCast,
    FnPtrAddrCast,
}

impl<'tcx> CastTy<'tcx> {
    /// Returns `Some` for integral/pointer casts.
    /// Casts like unsizing casts will return `None`.
    pub fn from_ty(t: Ty<'tcx>) -> Option<CastTy<'tcx>> {
        match *t.kind() {
            ty::Bool => Some(CastTy::Int(IntTy::Bool)),
            ty::Char => Some(CastTy::Int(IntTy::Char)),
            ty::Int(_) => Some(CastTy::Int(IntTy::I)),
            ty::Infer(ty::InferTy::IntVar(_)) => Some(CastTy::Int(IntTy::I)),
            ty::Infer(ty::InferTy::FloatVar(_)) => Some(CastTy::Float),
            ty::Uint(u) => Some(CastTy::Int(IntTy::U(u))),
            ty::Float(_) => Some(CastTy::Float),
            ty::Adt(d, _) if d.is_enum() && d.is_payloadfree() => Some(CastTy::Int(IntTy::CEnum)),
            ty::RawPtr(ty, mutbl) => Some(CastTy::Ptr(ty::TypeAndMut { ty, mutbl })),
            ty::FnPtr(..) => Some(CastTy::FnPtr),
            _ => None,
        }
    }
}

/// Returns `mir::CastKind` from the given parameters.
pub fn mir_cast_kind<'tcx>(from_ty: Ty<'tcx>, cast_ty: Ty<'tcx>) -> mir::CastKind {
    let from = CastTy::from_ty(from_ty);
    let cast = CastTy::from_ty(cast_ty);
    let cast_kind = match (from, cast) {
        (Some(CastTy::Ptr(_) | CastTy::FnPtr), Some(CastTy::Int(_))) => {
            mir::CastKind::PointerExposeProvenance
        }
        (Some(CastTy::Int(_)), Some(CastTy::Ptr(_))) => mir::CastKind::PointerWithExposedProvenance,
        (Some(CastTy::Int(_)), Some(CastTy::Int(_))) => mir::CastKind::IntToInt,
        (Some(CastTy::FnPtr), Some(CastTy::Ptr(_))) => mir::CastKind::FnPtrToPtr,

        (Some(CastTy::Float), Some(CastTy::Int(_))) => mir::CastKind::FloatToInt,
        (Some(CastTy::Int(_)), Some(CastTy::Float)) => mir::CastKind::IntToFloat,
        (Some(CastTy::Float), Some(CastTy::Float)) => mir::CastKind::FloatToFloat,
        (Some(CastTy::Ptr(_)), Some(CastTy::Ptr(_))) => mir::CastKind::PtrToPtr,

        (_, _) => {
            bug!("Attempting to cast non-castable types {:?} and {:?}", from_ty, cast_ty)
        }
    };
    cast_kind
}
