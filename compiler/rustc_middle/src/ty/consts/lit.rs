use rustc_ast::LitKind;
use rustc_hir;
use rustc_macros::HashStable;

use crate::ty::{self, Ty, TyCtxt};

/// Input argument for `tcx.lit_to_const`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, HashStable)]
pub struct LitToConstInput<'tcx> {
    /// The absolute value of the resultant constant.
    pub lit: LitKind,
    /// The type of the constant.
    pub ty: Ty<'tcx>,
    /// If the constant is negative.
    pub neg: bool,
}

/// Checks whether a literal can be interpreted as a const of the given type.
pub fn const_lit_matches_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: &LitKind,
    ty: Ty<'tcx>,
    neg: bool,
) -> bool {
    match (*kind, ty.kind()) {
        (LitKind::Str(..), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => true,
        (LitKind::Str(..), ty::Str) if tcx.features().deref_patterns() => true,
        (LitKind::ByteStr(..), ty::Ref(_, inner_ty, _))
            if let ty::Slice(ty) | ty::Array(ty, _) = inner_ty.kind()
                && matches!(ty.kind(), ty::Uint(ty::UintTy::U8)) =>
        {
            true
        }
        (LitKind::ByteStr(..), ty::Slice(inner_ty) | ty::Array(inner_ty, _))
            if tcx.features().deref_patterns()
                && matches!(inner_ty.kind(), ty::Uint(ty::UintTy::U8)) =>
        {
            true
        }
        (LitKind::Byte(..), ty::Uint(ty::UintTy::U8)) => true,
        (LitKind::CStr(..), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Adt(def, _)
                if tcx.is_lang_item(def.did(), rustc_hir::LangItem::CStr)) =>
        {
            true
        }
        (LitKind::Int(..), ty::Uint(_)) if !neg => true,
        (LitKind::Int(..), ty::Int(_)) => true,
        (LitKind::Bool(..), ty::Bool) => true,
        (LitKind::Float(..), ty::Float(_)) => true,
        (LitKind::Char(..), ty::Char) => true,
        (LitKind::Err(..), _) => true,
        _ => false,
    }
}
