use std::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use syntax::ast;

use crate::ty::{
    self, AdtDef, Binder, BoundTy, ExistentialPredicate, InferTy, List, ParamTy, PolyFnSig,
    ProjectionTy, Region, SubstsRef, Ty, TypeAndMut,
};
use rustc_hir::{self as hir, def_id::DefId};

pub use self::ViewKind::*;

/// `View<'tcx, T>` contains a value of `T` but stores the `Ty<'tcx>` ptr that contains the `T`
/// This allows for cheap access to the `Ty<'tcx>` without needing to ask the type interner or
/// losing the `T` type.
#[derive(TypeFoldable, Lift)]
pub struct View<'tcx, T> {
    ty: Ty<'tcx>,
    _marker: PhantomData<T>,
}

impl<'tcx, T> PartialEq for View<'tcx, T>
where
    T: PartialEq + TyDeref<'tcx>,
{
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<'tcx, T> Eq for View<'tcx, T> where T: Eq + TyDeref<'tcx> {}

impl<'tcx, T> Hash for View<'tcx, T>
where
    T: Hash + TyDeref<'tcx>,
{
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        (**self).hash(hasher)
    }
}

impl<T> Copy for View<'_, T> {}

impl<T> Clone for View<'_, T> {
    fn clone(&self) -> Self {
        View { ty: self.ty, _marker: PhantomData }
    }
}

impl<'tcx, T> fmt::Debug for View<'tcx, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.ty.fmt(f)
    }
}

impl<'tcx, T> fmt::Display for View<'tcx, T>
where
    T: fmt::Display + TyDeref<'tcx>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}
impl<'tcx, T> std::ops::Deref for View<'tcx, T>
where
    T: TyDeref<'tcx>,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match T::ty_deref(self.ty) {
            Some(t) => t,
            // SAFETY verified by `View::new`
            None => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}

impl<'tcx, T> View<'tcx, T>
where
    T: TyDeref<'tcx>,
{
    pub fn new(ty: Ty<'tcx>) -> Option<Self> {
        T::ty_deref(ty)?;
        Some(View { ty, _marker: PhantomData })
    }
}

impl<'tcx, T> View<'tcx, T> {
    pub fn as_ty(&self) -> Ty<'tcx> {
        self.ty
    }
}

/// SAFETY If `Some` is returned for `ty` then `Some` must always be returned for any subsequent
/// call with the same `Ty` value
pub unsafe trait TyDeref<'tcx>: Sized + 'tcx {
    fn ty_deref(ty: Ty<'tcx>) -> Option<&'tcx Self>;
}

macro_rules! impl_ty_deref {
    ($ty: ty, $variant: ident) => {
        unsafe impl<'tcx> TyDeref<'tcx> for $ty {
            fn ty_deref(ty: Ty<'tcx>) -> Option<&'tcx Self> {
                match &ty.kind {
                    ty::$variant(p) => Some(p),
                    _ => None,
                }
            }
        }
    };
}

impl_ty_deref! { ty::ParamTy, Param }
impl_ty_deref! { ty::ProjectionTy<'tcx>, Projection }

/// Mirror of `TyKind`, but with `View` fields where there is need for it
pub enum ViewKind<'tcx> {
    Bool,
    Char,
    Int(ast::IntTy),
    Uint(ast::UintTy),
    Float(ast::FloatTy),
    Adt(&'tcx AdtDef, SubstsRef<'tcx>),
    Foreign(DefId),
    Str,
    Array(Ty<'tcx>, &'tcx ty::Const<'tcx>),
    Slice(Ty<'tcx>),
    RawPtr(TypeAndMut<'tcx>),
    Ref(Region<'tcx>, Ty<'tcx>, hir::Mutability),
    FnDef(DefId, SubstsRef<'tcx>),
    FnPtr(PolyFnSig<'tcx>),
    Dynamic(Binder<&'tcx List<ExistentialPredicate<'tcx>>>, ty::Region<'tcx>),
    Closure(DefId, SubstsRef<'tcx>),
    Generator(DefId, SubstsRef<'tcx>, hir::Movability),
    GeneratorWitness(Binder<&'tcx List<Ty<'tcx>>>),
    Never,
    Tuple(SubstsRef<'tcx>),
    Projection(View<'tcx, ProjectionTy<'tcx>>),
    UnnormalizedProjection(ProjectionTy<'tcx>),
    Opaque(DefId, SubstsRef<'tcx>),
    Param(View<'tcx, ParamTy>),
    Bound(ty::DebruijnIndex, BoundTy),
    Placeholder(ty::PlaceholderType),
    Infer(InferTy),
    Error,
}

impl<'tcx> From<Ty<'tcx>> for ViewKind<'tcx> {
    fn from(ty: Ty<'tcx>) -> Self {
        match ty.kind {
            ty::RawPtr(tm) => Self::RawPtr(tm),
            ty::Array(typ, sz) => Self::Array(typ, sz),
            ty::Slice(typ) => Self::Slice(typ),
            ty::Adt(tid, substs) => Self::Adt(tid, substs),
            ty::Dynamic(trait_ty, region) => Self::Dynamic(trait_ty, region),
            ty::Tuple(ts) => Self::Tuple(ts),
            ty::FnDef(def_id, substs) => Self::FnDef(def_id, substs),
            ty::FnPtr(f) => Self::FnPtr(f),
            ty::Ref(r, ty, mutbl) => Self::Ref(r, ty, mutbl),
            ty::Generator(did, substs, movability) => Self::Generator(did, substs, movability),
            ty::GeneratorWitness(types) => Self::GeneratorWitness(types),
            ty::Closure(did, substs) => Self::Closure(did, substs),
            ty::Projection(_) => Self::Projection(View::new(ty).unwrap()),
            ty::UnnormalizedProjection(data) => Self::UnnormalizedProjection(data),
            ty::Opaque(did, substs) => Self::Opaque(did, substs),
            ty::Bool => Self::Bool,
            ty::Char => Self::Char,
            ty::Str => Self::Str,
            ty::Int(i) => Self::Int(i),
            ty::Uint(i) => Self::Uint(i),
            ty::Float(f) => Self::Float(f),
            ty::Error => Self::Error,
            ty::Infer(i) => Self::Infer(i),
            ty::Param(_) => Self::Param(View::new(ty).unwrap()),
            ty::Bound(b, c) => Self::Bound(b, c),
            ty::Placeholder(p) => Self::Placeholder(p),
            ty::Never => Self::Never,
            ty::Foreign(f) => Self::Foreign(f),
        }
    }
}
