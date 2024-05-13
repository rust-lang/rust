use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

use crate::fold::TypeSuperFoldable;
use crate::visit::{Flags, TypeSuperVisitable};
use crate::{
    BoundVar, ConstKind, DebruijnIndex, DebugWithInfcx, Interner, RegionKind, TyKind, UniverseIndex,
};

pub trait Ty<I: Interner<Ty = Self>>:
    Copy
    + DebugWithInfcx<I>
    + Hash
    + Eq
    + Into<I::GenericArg>
    + IntoKind<Kind = TyKind<I>>
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
{
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;
}

pub trait Region<I: Interner<Region = Self>>:
    Copy + DebugWithInfcx<I> + Hash + Eq + Into<I::GenericArg> + IntoKind<Kind = RegionKind<I>> + Flags
{
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar) -> Self;

    fn new_static(interner: I) -> Self;
}

pub trait Const<I: Interner<Const = Self>>:
    Copy
    + DebugWithInfcx<I>
    + Hash
    + Eq
    + Into<I::GenericArg>
    + IntoKind<Kind = ConstKind<I>>
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
{
    fn new_anon_bound(interner: I, debruijn: DebruijnIndex, var: BoundVar, ty: I::Ty) -> Self;

    fn ty(self) -> I::Ty;
}

pub trait GenericsOf<I: Interner<GenericsOf = Self>> {
    fn count(&self) -> usize;
}

pub trait GenericArgs<I: Interner<GenericArgs = Self>>:
    Copy
    + DebugWithInfcx<I>
    + Hash
    + Eq
    + IntoIterator<Item = I::GenericArg>
    + Deref<Target: Deref<Target = [I::GenericArg]>>
{
    fn type_at(self, i: usize) -> I::Ty;

    fn identity_for_item(interner: I, def_id: I::DefId) -> I::GenericArgs;
}

pub trait Predicate<I: Interner<Predicate = Self>>:
    Copy + Debug + Hash + Eq + TypeSuperVisitable<I> + TypeSuperFoldable<I> + Flags
{
}

/// Common capabilities of placeholder kinds
pub trait PlaceholderLike: Copy + Debug + Hash + Eq {
    fn universe(self) -> UniverseIndex;
    fn var(self) -> BoundVar;

    fn with_updated_universe(self, ui: UniverseIndex) -> Self;

    fn new(ui: UniverseIndex, var: BoundVar) -> Self;
}

pub trait IntoKind {
    type Kind;

    fn kind(self) -> Self::Kind;
}

pub trait BoundVars<I: Interner> {
    fn bound_vars(&self) -> I::BoundVars;

    fn has_no_bound_vars(&self) -> bool;
}

pub trait AliasTerm<I: Interner>: Copy + DebugWithInfcx<I> + Hash + Eq + Sized {
    fn new(
        interner: I,
        trait_def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self;

    fn def_id(self) -> I::DefId;

    fn args(self) -> I::GenericArgs;

    fn trait_def_id(self, interner: I) -> I::DefId;

    fn self_ty(self) -> I::Ty;

    fn with_self_ty(self, tcx: I, self_ty: I::Ty) -> Self;
}
