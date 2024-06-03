//! Set of traits which are used to emulate the inherent impls that are present in `rustc_middle`.
//! It is customary to glob-import `rustc_type_ir::inherent::*` to bring all of these traits into
//! scope when programming in interner-agnostic settings, and to avoid importing any of these
//! directly elsewhere (i.e. specify the full path for an implementation downstream).

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

use crate::fold::{TypeFoldable, TypeSuperFoldable};
use crate::visit::{Flags, TypeSuperVisitable, TypeVisitable};
use crate::{self as ty, CollectAndApply, DebugWithInfcx, Interner, UpcastFrom};

pub trait Ty<I: Interner<Ty = Self>>:
    Copy
    + DebugWithInfcx<I>
    + Hash
    + Eq
    + Into<I::GenericArg>
    + Into<I::Term>
    + IntoKind<Kind = ty::TyKind<I>>
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
{
    fn new_bool(interner: I) -> Self;

    fn new_infer(interner: I, var: ty::InferTy) -> Self;

    fn new_var(interner: I, var: ty::TyVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundTy) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_alias(interner: I, kind: ty::AliasTyKind, alias_ty: ty::AliasTy<I>) -> Self;

    fn new_coroutine(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_tup_from_iter<It, T>(interner: I, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: CollectAndApply<Self, Self>;

    fn tuple_fields(self) -> I::Tys;

    fn to_opt_closure_kind(self) -> Option<ty::ClosureKind>;

    fn from_closure_kind(interner: I, kind: ty::ClosureKind) -> Self;

    fn from_coroutine_closure_kind(interner: I, kind: ty::ClosureKind) -> Self;
}

pub trait Tys<I: Interner<Tys = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + IntoIterator<Item = I::Ty>
    + Deref<Target: Deref<Target = [I::Ty]>>
    + TypeFoldable<I>
    + Default
{
    fn split_inputs_and_output(self) -> (I::FnInputTys, I::Ty);
}

pub trait Abi<I: Interner<Abi = Self>>: Copy + Debug + Hash + Eq + TypeVisitable<I> {
    /// Whether this ABI is `extern "Rust"`.
    fn is_rust(self) -> bool;
}

pub trait Safety<I: Interner<Safety = Self>>: Copy + Debug + Hash + Eq + TypeVisitable<I> {
    fn is_safe(self) -> bool;

    fn prefix_str(self) -> &'static str;
}

pub trait Region<I: Interner<Region = Self>>:
    Copy
    + DebugWithInfcx<I>
    + Hash
    + Eq
    + Into<I::GenericArg>
    + IntoKind<Kind = ty::RegionKind<I>>
    + Flags
    + TypeVisitable<I>
{
    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundRegion) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_static(interner: I) -> Self;
}

pub trait Const<I: Interner<Const = Self>>:
    Copy
    + DebugWithInfcx<I>
    + Hash
    + Eq
    + Into<I::GenericArg>
    + Into<I::Term>
    + IntoKind<Kind = ty::ConstKind<I>>
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
{
    fn new_infer(interner: I, var: ty::InferConst) -> Self;

    fn new_var(interner: I, var: ty::ConstVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundConst) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_unevaluated(interner: I, uv: ty::UnevaluatedConst<I>) -> Self;
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
    + Default
    + TypeFoldable<I>
{
    fn type_at(self, i: usize) -> I::Ty;

    fn identity_for_item(interner: I, def_id: I::DefId) -> I::GenericArgs;

    fn extend_with_error(
        tcx: I,
        def_id: I::DefId,
        original_args: &[I::GenericArg],
    ) -> I::GenericArgs;

    fn split_closure_args(self) -> ty::ClosureArgsParts<I>;
    fn split_coroutine_closure_args(self) -> ty::CoroutineClosureArgsParts<I>;
    fn split_coroutine_args(self) -> ty::CoroutineArgsParts<I>;
}

pub trait Predicate<I: Interner<Predicate = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    + TypeSuperVisitable<I>
    + TypeSuperFoldable<I>
    + Flags
    + UpcastFrom<I, ty::NormalizesTo<I>>
{
    fn is_coinductive(self, interner: I) -> bool;
}

pub trait Clause<I: Interner<Clause = Self>>:
    Copy
    + Debug
    + Hash
    + Eq
    // FIXME: Remove these, uplift the `Upcast` impls.
    + UpcastFrom<I, ty::Binder<I, ty::TraitRef<I>>>
    + UpcastFrom<I, ty::Binder<I, ty::ProjectionPredicate<I>>>
{
}

/// Common capabilities of placeholder kinds
pub trait PlaceholderLike: Copy + Debug + Hash + Eq {
    fn universe(self) -> ty::UniverseIndex;
    fn var(self) -> ty::BoundVar;

    fn with_updated_universe(self, ui: ty::UniverseIndex) -> Self;

    fn new(ui: ty::UniverseIndex, var: ty::BoundVar) -> Self;
}

pub trait IntoKind {
    type Kind;

    fn kind(self) -> Self::Kind;
}

pub trait BoundVarLike<I: Interner> {
    fn var(self) -> ty::BoundVar;

    fn assert_eq(self, var: I::BoundVarKind);
}

pub trait ParamLike {
    fn index(self) -> u32;
}
