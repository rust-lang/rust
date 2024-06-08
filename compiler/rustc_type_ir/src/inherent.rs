//! Set of traits which are used to emulate the inherent impls that are present in `rustc_middle`.
//! It is customary to glob-import `rustc_type_ir::inherent::*` to bring all of these traits into
//! scope when programming in interner-agnostic settings, and to avoid importing any of these
//! directly elsewhere (i.e. specify the full path for an implementation downstream).

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

use rustc_ast_ir::Mutability;

use crate::fold::{TypeFoldable, TypeSuperFoldable};
use crate::relate::Relate;
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
    + Relate<I>
    + Flags
{
    fn new_bool(interner: I) -> Self;

    fn new_infer(interner: I, var: ty::InferTy) -> Self;

    fn new_var(interner: I, var: ty::TyVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundTy) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_alias(interner: I, kind: ty::AliasTyKind, alias_ty: ty::AliasTy<I>) -> Self;

    fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self;

    fn new_adt(interner: I, adt_def: I::AdtDef, args: I::GenericArgs) -> Self;

    fn new_foreign(interner: I, def_id: I::DefId) -> Self;

    fn new_dynamic(
        interner: I,
        preds: I::BoundExistentialPredicates,
        region: I::Region,
        kind: ty::DynKind,
    ) -> Self;

    fn new_coroutine(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_coroutine_closure(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_closure(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_coroutine_witness(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_ptr(interner: I, ty: Self, mutbl: Mutability) -> Self;

    fn new_ref(interner: I, region: I::Region, ty: Self, mutbl: Mutability) -> Self;

    fn new_array_with_const_len(interner: I, ty: Self, len: I::Const) -> Self;

    fn new_slice(interner: I, ty: Self) -> Self;

    fn new_tup(interner: I, tys: &[I::Ty]) -> Self;

    fn new_tup_from_iter<It, T>(interner: I, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: CollectAndApply<Self, Self>;

    fn tuple_fields(self) -> I::Tys;

    fn to_opt_closure_kind(self) -> Option<ty::ClosureKind>;

    fn from_closure_kind(interner: I, kind: ty::ClosureKind) -> Self;

    fn from_coroutine_closure_kind(interner: I, kind: ty::ClosureKind) -> Self;

    fn new_fn_def(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self;

    fn new_fn_ptr(interner: I, sig: ty::Binder<I, ty::FnSig<I>>) -> Self;

    fn new_pat(interner: I, ty: Self, pat: I::Pat) -> Self;
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
    + Relate<I>
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
    + Relate<I>
    + Flags
{
    fn try_to_target_usize(self, interner: I) -> Option<u64>;

    fn new_infer(interner: I, var: ty::InferConst) -> Self;

    fn new_var(interner: I, var: ty::ConstVid) -> Self;

    fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: I::BoundConst) -> Self;

    fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self;

    fn new_unevaluated(interner: I, uv: ty::UnevaluatedConst<I>) -> Self;

    fn new_expr(interner: I, expr: I::ExprConst) -> Self;
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
    + Relate<I>
{
    fn type_at(self, i: usize) -> I::Ty;

    fn identity_for_item(interner: I, def_id: I::DefId) -> I::GenericArgs;

    fn extend_with_error(
        interner: I,
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

pub trait AdtDef<I: Interner>: Copy + Debug + Hash + Eq {
    fn def_id(self) -> I::DefId;
}

pub trait Features<I: Interner>: Copy {
    fn generic_const_exprs(self) -> bool;
}
