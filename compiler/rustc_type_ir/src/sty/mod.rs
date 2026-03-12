use std::fmt::{self, Debug};
use std::hash::Hash;
use std::ops::{ControlFlow, Deref, Range};

use derive_where::derive_where;
use rustc_abi::{FieldIdx, VariantIdx};
#[cfg(feature = "nightly")]
use rustc_data_structures::fingerprint::Fingerprint;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_span::sym;
use tracing::instrument;

use crate::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
use crate::inherent::*;
use crate::lang_items::{SolverLangItem, SolverTraitLangItem};
use crate::solve::SizedTraitKind;
use crate::visit::{
    Flags, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor, VisitorResult,
};
use crate::walk::TypeWalker;
use crate::{
    self as ty, AliasTy, CollectAndApply, DebruijnIndex, Interner, Mutability, TyKind, TypeFlags,
    WithCachedTypeInfo, try_visit,
};

/// Use this rather than `TyKind`, whenever possible.
#[derive_where(Copy; I: Interner, I::InternedTyKindWithCachedInfo: Copy)]
#[derive_where(Clone, PartialEq, Eq, Hash; I: Interner)]
#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
pub struct Ty<I: Interner>(pub I::InternedTyKindWithCachedInfo);

impl<I: Interner> Ty<I> {
    #[inline]
    pub fn from_interned(interned: I::InternedTyKindWithCachedInfo) -> Self {
        Ty(interned)
    }

    #[inline]
    pub fn interned(self) -> I::InternedTyKindWithCachedInfo {
        self.0
    }
}

#[cfg(feature = "nightly")]
impl<CTX, I: Interner> HashStable<CTX> for Ty<I> {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        // Hashing the cached fingerprint avoids recursive trait obligations like
        // `Ty: HashStable -> TyKind: HashStable -> Ty: HashStable`.
        let stable_hash = self.0.stable_hash;
        debug_assert_ne!(
            stable_hash,
            Fingerprint::ZERO,
            "Ty should only be stably hashed once it has a cached stable hash",
        );
        stable_hash.hash_stable(hcx, hasher);
    }
}

impl<I: Interner> fmt::Debug for Ty<I>
where
    I::InternedTyKindWithCachedInfo: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&(*self).kind(), f)
    }
}

impl<I: Interner> IntoKind for Ty<I>
where
    I::InternedTyKindWithCachedInfo: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    type Kind = TyKind<I>;

    #[inline]
    fn kind(self) -> TyKind<I> {
        (*self.0).internee
    }
}

impl<I: Interner> Flags for Ty<I>
where
    I::InternedTyKindWithCachedInfo: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    #[inline]
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    #[inline]
    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl<I: Interner> TypeVisitable<I> for Ty<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_ty(*self)
    }
}

impl<I: Interner> TypeSuperVisitable<I> for Ty<I>
where
    I::InternedTyKindWithCachedInfo: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
    I::BoundExistentialPredicates: TypeVisitable<I>,
    I::Const: TypeVisitable<I>,
    I::ErrorGuaranteed: TypeVisitable<I>,
    I::GenericArgs: TypeVisitable<I>,
    I::Pat: TypeVisitable<I>,
    I::Region: TypeVisitable<I>,
    I::Tys: TypeVisitable<I>,
{
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match (*self).kind() {
            ty::RawPtr(ty, _mutbl) => ty.visit_with(visitor),
            ty::Array(typ, sz) => {
                try_visit!(typ.visit_with(visitor));
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, args) => args.visit_with(visitor),
            ty::Dynamic(trait_ty, reg) => {
                try_visit!(trait_ty.visit_with(visitor));
                reg.visit_with(visitor)
            }
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, args) => args.visit_with(visitor),
            ty::FnPtr(sig_tys, _) => sig_tys.visit_with(visitor),
            ty::UnsafeBinder(f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => {
                try_visit!(r.visit_with(visitor));
                ty.visit_with(visitor)
            }
            ty::Coroutine(_did, args) => args.visit_with(visitor),
            ty::CoroutineWitness(_did, args) => args.visit_with(visitor),
            ty::Closure(_did, args) => args.visit_with(visitor),
            ty::CoroutineClosure(_did, args) => args.visit_with(visitor),
            ty::Alias(_, data) => data.visit_with(visitor),
            ty::Pat(ty, pat) => {
                try_visit!(ty.visit_with(visitor));
                pat.visit_with(visitor)
            }
            ty::Error(guar) => guar.visit_with(visitor),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => V::Result::output(),
        }
    }
}

impl<I: Interner> TypeFoldable<I> for Ty<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_ty(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_ty(self)
    }
}

impl<I: Interner> TypeSuperFoldable<I> for Ty<I>
where
    I::InternedTyKindWithCachedInfo: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
    I::BoundExistentialPredicates: TypeFoldable<I>,
    I::Const: TypeFoldable<I>,
    I::GenericArgs: TypeFoldable<I>,
    I::Pat: TypeFoldable<I>,
    I::Region: TypeFoldable<I>,
    I::Tys: TypeFoldable<I>,
{
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match self.kind() {
            ty::RawPtr(ty, mutbl) => ty::RawPtr(ty.try_fold_with(folder)?, mutbl),
            ty::Array(typ, sz) => ty::Array(typ.try_fold_with(folder)?, sz.try_fold_with(folder)?),
            ty::Slice(typ) => ty::Slice(typ.try_fold_with(folder)?),
            ty::Adt(tid, args) => ty::Adt(tid, args.try_fold_with(folder)?),
            ty::Dynamic(trait_ty, region) => {
                ty::Dynamic(trait_ty.try_fold_with(folder)?, region.try_fold_with(folder)?)
            }
            ty::Tuple(ts) => ty::Tuple(ts.try_fold_with(folder)?),
            ty::FnDef(def_id, args) => ty::FnDef(def_id, args.try_fold_with(folder)?),
            ty::FnPtr(sig_tys, hdr) => ty::FnPtr(sig_tys.try_fold_with(folder)?, hdr),
            ty::UnsafeBinder(f) => ty::UnsafeBinder(f.try_fold_with(folder)?),
            ty::Ref(r, ty, mutbl) => {
                ty::Ref(r.try_fold_with(folder)?, ty.try_fold_with(folder)?, mutbl)
            }
            ty::Coroutine(did, args) => ty::Coroutine(did, args.try_fold_with(folder)?),
            ty::CoroutineWitness(did, args) => {
                ty::CoroutineWitness(did, args.try_fold_with(folder)?)
            }
            ty::Closure(did, args) => ty::Closure(did, args.try_fold_with(folder)?),
            ty::CoroutineClosure(did, args) => {
                ty::CoroutineClosure(did, args.try_fold_with(folder)?)
            }
            ty::Alias(kind, data) => ty::Alias(kind, data.try_fold_with(folder)?),
            ty::Pat(ty, pat) => ty::Pat(ty.try_fold_with(folder)?, pat.try_fold_with(folder)?),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => return Ok(self),
        };

        Ok(if self.kind() == kind { self } else { folder.cx().mk_ty_from_kind(kind) })
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        let kind = match self.kind() {
            ty::RawPtr(ty, mutbl) => ty::RawPtr(ty.fold_with(folder), mutbl),
            ty::Array(typ, sz) => ty::Array(typ.fold_with(folder), sz.fold_with(folder)),
            ty::Slice(typ) => ty::Slice(typ.fold_with(folder)),
            ty::Adt(tid, args) => ty::Adt(tid, args.fold_with(folder)),
            ty::Dynamic(trait_ty, region) => {
                ty::Dynamic(trait_ty.fold_with(folder), region.fold_with(folder))
            }
            ty::Tuple(ts) => ty::Tuple(ts.fold_with(folder)),
            ty::FnDef(def_id, args) => ty::FnDef(def_id, args.fold_with(folder)),
            ty::FnPtr(sig_tys, hdr) => ty::FnPtr(sig_tys.fold_with(folder), hdr),
            ty::UnsafeBinder(f) => ty::UnsafeBinder(f.fold_with(folder)),
            ty::Ref(r, ty, mutbl) => ty::Ref(r.fold_with(folder), ty.fold_with(folder), mutbl),
            ty::Coroutine(did, args) => ty::Coroutine(did, args.fold_with(folder)),
            ty::CoroutineWitness(did, args) => ty::CoroutineWitness(did, args.fold_with(folder)),
            ty::Closure(did, args) => ty::Closure(did, args.fold_with(folder)),
            ty::CoroutineClosure(did, args) => ty::CoroutineClosure(did, args.fold_with(folder)),
            ty::Alias(kind, data) => ty::Alias(kind, data.fold_with(folder)),
            ty::Pat(ty, pat) => ty::Pat(ty.fold_with(folder), pat.fold_with(folder)),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => return self,
        };

        if self.kind() == kind { self } else { folder.cx().mk_ty_from_kind(kind) }
    }
}

// Simple constructors and transformations
impl<I: Interner> Ty<I>
where
    I::BoundExistentialPredicates: TypeFoldable<I> + TypeVisitable<I>,
    I::GenericArg: From<Ty<I>>,
    I::GenericArgs: TypeFoldable<I> + TypeVisitable<I>,
    I::Const: TypeFoldable<I> + TypeVisitable<I>,
    I::ErrorGuaranteed: TypeVisitable<I>,
    I::Pat: TypeFoldable<I> + TypeVisitable<I>,
    I::Region: TypeFoldable<I> + TypeVisitable<I>,
    I::Term: From<Ty<I>>,
    I::Tys: TypeFoldable<I> + TypeVisitable<I>,
    I::InternedTyKindWithCachedInfo:
        Copy + Clone + Debug + Hash + Eq + Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    /// Avoid using this in favour of more specific `new_*` methods, where possible.
    /// The more specific methods will often optimize their creation.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    pub fn new(interner: I, st: TyKind<I>) -> Self {
        interner.mk_ty_from_kind(st)
    }

    #[inline]
    pub fn new_unit(interner: I) -> Self {
        Self::new(interner, ty::Tuple(Default::default()))
    }

    #[inline]
    pub fn new_bool(interner: I) -> Self {
        Self::new(interner, ty::Bool)
    }

    #[inline]
    pub fn new_u8(interner: I) -> Self {
        Self::new(interner, ty::Uint(ty::UintTy::U8))
    }

    #[inline]
    pub fn new_usize(interner: I) -> Self {
        Self::new(interner, ty::Uint(ty::UintTy::Usize))
    }

    #[inline]
    pub fn new_infer(interner: I, var: ty::InferTy) -> Self {
        Self::new(interner, ty::Infer(var))
    }

    #[inline]
    pub fn new_var(interner: I, var: ty::TyVid) -> Self {
        // Use a pre-interned one when possible.
        interner
            .get_ty_var(var.as_usize())
            .unwrap_or_else(|| Self::new(interner, ty::Infer(ty::InferTy::TyVar(var))))
    }

    #[inline]
    pub fn new_int_var(interner: I, v: ty::IntVid) -> Self {
        Self::new_infer(interner, ty::IntVar(v))
    }

    #[inline]
    pub fn new_float_var(interner: I, v: ty::FloatVid) -> Self {
        Self::new_infer(interner, ty::FloatVar(v))
    }

    #[inline]
    pub fn new_fresh(interner: I, n: u32) -> Self {
        // Use a pre-interned one when possible.
        interner
            .get_fresh_ty(n as usize)
            .unwrap_or_else(|| Self::new_infer(interner, ty::FreshTy(n)))
    }

    #[inline]
    pub fn new_fresh_int(interner: I, n: u32) -> Self {
        // Use a pre-interned one when possible.
        interner
            .get_fresh_ty_int(n as usize)
            .unwrap_or_else(|| Self::new_infer(interner, ty::FreshIntTy(n)))
    }

    #[inline]
    pub fn new_fresh_float(interner: I, n: u32) -> Self {
        // Use a pre-interned one when possible.
        interner
            .get_fresh_ty_float(n as usize)
            .unwrap_or_else(|| Self::new_infer(interner, ty::FreshFloatTy(n)))
    }

    #[inline]
    pub fn new_param(interner: I, param: I::ParamTy) -> Self {
        Self::new(interner, ty::Param(param))
    }

    #[inline]
    pub fn new_placeholder(interner: I, param: ty::PlaceholderType<I>) -> Self {
        Self::new(interner, ty::Placeholder(param))
    }

    #[inline]
    pub fn new_bound(interner: I, index: ty::DebruijnIndex, bound_ty: ty::BoundTy<I>) -> Self {
        // Use a pre-interned one when possible.
        if let ty::BoundTy { var, kind: ty::BoundTyKind::Anon } = bound_ty
            && let Some(inner) = interner.get_anon_bound_ty(index.as_usize())
            && let Some(ty) = inner.get(var.as_usize()).copied()
        {
            ty
        } else {
            Self::new(interner, ty::Bound(ty::BoundVarIndexKind::Bound(index), bound_ty))
        }
    }

    #[inline]
    pub fn new_canonical_bound(interner: I, var: ty::BoundVar) -> Self {
        // Use a pre-interned one when possible.
        if let Some(ty) = interner.get_anon_canonical_bound_ty(var.as_usize()) {
            ty
        } else {
            Self::new(
                interner,
                ty::Bound(
                    ty::BoundVarIndexKind::Canonical,
                    ty::BoundTy { var, kind: ty::BoundTyKind::Anon },
                ),
            )
        }
    }

    #[inline]
    pub fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        let bound_ty = ty::BoundTy { var, kind: ty::BoundTyKind::Anon };
        Self::new(interner, ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty))
    }

    #[inline]
    pub fn new_alias(interner: I, kind: ty::AliasTyKind, alias_ty: ty::AliasTy<I>) -> Self {
        // @PROBLEM
        // - requires hir for `DefKind` and we need `def_kind()` as a method
        // - Is `interner.debug_assert_args_compatible` a suitable drop in for
        //   `debug_assert_matches!`?
        //
        // interner.debug_assert_args_compatible(
        //     (kind, interner.def_kind(alias_ty.def_id)),
        //     (ty::Opaque, DefKind::OpaqueTy)
        //         | (ty::Projection | ty::Inherent, DefKind::AssocTy)
        //         | (ty::Free, DefKind::TyAlias)
        // );
        Self::new(interner, ty::Alias(kind, alias_ty))
    }

    #[inline]
    pub fn new_projection_from_args(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self {
        Ty::new_alias(
            interner,
            ty::AliasTyKind::Projection,
            ty::AliasTy::new_from_args(interner, def_id, args),
        )
    }

    #[inline]
    pub fn new_projection(
        interner: I,
        def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        Ty::new_alias(
            interner,
            ty::AliasTyKind::Projection,
            ty::AliasTy::new(interner, def_id, args),
        )
    }

    /// Constructs a `TyKind::Error` type with current `ErrorGuaranteed`
    pub fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self {
        Self::new(interner, ty::Error(guar))
    }

    #[inline]
    pub fn new_adt(interner: I, adt_def: I::AdtDef, args: I::GenericArgs) -> Self {
        interner.debug_assert_args_compatible(adt_def.def_id().into(), args);
        interner.debug_assert_adt_def_compatible(adt_def);
        Self::new(interner, ty::Adt(adt_def, args))
    }

    #[inline]
    pub fn new_foreign(interner: I, def_id: I::ForeignId) -> Self {
        Self::new(interner, ty::Foreign(def_id))
    }

    #[inline]
    pub fn new_array(interner: I, ty: Ty<I>, n: u64) -> Ty<I> {
        Self::new(interner, ty::Array(ty, Const::from_target_usize(interner, n)))
    }

    #[inline]
    pub fn new_dynamic(
        interner: I,
        preds: I::BoundExistentialPredicates,
        region: I::Region,
    ) -> Self {
        Self::new(interner, ty::Dynamic(preds, region))
    }

    #[inline]
    pub fn new_coroutine(interner: I, def_id: I::CoroutineId, args: I::GenericArgs) -> Self {
        interner.debug_assert_args_compatible(def_id.into(), args);
        Self::new(interner, ty::Coroutine(def_id, args))
    }

    #[inline]
    pub fn new_coroutine_closure(
        interner: I,
        def_id: I::CoroutineClosureId,
        args: I::GenericArgs,
    ) -> Self {
        Self::new(interner, ty::CoroutineClosure(def_id, args))
    }

    #[inline]
    pub fn new_closure(interner: I, def_id: I::ClosureId, args: I::GenericArgs) -> Self {
        Self::new(interner, ty::Closure(def_id, args))
    }

    #[inline]
    pub fn new_coroutine_witness(
        interner: I,
        def_id: I::CoroutineId,
        args: I::GenericArgs,
    ) -> Self {
        if cfg!(debug_assertions) {
            interner.debug_assert_args_compatible(interner.typeck_root_def_id(def_id.into()), args);
        }
        Self::new(interner, ty::CoroutineWitness(def_id, args))
    }

    #[inline]
    pub fn new_coroutine_witness_for_coroutine(
        interner: I,
        def_id: I::CoroutineId,
        coroutine_args: I::GenericArgs,
    ) -> Self {
        interner.debug_assert_args_compatible(def_id.into(), coroutine_args);
        let args = interner.get_generic_args_for_item(def_id.into(), coroutine_args);
        Self::new_coroutine_witness(interner, def_id, args)
    }

    #[inline]
    pub fn new_static_str(interner: I) -> Self {
        interner.new_static_str()
    }

    fn new_generic_adt(interner: I, wrapper_def_id: I::DefId, ty_param: Ty<I>) -> Ty<I> {
        interner.new_generic_adt(wrapper_def_id, ty_param)
    }

    #[inline]
    pub fn new_lang_item(interner: I, ty: Ty<I>, item: I::LangItem) -> Option<Ty<I>> {
        let def_id = interner.get_lang_item(item)?;
        Some(Self::new_generic_adt(interner, def_id, ty))
    }

    #[inline]
    pub fn new_diagnostic_item(interner: I, ty: Ty<I>, name: I::Symbol) -> Option<Ty<I>> {
        let def_id = interner.get_diagnostic_item(name)?;
        Some(Self::new_generic_adt(interner, def_id, ty))
    }

    #[inline]
    pub fn new_box(interner: I, ty: Ty<I>) -> Self {
        let def_id = interner.require_lang_item_owned_box();
        Self::new_generic_adt(interner, def_id, ty)
    }

    #[inline]
    pub fn new_option(interner: I, ty: Self) -> Self {
        let def_id = interner.require_lang_item_option();
        Self::new_generic_adt(interner, def_id, ty)
    }

    #[inline]
    pub fn new_maybe_uninit(interner: I, ty: Self) -> Self {
        let def_id = interner.require_lang_item_maybe_uninit();
        Self::new_generic_adt(interner, def_id, ty)
    }

    /// Creates a `&mut Context<'_>` [`Ty`] with erased lifetimes.
    pub fn new_task_context(interner: I) -> Self {
        interner.new_task_context()
    }

    #[inline]
    pub fn new_ptr(interner: I, ty: Self, mutbl: Mutability) -> Self {
        Self::new(interner, ty::RawPtr(ty, mutbl))
    }

    #[inline]
    pub fn new_mut_ptr(interner: I, ty: Ty<I>) -> Ty<I> {
        Self::new_ptr(interner, ty, Mutability::Mut)
    }

    #[inline]
    pub fn new_imm_ptr(interner: I, ty: Ty<I>) -> Ty<I> {
        Self::new_ptr(interner, ty, Mutability::Not)
    }

    #[inline]
    pub fn new_ref(interner: I, region: I::Region, ty: Self, mutbl: Mutability) -> Self {
        Self::new(interner, ty::Ref(region, ty, mutbl))
    }

    #[inline]
    pub fn new_mut_ref(interner: I, r: I::Region, ty: Ty<I>) -> Ty<I> {
        Self::new_ref(interner, r, ty, Mutability::Mut)
    }

    #[inline]
    pub fn new_imm_ref(interner: I, r: I::Region, ty: Ty<I>) -> Ty<I> {
        Self::new_ref(interner, r, ty, Mutability::Not)
    }

    #[inline]
    pub fn new_array_with_const_len(interner: I, ty: Self, len: I::Const) -> Self {
        Self::new(interner, ty::Array(ty, len))
    }

    pub fn new_pinned_ref(interner: I, r: I::Region, ty: Ty<I>, mutbl: Mutability) -> Ty<I> {
        interner.new_pinned_ref(r, ty, mutbl)
    }

    #[inline]
    pub fn new_slice(interner: I, ty: Self) -> Self {
        Self::new(interner, ty::Slice(ty))
    }

    #[inline]
    pub fn new_tup(interner: I, tys: &[Ty<I>]) -> Self {
        if tys.is_empty() {
            return Self::new_unit(interner);
        }

        let tys = interner.mk_type_list_from_iter(tys.iter().copied());
        Self::new(interner, ty::Tuple(tys))
    }

    #[inline]
    pub fn new_tup_from_iter<It, T>(interner: I, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: CollectAndApply<Self, Self>,
    {
        T::collect_and_apply(iter, |ts| Self::new_tup(interner, ts))
    }

    #[inline]
    pub fn new_fn_def(interner: I, def_id: I::FunctionId, args: I::GenericArgs) -> Self {
        Self::new(interner, ty::FnDef(def_id, args))
    }

    #[inline]
    pub fn new_fn_ptr(interner: I, sig: ty::Binder<I, ty::FnSig<I>>) -> Self {
        let (sig_tys, hdr) = sig.split();
        Self::new(interner, ty::FnPtr(sig_tys, hdr))
    }

    #[inline]
    pub fn new_pat(interner: I, ty: Self, pat: I::Pat) -> Self {
        Self::new(interner, ty::Pat(ty, pat))
    }

    #[inline]
    #[instrument(level = "debug", skip(interner))]
    pub fn new_opaque(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self {
        Self::new_alias(interner, ty::Opaque, AliasTy::new_from_args(interner, def_id, args))
    }

    #[inline]
    pub fn new_unsafe_binder(interner: I, ty: ty::Binder<I, Ty<I>>) -> Self {
        Self::new(interner, ty::UnsafeBinder(ty::UnsafeBinderInner::from(ty)))
    }
}

// Methods to determine what flavour `Ty` is
impl<I: Interner> Ty<I>
where
    I::InternedTyKindWithCachedInfo: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
    I::BoundExistentialPredicates: TypeVisitable<I>,
    I::Const: TypeVisitable<I>,
    I::ErrorGuaranteed: TypeVisitable<I>,
    I::GenericArgs: TypeVisitable<I>,
    I::Pat: TypeVisitable<I>,
    I::Region: TypeVisitable<I>,
    I::Tys: TypeVisitable<I>,
{
    // It would be nicer if this returned the value instead of a reference,
    // like how `Predicate::kind` and `Region::kind` do. (It would result in
    // many fewer subsequent dereferences.) But that gives a small but
    // noticeable performance hit. See #126069 for details.
    // Using `rustc::disallowed_pass_by_ref` otherwise we need to import the
    // `IntoKind` trait everywhere and deference which is a lot of noise
    #[expect(rustc::disallowed_pass_by_ref)]
    #[inline(always)]
    pub fn kind(&self) -> &TyKind<I> {
        &self.0
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    #[inline]
    pub fn is_unit(self) -> bool {
        match self.kind() {
            ty::Tuple(tys) => tys.is_empty(),
            _ => false,
        }
    }

    /// Check if type is an `usize`.
    #[inline]
    pub fn is_usize(self) -> bool {
        matches!(self.kind(), ty::Uint(ty::UintTy::Usize))
    }

    /// Check if type is an `usize` or an integral type variable.
    #[inline]
    pub fn is_usize_like(self) -> bool {
        matches!(self.kind(), ty::Uint(ty::UintTy::Usize) | ty::Infer(ty::IntVar(_)))
    }

    #[inline]
    pub fn is_never(self) -> bool {
        matches!(self.kind(), ty::Never)
    }

    #[inline]
    pub fn is_primitive(self) -> bool {
        matches!(self.kind(), ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_))
    }

    #[inline]
    pub fn is_adt(self) -> bool {
        matches!(self.kind(), ty::Adt(..))
    }

    #[inline]
    pub fn is_ref(self) -> bool {
        matches!(self.kind(), ty::Ref(..))
    }

    #[inline]
    pub fn is_ty_var(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::TyVar(_)))
    }

    #[inline]
    pub fn ty_vid(self) -> Option<ty::TyVid> {
        match self.kind() {
            ty::Infer(ty::TyVar(vid)) => Some(vid),
            _ => None,
        }
    }

    #[inline]
    pub fn is_ty_or_numeric_infer(self) -> bool {
        matches!(self.kind(), ty::Infer(_))
    }

    #[inline]
    pub fn is_phantom_data(self) -> bool {
        if let ty::Adt(def, _) = self.kind() { def.is_phantom_data() } else { false }
    }

    #[inline]
    pub fn is_bool(self) -> bool {
        self.kind() == ty::Bool
    }

    /// Returns `true` if this type is a `str`.
    #[inline]
    pub fn is_str(self) -> bool {
        self.kind() == ty::Str
    }

    /// Returns true if this type is `&str`. The reference's lifetime is ignored.
    #[inline]
    pub fn is_imm_ref_str(self) -> bool {
        matches!(self.kind(), ty::Ref(_, inner, Mutability::Not) if inner.is_str())
    }

    #[inline]
    pub fn is_param(self, index: u32) -> bool {
        match self.kind() {
            ty::Param(data) => data.index() == index,
            _ => false,
        }
    }

    #[inline]
    pub fn is_slice(self) -> bool {
        matches!(self.kind(), ty::Slice(_))
    }

    #[inline]
    pub fn is_array_slice(self) -> bool {
        match self.kind() {
            ty::Slice(_) => true,
            ty::RawPtr(ty, _) | ty::Ref(_, ty, _) => matches!(ty.kind(), ty::Slice(_)),
            _ => false,
        }
    }

    #[inline]
    pub fn is_array(self) -> bool {
        matches!(self.kind(), ty::Array(..))
    }

    #[inline]
    pub fn is_simd(self) -> bool {
        match self.kind() {
            ty::Adt(def, _) => def.repr_is_simd(),
            _ => false,
        }
    }

    #[inline]
    pub fn is_scalable_vector(self) -> bool {
        match self.kind() {
            ty::Adt(def, _) => def.scalable_element_cnt().is_some(),
            _ => false,
        }
    }

    pub fn sequence_element_type(self, interner: I) -> Ty<I> {
        match self.kind() {
            ty::Array(ty, _) | ty::Slice(ty) => ty,
            ty::Str => interner.u8_type(),
            _ => panic!("`sequence_element_type` called on non-sequence value: {:?}", self),
        }
    }

    pub fn scalable_vector_element_count_and_type(self, interner: I) -> (u16, Ty<I>) {
        let ty::Adt(def, args) = self.kind() else {
            panic!("`scalable_vector_size_and_type` called on invalid type")
        };
        let Some(element_count) = def.scalable_element_cnt() else {
            panic!("`scalable_vector_size_and_type` called on non-scalable vector type");
        };
        assert_eq!(def.non_enum_variant().fields_len(), 1);
        let field_ty = def.non_enum_variant().field_zero_ty(interner, args);
        (element_count, field_ty)
    }

    pub fn simd_size_and_type(self, interner: I) -> (u64, Ty<I>) {
        let ty::Adt(def, args) = self.kind() else {
            panic!("`simd_size_and_type` called on invalid type")
        };
        assert!(def.repr_is_simd(), "`simd_size_and_type` called on non-SIMD type");
        assert_eq!(def.non_enum_variant().fields_len(), 1);

        let field_ty = def.non_enum_variant().field_zero_ty(interner, args);

        let ty::Array(f0_elem_ty, f0_len) = field_ty.kind() else {
            panic!("Simd type has non-array field type {field_ty:?}")
        };
        // FIXME(repr_simd): https://github.com/rust-lang/rust/pull/78863#discussion_r522784112
        // The way we evaluate the `N` in `[T; N]` here only works since we use
        // `simd_size_and_type` post-monomorphization. It will probably start to ICE
        // if we use it in generic code. See the `simd-array-trait` ui test.
        (
            f0_len
                .try_to_target_usize(interner)
                .expect("expected SIMD field to have definite array size"),
            f0_elem_ty,
        )
    }

    pub fn pinned_ty(self) -> Option<Ty<I>> {
        match self.kind() {
            ty::Adt(def, args) if def.is_pin() => Some(args.type_at(0)),
            _ => None,
        }
    }

    /// Returns the type, pinnedness, mutability, and the region of a reference (`&T` or `&mut T`)
    /// or a pinned-reference type (`Pin<&T>` or `Pin<&mut T>`).
    ///
    /// Regarding the [`pin_ergonomics`] feature, one of the goals is to make pinned references
    /// (`Pin<&T>` and `Pin<&mut T>`) behaves similar to normal references (`&T` and `&mut T`).
    /// This function is useful when references and pinned references are processed similarly.
    ///
    /// [`pin_ergonomics`]: https://github.com/rust-lang/rust/issues/130494
    pub fn maybe_pinned_ref(self) -> Option<(Ty<I>, ty::Pinnedness, ty::Mutability, I::Region)> {
        match self.kind() {
            ty::Adt(def, args)
                if def.is_pin()
                    && let ty::Ref(region, ty, mutbl) = args.type_at(0).kind() =>
            {
                Some((ty, ty::Pinnedness::Pinned, mutbl, region))
            }
            ty::Ref(region, ty, mutbl) => Some((ty, ty::Pinnedness::Not, mutbl, region)),
            _ => None,
        }
    }

    /// Panics if called on any type other than `Box<T>`.
    pub fn expect_boxed_ty(self) -> Ty<I> {
        self.boxed_ty()
            .unwrap_or_else(|| panic!("`expect_boxed_ty` is called on non-box type {:?}", self))
    }

    /// A scalar type is one that denotes an atomic datum, with no sub-components.
    /// (A RawPtr is scalar because it represents a non-managed pointer, so its
    /// contents are abstract to rustc.)
    #[inline]
    pub fn is_scalar(self) -> bool {
        matches!(
            self.kind(),
            ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Float(_)
                | ty::Uint(_)
                | ty::FnDef(..)
                | ty::FnPtr(..)
                | ty::RawPtr(_, _)
                | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        )
    }

    #[inline]
    pub fn is_enum(self) -> bool {
        matches!(self.kind(), ty::Adt(adt_def, _) if adt_def.is_enum())
    }

    #[inline]
    pub fn is_union(self) -> bool {
        matches!(self.kind(), ty::Adt(adt_def, _) if adt_def.is_union())
    }

    #[inline]
    pub fn is_closure(self) -> bool {
        matches!(self.kind(), ty::Closure(..))
    }

    #[inline]
    pub fn is_coroutine(self) -> bool {
        matches!(self.kind(), ty::Coroutine(..))
    }

    #[inline]
    pub fn is_coroutine_closure(self) -> bool {
        matches!(self.kind(), ty::CoroutineClosure(..))
    }

    #[inline]
    pub fn is_fresh_ty(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::FreshTy(_)))
    }

    #[inline]
    pub fn is_fresh(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)))
    }

    #[inline]
    pub fn is_char(self) -> bool {
        matches!(self.kind(), ty::Char)
    }

    #[inline]
    pub fn is_numeric(self) -> bool {
        self.is_integral() || self.is_floating_point()
    }

    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(self.kind(), ty::Int(_))
    }

    #[inline]
    pub fn is_ptr_sized_integral(self) -> bool {
        matches!(self.kind(), ty::Int(ty::IntTy::Isize) | ty::Uint(ty::UintTy::Usize))
    }

    #[inline]
    pub fn has_concrete_skeleton(self) -> bool {
        !matches!(self.kind(), ty::Param(_) | ty::Infer(_) | ty::Error(_))
    }

    /// Checks whether a type recursively contains another type
    ///
    /// Example: `Option<()>` contains `()`
    pub fn contains(self, other: Ty<I>) -> bool {
        struct ContainsTyVisitor<I: Interner>(Ty<I>);

        impl<I: Interner> TypeVisitor<I> for ContainsTyVisitor<I> {
            type Result = ControlFlow<()>;

            fn visit_ty(&mut self, t: Ty<I>) -> Self::Result {
                if self.0 == t { ControlFlow::Break(()) } else { t.super_visit_with(self) }
            }
        }

        let cf = self.visit_with(&mut ContainsTyVisitor(other));
        cf.is_break()
    }

    /// Checks whether a type recursively contains any closure
    ///
    /// Example: `Option<{closure@file.rs:4:20}>` returns true
    pub fn contains_closure(self) -> bool {
        struct ContainsClosureVisitor;

        impl<I: Interner> TypeVisitor<I> for ContainsClosureVisitor {
            type Result = ControlFlow<()>;

            fn visit_ty(&mut self, t: Ty<I>) -> Self::Result {
                if let ty::Closure(..) = t.kind() {
                    ControlFlow::Break(())
                } else {
                    t.super_visit_with(self)
                }
            }
        }

        let cf = self.visit_with(&mut ContainsClosureVisitor);
        cf.is_break()
    }

    /// Returns the deepest `async_drop_in_place::{closure}` implementation.
    ///
    /// `async_drop_in_place<T>::{closure}`, when T is a coroutine, is a proxy-impl
    /// to call async drop poll from impl coroutine.
    pub fn find_async_drop_impl_coroutine<F: FnMut(Ty<I>)>(self, interner: I, mut f: F) -> Ty<I> {
        assert!(self.is_coroutine());
        let mut cor_ty = self;
        let mut ty = cor_ty;
        loop {
            let ty::Coroutine(def_id, args) = ty.kind() else { return cor_ty };
            cor_ty = ty;
            f(ty);
            if !interner.is_async_drop_in_place_coroutine(def_id.into()) {
                return cor_ty;
            }
            ty = args.first().unwrap().expect_ty();
        }
    }

    /// Returns the type of `*ty`.
    ///
    /// The parameter `explicit` indicates if this is an *explicit* dereference.
    /// Some types -- notably raw ptrs -- can only be dereferenced explicitly.
    pub fn builtin_deref(self, explicit: bool) -> Option<Ty<I>> {
        match self.kind() {
            _ if let Some(boxed) = self.boxed_ty() => Some(boxed),
            ty::Ref(_, ty, _) => Some(ty),
            ty::RawPtr(ty, _) if explicit => Some(ty),
            _ => None,
        }
    }

    /// Returns the type of `ty[i]`.
    pub fn builtin_index(self) -> Option<Ty<I>> {
        match self.kind() {
            ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
            _ => None,
        }
    }

    #[inline]
    pub fn is_fn(self) -> bool {
        matches!(self.kind(), ty::FnDef(..) | ty::FnPtr(..))
    }

    #[inline]
    pub fn is_impl_trait(self) -> bool {
        matches!(self.kind(), ty::Alias(ty::Opaque, ..))
    }

    #[inline]
    pub fn ty_adt_def(self) -> Option<I::AdtDef> {
        match self.kind() {
            ty::Adt(adt, _) => Some(adt),
            _ => None,
        }
    }

    /// Returns a list of tuple type arguments
    ///
    /// Panics when called on anything but a tuple.
    #[inline]
    pub fn tuple_fields(self) -> I::Tys {
        match self.kind() {
            ty::Tuple(tys) => tys,
            _ => panic!("tuple_fields called on non-tuple: {self:?}"),
        }
    }

    /// Returns a list of tuple type arguments, or `None` if `self` isn't a tuple.
    #[inline]
    pub fn opt_tuple_fields(self) -> Option<I::Tys> {
        match self.kind() {
            ty::Tuple(args) => Some(args),
            _ => None,
        }
    }

    /// If the type contains variants, returns the valid range of variant indices.
    //
    // FIXME: This requires the optimized MIR in the case of coroutines.
    #[inline]
    pub fn variant_range(self, interner: I) -> Option<Range<I::VariantIdx>> {
        match self.kind() {
            TyKind::Adt(adt, _) => Some(adt.variant_range()),
            TyKind::Coroutine(def_id, args) => {
                // @PROBLEM - I am not sure `coroutine_variant_range(...)` should be on `interner`
                let coroutine_args =
                    interner.coroutine_variant_range(def_id.into(), args.as_coroutine());
                Some(coroutine_args)
            }
            _ => None,
        }
    }

    /// If the type contains variants, returns the variant for `variant_index`.
    /// Panics if `variant_index` is out of range.
    //
    // FIXME: This requires the optimized MIR in the case of coroutines.
    #[inline]
    pub fn discriminant_for_variant(
        self,
        interner: I,
        variant_index: I::VariantIdx,
    ) -> Option<I::Discr> {
        match self.kind() {
            TyKind::Adt(adt, _) if adt.is_enum() => {
                Some(adt.discriminant_for_variant(interner, variant_index))
            }
            TyKind::Coroutine(def_id, args) => Some(args.coroutine_discriminant_for_variant(
                def_id.into(),
                interner,
                variant_index,
            )),
            _ => None,
        }
    }

    #[inline]
    pub fn is_mutable_ptr(self) -> bool {
        matches!(self.kind(), ty::RawPtr(_, Mutability::Mut) | ty::Ref(_, _, Mutability::Mut))
    }

    /// Get the mutability of the reference or `None` when not a reference
    #[inline]
    pub fn ref_mutability(self) -> Option<Mutability> {
        match self.kind() {
            ty::Ref(_, _, mutability) => Some(mutability),
            _ => None,
        }
    }

    #[inline]
    pub fn is_raw_ptr(self) -> bool {
        matches!(self.kind(), ty::RawPtr(_, _))
    }

    /// Tests if this is any kind of primitive pointer type (reference, raw pointer, fn pointer).
    /// `Box` is *not* considered a pointer here!
    #[inline]
    pub fn is_any_ptr(self) -> bool {
        self.is_ref() || self.is_raw_ptr() || self.is_fn_ptr()
    }

    #[inline]
    pub fn is_box(self) -> bool {
        match self.kind() {
            ty::Adt(def, _) => def.is_box(),
            _ => false,
        }
    }

    /// Tests whether this is a Box definitely using the global allocator.
    ///
    /// If the allocator is still generic, the answer is `false`, but it may
    /// later turn out that it does use the global allocator.
    #[inline]
    pub fn is_box_global(self, interner: I) -> bool {
        match self.kind() {
            ty::Adt(def, args) if def.is_box() => {
                let Some(alloc) = args.get(1) else {
                    // Single-argument Box is always global. (for "minicore" tests)
                    return true;
                };
                alloc.expect_ty().ty_adt_def().is_some_and(|alloc_adt| {
                    interner.is_lang_item(alloc_adt.def_id().into(), SolverLangItem::GlobalAlloc)
                })
            }
            _ => false,
        }
    }

    pub fn boxed_ty(self) -> Option<Ty<I>> {
        match self.kind() {
            ty::Adt(def, args) if def.is_box() => Some(args.type_at(0)),
            _ => None,
        }
    }

    #[inline]
    pub fn is_ty_error(self) -> bool {
        matches!(self.kind(), ty::Error(_))
    }

    /// Returns `true` if this type is a floating point type.
    #[inline]
    pub fn is_floating_point(self) -> bool {
        matches!(self.kind(), ty::Float(_) | ty::Infer(ty::FloatVar(_)))
    }

    #[inline]
    pub fn is_trait(self) -> bool {
        matches!(self.kind(), ty::Dynamic(_, _))
    }

    #[inline]
    pub fn is_integral(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::IntVar(_)) | ty::Int(_) | ty::Uint(_))
    }

    #[inline]
    pub fn is_fn_ptr(self) -> bool {
        matches!(self.kind(), ty::FnPtr(..))
    }

    #[inline]
    pub fn has_unsafe_fields(self) -> bool {
        match self.kind() {
            ty::Adt(adt_def, _) => adt_def.has_unsafe_fields(),
            _ => false,
        }
    }

    #[inline]
    #[tracing::instrument(level = "trace", skip(interner))]
    pub fn fn_sig(self, interner: I) -> ty::Binder<I, ty::FnSig<I>> {
        self.kind().fn_sig(interner)
    }

    /// Returns the type of the discriminant of this type.
    #[inline]
    pub fn discriminant_ty(self, interner: I) -> Ty<I> {
        match self.kind() {
            ty::Adt(adt, _) if adt.is_enum() => adt.repr_discr_type_to_ty(interner),
            ty::Coroutine(_, args) => args.as_coroutine_discr_ty(interner),

            ty::Param(_) | ty::Alias(..) | ty::Infer(ty::TyVar(_)) => {
                let mut assoc_items = interner
                    .associated_type_def_ids(
                        interner.require_trait_lang_item(SolverTraitLangItem::DiscriminantKind),
                    )
                    .into_iter();
                Ty::new_projection_from_args(
                    interner,
                    assoc_items.next().expect("DiscriminantKind should have an associated type"),
                    interner.mk_args(&[self.into()]),
                )
            }

            ty::Pat(ty, _) => ty.discriminant_ty(interner),

            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(..)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::UnsafeBinder(_)
            | ty::Error(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_)) => interner.u8_type(),

            ty::Bound(..)
            | ty::Placeholder(_)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                panic!("`discriminant_ty` applied to unexpected type: {:?}", self)
            }
        }
    }

    /// Returns the type of metadata for (potentially wide) pointers to this type,
    /// or the struct tail if the metadata type cannot be determined.
    pub fn ptr_metadata_ty_or_tail(
        self,
        interner: I,
        cause: &I::ObligationCause,
        normalize: impl FnMut(Ty<I>) -> Ty<I>,
    ) -> Result<Ty<I>, Ty<I>> {
        let tail = interner.struct_tail_raw(self, cause, normalize, || {});
        match tail.kind() {
            // Sized types
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Never
            | ty::Error(_)
            // Extern types have metadata = ().
            | ty::Foreign(..)
            // If returned by `struct_tail_raw` this is a unit struct
            // without any fields, or not a struct, and therefore is Sized.
            | ty::Adt(..)
            // If returned by `struct_tail_raw` this is the empty tuple,
            // a.k.a. unit type, which is Sized
            | ty::Tuple(..) => Ok(interner.unit_type()),

            ty::Str | ty::Slice(_) => Ok(interner.usize_type()),

            ty::Dynamic(_, _) => {
                let dyn_metadata = interner.require_lang_item(SolverLangItem::DynMetadata);
                Ok(interner.type_of(dyn_metadata).instantiate(interner, &[tail.into()]))
            }

            // We don't know the metadata of `self`, but it must be equal to the
            // metadata of `tail`.
            ty::Param(_) | ty::Alias(..) => Err(tail),

            | ty::UnsafeBinder(_) => todo!("FIXME(unsafe_binder)"),

            ty::Infer(ty::TyVar(_))
            | ty::Pat(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => panic!(
                "`ptr_metadata_ty_or_tail` applied to unexpected type: {self:?} (tail = {tail:?})"
            ),
        }
    }

    /// Returns the type of metadata for (potentially wide) pointers to this type.
    /// Causes an ICE if the metadata type cannot be determined.
    pub fn ptr_metadata_ty(
        self,
        interner: I,
        cause: &I::ObligationCause,
        normalize: impl FnMut(Ty<I>) -> Ty<I>,
    ) -> Ty<I> {
        match self.ptr_metadata_ty_or_tail(interner, cause, normalize) {
            Ok(metadata) => metadata,
            Err(tail) => panic!(
                "`ptr_metadata_ty` failed to get metadata for type: {self:?} (tail = {tail:?})"
            ),
        }
    }

    /// Given a pointer or reference type, returns the type of the *pointee*'s
    /// metadata. If it can't be determined exactly (perhaps due to still
    /// being generic) then a projection through `ptr::Pointee` will be returned.
    ///
    /// This is particularly useful for getting the type of the result of
    /// [`UnOp::PtrMetadata`](crate::mir::UnOp::PtrMetadata).
    ///
    /// Panics if `self` is not dereferenceable.
    #[track_caller]
    pub fn pointee_metadata_ty_or_projection(
        self,
        cause: &I::ObligationCause,
        interner: I,
    ) -> Ty<I> {
        let Some(pointee_ty) = self.builtin_deref(true) else {
            panic!("Type {self:?} is not a pointer or reference type")
        };
        if pointee_ty.has_trivial_sizedness(interner, SizedTraitKind::Sized) {
            interner.unit_type()
        } else {
            match pointee_ty.ptr_metadata_ty_or_tail(interner, cause, |x| x) {
                Ok(metadata_ty) => metadata_ty,
                Err(tail_ty) => {
                    let metadata_def_id = interner.require_lang_item(SolverLangItem::Metadata);
                    Ty::new_projection(interner, metadata_def_id, [tail_ty])
                }
            }
        }
    }

    /// When we create a closure, we record its kind (i.e., what trait
    /// it implements, constrained by how it uses its borrows) into its
    /// [`ty::ClosureArgs`] or [`ty::CoroutineClosureArgs`] using a type
    /// parameter. This is kind of a phantom type, except that the
    /// most convenient thing for us to are the integral types. This
    /// function converts such a special type into the closure
    /// kind. To go the other way, use [`Ty::from_closure_kind`].
    ///
    /// Note that during type checking, we use an inference variable
    /// to represent the closure kind, because it has not yet been
    /// inferred. Once upvar inference (in `rustc_hir_analysis/src/check/upvar.rs`)
    /// is complete, that type variable will be unified with one of
    /// the integral types.
    ///
    /// ```rust,ignore (snippet of compiler code)
    /// if let TyKind::Closure(def_id, args) = closure_ty.kind()
    ///     && let Some(closure_kind) = args.as_closure().kind_ty().to_opt_closure_kind()
    /// {
    ///     println!("{closure_kind:?}");
    /// } else if let TyKind::CoroutineClosure(def_id, args) = closure_ty.kind()
    ///     && let Some(closure_kind) = args.as_coroutine_closure().kind_ty().to_opt_closure_kind()
    /// {
    ///     println!("{closure_kind:?}");
    /// }
    /// ```
    ///
    /// After upvar analysis, you should instead use [`ty::ClosureArgs::kind()`]
    /// or [`ty::CoroutineClosureArgs::kind()`] to assert that the `ClosureKind`
    /// has been constrained instead of manually calling this method.
    ///
    /// ```rust,ignore (snippet of compiler code)
    /// if let TyKind::Closure(def_id, args) = closure_ty.kind()
    /// {
    ///     println!("{:?}", args.as_closure().kind());
    /// } else if let TyKind::CoroutineClosure(def_id, args) = closure_ty.kind()
    /// {
    ///     println!("{:?}", args.as_coroutine_closure().kind());
    /// }
    /// ```
    pub fn to_opt_closure_kind(self) -> Option<ty::ClosureKind> {
        match self.kind() {
            ty::Int(int_ty) => match int_ty {
                ty::IntTy::I8 => Some(ty::ClosureKind::Fn),
                ty::IntTy::I16 => Some(ty::ClosureKind::FnMut),
                ty::IntTy::I32 => Some(ty::ClosureKind::FnOnce),
                _ => panic!("cannot convert type `{:?}` to a closure kind", self),
            },

            // "Bound" types appear in canonical queries when the
            // closure type is not yet known, and `Placeholder` and `Param`
            // may be encountered in generic `AsyncFnKindHelper` goals.
            ty::Bound(..) | ty::Placeholder(_) | ty::Param(_) | ty::Infer(_) => None,

            ty::Error(_) => Some(ty::ClosureKind::Fn),

            _ => panic!("cannot convert type `{:?}` to a closure kind", self),
        }
    }

    /// Inverse of [`Ty::to_opt_closure_kind`]. See docs on that method
    /// for explanation of the relationship between `Ty` and [`ty::ClosureKind`].
    pub fn from_closure_kind(interner: I, kind: ty::ClosureKind) -> Ty<I> {
        match kind {
            ty::ClosureKind::Fn => interner.i8_type(),
            ty::ClosureKind::FnMut => interner.i16_type(),
            ty::ClosureKind::FnOnce => interner.i32_type(),
        }
    }

    /// Like [`Ty::to_opt_closure_kind`], but it caps the "maximum" closure kind
    /// to `FnMut`. This is because although we have three capability states,
    /// `AsyncFn`/`AsyncFnMut`/`AsyncFnOnce`, we only need to distinguish two coroutine
    /// bodies: by-ref and by-value.
    ///
    /// See the definition of `AsyncFn` and `AsyncFnMut` and the `CallRefFuture`
    /// associated type for why we don't distinguish [`ty::ClosureKind::Fn`] and
    /// [`ty::ClosureKind::FnMut`] for the purpose of the generated MIR bodies.
    ///
    /// This method should be used when constructing a `Coroutine` out of a
    /// `CoroutineClosure`, when the `Coroutine`'s `kind` field is being populated
    /// directly from the `CoroutineClosure`'s `kind`.
    pub fn from_coroutine_closure_kind(interner: I, kind: ty::ClosureKind) -> Ty<I> {
        match kind {
            ty::ClosureKind::Fn | ty::ClosureKind::FnMut => interner.i16_type(),
            ty::ClosureKind::FnOnce => interner.i32_type(),
        }
    }

    /// Fast path helper for testing if a type is `Sized` or `MetaSized`.
    ///
    /// Returning true means the type is known to implement the sizedness trait. Returning `false`
    /// means nothing -- could be sized, might not be.
    ///
    /// Note that we could never rely on the fact that a type such as `[_]` is trivially `!Sized`
    /// because we could be in a type environment with a bound such as `[_]: Copy`. A function with
    /// such a bound obviously never can be called, but that doesn't mean it shouldn't typecheck.
    /// This is why this method doesn't return `Option<bool>`.
    #[instrument(skip(interner), level = "debug")]
    pub fn has_trivial_sizedness(self, interner: I, sizedness: SizedTraitKind) -> bool {
        match self.kind() {
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::UnsafeBinder(_)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Array(..)
            | ty::Pat(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Never
            | ty::Error(_) => true,

            ty::Str | ty::Slice(_) | ty::Dynamic(_, _) => match sizedness {
                SizedTraitKind::Sized => false,
                SizedTraitKind::MetaSized => true,
            },

            ty::Foreign(..) => match sizedness {
                SizedTraitKind::Sized | SizedTraitKind::MetaSized => false,
            },

            ty::Tuple(tys) => {
                tys.last().is_none_or(|ty| ty.has_trivial_sizedness(interner, sizedness))
            }

            ty::Adt(def, args) => def.sizedness_constraint(interner, sizedness).is_none_or(|ty| {
                ty.instantiate(interner, args).has_trivial_sizedness(interner, sizedness)
            }),

            ty::Alias(..) | ty::Param(_) | ty::Placeholder(..) | ty::Bound(..) => false,

            ty::Infer(ty::TyVar(_)) => false,

            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                panic!("`has_trivial_sizedness` applied to unexpected type: {:?}", self)
            }
        }
    }

    /// Fast path helper for primitives which are always `Copy` and which
    /// have a side-effect-free `Clone` impl.
    ///
    /// Returning true means the type is known to be pure and `Copy+Clone`.
    /// Returning `false` means nothing -- could be `Copy`, might not be.
    ///
    /// This is mostly useful for optimizations, as these are the types
    /// on which we can replace cloning with dereferencing.
    pub fn is_trivially_pure_clone_copy(self) -> bool {
        match self.kind() {
            ty::Bool | ty::Char | ty::Never => true,

            // These aren't even `Clone`
            ty::Str | ty::Slice(..) | ty::Foreign(..) | ty::Dynamic(..) => false,

            ty::Infer(ty::InferTy::FloatVar(_) | ty::InferTy::IntVar(_))
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..) => true,

            // ZST which can't be named are fine.
            ty::FnDef(..) => true,

            ty::Array(element_ty, _len) => element_ty.is_trivially_pure_clone_copy(),

            // A 100-tuple isn't "trivial", so doing this only for reasonable sizes.
            ty::Tuple(field_tys) => {
                field_tys.len() <= 3 && field_tys.iter().all(Self::is_trivially_pure_clone_copy)
            }

            ty::Pat(ty, _) => ty.is_trivially_pure_clone_copy(),

            // Sometimes traits aren't implemented for every ABI or arity,
            // because we can't be generic over everything yet.
            ty::FnPtr(..) => false,

            // Definitely absolutely not copy.
            ty::Ref(_, _, Mutability::Mut) => false,

            // The standard library has a blanket Copy impl for shared references and raw pointers,
            // for all unsized types.
            ty::Ref(_, _, Mutability::Not) | ty::RawPtr(..) => true,

            ty::Coroutine(..) | ty::CoroutineWitness(..) => false,

            // Might be, but not "trivial" so just giving the safe answer.
            ty::Adt(..) | ty::Closure(..) | ty::CoroutineClosure(..) => false,

            ty::UnsafeBinder(_) => false,

            // Needs normalisation or revealing to determine, so no is the safe answer.
            ty::Alias(..) => false,

            ty::Param(..) | ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) | ty::Error(..) => {
                false
            }
        }
    }

    pub fn is_trivially_wf(self, interner: I) -> bool {
        match self.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never
            | ty::Param(_)
            | ty::Placeholder(_)
            | ty::Bound(..) => true,

            ty::Slice(ty) => {
                ty.is_trivially_wf(interner)
                    && ty.has_trivial_sizedness(interner, SizedTraitKind::Sized)
            }
            ty::RawPtr(ty, _) => ty.is_trivially_wf(interner),

            ty::FnPtr(sig_tys, _) => sig_tys
                .skip_binder()
                .inputs_and_output
                .iter()
                .all(|ty: Ty<I>| ty.is_trivially_wf(interner)),
            ty::Ref(_, ty, _) => ty.is_global() && ty.is_trivially_wf(interner),

            ty::Infer(infer) => match infer {
                ty::TyVar(_) => false,
                ty::IntVar(_) | ty::FloatVar(_) => true,
                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => true,
            },

            ty::Adt(_, _)
            | ty::Tuple(_)
            | ty::Array(..)
            | ty::Foreign(_)
            | ty::Pat(_, _)
            | ty::FnDef(..)
            | ty::UnsafeBinder(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Alias(..)
            | ty::Error(_) => false,
        }
    }

    /// If `self` is a primitive, return its [`Symbol`].
    pub fn primitive_symbol(self) -> Option<I::Symbol>
    where
        I::Symbol: From<rustc_span::Symbol>,
    {
        match self.kind() {
            ty::Bool => Some(sym::bool.into()),
            ty::Char => Some(sym::char.into()),
            ty::Float(f) => match f {
                ty::FloatTy::F16 => Some(sym::f16.into()),
                ty::FloatTy::F32 => Some(sym::f32.into()),
                ty::FloatTy::F64 => Some(sym::f64.into()),
                ty::FloatTy::F128 => Some(sym::f128.into()),
            },
            ty::Int(f) => match f {
                ty::IntTy::Isize => Some(sym::isize.into()),
                ty::IntTy::I8 => Some(sym::i8.into()),
                ty::IntTy::I16 => Some(sym::i16.into()),
                ty::IntTy::I32 => Some(sym::i32.into()),
                ty::IntTy::I64 => Some(sym::i64.into()),
                ty::IntTy::I128 => Some(sym::i128.into()),
            },
            ty::Uint(f) => match f {
                ty::UintTy::Usize => Some(sym::usize.into()),
                ty::UintTy::U8 => Some(sym::u8.into()),
                ty::UintTy::U16 => Some(sym::u16.into()),
                ty::UintTy::U32 => Some(sym::u32.into()),
                ty::UintTy::U64 => Some(sym::u64.into()),
                ty::UintTy::U128 => Some(sym::u128.into()),
            },
            ty::Str => Some(sym::str.into()),
            _ => None,
        }
    }

    pub fn is_c_void(self, interner: I) -> bool {
        match self.kind() {
            ty::Adt(adt, _) => interner.is_c_void(adt),
            _ => false,
        }
    }

    pub fn is_async_drop_in_place_coroutine(self, interner: I) -> bool {
        match self.kind() {
            ty::Coroutine(def, ..) => interner.is_async_drop_in_place_coroutine(def.into()),
            _ => false,
        }
    }

    /// Returns `true` when the outermost type cannot be further normalized,
    /// resolved, or instantiated. This includes all primitive types, but also
    /// things like ADTs and trait objects, since even if their arguments or
    /// nested types may be further simplified, the outermost [`TyKind`] or
    /// type constructor remains the same.
    #[inline]
    pub fn is_known_rigid(self) -> bool {
        self.kind().is_known_rigid()
    }

    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(self) -> TypeWalker<I> {
        TypeWalker::new(self.into())
    }

    #[inline]
    pub fn is_guaranteed_unsized_raw(self) -> bool {
        match self.kind() {
            ty::Dynamic(_, _) | ty::Slice(_) | ty::Str => true,
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_, _)
            | ty::UnsafeBinder(_)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Placeholder(_)
            | ty::Infer(_)
            | ty::Error(_) => false,
        }
    }

    #[inline]
    pub fn new_int(interner: I, i: ty::IntTy) -> Self {
        use ty::IntTy::*;
        let int_ty = match i {
            Isize => ty::Int(ty::IntTy::Isize),
            I8 => ty::Int(ty::IntTy::I8),
            I16 => ty::Int(ty::IntTy::I16),
            I32 => ty::Int(ty::IntTy::I32),
            I64 => ty::Int(ty::IntTy::I64),
            I128 => ty::Int(ty::IntTy::I128),
        };
        Self::new(interner, int_ty)
    }

    #[inline]
    pub fn new_uint(interner: I, ui: ty::UintTy) -> Self {
        use ty::UintTy::*;
        let unsigned_int_ty = match ui {
            Usize => ty::Uint(ty::UintTy::Usize),
            U8 => ty::Uint(ty::UintTy::U8),
            U16 => ty::Uint(ty::UintTy::U16),
            U32 => ty::Uint(ty::UintTy::U32),
            U64 => ty::Uint(ty::UintTy::U64),
            U128 => ty::Uint(ty::UintTy::U128),
        };
        Self::new(interner, unsigned_int_ty)
    }

    #[inline]
    pub fn new_float(interner: I, f: ty::FloatTy) -> Self {
        use ty::FloatTy::*;
        let float_ty = match f {
            F16 => ty::Float(ty::FloatTy::F16),
            F32 => ty::Float(ty::FloatTy::F32),
            F64 => ty::Float(ty::FloatTy::F64),
            F128 => ty::Float(ty::FloatTy::F128),
        };
        Self::new(interner, float_ty)
    }

    #[inline]
    pub fn new_field_representing_type(
        interner: I,
        base: Self,
        variant: VariantIdx,
        field: FieldIdx,
    ) -> Self {
        interner.new_field_representing_type(base, variant, field)
    }
}
