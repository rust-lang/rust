//! Things related to generic args in the next-trait-solver (`GenericArg`, `GenericArgs`, `Term`).
//!
//! Implementations of `GenericArg` and `Term` are pointer-tagged instead of an enum (rustc does
//! the same). This is done to save memory (which also helps speed) - one `GenericArg` is a machine
//! word instead of two, while matching on it is basically as cheap. The implementation for both
//! `GenericArg` and `Term` is shared in [`GenericArgImpl`]. This both simplifies the implementation,
//! as well as enables a noop conversion from `Term` to `GenericArg`.

use std::{hint::unreachable_unchecked, marker::PhantomData, ptr::NonNull};

use hir_def::{GenericDefId, GenericParamId};
use intern::InternedRef;
use rustc_type_ir::{
    ClosureArgs, ConstVid, CoroutineArgs, CoroutineClosureArgs, FallibleTypeFolder,
    GenericTypeVisitable, Interner, TyVid, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor,
    Variance,
    inherent::{GenericArg as _, GenericsOf, IntoKind, SliceLike, Term as _, Ty as _},
    relate::{Relate, VarianceDiagInfo},
    walk::TypeWalker,
};
use smallvec::SmallVec;

use crate::next_solver::{
    ConstInterned, RegionInterned, TyInterned, impl_foldable_for_interned_slice, interned_slice,
};

use super::{
    Const, DbInterner, EarlyParamRegion, ErrorGuaranteed, ParamConst, Region, SolverDefId, Ty,
    generics::Generics,
};

pub type GenericArgKind<'db> = rustc_type_ir::GenericArgKind<DbInterner<'db>>;
pub type TermKind<'db> = rustc_type_ir::TermKind<DbInterner<'db>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GenericArgImpl<'db> {
    /// # Invariant
    ///
    /// Contains an [`InternedRef`] of a [`Ty`], [`Const`] or [`Region`], bit-tagged as per the consts below.
    ptr: NonNull<()>,
    _marker: PhantomData<(Ty<'db>, Const<'db>, Region<'db>)>,
}

// SAFETY: We essentially own the `Ty`, `Const` or `Region`, and they are `Send + Sync`.
unsafe impl Send for GenericArgImpl<'_> {}
unsafe impl Sync for GenericArgImpl<'_> {}

impl<'db> GenericArgImpl<'db> {
    const KIND_MASK: usize = 0b11;
    const PTR_MASK: usize = !Self::KIND_MASK;
    const TY_TAG: usize = 0b00;
    const CONST_TAG: usize = 0b01;
    const REGION_TAG: usize = 0b10;

    #[inline]
    fn new_ty(ty: Ty<'db>) -> Self {
        Self {
            // SAFETY: We create it from an `InternedRef`, and it's never null.
            ptr: unsafe {
                NonNull::new_unchecked(
                    ty.interned
                        .as_raw()
                        .cast::<()>()
                        .cast_mut()
                        .map_addr(|addr| addr | Self::TY_TAG),
                )
            },
            _marker: PhantomData,
        }
    }

    #[inline]
    fn new_const(ty: Const<'db>) -> Self {
        Self {
            // SAFETY: We create it from an `InternedRef`, and it's never null.
            ptr: unsafe {
                NonNull::new_unchecked(
                    ty.interned
                        .as_raw()
                        .cast::<()>()
                        .cast_mut()
                        .map_addr(|addr| addr | Self::CONST_TAG),
                )
            },
            _marker: PhantomData,
        }
    }

    #[inline]
    fn new_region(ty: Region<'db>) -> Self {
        Self {
            // SAFETY: We create it from an `InternedRef`, and it's never null.
            ptr: unsafe {
                NonNull::new_unchecked(
                    ty.interned
                        .as_raw()
                        .cast::<()>()
                        .cast_mut()
                        .map_addr(|addr| addr | Self::REGION_TAG),
                )
            },
            _marker: PhantomData,
        }
    }

    #[inline]
    fn kind(self) -> GenericArgKind<'db> {
        let ptr = self.ptr.as_ptr().map_addr(|addr| addr & Self::PTR_MASK);
        // SAFETY: We can only be created from a `Ty`, a `Const` or a `Region`, and the tag will match.
        unsafe {
            match self.ptr.addr().get() & Self::KIND_MASK {
                Self::TY_TAG => GenericArgKind::Type(Ty {
                    interned: InternedRef::from_raw(ptr.cast::<TyInterned>()),
                }),
                Self::CONST_TAG => GenericArgKind::Const(Const {
                    interned: InternedRef::from_raw(ptr.cast::<ConstInterned>()),
                }),
                Self::REGION_TAG => GenericArgKind::Lifetime(Region {
                    interned: InternedRef::from_raw(ptr.cast::<RegionInterned>()),
                }),
                _ => unreachable_unchecked(),
            }
        }
    }

    #[inline]
    fn term_kind(self) -> TermKind<'db> {
        let ptr = self.ptr.as_ptr().map_addr(|addr| addr & Self::PTR_MASK);
        // SAFETY: We can only be created from a `Ty`, a `Const` or a `Region`, and the tag will match.
        // It is the caller's responsibility (encapsulated within this module) to only call this with
        // `Term`, which cannot be constructed from a `Region`.
        unsafe {
            match self.ptr.addr().get() & Self::KIND_MASK {
                Self::TY_TAG => {
                    TermKind::Ty(Ty { interned: InternedRef::from_raw(ptr.cast::<TyInterned>()) })
                }
                Self::CONST_TAG => TermKind::Const(Const {
                    interned: InternedRef::from_raw(ptr.cast::<ConstInterned>()),
                }),
                _ => unreachable_unchecked(),
            }
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct StoredGenericArg {
    ptr: GenericArgImpl<'static>,
}

impl Clone for StoredGenericArg {
    #[inline]
    fn clone(&self) -> Self {
        match self.ptr.kind() {
            GenericArgKind::Lifetime(it) => std::mem::forget(it.interned.to_owned()),
            GenericArgKind::Type(it) => std::mem::forget(it.interned.to_owned()),
            GenericArgKind::Const(it) => std::mem::forget(it.interned.to_owned()),
        }
        Self { ptr: self.ptr }
    }
}

impl Drop for StoredGenericArg {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            match self.ptr.kind() {
                GenericArgKind::Lifetime(it) => it.interned.decrement_refcount(),
                GenericArgKind::Type(it) => it.interned.decrement_refcount(),
                GenericArgKind::Const(it) => it.interned.decrement_refcount(),
            }
        }
    }
}

impl StoredGenericArg {
    #[inline]
    fn new(value: GenericArg<'_>) -> Self {
        let result = Self { ptr: GenericArgImpl { ptr: value.ptr.ptr, _marker: PhantomData } };
        // Increase refcount.
        std::mem::forget(result.clone());
        result
    }

    #[inline]
    pub fn as_ref<'db>(&self) -> GenericArg<'db> {
        GenericArg { ptr: self.ptr }
    }
}

impl std::fmt::Debug for StoredGenericArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct GenericArg<'db> {
    ptr: GenericArgImpl<'db>,
}

impl<'db> std::fmt::Debug for GenericArg<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            GenericArgKind::Type(t) => std::fmt::Debug::fmt(&t, f),
            GenericArgKind::Lifetime(r) => std::fmt::Debug::fmt(&r, f),
            GenericArgKind::Const(c) => std::fmt::Debug::fmt(&c, f),
        }
    }
}

impl<'db> GenericArg<'db> {
    #[inline]
    pub fn store(self) -> StoredGenericArg {
        StoredGenericArg::new(self)
    }

    #[inline]
    pub fn kind(self) -> GenericArgKind<'db> {
        self.ptr.kind()
    }

    pub fn ty(self) -> Option<Ty<'db>> {
        match self.kind() {
            GenericArgKind::Type(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn expect_ty(self) -> Ty<'db> {
        match self.kind() {
            GenericArgKind::Type(ty) => ty,
            _ => panic!("Expected ty, got {self:?}"),
        }
    }

    pub fn konst(self) -> Option<Const<'db>> {
        match self.kind() {
            GenericArgKind::Const(konst) => Some(konst),
            _ => None,
        }
    }

    pub fn region(self) -> Option<Region<'db>> {
        match self.kind() {
            GenericArgKind::Lifetime(r) => Some(r),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn expect_region(self) -> Region<'db> {
        match self.kind() {
            GenericArgKind::Lifetime(region) => region,
            _ => panic!("expected a region, got {self:?}"),
        }
    }

    pub fn error_from_id(interner: DbInterner<'db>, id: GenericParamId) -> GenericArg<'db> {
        match id {
            GenericParamId::TypeParamId(_) => Ty::new_error(interner, ErrorGuaranteed).into(),
            GenericParamId::ConstParamId(_) => Const::error(interner).into(),
            GenericParamId::LifetimeParamId(_) => Region::error(interner).into(),
        }
    }

    #[inline]
    pub fn walk(self) -> TypeWalker<DbInterner<'db>> {
        TypeWalker::new(self)
    }
}

impl<'db> From<Term<'db>> for GenericArg<'db> {
    #[inline]
    fn from(value: Term<'db>) -> Self {
        GenericArg { ptr: value.ptr }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Term<'db> {
    ptr: GenericArgImpl<'db>,
}

impl<'db> std::fmt::Debug for Term<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            TermKind::Ty(t) => std::fmt::Debug::fmt(&t, f),
            TermKind::Const(c) => std::fmt::Debug::fmt(&c, f),
        }
    }
}

impl<'db> Term<'db> {
    #[inline]
    pub fn kind(self) -> TermKind<'db> {
        self.ptr.term_kind()
    }

    pub fn expect_type(&self) -> Ty<'db> {
        self.as_type().expect("expected a type, but found a const")
    }

    pub fn is_trivially_wf(&self, tcx: DbInterner<'db>) -> bool {
        match self.kind() {
            TermKind::Ty(ty) => ty.is_trivially_wf(tcx),
            TermKind::Const(ct) => ct.is_trivially_wf(),
        }
    }
}

impl<'db> From<Ty<'db>> for GenericArg<'db> {
    #[inline]
    fn from(value: Ty<'db>) -> Self {
        GenericArg { ptr: GenericArgImpl::new_ty(value) }
    }
}

impl<'db> From<Region<'db>> for GenericArg<'db> {
    #[inline]
    fn from(value: Region<'db>) -> Self {
        GenericArg { ptr: GenericArgImpl::new_region(value) }
    }
}

impl<'db> From<Const<'db>> for GenericArg<'db> {
    #[inline]
    fn from(value: Const<'db>) -> Self {
        GenericArg { ptr: GenericArgImpl::new_const(value) }
    }
}

impl<'db> IntoKind for GenericArg<'db> {
    type Kind = GenericArgKind<'db>;

    #[inline]
    fn kind(self) -> Self::Kind {
        self.ptr.kind()
    }
}

impl<'db, V> GenericTypeVisitable<V> for GenericArg<'db>
where
    GenericArgKind<'db>: GenericTypeVisitable<V>,
{
    fn generic_visit_with(&self, visitor: &mut V) {
        self.kind().generic_visit_with(visitor);
    }
}

impl<'db, V> GenericTypeVisitable<V> for Term<'db>
where
    TermKind<'db>: GenericTypeVisitable<V>,
{
    fn generic_visit_with(&self, visitor: &mut V) {
        self.kind().generic_visit_with(visitor);
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for GenericArg<'db> {
    fn visit_with<V: TypeVisitor<DbInterner<'db>>>(&self, visitor: &mut V) -> V::Result {
        match self.kind() {
            GenericArgKind::Lifetime(it) => it.visit_with(visitor),
            GenericArgKind::Type(it) => it.visit_with(visitor),
            GenericArgKind::Const(it) => it.visit_with(visitor),
        }
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for Term<'db> {
    fn visit_with<V: TypeVisitor<DbInterner<'db>>>(&self, visitor: &mut V) -> V::Result {
        match self.kind() {
            TermKind::Ty(it) => it.visit_with(visitor),
            TermKind::Const(it) => it.visit_with(visitor),
        }
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for GenericArg<'db> {
    fn try_fold_with<F: FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(match self.kind() {
            GenericArgKind::Lifetime(it) => it.try_fold_with(folder)?.into(),
            GenericArgKind::Type(it) => it.try_fold_with(folder)?.into(),
            GenericArgKind::Const(it) => it.try_fold_with(folder)?.into(),
        })
    }

    fn fold_with<F: TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        match self.kind() {
            GenericArgKind::Lifetime(it) => it.fold_with(folder).into(),
            GenericArgKind::Type(it) => it.fold_with(folder).into(),
            GenericArgKind::Const(it) => it.fold_with(folder).into(),
        }
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for Term<'db> {
    fn try_fold_with<F: FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(match self.kind() {
            TermKind::Ty(it) => it.try_fold_with(folder)?.into(),
            TermKind::Const(it) => it.try_fold_with(folder)?.into(),
        })
    }

    fn fold_with<F: TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        match self.kind() {
            TermKind::Ty(it) => it.fold_with(folder).into(),
            TermKind::Const(it) => it.fold_with(folder).into(),
        }
    }
}

impl<'db> Relate<DbInterner<'db>> for GenericArg<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        match (a.kind(), b.kind()) {
            (GenericArgKind::Lifetime(a_lt), GenericArgKind::Lifetime(b_lt)) => {
                Ok(relation.relate(a_lt, b_lt)?.into())
            }
            (GenericArgKind::Type(a_ty), GenericArgKind::Type(b_ty)) => {
                Ok(relation.relate(a_ty, b_ty)?.into())
            }
            (GenericArgKind::Const(a_ct), GenericArgKind::Const(b_ct)) => {
                Ok(relation.relate(a_ct, b_ct)?.into())
            }
            (GenericArgKind::Lifetime(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (GenericArgKind::Type(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (GenericArgKind::Const(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
        }
    }
}

interned_slice!(
    GenericArgsStorage,
    GenericArgs,
    StoredGenericArgs,
    generic_args,
    GenericArg<'db>,
    GenericArg<'static>,
);
impl_foldable_for_interned_slice!(GenericArgs);

impl<'db> rustc_type_ir::inherent::GenericArg<DbInterner<'db>> for GenericArg<'db> {}

impl<'db> GenericArgs<'db> {
    /// Creates an `GenericArgs` for generic parameter definitions,
    /// by calling closures to obtain each kind.
    /// The closures get to observe the `GenericArgs` as they're
    /// being built, which can be used to correctly
    /// replace defaults of generic parameters.
    pub fn for_item<F>(
        interner: DbInterner<'db>,
        def_id: SolverDefId,
        mut mk_kind: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let defs = interner.generics_of(def_id);
        let count = defs.count();

        if count == 0 {
            return Default::default();
        }

        let mut args = SmallVec::with_capacity(count);
        Self::fill_item(&mut args, interner, defs, &mut mk_kind);
        interner.mk_args(&args)
    }

    /// Creates an all-error `GenericArgs`.
    pub fn error_for_item(interner: DbInterner<'db>, def_id: SolverDefId) -> GenericArgs<'db> {
        GenericArgs::for_item(interner, def_id, |_, id, _| GenericArg::error_from_id(interner, id))
    }

    /// Like `for_item`, but prefers the default of a parameter if it has any.
    pub fn for_item_with_defaults<F>(
        interner: DbInterner<'db>,
        def_id: GenericDefId,
        mut fallback: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let defaults = interner.db.generic_defaults(def_id);
        Self::for_item(interner, def_id.into(), |idx, id, prev| match defaults.get(idx as usize) {
            Some(default) => default.instantiate(interner, prev),
            None => fallback(idx, id, prev),
        })
    }

    /// Like `for_item()`, but calls first uses the args from `first`.
    pub fn fill_rest<F>(
        interner: DbInterner<'db>,
        def_id: SolverDefId,
        first: impl IntoIterator<Item = GenericArg<'db>>,
        mut fallback: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let mut iter = first.into_iter();
        Self::for_item(interner, def_id, |idx, id, prev| {
            iter.next().unwrap_or_else(|| fallback(idx, id, prev))
        })
    }

    /// Appends default param values to `first` if needed. Params without default will call `fallback()`.
    pub fn fill_with_defaults<F>(
        interner: DbInterner<'db>,
        def_id: GenericDefId,
        first: impl IntoIterator<Item = GenericArg<'db>>,
        mut fallback: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let defaults = interner.db.generic_defaults(def_id);
        Self::fill_rest(interner, def_id.into(), first, |idx, id, prev| {
            defaults
                .get(idx as usize)
                .map(|default| default.instantiate(interner, prev))
                .unwrap_or_else(|| fallback(idx, id, prev))
        })
    }

    fn fill_item<F>(
        args: &mut SmallVec<[GenericArg<'db>; 8]>,
        interner: DbInterner<'_>,
        defs: Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        if let Some(def_id) = defs.parent {
            let parent_defs = interner.generics_of(def_id.into());
            Self::fill_item(args, interner, parent_defs, mk_kind);
        }
        Self::fill_single(args, &defs, mk_kind);
    }

    fn fill_single<F>(args: &mut SmallVec<[GenericArg<'db>; 8]>, defs: &Generics, mk_kind: &mut F)
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        args.reserve(defs.own_params.len());
        for param in &defs.own_params {
            let kind = mk_kind(args.len() as u32, param.id, args);
            args.push(kind);
        }
    }

    pub fn types(self) -> impl Iterator<Item = Ty<'db>> {
        self.iter().filter_map(|it| it.as_type())
    }

    pub fn consts(self) -> impl Iterator<Item = Const<'db>> {
        self.iter().filter_map(|it| it.as_const())
    }

    pub fn regions(self) -> impl Iterator<Item = Region<'db>> {
        self.iter().filter_map(|it| it.as_region())
    }
}

impl<'db> rustc_type_ir::relate::Relate<DbInterner<'db>> for GenericArgs<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        GenericArgs::new_from_iter(
            relation.cx(),
            std::iter::zip(a.iter(), b.iter()).map(|(a, b)| {
                relation.relate_with_variance(
                    Variance::Invariant,
                    VarianceDiagInfo::default(),
                    a,
                    b,
                )
            }),
        )
    }
}

impl<'db> rustc_type_ir::inherent::GenericArgs<DbInterner<'db>> for GenericArgs<'db> {
    fn as_closure(self) -> ClosureArgs<DbInterner<'db>> {
        ClosureArgs { args: self }
    }
    fn as_coroutine(self) -> CoroutineArgs<DbInterner<'db>> {
        CoroutineArgs { args: self }
    }
    fn as_coroutine_closure(self) -> CoroutineClosureArgs<DbInterner<'db>> {
        CoroutineClosureArgs { args: self }
    }
    fn rebase_onto(
        self,
        interner: DbInterner<'db>,
        source_def_id: <DbInterner<'db> as rustc_type_ir::Interner>::DefId,
        target: <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs,
    ) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        let defs = interner.generics_of(source_def_id);
        interner.mk_args_from_iter(target.iter().chain(self.iter().skip(defs.count())))
    }

    fn identity_for_item(
        interner: DbInterner<'db>,
        def_id: <DbInterner<'db> as rustc_type_ir::Interner>::DefId,
    ) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        Self::for_item(interner, def_id, |index, kind, _| mk_param(interner, index, kind))
    }

    fn extend_with_error(
        interner: DbInterner<'db>,
        def_id: <DbInterner<'db> as rustc_type_ir::Interner>::DefId,
        original_args: &[<DbInterner<'db> as rustc_type_ir::Interner>::GenericArg],
    ) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        Self::for_item(interner, def_id, |index, kind, _| {
            if let Some(arg) = original_args.get(index as usize) {
                *arg
            } else {
                error_for_param_kind(kind, interner)
            }
        })
    }
    fn type_at(self, i: usize) -> <DbInterner<'db> as rustc_type_ir::Interner>::Ty {
        self.get(i)
            .and_then(|g| g.as_type())
            .unwrap_or_else(|| Ty::new_error(DbInterner::conjure(), ErrorGuaranteed))
    }

    fn region_at(self, i: usize) -> <DbInterner<'db> as rustc_type_ir::Interner>::Region {
        self.get(i)
            .and_then(|g| g.as_region())
            .unwrap_or_else(|| Region::error(DbInterner::conjure()))
    }

    fn const_at(self, i: usize) -> <DbInterner<'db> as rustc_type_ir::Interner>::Const {
        self.get(i)
            .and_then(|g| g.as_const())
            .unwrap_or_else(|| Const::error(DbInterner::conjure()))
    }

    fn split_closure_args(self) -> rustc_type_ir::ClosureArgsParts<DbInterner<'db>> {
        // FIXME: should use `ClosureSubst` when possible
        match self.as_slice() {
            [parent_args @ .., closure_kind_ty, sig_ty, tupled_upvars_ty] => {
                rustc_type_ir::ClosureArgsParts {
                    parent_args,
                    closure_sig_as_fn_ptr_ty: sig_ty.expect_ty(),
                    closure_kind_ty: closure_kind_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => {
                unreachable!("unexpected closure sig");
            }
        }
    }

    fn split_coroutine_closure_args(
        self,
    ) -> rustc_type_ir::CoroutineClosureArgsParts<DbInterner<'db>> {
        match self.as_slice() {
            [
                parent_args @ ..,
                closure_kind_ty,
                signature_parts_ty,
                tupled_upvars_ty,
                coroutine_captures_by_ref_ty,
            ] => rustc_type_ir::CoroutineClosureArgsParts {
                parent_args,
                closure_kind_ty: closure_kind_ty.expect_ty(),
                signature_parts_ty: signature_parts_ty.expect_ty(),
                tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                coroutine_captures_by_ref_ty: coroutine_captures_by_ref_ty.expect_ty(),
            },
            _ => panic!("GenericArgs were likely not for a CoroutineClosure."),
        }
    }

    fn split_coroutine_args(self) -> rustc_type_ir::CoroutineArgsParts<DbInterner<'db>> {
        match self.as_slice() {
            [parent_args @ .., kind_ty, resume_ty, yield_ty, return_ty, tupled_upvars_ty] => {
                rustc_type_ir::CoroutineArgsParts {
                    parent_args,
                    kind_ty: kind_ty.expect_ty(),
                    resume_ty: resume_ty.expect_ty(),
                    yield_ty: yield_ty.expect_ty(),
                    return_ty: return_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => panic!("GenericArgs were likely not for a Coroutine."),
        }
    }
}

pub fn mk_param<'db>(interner: DbInterner<'db>, index: u32, id: GenericParamId) -> GenericArg<'db> {
    match id {
        GenericParamId::LifetimeParamId(id) => {
            Region::new_early_param(interner, EarlyParamRegion { index, id }).into()
        }
        GenericParamId::TypeParamId(id) => Ty::new_param(interner, id, index).into(),
        GenericParamId::ConstParamId(id) => {
            Const::new_param(interner, ParamConst { index, id }).into()
        }
    }
}

pub fn error_for_param_kind<'db>(id: GenericParamId, interner: DbInterner<'db>) -> GenericArg<'db> {
    match id {
        GenericParamId::LifetimeParamId(_) => Region::error(interner).into(),
        GenericParamId::TypeParamId(_) => Ty::new_error(interner, ErrorGuaranteed).into(),
        GenericParamId::ConstParamId(_) => Const::error(interner).into(),
    }
}

impl<'db> IntoKind for Term<'db> {
    type Kind = TermKind<'db>;

    #[inline]
    fn kind(self) -> Self::Kind {
        self.ptr.term_kind()
    }
}

impl<'db> From<Ty<'db>> for Term<'db> {
    #[inline]
    fn from(value: Ty<'db>) -> Self {
        Term { ptr: GenericArgImpl::new_ty(value) }
    }
}

impl<'db> From<Const<'db>> for Term<'db> {
    #[inline]
    fn from(value: Const<'db>) -> Self {
        Term { ptr: GenericArgImpl::new_const(value) }
    }
}

impl<'db> Relate<DbInterner<'db>> for Term<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        match (a.kind(), b.kind()) {
            (TermKind::Ty(a_ty), TermKind::Ty(b_ty)) => Ok(relation.relate(a_ty, b_ty)?.into()),
            (TermKind::Const(a_ct), TermKind::Const(b_ct)) => {
                Ok(relation.relate(a_ct, b_ct)?.into())
            }
            (TermKind::Ty(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (TermKind::Const(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
        }
    }
}

impl<'db> rustc_type_ir::inherent::Term<DbInterner<'db>> for Term<'db> {}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum TermVid {
    Ty(TyVid),
    Const(ConstVid),
}

impl From<TyVid> for TermVid {
    fn from(value: TyVid) -> Self {
        TermVid::Ty(value)
    }
}

impl From<ConstVid> for TermVid {
    fn from(value: ConstVid) -> Self {
        TermVid::Const(value)
    }
}

impl<'db> DbInterner<'db> {
    pub(super) fn mk_args(self, args: &[GenericArg<'db>]) -> GenericArgs<'db> {
        GenericArgs::new_from_slice(args)
    }

    pub(super) fn mk_args_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<GenericArg<'db>, GenericArgs<'db>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_args(xs))
    }
}
