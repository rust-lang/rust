//! Things related to the Interner in the next-trait-solver.

use std::{fmt, ops::ControlFlow};

pub use tls_cache::clear_tls_solver_cache;
pub use tls_db::{attach_db, attach_db_allow_change, with_attached_db};

use base_db::Crate;
use hir_def::{
    AdtId, AttrDefId, BlockId, CallableDefId, DefWithBodyId, EnumVariantId, ItemContainerId,
    StructId, UnionId, VariantId,
    lang_item::LangItem,
    signatures::{FieldData, FnFlags, ImplFlags, StructFlags, TraitFlags},
};
use la_arena::Idx;
use rustc_abi::{ReprFlags, ReprOptions};
use rustc_hash::FxHashSet;
use rustc_index::bit_set::DenseBitSet;
use rustc_type_ir::{
    AliasTermKind, AliasTyKind, BoundVar, CollectAndApply, CoroutineWitnessTypes, DebruijnIndex,
    EarlyBinder, FlagComputation, Flags, GenericArgKind, ImplPolarity, InferTy, Interner, TraitRef,
    TypeVisitableExt, UniverseIndex, Upcast, Variance,
    elaborate::elaborate,
    error::TypeError,
    inherent::{self, GenericsOf, IntoKind, SliceLike as _, Span as _, Ty as _},
    lang_items::{SolverAdtLangItem, SolverLangItem, SolverTraitLangItem},
    solve::SizedTraitKind,
};

use crate::{
    FnAbi,
    db::{HirDatabase, InternedCoroutine, InternedCoroutineId},
    method_resolution::{ALL_FLOAT_FPS, ALL_INT_FPS, TyFingerprint},
    next_solver::{
        AdtIdWrapper, BoundConst, CallableIdWrapper, CanonicalVarKind, ClosureIdWrapper,
        CoroutineIdWrapper, Ctor, FnSig, FxIndexMap, ImplIdWrapper, OpaqueTypeKey,
        RegionAssumptions, SolverContext, SolverDefIds, TraitIdWrapper, TypeAliasIdWrapper,
        util::{ContainsTypeErrors, explicit_item_bounds, for_trait_impls},
    },
};

use super::{
    Binder, BoundExistentialPredicates, BoundTy, BoundTyKind, Clause, ClauseKind, Clauses, Const,
    ErrorGuaranteed, ExprConst, ExternalConstraints, GenericArg, GenericArgs, ParamConst, ParamEnv,
    ParamTy, PlaceholderConst, PlaceholderTy, PredefinedOpaques, Predicate, SolverDefId, Term, Ty,
    TyKind, Tys, Valtree, ValueConst,
    abi::Safety,
    fold::{BoundVarReplacer, BoundVarReplacerDelegate, FnMutDelegate},
    generics::{Generics, generics},
    region::{
        BoundRegion, BoundRegionKind, EarlyParamRegion, LateParamRegion, PlaceholderRegion, Region,
    },
    util::sizedness_constraint_for_ty,
};

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub struct InternedWrapperNoDebug<T>(pub(crate) T);

#[macro_export]
#[doc(hidden)]
macro_rules! _interned_vec_nolifetime_salsa {
    ($name:ident, $ty:ty) => {
        interned_vec_nolifetime_salsa!($name, $ty, nofold);

        impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for $name<'db> {
            fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
                self,
                folder: &mut F,
            ) -> Result<Self, F::Error> {
                use rustc_type_ir::inherent::SliceLike as _;
                let inner: smallvec::SmallVec<[_; 2]> =
                    self.iter().map(|v| v.try_fold_with(folder)).collect::<Result<_, _>>()?;
                Ok($name::new_(folder.cx().db(), inner))
            }
            fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
                self,
                folder: &mut F,
            ) -> Self {
                use rustc_type_ir::inherent::SliceLike as _;
                let inner: smallvec::SmallVec<[_; 2]> =
                    self.iter().map(|v| v.fold_with(folder)).collect();
                $name::new_(folder.cx().db(), inner)
            }
        }

        impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for $name<'db> {
            fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
                &self,
                visitor: &mut V,
            ) -> V::Result {
                use rustc_ast_ir::visit::VisitorResult;
                use rustc_type_ir::inherent::SliceLike as _;
                rustc_ast_ir::walk_visitable_list!(visitor, self.as_slice().iter());
                V::Result::output()
            }
        }
    };
    ($name:ident, $ty:ty, nofold) => {
        #[salsa::interned(constructor = new_)]
        pub struct $name {
            #[returns(ref)]
            inner_: smallvec::SmallVec<[$ty; 2]>,
        }

        impl<'db> $name<'db> {
            pub fn new_from_iter(
                interner: DbInterner<'db>,
                data: impl IntoIterator<Item = $ty>,
            ) -> Self {
                $name::new_(interner.db(), data.into_iter().collect::<smallvec::SmallVec<[_; 2]>>())
            }

            pub fn inner(&self) -> &smallvec::SmallVec<[$ty; 2]> {
                // SAFETY: ¯\_(ツ)_/¯
                $crate::with_attached_db(|db| {
                    let inner = self.inner_(db);
                    unsafe { std::mem::transmute(inner) }
                })
            }
        }

        impl<'db> std::fmt::Debug for $name<'db> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.as_slice().fmt(fmt)
            }
        }

        impl<'db> rustc_type_ir::inherent::SliceLike for $name<'db> {
            type Item = $ty;

            type IntoIter = <smallvec::SmallVec<[$ty; 2]> as IntoIterator>::IntoIter;

            fn iter(self) -> Self::IntoIter {
                self.inner().clone().into_iter()
            }

            fn as_slice(&self) -> &[Self::Item] {
                self.inner().as_slice()
            }
        }

        impl<'db> IntoIterator for $name<'db> {
            type Item = $ty;
            type IntoIter = <Self as rustc_type_ir::inherent::SliceLike>::IntoIter;

            fn into_iter(self) -> Self::IntoIter {
                rustc_type_ir::inherent::SliceLike::iter(self)
            }
        }

        impl<'db> Default for $name<'db> {
            fn default() -> Self {
                $name::new_from_iter(DbInterner::conjure(), [])
            }
        }
    };
}

pub use crate::_interned_vec_nolifetime_salsa as interned_vec_nolifetime_salsa;

#[macro_export]
#[doc(hidden)]
macro_rules! _interned_vec_db {
    ($name:ident, $ty:ident) => {
        interned_vec_db!($name, $ty, nofold);

        impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for $name<'db> {
            fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
                self,
                folder: &mut F,
            ) -> Result<Self, F::Error> {
                use rustc_type_ir::inherent::SliceLike as _;
                let inner: smallvec::SmallVec<[_; 2]> =
                    self.iter().map(|v| v.try_fold_with(folder)).collect::<Result<_, _>>()?;
                Ok($name::new_(folder.cx().db(), inner))
            }
            fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
                self,
                folder: &mut F,
            ) -> Self {
                use rustc_type_ir::inherent::SliceLike as _;
                let inner: smallvec::SmallVec<[_; 2]> =
                    self.iter().map(|v| v.fold_with(folder)).collect();
                $name::new_(folder.cx().db(), inner)
            }
        }

        impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for $name<'db> {
            fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
                &self,
                visitor: &mut V,
            ) -> V::Result {
                use rustc_ast_ir::visit::VisitorResult;
                use rustc_type_ir::inherent::SliceLike as _;
                rustc_ast_ir::walk_visitable_list!(visitor, self.as_slice().iter());
                V::Result::output()
            }
        }
    };
    ($name:ident, $ty:ident, nofold) => {
        #[salsa::interned(constructor = new_)]
        pub struct $name<'db> {
            #[returns(ref)]
            inner_: smallvec::SmallVec<[$ty<'db>; 2]>,
        }

        impl<'db> std::fmt::Debug for $name<'db> {
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.as_slice().fmt(fmt)
            }
        }

        impl<'db> $name<'db> {
            pub fn new_from_iter(
                interner: DbInterner<'db>,
                data: impl IntoIterator<Item = $ty<'db>>,
            ) -> Self {
                $name::new_(interner.db(), data.into_iter().collect::<smallvec::SmallVec<[_; 2]>>())
            }

            pub fn inner(&self) -> &smallvec::SmallVec<[$ty<'db>; 2]> {
                // SAFETY: ¯\_(ツ)_/¯
                $crate::with_attached_db(|db| {
                    let inner = self.inner_(db);
                    unsafe { std::mem::transmute(inner) }
                })
            }
        }

        impl<'db> rustc_type_ir::inherent::SliceLike for $name<'db> {
            type Item = $ty<'db>;

            type IntoIter = <smallvec::SmallVec<[$ty<'db>; 2]> as IntoIterator>::IntoIter;

            fn iter(self) -> Self::IntoIter {
                self.inner().clone().into_iter()
            }

            fn as_slice(&self) -> &[Self::Item] {
                self.inner().as_slice()
            }
        }

        impl<'db> IntoIterator for $name<'db> {
            type Item = $ty<'db>;
            type IntoIter = <Self as rustc_type_ir::inherent::SliceLike>::IntoIter;

            fn into_iter(self) -> Self::IntoIter {
                rustc_type_ir::inherent::SliceLike::iter(self)
            }
        }

        impl<'db> Default for $name<'db> {
            fn default() -> Self {
                $name::new_from_iter(DbInterner::conjure(), [])
            }
        }
    };
}

pub use crate::_interned_vec_db as interned_vec_db;

#[derive(Debug, Copy, Clone)]
pub struct DbInterner<'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) krate: Option<Crate>,
    pub(crate) block: Option<BlockId>,
}

// FIXME: very wrong, see https://github.com/rust-lang/rust/pull/144808
unsafe impl Send for DbInterner<'_> {}
unsafe impl Sync for DbInterner<'_> {}

impl<'db> DbInterner<'db> {
    // FIXME(next-solver): remove this method
    pub fn conjure() -> DbInterner<'db> {
        crate::with_attached_db(|db| DbInterner {
            db: unsafe { std::mem::transmute::<&dyn HirDatabase, &'db dyn HirDatabase>(db) },
            krate: None,
            block: None,
        })
    }

    pub fn new_with(
        db: &'db dyn HirDatabase,
        krate: Option<Crate>,
        block: Option<BlockId>,
    ) -> DbInterner<'db> {
        DbInterner { db, krate, block }
    }

    #[inline]
    pub fn db(&self) -> &'db dyn HirDatabase {
        self.db
    }
}

// This is intentionally left as `()`
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Span(());

impl<'db> inherent::Span<DbInterner<'db>> for Span {
    fn dummy() -> Self {
        Span(())
    }
}

interned_vec_nolifetime_salsa!(BoundVarKinds, BoundVarKind, nofold);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum BoundVarKind {
    Ty(BoundTyKind),
    Region(BoundRegionKind),
    Const,
}

impl BoundVarKind {
    pub fn expect_region(self) -> BoundRegionKind {
        match self {
            BoundVarKind::Region(lt) => lt,
            _ => panic!("expected a region, but found another kind"),
        }
    }

    pub fn expect_ty(self) -> BoundTyKind {
        match self {
            BoundVarKind::Ty(ty) => ty,
            _ => panic!("expected a type, but found another kind"),
        }
    }

    pub fn expect_const(self) {
        match self {
            BoundVarKind::Const => (),
            _ => panic!("expected a const, but found another kind"),
        }
    }
}

interned_vec_db!(CanonicalVars, CanonicalVarKind, nofold);

pub struct DepNodeIndex;

#[derive(Debug)]
pub struct Tracked<T: fmt::Debug + Clone>(T);

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub bound: T,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Placeholder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        if self.universe == UniverseIndex::ROOT {
            write!(f, "!{:?}", self.bound)
        } else {
            write!(f, "!{}_{:?}", self.universe.index(), self.bound)
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct AllocId;

interned_vec_nolifetime_salsa!(VariancesOf, Variance, nofold);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VariantIdx(usize);

// FIXME: could/should store actual data?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum VariantDef {
    Struct(StructId),
    Union(UnionId),
    Enum(EnumVariantId),
}

impl VariantDef {
    pub fn id(&self) -> VariantId {
        match self {
            VariantDef::Struct(struct_id) => VariantId::StructId(*struct_id),
            VariantDef::Union(union_id) => VariantId::UnionId(*union_id),
            VariantDef::Enum(enum_variant_id) => VariantId::EnumVariantId(*enum_variant_id),
        }
    }

    pub fn fields(&self, db: &dyn HirDatabase) -> Vec<(Idx<FieldData>, FieldData)> {
        let id: VariantId = match self {
            VariantDef::Struct(it) => (*it).into(),
            VariantDef::Union(it) => (*it).into(),
            VariantDef::Enum(it) => (*it).into(),
        };
        id.fields(db).fields().iter().map(|(id, data)| (id, data.clone())).collect()
    }
}

/*
/// Definition of a variant -- a struct's fields or an enum variant.
#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct VariantDef {
    /// `DefId` that identifies the variant itself.
    /// If this variant belongs to a struct or union, then this is a copy of its `DefId`.
    pub def_id: DefId,
    /// `DefId` that identifies the variant's constructor.
    /// If this variant is a struct variant, then this is `None`.
    pub ctor: Option<(CtorKind, DefId)>,
    /// Variant or struct name, maybe empty for anonymous adt (struct or union).
    pub name: Symbol,
    /// Discriminant of this variant.
    pub discr: VariantDiscr,
    /// Fields of this variant.
    pub fields: IndexVec<FieldIdx, FieldDef>,
    /// The error guarantees from parser, if any.
    tainted: Option<ErrorGuaranteed>,
    /// Flags of the variant (e.g. is field list non-exhaustive)?
    flags: VariantFlags,
}
*/

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AdtFlags {
    is_enum: bool,
    is_union: bool,
    is_struct: bool,
    is_phantom_data: bool,
    is_fundamental: bool,
    is_box: bool,
    is_manually_drop: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdtDefInner {
    pub id: AdtId,
    variants: Vec<(VariantIdx, VariantDef)>,
    flags: AdtFlags,
    repr: ReprOptions,
}

// We're gonna cheat a little bit and implement `Hash` on only the `DefId` and
// accept there might be collisions for def ids from different crates (or across
// different tests, oh my).
impl std::hash::Hash for AdtDefInner {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) {
        self.id.hash(s)
    }
}

#[salsa::interned(no_lifetime, constructor = new_)]
pub struct AdtDef {
    #[returns(ref)]
    data_: AdtDefInner,
}

impl AdtDef {
    pub fn new<'db>(def_id: AdtId, interner: DbInterner<'db>) -> Self {
        let db = interner.db();
        let (flags, variants, repr) = match def_id {
            AdtId::StructId(struct_id) => {
                let data = db.struct_signature(struct_id);

                let flags = AdtFlags {
                    is_enum: false,
                    is_union: false,
                    is_struct: true,
                    is_phantom_data: data.flags.contains(StructFlags::IS_PHANTOM_DATA),
                    is_fundamental: data.flags.contains(StructFlags::FUNDAMENTAL),
                    is_box: data.flags.contains(StructFlags::IS_BOX),
                    is_manually_drop: data.flags.contains(StructFlags::IS_MANUALLY_DROP),
                };

                let variants = vec![(VariantIdx(0), VariantDef::Struct(struct_id))];

                let mut repr = ReprOptions::default();
                repr.align = data.repr.and_then(|r| r.align);
                repr.pack = data.repr.and_then(|r| r.pack);
                repr.int = data.repr.and_then(|r| r.int);

                let mut repr_flags = ReprFlags::empty();
                if flags.is_box {
                    repr_flags.insert(ReprFlags::IS_LINEAR);
                }
                if data.repr.is_some_and(|r| r.c()) {
                    repr_flags.insert(ReprFlags::IS_C);
                }
                if data.repr.is_some_and(|r| r.simd()) {
                    repr_flags.insert(ReprFlags::IS_SIMD);
                }
                repr.flags = repr_flags;

                (flags, variants, repr)
            }
            AdtId::UnionId(union_id) => {
                let data = db.union_signature(union_id);

                let flags = AdtFlags {
                    is_enum: false,
                    is_union: true,
                    is_struct: false,
                    is_phantom_data: false,
                    is_fundamental: false,
                    is_box: false,
                    is_manually_drop: false,
                };

                let variants = vec![(VariantIdx(0), VariantDef::Union(union_id))];

                let mut repr = ReprOptions::default();
                repr.align = data.repr.and_then(|r| r.align);
                repr.pack = data.repr.and_then(|r| r.pack);
                repr.int = data.repr.and_then(|r| r.int);

                let mut repr_flags = ReprFlags::empty();
                if flags.is_box {
                    repr_flags.insert(ReprFlags::IS_LINEAR);
                }
                if data.repr.is_some_and(|r| r.c()) {
                    repr_flags.insert(ReprFlags::IS_C);
                }
                if data.repr.is_some_and(|r| r.simd()) {
                    repr_flags.insert(ReprFlags::IS_SIMD);
                }
                repr.flags = repr_flags;

                (flags, variants, repr)
            }
            AdtId::EnumId(enum_id) => {
                let flags = AdtFlags {
                    is_enum: true,
                    is_union: false,
                    is_struct: false,
                    is_phantom_data: false,
                    is_fundamental: false,
                    is_box: false,
                    is_manually_drop: false,
                };

                let variants = enum_id
                    .enum_variants(db)
                    .variants
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| (VariantIdx(idx), v))
                    .map(|(idx, v)| (idx, VariantDef::Enum(v.0)))
                    .collect();

                let data = db.enum_signature(enum_id);

                let mut repr = ReprOptions::default();
                repr.align = data.repr.and_then(|r| r.align);
                repr.pack = data.repr.and_then(|r| r.pack);
                repr.int = data.repr.and_then(|r| r.int);

                let mut repr_flags = ReprFlags::empty();
                if flags.is_box {
                    repr_flags.insert(ReprFlags::IS_LINEAR);
                }
                if data.repr.is_some_and(|r| r.c()) {
                    repr_flags.insert(ReprFlags::IS_C);
                }
                if data.repr.is_some_and(|r| r.simd()) {
                    repr_flags.insert(ReprFlags::IS_SIMD);
                }
                repr.flags = repr_flags;

                (flags, variants, repr)
            }
        };

        AdtDef::new_(db, AdtDefInner { id: def_id, variants, flags, repr })
    }

    pub fn inner(&self) -> &AdtDefInner {
        crate::with_attached_db(|db| {
            let inner = self.data_(db);
            // SAFETY: ¯\_(ツ)_/¯
            unsafe { std::mem::transmute(inner) }
        })
    }

    pub fn is_enum(&self) -> bool {
        self.inner().flags.is_enum
    }

    #[inline]
    pub fn repr(self) -> ReprOptions {
        self.inner().repr
    }

    /// Asserts this is a struct or union and returns its unique variant.
    pub fn non_enum_variant(self) -> VariantDef {
        assert!(self.inner().flags.is_struct || self.inner().flags.is_union);
        self.inner().variants[0].1.clone()
    }
}

impl<'db> inherent::AdtDef<DbInterner<'db>> for AdtDef {
    fn def_id(self) -> AdtIdWrapper {
        self.inner().id.into()
    }

    fn is_struct(self) -> bool {
        self.inner().flags.is_struct
    }

    fn is_phantom_data(self) -> bool {
        self.inner().flags.is_phantom_data
    }

    fn is_fundamental(self) -> bool {
        self.inner().flags.is_fundamental
    }

    fn struct_tail_ty(
        self,
        interner: DbInterner<'db>,
    ) -> Option<EarlyBinder<DbInterner<'db>, Ty<'db>>> {
        let hir_def::AdtId::StructId(struct_id) = self.inner().id else {
            return None;
        };
        let id: VariantId = struct_id.into();
        let field_types = interner.db().field_types(id);

        field_types.iter().last().map(|f| *f.1)
    }

    fn all_field_tys(
        self,
        interner: DbInterner<'db>,
    ) -> EarlyBinder<DbInterner<'db>, impl IntoIterator<Item = Ty<'db>>> {
        let db = interner.db();
        // FIXME: this is disabled just to match the behavior with chalk right now
        let _field_tys = |id: VariantId| {
            db.field_types(id).iter().map(|(_, ty)| ty.skip_binder()).collect::<Vec<_>>()
        };
        let field_tys = |_id: VariantId| vec![];
        let tys: Vec<_> = match self.inner().id {
            hir_def::AdtId::StructId(id) => field_tys(id.into()),
            hir_def::AdtId::UnionId(id) => field_tys(id.into()),
            hir_def::AdtId::EnumId(id) => id
                .enum_variants(db)
                .variants
                .iter()
                .flat_map(|&(variant_id, _, _)| field_tys(variant_id.into()))
                .collect(),
        };

        EarlyBinder::bind(tys)
    }

    fn sizedness_constraint(
        self,
        interner: DbInterner<'db>,
        sizedness: SizedTraitKind,
    ) -> Option<EarlyBinder<DbInterner<'db>, Ty<'db>>> {
        if self.is_struct() {
            let tail_ty = self.all_field_tys(interner).skip_binder().into_iter().last()?;

            let constraint_ty = sizedness_constraint_for_ty(interner, sizedness, tail_ty)?;

            Some(EarlyBinder::bind(constraint_ty))
        } else {
            None
        }
    }

    fn destructor(
        self,
        _interner: DbInterner<'db>,
    ) -> Option<rustc_type_ir::solve::AdtDestructorKind> {
        // FIXME(next-solver)
        None
    }

    fn is_manually_drop(self) -> bool {
        self.inner().flags.is_manually_drop
    }
}

impl fmt::Debug for AdtDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::with_attached_db(|db| match self.inner().id {
            AdtId::StructId(struct_id) => {
                let data = db.struct_signature(struct_id);
                f.write_str(data.name.as_str())
            }
            AdtId::UnionId(union_id) => {
                let data = db.union_signature(union_id);
                f.write_str(data.name.as_str())
            }
            AdtId::EnumId(enum_id) => {
                let data = db.enum_signature(enum_id);
                f.write_str(data.name.as_str())
            }
        })
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Features;

impl<'db> inherent::Features<DbInterner<'db>> for Features {
    fn generic_const_exprs(self) -> bool {
        false
    }

    fn coroutine_clone(self) -> bool {
        false
    }

    fn associated_const_equality(self) -> bool {
        false
    }

    fn feature_bound_holds_in_crate(self, _symbol: ()) -> bool {
        false
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct UnsizingParams(pub(crate) DenseBitSet<u32>);

impl std::ops::Deref for UnsizingParams {
    type Target = DenseBitSet<u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub type PatternKind<'db> = rustc_type_ir::PatternKind<DbInterner<'db>>;

#[salsa::interned(constructor = new_, debug)]
pub struct Pattern<'db> {
    #[returns(ref)]
    kind_: InternedWrapperNoDebug<PatternKind<'db>>,
}

impl<'db> std::fmt::Debug for InternedWrapperNoDebug<PatternKind<'db>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<'db> Pattern<'db> {
    pub fn new(interner: DbInterner<'db>, kind: PatternKind<'db>) -> Self {
        Pattern::new_(interner.db(), InternedWrapperNoDebug(kind))
    }

    pub fn inner(&self) -> &PatternKind<'db> {
        crate::with_attached_db(|db| {
            let inner = &self.kind_(db).0;
            // SAFETY: The caller already has access to a `Ty<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
    }
}

impl<'db> Flags for Pattern<'db> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        match self.inner() {
            PatternKind::Range { start, end } => {
                FlagComputation::for_const_kind(&start.kind()).flags
                    | FlagComputation::for_const_kind(&end.kind()).flags
            }
            PatternKind::Or(pats) => {
                let mut flags = pats.as_slice()[0].flags();
                for pat in pats.as_slice()[1..].iter() {
                    flags |= pat.flags();
                }
                flags
            }
        }
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        match self.inner() {
            PatternKind::Range { start, end } => {
                start.outer_exclusive_binder().max(end.outer_exclusive_binder())
            }
            PatternKind::Or(pats) => {
                let mut idx = pats.as_slice()[0].outer_exclusive_binder();
                for pat in pats.as_slice()[1..].iter() {
                    idx = idx.max(pat.outer_exclusive_binder());
                }
                idx
            }
        }
    }
}

impl<'db> rustc_type_ir::inherent::IntoKind for Pattern<'db> {
    type Kind = rustc_type_ir::PatternKind<DbInterner<'db>>;
    fn kind(self) -> Self::Kind {
        *self.inner()
    }
}

impl<'db> rustc_type_ir::relate::Relate<DbInterner<'db>> for Pattern<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        let tcx = relation.cx();
        match (a.kind(), b.kind()) {
            (
                PatternKind::Range { start: start_a, end: end_a },
                PatternKind::Range { start: start_b, end: end_b },
            ) => {
                let start = relation.relate(start_a, start_b)?;
                let end = relation.relate(end_a, end_b)?;
                Ok(Pattern::new(tcx, PatternKind::Range { start, end }))
            }
            (PatternKind::Or(a), PatternKind::Or(b)) => {
                if a.len() != b.len() {
                    return Err(TypeError::Mismatch);
                }
                let pats = CollectAndApply::collect_and_apply(
                    std::iter::zip(a.iter(), b.iter()).map(|(a, b)| relation.relate(a, b)),
                    |g| PatList::new_from_iter(tcx, g.iter().cloned()),
                )?;
                Ok(Pattern::new(tcx, PatternKind::Or(pats)))
            }
            (PatternKind::Range { .. } | PatternKind::Or(_), _) => Err(TypeError::Mismatch),
        }
    }
}

interned_vec_db!(PatList, Pattern);

macro_rules! as_lang_item {
    (
        $solver_enum:ident, $var:ident;

        ignore = {
            $( $ignore:ident ),* $(,)?
        }

        $( $variant:ident ),* $(,)?
    ) => {{
        // Ensure exhaustiveness.
        if let Some(it) = None::<$solver_enum> {
            match it {
                $( $solver_enum::$variant => {} )*
                $( $solver_enum::$ignore => {} )*
            }
        }
        match $var {
            $( LangItem::$variant => Some($solver_enum::$variant), )*
            _ => None
        }
    }};
}

impl<'db> Interner for DbInterner<'db> {
    type DefId = SolverDefId;
    type LocalDefId = SolverDefId;
    type LocalDefIds = SolverDefIds<'db>;
    type TraitId = TraitIdWrapper;
    type ForeignId = TypeAliasIdWrapper;
    type FunctionId = CallableIdWrapper;
    type ClosureId = ClosureIdWrapper;
    type CoroutineClosureId = CoroutineIdWrapper;
    type CoroutineId = CoroutineIdWrapper;
    type AdtId = AdtIdWrapper;
    type ImplId = ImplIdWrapper;
    type Span = Span;

    type GenericArgs = GenericArgs<'db>;
    type GenericArgsSlice = GenericArgs<'db>;
    type GenericArg = GenericArg<'db>;

    type Term = Term<'db>;

    type BoundVarKinds = BoundVarKinds<'db>;
    type BoundVarKind = BoundVarKind;

    type PredefinedOpaques = PredefinedOpaques<'db>;

    fn mk_predefined_opaques_in_body(
        self,
        data: &[(OpaqueTypeKey<'db>, Self::Ty)],
    ) -> Self::PredefinedOpaques {
        PredefinedOpaques::new_from_iter(self, data.iter().cloned())
    }

    type CanonicalVarKinds = CanonicalVars<'db>;

    fn mk_canonical_var_kinds(
        self,
        kinds: &[rustc_type_ir::CanonicalVarKind<Self>],
    ) -> Self::CanonicalVarKinds {
        CanonicalVars::new_from_iter(self, kinds.iter().cloned())
    }

    type ExternalConstraints = ExternalConstraints<'db>;

    fn mk_external_constraints(
        self,
        data: rustc_type_ir::solve::ExternalConstraintsData<Self>,
    ) -> Self::ExternalConstraints {
        ExternalConstraints::new(self, data)
    }

    type DepNodeIndex = DepNodeIndex;

    type Tracked<T: fmt::Debug + Clone> = Tracked<T>;

    type Ty = Ty<'db>;
    type Tys = Tys<'db>;
    type FnInputTys = Tys<'db>;
    type ParamTy = ParamTy;
    type BoundTy = BoundTy;
    type PlaceholderTy = PlaceholderTy;
    type Symbol = ();

    type ErrorGuaranteed = ErrorGuaranteed;
    type BoundExistentialPredicates = BoundExistentialPredicates<'db>;
    type AllocId = AllocId;
    type Pat = Pattern<'db>;
    type PatList = PatList<'db>;
    type Safety = Safety;
    type Abi = FnAbi;

    type Const = Const<'db>;
    type PlaceholderConst = PlaceholderConst;
    type ParamConst = ParamConst;
    type BoundConst = BoundConst;
    type ValueConst = ValueConst<'db>;
    type ValTree = Valtree<'db>;
    type ExprConst = ExprConst;

    type Region = Region<'db>;
    type EarlyParamRegion = EarlyParamRegion;
    type LateParamRegion = LateParamRegion;
    type BoundRegion = BoundRegion;
    type PlaceholderRegion = PlaceholderRegion;

    type RegionAssumptions = RegionAssumptions<'db>;

    type ParamEnv = ParamEnv<'db>;
    type Predicate = Predicate<'db>;
    type Clause = Clause<'db>;
    type Clauses = Clauses<'db>;

    type GenericsOf = Generics;

    type VariancesOf = VariancesOf<'db>;

    type AdtDef = AdtDef;

    type Features = Features;

    fn mk_args(self, args: &[Self::GenericArg]) -> Self::GenericArgs {
        GenericArgs::new_from_iter(self, args.iter().cloned())
    }

    fn mk_args_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self::GenericArg, Self::GenericArgs>,
    {
        CollectAndApply::collect_and_apply(args, |g| {
            GenericArgs::new_from_iter(self, g.iter().cloned())
        })
    }

    type UnsizingParams = UnsizingParams;

    fn mk_tracked<T: fmt::Debug + Clone>(
        self,
        data: T,
        _dep_node: Self::DepNodeIndex,
    ) -> Self::Tracked<T> {
        Tracked(data)
    }

    fn get_tracked<T: fmt::Debug + Clone>(self, tracked: &Self::Tracked<T>) -> T {
        tracked.0.clone()
    }

    fn with_cached_task<T>(self, task: impl FnOnce() -> T) -> (T, Self::DepNodeIndex) {
        (task(), DepNodeIndex)
    }

    fn with_global_cache<R>(
        self,
        f: impl FnOnce(&mut rustc_type_ir::search_graph::GlobalCache<Self>) -> R,
    ) -> R {
        tls_cache::with_cache(self.db, f)
    }

    fn canonical_param_env_cache_get_or_insert<R>(
        self,
        _param_env: Self::ParamEnv,
        f: impl FnOnce() -> rustc_type_ir::CanonicalParamEnvCacheEntry<Self>,
        from_entry: impl FnOnce(&rustc_type_ir::CanonicalParamEnvCacheEntry<Self>) -> R,
    ) -> R {
        from_entry(&f())
    }

    fn assert_evaluation_is_concurrent(&self) {
        panic!("evaluation shouldn't be concurrent yet")
    }

    fn expand_abstract_consts<T: rustc_type_ir::TypeFoldable<Self>>(self, _: T) -> T {
        unreachable!("only used by the old trait solver in rustc");
    }

    fn generics_of(self, def_id: Self::DefId) -> Self::GenericsOf {
        generics(self.db(), def_id)
    }

    fn variances_of(self, def_id: Self::DefId) -> Self::VariancesOf {
        let generic_def = match def_id {
            SolverDefId::Ctor(Ctor::Enum(def_id)) | SolverDefId::EnumVariantId(def_id) => {
                def_id.loc(self.db).parent.into()
            }
            SolverDefId::InternedOpaqueTyId(_def_id) => {
                // FIXME(next-solver): track variances
                //
                // We compute them based on the only `Ty` level info in rustc,
                // move `variances_of_opaque` into `rustc_next_trait_solver` for reuse.
                return VariancesOf::new_from_iter(
                    self,
                    (0..self.generics_of(def_id).count()).map(|_| Variance::Invariant),
                );
            }
            SolverDefId::Ctor(Ctor::Struct(def_id)) => def_id.into(),
            SolverDefId::AdtId(def_id) => def_id.into(),
            SolverDefId::FunctionId(def_id) => def_id.into(),
            SolverDefId::ConstId(_)
            | SolverDefId::StaticId(_)
            | SolverDefId::TraitId(_)
            | SolverDefId::TypeAliasId(_)
            | SolverDefId::ImplId(_)
            | SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_) => {
                return VariancesOf::new_from_iter(self, []);
            }
        };
        self.db.variances_of(generic_def)
    }

    fn type_of(self, def_id: Self::DefId) -> EarlyBinder<Self, Self::Ty> {
        match def_id {
            SolverDefId::TypeAliasId(id) => {
                use hir_def::Lookup;
                match id.lookup(self.db()).container {
                    ItemContainerId::ImplId(it) => it,
                    _ => panic!("assoc ty value should be in impl"),
                };
                self.db().ty(id.into())
            }
            SolverDefId::AdtId(id) => self.db().ty(id.into()),
            // FIXME(next-solver): This uses the types of `query mir_borrowck` in rustc.
            //
            // We currently always use the type from HIR typeck which ignores regions. This
            // should be fine.
            SolverDefId::InternedOpaqueTyId(_) => self.type_of_opaque_hir_typeck(def_id),
            SolverDefId::FunctionId(id) => self.db.value_ty(id.into()).unwrap(),
            SolverDefId::Ctor(id) => {
                let id = match id {
                    Ctor::Struct(id) => id.into(),
                    Ctor::Enum(id) => id.into(),
                };
                self.db.value_ty(id).expect("`SolverDefId::Ctor` should have a function-like ctor")
            }
            _ => panic!("Unexpected def_id `{def_id:?}` provided for `type_of`"),
        }
    }

    fn adt_def(self, def_id: Self::AdtId) -> Self::AdtDef {
        AdtDef::new(def_id.0, self)
    }

    fn alias_ty_kind(self, alias: rustc_type_ir::AliasTy<Self>) -> AliasTyKind {
        match alias.def_id {
            SolverDefId::InternedOpaqueTyId(_) => AliasTyKind::Opaque,
            SolverDefId::TypeAliasId(type_alias) => match type_alias.loc(self.db).container {
                ItemContainerId::ImplId(impl_)
                    if self.db.impl_signature(impl_).target_trait.is_none() =>
                {
                    AliasTyKind::Inherent
                }
                ItemContainerId::TraitId(_) | ItemContainerId::ImplId(_) => AliasTyKind::Projection,
                _ => AliasTyKind::Free,
            },
            _ => unimplemented!("Unexpected alias: {:?}", alias.def_id),
        }
    }

    fn alias_term_kind(
        self,
        alias: rustc_type_ir::AliasTerm<Self>,
    ) -> rustc_type_ir::AliasTermKind {
        match alias.def_id {
            SolverDefId::InternedOpaqueTyId(_) => AliasTermKind::OpaqueTy,
            SolverDefId::TypeAliasId(type_alias) => match type_alias.loc(self.db).container {
                ItemContainerId::ImplId(impl_)
                    if self.db.impl_signature(impl_).target_trait.is_none() =>
                {
                    AliasTermKind::InherentTy
                }
                ItemContainerId::TraitId(_) | ItemContainerId::ImplId(_) => {
                    AliasTermKind::ProjectionTy
                }
                _ => AliasTermKind::FreeTy,
            },
            // rustc creates an `AnonConst` for consts, and evaluates them with CTFE (normalizing projections
            // via selection, similar to ours `find_matching_impl()`, and not with the trait solver), so mimic it.
            SolverDefId::ConstId(_) => AliasTermKind::UnevaluatedConst,
            _ => unimplemented!("Unexpected alias: {:?}", alias.def_id),
        }
    }

    fn trait_ref_and_own_args_for_alias(
        self,
        def_id: Self::DefId,
        args: Self::GenericArgs,
    ) -> (rustc_type_ir::TraitRef<Self>, Self::GenericArgsSlice) {
        let trait_def_id = self.parent(def_id);
        let trait_generics = self.generics_of(trait_def_id);
        let trait_args = GenericArgs::new_from_iter(
            self,
            args.as_slice()[0..trait_generics.own_params.len()].iter().cloned(),
        );
        let alias_args =
            GenericArgs::new_from_iter(self, args.iter().skip(trait_generics.own_params.len()));
        (TraitRef::new_from_args(self, trait_def_id.try_into().unwrap(), trait_args), alias_args)
    }

    fn check_args_compatible(self, _def_id: Self::DefId, _args: Self::GenericArgs) -> bool {
        // FIXME
        true
    }

    fn debug_assert_args_compatible(self, _def_id: Self::DefId, _args: Self::GenericArgs) {}

    fn debug_assert_existential_args_compatible(
        self,
        _def_id: Self::DefId,
        _args: Self::GenericArgs,
    ) {
    }

    fn mk_type_list_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self::Ty, Self::Tys>,
    {
        CollectAndApply::collect_and_apply(args, |g| Tys::new_from_iter(self, g.iter().cloned()))
    }

    fn parent(self, def_id: Self::DefId) -> Self::DefId {
        use hir_def::Lookup;

        let container = match def_id {
            SolverDefId::FunctionId(it) => it.lookup(self.db()).container,
            SolverDefId::TypeAliasId(it) => it.lookup(self.db()).container,
            SolverDefId::ConstId(it) => it.lookup(self.db()).container,
            SolverDefId::InternedClosureId(it) => {
                return self
                    .db()
                    .lookup_intern_closure(it)
                    .0
                    .as_generic_def_id(self.db())
                    .unwrap()
                    .into();
            }
            SolverDefId::InternedCoroutineId(it) => {
                return self
                    .db()
                    .lookup_intern_coroutine(it)
                    .0
                    .as_generic_def_id(self.db())
                    .unwrap()
                    .into();
            }
            SolverDefId::StaticId(_)
            | SolverDefId::AdtId(_)
            | SolverDefId::TraitId(_)
            | SolverDefId::ImplId(_)
            | SolverDefId::EnumVariantId(..)
            | SolverDefId::Ctor(..)
            | SolverDefId::InternedOpaqueTyId(..) => panic!(),
        };

        match container {
            ItemContainerId::ImplId(it) => it.into(),
            ItemContainerId::TraitId(it) => it.into(),
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => panic!(),
        }
    }

    fn recursion_limit(self) -> usize {
        50
    }

    fn features(self) -> Self::Features {
        Features
    }

    fn fn_sig(
        self,
        def_id: Self::FunctionId,
    ) -> EarlyBinder<Self, rustc_type_ir::Binder<Self, rustc_type_ir::FnSig<Self>>> {
        self.db().callable_item_signature(def_id.0)
    }

    fn coroutine_movability(self, def_id: Self::CoroutineId) -> rustc_ast_ir::Movability {
        // FIXME: Make this a query? I don't believe this can be accessed from bodies other than
        // the current infer query, except with revealed opaques - is it rare enough to not matter?
        let InternedCoroutine(owner, expr_id) = def_id.0.loc(self.db);
        let body = self.db.body(owner);
        let expr = &body[expr_id];
        match *expr {
            hir_def::hir::Expr::Closure { closure_kind, .. } => match closure_kind {
                hir_def::hir::ClosureKind::Coroutine(movability) => match movability {
                    hir_def::hir::Movability::Static => rustc_ast_ir::Movability::Static,
                    hir_def::hir::Movability::Movable => rustc_ast_ir::Movability::Movable,
                },
                hir_def::hir::ClosureKind::Async => rustc_ast_ir::Movability::Static,
                _ => panic!("unexpected expression for a coroutine: {expr:?}"),
            },
            hir_def::hir::Expr::Async { .. } => rustc_ast_ir::Movability::Static,
            _ => panic!("unexpected expression for a coroutine: {expr:?}"),
        }
    }

    fn coroutine_for_closure(self, def_id: Self::CoroutineClosureId) -> Self::CoroutineId {
        def_id
    }

    fn generics_require_sized_self(self, def_id: Self::DefId) -> bool {
        let sized_trait =
            LangItem::Sized.resolve_trait(self.db(), self.krate.expect("Must have self.krate"));
        let Some(sized_id) = sized_trait else {
            return false; /* No Sized trait, can't require it! */
        };
        let sized_def_id = sized_id.into();

        // Search for a predicate like `Self : Sized` amongst the trait bounds.
        let predicates = self.predicates_of(def_id);
        elaborate(self, predicates.iter_identity()).any(|pred| match pred.kind().skip_binder() {
            ClauseKind::Trait(ref trait_pred) => {
                trait_pred.def_id() == sized_def_id
                    && matches!(
                        trait_pred.self_ty().kind(),
                        TyKind::Param(ParamTy { index: 0, .. })
                    )
            }
            ClauseKind::RegionOutlives(_)
            | ClauseKind::TypeOutlives(_)
            | ClauseKind::Projection(_)
            | ClauseKind::ConstArgHasType(_, _)
            | ClauseKind::WellFormed(_)
            | ClauseKind::ConstEvaluatable(_)
            | ClauseKind::HostEffect(..)
            | ClauseKind::UnstableFeature(_) => false,
        })
    }

    #[tracing::instrument(skip(self), ret)]
    fn item_bounds(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        explicit_item_bounds(self, def_id).map_bound(|bounds| {
            Clauses::new_from_iter(self, elaborate(self, bounds).collect::<Vec<_>>())
        })
    }

    #[tracing::instrument(skip(self), ret)]
    fn item_self_bounds(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        explicit_item_bounds(self, def_id).map_bound(|bounds| {
            Clauses::new_from_iter(
                self,
                elaborate(self, bounds).filter_only_self().collect::<Vec<_>>(),
            )
        })
    }

    fn item_non_self_bounds(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        let all_bounds: FxHashSet<_> = self.item_bounds(def_id).skip_binder().into_iter().collect();
        let own_bounds: FxHashSet<_> =
            self.item_self_bounds(def_id).skip_binder().into_iter().collect();
        if all_bounds.len() == own_bounds.len() {
            EarlyBinder::bind(Clauses::new_from_iter(self, []))
        } else {
            EarlyBinder::bind(Clauses::new_from_iter(
                self,
                all_bounds.difference(&own_bounds).cloned(),
            ))
        }
    }

    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn predicates_of(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        let predicates = self.db().generic_predicates(def_id.try_into().unwrap());
        let predicates: Vec<_> = predicates.iter().cloned().collect();
        EarlyBinder::bind(predicates.into_iter())
    }

    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn own_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        let predicates = self.db().generic_predicates_without_parent(def_id.try_into().unwrap());
        let predicates: Vec<_> = predicates.iter().cloned().collect();
        EarlyBinder::bind(predicates.into_iter())
    }

    #[tracing::instrument(skip(self), ret)]
    fn explicit_super_predicates_of(
        self,
        def_id: Self::TraitId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>> {
        let is_self = |ty: Ty<'db>| match ty.kind() {
            rustc_type_ir::TyKind::Param(param) => param.index == 0,
            _ => false,
        };

        let predicates: Vec<(Clause<'db>, Span)> = self
            .db()
            .generic_predicates(def_id.0.into())
            .iter()
            .filter(|p| match p.kind().skip_binder() {
                // rustc has the following assertion:
                // https://github.com/rust-lang/rust/blob/52618eb338609df44978b0ca4451ab7941fd1c7a/compiler/rustc_hir_analysis/src/hir_ty_lowering/bounds.rs#L525-L608
                rustc_type_ir::ClauseKind::Trait(it) => is_self(it.self_ty()),
                rustc_type_ir::ClauseKind::TypeOutlives(it) => is_self(it.0),
                rustc_type_ir::ClauseKind::Projection(it) => is_self(it.self_ty()),
                rustc_type_ir::ClauseKind::HostEffect(it) => is_self(it.self_ty()),
                _ => false,
            })
            .cloned()
            .map(|p| (p, Span::dummy()))
            .collect();
        EarlyBinder::bind(predicates)
    }

    #[tracing::instrument(skip(self), ret)]
    fn explicit_implied_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>> {
        fn is_self_or_assoc(ty: Ty<'_>) -> bool {
            match ty.kind() {
                rustc_type_ir::TyKind::Param(param) => param.index == 0,
                rustc_type_ir::TyKind::Alias(rustc_type_ir::AliasTyKind::Projection, alias) => {
                    is_self_or_assoc(alias.self_ty())
                }
                _ => false,
            }
        }

        let predicates: Vec<(Clause<'db>, Span)> = self
            .db()
            .generic_predicates(def_id.try_into().unwrap())
            .iter()
            .filter(|p| match p.kind().skip_binder() {
                rustc_type_ir::ClauseKind::Trait(it) => is_self_or_assoc(it.self_ty()),
                rustc_type_ir::ClauseKind::TypeOutlives(it) => is_self_or_assoc(it.0),
                rustc_type_ir::ClauseKind::Projection(it) => is_self_or_assoc(it.self_ty()),
                rustc_type_ir::ClauseKind::HostEffect(it) => is_self_or_assoc(it.self_ty()),
                // FIXME: Not sure is this correct to allow other clauses but we might replace
                // `generic_predicates_ns` query here with something closer to rustc's
                // `implied_bounds_with_filter`, which is more granular lowering than this
                // "lower at once and then filter" implementation.
                _ => true,
            })
            .cloned()
            .map(|p| (p, Span::dummy()))
            .collect();
        EarlyBinder::bind(predicates)
    }

    fn impl_super_outlives(
        self,
        impl_id: Self::ImplId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        let trait_ref = self.db().impl_trait(impl_id.0).expect("expected an impl of trait");
        trait_ref.map_bound(|trait_ref| {
            let clause: Clause<'_> = trait_ref.upcast(self);
            Clauses::new_from_iter(
                self,
                rustc_type_ir::elaborate::elaborate(self, [clause]).filter(|clause| {
                    matches!(
                        clause.kind().skip_binder(),
                        ClauseKind::TypeOutlives(_) | ClauseKind::RegionOutlives(_)
                    )
                }),
            )
        })
    }

    #[expect(unreachable_code)]
    fn const_conditions(
        self,
        _def_id: Self::DefId,
    ) -> EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        EarlyBinder::bind([unimplemented!()])
    }

    fn has_target_features(self, _def_id: Self::FunctionId) -> bool {
        false
    }

    fn require_lang_item(self, lang_item: SolverLangItem) -> Self::DefId {
        let lang_item = match lang_item {
            SolverLangItem::AsyncFnKindUpvars => unimplemented!(),
            SolverLangItem::AsyncFnOnceOutput => LangItem::AsyncFnOnceOutput,
            SolverLangItem::CallOnceFuture => LangItem::CallOnceFuture,
            SolverLangItem::CallRefFuture => LangItem::CallRefFuture,
            SolverLangItem::CoroutineReturn => LangItem::CoroutineReturn,
            SolverLangItem::CoroutineYield => LangItem::CoroutineYield,
            SolverLangItem::DynMetadata => LangItem::DynMetadata,
            SolverLangItem::FutureOutput => LangItem::FutureOutput,
            SolverLangItem::Metadata => LangItem::Metadata,
        };
        let target = hir_def::lang_item::lang_item(
            self.db(),
            self.krate.expect("Must have self.krate"),
            lang_item,
        )
        .unwrap_or_else(|| panic!("Lang item {lang_item:?} required but not found."));
        match target {
            hir_def::lang_item::LangItemTarget::EnumId(enum_id) => enum_id.into(),
            hir_def::lang_item::LangItemTarget::Function(function_id) => function_id.into(),
            hir_def::lang_item::LangItemTarget::ImplDef(impl_id) => impl_id.into(),
            hir_def::lang_item::LangItemTarget::Static(static_id) => static_id.into(),
            hir_def::lang_item::LangItemTarget::Struct(struct_id) => struct_id.into(),
            hir_def::lang_item::LangItemTarget::Union(union_id) => union_id.into(),
            hir_def::lang_item::LangItemTarget::TypeAlias(type_alias_id) => type_alias_id.into(),
            hir_def::lang_item::LangItemTarget::Trait(trait_id) => trait_id.into(),
            hir_def::lang_item::LangItemTarget::EnumVariant(_) => unimplemented!(),
        }
    }

    fn require_trait_lang_item(self, lang_item: SolverTraitLangItem) -> TraitIdWrapper {
        let lang_item = match lang_item {
            SolverTraitLangItem::AsyncFn => LangItem::AsyncFn,
            SolverTraitLangItem::AsyncFnKindHelper => unimplemented!(),
            SolverTraitLangItem::AsyncFnMut => LangItem::AsyncFnMut,
            SolverTraitLangItem::AsyncFnOnce => LangItem::AsyncFnOnce,
            SolverTraitLangItem::AsyncFnOnceOutput => LangItem::AsyncFnOnceOutput,
            SolverTraitLangItem::AsyncIterator => unimplemented!(),
            SolverTraitLangItem::Clone => LangItem::Clone,
            SolverTraitLangItem::Copy => LangItem::Copy,
            SolverTraitLangItem::Coroutine => LangItem::Coroutine,
            SolverTraitLangItem::Destruct => LangItem::Destruct,
            SolverTraitLangItem::DiscriminantKind => LangItem::DiscriminantKind,
            SolverTraitLangItem::Drop => LangItem::Drop,
            SolverTraitLangItem::Fn => LangItem::Fn,
            SolverTraitLangItem::FnMut => LangItem::FnMut,
            SolverTraitLangItem::FnOnce => LangItem::FnOnce,
            SolverTraitLangItem::FnPtrTrait => LangItem::FnPtrTrait,
            SolverTraitLangItem::FusedIterator => unimplemented!(),
            SolverTraitLangItem::Future => LangItem::Future,
            SolverTraitLangItem::Iterator => LangItem::Iterator,
            SolverTraitLangItem::PointeeTrait => LangItem::PointeeTrait,
            SolverTraitLangItem::Sized => LangItem::Sized,
            SolverTraitLangItem::MetaSized => LangItem::MetaSized,
            SolverTraitLangItem::PointeeSized => LangItem::PointeeSized,
            SolverTraitLangItem::TransmuteTrait => LangItem::TransmuteTrait,
            SolverTraitLangItem::Tuple => LangItem::Tuple,
            SolverTraitLangItem::Unpin => LangItem::Unpin,
            SolverTraitLangItem::Unsize => LangItem::Unsize,
            SolverTraitLangItem::BikeshedGuaranteedNoDrop => {
                unimplemented!()
            }
        };
        lang_item
            .resolve_trait(self.db(), self.krate.expect("Must have self.krate"))
            .unwrap_or_else(|| panic!("Lang item {lang_item:?} required but not found."))
            .into()
    }

    fn require_adt_lang_item(self, lang_item: SolverAdtLangItem) -> AdtIdWrapper {
        let lang_item = match lang_item {
            SolverAdtLangItem::Option => LangItem::Option,
            SolverAdtLangItem::Poll => LangItem::Poll,
        };
        lang_item
            .resolve_adt(self.db(), self.krate.expect("Must have self.krate"))
            .unwrap_or_else(|| panic!("Lang item {lang_item:?} required but not found."))
            .into()
    }

    fn is_lang_item(self, def_id: Self::DefId, lang_item: SolverLangItem) -> bool {
        self.as_lang_item(def_id)
            .map_or(false, |l| std::mem::discriminant(&l) == std::mem::discriminant(&lang_item))
    }

    fn is_trait_lang_item(self, def_id: Self::TraitId, lang_item: SolverTraitLangItem) -> bool {
        self.as_trait_lang_item(def_id)
            .map_or(false, |l| std::mem::discriminant(&l) == std::mem::discriminant(&lang_item))
    }

    fn is_adt_lang_item(self, def_id: Self::AdtId, lang_item: SolverAdtLangItem) -> bool {
        // FIXME: derive PartialEq on SolverTraitLangItem
        self.as_adt_lang_item(def_id)
            .map_or(false, |l| std::mem::discriminant(&l) == std::mem::discriminant(&lang_item))
    }

    fn as_lang_item(self, def_id: Self::DefId) -> Option<SolverLangItem> {
        let def_id: AttrDefId = match def_id {
            SolverDefId::TraitId(id) => id.into(),
            SolverDefId::TypeAliasId(id) => id.into(),
            SolverDefId::AdtId(id) => id.into(),
            _ => panic!("Unexpected SolverDefId in as_lang_item"),
        };
        let lang_item = self.db().lang_attr(def_id)?;
        as_lang_item!(
            SolverLangItem, lang_item;

            ignore = {
                AsyncFnKindUpvars,
            }

            Metadata,
            DynMetadata,
            CoroutineReturn,
            CoroutineYield,
            FutureOutput,
            CallRefFuture,
            CallOnceFuture,
            AsyncFnOnceOutput,
        )
    }

    fn as_trait_lang_item(self, def_id: Self::TraitId) -> Option<SolverTraitLangItem> {
        let def_id: AttrDefId = def_id.0.into();
        let lang_item = self.db().lang_attr(def_id)?;
        as_lang_item!(
            SolverTraitLangItem, lang_item;

            ignore = {
                AsyncFnKindHelper,
                AsyncIterator,
                BikeshedGuaranteedNoDrop,
                FusedIterator,
            }

            Sized,
            MetaSized,
            PointeeSized,
            Unsize,
            Copy,
            Clone,
            DiscriminantKind,
            PointeeTrait,
            FnPtrTrait,
            Drop,
            Destruct,
            TransmuteTrait,
            Fn,
            FnMut,
            FnOnce,
            Future,
            Coroutine,
            Unpin,
            Tuple,
            Iterator,
            AsyncFn,
            AsyncFnMut,
            AsyncFnOnce,
            AsyncFnOnceOutput,
        )
    }

    fn as_adt_lang_item(self, def_id: Self::AdtId) -> Option<SolverAdtLangItem> {
        let def_id: AttrDefId = def_id.0.into();
        let lang_item = self.db().lang_attr(def_id)?;
        as_lang_item!(
            SolverAdtLangItem, lang_item;

            ignore = {}

            Option,
            Poll,
        )
    }

    fn associated_type_def_ids(self, def_id: Self::DefId) -> impl IntoIterator<Item = Self::DefId> {
        let trait_ = match def_id {
            SolverDefId::TraitId(id) => id,
            _ => unreachable!(),
        };
        trait_.trait_items(self.db()).associated_types().map(|id| id.into())
    }

    fn for_each_relevant_impl(
        self,
        trait_: Self::TraitId,
        self_ty: Self::Ty,
        mut f: impl FnMut(Self::ImplId),
    ) {
        let trait_ = trait_.0;
        let self_ty_fp = TyFingerprint::for_trait_impl(self_ty);
        let fps: &[TyFingerprint] = match self_ty.kind() {
            TyKind::Infer(InferTy::IntVar(..)) => &ALL_INT_FPS,
            TyKind::Infer(InferTy::FloatVar(..)) => &ALL_FLOAT_FPS,
            _ => self_ty_fp.as_slice(),
        };

        if fps.is_empty() {
            _ = for_trait_impls(
                self.db(),
                self.krate.expect("Must have self.krate"),
                self.block,
                trait_,
                self_ty_fp,
                |impls| {
                    for i in impls.for_trait(trait_) {
                        use rustc_type_ir::TypeVisitable;
                        let contains_errors = self.db().impl_trait(i).map_or(false, |b| {
                            b.skip_binder().visit_with(&mut ContainsTypeErrors).is_break()
                        });
                        if contains_errors {
                            continue;
                        }

                        f(i.into());
                    }
                    ControlFlow::Continue(())
                },
            );
        } else {
            _ = for_trait_impls(
                self.db(),
                self.krate.expect("Must have self.krate"),
                self.block,
                trait_,
                self_ty_fp,
                |impls| {
                    for fp in fps {
                        for i in impls.for_trait_and_self_ty(trait_, *fp) {
                            use rustc_type_ir::TypeVisitable;
                            let contains_errors = self.db().impl_trait(i).map_or(false, |b| {
                                b.skip_binder().visit_with(&mut ContainsTypeErrors).is_break()
                            });
                            if contains_errors {
                                continue;
                            }

                            f(i.into());
                        }
                    }
                    ControlFlow::Continue(())
                },
            );
        }
    }

    fn for_each_blanket_impl(self, trait_def_id: Self::TraitId, mut f: impl FnMut(Self::ImplId)) {
        let Some(krate) = self.krate else { return };

        for impls in self.db.trait_impls_in_deps(krate).iter() {
            for impl_id in impls.for_trait(trait_def_id.0) {
                let impl_data = self.db.impl_signature(impl_id);
                let self_ty_ref = &impl_data.store[impl_data.self_ty];
                if matches!(self_ty_ref, hir_def::type_ref::TypeRef::TypeParam(_)) {
                    f(impl_id.into());
                }
            }
        }
    }

    fn has_item_definition(self, _def_id: Self::DefId) -> bool {
        // FIXME(next-solver): should check if the associated item has a value.
        true
    }

    fn impl_is_default(self, impl_def_id: Self::ImplId) -> bool {
        self.db.impl_signature(impl_def_id.0).is_default()
    }

    #[tracing::instrument(skip(self), ret)]
    fn impl_trait_ref(
        self,
        impl_id: Self::ImplId,
    ) -> EarlyBinder<Self, rustc_type_ir::TraitRef<Self>> {
        let db = self.db();
        db.impl_trait(impl_id.0)
            // ImplIds for impls where the trait ref can't be resolved should never reach trait solving
            .expect("invalid impl passed to trait solver")
    }

    fn impl_polarity(self, impl_id: Self::ImplId) -> rustc_type_ir::ImplPolarity {
        let impl_data = self.db().impl_signature(impl_id.0);
        if impl_data.flags.contains(ImplFlags::NEGATIVE) {
            ImplPolarity::Negative
        } else {
            ImplPolarity::Positive
        }
    }

    fn trait_is_auto(self, trait_: Self::TraitId) -> bool {
        let trait_data = self.db().trait_signature(trait_.0);
        trait_data.flags.contains(TraitFlags::AUTO)
    }

    fn trait_is_alias(self, trait_: Self::TraitId) -> bool {
        let trait_data = self.db().trait_signature(trait_.0);
        trait_data.flags.contains(TraitFlags::ALIAS)
    }

    fn trait_is_dyn_compatible(self, trait_: Self::TraitId) -> bool {
        crate::dyn_compatibility::dyn_compatibility(self.db(), trait_.0).is_none()
    }

    fn trait_is_fundamental(self, trait_: Self::TraitId) -> bool {
        let trait_data = self.db().trait_signature(trait_.0);
        trait_data.flags.contains(TraitFlags::FUNDAMENTAL)
    }

    fn trait_may_be_implemented_via_object(self, _trait_def_id: Self::TraitId) -> bool {
        // FIXME(next-solver): should check the `TraitFlags` for
        // the `#[rustc_do_not_implement_via_object]` flag
        true
    }

    fn is_impl_trait_in_trait(self, _def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        false
    }

    fn delay_bug(self, msg: impl ToString) -> Self::ErrorGuaranteed {
        panic!("Bug encountered in next-trait-solver: {}", msg.to_string())
    }

    fn is_general_coroutine(self, def_id: Self::CoroutineId) -> bool {
        // FIXME: Make this a query? I don't believe this can be accessed from bodies other than
        // the current infer query, except with revealed opaques - is it rare enough to not matter?
        let InternedCoroutine(owner, expr_id) = def_id.0.loc(self.db);
        let body = self.db.body(owner);
        matches!(
            body[expr_id],
            hir_def::hir::Expr::Closure {
                closure_kind: hir_def::hir::ClosureKind::Coroutine(_),
                ..
            }
        )
    }

    fn coroutine_is_async(self, def_id: Self::CoroutineId) -> bool {
        // FIXME: Make this a query? I don't believe this can be accessed from bodies other than
        // the current infer query, except with revealed opaques - is it rare enough to not matter?
        let InternedCoroutine(owner, expr_id) = def_id.0.loc(self.db);
        let body = self.db.body(owner);
        matches!(
            body[expr_id],
            hir_def::hir::Expr::Closure { closure_kind: hir_def::hir::ClosureKind::Async, .. }
                | hir_def::hir::Expr::Async { .. }
        )
    }

    fn coroutine_is_gen(self, _coroutine_def_id: Self::CoroutineId) -> bool {
        // We don't handle gen coroutines yet.
        false
    }

    fn coroutine_is_async_gen(self, _coroutine_def_id: Self::CoroutineId) -> bool {
        // We don't handle gen coroutines yet.
        false
    }

    fn unsizing_params_for_adt(self, id: Self::AdtId) -> Self::UnsizingParams {
        let def = AdtDef::new(id.0, self);
        let num_params = self.generics_of(id.into()).count();

        let maybe_unsizing_param_idx = |arg: GenericArg<'db>| match arg.kind() {
            GenericArgKind::Type(ty) => match ty.kind() {
                rustc_type_ir::TyKind::Param(p) => Some(p.index),
                _ => None,
            },
            GenericArgKind::Lifetime(_) => None,
            GenericArgKind::Const(ct) => match ct.kind() {
                rustc_type_ir::ConstKind::Param(p) => Some(p.index),
                _ => None,
            },
        };

        // The last field of the structure has to exist and contain type/const parameters.
        let variant = def.non_enum_variant();
        let fields = variant.fields(self.db());
        let Some((tail_field, prefix_fields)) = fields.split_last() else {
            return UnsizingParams(DenseBitSet::new_empty(num_params));
        };

        let field_types = self.db().field_types(variant.id());
        let mut unsizing_params = DenseBitSet::new_empty(num_params);
        let ty = field_types[tail_field.0];
        for arg in ty.instantiate_identity().walk() {
            if let Some(i) = maybe_unsizing_param_idx(arg) {
                unsizing_params.insert(i);
            }
        }

        // Ensure none of the other fields mention the parameters used
        // in unsizing.
        for field in prefix_fields {
            for arg in field_types[field.0].instantiate_identity().walk() {
                if let Some(i) = maybe_unsizing_param_idx(arg) {
                    unsizing_params.remove(i);
                }
            }
        }

        UnsizingParams(unsizing_params)
    }

    fn anonymize_bound_vars<T: rustc_type_ir::TypeFoldable<Self>>(
        self,
        value: rustc_type_ir::Binder<Self, T>,
    ) -> rustc_type_ir::Binder<Self, T> {
        struct Anonymize<'a, 'db> {
            interner: DbInterner<'db>,
            map: &'a mut FxIndexMap<BoundVar, BoundVarKind>,
        }
        impl<'db> BoundVarReplacerDelegate<'db> for Anonymize<'_, 'db> {
            fn replace_region(&mut self, br: BoundRegion) -> Region<'db> {
                let entry = self.map.entry(br.var);
                let index = entry.index();
                let var = BoundVar::from_usize(index);
                let kind = (*entry.or_insert_with(|| BoundVarKind::Region(BoundRegionKind::Anon)))
                    .expect_region();
                let br = BoundRegion { var, kind };
                Region::new_bound(self.interner, DebruijnIndex::ZERO, br)
            }
            fn replace_ty(&mut self, bt: BoundTy) -> Ty<'db> {
                let entry = self.map.entry(bt.var);
                let index = entry.index();
                let var = BoundVar::from_usize(index);
                let kind =
                    (*entry.or_insert_with(|| BoundVarKind::Ty(BoundTyKind::Anon))).expect_ty();
                Ty::new_bound(self.interner, DebruijnIndex::ZERO, BoundTy { var, kind })
            }
            fn replace_const(&mut self, bv: BoundConst) -> Const<'db> {
                let entry = self.map.entry(bv.var);
                let index = entry.index();
                let var = BoundVar::from_usize(index);
                let () = (*entry.or_insert_with(|| BoundVarKind::Const)).expect_const();
                Const::new_bound(self.interner, DebruijnIndex::ZERO, BoundConst { var })
            }
        }

        let mut map = Default::default();
        let delegate = Anonymize { interner: self, map: &mut map };
        let inner = self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate);
        let bound_vars = CollectAndApply::collect_and_apply(map.into_values(), |xs| {
            BoundVarKinds::new_from_iter(self, xs.iter().cloned())
        });
        Binder::bind_with_vars(inner, bound_vars)
    }

    fn opaque_types_defined_by(self, def_id: Self::LocalDefId) -> Self::LocalDefIds {
        let Ok(def_id) = DefWithBodyId::try_from(def_id) else {
            return SolverDefIds::default();
        };
        let mut result = Vec::new();
        crate::opaques::opaque_types_defined_by(self.db, def_id, &mut result);
        SolverDefIds::new_from_iter(self, result)
    }

    fn opaque_types_and_coroutines_defined_by(self, def_id: Self::LocalDefId) -> Self::LocalDefIds {
        let Ok(def_id) = DefWithBodyId::try_from(def_id) else {
            return SolverDefIds::default();
        };
        let mut result = Vec::new();

        crate::opaques::opaque_types_defined_by(self.db, def_id, &mut result);

        // Collect coroutines.
        let body = self.db.body(def_id);
        body.exprs().for_each(|(expr_id, expr)| {
            if matches!(
                expr,
                hir_def::hir::Expr::Async { .. }
                    | hir_def::hir::Expr::Closure {
                        closure_kind: hir_def::hir::ClosureKind::Async
                            | hir_def::hir::ClosureKind::Coroutine(_),
                        ..
                    }
            ) {
                let coroutine =
                    InternedCoroutineId::new(self.db, InternedCoroutine(def_id, expr_id));
                result.push(coroutine.into());
            }
        });

        SolverDefIds::new_from_iter(self, result)
    }

    fn alias_has_const_conditions(self, _def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        false
    }

    fn explicit_implied_const_bounds(
        self,
        _def_id: Self::DefId,
    ) -> EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        // FIXME(next-solver)
        EarlyBinder::bind([])
    }

    fn fn_is_const(self, id: Self::FunctionId) -> bool {
        let id = match id.0 {
            CallableDefId::FunctionId(id) => id,
            _ => return false,
        };
        self.db().function_signature(id).flags.contains(FnFlags::CONST)
    }

    fn impl_is_const(self, _def_id: Self::ImplId) -> bool {
        false
    }

    fn opt_alias_variances(
        self,
        _kind: impl Into<rustc_type_ir::AliasTermKind>,
        _def_id: Self::DefId,
    ) -> Option<Self::VariancesOf> {
        None
    }

    fn type_of_opaque_hir_typeck(self, def_id: Self::LocalDefId) -> EarlyBinder<Self, Self::Ty> {
        match def_id {
            SolverDefId::InternedOpaqueTyId(opaque) => {
                let impl_trait_id = self.db().lookup_intern_impl_trait_id(opaque);
                match impl_trait_id {
                    crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        crate::opaques::rpit_hidden_types(self.db, func)[idx]
                    }
                    crate::ImplTraitId::TypeAliasImplTrait(type_alias, idx) => {
                        crate::opaques::tait_hidden_types(self.db, type_alias)[idx]
                    }
                }
            }
            _ => panic!("Unexpected SolverDefId in type_of_opaque_hir_typeck"),
        }
    }

    fn coroutine_hidden_types(
        self,
        _def_id: Self::CoroutineId,
    ) -> EarlyBinder<Self, Binder<'db, CoroutineWitnessTypes<Self>>> {
        // FIXME: Actually implement this.
        EarlyBinder::bind(Binder::dummy(CoroutineWitnessTypes {
            types: Tys::default(),
            assumptions: RegionAssumptions::default(),
        }))
    }

    fn is_default_trait(self, def_id: Self::TraitId) -> bool {
        self.as_trait_lang_item(def_id).map_or(false, |l| matches!(l, SolverTraitLangItem::Sized))
    }

    fn trait_is_coinductive(self, trait_: Self::TraitId) -> bool {
        self.db().trait_signature(trait_.0).flags.contains(TraitFlags::COINDUCTIVE)
    }

    fn trait_is_unsafe(self, trait_: Self::TraitId) -> bool {
        self.db().trait_signature(trait_.0).flags.contains(TraitFlags::UNSAFE)
    }

    fn impl_self_is_guaranteed_unsized(self, _def_id: Self::ImplId) -> bool {
        false
    }

    fn impl_specializes(
        self,
        specializing_impl_def_id: Self::ImplId,
        parent_impl_def_id: Self::ImplId,
    ) -> bool {
        crate::specialization::specializes(
            self.db,
            specializing_impl_def_id.0,
            parent_impl_def_id.0,
        )
    }

    fn next_trait_solver_globally(self) -> bool {
        true
    }

    type Probe = rustc_type_ir::solve::inspect::Probe<DbInterner<'db>>;
    fn mk_probe(self, probe: rustc_type_ir::solve::inspect::Probe<Self>) -> Self::Probe {
        probe
    }
    fn evaluate_root_goal_for_proof_tree_raw(
        self,
        canonical_goal: rustc_type_ir::solve::CanonicalInput<Self>,
    ) -> (rustc_type_ir::solve::QueryResult<Self>, Self::Probe) {
        rustc_next_trait_solver::solve::evaluate_root_goal_for_proof_tree_raw_provider::<
            SolverContext<'db>,
            Self,
        >(self, canonical_goal)
    }

    fn is_sizedness_trait(self, def_id: Self::TraitId) -> bool {
        matches!(
            self.as_trait_lang_item(def_id),
            Some(SolverTraitLangItem::Sized | SolverTraitLangItem::MetaSized)
        )
    }
}

impl<'db> DbInterner<'db> {
    pub fn shift_bound_var_indices<T>(self, bound_vars: usize, value: T) -> T
    where
        T: rustc_type_ir::TypeFoldable<Self>,
    {
        let shift_bv = |bv: BoundVar| BoundVar::from_usize(bv.as_usize() + bound_vars);
        self.replace_escaping_bound_vars_uncached(
            value,
            FnMutDelegate {
                regions: &mut |r: BoundRegion| {
                    Region::new_bound(
                        self,
                        DebruijnIndex::ZERO,
                        BoundRegion { var: shift_bv(r.var), kind: r.kind },
                    )
                },
                types: &mut |t: BoundTy| {
                    Ty::new_bound(
                        self,
                        DebruijnIndex::ZERO,
                        BoundTy { var: shift_bv(t.var), kind: t.kind },
                    )
                },
                consts: &mut |c| {
                    Const::new_bound(self, DebruijnIndex::ZERO, BoundConst { var: shift_bv(c.var) })
                },
            },
        )
    }

    pub fn replace_escaping_bound_vars_uncached<T: rustc_type_ir::TypeFoldable<DbInterner<'db>>>(
        self,
        value: T,
        delegate: impl BoundVarReplacerDelegate<'db>,
    ) -> T {
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    pub fn replace_bound_vars_uncached<T: rustc_type_ir::TypeFoldable<DbInterner<'db>>>(
        self,
        value: Binder<'db, T>,
        delegate: impl BoundVarReplacerDelegate<'db>,
    ) -> T {
        self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate)
    }

    pub fn mk_fn_sig<I>(
        self,
        inputs: I,
        output: Ty<'db>,
        c_variadic: bool,
        safety: Safety,
        abi: FnAbi,
    ) -> FnSig<'db>
    where
        I: IntoIterator<Item = Ty<'db>>,
    {
        FnSig {
            inputs_and_output: Tys::new_from_iter(
                self,
                inputs.into_iter().chain(std::iter::once(output)),
            ),
            c_variadic,
            safety,
            abi,
        }
    }
}

macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty,)+) => {
        $(
            impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for $ty {
                fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for $ty {
                #[inline]
                fn visit_with<F: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
                    &self,
                    _: &mut F)
                    -> F::Result
                {
                    <F::Result as rustc_ast_ir::visit::VisitorResult>::output()
                }
            }
        )+
    };
}

TrivialTypeTraversalImpls! {
    SolverDefId,
    TraitIdWrapper,
    TypeAliasIdWrapper,
    CallableIdWrapper,
    ClosureIdWrapper,
    CoroutineIdWrapper,
    AdtIdWrapper,
    ImplIdWrapper,
    Pattern<'db>,
    Safety,
    FnAbi,
    Span,
    ParamConst,
    ParamTy,
    BoundRegion,
    BoundVar,
    Placeholder<BoundRegion>,
    Placeholder<BoundTy>,
    Placeholder<BoundVar>,
}

mod tls_db {
    use std::{cell::Cell, ptr::NonNull};

    use crate::db::HirDatabase;

    struct Attached {
        database: Cell<Option<NonNull<dyn HirDatabase>>>,
    }

    impl Attached {
        #[inline]
        fn attach<R>(&self, db: &dyn HirDatabase, op: impl FnOnce() -> R) -> R {
            struct DbGuard<'s> {
                state: Option<&'s Attached>,
            }

            impl<'s> DbGuard<'s> {
                #[inline]
                fn new(attached: &'s Attached, db: &dyn HirDatabase) -> Self {
                    match attached.database.get() {
                        Some(current_db) => {
                            let new_db = NonNull::from(db);
                            if !std::ptr::addr_eq(current_db.as_ptr(), new_db.as_ptr()) {
                                panic!(
                                    "Cannot change attached database. This is likely a bug.\n\
                                    If this is not a bug, you can use `attach_db_allow_change()`."
                                );
                            }
                            Self { state: None }
                        }
                        None => {
                            // Otherwise, set the database.
                            attached.database.set(Some(NonNull::from(db)));
                            Self { state: Some(attached) }
                        }
                    }
                }
            }

            impl Drop for DbGuard<'_> {
                #[inline]
                fn drop(&mut self) {
                    // Reset database to null if we did anything in `DbGuard::new`.
                    if let Some(attached) = self.state {
                        attached.database.set(None);
                    }
                }
            }

            let _guard = DbGuard::new(self, db);
            op()
        }

        #[inline]
        fn attach_allow_change<R>(&self, db: &dyn HirDatabase, op: impl FnOnce() -> R) -> R {
            struct DbGuard<'s> {
                state: &'s Attached,
                prev: Option<NonNull<dyn HirDatabase>>,
            }

            impl<'s> DbGuard<'s> {
                #[inline]
                fn new(attached: &'s Attached, db: &dyn HirDatabase) -> Self {
                    let prev = attached.database.replace(Some(NonNull::from(db)));
                    Self { state: attached, prev }
                }
            }

            impl Drop for DbGuard<'_> {
                #[inline]
                fn drop(&mut self) {
                    self.state.database.set(self.prev);
                }
            }

            let _guard = DbGuard::new(self, db);
            op()
        }

        #[inline]
        fn with<R>(&self, op: impl FnOnce(&dyn HirDatabase) -> R) -> R {
            let db = self.database.get().expect("Try to use attached db, but not db is attached");

            // SAFETY: The db is attached, so it must be valid.
            op(unsafe { db.as_ref() })
        }
    }

    thread_local! {
        static GLOBAL_DB: Attached = const { Attached { database: Cell::new(None) } };
    }

    #[inline]
    pub fn attach_db<R>(db: &dyn HirDatabase, op: impl FnOnce() -> R) -> R {
        GLOBAL_DB.with(|global_db| global_db.attach(db, op))
    }

    #[inline]
    pub fn attach_db_allow_change<R>(db: &dyn HirDatabase, op: impl FnOnce() -> R) -> R {
        GLOBAL_DB.with(|global_db| global_db.attach_allow_change(db, op))
    }

    #[inline]
    pub fn with_attached_db<R>(op: impl FnOnce(&dyn HirDatabase) -> R) -> R {
        GLOBAL_DB.with(
            #[inline]
            |a| a.with(op),
        )
    }
}

mod tls_cache {
    use crate::db::HirDatabase;

    use super::DbInterner;
    use base_db::Nonce;
    use rustc_type_ir::search_graph::GlobalCache;
    use salsa::Revision;
    use std::cell::RefCell;

    struct Cache {
        cache: GlobalCache<DbInterner<'static>>,
        revision: Revision,
        db_nonce: Nonce,
    }

    thread_local! {
        static GLOBAL_CACHE: RefCell<Option<Cache>> = const { RefCell::new(None) };
    }

    pub(super) fn with_cache<'db, T>(
        db: &'db dyn HirDatabase,
        f: impl FnOnce(&mut GlobalCache<DbInterner<'db>>) -> T,
    ) -> T {
        GLOBAL_CACHE.with_borrow_mut(|handle| {
            let (db_nonce, revision) = db.nonce_and_revision();
            let handle = match handle {
                Some(handle) => {
                    if handle.revision != revision || db_nonce != handle.db_nonce {
                        *handle = Cache { cache: GlobalCache::default(), revision, db_nonce };
                    }
                    handle
                }
                None => handle.insert(Cache { cache: GlobalCache::default(), revision, db_nonce }),
            };

            // SAFETY: No idea
            f(unsafe {
                std::mem::transmute::<
                    &mut GlobalCache<DbInterner<'static>>,
                    &mut GlobalCache<DbInterner<'db>>,
                >(&mut handle.cache)
            })
        })
    }

    /// Clears the thread-local trait solver cache.
    ///
    /// Should be called before getting memory usage estimations, as the solver cache
    /// is per-revision and usually should be excluded from estimations.
    pub fn clear_tls_solver_cache() {
        GLOBAL_CACHE.with_borrow_mut(|handle| *handle = None);
    }
}
