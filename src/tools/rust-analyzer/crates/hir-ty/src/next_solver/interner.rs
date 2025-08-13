//! Things related to the Interner in the next-trait-solver.
#![allow(unused)]

use base_db::Crate;
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef, Variances};
use hir_def::lang_item::LangItem;
use hir_def::signatures::{FieldData, FnFlags, ImplFlags, StructFlags, TraitFlags};
use hir_def::{AdtId, BlockId, TypeAliasId, VariantId};
use hir_def::{AttrDefId, Lookup};
use hir_def::{CallableDefId, EnumVariantId, ItemContainerId, StructId, UnionId};
use intern::sym::non_exhaustive;
use intern::{Interned, impl_internable, sym};
use la_arena::Idx;
use rustc_abi::{Align, ReprFlags, ReprOptions};
use rustc_hash::FxHashSet;
use rustc_index::bit_set::DenseBitSet;
use rustc_type_ir::elaborate::elaborate;
use rustc_type_ir::error::TypeError;
use rustc_type_ir::inherent::{
    AdtDef as _, GenericArgs as _, GenericsOf, IntoKind, SliceLike as _, Span as _,
};
use rustc_type_ir::lang_items::TraitSolverLangItem;
use rustc_type_ir::solve::SizedTraitKind;
use rustc_type_ir::{
    AliasTerm, AliasTermKind, AliasTy, AliasTyKind, EarlyBinder, FlagComputation, Flags,
    ImplPolarity, InferTy, ProjectionPredicate, TraitPredicate, TraitRef, Upcast,
};
use salsa::plumbing::AsId;
use smallvec::{SmallVec, smallvec};
use std::fmt;
use std::ops::ControlFlow;
use syntax::ast::SelfParamKind;
use triomphe::Arc;

use rustc_ast_ir::visit::VisitorResult;
use rustc_index::IndexVec;
use rustc_type_ir::TypeVisitableExt;
use rustc_type_ir::{
    BoundVar, CollectAndApply, DebruijnIndex, GenericArgKind, RegionKind, TermKind, UniverseIndex,
    Variance, WithCachedTypeInfo, elaborate,
    inherent::{self, Const as _, Region as _, Ty as _},
    ir_print, relate,
};

use crate::lower_nextsolver::{self, TyLoweringContext};
use crate::method_resolution::{ALL_FLOAT_FPS, ALL_INT_FPS, TyFingerprint};
use crate::next_solver::util::{ContainsTypeErrors, explicit_item_bounds, for_trait_impls};
use crate::next_solver::{
    CanonicalVarKind, FxIndexMap, InternedWrapperNoDebug, RegionAssumptions, SolverDefIds,
};
use crate::{ConstScalar, FnAbi, Interner, db::HirDatabase};

use super::generics::generics;
use super::util::sizedness_constraint_for_ty;
use super::{
    Binder, BoundExistentialPredicate, BoundExistentialPredicates, BoundTy, BoundTyKind, Clause,
    Clauses, Const, ConstKind, ErrorGuaranteed, ExprConst, ExternalConstraints,
    ExternalConstraintsData, GenericArg, GenericArgs, InternedClausesWrapper, ParamConst, ParamEnv,
    ParamTy, PlaceholderConst, PlaceholderTy, PredefinedOpaques, PredefinedOpaquesData, Predicate,
    PredicateKind, Term, Ty, TyKind, Tys, ValueConst,
    abi::Safety,
    fold::{BoundVarReplacer, BoundVarReplacerDelegate, FnMutDelegate},
    generics::Generics,
    mapping::ChalkToNextSolver,
    region::{
        BoundRegion, BoundRegionKind, EarlyParamRegion, LateParamRegion, PlaceholderRegion, Region,
    },
};
use super::{ClauseKind, SolverDefId, Valtree};

#[macro_export]
#[doc(hidden)]
macro_rules! _interned_vec_nolifetime_salsa {
    ($name:ident, $ty:ty) => {
        interned_vec_nolifetime_salsa!($name, $ty, nofold);

        impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for $name {
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

        impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for $name {
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
        #[salsa::interned(no_lifetime, constructor = new_, debug)]
        pub struct $name {
            #[returns(ref)]
            inner_: smallvec::SmallVec<[$ty; 2]>,
        }

        impl $name {
            pub fn new_from_iter<'db>(
                interner: DbInterner<'db>,
                data: impl IntoIterator<Item = $ty>,
            ) -> Self {
                $name::new_(interner.db(), data.into_iter().collect::<smallvec::SmallVec<[_; 2]>>())
            }

            pub fn inner(&self) -> &smallvec::SmallVec<[$ty; 2]> {
                // SAFETY: ¯\_(ツ)_/¯
                salsa::with_attached_database(|db| {
                    let inner = self.inner_(db);
                    unsafe { std::mem::transmute(inner) }
                })
                .unwrap()
            }
        }

        impl rustc_type_ir::inherent::SliceLike for $name {
            type Item = $ty;

            type IntoIter = <smallvec::SmallVec<[$ty; 2]> as IntoIterator>::IntoIter;

            fn iter(self) -> Self::IntoIter {
                self.inner().clone().into_iter()
            }

            fn as_slice(&self) -> &[Self::Item] {
                self.inner().as_slice()
            }
        }

        impl IntoIterator for $name {
            type Item = $ty;
            type IntoIter = <Self as rustc_type_ir::inherent::SliceLike>::IntoIter;

            fn into_iter(self) -> Self::IntoIter {
                rustc_type_ir::inherent::SliceLike::iter(self)
            }
        }

        impl Default for $name {
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
                salsa::with_attached_database(|db| {
                    let inner = self.inner_(db);
                    unsafe { std::mem::transmute(inner) }
                })
                .unwrap()
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
        salsa::with_attached_database(|db| DbInterner {
            db: unsafe {
                std::mem::transmute::<&dyn HirDatabase, &'db dyn HirDatabase>(db.as_view())
            },
            krate: None,
            block: None,
        })
        .expect("db is expected to be attached")
    }

    pub fn new_with(
        db: &'db dyn HirDatabase,
        krate: Option<Crate>,
        block: Option<BlockId>,
    ) -> DbInterner<'db> {
        DbInterner { db, krate, block }
    }

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
        salsa::with_attached_database(|db| {
            let inner = self.data_(db);
            // SAFETY: ¯\_(ツ)_/¯
            unsafe { std::mem::transmute(inner) }
        })
        .unwrap()
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
    fn def_id(self) -> <DbInterner<'db> as rustc_type_ir::Interner>::DefId {
        SolverDefId::AdtId(self.inner().id)
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
        let db = interner.db();
        let hir_def::AdtId::StructId(struct_id) = self.inner().id else {
            return None;
        };
        let id: VariantId = struct_id.into();
        let field_types = interner.db().field_types_ns(id);

        field_types.iter().last().map(|f| *f.1)
    }

    fn all_field_tys(
        self,
        interner: DbInterner<'db>,
    ) -> EarlyBinder<DbInterner<'db>, impl IntoIterator<Item = Ty<'db>>> {
        let db = interner.db();
        // FIXME: this is disabled just to match the behavior with chalk right now
        let field_tys = |id: VariantId| {
            let variant_data = id.fields(db);
            let fields = if variant_data.fields().is_empty() {
                vec![]
            } else {
                let field_types = db.field_types_ns(id);
                variant_data
                    .fields()
                    .iter()
                    .map(|(idx, _)| {
                        let ty = field_types[idx];
                        ty.skip_binder()
                    })
                    .collect()
            };
        };
        let field_tys = |id: VariantId| vec![];
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
        interner: DbInterner<'db>,
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
        salsa::with_attached_database(|db| match self.inner().id {
            AdtId::StructId(struct_id) => {
                let data = db.as_view::<dyn HirDatabase>().struct_signature(struct_id);
                f.write_str(data.name.as_str())
            }
            AdtId::UnionId(union_id) => {
                let data = db.as_view::<dyn HirDatabase>().union_signature(union_id);
                f.write_str(data.name.as_str())
            }
            AdtId::EnumId(enum_id) => {
                let data = db.as_view::<dyn HirDatabase>().enum_signature(enum_id);
                f.write_str(data.name.as_str())
            }
        })
        .unwrap_or_else(|| f.write_str(&format!("AdtDef({:?})", self.inner().id)))
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

    fn feature_bound_holds_in_crate(self, symbol: ()) -> bool {
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
        salsa::with_attached_database(|db| {
            let inner = &self.kind_(db).0;
            // SAFETY: The caller already has access to a `Ty<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
        .unwrap()
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

impl<'db> rustc_type_ir::Interner for DbInterner<'db> {
    type DefId = SolverDefId;
    type LocalDefId = SolverDefId;
    type LocalDefIds = SolverDefIds;
    type Span = Span;

    type GenericArgs = GenericArgs<'db>;
    type GenericArgsSlice = GenericArgs<'db>;
    type GenericArg = GenericArg<'db>;

    type Term = Term<'db>;

    type BoundVarKinds = BoundVarKinds;
    type BoundVarKind = BoundVarKind;

    type PredefinedOpaques = PredefinedOpaques<'db>;

    fn mk_predefined_opaques_in_body(
        self,
        data: rustc_type_ir::solve::PredefinedOpaquesData<Self>,
    ) -> Self::PredefinedOpaques {
        PredefinedOpaques::new(self, data)
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
    type BoundConst = rustc_type_ir::BoundVar;
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

    type VariancesOf = VariancesOf;

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
        dep_node: Self::DepNodeIndex,
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
        salsa::with_attached_database(|db| {
            tls_cache::with_cache(
                unsafe {
                    std::mem::transmute::<&dyn HirDatabase, &'db dyn HirDatabase>(
                        db.as_view::<dyn HirDatabase>(),
                    )
                },
                f,
            )
        })
        .unwrap()
    }

    fn canonical_param_env_cache_get_or_insert<R>(
        self,
        param_env: Self::ParamEnv,
        f: impl FnOnce() -> rustc_type_ir::CanonicalParamEnvCacheEntry<Self>,
        from_entry: impl FnOnce(&rustc_type_ir::CanonicalParamEnvCacheEntry<Self>) -> R,
    ) -> R {
        from_entry(&f())
    }

    fn evaluation_is_concurrent(&self) -> bool {
        false
    }

    fn expand_abstract_consts<T: rustc_type_ir::TypeFoldable<Self>>(self, _: T) -> T {
        unreachable!("only used by the old trait solver in rustc");
    }

    fn generics_of(self, def_id: Self::DefId) -> Self::GenericsOf {
        generics(self.db(), def_id)
    }

    fn variances_of(self, def_id: Self::DefId) -> Self::VariancesOf {
        match def_id {
            SolverDefId::FunctionId(def_id) => VariancesOf::new_from_iter(
                self,
                self.db()
                    .variances_of(hir_def::GenericDefId::FunctionId(def_id))
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .map(|v| v.to_nextsolver(self)),
            ),
            SolverDefId::AdtId(def_id) => VariancesOf::new_from_iter(
                self,
                self.db()
                    .variances_of(hir_def::GenericDefId::AdtId(def_id))
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .map(|v| v.to_nextsolver(self)),
            ),
            SolverDefId::InternedOpaqueTyId(_def_id) => {
                // FIXME(next-solver): track variances
                //
                // We compute them based on the only `Ty` level info in rustc,
                // move `variances_of_opaque` into `rustc_next_trait_solver` for reuse.
                VariancesOf::new_from_iter(
                    self,
                    (0..self.generics_of(def_id).count()).map(|_| Variance::Invariant),
                )
            }
            _ => VariancesOf::new_from_iter(self, []),
        }
    }

    fn type_of(self, def_id: Self::DefId) -> EarlyBinder<Self, Self::Ty> {
        let def_id = match def_id {
            SolverDefId::TypeAliasId(id) => {
                use hir_def::Lookup;
                match id.lookup(self.db()).container {
                    ItemContainerId::ImplId(it) => it,
                    _ => panic!("assoc ty value should be in impl"),
                };
                crate::TyDefId::TypeAliasId(id)
            }
            SolverDefId::AdtId(id) => crate::TyDefId::AdtId(id),
            // FIXME(next-solver): This uses the types of `query mir_borrowck` in rustc.
            //
            // We currently always use the type from HIR typeck which ignores regions. This
            // should be fine.
            SolverDefId::InternedOpaqueTyId(_) => {
                return self.type_of_opaque_hir_typeck(def_id);
            }
            _ => panic!("Unexpected def_id `{def_id:?}` provided for `type_of`"),
        };
        self.db().ty_ns(def_id)
    }

    fn adt_def(self, adt_def_id: Self::DefId) -> Self::AdtDef {
        let def_id = match adt_def_id {
            SolverDefId::AdtId(adt_id) => adt_id,
            _ => panic!("Invalid DefId passed to adt_def"),
        };
        AdtDef::new(def_id, self)
    }

    fn alias_ty_kind(self, alias: rustc_type_ir::AliasTy<Self>) -> AliasTyKind {
        match alias.def_id {
            SolverDefId::InternedOpaqueTyId(_) => AliasTyKind::Opaque,
            SolverDefId::TypeAliasId(_) => AliasTyKind::Projection,
            _ => unimplemented!("Unexpected alias: {:?}", alias.def_id),
        }
    }

    fn alias_term_kind(
        self,
        alias: rustc_type_ir::AliasTerm<Self>,
    ) -> rustc_type_ir::AliasTermKind {
        match alias.def_id {
            SolverDefId::InternedOpaqueTyId(_) => AliasTermKind::OpaqueTy,
            SolverDefId::TypeAliasId(_) => AliasTermKind::ProjectionTy,
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
        (TraitRef::new_from_args(self, trait_def_id, trait_args), alias_args)
    }

    fn check_args_compatible(self, def_id: Self::DefId, args: Self::GenericArgs) -> bool {
        // FIXME
        true
    }

    fn debug_assert_args_compatible(self, def_id: Self::DefId, args: Self::GenericArgs) {}

    fn debug_assert_existential_args_compatible(
        self,
        def_id: Self::DefId,
        args: Self::GenericArgs,
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
            SolverDefId::ForeignId(it) => it.lookup(self.db()).container,
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
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, rustc_type_ir::Binder<Self, rustc_type_ir::FnSig<Self>>> {
        let id = match def_id {
            SolverDefId::FunctionId(id) => CallableDefId::FunctionId(id),
            SolverDefId::Ctor(ctor) => match ctor {
                super::Ctor::Struct(struct_id) => CallableDefId::StructId(struct_id),
                super::Ctor::Enum(enum_variant_id) => CallableDefId::EnumVariantId(enum_variant_id),
            },
            def => unreachable!("{:?}", def),
        };
        self.db().callable_item_signature_ns(id)
    }

    fn coroutine_movability(self, def_id: Self::DefId) -> rustc_ast_ir::Movability {
        unimplemented!()
    }

    fn coroutine_for_closure(self, def_id: Self::DefId) -> Self::DefId {
        unimplemented!()
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
        let predicates = self.db().generic_predicates_ns(def_id.try_into().unwrap());
        let predicates: Vec<_> = predicates.iter().cloned().collect();
        EarlyBinder::bind(predicates.into_iter())
    }

    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn own_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        let predicates = self.db().generic_predicates_without_parent_ns(def_id.try_into().unwrap());
        let predicates: Vec<_> = predicates.iter().cloned().collect();
        EarlyBinder::bind(predicates.into_iter())
    }

    #[tracing::instrument(skip(self), ret)]
    fn explicit_super_predicates_of(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = (Self::Clause, Self::Span)>> {
        let predicates: Vec<(Clause<'db>, Span)> = self
            .db()
            .generic_predicates_ns(def_id.try_into().unwrap())
            .iter()
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
        let predicates: Vec<(Clause<'db>, Span)> = self
            .db()
            .generic_predicates_ns(def_id.try_into().unwrap())
            .iter()
            .cloned()
            .map(|p| (p, Span::dummy()))
            .collect();
        EarlyBinder::bind(predicates)
    }

    fn impl_super_outlives(
        self,
        impl_def_id: Self::DefId,
    ) -> EarlyBinder<Self, impl IntoIterator<Item = Self::Clause>> {
        let impl_id = match impl_def_id {
            SolverDefId::ImplId(id) => id,
            _ => unreachable!(),
        };
        let trait_ref = self.db().impl_trait_ns(impl_id).expect("expected an impl of trait");
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

    fn const_conditions(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        EarlyBinder::bind([unimplemented!()])
    }

    fn has_target_features(self, def_id: Self::DefId) -> bool {
        false
    }

    fn require_lang_item(
        self,
        lang_item: rustc_type_ir::lang_items::TraitSolverLangItem,
    ) -> Self::DefId {
        let lang_item = match lang_item {
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncFn => LangItem::AsyncFn,
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncFnKindHelper => unimplemented!(),
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncFnKindUpvars => unimplemented!(),
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncFnMut => LangItem::AsyncFnMut,
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncFnOnce => LangItem::AsyncFnOnce,
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncFnOnceOutput => {
                LangItem::AsyncFnOnceOutput
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::AsyncIterator => unimplemented!(),
            rustc_type_ir::lang_items::TraitSolverLangItem::CallOnceFuture => {
                LangItem::CallOnceFuture
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::CallRefFuture => {
                LangItem::CallRefFuture
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::Clone => LangItem::Clone,
            rustc_type_ir::lang_items::TraitSolverLangItem::Copy => LangItem::Copy,
            rustc_type_ir::lang_items::TraitSolverLangItem::Coroutine => LangItem::Coroutine,
            rustc_type_ir::lang_items::TraitSolverLangItem::CoroutineReturn => {
                LangItem::CoroutineReturn
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::CoroutineYield => {
                LangItem::CoroutineYield
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::Destruct => LangItem::Destruct,
            rustc_type_ir::lang_items::TraitSolverLangItem::DiscriminantKind => {
                LangItem::DiscriminantKind
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::Drop => LangItem::Drop,
            rustc_type_ir::lang_items::TraitSolverLangItem::DynMetadata => LangItem::DynMetadata,
            rustc_type_ir::lang_items::TraitSolverLangItem::Fn => LangItem::Fn,
            rustc_type_ir::lang_items::TraitSolverLangItem::FnMut => LangItem::FnMut,
            rustc_type_ir::lang_items::TraitSolverLangItem::FnOnce => LangItem::FnOnce,
            rustc_type_ir::lang_items::TraitSolverLangItem::FnPtrTrait => LangItem::FnPtrTrait,
            rustc_type_ir::lang_items::TraitSolverLangItem::FusedIterator => unimplemented!(),
            rustc_type_ir::lang_items::TraitSolverLangItem::Future => LangItem::Future,
            rustc_type_ir::lang_items::TraitSolverLangItem::FutureOutput => LangItem::FutureOutput,
            rustc_type_ir::lang_items::TraitSolverLangItem::Iterator => LangItem::Iterator,
            rustc_type_ir::lang_items::TraitSolverLangItem::Metadata => LangItem::Metadata,
            rustc_type_ir::lang_items::TraitSolverLangItem::Option => LangItem::Option,
            rustc_type_ir::lang_items::TraitSolverLangItem::PointeeTrait => LangItem::PointeeTrait,
            rustc_type_ir::lang_items::TraitSolverLangItem::Poll => LangItem::Poll,
            rustc_type_ir::lang_items::TraitSolverLangItem::Sized => LangItem::Sized,
            rustc_type_ir::lang_items::TraitSolverLangItem::MetaSized => LangItem::MetaSized,
            rustc_type_ir::lang_items::TraitSolverLangItem::PointeeSized => LangItem::PointeeSized,
            rustc_type_ir::lang_items::TraitSolverLangItem::TransmuteTrait => {
                LangItem::TransmuteTrait
            }
            rustc_type_ir::lang_items::TraitSolverLangItem::Tuple => LangItem::Tuple,
            rustc_type_ir::lang_items::TraitSolverLangItem::Unpin => LangItem::Unpin,
            rustc_type_ir::lang_items::TraitSolverLangItem::Unsize => LangItem::Unsize,
            rustc_type_ir::lang_items::TraitSolverLangItem::BikeshedGuaranteedNoDrop => {
                unimplemented!()
            }
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
            hir_def::lang_item::LangItemTarget::EnumVariant(enum_variant_id) => unimplemented!(),
        }
    }

    #[allow(clippy::match_like_matches_macro)]
    fn is_lang_item(
        self,
        def_id: Self::DefId,
        lang_item: rustc_type_ir::lang_items::TraitSolverLangItem,
    ) -> bool {
        use rustc_type_ir::lang_items::TraitSolverLangItem::*;

        // FIXME: derive PartialEq on TraitSolverLangItem
        self.as_lang_item(def_id).map_or(false, |l| match (l, lang_item) {
            (AsyncFn, AsyncFn) => true,
            (AsyncFnKindHelper, AsyncFnKindHelper) => true,
            (AsyncFnKindUpvars, AsyncFnKindUpvars) => true,
            (AsyncFnMut, AsyncFnMut) => true,
            (AsyncFnOnce, AsyncFnOnce) => true,
            (AsyncFnOnceOutput, AsyncFnOnceOutput) => true,
            (AsyncIterator, AsyncIterator) => true,
            (CallOnceFuture, CallOnceFuture) => true,
            (CallRefFuture, CallRefFuture) => true,
            (Clone, Clone) => true,
            (Copy, Copy) => true,
            (Coroutine, Coroutine) => true,
            (CoroutineReturn, CoroutineReturn) => true,
            (CoroutineYield, CoroutineYield) => true,
            (Destruct, Destruct) => true,
            (DiscriminantKind, DiscriminantKind) => true,
            (Drop, Drop) => true,
            (DynMetadata, DynMetadata) => true,
            (Fn, Fn) => true,
            (FnMut, FnMut) => true,
            (FnOnce, FnOnce) => true,
            (FnPtrTrait, FnPtrTrait) => true,
            (FusedIterator, FusedIterator) => true,
            (Future, Future) => true,
            (FutureOutput, FutureOutput) => true,
            (Iterator, Iterator) => true,
            (Metadata, Metadata) => true,
            (Option, Option) => true,
            (PointeeTrait, PointeeTrait) => true,
            (Poll, Poll) => true,
            (Sized, Sized) => true,
            (TransmuteTrait, TransmuteTrait) => true,
            (Tuple, Tuple) => true,
            (Unpin, Unpin) => true,
            (Unsize, Unsize) => true,
            _ => false,
        })
    }

    fn as_lang_item(
        self,
        def_id: Self::DefId,
    ) -> Option<rustc_type_ir::lang_items::TraitSolverLangItem> {
        use rustc_type_ir::lang_items::TraitSolverLangItem;

        let def_id: AttrDefId = match def_id {
            SolverDefId::TraitId(id) => id.into(),
            SolverDefId::TypeAliasId(id) => id.into(),
            _ => panic!("Unexpected SolverDefId in as_lang_item"),
        };
        let lang_item = self.db().lang_attr(def_id)?;
        Some(match lang_item {
            LangItem::Sized => TraitSolverLangItem::Sized,
            LangItem::MetaSized => TraitSolverLangItem::MetaSized,
            LangItem::PointeeSized => TraitSolverLangItem::PointeeSized,
            LangItem::Unsize => TraitSolverLangItem::Unsize,
            LangItem::StructuralPeq => return None,
            LangItem::StructuralTeq => return None,
            LangItem::Copy => TraitSolverLangItem::Copy,
            LangItem::Clone => TraitSolverLangItem::Clone,
            LangItem::Sync => return None,
            LangItem::DiscriminantKind => TraitSolverLangItem::DiscriminantKind,
            LangItem::Discriminant => return None,
            LangItem::PointeeTrait => TraitSolverLangItem::PointeeTrait,
            LangItem::Metadata => TraitSolverLangItem::Metadata,
            LangItem::DynMetadata => TraitSolverLangItem::DynMetadata,
            LangItem::Freeze => return None,
            LangItem::FnPtrTrait => TraitSolverLangItem::FnPtrTrait,
            LangItem::FnPtrAddr => return None,
            LangItem::Drop => TraitSolverLangItem::Drop,
            LangItem::Destruct => TraitSolverLangItem::Destruct,
            LangItem::CoerceUnsized => return None,
            LangItem::DispatchFromDyn => return None,
            LangItem::TransmuteOpts => return None,
            LangItem::TransmuteTrait => TraitSolverLangItem::TransmuteTrait,
            LangItem::Add => return None,
            LangItem::Sub => return None,
            LangItem::Mul => return None,
            LangItem::Div => return None,
            LangItem::Rem => return None,
            LangItem::Neg => return None,
            LangItem::Not => return None,
            LangItem::BitXor => return None,
            LangItem::BitAnd => return None,
            LangItem::BitOr => return None,
            LangItem::Shl => return None,
            LangItem::Shr => return None,
            LangItem::AddAssign => return None,
            LangItem::SubAssign => return None,
            LangItem::MulAssign => return None,
            LangItem::DivAssign => return None,
            LangItem::RemAssign => return None,
            LangItem::BitXorAssign => return None,
            LangItem::BitAndAssign => return None,
            LangItem::BitOrAssign => return None,
            LangItem::ShlAssign => return None,
            LangItem::ShrAssign => return None,
            LangItem::Index => return None,
            LangItem::IndexMut => return None,
            LangItem::UnsafeCell => return None,
            LangItem::VaList => return None,
            LangItem::Deref => return None,
            LangItem::DerefMut => return None,
            LangItem::DerefTarget => return None,
            LangItem::Receiver => return None,
            LangItem::Fn => TraitSolverLangItem::Fn,
            LangItem::FnMut => TraitSolverLangItem::FnMut,
            LangItem::FnOnce => TraitSolverLangItem::FnOnce,
            LangItem::FnOnceOutput => return None,
            LangItem::Future => TraitSolverLangItem::Future,
            LangItem::CoroutineState => return None,
            LangItem::Coroutine => TraitSolverLangItem::Coroutine,
            LangItem::CoroutineReturn => TraitSolverLangItem::CoroutineReturn,
            LangItem::CoroutineYield => TraitSolverLangItem::CoroutineYield,
            LangItem::Unpin => TraitSolverLangItem::Unpin,
            LangItem::Pin => return None,
            LangItem::PartialEq => return None,
            LangItem::PartialOrd => return None,
            LangItem::CVoid => return None,
            LangItem::Panic => return None,
            LangItem::PanicNounwind => return None,
            LangItem::PanicFmt => return None,
            LangItem::PanicDisplay => return None,
            LangItem::ConstPanicFmt => return None,
            LangItem::PanicBoundsCheck => return None,
            LangItem::PanicMisalignedPointerDereference => return None,
            LangItem::PanicInfo => return None,
            LangItem::PanicLocation => return None,
            LangItem::PanicImpl => return None,
            LangItem::PanicCannotUnwind => return None,
            LangItem::BeginPanic => return None,
            LangItem::FormatAlignment => return None,
            LangItem::FormatArgument => return None,
            LangItem::FormatArguments => return None,
            LangItem::FormatCount => return None,
            LangItem::FormatPlaceholder => return None,
            LangItem::FormatUnsafeArg => return None,
            LangItem::ExchangeMalloc => return None,
            LangItem::BoxFree => return None,
            LangItem::DropInPlace => return None,
            LangItem::AllocLayout => return None,
            LangItem::Start => return None,
            LangItem::EhPersonality => return None,
            LangItem::EhCatchTypeinfo => return None,
            LangItem::OwnedBox => return None,
            LangItem::PhantomData => return None,
            LangItem::ManuallyDrop => return None,
            LangItem::MaybeUninit => return None,
            LangItem::AlignOffset => return None,
            LangItem::Termination => return None,
            LangItem::Try => return None,
            LangItem::Tuple => TraitSolverLangItem::Tuple,
            LangItem::SliceLen => return None,
            LangItem::TryTraitFromResidual => return None,
            LangItem::TryTraitFromOutput => return None,
            LangItem::TryTraitBranch => return None,
            LangItem::TryTraitFromYeet => return None,
            LangItem::PointerLike => return None,
            LangItem::ConstParamTy => return None,
            LangItem::Poll => TraitSolverLangItem::Poll,
            LangItem::PollReady => return None,
            LangItem::PollPending => return None,
            LangItem::ResumeTy => return None,
            LangItem::GetContext => return None,
            LangItem::Context => return None,
            LangItem::FuturePoll => return None,
            LangItem::FutureOutput => TraitSolverLangItem::FutureOutput,
            LangItem::Option => TraitSolverLangItem::Option,
            LangItem::OptionSome => return None,
            LangItem::OptionNone => return None,
            LangItem::ResultOk => return None,
            LangItem::ResultErr => return None,
            LangItem::ControlFlowContinue => return None,
            LangItem::ControlFlowBreak => return None,
            LangItem::IntoFutureIntoFuture => return None,
            LangItem::IntoIterIntoIter => return None,
            LangItem::IteratorNext => return None,
            LangItem::Iterator => TraitSolverLangItem::Iterator,
            LangItem::PinNewUnchecked => return None,
            LangItem::RangeFrom => return None,
            LangItem::RangeFull => return None,
            LangItem::RangeInclusiveStruct => return None,
            LangItem::RangeInclusiveNew => return None,
            LangItem::Range => return None,
            LangItem::RangeToInclusive => return None,
            LangItem::RangeTo => return None,
            LangItem::String => return None,
            LangItem::CStr => return None,
            LangItem::AsyncFn => TraitSolverLangItem::AsyncFn,
            LangItem::AsyncFnMut => TraitSolverLangItem::AsyncFnMut,
            LangItem::AsyncFnOnce => TraitSolverLangItem::AsyncFnOnce,
            LangItem::AsyncFnOnceOutput => TraitSolverLangItem::AsyncFnOnceOutput,
            LangItem::CallRefFuture => TraitSolverLangItem::CallRefFuture,
            LangItem::CallOnceFuture => TraitSolverLangItem::CallOnceFuture,
            LangItem::Ordering => return None,
            LangItem::PanicNullPointerDereference => return None,
            LangItem::ReceiverTarget => return None,
            LangItem::UnsafePinned => return None,
            LangItem::AsyncFnOnceOutput => TraitSolverLangItem::AsyncFnOnceOutput,
        })
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
        trait_def_id: Self::DefId,
        self_ty: Self::Ty,
        mut f: impl FnMut(Self::DefId),
    ) {
        let trait_ = match trait_def_id {
            SolverDefId::TraitId(id) => id,
            _ => panic!("for_each_relevant_impl called for non-trait"),
        };

        let self_ty_fp = TyFingerprint::for_trait_impl_ns(&self_ty);
        let fps: &[TyFingerprint] = match self_ty.kind() {
            TyKind::Infer(InferTy::IntVar(..)) => &ALL_INT_FPS,
            TyKind::Infer(InferTy::FloatVar(..)) => &ALL_FLOAT_FPS,
            _ => self_ty_fp.as_slice(),
        };

        if fps.is_empty() {
            for_trait_impls(
                self.db(),
                self.krate.expect("Must have self.krate"),
                self.block,
                trait_,
                self_ty_fp,
                |impls| {
                    for i in impls.for_trait(trait_) {
                        use rustc_type_ir::TypeVisitable;
                        let contains_errors = self.db().impl_trait_ns(i).map_or(false, |b| {
                            b.skip_binder().visit_with(&mut ContainsTypeErrors).is_break()
                        });
                        if contains_errors {
                            continue;
                        }

                        f(SolverDefId::ImplId(i));
                    }
                    ControlFlow::Continue(())
                },
            );
        } else {
            for_trait_impls(
                self.db(),
                self.krate.expect("Must have self.krate"),
                self.block,
                trait_,
                self_ty_fp,
                |impls| {
                    for fp in fps {
                        for i in impls.for_trait_and_self_ty(trait_, *fp) {
                            use rustc_type_ir::TypeVisitable;
                            let contains_errors = self.db().impl_trait_ns(i).map_or(false, |b| {
                                b.skip_binder().visit_with(&mut ContainsTypeErrors).is_break()
                            });
                            if contains_errors {
                                continue;
                            }

                            f(SolverDefId::ImplId(i));
                        }
                    }
                    ControlFlow::Continue(())
                },
            );
        }
    }

    fn has_item_definition(self, def_id: Self::DefId) -> bool {
        // FIXME(next-solver): should check if the associated item has a value.
        true
    }

    fn impl_is_default(self, impl_def_id: Self::DefId) -> bool {
        // FIXME
        false
    }

    #[tracing::instrument(skip(self), ret)]
    fn impl_trait_ref(
        self,
        impl_def_id: Self::DefId,
    ) -> EarlyBinder<Self, rustc_type_ir::TraitRef<Self>> {
        let impl_id = match impl_def_id {
            SolverDefId::ImplId(id) => id,
            _ => panic!("Unexpected SolverDefId in impl_trait_ref"),
        };

        let db = self.db();

        db.impl_trait_ns(impl_id)
            // ImplIds for impls where the trait ref can't be resolved should never reach trait solving
            .expect("invalid impl passed to trait solver")
    }

    fn impl_polarity(self, impl_def_id: Self::DefId) -> rustc_type_ir::ImplPolarity {
        let impl_id = match impl_def_id {
            SolverDefId::ImplId(id) => id,
            _ => unreachable!(),
        };
        let impl_data = self.db().impl_signature(impl_id);
        if impl_data.flags.contains(ImplFlags::NEGATIVE) {
            ImplPolarity::Negative
        } else {
            ImplPolarity::Positive
        }
    }

    fn trait_is_auto(self, trait_def_id: Self::DefId) -> bool {
        let trait_ = match trait_def_id {
            SolverDefId::TraitId(id) => id,
            _ => panic!("Unexpected SolverDefId in trait_is_auto"),
        };
        let trait_data = self.db().trait_signature(trait_);
        trait_data.flags.contains(TraitFlags::AUTO)
    }

    fn trait_is_alias(self, trait_def_id: Self::DefId) -> bool {
        let trait_ = match trait_def_id {
            SolverDefId::TraitId(id) => id,
            _ => panic!("Unexpected SolverDefId in trait_is_alias"),
        };
        let trait_data = self.db().trait_signature(trait_);
        trait_data.flags.contains(TraitFlags::ALIAS)
    }

    fn trait_is_dyn_compatible(self, trait_def_id: Self::DefId) -> bool {
        let trait_ = match trait_def_id {
            SolverDefId::TraitId(id) => id,
            _ => unreachable!(),
        };
        crate::dyn_compatibility::dyn_compatibility(self.db(), trait_).is_none()
    }

    fn trait_is_fundamental(self, def_id: Self::DefId) -> bool {
        let trait_ = match def_id {
            SolverDefId::TraitId(id) => id,
            _ => panic!("Unexpected SolverDefId in trait_is_fundamental"),
        };
        let trait_data = self.db().trait_signature(trait_);
        trait_data.flags.contains(TraitFlags::FUNDAMENTAL)
    }

    fn trait_may_be_implemented_via_object(self, trait_def_id: Self::DefId) -> bool {
        // FIXME(next-solver): should check the `TraitFlags` for
        // the `#[rustc_do_not_implement_via_object]` flag
        true
    }

    fn is_impl_trait_in_trait(self, def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        false
    }

    fn delay_bug(self, msg: impl ToString) -> Self::ErrorGuaranteed {
        panic!("Bug encountered in next-trait-solver.")
    }

    fn is_general_coroutine(self, coroutine_def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        true
    }

    fn coroutine_is_async(self, coroutine_def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        true
    }

    fn coroutine_is_gen(self, coroutine_def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        false
    }

    fn coroutine_is_async_gen(self, coroutine_def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        false
    }

    fn unsizing_params_for_adt(self, adt_def_id: Self::DefId) -> Self::UnsizingParams {
        let id = match adt_def_id {
            SolverDefId::AdtId(id) => id,
            _ => unreachable!(),
        };
        let def = AdtDef::new(id, self);
        let num_params = self.generics_of(adt_def_id).count();

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

        let field_types = self.db().field_types_ns(variant.id());
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
            fn replace_const(&mut self, bv: BoundVar) -> Const<'db> {
                let entry = self.map.entry(bv);
                let index = entry.index();
                let var = BoundVar::from_usize(index);
                let () = (*entry.or_insert_with(|| BoundVarKind::Const)).expect_const();
                Const::new_bound(self.interner, DebruijnIndex::ZERO, var)
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

    fn opaque_types_defined_by(self, defining_anchor: Self::LocalDefId) -> Self::LocalDefIds {
        // FIXME(next-solver)
        SolverDefIds::new_from_iter(self, [])
    }

    fn alias_has_const_conditions(self, def_id: Self::DefId) -> bool {
        // FIXME(next-solver)
        false
    }

    fn explicit_implied_const_bounds(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<
        Self,
        impl IntoIterator<Item = rustc_type_ir::Binder<Self, rustc_type_ir::TraitRef<Self>>>,
    > {
        // FIXME(next-solver)
        EarlyBinder::bind([])
    }

    fn fn_is_const(self, def_id: Self::DefId) -> bool {
        let id = match def_id {
            SolverDefId::FunctionId(id) => id,
            _ => unreachable!(),
        };
        self.db().function_signature(id).flags.contains(FnFlags::CONST)
    }

    fn impl_is_const(self, def_id: Self::DefId) -> bool {
        false
    }

    fn opt_alias_variances(
        self,
        kind: impl Into<rustc_type_ir::AliasTermKind>,
        def_id: Self::DefId,
    ) -> Option<Self::VariancesOf> {
        None
    }

    fn type_of_opaque_hir_typeck(self, def_id: Self::LocalDefId) -> EarlyBinder<Self, Self::Ty> {
        match def_id {
            SolverDefId::InternedOpaqueTyId(opaque) => {
                let impl_trait_id = self.db().lookup_intern_impl_trait_id(opaque);
                match impl_trait_id {
                    crate::ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let infer = self.db().infer(func.into());
                        EarlyBinder::bind(infer.type_of_rpit[idx].to_nextsolver(self))
                    }
                    crate::ImplTraitId::TypeAliasImplTrait(..)
                    | crate::ImplTraitId::AsyncBlockTypeImplTrait(_, _) => {
                        // FIXME(next-solver)
                        EarlyBinder::bind(Ty::new_error(self, ErrorGuaranteed))
                    }
                }
            }
            _ => panic!("Unexpected SolverDefId in type_of_opaque_hir_typeck"),
        }
    }

    fn coroutine_hidden_types(
        self,
        def_id: Self::DefId,
    ) -> EarlyBinder<Self, rustc_type_ir::Binder<Self, rustc_type_ir::CoroutineWitnessTypes<Self>>>
    {
        // FIXME(next-solver)
        unimplemented!()
    }

    fn is_default_trait(self, def_id: Self::DefId) -> bool {
        self.as_lang_item(def_id).map_or(false, |l| matches!(l, TraitSolverLangItem::Sized))
    }

    fn trait_is_coinductive(self, trait_def_id: Self::DefId) -> bool {
        let id = match trait_def_id {
            SolverDefId::TraitId(id) => id,
            _ => unreachable!(),
        };
        self.db().trait_signature(id).flags.contains(TraitFlags::COINDUCTIVE)
    }

    fn trait_is_unsafe(self, trait_def_id: Self::DefId) -> bool {
        let id = match trait_def_id {
            SolverDefId::TraitId(id) => id,
            _ => unreachable!(),
        };
        self.db().trait_signature(id).flags.contains(TraitFlags::UNSAFE)
    }

    fn impl_self_is_guaranteed_unsized(self, def_id: Self::DefId) -> bool {
        false
    }

    fn impl_specializes(self, impl_def_id: Self::DefId, victim_def_id: Self::DefId) -> bool {
        false
    }

    fn next_trait_solver_globally(self) -> bool {
        true
    }

    fn opaque_types_and_coroutines_defined_by(
        self,
        defining_anchor: Self::LocalDefId,
    ) -> Self::LocalDefIds {
        // FIXME(next-solver)
        unimplemented!()
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
                consts: &mut |c| Const::new_bound(self, DebruijnIndex::ZERO, shift_bv(c)),
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

pub(crate) use tls_cache::with_new_cache;
mod tls_cache {
    use crate::db::HirDatabase;

    use super::DbInterner;
    use rustc_type_ir::search_graph::GlobalCache;
    use std::cell::RefCell;

    scoped_tls::scoped_thread_local!(static GLOBAL_CACHE: RefCell<rustc_type_ir::search_graph::GlobalCache<DbInterner<'static>>>);

    pub(crate) fn with_new_cache<T>(f: impl FnOnce() -> T) -> T {
        GLOBAL_CACHE.set(&RefCell::new(GlobalCache::default()), f)
    }

    pub(super) fn with_cache<'db, T>(
        _db: &'db dyn HirDatabase,
        f: impl FnOnce(&mut GlobalCache<DbInterner<'db>>) -> T,
    ) -> T {
        // SAFETY: No idea
        let call = move |slot: &RefCell<_>| {
            f(unsafe {
                std::mem::transmute::<
                    &mut GlobalCache<DbInterner<'static>>,
                    &mut GlobalCache<DbInterner<'db>>,
                >(&mut *slot.borrow_mut())
            })
        };
        if GLOBAL_CACHE.is_set() {
            GLOBAL_CACHE.with(call)
        } else {
            GLOBAL_CACHE.set(&RefCell::new(GlobalCache::default()), || GLOBAL_CACHE.with(call))
        }
    }
}
