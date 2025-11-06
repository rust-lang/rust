//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in rustc_hir_analysis/check/method/probe.rs.
use std::ops::ControlFlow;

use base_db::Crate;
use hir_def::{
    AdtId, AssocItemId, BlockId, ConstId, FunctionId, HasModule, ImplId, ItemContainerId, Lookup,
    ModuleId, TraitId, TypeAliasId,
    nameres::{DefMap, block_def_map, crate_def_map},
    signatures::{ConstFlags, EnumFlags, FnFlags, StructFlags, TraitFlags, TypeAliasFlags},
};
use hir_expand::name::Name;
use intern::sym;
use rustc_ast_ir::Mutability;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    FloatTy, IntTy, TypeVisitableExt, UintTy,
    inherent::{
        AdtDef, BoundExistentialPredicates, GenericArgs as _, IntoKind, SliceLike, Ty as _,
    },
};
use smallvec::{SmallVec, smallvec};
use stdx::never;
use triomphe::Arc;

use crate::{
    TraitEnvironment,
    autoderef::{self, AutoderefKind},
    db::HirDatabase,
    infer::{Adjust, Adjustment, OverloadedDeref, PointerCast, unify::InferenceTable},
    lang_items::is_box,
    next_solver::{
        Canonical, DbInterner, ErrorGuaranteed, GenericArgs, Goal, Predicate, Region, SolverDefId,
        TraitRef, Ty, TyKind, TypingMode,
        infer::{
            DbInternerInferExt, InferCtxt,
            select::ImplSource,
            traits::{Obligation, ObligationCause, PredicateObligation},
        },
        obligation_ctxt::ObligationCtxt,
    },
    traits::next_trait_solve_canonical_in_ctxt,
    utils::all_super_traits,
};

/// This is used as a key for indexing impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TyFingerprint {
    // These are lang item impls:
    Str,
    Slice,
    Array,
    Never,
    Ref(Mutability),
    RawPtr(Mutability),
    Bool,
    Char,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    // These can have user-defined impls:
    Adt(hir_def::AdtId),
    Dyn(TraitId),
    ForeignType(TypeAliasId),
    // These only exist for trait impls
    Unit,
    Unnameable,
    Function(u32),
}

impl TyFingerprint {
    /// Creates a TyFingerprint for looking up an inherent impl. Only certain
    /// types can have inherent impls: if we have some `struct S`, we can have
    /// an `impl S`, but not `impl &S`. Hence, this will return `None` for
    /// reference types and such.
    pub fn for_inherent_impl<'db>(ty: Ty<'db>) -> Option<TyFingerprint> {
        let fp = match ty.kind() {
            TyKind::Str => TyFingerprint::Str,
            TyKind::Never => TyFingerprint::Never,
            TyKind::Slice(..) => TyFingerprint::Slice,
            TyKind::Array(..) => TyFingerprint::Array,
            TyKind::Bool => TyFingerprint::Bool,
            TyKind::Char => TyFingerprint::Char,
            TyKind::Int(int) => TyFingerprint::Int(int),
            TyKind::Uint(int) => TyFingerprint::Uint(int),
            TyKind::Float(float) => TyFingerprint::Float(float),
            TyKind::Adt(adt_def, _) => TyFingerprint::Adt(adt_def.def_id().0),
            TyKind::RawPtr(_, mutability) => TyFingerprint::RawPtr(mutability),
            TyKind::Foreign(alias_id, ..) => TyFingerprint::ForeignType(alias_id.0),
            TyKind::Dynamic(bounds, _) => {
                bounds.principal_def_id().map(|trait_| TyFingerprint::Dyn(trait_.0))?
            }
            _ => return None,
        };
        Some(fp)
    }

    /// Creates a TyFingerprint for looking up a trait impl.
    pub fn for_trait_impl<'db>(ty: Ty<'db>) -> Option<TyFingerprint> {
        let fp = match ty.kind() {
            TyKind::Str => TyFingerprint::Str,
            TyKind::Never => TyFingerprint::Never,
            TyKind::Slice(..) => TyFingerprint::Slice,
            TyKind::Array(..) => TyFingerprint::Array,
            TyKind::Bool => TyFingerprint::Bool,
            TyKind::Char => TyFingerprint::Char,
            TyKind::Int(int) => TyFingerprint::Int(int),
            TyKind::Uint(int) => TyFingerprint::Uint(int),
            TyKind::Float(float) => TyFingerprint::Float(float),
            TyKind::Adt(adt_def, _) => TyFingerprint::Adt(adt_def.def_id().0),
            TyKind::RawPtr(_, mutability) => TyFingerprint::RawPtr(mutability),
            TyKind::Foreign(alias_id, ..) => TyFingerprint::ForeignType(alias_id.0),
            TyKind::Dynamic(bounds, _) => {
                bounds.principal_def_id().map(|trait_| TyFingerprint::Dyn(trait_.0))?
            }
            TyKind::Ref(_, _, mutability) => TyFingerprint::Ref(mutability),
            TyKind::Tuple(subst) => {
                let first_ty = subst.as_slice().first();
                match first_ty {
                    Some(ty) => return TyFingerprint::for_trait_impl(*ty),
                    None => TyFingerprint::Unit,
                }
            }
            // FIXME(next-solver): Putting `Alias` here is *probably* incorrect, AFAIK it should return `None`. But this breaks
            // flyimport, which uses an incorrect but fast method resolution algorithm. Therefore we put it here,
            // because this function is only called by flyimport, and anyway we should get rid of `TyFingerprint`
            // and switch to `rustc_type_ir`'s `SimplifiedType`.
            TyKind::Alias(..)
            | TyKind::FnDef(_, _)
            | TyKind::Closure(_, _)
            | TyKind::Coroutine(..)
            | TyKind::CoroutineClosure(..)
            | TyKind::CoroutineWitness(..) => TyFingerprint::Unnameable,
            TyKind::FnPtr(sig, _) => {
                TyFingerprint::Function(sig.skip_binder().inputs_and_output.inner().len() as u32)
            }
            TyKind::Param(_)
            | TyKind::Bound(..)
            | TyKind::Placeholder(..)
            | TyKind::Infer(_)
            | TyKind::Error(_)
            | TyKind::Pat(..)
            | TyKind::UnsafeBinder(..) => return None,
        };
        Some(fp)
    }
}

pub(crate) const ALL_INT_FPS: [TyFingerprint; 12] = [
    TyFingerprint::Int(IntTy::I8),
    TyFingerprint::Int(IntTy::I16),
    TyFingerprint::Int(IntTy::I32),
    TyFingerprint::Int(IntTy::I64),
    TyFingerprint::Int(IntTy::I128),
    TyFingerprint::Int(IntTy::Isize),
    TyFingerprint::Uint(UintTy::U8),
    TyFingerprint::Uint(UintTy::U16),
    TyFingerprint::Uint(UintTy::U32),
    TyFingerprint::Uint(UintTy::U64),
    TyFingerprint::Uint(UintTy::U128),
    TyFingerprint::Uint(UintTy::Usize),
];

pub(crate) const ALL_FLOAT_FPS: [TyFingerprint; 4] = [
    TyFingerprint::Float(FloatTy::F16),
    TyFingerprint::Float(FloatTy::F32),
    TyFingerprint::Float(FloatTy::F64),
    TyFingerprint::Float(FloatTy::F128),
];

type TraitFpMap = FxHashMap<TraitId, FxHashMap<Option<TyFingerprint>, Box<[ImplId]>>>;
type TraitFpMapCollector = FxHashMap<TraitId, FxHashMap<Option<TyFingerprint>, Vec<ImplId>>>;

/// Trait impls defined or available in some crate.
#[derive(Debug, Eq, PartialEq)]
pub struct TraitImpls {
    // If the `Option<TyFingerprint>` is `None`, the impl may apply to any self type.
    map: TraitFpMap,
}

impl TraitImpls {
    pub(crate) fn trait_impls_in_crate_query(db: &dyn HirDatabase, krate: Crate) -> Arc<Self> {
        let _p = tracing::info_span!("trait_impls_in_crate_query", ?krate).entered();
        let mut impls = FxHashMap::default();

        Self::collect_def_map(db, &mut impls, crate_def_map(db, krate));

        Arc::new(Self::finish(impls))
    }

    pub(crate) fn trait_impls_in_block_query(
        db: &dyn HirDatabase,
        block: BlockId,
    ) -> Option<Arc<Self>> {
        let _p = tracing::info_span!("trait_impls_in_block_query").entered();
        let mut impls = FxHashMap::default();

        Self::collect_def_map(db, &mut impls, block_def_map(db, block));

        if impls.is_empty() { None } else { Some(Arc::new(Self::finish(impls))) }
    }

    pub(crate) fn trait_impls_in_deps_query(
        db: &dyn HirDatabase,
        krate: Crate,
    ) -> Arc<[Arc<Self>]> {
        let _p = tracing::info_span!("trait_impls_in_deps_query", ?krate).entered();
        Arc::from_iter(
            db.transitive_deps(krate).into_iter().map(|krate| db.trait_impls_in_crate(krate)),
        )
    }

    fn finish(map: TraitFpMapCollector) -> TraitImpls {
        TraitImpls {
            map: map
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().map(|(k, v)| (k, v.into_boxed_slice())).collect()))
                .collect(),
        }
    }

    fn collect_def_map(db: &dyn HirDatabase, map: &mut TraitFpMapCollector, def_map: &DefMap) {
        for (_module_id, module_data) in def_map.modules() {
            for impl_id in module_data.scope.impls() {
                // Reservation impls should be ignored during trait resolution, so we never need
                // them during type analysis. See rust-lang/rust#64631 for details.
                //
                // FIXME: Reservation impls should be considered during coherence checks. If we are
                // (ever) to implement coherence checks, this filtering should be done by the trait
                // solver.
                if db.attrs(impl_id.into()).by_key(sym::rustc_reservation_impl).exists() {
                    continue;
                }
                let target_trait = match db.impl_trait(impl_id) {
                    Some(tr) => tr.skip_binder().def_id.0,
                    None => continue,
                };
                let self_ty = db.impl_self_ty(impl_id);
                let self_ty_fp = TyFingerprint::for_trait_impl(self_ty.instantiate_identity());
                map.entry(target_trait).or_default().entry(self_ty_fp).or_default().push(impl_id);
            }

            // To better support custom derives, collect impls in all unnamed const items.
            // const _: () = { ... };
            for konst in module_data.scope.unnamed_consts() {
                let body = db.body(konst.into());
                for (_, block_def_map) in body.blocks(db) {
                    Self::collect_def_map(db, map, block_def_map);
                }
            }
        }
    }

    /// Queries all trait impls for the given type.
    pub fn for_self_ty_without_blanket_impls(
        &self,
        fp: TyFingerprint,
    ) -> impl Iterator<Item = ImplId> + '_ {
        self.map
            .values()
            .flat_map(move |impls| impls.get(&Some(fp)).into_iter())
            .flat_map(|it| it.iter().copied())
    }

    /// Queries all impls of the given trait.
    pub fn for_trait(&self, trait_: TraitId) -> impl Iterator<Item = ImplId> + '_ {
        self.map
            .get(&trait_)
            .into_iter()
            .flat_map(|map| map.values().flat_map(|v| v.iter().copied()))
    }

    /// Queries all impls of `trait_` that may apply to `self_ty`.
    pub fn for_trait_and_self_ty(
        &self,
        trait_: TraitId,
        self_ty: TyFingerprint,
    ) -> impl Iterator<Item = ImplId> + '_ {
        self.map
            .get(&trait_)
            .into_iter()
            .flat_map(move |map| map.get(&Some(self_ty)).into_iter().chain(map.get(&None)))
            .flat_map(|v| v.iter().copied())
    }

    /// Queries whether `self_ty` has potentially applicable implementations of `trait_`.
    pub fn has_impls_for_trait_and_self_ty(&self, trait_: TraitId, self_ty: TyFingerprint) -> bool {
        self.for_trait_and_self_ty(trait_, self_ty).next().is_some()
    }

    pub fn all_impls(&self) -> impl Iterator<Item = ImplId> + '_ {
        self.map.values().flat_map(|map| map.values().flat_map(|v| v.iter().copied()))
    }
}

/// Inherent impls defined in some crate.
///
/// Inherent impls can only be defined in the crate that also defines the self type of the impl
/// (note that some primitives are considered to be defined by both libcore and liballoc).
///
/// This makes inherent impl lookup easier than trait impl lookup since we only have to consider a
/// single crate.
#[derive(Debug, Eq, PartialEq)]
pub struct InherentImpls {
    map: FxHashMap<TyFingerprint, Vec<ImplId>>,
    invalid_impls: Vec<ImplId>,
}

impl InherentImpls {
    pub(crate) fn inherent_impls_in_crate_query(db: &dyn HirDatabase, krate: Crate) -> Arc<Self> {
        let _p = tracing::info_span!("inherent_impls_in_crate_query", ?krate).entered();
        let mut impls = Self { map: FxHashMap::default(), invalid_impls: Vec::default() };

        let crate_def_map = crate_def_map(db, krate);
        impls.collect_def_map(db, crate_def_map);
        impls.shrink_to_fit();

        Arc::new(impls)
    }

    pub(crate) fn inherent_impls_in_block_query(
        db: &dyn HirDatabase,
        block: BlockId,
    ) -> Option<Arc<Self>> {
        let _p = tracing::info_span!("inherent_impls_in_block_query").entered();
        let mut impls = Self { map: FxHashMap::default(), invalid_impls: Vec::default() };

        let block_def_map = block_def_map(db, block);
        impls.collect_def_map(db, block_def_map);
        impls.shrink_to_fit();

        if impls.map.is_empty() && impls.invalid_impls.is_empty() {
            None
        } else {
            Some(Arc::new(impls))
        }
    }

    fn shrink_to_fit(&mut self) {
        self.map.values_mut().for_each(Vec::shrink_to_fit);
        self.map.shrink_to_fit();
    }

    fn collect_def_map(&mut self, db: &dyn HirDatabase, def_map: &DefMap) {
        for (_module_id, module_data) in def_map.modules() {
            for impl_id in module_data.scope.impls() {
                let data = db.impl_signature(impl_id);
                if data.target_trait.is_some() {
                    continue;
                }

                let self_ty = db.impl_self_ty(impl_id);
                let self_ty = self_ty.instantiate_identity();

                match is_inherent_impl_coherent(db, def_map, impl_id, self_ty) {
                    true => {
                        // `fp` should only be `None` in error cases (either erroneous code or incomplete name resolution)
                        if let Some(fp) = TyFingerprint::for_inherent_impl(self_ty) {
                            self.map.entry(fp).or_default().push(impl_id);
                        }
                    }
                    false => self.invalid_impls.push(impl_id),
                }
            }

            // To better support custom derives, collect impls in all unnamed const items.
            // const _: () = { ... };
            for konst in module_data.scope.unnamed_consts() {
                let body = db.body(konst.into());
                for (_, block_def_map) in body.blocks(db) {
                    self.collect_def_map(db, block_def_map);
                }
            }
        }
    }

    pub fn for_self_ty<'db>(&self, self_ty: Ty<'db>) -> &[ImplId] {
        match TyFingerprint::for_inherent_impl(self_ty) {
            Some(fp) => self.map.get(&fp).map(|vec| vec.as_ref()).unwrap_or(&[]),
            None => &[],
        }
    }

    pub fn all_impls(&self) -> impl Iterator<Item = ImplId> + '_ {
        self.map.values().flat_map(|v| v.iter().copied())
    }

    pub fn invalid_impls(&self) -> &[ImplId] {
        &self.invalid_impls
    }
}

pub(crate) fn incoherent_inherent_impl_crates(
    db: &dyn HirDatabase,
    krate: Crate,
    fp: TyFingerprint,
) -> SmallVec<[Crate; 2]> {
    let _p = tracing::info_span!("incoherent_inherent_impl_crates").entered();
    let mut res = SmallVec::new();

    // should pass crate for finger print and do reverse deps

    for krate in db.transitive_deps(krate) {
        let impls = db.inherent_impls_in_crate(krate);
        if impls.map.get(&fp).is_some_and(|v| !v.is_empty()) {
            res.push(krate);
        }
    }

    res
}

pub fn def_crates<'db>(
    db: &'db dyn HirDatabase,
    ty: Ty<'db>,
    cur_crate: Crate,
) -> Option<SmallVec<[Crate; 2]>> {
    match ty.kind() {
        TyKind::Adt(adt_def, _) => {
            let def_id = adt_def.def_id().0;
            let rustc_has_incoherent_inherent_impls = match def_id {
                hir_def::AdtId::StructId(id) => db
                    .struct_signature(id)
                    .flags
                    .contains(StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
                hir_def::AdtId::UnionId(id) => db
                    .union_signature(id)
                    .flags
                    .contains(StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
                hir_def::AdtId::EnumId(id) => db
                    .enum_signature(id)
                    .flags
                    .contains(EnumFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
            };
            Some(if rustc_has_incoherent_inherent_impls {
                db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::Adt(def_id))
            } else {
                smallvec![def_id.module(db).krate()]
            })
        }
        TyKind::Foreign(alias) => {
            let alias = alias.0;
            Some(
                if db
                    .type_alias_signature(alias)
                    .flags
                    .contains(TypeAliasFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPL)
                {
                    db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::ForeignType(alias))
                } else {
                    smallvec![alias.module(db).krate()]
                },
            )
        }
        TyKind::Dynamic(bounds, _) => {
            let trait_id = bounds.principal_def_id()?.0;
            Some(
                if db
                    .trait_signature(trait_id)
                    .flags
                    .contains(TraitFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS)
                {
                    db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::Dyn(trait_id))
                } else {
                    smallvec![trait_id.module(db).krate()]
                },
            )
        }
        // for primitives, there may be impls in various places (core and alloc
        // mostly). We just check the whole crate graph for crates with impls
        // (cached behind a query).
        TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::Str
        | TyKind::Slice(_)
        | TyKind::Array(..)
        | TyKind::RawPtr(..) => Some(db.incoherent_inherent_impl_crates(
            cur_crate,
            TyFingerprint::for_inherent_impl(ty).expect("fingerprint for primitive"),
        )),
        _ => None,
    }
}

/// Look up the method with the given name.
pub(crate) fn lookup_method<'db>(
    ty: &Canonical<'db, Ty<'db>>,
    table: &mut InferenceTable<'db>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: &Name,
) -> Option<(ReceiverAdjustments, FunctionId, bool)> {
    let mut not_visible = None;
    let res = iterate_method_candidates(
        ty,
        table,
        traits_in_scope,
        visible_from_module,
        Some(name),
        LookupMode::MethodCall,
        |adjustments, f, visible| match f {
            AssocItemId::FunctionId(f) if visible => Some((adjustments, f, true)),
            AssocItemId::FunctionId(f) if not_visible.is_none() => {
                not_visible = Some((adjustments, f, false));
                None
            }
            _ => None,
        },
    );
    res.or(not_visible)
}

/// Whether we're looking up a dotted method call (like `v.len()`) or a path
/// (like `Vec::new`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LookupMode {
    /// Looking up a method call like `v.len()`: We only consider candidates
    /// that have a `self` parameter, and do autoderef.
    MethodCall,
    /// Looking up a path like `Vec::new` or `Vec::default`: We consider all
    /// candidates including associated constants, but don't do autoderef.
    Path,
}

#[derive(Clone, Copy)]
pub enum VisibleFromModule {
    /// Filter for results that are visible from the given module
    Filter(ModuleId),
    /// Include impls from the given block.
    IncludeBlock(BlockId),
    /// Do nothing special in regards visibility
    None,
}

impl From<Option<ModuleId>> for VisibleFromModule {
    fn from(module: Option<ModuleId>) -> Self {
        match module {
            Some(module) => Self::Filter(module),
            None => Self::None,
        }
    }
}

impl From<Option<BlockId>> for VisibleFromModule {
    fn from(block: Option<BlockId>) -> Self {
        match block {
            Some(block) => Self::IncludeBlock(block),
            None => Self::None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AutorefOrPtrAdjustment {
    Autoref(Mutability),
    ToConstPtr,
}

#[derive(Debug, Clone, Default)]
pub struct ReceiverAdjustments {
    autoref: Option<AutorefOrPtrAdjustment>,
    autoderefs: usize,
    unsize_array: bool,
}

impl ReceiverAdjustments {
    pub(crate) fn apply<'db>(
        &self,
        table: &mut InferenceTable<'db>,
        mut ty: Ty<'db>,
    ) -> (Ty<'db>, Vec<Adjustment<'db>>) {
        let mut adjust = Vec::new();
        let mut autoderef = table.autoderef(ty);
        autoderef.next();
        for _ in 0..self.autoderefs {
            match autoderef.next() {
                None => {
                    never!("autoderef not possible for {:?}", ty);
                    ty = Ty::new_error(table.interner(), ErrorGuaranteed);
                    break;
                }
                Some((new_ty, _)) => {
                    ty = new_ty;
                    let mutbl = match self.autoref {
                        Some(AutorefOrPtrAdjustment::Autoref(m)) => Some(m),
                        Some(AutorefOrPtrAdjustment::ToConstPtr) => Some(Mutability::Not),
                        // FIXME should we know the mutability here, when autoref is `None`?
                        None => None,
                    };
                    adjust.push(Adjustment {
                        kind: Adjust::Deref(match autoderef.steps().last().unwrap().1 {
                            AutoderefKind::Overloaded => Some(OverloadedDeref(mutbl)),
                            AutoderefKind::Builtin => None,
                        }),
                        target: ty,
                    });
                }
            }
        }
        if let Some(autoref) = &self.autoref {
            let lt = table.next_region_var();
            match autoref {
                AutorefOrPtrAdjustment::Autoref(m) => {
                    let a = Adjustment::borrow(table.interner(), *m, ty, lt);
                    ty = a.target;
                    adjust.push(a);
                }
                AutorefOrPtrAdjustment::ToConstPtr => {
                    if let TyKind::RawPtr(pointee, Mutability::Mut) = ty.kind() {
                        let a = Adjustment {
                            kind: Adjust::Pointer(PointerCast::MutToConstPointer),
                            target: Ty::new_ptr(table.interner(), pointee, Mutability::Not),
                        };
                        ty = a.target;
                        adjust.push(a);
                    } else {
                        never!("`ToConstPtr` target is not a raw mutable pointer");
                    }
                }
            };
        }
        if self.unsize_array {
            ty = 'it: {
                if let TyKind::Ref(l, inner, m) = ty.kind()
                    && let TyKind::Array(inner, _) = inner.kind()
                {
                    break 'it Ty::new_ref(
                        table.interner(),
                        l,
                        Ty::new_slice(table.interner(), inner),
                        m,
                    );
                }
                // FIXME: report diagnostic if array unsizing happens without indirection.
                ty
            };
            adjust.push(Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), target: ty });
        }
        (ty, adjust)
    }

    fn with_autoref(&self, a: AutorefOrPtrAdjustment) -> ReceiverAdjustments {
        Self { autoref: Some(a), ..*self }
    }
}

// This would be nicer if it just returned an iterator, but that runs into
// lifetime problems, because we need to borrow temp `CrateImplDefs`.
// FIXME add a context type here?
pub(crate) fn iterate_method_candidates<'db, T>(
    ty: &Canonical<'db, Ty<'db>>,
    table: &mut InferenceTable<'db>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(ReceiverAdjustments, AssocItemId, bool) -> Option<T>,
) -> Option<T> {
    let mut slot = None;
    _ = iterate_method_candidates_dyn_impl(
        ty,
        table,
        traits_in_scope,
        visible_from_module,
        name,
        mode,
        &mut |adj, item, visible| {
            assert!(slot.is_none());
            if let Some(it) = callback(adj, item, visible) {
                slot = Some(it);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        },
    );
    slot
}

pub fn lookup_impl_const<'db>(
    infcx: &InferCtxt<'db>,
    env: Arc<TraitEnvironment<'db>>,
    const_id: ConstId,
    subs: GenericArgs<'db>,
) -> (ConstId, GenericArgs<'db>) {
    let interner = infcx.interner;
    let db = interner.db;

    let trait_id = match const_id.lookup(db).container {
        ItemContainerId::TraitId(id) => id,
        _ => return (const_id, subs),
    };
    let trait_ref = TraitRef::new(interner, trait_id.into(), subs);

    let const_signature = db.const_signature(const_id);
    let name = match const_signature.name.as_ref() {
        Some(name) => name,
        None => return (const_id, subs),
    };

    lookup_impl_assoc_item_for_trait_ref(infcx, trait_ref, env, name)
        .and_then(
            |assoc| if let (AssocItemId::ConstId(id), s) = assoc { Some((id, s)) } else { None },
        )
        .unwrap_or((const_id, subs))
}

/// Checks if the self parameter of `Trait` method is the `dyn Trait` and we should
/// call the method using the vtable.
pub fn is_dyn_method<'db>(
    interner: DbInterner<'db>,
    _env: Arc<TraitEnvironment<'db>>,
    func: FunctionId,
    fn_subst: GenericArgs<'db>,
) -> Option<usize> {
    let db = interner.db;

    let ItemContainerId::TraitId(trait_id) = func.lookup(db).container else {
        return None;
    };
    let trait_params = db.generic_params(trait_id.into()).len();
    let fn_params = fn_subst.len() - trait_params;
    let trait_ref = TraitRef::new(
        interner,
        trait_id.into(),
        GenericArgs::new_from_iter(interner, fn_subst.iter().take(trait_params)),
    );
    let self_ty = trait_ref.self_ty();
    if let TyKind::Dynamic(d, _) = self_ty.kind() {
        // rustc doesn't accept `impl Foo<2> for dyn Foo<5>`, so if the trait id is equal, no matter
        // what the generics are, we are sure that the method is come from the vtable.
        let is_my_trait_in_bounds = d
            .principal_def_id()
            .is_some_and(|trait_| all_super_traits(db, trait_.0).contains(&trait_id));
        if is_my_trait_in_bounds {
            return Some(fn_params);
        }
    }
    None
}

/// Looks up the impl method that actually runs for the trait method `func`.
///
/// Returns `func` if it's not a method defined in a trait or the lookup failed.
pub(crate) fn lookup_impl_method_query<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    func: FunctionId,
    fn_subst: GenericArgs<'db>,
) -> (FunctionId, GenericArgs<'db>) {
    let interner = DbInterner::new_with(db, Some(env.krate), env.block);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);

    let ItemContainerId::TraitId(trait_id) = func.lookup(db).container else {
        return (func, fn_subst);
    };
    let trait_params = db.generic_params(trait_id.into()).len();
    let trait_ref = TraitRef::new(
        interner,
        trait_id.into(),
        GenericArgs::new_from_iter(interner, fn_subst.iter().take(trait_params)),
    );

    let name = &db.function_signature(func).name;
    let Some((impl_fn, impl_subst)) =
        lookup_impl_assoc_item_for_trait_ref(&infcx, trait_ref, env, name).and_then(|assoc| {
            if let (AssocItemId::FunctionId(id), subst) = assoc { Some((id, subst)) } else { None }
        })
    else {
        return (func, fn_subst);
    };

    (
        impl_fn,
        GenericArgs::new_from_iter(
            interner,
            impl_subst.iter().chain(fn_subst.iter().skip(trait_params)),
        ),
    )
}

fn lookup_impl_assoc_item_for_trait_ref<'db>(
    infcx: &InferCtxt<'db>,
    trait_ref: TraitRef<'db>,
    env: Arc<TraitEnvironment<'db>>,
    name: &Name,
) -> Option<(AssocItemId, GenericArgs<'db>)> {
    let (impl_id, impl_subst) = find_matching_impl(infcx, &env, trait_ref)?;
    let item =
        impl_id.impl_items(infcx.interner.db).items.iter().find_map(|(n, it)| match *it {
            AssocItemId::FunctionId(f) => (n == name).then_some(AssocItemId::FunctionId(f)),
            AssocItemId::ConstId(c) => (n == name).then_some(AssocItemId::ConstId(c)),
            AssocItemId::TypeAliasId(_) => None,
        })?;
    Some((item, impl_subst))
}

pub(crate) fn find_matching_impl<'db>(
    infcx: &InferCtxt<'db>,
    env: &TraitEnvironment<'db>,
    trait_ref: TraitRef<'db>,
) -> Option<(ImplId, GenericArgs<'db>)> {
    let trait_ref =
        infcx.at(&ObligationCause::dummy(), env.env).deeply_normalize(trait_ref).ok()?;

    let obligation = Obligation::new(infcx.interner, ObligationCause::dummy(), env.env, trait_ref);

    let selection = infcx.select(&obligation).ok()??;

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters.
    let mut ocx = ObligationCtxt::new(infcx);
    let impl_source = selection.map(|obligation| ocx.register_obligation(obligation));

    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    if !errors.is_empty() {
        return None;
    }

    let impl_source = infcx.resolve_vars_if_possible(impl_source);
    if impl_source.has_non_region_infer() {
        return None;
    }

    match impl_source {
        ImplSource::UserDefined(impl_source) => Some((impl_source.impl_def_id, impl_source.args)),
        ImplSource::Param(_) | ImplSource::Builtin(..) => None,
    }
}

fn is_inherent_impl_coherent<'db>(
    db: &'db dyn HirDatabase,
    def_map: &DefMap,
    impl_id: ImplId,
    self_ty: Ty<'db>,
) -> bool {
    let self_ty = self_ty.kind();
    let impl_allowed = match self_ty {
        TyKind::Tuple(_)
        | TyKind::FnDef(_, _)
        | TyKind::Array(_, _)
        | TyKind::Never
        | TyKind::RawPtr(_, _)
        | TyKind::Ref(_, _, _)
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_) => def_map.is_rustc_coherence_is_core(),

        TyKind::Adt(adt_def, _) => adt_def.def_id().0.module(db).krate() == def_map.krate(),
        TyKind::Dynamic(it, _) => it
            .principal_def_id()
            .is_some_and(|trait_id| trait_id.0.module(db).krate() == def_map.krate()),

        _ => true,
    };
    impl_allowed || {
        let rustc_has_incoherent_inherent_impls = match self_ty {
            TyKind::Tuple(_)
            | TyKind::FnDef(_, _)
            | TyKind::Array(_, _)
            | TyKind::Never
            | TyKind::RawPtr(_, _)
            | TyKind::Ref(_, _, _)
            | TyKind::Slice(_)
            | TyKind::Str
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_) => true,

            TyKind::Adt(adt_def, _) => match adt_def.def_id().0 {
                hir_def::AdtId::StructId(id) => db
                    .struct_signature(id)
                    .flags
                    .contains(StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
                hir_def::AdtId::UnionId(id) => db
                    .union_signature(id)
                    .flags
                    .contains(StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
                hir_def::AdtId::EnumId(it) => db
                    .enum_signature(it)
                    .flags
                    .contains(EnumFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
            },
            TyKind::Dynamic(it, _) => it.principal_def_id().is_some_and(|trait_id| {
                db.trait_signature(trait_id.0)
                    .flags
                    .contains(TraitFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS)
            }),

            _ => false,
        };
        let items = impl_id.impl_items(db);
        rustc_has_incoherent_inherent_impls
            && !items.items.is_empty()
            && items.items.iter().all(|&(_, assoc)| match assoc {
                AssocItemId::FunctionId(it) => {
                    db.function_signature(it).flags.contains(FnFlags::RUSTC_ALLOW_INCOHERENT_IMPL)
                }
                AssocItemId::ConstId(it) => {
                    db.const_signature(it).flags.contains(ConstFlags::RUSTC_ALLOW_INCOHERENT_IMPL)
                }
                AssocItemId::TypeAliasId(it) => db
                    .type_alias_signature(it)
                    .flags
                    .contains(TypeAliasFlags::RUSTC_ALLOW_INCOHERENT_IMPL),
            })
    }
}

/// Checks whether the impl satisfies the orphan rules.
///
/// Given `impl<P1..=Pn> Trait<T1..=Tn> for T0`, an `impl`` is valid only if at least one of the following is true:
/// - Trait is a local trait
/// - All of
///   - At least one of the types `T0..=Tn`` must be a local type. Let `Ti`` be the first such type.
///   - No uncovered type parameters `P1..=Pn` may appear in `T0..Ti`` (excluding `Ti`)
pub fn check_orphan_rules<'db>(db: &'db dyn HirDatabase, impl_: ImplId) -> bool {
    let Some(impl_trait) = db.impl_trait(impl_) else {
        // not a trait impl
        return true;
    };

    let local_crate = impl_.lookup(db).container.krate();
    let is_local = |tgt_crate| tgt_crate == local_crate;

    let trait_ref = impl_trait.instantiate_identity();
    let trait_id = trait_ref.def_id.0;
    if is_local(trait_id.module(db).krate()) {
        // trait to be implemented is local
        return true;
    }

    let unwrap_fundamental = |mut ty: Ty<'db>| {
        // Unwrap all layers of fundamental types with a loop.
        loop {
            match ty.kind() {
                TyKind::Ref(_, referenced, _) => ty = referenced,
                TyKind::Adt(adt_def, subs) => {
                    let AdtId::StructId(s) = adt_def.def_id().0 else {
                        break ty;
                    };
                    let struct_signature = db.struct_signature(s);
                    if struct_signature.flags.contains(StructFlags::FUNDAMENTAL) {
                        let next = subs.types().next();
                        match next {
                            Some(it) => ty = it,
                            None => break ty,
                        }
                    } else {
                        break ty;
                    }
                }
                _ => break ty,
            }
        }
    };
    //   - At least one of the types `T0..=Tn`` must be a local type. Let `Ti`` be the first such type.

    // FIXME: param coverage
    //   - No uncovered type parameters `P1..=Pn` may appear in `T0..Ti`` (excluding `Ti`)
    let is_not_orphan = trait_ref.args.types().any(|ty| match unwrap_fundamental(ty).kind() {
        TyKind::Adt(adt_def, _) => is_local(adt_def.def_id().0.module(db).krate()),
        TyKind::Error(_) => true,
        TyKind::Dynamic(it, _) => {
            it.principal_def_id().is_some_and(|trait_id| is_local(trait_id.0.module(db).krate()))
        }
        _ => false,
    });
    #[allow(clippy::let_and_return)]
    is_not_orphan
}

/// To be used from `hir` only.
pub fn iterate_path_candidates<'db>(
    ty: &Canonical<'db, Ty<'db>>,
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    iterate_method_candidates_dyn(
        ty,
        db,
        env,
        traits_in_scope,
        visible_from_module,
        name,
        LookupMode::Path,
        // the adjustments are not relevant for path lookup
        callback,
    )
}

/// To be used from `hir` only.
pub fn iterate_method_candidates_dyn<'db>(
    ty: &Canonical<'db, Ty<'db>>,
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mode: LookupMode,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    iterate_method_candidates_dyn_impl(
        ty,
        &mut InferenceTable::new(db, env, None),
        traits_in_scope,
        visible_from_module,
        name,
        mode,
        callback,
    )
}

fn iterate_method_candidates_dyn_impl<'db>(
    ty: &Canonical<'db, Ty<'db>>,
    table: &mut InferenceTable<'db>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mode: LookupMode,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    let _p = tracing::info_span!(
        "iterate_method_candidates_dyn",
        ?mode,
        ?name,
        traits_in_scope_len = traits_in_scope.len()
    )
    .entered();

    match mode {
        LookupMode::MethodCall => {
            // For method calls, rust first does any number of autoderef, and
            // then one autoref (i.e. when the method takes &self or &mut self).
            // Note that when we've got a receiver like &S, even if the method
            // we find in the end takes &self, we still do the autoderef step
            // (just as rustc does an autoderef and then autoref again).

            // We have to be careful about the order we're looking at candidates
            // in here. Consider the case where we're resolving `it.clone()`
            // where `it: &Vec<_>`. This resolves to the clone method with self
            // type `Vec<_>`, *not* `&_`. I.e. we need to consider methods where
            // the receiver type exactly matches before cases where we have to
            // do autoref. But in the autoderef steps, the `&_` self type comes
            // up *before* the `Vec<_>` self type.
            //
            // On the other hand, we don't want to just pick any by-value method
            // before any by-autoref method; it's just that we need to consider
            // the methods by autoderef order of *receiver types*, not *self
            // types*.

            table.run_in_snapshot(|table| {
                let ty = table.instantiate_canonical(*ty);
                let deref_chain = autoderef_method_receiver(table, ty);

                deref_chain.into_iter().try_for_each(|(receiver_ty, adj)| {
                    iterate_method_candidates_with_autoref(
                        table,
                        receiver_ty,
                        adj,
                        traits_in_scope,
                        visible_from_module,
                        name,
                        callback,
                    )
                })
            })
        }
        LookupMode::Path => {
            // No autoderef for path lookups
            iterate_method_candidates_for_self_ty(
                ty,
                table,
                traits_in_scope,
                visible_from_module,
                name,
                callback,
            )
        }
    }
}

#[tracing::instrument(skip_all, fields(name = ?name))]
fn iterate_method_candidates_with_autoref<'db>(
    table: &mut InferenceTable<'db>,
    receiver_ty: Canonical<'db, Ty<'db>>,
    first_adjustment: ReceiverAdjustments,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    let interner = table.interner();

    let mut iterate_method_candidates_by_receiver = move |receiver_ty, first_adjustment| {
        iterate_method_candidates_by_receiver(
            table,
            receiver_ty,
            first_adjustment,
            traits_in_scope,
            visible_from_module,
            name,
            callback,
        )
    };

    let mut maybe_reborrowed = first_adjustment.clone();
    if let TyKind::Ref(_, _, m) = receiver_ty.value.kind() {
        // Prefer reborrow of references to move
        maybe_reborrowed.autoref = Some(AutorefOrPtrAdjustment::Autoref(m));
        maybe_reborrowed.autoderefs += 1;
    }

    iterate_method_candidates_by_receiver(receiver_ty, maybe_reborrowed)?;

    let refed = Canonical {
        max_universe: receiver_ty.max_universe,
        variables: receiver_ty.variables,
        value: Ty::new_ref(interner, Region::error(interner), receiver_ty.value, Mutability::Not),
    };

    iterate_method_candidates_by_receiver(
        refed,
        first_adjustment.with_autoref(AutorefOrPtrAdjustment::Autoref(Mutability::Not)),
    )?;

    let ref_muted = Canonical {
        max_universe: receiver_ty.max_universe,
        variables: receiver_ty.variables,
        value: Ty::new_ref(interner, Region::error(interner), receiver_ty.value, Mutability::Mut),
    };

    iterate_method_candidates_by_receiver(
        ref_muted,
        first_adjustment.with_autoref(AutorefOrPtrAdjustment::Autoref(Mutability::Mut)),
    )?;

    if let TyKind::RawPtr(ty, Mutability::Mut) = receiver_ty.value.kind() {
        let const_ptr_ty = rustc_type_ir::Canonical {
            max_universe: rustc_type_ir::UniverseIndex::ZERO,
            value: Ty::new_ptr(interner, ty, Mutability::Not),
            variables: receiver_ty.variables,
        };
        iterate_method_candidates_by_receiver(
            const_ptr_ty,
            first_adjustment.with_autoref(AutorefOrPtrAdjustment::ToConstPtr),
        )?;
    }

    ControlFlow::Continue(())
}

pub trait MethodCandidateCallback {
    fn on_inherent_method(
        &mut self,
        adjustments: ReceiverAdjustments,
        item: AssocItemId,
        is_visible: bool,
    ) -> ControlFlow<()>;

    fn on_trait_method(
        &mut self,
        adjustments: ReceiverAdjustments,
        item: AssocItemId,
        is_visible: bool,
    ) -> ControlFlow<()>;
}

impl<F> MethodCandidateCallback for F
where
    F: FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
{
    fn on_inherent_method(
        &mut self,
        adjustments: ReceiverAdjustments,
        item: AssocItemId,
        is_visible: bool,
    ) -> ControlFlow<()> {
        self(adjustments, item, is_visible)
    }

    fn on_trait_method(
        &mut self,
        adjustments: ReceiverAdjustments,
        item: AssocItemId,
        is_visible: bool,
    ) -> ControlFlow<()> {
        self(adjustments, item, is_visible)
    }
}

#[tracing::instrument(skip_all, fields(name = ?name))]
fn iterate_method_candidates_by_receiver<'db>(
    table: &mut InferenceTable<'db>,
    receiver_ty: Canonical<'db, Ty<'db>>,
    receiver_adjustments: ReceiverAdjustments,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    let receiver_ty = table.instantiate_canonical(receiver_ty);
    // We're looking for methods with *receiver* type receiver_ty. These could
    // be found in any of the derefs of receiver_ty, so we have to go through
    // that, including raw derefs.
    table.run_in_snapshot(|table| {
        let mut autoderef = autoderef::Autoderef::new_no_tracking(table, receiver_ty)
            .include_raw_pointers()
            .use_receiver_trait();
        while let Some((self_ty, _)) = autoderef.next() {
            iterate_inherent_methods(
                self_ty,
                autoderef.table,
                name,
                Some(receiver_ty),
                Some(receiver_adjustments.clone()),
                visible_from_module,
                LookupMode::MethodCall,
                &mut |adjustments, item, is_visible| {
                    callback.on_inherent_method(adjustments, item, is_visible)
                },
            )?
        }
        ControlFlow::Continue(())
    })?;
    table.run_in_snapshot(|table| {
        let mut autoderef = autoderef::Autoderef::new_no_tracking(table, receiver_ty)
            .include_raw_pointers()
            .use_receiver_trait();
        while let Some((self_ty, _)) = autoderef.next() {
            if matches!(self_ty.kind(), TyKind::Infer(rustc_type_ir::TyVar(_))) {
                // don't try to resolve methods on unknown types
                return ControlFlow::Continue(());
            }

            iterate_trait_method_candidates(
                self_ty,
                autoderef.table,
                traits_in_scope,
                name,
                Some(receiver_ty),
                Some(receiver_adjustments.clone()),
                LookupMode::MethodCall,
                &mut |adjustments, item, is_visible| {
                    callback.on_trait_method(adjustments, item, is_visible)
                },
            )?
        }
        ControlFlow::Continue(())
    })
}

#[tracing::instrument(skip_all, fields(name = ?name))]
fn iterate_method_candidates_for_self_ty<'db>(
    self_ty: &Canonical<'db, Ty<'db>>,
    table: &mut InferenceTable<'db>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    table.run_in_snapshot(|table| {
        let self_ty = table.instantiate_canonical(*self_ty);
        iterate_inherent_methods(
            self_ty,
            table,
            name,
            None,
            None,
            visible_from_module,
            LookupMode::Path,
            &mut |adjustments, item, is_visible| {
                callback.on_inherent_method(adjustments, item, is_visible)
            },
        )?;
        iterate_trait_method_candidates(
            self_ty,
            table,
            traits_in_scope,
            name,
            None,
            None,
            LookupMode::Path,
            &mut |adjustments, item, is_visible| {
                callback.on_trait_method(adjustments, item, is_visible)
            },
        )
    })
}

#[tracing::instrument(skip_all, fields(name = ?name, visible_from_module, receiver_ty))]
fn iterate_trait_method_candidates<'db>(
    self_ty: Ty<'db>,
    table: &mut InferenceTable<'db>,
    traits_in_scope: &FxHashSet<TraitId>,
    name: Option<&Name>,
    receiver_ty: Option<Ty<'db>>,
    receiver_adjustments: Option<ReceiverAdjustments>,
    mode: LookupMode,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let db = table.db;

    let canonical_self_ty = table.canonicalize(self_ty);
    let krate = table.trait_env.krate;

    'traits: for &t in traits_in_scope {
        let data = db.trait_signature(t);

        // Traits annotated with `#[rustc_skip_during_method_dispatch]` are skipped during
        // method resolution, if the receiver is an array, and we're compiling for editions before
        // 2021.
        // This is to make `[a].into_iter()` not break code with the new `IntoIterator` impl for
        // arrays.
        if data.flags.contains(TraitFlags::SKIP_ARRAY_DURING_METHOD_DISPATCH)
            && matches!(self_ty.kind(), TyKind::Array(..))
        {
            // FIXME: this should really be using the edition of the method name's span, in case it
            // comes from a macro
            if !krate.data(db).edition.at_least_2021() {
                continue;
            }
        }
        if data.flags.contains(TraitFlags::SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH)
            && matches!(
                self_ty.kind(), TyKind::Adt(adt_def, subst)
                if is_box(table.db, adt_def.def_id().0)
                    && matches!(subst.type_at(0).kind(), TyKind::Slice(..))
            )
        {
            // FIXME: this should really be using the edition of the method name's span, in case it
            // comes from a macro
            if !krate.data(db).edition.at_least_2024() {
                continue;
            }
        }

        // we'll be lazy about checking whether the type implements the
        // trait, but if we find out it doesn't, we'll skip the rest of the
        // iteration
        let mut known_implemented = false;
        for &(_, item) in t.trait_items(db).items.iter() {
            // Don't pass a `visible_from_module` down to `is_valid_candidate`,
            // since only inherent methods should be included into visibility checking.
            let visible = match is_valid_trait_method_candidate(
                table,
                t,
                name,
                receiver_ty,
                item,
                self_ty,
                mode,
            ) {
                IsValidCandidate::Yes => true,
                IsValidCandidate::NotVisible => false,
                IsValidCandidate::No => continue,
            };
            if !known_implemented {
                let goal = generic_implements_goal_ns(table, t, canonical_self_ty);
                if next_trait_solve_canonical_in_ctxt(&table.infer_ctxt, goal).no_solution() {
                    continue 'traits;
                }
            }
            known_implemented = true;
            callback(receiver_adjustments.clone().unwrap_or_default(), item, visible)?;
        }
    }
    ControlFlow::Continue(())
}

#[tracing::instrument(skip_all, fields(name = ?name, visible_from_module, receiver_ty))]
fn iterate_inherent_methods<'db>(
    self_ty: Ty<'db>,
    table: &mut InferenceTable<'db>,
    name: Option<&Name>,
    receiver_ty: Option<Ty<'db>>,
    receiver_adjustments: Option<ReceiverAdjustments>,
    visible_from_module: VisibleFromModule,
    mode: LookupMode,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let db = table.db;
    let env = table.trait_env.clone();

    // For trait object types and placeholder types with trait bounds, the methods of the trait and
    // its super traits are considered inherent methods. This matters because these methods have
    // higher priority than the other traits' methods, which would be considered in
    // `iterate_trait_method_candidates()` only after this function.
    match self_ty.kind() {
        TyKind::Param(_) => {
            let env = table.trait_env.clone();
            let traits =
                env.traits_in_scope_from_clauses(self_ty).flat_map(|t| all_super_traits(db, t));
            iterate_inherent_trait_methods(
                self_ty,
                table,
                name,
                receiver_ty,
                receiver_adjustments.clone(),
                callback,
                traits,
                mode,
            )?;
        }
        TyKind::Dynamic(bounds, _) => {
            if let Some(principal_trait) = bounds.principal_def_id() {
                let traits = all_super_traits(db, principal_trait.0);
                iterate_inherent_trait_methods(
                    self_ty,
                    table,
                    name,
                    receiver_ty,
                    receiver_adjustments.clone(),
                    callback,
                    traits.into_iter(),
                    mode,
                )?;
            }
        }
        _ => {}
    }

    let def_crates = match def_crates(db, self_ty, env.krate) {
        Some(k) => k,
        None => return ControlFlow::Continue(()),
    };

    let (module, mut block) = match visible_from_module {
        VisibleFromModule::Filter(module) => (Some(module), module.containing_block()),
        VisibleFromModule::IncludeBlock(block) => (None, Some(block)),
        VisibleFromModule::None => (None, None),
    };

    while let Some(block_id) = block {
        if let Some(impls) = db.inherent_impls_in_block(block_id) {
            impls_for_self_ty(
                &impls,
                self_ty,
                table,
                name,
                receiver_ty,
                receiver_adjustments.clone(),
                module,
                callback,
            )?;
        }

        block = block_def_map(db, block_id).parent().and_then(|module| module.containing_block());
    }

    for krate in def_crates {
        let impls = db.inherent_impls_in_crate(krate);
        impls_for_self_ty(
            &impls,
            self_ty,
            table,
            name,
            receiver_ty,
            receiver_adjustments.clone(),
            module,
            callback,
        )?;
    }
    return ControlFlow::Continue(());

    #[tracing::instrument(skip_all, fields(name = ?name, visible_from_module, receiver_ty))]
    fn iterate_inherent_trait_methods<'db>(
        self_ty: Ty<'db>,
        table: &mut InferenceTable<'db>,
        name: Option<&Name>,
        receiver_ty: Option<Ty<'db>>,
        receiver_adjustments: Option<ReceiverAdjustments>,
        callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
        traits: impl Iterator<Item = TraitId>,
        mode: LookupMode,
    ) -> ControlFlow<()> {
        let db = table.db;
        for t in traits {
            let data = t.trait_items(db);
            for &(_, item) in data.items.iter() {
                // We don't pass `visible_from_module` as all trait items should be visible.
                let visible = match is_valid_trait_method_candidate(
                    table,
                    t,
                    name,
                    receiver_ty,
                    item,
                    self_ty,
                    mode,
                ) {
                    IsValidCandidate::Yes => true,
                    IsValidCandidate::NotVisible => false,
                    IsValidCandidate::No => continue,
                };
                callback(receiver_adjustments.clone().unwrap_or_default(), item, visible)?;
            }
        }
        ControlFlow::Continue(())
    }

    #[tracing::instrument(skip_all, fields(name = ?name, visible_from_module, receiver_ty))]
    fn impls_for_self_ty<'db>(
        impls: &InherentImpls,
        self_ty: Ty<'db>,
        table: &mut InferenceTable<'db>,
        name: Option<&Name>,
        receiver_ty: Option<Ty<'db>>,
        receiver_adjustments: Option<ReceiverAdjustments>,
        visible_from_module: Option<ModuleId>,
        callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        for &impl_id in impls.for_self_ty(self_ty) {
            for &(ref item_name, item) in impl_id.impl_items(table.db).items.iter() {
                let visible = match is_valid_impl_method_candidate(
                    table,
                    self_ty,
                    receiver_ty,
                    visible_from_module,
                    name,
                    impl_id,
                    item,
                    item_name,
                ) {
                    IsValidCandidate::Yes => true,
                    IsValidCandidate::NotVisible => false,
                    IsValidCandidate::No => continue,
                };
                callback(receiver_adjustments.clone().unwrap_or_default(), item, visible)?;
            }
        }
        ControlFlow::Continue(())
    }
}

/// Returns the receiver type for the index trait call.
pub(crate) fn resolve_indexing_op<'db>(
    table: &mut InferenceTable<'db>,
    ty: Canonical<'db, Ty<'db>>,
    index_trait: TraitId,
) -> Option<ReceiverAdjustments> {
    let ty = table.instantiate_canonical(ty);
    let deref_chain = autoderef_method_receiver(table, ty);
    for (ty, adj) in deref_chain {
        let goal = generic_implements_goal_ns(table, index_trait, ty);
        if !next_trait_solve_canonical_in_ctxt(&table.infer_ctxt, goal).no_solution() {
            return Some(adj);
        }
    }
    None
}

// FIXME: Replace this with a `Try` impl once stable
macro_rules! check_that {
    ($cond:expr) => {
        if !$cond {
            return IsValidCandidate::No;
        }
    };
}

#[derive(Debug)]
enum IsValidCandidate {
    Yes,
    No,
    NotVisible,
}

#[tracing::instrument(skip_all, fields(name))]
fn is_valid_impl_method_candidate<'db>(
    table: &mut InferenceTable<'db>,
    self_ty: Ty<'db>,
    receiver_ty: Option<Ty<'db>>,
    visible_from_module: Option<ModuleId>,
    name: Option<&Name>,
    impl_id: ImplId,
    item: AssocItemId,
    item_name: &Name,
) -> IsValidCandidate {
    match item {
        AssocItemId::FunctionId(f) => is_valid_impl_fn_candidate(
            table,
            impl_id,
            f,
            name,
            receiver_ty,
            self_ty,
            visible_from_module,
            item_name,
        ),
        AssocItemId::ConstId(c) => {
            let db = table.db;
            check_that!(receiver_ty.is_none());
            check_that!(name.is_none_or(|n| n == item_name));

            if let Some(from_module) = visible_from_module
                && !db.assoc_visibility(c.into()).is_visible_from(db, from_module)
            {
                cov_mark::hit!(const_candidate_not_visible);
                return IsValidCandidate::NotVisible;
            }
            let self_ty_matches = table.run_in_snapshot(|table| {
                let impl_args = table.fresh_args_for_item(impl_id.into());
                let expected_self_ty =
                    db.impl_self_ty(impl_id).instantiate(table.interner(), impl_args);
                table.unify(expected_self_ty, self_ty)
            });
            if !self_ty_matches {
                cov_mark::hit!(const_candidate_self_type_mismatch);
                return IsValidCandidate::No;
            }
            IsValidCandidate::Yes
        }
        _ => IsValidCandidate::No,
    }
}

/// Checks whether a given `AssocItemId` is applicable for `receiver_ty`.
#[tracing::instrument(skip_all, fields(name))]
fn is_valid_trait_method_candidate<'db>(
    table: &mut InferenceTable<'db>,
    trait_id: TraitId,
    name: Option<&Name>,
    receiver_ty: Option<Ty<'db>>,
    item: AssocItemId,
    self_ty: Ty<'db>,
    mode: LookupMode,
) -> IsValidCandidate {
    let db = table.db;
    match item {
        AssocItemId::FunctionId(fn_id) => {
            let data = db.function_signature(fn_id);

            check_that!(name.is_none_or(|n| n == &data.name));

            table.run_in_snapshot(|table| {
                let impl_subst = table.fresh_args_for_item(trait_id.into());
                let expect_self_ty = impl_subst.type_at(0);

                check_that!(table.unify(expect_self_ty, self_ty));

                if let Some(receiver_ty) = receiver_ty {
                    check_that!(data.has_self_param());

                    let args = table.fill_rest_fresh_args(fn_id.into(), impl_subst);

                    let sig = db.callable_item_signature(fn_id.into());
                    let expected_receiver = sig
                        .map_bound(|s| s.skip_binder().inputs_and_output.as_slice()[0])
                        .instantiate(table.interner(), args);

                    // FIXME: Clean up this mess with some context struct like rustc's `ProbeContext`
                    let variance = match mode {
                        LookupMode::MethodCall => rustc_type_ir::Variance::Covariant,
                        LookupMode::Path => rustc_type_ir::Variance::Invariant,
                    };
                    let res = table
                        .infer_ctxt
                        .at(&ObligationCause::dummy(), table.trait_env.env)
                        .relate(expected_receiver, variance, receiver_ty);
                    let Ok(infer_ok) = res else {
                        return IsValidCandidate::No;
                    };

                    if !infer_ok.obligations.is_empty() {
                        let mut ctxt = ObligationCtxt::new(&table.infer_ctxt);
                        ctxt.register_obligations(infer_ok.into_obligations());
                        // FIXME: Are we doing this correctly? Probably better to follow rustc more closely.
                        check_that!(ctxt.try_evaluate_obligations().is_empty());
                    }

                    check_that!(table.unify(receiver_ty, expected_receiver));
                }

                IsValidCandidate::Yes
            })
        }
        AssocItemId::ConstId(c) => {
            check_that!(receiver_ty.is_none());
            check_that!(name.is_none_or(|n| db.const_signature(c).name.as_ref() == Some(n)));

            IsValidCandidate::Yes
        }
        _ => IsValidCandidate::No,
    }
}

#[tracing::instrument(skip_all, fields(name))]
fn is_valid_impl_fn_candidate<'db>(
    table: &mut InferenceTable<'db>,
    impl_id: ImplId,
    fn_id: FunctionId,
    name: Option<&Name>,
    receiver_ty: Option<Ty<'db>>,
    self_ty: Ty<'db>,
    visible_from_module: Option<ModuleId>,
    item_name: &Name,
) -> IsValidCandidate {
    check_that!(name.is_none_or(|n| n == item_name));

    let db = table.db;
    let data = db.function_signature(fn_id);

    if let Some(from_module) = visible_from_module
        && !db.assoc_visibility(fn_id.into()).is_visible_from(db, from_module)
    {
        cov_mark::hit!(autoderef_candidate_not_visible);
        return IsValidCandidate::NotVisible;
    }
    table.run_in_snapshot(|table| {
        let _p = tracing::info_span!("subst_for_def").entered();
        let impl_subst = table.infer_ctxt.fresh_args_for_item(impl_id.into());
        let expect_self_ty = db.impl_self_ty(impl_id).instantiate(table.interner(), &impl_subst);

        check_that!(table.unify(expect_self_ty, self_ty));

        if let Some(receiver_ty) = receiver_ty {
            let _p = tracing::info_span!("check_receiver_ty").entered();
            check_that!(data.has_self_param());

            let args = table.infer_ctxt.fresh_args_for_item(fn_id.into());

            let sig = db.callable_item_signature(fn_id.into());
            let expected_receiver = sig
                .map_bound(|s| s.skip_binder().inputs_and_output.as_slice()[0])
                .instantiate(table.interner(), args);

            check_that!(table.unify(receiver_ty, expected_receiver));
        }

        // We need to consider the bounds on the impl to distinguish functions of the same name
        // for a type.
        let predicates = db.generic_predicates(impl_id.into());
        let Some(predicates) = predicates.instantiate(table.interner(), impl_subst) else {
            return IsValidCandidate::Yes;
        };

        let mut ctxt = ObligationCtxt::new(&table.infer_ctxt);

        ctxt.register_obligations(predicates.into_iter().map(|p| {
            PredicateObligation::new(
                table.interner(),
                ObligationCause::new(),
                table.trait_env.env,
                p.0,
            )
        }));

        if ctxt.try_evaluate_obligations().is_empty() {
            IsValidCandidate::Yes
        } else {
            IsValidCandidate::No
        }
    })
}

/// This creates Substs for a trait with the given Self type and type variables
/// for all other parameters, to query the trait solver with it.
#[tracing::instrument(skip_all)]
fn generic_implements_goal_ns<'db>(
    table: &mut InferenceTable<'db>,
    trait_: TraitId,
    self_ty: Canonical<'db, Ty<'db>>,
) -> Canonical<'db, Goal<'db, Predicate<'db>>> {
    let args = table.infer_ctxt.fresh_args_for_item(SolverDefId::TraitId(trait_));
    let self_ty = table.instantiate_canonical(self_ty);
    let trait_ref =
        rustc_type_ir::TraitRef::new_from_args(table.infer_ctxt.interner, trait_.into(), args)
            .with_replaced_self_ty(table.infer_ctxt.interner, self_ty);
    let goal = Goal::new(table.infer_ctxt.interner, table.trait_env.env, trait_ref);

    table.canonicalize(goal)
}

fn autoderef_method_receiver<'db>(
    table: &mut InferenceTable<'db>,
    ty: Ty<'db>,
) -> Vec<(Canonical<'db, Ty<'db>>, ReceiverAdjustments)> {
    let interner = table.interner();
    let mut deref_chain = Vec::new();
    let mut autoderef = autoderef::Autoderef::new_no_tracking(table, ty).use_receiver_trait();
    while let Some((ty, derefs)) = autoderef.next() {
        deref_chain.push((
            autoderef.table.canonicalize(ty),
            ReceiverAdjustments { autoref: None, autoderefs: derefs, unsize_array: false },
        ));
    }
    // As a last step, we can do array unsizing (that's the only unsizing that rustc does for method receivers!)
    if let Some((rustc_type_ir::Array(parameters, _), variables, max_universe, adj)) =
        deref_chain.last().map(|d| (d.0.value.kind(), d.0.variables, d.0.max_universe, d.1.clone()))
    {
        let unsized_ty = Ty::new_slice(interner, parameters);
        deref_chain.push((
            Canonical { max_universe, value: unsized_ty, variables },
            ReceiverAdjustments { unsize_array: true, ..adj.clone() },
        ));
    }
    deref_chain
}
