//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in rustc_hir_analysis/check/method/probe.rs.
use std::ops::ControlFlow;

use arrayvec::ArrayVec;
use base_db::Crate;
use chalk_ir::{UniverseIndex, WithKind, cast::Cast};
use hir_def::{
    AssocItemId, BlockId, ConstId, FunctionId, HasModule, ImplId, ItemContainerId, Lookup,
    ModuleId, TraitId,
    nameres::{DefMap, assoc::ImplItems, block_def_map, crate_def_map},
    signatures::{ConstFlags, EnumFlags, FnFlags, StructFlags, TraitFlags, TypeAliasFlags},
};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use stdx::never;
use triomphe::Arc;

use crate::{
    AdtId, Canonical, CanonicalVarKinds, DebruijnIndex, DynTyExt, ForeignDefId, GenericArgData,
    Goal, Guidance, InEnvironment, Interner, Mutability, Scalar, Solution, Substitution,
    TraitEnvironment, TraitRef, TraitRefExt, Ty, TyBuilder, TyExt, TyKind, TyVariableKind,
    VariableKind, WhereClause,
    autoderef::{self, AutoderefKind},
    db::HirDatabase,
    error_lifetime, from_chalk_trait_id, from_foreign_def_id,
    infer::{Adjust, Adjustment, OverloadedDeref, PointerCast, unify::InferenceTable},
    lang_items::is_box,
    primitive::{FloatTy, IntTy, UintTy},
    to_chalk_trait_id,
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
    RawPtr(Mutability),
    Scalar(Scalar),
    // These can have user-defined impls:
    Adt(hir_def::AdtId),
    Dyn(TraitId),
    ForeignType(ForeignDefId),
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
    pub fn for_inherent_impl(ty: &Ty) -> Option<TyFingerprint> {
        let fp = match ty.kind(Interner) {
            TyKind::Str => TyFingerprint::Str,
            TyKind::Never => TyFingerprint::Never,
            TyKind::Slice(..) => TyFingerprint::Slice,
            TyKind::Array(..) => TyFingerprint::Array,
            TyKind::Scalar(scalar) => TyFingerprint::Scalar(*scalar),
            TyKind::Adt(AdtId(adt), _) => TyFingerprint::Adt(*adt),
            TyKind::Raw(mutability, ..) => TyFingerprint::RawPtr(*mutability),
            TyKind::Foreign(alias_id, ..) => TyFingerprint::ForeignType(*alias_id),
            TyKind::Dyn(_) => ty.dyn_trait().map(TyFingerprint::Dyn)?,
            _ => return None,
        };
        Some(fp)
    }

    /// Creates a TyFingerprint for looking up a trait impl.
    pub fn for_trait_impl(ty: &Ty) -> Option<TyFingerprint> {
        let fp = match ty.kind(Interner) {
            TyKind::Str => TyFingerprint::Str,
            TyKind::Never => TyFingerprint::Never,
            TyKind::Slice(..) => TyFingerprint::Slice,
            TyKind::Array(..) => TyFingerprint::Array,
            TyKind::Scalar(scalar) => TyFingerprint::Scalar(*scalar),
            TyKind::Adt(AdtId(adt), _) => TyFingerprint::Adt(*adt),
            TyKind::Raw(mutability, ..) => TyFingerprint::RawPtr(*mutability),
            TyKind::Foreign(alias_id, ..) => TyFingerprint::ForeignType(*alias_id),
            TyKind::Dyn(_) => ty.dyn_trait().map(TyFingerprint::Dyn)?,
            TyKind::Ref(_, _, ty) => return TyFingerprint::for_trait_impl(ty),
            TyKind::Tuple(_, subst) => {
                let first_ty = subst.interned().first().map(|arg| arg.assert_ty_ref(Interner));
                match first_ty {
                    Some(ty) => return TyFingerprint::for_trait_impl(ty),
                    None => TyFingerprint::Unit,
                }
            }
            TyKind::AssociatedType(_, _)
            | TyKind::OpaqueType(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Closure(_, _)
            | TyKind::Coroutine(..)
            | TyKind::CoroutineWitness(..) => TyFingerprint::Unnameable,
            TyKind::Function(fn_ptr) => {
                TyFingerprint::Function(fn_ptr.substitution.0.len(Interner) as u32)
            }
            TyKind::Alias(_)
            | TyKind::Placeholder(_)
            | TyKind::BoundVar(_)
            | TyKind::InferenceVar(_, _)
            | TyKind::Error => return None,
        };
        Some(fp)
    }
}

pub(crate) const ALL_INT_FPS: [TyFingerprint; 12] = [
    TyFingerprint::Scalar(Scalar::Int(IntTy::I8)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I16)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I32)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I64)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::I128)),
    TyFingerprint::Scalar(Scalar::Int(IntTy::Isize)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U8)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U16)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U32)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U64)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::U128)),
    TyFingerprint::Scalar(Scalar::Uint(UintTy::Usize)),
];

pub(crate) const ALL_FLOAT_FPS: [TyFingerprint; 4] = [
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F16)),
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F32)),
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F64)),
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F128)),
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
                    Some(tr) => tr.skip_binders().hir_trait_id(),
                    None => continue,
                };
                let self_ty = db.impl_self_ty(impl_id);
                let self_ty_fp = TyFingerprint::for_trait_impl(self_ty.skip_binders());
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
                let self_ty = self_ty.skip_binders();

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

    pub fn for_self_ty(&self, self_ty: &Ty) -> &[ImplId] {
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

pub fn def_crates(db: &dyn HirDatabase, ty: &Ty, cur_crate: Crate) -> Option<SmallVec<[Crate; 2]>> {
    match ty.kind(Interner) {
        &TyKind::Adt(AdtId(def_id), _) => {
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
        &TyKind::Foreign(id) => {
            let alias = from_foreign_def_id(id);
            Some(
                if db
                    .type_alias_signature(alias)
                    .flags
                    .contains(TypeAliasFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPL)
                {
                    db.incoherent_inherent_impl_crates(cur_crate, TyFingerprint::ForeignType(id))
                } else {
                    smallvec![alias.module(db).krate()]
                },
            )
        }
        TyKind::Dyn(_) => {
            let trait_id = ty.dyn_trait()?;
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
        TyKind::Scalar(_)
        | TyKind::Str
        | TyKind::Slice(_)
        | TyKind::Array(..)
        | TyKind::Raw(..) => Some(db.incoherent_inherent_impl_crates(
            cur_crate,
            TyFingerprint::for_inherent_impl(ty).expect("fingerprint for primitive"),
        )),
        _ => None,
    }
}

/// Look up the method with the given name.
pub(crate) fn lookup_method(
    db: &dyn HirDatabase,
    ty: &Canonical<Ty>,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: &Name,
) -> Option<(ReceiverAdjustments, FunctionId, bool)> {
    let mut not_visible = None;
    let res = iterate_method_candidates(
        ty,
        db,
        env,
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
    pub(crate) fn apply(&self, table: &mut InferenceTable<'_>, ty: Ty) -> (Ty, Vec<Adjustment>) {
        let mut ty = table.resolve_ty_shallow(&ty);
        let mut adjust = Vec::new();
        for _ in 0..self.autoderefs {
            match autoderef::autoderef_step(table, ty.clone(), true, false) {
                None => {
                    never!("autoderef not possible for {:?}", ty);
                    ty = TyKind::Error.intern(Interner);
                    break;
                }
                Some((kind, new_ty)) => {
                    ty = new_ty.clone();
                    let mutbl = match self.autoref {
                        Some(AutorefOrPtrAdjustment::Autoref(m)) => Some(m),
                        Some(AutorefOrPtrAdjustment::ToConstPtr) => Some(Mutability::Not),
                        // FIXME should we know the mutability here, when autoref is `None`?
                        None => None,
                    };
                    adjust.push(Adjustment {
                        kind: Adjust::Deref(match kind {
                            AutoderefKind::Overloaded => Some(OverloadedDeref(mutbl)),
                            AutoderefKind::Builtin => None,
                        }),
                        target: new_ty,
                    });
                }
            }
        }
        if let Some(autoref) = &self.autoref {
            let lt = table.new_lifetime_var();
            match autoref {
                AutorefOrPtrAdjustment::Autoref(m) => {
                    let a = Adjustment::borrow(*m, ty, lt);
                    ty = a.target.clone();
                    adjust.push(a);
                }
                AutorefOrPtrAdjustment::ToConstPtr => {
                    if let TyKind::Raw(Mutability::Mut, pointee) = ty.kind(Interner) {
                        let a = Adjustment {
                            kind: Adjust::Pointer(PointerCast::MutToConstPointer),
                            target: TyKind::Raw(Mutability::Not, pointee.clone()).intern(Interner),
                        };
                        ty = a.target.clone();
                        adjust.push(a);
                    } else {
                        never!("`ToConstPtr` target is not a raw mutable pointer");
                    }
                }
            };
        }
        if self.unsize_array {
            ty = 'it: {
                if let TyKind::Ref(m, l, inner) = ty.kind(Interner) {
                    if let TyKind::Array(inner, _) = inner.kind(Interner) {
                        break 'it TyKind::Ref(
                            *m,
                            l.clone(),
                            TyKind::Slice(inner.clone()).intern(Interner),
                        )
                        .intern(Interner);
                    }
                }
                // FIXME: report diagnostic if array unsizing happens without indirection.
                ty
            };
            adjust.push(Adjustment {
                kind: Adjust::Pointer(PointerCast::Unsize),
                target: ty.clone(),
            });
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
pub(crate) fn iterate_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(ReceiverAdjustments, AssocItemId, bool) -> Option<T>,
) -> Option<T> {
    let mut slot = None;
    _ = iterate_method_candidates_dyn(
        ty,
        db,
        env,
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

pub fn lookup_impl_const(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    const_id: ConstId,
    subs: Substitution,
) -> (ConstId, Substitution) {
    let trait_id = match const_id.lookup(db).container {
        ItemContainerId::TraitId(id) => id,
        _ => return (const_id, subs),
    };
    let substitution = Substitution::from_iter(Interner, subs.iter(Interner));
    let trait_ref = TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution };

    let const_signature = db.const_signature(const_id);
    let name = match const_signature.name.as_ref() {
        Some(name) => name,
        None => return (const_id, subs),
    };

    lookup_impl_assoc_item_for_trait_ref(trait_ref, db, env, name)
        .and_then(
            |assoc| if let (AssocItemId::ConstId(id), s) = assoc { Some((id, s)) } else { None },
        )
        .unwrap_or((const_id, subs))
}

/// Checks if the self parameter of `Trait` method is the `dyn Trait` and we should
/// call the method using the vtable.
pub fn is_dyn_method(
    db: &dyn HirDatabase,
    _env: Arc<TraitEnvironment>,
    func: FunctionId,
    fn_subst: Substitution,
) -> Option<usize> {
    let ItemContainerId::TraitId(trait_id) = func.lookup(db).container else {
        return None;
    };
    let trait_params = db.generic_params(trait_id.into()).len();
    let fn_params = fn_subst.len(Interner) - trait_params;
    let trait_ref = TraitRef {
        trait_id: to_chalk_trait_id(trait_id),
        substitution: Substitution::from_iter(Interner, fn_subst.iter(Interner).take(trait_params)),
    };
    let self_ty = trait_ref.self_type_parameter(Interner);
    if let TyKind::Dyn(d) = self_ty.kind(Interner) {
        let is_my_trait_in_bounds = d
            .bounds
            .skip_binders()
            .as_slice(Interner)
            .iter()
            .map(|it| it.skip_binders())
            .flat_map(|it| match it {
                WhereClause::Implemented(tr) => {
                    all_super_traits(db, from_chalk_trait_id(tr.trait_id))
                }
                _ => smallvec![],
            })
            // rustc doesn't accept `impl Foo<2> for dyn Foo<5>`, so if the trait id is equal, no matter
            // what the generics are, we are sure that the method is come from the vtable.
            .any(|x| x == trait_id);
        if is_my_trait_in_bounds {
            return Some(fn_params);
        }
    }
    None
}

/// Looks up the impl method that actually runs for the trait method `func`.
///
/// Returns `func` if it's not a method defined in a trait or the lookup failed.
pub(crate) fn lookup_impl_method_query(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    func: FunctionId,
    fn_subst: Substitution,
) -> (FunctionId, Substitution) {
    let ItemContainerId::TraitId(trait_id) = func.lookup(db).container else {
        return (func, fn_subst);
    };
    let trait_params = db.generic_params(trait_id.into()).len();
    let trait_ref = TraitRef {
        trait_id: to_chalk_trait_id(trait_id),
        substitution: Substitution::from_iter(Interner, fn_subst.iter(Interner).take(trait_params)),
    };

    let name = &db.function_signature(func).name;
    let Some((impl_fn, impl_subst)) =
        lookup_impl_assoc_item_for_trait_ref(trait_ref, db, env, name).and_then(|assoc| {
            if let (AssocItemId::FunctionId(id), subst) = assoc { Some((id, subst)) } else { None }
        })
    else {
        return (func, fn_subst);
    };

    (
        impl_fn,
        Substitution::from_iter(
            Interner,
            impl_subst.iter(Interner).chain(fn_subst.iter(Interner).skip(trait_params)),
        ),
    )
}

fn lookup_impl_assoc_item_for_trait_ref(
    trait_ref: TraitRef,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    name: &Name,
) -> Option<(AssocItemId, Substitution)> {
    let hir_trait_id = trait_ref.hir_trait_id();
    let self_ty = trait_ref.self_type_parameter(Interner);
    let self_ty_fp = TyFingerprint::for_trait_impl(&self_ty)?;
    let impls = db.trait_impls_in_deps(env.krate);

    let trait_module = hir_trait_id.module(db);
    let type_module = match self_ty_fp {
        TyFingerprint::Adt(adt_id) => Some(adt_id.module(db)),
        TyFingerprint::ForeignType(type_id) => Some(from_foreign_def_id(type_id).module(db)),
        TyFingerprint::Dyn(trait_id) => Some(trait_id.module(db)),
        _ => None,
    };

    let def_blocks: ArrayVec<_, 2> =
        [trait_module.containing_block(), type_module.and_then(|it| it.containing_block())]
            .into_iter()
            .flatten()
            .filter_map(|block_id| db.trait_impls_in_block(block_id))
            .collect();

    let impls = impls
        .iter()
        .chain(&def_blocks)
        .flat_map(|impls| impls.for_trait_and_self_ty(hir_trait_id, self_ty_fp));

    let table = InferenceTable::new(db, env);

    let (impl_data, impl_subst) = find_matching_impl(impls, table, trait_ref)?;
    let item = impl_data.items.iter().find_map(|(n, it)| match *it {
        AssocItemId::FunctionId(f) => (n == name).then_some(AssocItemId::FunctionId(f)),
        AssocItemId::ConstId(c) => (n == name).then_some(AssocItemId::ConstId(c)),
        AssocItemId::TypeAliasId(_) => None,
    })?;
    Some((item, impl_subst))
}

fn find_matching_impl(
    mut impls: impl Iterator<Item = ImplId>,
    mut table: InferenceTable<'_>,
    actual_trait_ref: TraitRef,
) -> Option<(&ImplItems, Substitution)> {
    let db = table.db;
    impls.find_map(|impl_| {
        table.run_in_snapshot(|table| {
            let impl_substs =
                TyBuilder::subst_for_def(db, impl_, None).fill_with_inference_vars(table).build();
            let trait_ref = db
                .impl_trait(impl_)
                .expect("non-trait method in find_matching_impl")
                .substitute(Interner, &impl_substs);

            if !table.unify(&trait_ref, &actual_trait_ref) {
                return None;
            }

            let wcs = crate::chalk_db::convert_where_clauses(db, impl_.into(), &impl_substs)
                .into_iter()
                .map(|b| b.cast(Interner));
            let goal = crate::Goal::all(Interner, wcs);
            table.try_obligation(goal.clone())?;
            table.register_obligation(goal);
            Some((impl_.impl_items(db), table.resolve_completely(impl_substs)))
        })
    })
}

fn is_inherent_impl_coherent(
    db: &dyn HirDatabase,
    def_map: &DefMap,
    impl_id: ImplId,
    self_ty: &Ty,
) -> bool {
    let self_ty = self_ty.kind(Interner);
    let impl_allowed = match self_ty {
        TyKind::Tuple(_, _)
        | TyKind::FnDef(_, _)
        | TyKind::Array(_, _)
        | TyKind::Never
        | TyKind::Raw(_, _)
        | TyKind::Ref(_, _, _)
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Scalar(_) => def_map.is_rustc_coherence_is_core(),

        &TyKind::Adt(AdtId(adt), _) => adt.module(db).krate() == def_map.krate(),
        TyKind::Dyn(it) => it.principal_id().is_some_and(|trait_id| {
            from_chalk_trait_id(trait_id).module(db).krate() == def_map.krate()
        }),

        _ => true,
    };
    impl_allowed || {
        let rustc_has_incoherent_inherent_impls = match self_ty {
            TyKind::Tuple(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Array(_, _)
            | TyKind::Never
            | TyKind::Raw(_, _)
            | TyKind::Ref(_, _, _)
            | TyKind::Slice(_)
            | TyKind::Str
            | TyKind::Scalar(_) => true,

            &TyKind::Adt(AdtId(adt), _) => match adt {
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
            TyKind::Dyn(it) => it.principal_id().is_some_and(|trait_id| {
                db.trait_signature(from_chalk_trait_id(trait_id))
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
pub fn check_orphan_rules(db: &dyn HirDatabase, impl_: ImplId) -> bool {
    let substs = TyBuilder::placeholder_subst(db, impl_);
    let Some(impl_trait) = db.impl_trait(impl_) else {
        // not a trait impl
        return true;
    };

    let local_crate = impl_.lookup(db).container.krate();
    let is_local = |tgt_crate| tgt_crate == local_crate;

    let trait_ref = impl_trait.substitute(Interner, &substs);
    let trait_id = from_chalk_trait_id(trait_ref.trait_id);
    if is_local(trait_id.module(db).krate()) {
        // trait to be implemented is local
        return true;
    }

    let unwrap_fundamental = |mut ty: Ty| {
        // Unwrap all layers of fundamental types with a loop.
        loop {
            match ty.kind(Interner) {
                TyKind::Ref(_, _, referenced) => ty = referenced.clone(),
                &TyKind::Adt(AdtId(hir_def::AdtId::StructId(s)), ref subs) => {
                    let struct_signature = db.struct_signature(s);
                    if struct_signature.flags.contains(StructFlags::FUNDAMENTAL) {
                        let next = subs.type_parameters(Interner).next();
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
    let is_not_orphan = trait_ref.substitution.type_parameters(Interner).any(|ty| {
        match unwrap_fundamental(ty).kind(Interner) {
            &TyKind::Adt(AdtId(id), _) => is_local(id.module(db).krate()),
            TyKind::Error => true,
            TyKind::Dyn(it) => it
                .principal_id()
                .is_some_and(|trait_id| is_local(from_chalk_trait_id(trait_id).module(db).krate())),
            _ => false,
        }
    });
    #[allow(clippy::let_and_return)]
    is_not_orphan
}

pub fn iterate_path_candidates(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
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

pub fn iterate_method_candidates_dyn(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
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

            let mut table = InferenceTable::new(db, env);
            let ty = table.instantiate_canonical(ty.clone());
            let deref_chain = autoderef_method_receiver(&mut table, ty);

            deref_chain.into_iter().try_for_each(|(receiver_ty, adj)| {
                iterate_method_candidates_with_autoref(
                    &mut table,
                    receiver_ty,
                    adj,
                    traits_in_scope,
                    visible_from_module,
                    name,
                    callback,
                )
            })
        }
        LookupMode::Path => {
            // No autoderef for path lookups
            iterate_method_candidates_for_self_ty(
                ty,
                db,
                env,
                traits_in_scope,
                visible_from_module,
                name,
                callback,
            )
        }
    }
}

#[tracing::instrument(skip_all, fields(name = ?name))]
fn iterate_method_candidates_with_autoref(
    table: &mut InferenceTable<'_>,
    receiver_ty: Canonical<Ty>,
    first_adjustment: ReceiverAdjustments,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    if receiver_ty.value.is_general_var(Interner, &receiver_ty.binders) {
        // don't try to resolve methods on unknown types
        return ControlFlow::Continue(());
    }

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
    if let Some((_, _, m)) = receiver_ty.value.as_reference() {
        // Prefer reborrow of references to move
        maybe_reborrowed.autoref = Some(AutorefOrPtrAdjustment::Autoref(m));
        maybe_reborrowed.autoderefs += 1;
    }

    iterate_method_candidates_by_receiver(receiver_ty.clone(), maybe_reborrowed)?;

    let refed = Canonical {
        value: TyKind::Ref(Mutability::Not, error_lifetime(), receiver_ty.value.clone())
            .intern(Interner),
        binders: receiver_ty.binders.clone(),
    };

    iterate_method_candidates_by_receiver(
        refed,
        first_adjustment.with_autoref(AutorefOrPtrAdjustment::Autoref(Mutability::Not)),
    )?;

    let ref_muted = Canonical {
        value: TyKind::Ref(Mutability::Mut, error_lifetime(), receiver_ty.value.clone())
            .intern(Interner),
        binders: receiver_ty.binders.clone(),
    };

    iterate_method_candidates_by_receiver(
        ref_muted,
        first_adjustment.with_autoref(AutorefOrPtrAdjustment::Autoref(Mutability::Mut)),
    )?;

    if let Some((ty, Mutability::Mut)) = receiver_ty.value.as_raw_ptr() {
        let const_ptr_ty = Canonical {
            value: TyKind::Raw(Mutability::Not, ty.clone()).intern(Interner),
            binders: receiver_ty.binders,
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
fn iterate_method_candidates_by_receiver(
    table: &mut InferenceTable<'_>,
    receiver_ty: Canonical<Ty>,
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
        let mut autoderef =
            autoderef::Autoderef::new_no_tracking(table, receiver_ty.clone(), true, true);
        while let Some((self_ty, _)) = autoderef.next() {
            iterate_inherent_methods(
                &self_ty,
                autoderef.table,
                name,
                Some(&receiver_ty),
                Some(receiver_adjustments.clone()),
                visible_from_module,
                &mut |adjustments, item, is_visible| {
                    callback.on_inherent_method(adjustments, item, is_visible)
                },
            )?
        }
        ControlFlow::Continue(())
    })?;
    table.run_in_snapshot(|table| {
        let mut autoderef =
            autoderef::Autoderef::new_no_tracking(table, receiver_ty.clone(), true, true);
        while let Some((self_ty, _)) = autoderef.next() {
            if matches!(self_ty.kind(Interner), TyKind::InferenceVar(_, TyVariableKind::General)) {
                // don't try to resolve methods on unknown types
                return ControlFlow::Continue(());
            }

            iterate_trait_method_candidates(
                &self_ty,
                autoderef.table,
                traits_in_scope,
                name,
                Some(&receiver_ty),
                Some(receiver_adjustments.clone()),
                &mut |adjustments, item, is_visible| {
                    callback.on_trait_method(adjustments, item, is_visible)
                },
            )?
        }
        ControlFlow::Continue(())
    })
}

#[tracing::instrument(skip_all, fields(name = ?name))]
fn iterate_method_candidates_for_self_ty(
    self_ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: VisibleFromModule,
    name: Option<&Name>,
    callback: &mut dyn MethodCandidateCallback,
) -> ControlFlow<()> {
    let mut table = InferenceTable::new(db, env);
    let self_ty = table.instantiate_canonical(self_ty.clone());
    iterate_inherent_methods(
        &self_ty,
        &mut table,
        name,
        None,
        None,
        visible_from_module,
        &mut |adjustments, item, is_visible| {
            callback.on_inherent_method(adjustments, item, is_visible)
        },
    )?;
    iterate_trait_method_candidates(
        &self_ty,
        &mut table,
        traits_in_scope,
        name,
        None,
        None,
        &mut |adjustments, item, is_visible| {
            callback.on_trait_method(adjustments, item, is_visible)
        },
    )
}

#[tracing::instrument(skip_all, fields(name = ?name, visible_from_module, receiver_ty))]
fn iterate_trait_method_candidates(
    self_ty: &Ty,
    table: &mut InferenceTable<'_>,
    traits_in_scope: &FxHashSet<TraitId>,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    receiver_adjustments: Option<ReceiverAdjustments>,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let db = table.db;

    let canonical_self_ty = table.canonicalize(self_ty.clone());
    let TraitEnvironment { krate, block, .. } = *table.trait_env;

    'traits: for &t in traits_in_scope {
        let data = db.trait_signature(t);

        // Traits annotated with `#[rustc_skip_during_method_dispatch]` are skipped during
        // method resolution, if the receiver is an array, and we're compiling for editions before
        // 2021.
        // This is to make `[a].into_iter()` not break code with the new `IntoIterator` impl for
        // arrays.
        if data.flags.contains(TraitFlags::SKIP_ARRAY_DURING_METHOD_DISPATCH)
            && matches!(self_ty.kind(Interner), TyKind::Array(..))
        {
            // FIXME: this should really be using the edition of the method name's span, in case it
            // comes from a macro
            if !krate.data(db).edition.at_least_2021() {
                continue;
            }
        }
        if data.flags.contains(TraitFlags::SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH)
            && matches!(
                self_ty.kind(Interner), TyKind::Adt(AdtId(def), subst)
                if is_box(table.db, *def)
                    && matches!(subst.at(Interner, 0).assert_ty_ref(Interner).kind(Interner), TyKind::Slice(..))
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
            let visible =
                match is_valid_trait_method_candidate(table, t, name, receiver_ty, item, self_ty) {
                    IsValidCandidate::Yes => true,
                    IsValidCandidate::NotVisible => false,
                    IsValidCandidate::No => continue,
                };
            if !known_implemented {
                let goal = generic_implements_goal(db, &table.trait_env, t, &canonical_self_ty);
                if db.trait_solve(krate, block, goal.cast(Interner)).is_none() {
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
fn iterate_inherent_methods(
    self_ty: &Ty,
    table: &mut InferenceTable<'_>,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    receiver_adjustments: Option<ReceiverAdjustments>,
    visible_from_module: VisibleFromModule,
    callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let db = table.db;
    let env = table.trait_env.clone();

    // For trait object types and placeholder types with trait bounds, the methods of the trait and
    // its super traits are considered inherent methods. This matters because these methods have
    // higher priority than the other traits' methods, which would be considered in
    // `iterate_trait_method_candidates()` only after this function.
    match self_ty.kind(Interner) {
        TyKind::Placeholder(_) => {
            let env = table.trait_env.clone();
            let traits = env
                .traits_in_scope_from_clauses(self_ty.clone())
                .flat_map(|t| all_super_traits(db, t));
            iterate_inherent_trait_methods(
                self_ty,
                table,
                name,
                receiver_ty,
                receiver_adjustments.clone(),
                callback,
                traits,
            )?;
        }
        TyKind::Dyn(_) => {
            if let Some(principal_trait) = self_ty.dyn_trait() {
                let traits = all_super_traits(db, principal_trait);
                iterate_inherent_trait_methods(
                    self_ty,
                    table,
                    name,
                    receiver_ty,
                    receiver_adjustments.clone(),
                    callback,
                    traits.into_iter(),
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
    fn iterate_inherent_trait_methods(
        self_ty: &Ty,
        table: &mut InferenceTable<'_>,
        name: Option<&Name>,
        receiver_ty: Option<&Ty>,
        receiver_adjustments: Option<ReceiverAdjustments>,
        callback: &mut dyn FnMut(ReceiverAdjustments, AssocItemId, bool) -> ControlFlow<()>,
        traits: impl Iterator<Item = TraitId>,
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
    fn impls_for_self_ty(
        impls: &InherentImpls,
        self_ty: &Ty,
        table: &mut InferenceTable<'_>,
        name: Option<&Name>,
        receiver_ty: Option<&Ty>,
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
pub(crate) fn resolve_indexing_op(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    ty: Canonical<Ty>,
    index_trait: TraitId,
) -> Option<ReceiverAdjustments> {
    let mut table = InferenceTable::new(db, env);
    let ty = table.instantiate_canonical(ty);
    let deref_chain = autoderef_method_receiver(&mut table, ty);
    for (ty, adj) in deref_chain {
        let goal = generic_implements_goal(db, &table.trait_env, index_trait, &ty);
        if db
            .trait_solve(table.trait_env.krate, table.trait_env.block, goal.cast(Interner))
            .is_some()
        {
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
fn is_valid_impl_method_candidate(
    table: &mut InferenceTable<'_>,
    self_ty: &Ty,
    receiver_ty: Option<&Ty>,
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

            if let Some(from_module) = visible_from_module {
                if !db.assoc_visibility(c.into()).is_visible_from(db, from_module) {
                    cov_mark::hit!(const_candidate_not_visible);
                    return IsValidCandidate::NotVisible;
                }
            }
            let self_ty_matches = table.run_in_snapshot(|table| {
                let expected_self_ty =
                    TyBuilder::impl_self_ty(db, impl_id).fill_with_inference_vars(table).build();
                table.unify(&expected_self_ty, self_ty)
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
fn is_valid_trait_method_candidate(
    table: &mut InferenceTable<'_>,
    trait_id: TraitId,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    item: AssocItemId,
    self_ty: &Ty,
) -> IsValidCandidate {
    let db = table.db;
    match item {
        AssocItemId::FunctionId(fn_id) => {
            let data = db.function_signature(fn_id);

            check_that!(name.is_none_or(|n| n == &data.name));

            table.run_in_snapshot(|table| {
                let impl_subst = TyBuilder::subst_for_def(db, trait_id, None)
                    .fill_with_inference_vars(table)
                    .build();
                let expect_self_ty = impl_subst.at(Interner, 0).assert_ty_ref(Interner).clone();

                check_that!(table.unify(&expect_self_ty, self_ty));

                if let Some(receiver_ty) = receiver_ty {
                    check_that!(data.has_self_param());

                    let fn_subst = TyBuilder::subst_for_def(db, fn_id, Some(impl_subst))
                        .fill_with_inference_vars(table)
                        .build();

                    let sig = db.callable_item_signature(fn_id.into());
                    let expected_receiver =
                        sig.map(|s| s.params()[0].clone()).substitute(Interner, &fn_subst);

                    check_that!(table.unify(receiver_ty, &expected_receiver));
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
fn is_valid_impl_fn_candidate(
    table: &mut InferenceTable<'_>,
    impl_id: ImplId,
    fn_id: FunctionId,
    name: Option<&Name>,
    receiver_ty: Option<&Ty>,
    self_ty: &Ty,
    visible_from_module: Option<ModuleId>,
    item_name: &Name,
) -> IsValidCandidate {
    check_that!(name.is_none_or(|n| n == item_name));

    let db = table.db;
    let data = db.function_signature(fn_id);

    if let Some(from_module) = visible_from_module {
        if !db.assoc_visibility(fn_id.into()).is_visible_from(db, from_module) {
            cov_mark::hit!(autoderef_candidate_not_visible);
            return IsValidCandidate::NotVisible;
        }
    }
    table.run_in_snapshot(|table| {
        let _p = tracing::info_span!("subst_for_def").entered();
        let impl_subst =
            TyBuilder::subst_for_def(db, impl_id, None).fill_with_inference_vars(table).build();
        let expect_self_ty = db.impl_self_ty(impl_id).substitute(Interner, &impl_subst);

        check_that!(table.unify(&expect_self_ty, self_ty));

        if let Some(receiver_ty) = receiver_ty {
            let _p = tracing::info_span!("check_receiver_ty").entered();
            check_that!(data.has_self_param());

            let fn_subst = TyBuilder::subst_for_def(db, fn_id, Some(impl_subst.clone()))
                .fill_with_inference_vars(table)
                .build();

            let sig = db.callable_item_signature(fn_id.into());
            let expected_receiver =
                sig.map(|s| s.params()[0].clone()).substitute(Interner, &fn_subst);

            check_that!(table.unify(receiver_ty, &expected_receiver));
        }

        // We need to consider the bounds on the impl to distinguish functions of the same name
        // for a type.
        let predicates = db.generic_predicates(impl_id.into());
        let goals = predicates.iter().map(|p| {
            let (p, b) = p
                .clone()
                .substitute(Interner, &impl_subst)
                // Skipping the inner binders is ok, as we don't handle quantified where
                // clauses yet.
                .into_value_and_skipped_binders();
            stdx::always!(b.len(Interner) == 0);

            p.cast::<Goal>(Interner)
        });

        for goal in goals.clone() {
            let in_env = InEnvironment::new(&table.trait_env.env, goal);
            let canonicalized = table.canonicalize_with_free_vars(in_env);
            let solution = table.db.trait_solve(
                table.trait_env.krate,
                table.trait_env.block,
                canonicalized.value.clone(),
            );

            match solution {
                Some(Solution::Unique(canonical_subst)) => {
                    canonicalized.apply_solution(
                        table,
                        Canonical {
                            binders: canonical_subst.binders,
                            value: canonical_subst.value.subst,
                        },
                    );
                }
                Some(Solution::Ambig(Guidance::Definite(substs))) => {
                    canonicalized.apply_solution(table, substs);
                }
                Some(_) => (),
                None => return IsValidCandidate::No,
            }
        }

        for goal in goals {
            if table.try_obligation(goal).is_none() {
                return IsValidCandidate::No;
            }
        }

        IsValidCandidate::Yes
    })
}

pub fn implements_trait(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: &TraitEnvironment,
    trait_: TraitId,
) -> bool {
    let goal = generic_implements_goal(db, env, trait_, ty);
    let solution = db.trait_solve(env.krate, env.block, goal.cast(Interner));

    solution.is_some()
}

pub fn implements_trait_unique(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: &TraitEnvironment,
    trait_: TraitId,
) -> bool {
    let goal = generic_implements_goal(db, env, trait_, ty);
    let solution = db.trait_solve(env.krate, env.block, goal.cast(Interner));

    matches!(solution, Some(crate::Solution::Unique(_)))
}

/// This creates Substs for a trait with the given Self type and type variables
/// for all other parameters, to query Chalk with it.
#[tracing::instrument(skip_all)]
fn generic_implements_goal(
    db: &dyn HirDatabase,
    env: &TraitEnvironment,
    trait_: TraitId,
    self_ty: &Canonical<Ty>,
) -> Canonical<InEnvironment<super::DomainGoal>> {
    let binders = self_ty.binders.interned();
    let trait_ref = TyBuilder::trait_ref(db, trait_)
        .push(self_ty.value.clone())
        .fill_with_bound_vars(DebruijnIndex::INNERMOST, binders.len())
        .build();

    let kinds =
        binders.iter().cloned().chain(trait_ref.substitution.iter(Interner).skip(1).map(|it| {
            let vk = match it.data(Interner) {
                GenericArgData::Ty(_) => VariableKind::Ty(chalk_ir::TyVariableKind::General),
                GenericArgData::Lifetime(_) => VariableKind::Lifetime,
                GenericArgData::Const(c) => VariableKind::Const(c.data(Interner).ty.clone()),
            };
            WithKind::new(vk, UniverseIndex::ROOT)
        }));
    let binders = CanonicalVarKinds::from_iter(Interner, kinds);

    let obligation = trait_ref.cast(Interner);
    let value = InEnvironment::new(&env.env, obligation);
    Canonical { binders, value }
}

fn autoderef_method_receiver(
    table: &mut InferenceTable<'_>,
    ty: Ty,
) -> Vec<(Canonical<Ty>, ReceiverAdjustments)> {
    let mut deref_chain: Vec<_> = Vec::new();
    let mut autoderef = autoderef::Autoderef::new_no_tracking(table, ty, false, true);
    while let Some((ty, derefs)) = autoderef.next() {
        deref_chain.push((
            autoderef.table.canonicalize(ty),
            ReceiverAdjustments { autoref: None, autoderefs: derefs, unsize_array: false },
        ));
    }
    // As a last step, we can do array unsizing (that's the only unsizing that rustc does for method receivers!)
    if let Some((TyKind::Array(parameters, _), binders, adj)) =
        deref_chain.last().map(|(ty, adj)| (ty.value.kind(Interner), ty.binders.clone(), adj))
    {
        let unsized_ty = TyKind::Slice(parameters.clone()).intern(Interner);
        deref_chain.push((
            Canonical { value: unsized_ty, binders },
            ReceiverAdjustments { unsize_array: true, ..adj.clone() },
        ));
    }
    deref_chain
}
