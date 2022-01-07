//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in librustc_typeck/check/method/probe.rs.
use std::{iter, ops::ControlFlow, sync::Arc};

use arrayvec::ArrayVec;
use base_db::{CrateId, Edition};
use chalk_ir::{cast::Cast, Mutability, UniverseIndex};
use hir_def::{
    item_scope::ItemScope, lang_item::LangItemTarget, nameres::DefMap, AssocItemId, BlockId,
    ConstId, FunctionId, GenericDefId, HasModule, ImplId, ItemContainerId, Lookup, ModuleDefId,
    ModuleId, TraitId,
};
use hir_expand::name::Name;
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::never;

use crate::{
    autoderef,
    consteval::{self, ConstExt},
    db::HirDatabase,
    from_foreign_def_id,
    primitive::{self, FloatTy, IntTy, UintTy},
    static_lifetime,
    utils::all_super_traits,
    AdtId, Canonical, CanonicalVarKinds, DebruijnIndex, ForeignDefId, InEnvironment, Interner,
    Scalar, Substitution, TraitEnvironment, TraitRefExt, Ty, TyBuilder, TyExt, TyKind,
};

/// This is used as a key for indexing impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
                let first_ty = subst.interned().get(0).map(|arg| arg.assert_ty_ref(Interner));
                match first_ty {
                    Some(ty) => return TyFingerprint::for_trait_impl(ty),
                    None => TyFingerprint::Unit,
                }
            }
            TyKind::AssociatedType(_, _)
            | TyKind::OpaqueType(_, _)
            | TyKind::FnDef(_, _)
            | TyKind::Closure(_, _)
            | TyKind::Generator(..)
            | TyKind::GeneratorWitness(..) => TyFingerprint::Unnameable,
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

pub(crate) const ALL_FLOAT_FPS: [TyFingerprint; 2] = [
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F32)),
    TyFingerprint::Scalar(Scalar::Float(FloatTy::F64)),
];

/// Trait impls defined or available in some crate.
#[derive(Debug, Eq, PartialEq)]
pub struct TraitImpls {
    // If the `Option<TyFingerprint>` is `None`, the impl may apply to any self type.
    map: FxHashMap<TraitId, FxHashMap<Option<TyFingerprint>, Vec<ImplId>>>,
}

impl TraitImpls {
    pub(crate) fn trait_impls_in_crate_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("trait_impls_in_crate_query");
        let mut impls = Self { map: FxHashMap::default() };

        let crate_def_map = db.crate_def_map(krate);
        impls.collect_def_map(db, &crate_def_map);
        impls.shrink_to_fit();

        Arc::new(impls)
    }

    pub(crate) fn trait_impls_in_block_query(
        db: &dyn HirDatabase,
        block: BlockId,
    ) -> Option<Arc<Self>> {
        let _p = profile::span("trait_impls_in_block_query");
        let mut impls = Self { map: FxHashMap::default() };

        let block_def_map = db.block_def_map(block)?;
        impls.collect_def_map(db, &block_def_map);
        impls.shrink_to_fit();

        Some(Arc::new(impls))
    }

    pub(crate) fn trait_impls_in_deps_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("trait_impls_in_deps_query");
        let crate_graph = db.crate_graph();
        let mut res = Self { map: FxHashMap::default() };

        for krate in crate_graph.transitive_deps(krate) {
            res.merge(&db.trait_impls_in_crate(krate));
        }
        res.shrink_to_fit();

        Arc::new(res)
    }

    fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
        self.map.values_mut().for_each(|map| {
            map.shrink_to_fit();
            map.values_mut().for_each(Vec::shrink_to_fit);
        });
    }

    fn collect_def_map(&mut self, db: &dyn HirDatabase, def_map: &DefMap) {
        for (_module_id, module_data) in def_map.modules() {
            for impl_id in module_data.scope.impls() {
                let target_trait = match db.impl_trait(impl_id) {
                    Some(tr) => tr.skip_binders().hir_trait_id(),
                    None => continue,
                };
                let self_ty = db.impl_self_ty(impl_id);
                let self_ty_fp = TyFingerprint::for_trait_impl(self_ty.skip_binders());
                self.map
                    .entry(target_trait)
                    .or_default()
                    .entry(self_ty_fp)
                    .or_default()
                    .push(impl_id);
            }

            // To better support custom derives, collect impls in all unnamed const items.
            // const _: () = { ... };
            for konst in collect_unnamed_consts(db, &module_data.scope) {
                let body = db.body(konst.into());
                for (_, block_def_map) in body.blocks(db.upcast()) {
                    self.collect_def_map(db, &block_def_map);
                }
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        for (trait_, other_map) in &other.map {
            let map = self.map.entry(*trait_).or_default();
            for (fp, impls) in other_map {
                let vec = map.entry(*fp).or_default();
                vec.extend(impls);
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
            .flat_map(move |map| map.get(&None).into_iter().chain(map.get(&Some(self_ty))))
            .flat_map(|v| v.iter().copied())
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
}

impl InherentImpls {
    pub(crate) fn inherent_impls_in_crate_query(db: &dyn HirDatabase, krate: CrateId) -> Arc<Self> {
        let mut impls = Self { map: FxHashMap::default() };

        let crate_def_map = db.crate_def_map(krate);
        impls.collect_def_map(db, &crate_def_map);
        impls.shrink_to_fit();

        return Arc::new(impls);
    }

    pub(crate) fn inherent_impls_in_block_query(
        db: &dyn HirDatabase,
        block: BlockId,
    ) -> Option<Arc<Self>> {
        let mut impls = Self { map: FxHashMap::default() };
        if let Some(block_def_map) = db.block_def_map(block) {
            impls.collect_def_map(db, &block_def_map);
            impls.shrink_to_fit();
            return Some(Arc::new(impls));
        }
        return None;
    }

    fn shrink_to_fit(&mut self) {
        self.map.values_mut().for_each(Vec::shrink_to_fit);
        self.map.shrink_to_fit();
    }

    fn collect_def_map(&mut self, db: &dyn HirDatabase, def_map: &DefMap) {
        for (_module_id, module_data) in def_map.modules() {
            for impl_id in module_data.scope.impls() {
                let data = db.impl_data(impl_id);
                if data.target_trait.is_some() {
                    continue;
                }

                let self_ty = db.impl_self_ty(impl_id);
                let fp = TyFingerprint::for_inherent_impl(self_ty.skip_binders());
                if let Some(fp) = fp {
                    self.map.entry(fp).or_default().push(impl_id);
                }
                // `fp` should only be `None` in error cases (either erroneous code or incomplete name resolution)
            }

            // To better support custom derives, collect impls in all unnamed const items.
            // const _: () = { ... };
            for konst in collect_unnamed_consts(db, &module_data.scope) {
                let body = db.body(konst.into());
                for (_, block_def_map) in body.blocks(db.upcast()) {
                    self.collect_def_map(db, &block_def_map);
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
}

fn collect_unnamed_consts<'a>(
    db: &'a dyn HirDatabase,
    scope: &'a ItemScope,
) -> impl Iterator<Item = ConstId> + 'a {
    let unnamed_consts = scope.unnamed_consts();

    // FIXME: Also treat consts named `_DERIVE_*` as unnamed, since synstructure generates those.
    // Should be removed once synstructure stops doing that.
    let synstructure_hack_consts = scope.values().filter_map(|(item, _)| match item {
        ModuleDefId::ConstId(id) => {
            let loc = id.lookup(db.upcast());
            let item_tree = loc.id.item_tree(db.upcast());
            if item_tree[loc.id.value]
                .name
                .as_ref()
                .map_or(false, |n| n.to_smol_str().starts_with("_DERIVE_"))
            {
                Some(id)
            } else {
                None
            }
        }
        _ => None,
    });

    unnamed_consts.chain(synstructure_hack_consts)
}

pub fn def_crates(
    db: &dyn HirDatabase,
    ty: &Ty,
    cur_crate: CrateId,
) -> Option<ArrayVec<CrateId, 2>> {
    // Types like slice can have inherent impls in several crates, (core and alloc).
    // The corresponding impls are marked with lang items, so we can use them to find the required crates.
    macro_rules! lang_item_crate {
            ($($name:expr),+ $(,)?) => {{
                let mut v = ArrayVec::<LangItemTarget, 2>::new();
                $(
                    v.extend(db.lang_item(cur_crate, $name.into()));
                )+
                v
            }};
        }

    let mod_to_crate_ids = |module: ModuleId| Some(iter::once(module.krate()).collect());

    let lang_item_targets = match ty.kind(Interner) {
        TyKind::Adt(AdtId(def_id), _) => {
            return mod_to_crate_ids(def_id.module(db.upcast()));
        }
        TyKind::Foreign(id) => {
            return mod_to_crate_ids(
                from_foreign_def_id(*id).lookup(db.upcast()).module(db.upcast()),
            );
        }
        TyKind::Scalar(Scalar::Bool) => lang_item_crate!("bool"),
        TyKind::Scalar(Scalar::Char) => lang_item_crate!("char"),
        TyKind::Scalar(Scalar::Float(f)) => match f {
            // There are two lang items: one in libcore (fXX) and one in libstd (fXX_runtime)
            FloatTy::F32 => lang_item_crate!("f32", "f32_runtime"),
            FloatTy::F64 => lang_item_crate!("f64", "f64_runtime"),
        },
        &TyKind::Scalar(Scalar::Int(t)) => {
            lang_item_crate!(primitive::int_ty_to_string(t))
        }
        &TyKind::Scalar(Scalar::Uint(t)) => {
            lang_item_crate!(primitive::uint_ty_to_string(t))
        }
        TyKind::Str => lang_item_crate!("str_alloc", "str"),
        TyKind::Slice(_) => lang_item_crate!("slice_alloc", "slice"),
        TyKind::Array(..) => lang_item_crate!("array"),
        TyKind::Raw(Mutability::Not, _) => lang_item_crate!("const_ptr"),
        TyKind::Raw(Mutability::Mut, _) => lang_item_crate!("mut_ptr"),
        TyKind::Dyn(_) => {
            return ty.dyn_trait().and_then(|trait_| {
                mod_to_crate_ids(GenericDefId::TraitId(trait_).module(db.upcast()))
            });
        }
        _ => return None,
    };
    let res = lang_item_targets
        .into_iter()
        .filter_map(|it| match it {
            LangItemTarget::ImplDefId(it) => Some(it),
            _ => None,
        })
        .map(|it| it.lookup(db.upcast()).container.krate())
        .collect();
    Some(res)
}

/// Look up the method with the given name, returning the actual autoderefed
/// receiver type (but without autoref applied yet).
pub(crate) fn lookup_method(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: Option<ModuleId>,
    name: &Name,
) -> Option<(Canonical<Ty>, FunctionId)> {
    iterate_method_candidates(
        ty,
        db,
        env,
        krate,
        traits_in_scope,
        visible_from_module,
        Some(name),
        LookupMode::MethodCall,
        |ty, f| match f {
            AssocItemId::FunctionId(f) => Some((ty.clone(), f)),
            _ => None,
        },
    )
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

// This would be nicer if it just returned an iterator, but that runs into
// lifetime problems, because we need to borrow temp `CrateImplDefs`.
// FIXME add a context type here?
pub fn iterate_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: Option<ModuleId>,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(&Canonical<Ty>, AssocItemId) -> Option<T>,
) -> Option<T> {
    let mut slot = None;
    iterate_method_candidates_dyn(
        ty,
        db,
        env,
        krate,
        traits_in_scope,
        visible_from_module,
        name,
        mode,
        &mut |ty, item| {
            assert!(slot.is_none());
            if let Some(it) = callback(ty, item) {
                slot = Some(it);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        },
    );
    slot
}

pub fn iterate_method_candidates_dyn(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: Option<ModuleId>,
    name: Option<&Name>,
    mode: LookupMode,
    callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    match mode {
        LookupMode::MethodCall => {
            // For method calls, rust first does any number of autoderef, and then one
            // autoref (i.e. when the method takes &self or &mut self). We just ignore
            // the autoref currently -- when we find a method matching the given name,
            // we assume it fits.

            // Also note that when we've got a receiver like &S, even if the method we
            // find in the end takes &self, we still do the autoderef step (just as
            // rustc does an autoderef and then autoref again).
            let ty = InEnvironment { goal: ty.clone(), environment: env.env.clone() };

            // We have to be careful about the order we're looking at candidates
            // in here. Consider the case where we're resolving `x.clone()`
            // where `x: &Vec<_>`. This resolves to the clone method with self
            // type `Vec<_>`, *not* `&_`. I.e. we need to consider methods where
            // the receiver type exactly matches before cases where we have to
            // do autoref. But in the autoderef steps, the `&_` self type comes
            // up *before* the `Vec<_>` self type.
            //
            // On the other hand, we don't want to just pick any by-value method
            // before any by-autoref method; it's just that we need to consider
            // the methods by autoderef order of *receiver types*, not *self
            // types*.

            let deref_chain = autoderef_method_receiver(db, krate, ty);
            let mut deref_chains = stdx::slice_tails(&deref_chain);

            deref_chains.try_for_each(|deref_chain| {
                iterate_method_candidates_with_autoref(
                    deref_chain,
                    db,
                    env.clone(),
                    krate,
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
                krate,
                traits_in_scope,
                visible_from_module,
                name,
                callback,
            )
        }
    }
}

fn iterate_method_candidates_with_autoref(
    deref_chain: &[Canonical<Ty>],
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: Option<ModuleId>,
    name: Option<&Name>,
    mut callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let (receiver_ty, rest) = match deref_chain.split_first() {
        Some((rec, rest)) => (rec.clone(), rest),
        None => {
            never!("received empty deref-chain");
            return ControlFlow::Break(());
        }
    };
    iterate_method_candidates_by_receiver(
        &receiver_ty,
        &rest,
        db,
        env.clone(),
        krate,
        traits_in_scope,
        visible_from_module,
        name,
        &mut callback,
    )?;

    let refed = Canonical {
        binders: receiver_ty.binders.clone(),
        value: TyKind::Ref(Mutability::Not, static_lifetime(), receiver_ty.value.clone())
            .intern(Interner),
    };

    iterate_method_candidates_by_receiver(
        &refed,
        deref_chain,
        db,
        env.clone(),
        krate,
        traits_in_scope,
        visible_from_module,
        name,
        &mut callback,
    )?;

    let ref_muted = Canonical {
        binders: receiver_ty.binders,
        value: TyKind::Ref(Mutability::Mut, static_lifetime(), receiver_ty.value).intern(Interner),
    };

    iterate_method_candidates_by_receiver(
        &ref_muted,
        deref_chain,
        db,
        env,
        krate,
        traits_in_scope,
        visible_from_module,
        name,
        &mut callback,
    )
}

fn iterate_method_candidates_by_receiver(
    receiver_ty: &Canonical<Ty>,
    rest_of_deref_chain: &[Canonical<Ty>],
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: Option<ModuleId>,
    name: Option<&Name>,
    mut callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    // We're looking for methods with *receiver* type receiver_ty. These could
    // be found in any of the derefs of receiver_ty, so we have to go through
    // that.
    for self_ty in iter::once(receiver_ty).chain(rest_of_deref_chain) {
        iterate_inherent_methods(
            self_ty,
            db,
            env.clone(),
            name,
            Some(receiver_ty),
            krate,
            visible_from_module,
            &mut callback,
        )?
    }

    for self_ty in iter::once(receiver_ty).chain(rest_of_deref_chain) {
        iterate_trait_method_candidates(
            self_ty,
            db,
            env.clone(),
            krate,
            traits_in_scope,
            name,
            Some(receiver_ty),
            &mut callback,
        )?
    }

    ControlFlow::Continue(())
}

fn iterate_method_candidates_for_self_ty(
    self_ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    visible_from_module: Option<ModuleId>,
    name: Option<&Name>,
    mut callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    iterate_inherent_methods(
        self_ty,
        db,
        env.clone(),
        name,
        None,
        krate,
        visible_from_module,
        &mut callback,
    )?;
    iterate_trait_method_candidates(self_ty, db, env, krate, traits_in_scope, name, None, callback)
}

fn iterate_trait_method_candidates(
    self_ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    traits_in_scope: &FxHashSet<TraitId>,
    name: Option<&Name>,
    receiver_ty: Option<&Canonical<Ty>>,
    callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let receiver_is_array = matches!(self_ty.value.kind(Interner), chalk_ir::TyKind::Array(..));
    // if ty is `dyn Trait`, the trait doesn't need to be in scope
    let inherent_trait =
        self_ty.value.dyn_trait().into_iter().flat_map(|t| all_super_traits(db.upcast(), t));
    let env_traits = matches!(self_ty.value.kind(Interner), TyKind::Placeholder(_))
        // if we have `T: Trait` in the param env, the trait doesn't need to be in scope
        .then(|| {
            env.traits_in_scope_from_clauses(self_ty.value.clone())
                .flat_map(|t| all_super_traits(db.upcast(), t))
        })
        .into_iter()
        .flatten();
    let traits = inherent_trait.chain(env_traits).chain(traits_in_scope.iter().copied());

    'traits: for t in traits {
        let data = db.trait_data(t);

        // Traits annotated with `#[rustc_skip_array_during_method_dispatch]` are skipped during
        // method resolution, if the receiver is an array, and we're compiling for editions before
        // 2021.
        // This is to make `[a].into_iter()` not break code with the new `IntoIterator` impl for
        // arrays.
        if data.skip_array_during_method_dispatch && receiver_is_array {
            // FIXME: this should really be using the edition of the method name's span, in case it
            // comes from a macro
            if db.crate_graph()[krate].edition < Edition::Edition2021 {
                continue;
            }
        }

        // we'll be lazy about checking whether the type implements the
        // trait, but if we find out it doesn't, we'll skip the rest of the
        // iteration
        let mut known_implemented = false;
        for &(_, item) in data.items.iter() {
            // Don't pass a `visible_from_module` down to `is_valid_candidate`,
            // since only inherent methods should be included into visibility checking.
            if !is_valid_candidate(db, env.clone(), name, receiver_ty, item, self_ty, None) {
                continue;
            }
            if !known_implemented {
                let goal = generic_implements_goal(db, env.clone(), t, self_ty.clone());
                if db.trait_solve(krate, goal.cast(Interner)).is_none() {
                    continue 'traits;
                }
            }
            known_implemented = true;
            // FIXME: we shouldn't be ignoring the binders here
            callback(self_ty, item)?
        }
    }
    ControlFlow::Continue(())
}

fn filter_inherent_impls_for_self_ty<'i>(
    impls: &'i InherentImpls,
    self_ty: &Ty,
) -> impl Iterator<Item = &'i ImplId> {
    // inherent methods on arrays are fingerprinted as [T; {unknown}], so we must also consider them when
    // resolving a method call on an array with a known len
    let array_impls = {
        match self_ty.kind(Interner) {
            TyKind::Array(parameters, array_len) if !array_len.is_unknown() => {
                let unknown_array_len_ty =
                    TyKind::Array(parameters.clone(), consteval::usize_const(None));

                Some(impls.for_self_ty(&unknown_array_len_ty.intern(Interner)))
            }
            _ => None,
        }
    }
    .into_iter()
    .flatten();

    impls.for_self_ty(self_ty).iter().chain(array_impls)
}

fn iterate_inherent_methods(
    self_ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    name: Option<&Name>,
    receiver_ty: Option<&Canonical<Ty>>,
    krate: CrateId,
    visible_from_module: Option<ModuleId>,
    callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let def_crates = match def_crates(db, &self_ty.value, krate) {
        Some(k) => k,
        None => return ControlFlow::Continue(()),
    };

    if let Some(module_id) = visible_from_module {
        if let Some(block_id) = module_id.containing_block() {
            if let Some(impls) = db.inherent_impls_in_block(block_id) {
                impls_for_self_ty(
                    &impls,
                    self_ty,
                    db,
                    env.clone(),
                    name,
                    receiver_ty,
                    visible_from_module,
                    callback,
                )?;
            }
        }
    }

    for krate in def_crates {
        let impls = db.inherent_impls_in_crate(krate);
        impls_for_self_ty(
            &impls,
            self_ty,
            db,
            env.clone(),
            name,
            receiver_ty,
            visible_from_module,
            callback,
        )?;
    }
    return ControlFlow::Continue(());

    fn impls_for_self_ty(
        impls: &InherentImpls,
        self_ty: &Canonical<Ty>,
        db: &dyn HirDatabase,
        env: Arc<TraitEnvironment>,
        name: Option<&Name>,
        receiver_ty: Option<&Canonical<Ty>>,
        visible_from_module: Option<ModuleId>,
        callback: &mut dyn FnMut(&Canonical<Ty>, AssocItemId) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let impls_for_self_ty = filter_inherent_impls_for_self_ty(impls, &self_ty.value);
        for &impl_def in impls_for_self_ty {
            for &item in &db.impl_data(impl_def).items {
                if !is_valid_candidate(
                    db,
                    env.clone(),
                    name,
                    receiver_ty,
                    item,
                    self_ty,
                    visible_from_module,
                ) {
                    continue;
                }
                // we have to check whether the self type unifies with the type
                // that the impl is for. If we have a receiver type, this
                // already happens in `is_valid_candidate` above; if not, we
                // check it here
                if receiver_ty.is_none()
                    && inherent_impl_substs(db, env.clone(), impl_def, self_ty).is_none()
                {
                    cov_mark::hit!(impl_self_type_match_without_receiver);
                    continue;
                }
                let receiver_ty = receiver_ty.unwrap_or(self_ty);
                callback(receiver_ty, item)?;
            }
        }
        ControlFlow::Continue(())
    }
}

/// Returns the self type for the index trait call.
pub fn resolve_indexing_op(
    db: &dyn HirDatabase,
    ty: &Canonical<Ty>,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    index_trait: TraitId,
) -> Option<Canonical<Ty>> {
    let ty = InEnvironment { goal: ty.clone(), environment: env.env.clone() };
    let deref_chain = autoderef_method_receiver(db, krate, ty);
    for ty in deref_chain {
        let goal = generic_implements_goal(db, env.clone(), index_trait, ty.clone());
        if db.trait_solve(krate, goal.cast(Interner)).is_some() {
            return Some(ty);
        }
    }
    None
}

fn is_transformed_receiver_ty_equal(transformed_receiver_ty: &Ty, receiver_ty: &Ty) -> bool {
    if transformed_receiver_ty == receiver_ty {
        return true;
    }

    // a transformed receiver may be considered equal (and a valid method call candidate) if it is an array
    // with an unknown (i.e. generic) length, and the receiver is an array with the same item type but a known len,
    // this allows inherent methods on arrays to be considered valid resolution candidates
    match (transformed_receiver_ty.kind(Interner), receiver_ty.kind(Interner)) {
        (
            TyKind::Array(transformed_array_ty, transformed_array_len),
            TyKind::Array(receiver_array_ty, receiver_array_len),
        ) if transformed_array_ty == receiver_array_ty
            && transformed_array_len.is_unknown()
            && !receiver_array_len.is_unknown() =>
        {
            true
        }
        _ => false,
    }
}

fn is_valid_candidate(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    name: Option<&Name>,
    receiver_ty: Option<&Canonical<Ty>>,
    item: AssocItemId,
    self_ty: &Canonical<Ty>,
    visible_from_module: Option<ModuleId>,
) -> bool {
    match item {
        AssocItemId::FunctionId(m) => {
            let data = db.function_data(m);
            if let Some(name) = name {
                if &data.name != name {
                    return false;
                }
            }
            if let Some(receiver_ty) = receiver_ty {
                if !data.has_self_param() {
                    return false;
                }
                let transformed_receiver_ty = match transform_receiver_ty(db, env, m, self_ty) {
                    Some(ty) => ty,
                    None => return false,
                };

                if !is_transformed_receiver_ty_equal(&transformed_receiver_ty, &receiver_ty.value) {
                    return false;
                }
            }
            if let Some(from_module) = visible_from_module {
                if !db.function_visibility(m).is_visible_from(db.upcast(), from_module) {
                    cov_mark::hit!(autoderef_candidate_not_visible);
                    return false;
                }
            }

            true
        }
        AssocItemId::ConstId(c) => {
            let data = db.const_data(c);
            name.map_or(true, |name| data.name.as_ref() == Some(name)) && receiver_ty.is_none()
        }
        _ => false,
    }
}

pub(crate) fn inherent_impl_substs(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    impl_id: ImplId,
    self_ty: &Canonical<Ty>,
) -> Option<Substitution> {
    // we create a var for each type parameter of the impl; we need to keep in
    // mind here that `self_ty` might have vars of its own
    let self_ty_vars = self_ty.binders.len(Interner);
    let vars = TyBuilder::subst_for_def(db, impl_id)
        .fill_with_bound_vars(DebruijnIndex::INNERMOST, self_ty_vars)
        .build();
    let self_ty_with_vars = db.impl_self_ty(impl_id).substitute(Interner, &vars);
    let mut kinds = self_ty.binders.interned().to_vec();
    kinds.extend(
        iter::repeat(chalk_ir::WithKind::new(
            chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
            UniverseIndex::ROOT,
        ))
        .take(vars.len(Interner)),
    );
    let tys = Canonical {
        binders: CanonicalVarKinds::from_iter(Interner, kinds),
        value: (self_ty_with_vars, self_ty.value.clone()),
    };
    let substs = super::infer::unify(db, env, &tys)?;
    // We only want the substs for the vars we added, not the ones from self_ty.
    // Also, if any of the vars we added are still in there, we replace them by
    // Unknown. I think this can only really happen if self_ty contained
    // Unknown, and in that case we want the result to contain Unknown in those
    // places again.
    let suffix =
        Substitution::from_iter(Interner, substs.iter(Interner).cloned().skip(self_ty_vars));
    Some(fallback_bound_vars(suffix, self_ty_vars))
}

/// This replaces any 'free' Bound vars in `s` (i.e. those with indices past
/// num_vars_to_keep) by `TyKind::Unknown`.
fn fallback_bound_vars(s: Substitution, num_vars_to_keep: usize) -> Substitution {
    crate::fold_free_vars(s, |bound, binders| {
        if bound.index >= num_vars_to_keep && bound.debruijn == DebruijnIndex::INNERMOST {
            TyKind::Error.intern(Interner)
        } else {
            bound.shifted_in_from(binders).to_ty(Interner)
        }
    })
}

fn transform_receiver_ty(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    function_id: FunctionId,
    self_ty: &Canonical<Ty>,
) -> Option<Ty> {
    let substs = match function_id.lookup(db.upcast()).container {
        ItemContainerId::TraitId(_) => TyBuilder::subst_for_def(db, function_id)
            .push(self_ty.value.clone())
            .fill_with_unknown()
            .build(),
        ItemContainerId::ImplId(impl_id) => {
            let impl_substs = inherent_impl_substs(db, env, impl_id, self_ty)?;
            TyBuilder::subst_for_def(db, function_id)
                .use_parent_substs(&impl_substs)
                .fill_with_unknown()
                .build()
        }
        // No receiver
        ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => unreachable!(),
    };
    let sig = db.callable_item_signature(function_id.into());
    Some(sig.map(|s| s.params()[0].clone()).substitute(Interner, &substs))
}

pub fn implements_trait(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    trait_: TraitId,
) -> bool {
    let goal = generic_implements_goal(db, env, trait_, ty.clone());
    let solution = db.trait_solve(krate, goal.cast(Interner));

    solution.is_some()
}

pub fn implements_trait_unique(
    ty: &Canonical<Ty>,
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    krate: CrateId,
    trait_: TraitId,
) -> bool {
    let goal = generic_implements_goal(db, env, trait_, ty.clone());
    let solution = db.trait_solve(krate, goal.cast(Interner));

    matches!(solution, Some(crate::Solution::Unique(_)))
}

/// This creates Substs for a trait with the given Self type and type variables
/// for all other parameters, to query Chalk with it.
fn generic_implements_goal(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    trait_: TraitId,
    self_ty: Canonical<Ty>,
) -> Canonical<InEnvironment<super::DomainGoal>> {
    let mut kinds = self_ty.binders.interned().to_vec();
    let trait_ref = TyBuilder::trait_ref(db, trait_)
        .push(self_ty.value)
        .fill_with_bound_vars(DebruijnIndex::INNERMOST, kinds.len())
        .build();
    kinds.extend(
        iter::repeat(chalk_ir::WithKind::new(
            chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General),
            UniverseIndex::ROOT,
        ))
        .take(trait_ref.substitution.len(Interner) - 1),
    );
    let obligation = trait_ref.cast(Interner);
    Canonical {
        binders: CanonicalVarKinds::from_iter(Interner, kinds),
        value: InEnvironment::new(&env.env, obligation),
    }
}

fn autoderef_method_receiver(
    db: &dyn HirDatabase,
    krate: CrateId,
    ty: InEnvironment<Canonical<Ty>>,
) -> Vec<Canonical<Ty>> {
    let mut deref_chain: Vec<_> = autoderef::autoderef(db, Some(krate), ty).collect();
    // As a last step, we can do array unsizing (that's the only unsizing that rustc does for method receivers!)
    if let Some(TyKind::Array(parameters, _)) = deref_chain.last().map(|ty| ty.value.kind(Interner))
    {
        let kinds = deref_chain.last().unwrap().binders.clone();
        let unsized_ty = TyKind::Slice(parameters.clone()).intern(Interner);
        deref_chain.push(Canonical { value: unsized_ty, binders: kinds })
    }
    deref_chain
}
