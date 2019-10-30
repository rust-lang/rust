//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in librustc_typeck/check/method/probe.rs.
use std::sync::Arc;

use arrayvec::ArrayVec;
use hir_def::CrateModuleId;
use rustc_hash::FxHashMap;

use super::{autoderef, lower, Canonical, InEnvironment, TraitEnvironment, TraitRef};
use crate::{
    db::HirDatabase,
    impl_block::{ImplBlock, ImplId},
    resolve::Resolver,
    ty::primitive::{FloatBitness, UncertainFloatTy, UncertainIntTy},
    ty::{Ty, TypeCtor},
    AssocItem, Crate, Function, Module, Mutability, Name, Trait,
};

/// This is used as a key for indexing impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TyFingerprint {
    Apply(TypeCtor),
}

impl TyFingerprint {
    /// Creates a TyFingerprint for looking up an impl. Only certain types can
    /// have impls: if we have some `struct S`, we can have an `impl S`, but not
    /// `impl &S`. Hence, this will return `None` for reference types and such.
    fn for_impl(ty: &Ty) -> Option<TyFingerprint> {
        match ty {
            Ty::Apply(a_ty) => Some(TyFingerprint::Apply(a_ty.ctor)),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct CrateImplBlocks {
    /// To make sense of the CrateModuleIds, we need the source root.
    krate: Crate,
    impls: FxHashMap<TyFingerprint, Vec<(CrateModuleId, ImplId)>>,
    impls_by_trait: FxHashMap<Trait, Vec<(CrateModuleId, ImplId)>>,
}

impl CrateImplBlocks {
    pub fn lookup_impl_blocks<'a>(&'a self, ty: &Ty) -> impl Iterator<Item = ImplBlock> + 'a {
        let fingerprint = TyFingerprint::for_impl(ty);
        fingerprint.and_then(|f| self.impls.get(&f)).into_iter().flat_map(|i| i.iter()).map(
            move |(module_id, impl_id)| {
                let module = Module::new(self.krate, *module_id);
                ImplBlock::from_id(module, *impl_id)
            },
        )
    }

    pub fn lookup_impl_blocks_for_trait<'a>(
        &'a self,
        tr: Trait,
    ) -> impl Iterator<Item = ImplBlock> + 'a {
        self.impls_by_trait.get(&tr).into_iter().flat_map(|i| i.iter()).map(
            move |(module_id, impl_id)| {
                let module = Module::new(self.krate, *module_id);
                ImplBlock::from_id(module, *impl_id)
            },
        )
    }

    pub fn all_impls<'a>(&'a self) -> impl Iterator<Item = ImplBlock> + 'a {
        self.impls.values().chain(self.impls_by_trait.values()).flat_map(|i| i.iter()).map(
            move |(module_id, impl_id)| {
                let module = Module::new(self.krate, *module_id);
                ImplBlock::from_id(module, *impl_id)
            },
        )
    }

    fn collect_recursive(&mut self, db: &impl HirDatabase, module: Module) {
        let module_impl_blocks = db.impls_in_module(module);

        for (impl_id, _) in module_impl_blocks.impls.iter() {
            let impl_block = ImplBlock::from_id(module_impl_blocks.module, impl_id);

            let target_ty = impl_block.target_ty(db);

            if impl_block.target_trait(db).is_some() {
                if let Some(tr) = impl_block.target_trait_ref(db) {
                    self.impls_by_trait
                        .entry(tr.trait_)
                        .or_insert_with(Vec::new)
                        .push((module.id.module_id, impl_id));
                }
            } else {
                if let Some(target_ty_fp) = TyFingerprint::for_impl(&target_ty) {
                    self.impls
                        .entry(target_ty_fp)
                        .or_insert_with(Vec::new)
                        .push((module.id.module_id, impl_id));
                }
            }
        }

        for child in module.children(db) {
            self.collect_recursive(db, child);
        }
    }

    pub(crate) fn impls_in_crate_query(
        db: &impl HirDatabase,
        krate: Crate,
    ) -> Arc<CrateImplBlocks> {
        let mut crate_impl_blocks = CrateImplBlocks {
            krate,
            impls: FxHashMap::default(),
            impls_by_trait: FxHashMap::default(),
        };
        if let Some(module) = krate.root_module(db) {
            crate_impl_blocks.collect_recursive(db, module);
        }
        Arc::new(crate_impl_blocks)
    }
}

fn def_crates(db: &impl HirDatabase, cur_crate: Crate, ty: &Ty) -> Option<ArrayVec<[Crate; 2]>> {
    // Types like slice can have inherent impls in several crates, (core and alloc).
    // The corresponding impls are marked with lang items, so we can use them to find the required crates.
    macro_rules! lang_item_crate {
        ($db:expr, $cur_crate:expr, $($name:expr),+ $(,)?) => {{
            let mut v = ArrayVec::<[Crate; 2]>::new();
            $(
                v.extend($db.lang_item($cur_crate, $name.into()).and_then(|item| item.krate($db)));
            )+
            Some(v)
        }};
    }

    match ty {
        Ty::Apply(a_ty) => match a_ty.ctor {
            TypeCtor::Adt(def_id) => Some(std::iter::once(def_id.krate(db)?).collect()),
            TypeCtor::Bool => lang_item_crate!(db, cur_crate, "bool"),
            TypeCtor::Char => lang_item_crate!(db, cur_crate, "char"),
            TypeCtor::Float(UncertainFloatTy::Known(f)) => match f.bitness {
                // There are two lang items: one in libcore (fXX) and one in libstd (fXX_runtime)
                FloatBitness::X32 => lang_item_crate!(db, cur_crate, "f32", "f32_runtime"),
                FloatBitness::X64 => lang_item_crate!(db, cur_crate, "f64", "f64_runtime"),
            },
            TypeCtor::Int(UncertainIntTy::Known(i)) => {
                lang_item_crate!(db, cur_crate, i.ty_to_string())
            }
            TypeCtor::Str => lang_item_crate!(db, cur_crate, "str_alloc", "str"),
            TypeCtor::Slice => lang_item_crate!(db, cur_crate, "slice_alloc", "slice"),
            TypeCtor::RawPtr(Mutability::Shared) => lang_item_crate!(db, cur_crate, "const_ptr"),
            TypeCtor::RawPtr(Mutability::Mut) => lang_item_crate!(db, cur_crate, "mut_ptr"),
            _ => None,
        },
        _ => None,
    }
}

/// Look up the method with the given name, returning the actual autoderefed
/// receiver type (but without autoref applied yet).
pub(crate) fn lookup_method(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    name: &Name,
    resolver: &Resolver,
) -> Option<(Ty, Function)> {
    iterate_method_candidates(ty, db, resolver, Some(name), |ty, f| Some((ty.clone(), f)))
}

// This would be nicer if it just returned an iterator, but that runs into
// lifetime problems, because we need to borrow temp `CrateImplBlocks`.
pub(crate) fn iterate_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mut callback: impl FnMut(&Ty, Function) -> Option<T>,
) -> Option<T> {
    // For method calls, rust first does any number of autoderef, and then one
    // autoref (i.e. when the method takes &self or &mut self). We just ignore
    // the autoref currently -- when we find a method matching the given name,
    // we assume it fits.

    // Also note that when we've got a receiver like &S, even if the method we
    // find in the end takes &self, we still do the autoderef step (just as
    // rustc does an autoderef and then autoref again).

    let krate = resolver.krate()?;
    for derefed_ty in autoderef::autoderef(db, resolver, ty.clone()) {
        if let Some(result) = iterate_inherent_methods(&derefed_ty, db, name, krate, &mut callback)
        {
            return Some(result);
        }
        if let Some(result) =
            iterate_trait_method_candidates(&derefed_ty, db, resolver, name, &mut callback)
        {
            return Some(result);
        }
    }
    None
}

fn iterate_trait_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mut callback: impl FnMut(&Ty, Function) -> Option<T>,
) -> Option<T> {
    let krate = resolver.krate()?;
    // FIXME: maybe put the trait_env behind a query (need to figure out good input parameters for that)
    let env = lower::trait_env(db, resolver);
    // if ty is `impl Trait` or `dyn Trait`, the trait doesn't need to be in scope
    let inherent_trait = ty.value.inherent_trait().into_iter();
    // if we have `T: Trait` in the param env, the trait doesn't need to be in scope
    let traits_from_env = env
        .trait_predicates_for_self_ty(&ty.value)
        .map(|tr| tr.trait_)
        .flat_map(|t| t.all_super_traits(db));
    let traits = inherent_trait.chain(traits_from_env).chain(resolver.traits_in_scope(db));
    'traits: for t in traits {
        let data = t.trait_data(db);

        // FIXME this is a bit of a hack, since Chalk should say the same thing
        // anyway, but currently Chalk doesn't implement `dyn/impl Trait` yet
        let inherently_implemented = ty.value.inherent_trait() == Some(t);

        // we'll be lazy about checking whether the type implements the
        // trait, but if we find out it doesn't, we'll skip the rest of the
        // iteration
        let mut known_implemented = inherently_implemented;
        for item in data.items() {
            if let AssocItem::Function(m) = *item {
                let data = m.data(db);
                if name.map_or(true, |name| data.name() == name) && data.has_self_param() {
                    if !known_implemented {
                        let goal = generic_implements_goal(db, env.clone(), t, ty.clone());
                        if db.trait_solve(krate, goal).is_none() {
                            continue 'traits;
                        }
                    }
                    known_implemented = true;
                    if let Some(result) = callback(&ty.value, m) {
                        return Some(result);
                    }
                }
            }
        }
    }
    None
}

fn iterate_inherent_methods<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    name: Option<&Name>,
    krate: Crate,
    mut callback: impl FnMut(&Ty, Function) -> Option<T>,
) -> Option<T> {
    for krate in def_crates(db, krate, &ty.value)? {
        let impls = db.impls_in_crate(krate);

        for impl_block in impls.lookup_impl_blocks(&ty.value) {
            for item in impl_block.items(db) {
                if let AssocItem::Function(f) = item {
                    let data = f.data(db);
                    if name.map_or(true, |name| data.name() == name) && data.has_self_param() {
                        if let Some(result) = callback(&ty.value, f) {
                            return Some(result);
                        }
                    }
                }
            }
        }
    }
    None
}

pub(crate) fn implements_trait(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    krate: Crate,
    trait_: Trait,
) -> bool {
    if ty.value.inherent_trait() == Some(trait_) {
        // FIXME this is a bit of a hack, since Chalk should say the same thing
        // anyway, but currently Chalk doesn't implement `dyn/impl Trait` yet
        return true;
    }
    let env = lower::trait_env(db, resolver);
    let goal = generic_implements_goal(db, env, trait_, ty.clone());
    let solution = db.trait_solve(krate, goal);

    solution.is_some()
}

impl Ty {
    // This would be nicer if it just returned an iterator, but that runs into
    // lifetime problems, because we need to borrow temp `CrateImplBlocks`.
    pub fn iterate_impl_items<T>(
        self,
        db: &impl HirDatabase,
        krate: Crate,
        mut callback: impl FnMut(AssocItem) -> Option<T>,
    ) -> Option<T> {
        for krate in def_crates(db, krate, &self)? {
            let impls = db.impls_in_crate(krate);

            for impl_block in impls.lookup_impl_blocks(&self) {
                for item in impl_block.items(db) {
                    if let Some(result) = callback(item) {
                        return Some(result);
                    }
                }
            }
        }
        None
    }
}

/// This creates Substs for a trait with the given Self type and type variables
/// for all other parameters, to query Chalk with it.
fn generic_implements_goal(
    db: &impl HirDatabase,
    env: Arc<TraitEnvironment>,
    trait_: Trait,
    self_ty: Canonical<Ty>,
) -> Canonical<InEnvironment<super::Obligation>> {
    let num_vars = self_ty.num_vars;
    let substs = super::Substs::build_for_def(db, trait_)
        .push(self_ty.value)
        .fill_with_bound_vars(num_vars as u32)
        .build();
    let num_vars = substs.len() - 1 + self_ty.num_vars;
    let trait_ref = TraitRef { trait_, substs };
    let obligation = super::Obligation::Trait(trait_ref);
    Canonical { num_vars, value: InEnvironment::new(env, obligation) }
}
