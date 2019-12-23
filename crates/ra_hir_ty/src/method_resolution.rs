//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in librustc_typeck/check/method/probe.rs.
use std::sync::Arc;

use arrayvec::ArrayVec;
use hir_def::{
    lang_item::LangItemTarget, resolver::Resolver, type_ref::Mutability, AssocContainerId,
    AssocItemId, FunctionId, HasModule, ImplId, Lookup, TraitId,
};
use hir_expand::name::Name;
use ra_db::CrateId;
use ra_prof::profile;
use rustc_hash::FxHashMap;

use super::Substs;
use crate::{
    autoderef,
    db::HirDatabase,
    primitive::{FloatBitness, Uncertain},
    utils::all_super_traits,
    Canonical, InEnvironment, TraitEnvironment, TraitRef, Ty, TypeCtor, TypeWalk,
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
    impls: FxHashMap<TyFingerprint, Vec<ImplId>>,
    impls_by_trait: FxHashMap<TraitId, Vec<ImplId>>,
}

impl CrateImplBlocks {
    pub(crate) fn impls_in_crate_query(
        db: &impl HirDatabase,
        krate: CrateId,
    ) -> Arc<CrateImplBlocks> {
        let _p = profile("impls_in_crate_query");
        let mut res =
            CrateImplBlocks { impls: FxHashMap::default(), impls_by_trait: FxHashMap::default() };

        let crate_def_map = db.crate_def_map(krate);
        for (_module_id, module_data) in crate_def_map.modules.iter() {
            for impl_id in module_data.scope.impls() {
                match db.impl_trait(impl_id) {
                    Some(tr) => {
                        res.impls_by_trait.entry(tr.trait_).or_default().push(impl_id);
                    }
                    None => {
                        let self_ty = db.impl_self_ty(impl_id);
                        if let Some(self_ty_fp) = TyFingerprint::for_impl(&self_ty) {
                            res.impls.entry(self_ty_fp).or_default().push(impl_id);
                        }
                    }
                }
            }
        }

        Arc::new(res)
    }
    pub fn lookup_impl_blocks(&self, ty: &Ty) -> impl Iterator<Item = ImplId> + '_ {
        let fingerprint = TyFingerprint::for_impl(ty);
        fingerprint.and_then(|f| self.impls.get(&f)).into_iter().flatten().copied()
    }

    pub fn lookup_impl_blocks_for_trait(&self, tr: TraitId) -> impl Iterator<Item = ImplId> + '_ {
        self.impls_by_trait.get(&tr).into_iter().flatten().copied()
    }

    pub fn all_impls<'a>(&'a self) -> impl Iterator<Item = ImplId> + 'a {
        self.impls.values().chain(self.impls_by_trait.values()).flatten().copied()
    }
}

impl Ty {
    pub fn def_crates(
        &self,
        db: &impl HirDatabase,
        cur_crate: CrateId,
    ) -> Option<ArrayVec<[CrateId; 2]>> {
        // Types like slice can have inherent impls in several crates, (core and alloc).
        // The corresponding impls are marked with lang items, so we can use them to find the required crates.
        macro_rules! lang_item_crate {
        ($($name:expr),+ $(,)?) => {{
            let mut v = ArrayVec::<[LangItemTarget; 2]>::new();
            $(
                v.extend(db.lang_item(cur_crate, $name.into()));
            )+
            v
        }};
    }

        let lang_item_targets = match self {
            Ty::Apply(a_ty) => match a_ty.ctor {
                TypeCtor::Adt(def_id) => {
                    return Some(std::iter::once(def_id.module(db).krate).collect())
                }
                TypeCtor::Bool => lang_item_crate!("bool"),
                TypeCtor::Char => lang_item_crate!("char"),
                TypeCtor::Float(Uncertain::Known(f)) => match f.bitness {
                    // There are two lang items: one in libcore (fXX) and one in libstd (fXX_runtime)
                    FloatBitness::X32 => lang_item_crate!("f32", "f32_runtime"),
                    FloatBitness::X64 => lang_item_crate!("f64", "f64_runtime"),
                },
                TypeCtor::Int(Uncertain::Known(i)) => lang_item_crate!(i.ty_to_string()),
                TypeCtor::Str => lang_item_crate!("str_alloc", "str"),
                TypeCtor::Slice => lang_item_crate!("slice_alloc", "slice"),
                TypeCtor::RawPtr(Mutability::Shared) => lang_item_crate!("const_ptr"),
                TypeCtor::RawPtr(Mutability::Mut) => lang_item_crate!("mut_ptr"),
                _ => return None,
            },
            _ => return None,
        };
        let res = lang_item_targets
            .into_iter()
            .filter_map(|it| match it {
                LangItemTarget::ImplBlockId(it) => Some(it),
                _ => None,
            })
            .map(|it| it.lookup(db).container.module(db).krate)
            .collect();
        Some(res)
    }
}
/// Look up the method with the given name, returning the actual autoderefed
/// receiver type (but without autoref applied yet).
pub(crate) fn lookup_method(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    name: &Name,
    resolver: &Resolver,
) -> Option<(Ty, FunctionId)> {
    iterate_method_candidates(ty, db, resolver, Some(name), LookupMode::MethodCall, |ty, f| match f
    {
        AssocItemId::FunctionId(f) => Some((ty.clone(), f)),
        _ => None,
    })
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
// lifetime problems, because we need to borrow temp `CrateImplBlocks`.
// FIXME add a context type here?
pub fn iterate_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(&Ty, AssocItemId) -> Option<T>,
) -> Option<T> {
    match mode {
        LookupMode::MethodCall => {
            // For method calls, rust first does any number of autoderef, and then one
            // autoref (i.e. when the method takes &self or &mut self). We just ignore
            // the autoref currently -- when we find a method matching the given name,
            // we assume it fits.

            // Also note that when we've got a receiver like &S, even if the method we
            // find in the end takes &self, we still do the autoderef step (just as
            // rustc does an autoderef and then autoref again).
            let environment = TraitEnvironment::lower(db, resolver);
            let ty = InEnvironment { value: ty.clone(), environment };
            let krate = resolver.krate()?;

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

            let deref_chain: Vec<_> = autoderef::autoderef(db, Some(krate), ty.clone()).collect();
            for i in 0..deref_chain.len() {
                if let Some(result) = iterate_method_candidates_with_autoref(
                    &deref_chain[i..],
                    db,
                    resolver,
                    name,
                    &mut callback,
                ) {
                    return Some(result);
                }
            }
            None
        }
        LookupMode::Path => {
            // No autoderef for path lookups
            iterate_method_candidates_for_self_ty(&ty, db, resolver, name, &mut callback)
        }
    }
}

fn iterate_method_candidates_with_autoref<T>(
    deref_chain: &[Canonical<Ty>],
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mut callback: impl FnMut(&Ty, AssocItemId) -> Option<T>,
) -> Option<T> {
    if let Some(result) = iterate_method_candidates_by_receiver(
        &deref_chain[0],
        &deref_chain[1..],
        db,
        resolver,
        name,
        &mut callback,
    ) {
        return Some(result);
    }
    let refed = Canonical {
        num_vars: deref_chain[0].num_vars,
        value: Ty::apply_one(TypeCtor::Ref(Mutability::Shared), deref_chain[0].value.clone()),
    };
    if let Some(result) = iterate_method_candidates_by_receiver(
        &refed,
        deref_chain,
        db,
        resolver,
        name,
        &mut callback,
    ) {
        return Some(result);
    }
    let ref_muted = Canonical {
        num_vars: deref_chain[0].num_vars,
        value: Ty::apply_one(TypeCtor::Ref(Mutability::Mut), deref_chain[0].value.clone()),
    };
    if let Some(result) = iterate_method_candidates_by_receiver(
        &ref_muted,
        deref_chain,
        db,
        resolver,
        name,
        &mut callback,
    ) {
        return Some(result);
    }
    None
}

fn iterate_method_candidates_by_receiver<T>(
    receiver_ty: &Canonical<Ty>,
    rest_of_deref_chain: &[Canonical<Ty>],
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mut callback: impl FnMut(&Ty, AssocItemId) -> Option<T>,
) -> Option<T> {
    // We're looking for methods with *receiver* type receiver_ty. These could
    // be found in any of the derefs of receiver_ty, so we have to go through
    // that.
    let krate = resolver.krate()?;
    for self_ty in std::iter::once(receiver_ty).chain(rest_of_deref_chain) {
        if let Some(result) =
            iterate_inherent_methods(self_ty, db, name, Some(receiver_ty), krate, &mut callback)
        {
            return Some(result);
        }
    }
    for self_ty in std::iter::once(receiver_ty).chain(rest_of_deref_chain) {
        if let Some(result) = iterate_trait_method_candidates(
            self_ty,
            db,
            resolver,
            name,
            Some(receiver_ty),
            &mut callback,
        ) {
            return Some(result);
        }
    }
    None
}

fn iterate_method_candidates_for_self_ty<T>(
    self_ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mut callback: impl FnMut(&Ty, AssocItemId) -> Option<T>,
) -> Option<T> {
    let krate = resolver.krate()?;
    if let Some(result) = iterate_inherent_methods(self_ty, db, name, None, krate, &mut callback) {
        return Some(result);
    }
    if let Some(result) =
        iterate_trait_method_candidates(self_ty, db, resolver, name, None, &mut callback)
    {
        return Some(result);
    }
    None
}

fn iterate_trait_method_candidates<T>(
    self_ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    receiver_ty: Option<&Canonical<Ty>>,
    mut callback: impl FnMut(&Ty, AssocItemId) -> Option<T>,
) -> Option<T> {
    let krate = resolver.krate()?;
    // FIXME: maybe put the trait_env behind a query (need to figure out good input parameters for that)
    let env = TraitEnvironment::lower(db, resolver);
    // if ty is `impl Trait` or `dyn Trait`, the trait doesn't need to be in scope
    let inherent_trait = self_ty.value.inherent_trait().into_iter();
    // if we have `T: Trait` in the param env, the trait doesn't need to be in scope
    let traits_from_env = env
        .trait_predicates_for_self_ty(&self_ty.value)
        .map(|tr| tr.trait_)
        .flat_map(|t| all_super_traits(db, t));
    let traits =
        inherent_trait.chain(traits_from_env).chain(resolver.traits_in_scope(db).into_iter());
    'traits: for t in traits {
        let data = db.trait_data(t);

        // we'll be lazy about checking whether the type implements the
        // trait, but if we find out it doesn't, we'll skip the rest of the
        // iteration
        let mut known_implemented = false;
        for (_name, item) in data.items.iter() {
            if !is_valid_candidate(db, name, receiver_ty, (*item).into(), self_ty) {
                continue;
            }
            if !known_implemented {
                let goal = generic_implements_goal(db, env.clone(), t, self_ty.clone());
                if db.trait_solve(krate.into(), goal).is_none() {
                    continue 'traits;
                }
            }
            known_implemented = true;
            if let Some(result) = callback(&self_ty.value, (*item).into()) {
                return Some(result);
            }
        }
    }
    None
}

fn iterate_inherent_methods<T>(
    self_ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    name: Option<&Name>,
    receiver_ty: Option<&Canonical<Ty>>,
    krate: CrateId,
    mut callback: impl FnMut(&Ty, AssocItemId) -> Option<T>,
) -> Option<T> {
    for krate in self_ty.value.def_crates(db, krate)? {
        let impls = db.impls_in_crate(krate);

        for impl_block in impls.lookup_impl_blocks(&self_ty.value) {
            for &item in db.impl_data(impl_block).items.iter() {
                if !is_valid_candidate(db, name, receiver_ty, item, self_ty) {
                    continue;
                }
                if let Some(result) = callback(&self_ty.value, item) {
                    return Some(result);
                }
            }
        }
    }
    None
}

fn is_valid_candidate(
    db: &impl HirDatabase,
    name: Option<&Name>,
    receiver_ty: Option<&Canonical<Ty>>,
    item: AssocItemId,
    self_ty: &Canonical<Ty>,
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
                if !data.has_self_param {
                    return false;
                }
                let transformed_receiver_ty = match transform_receiver_ty(db, m, self_ty) {
                    Some(ty) => ty,
                    None => return false,
                };
                if transformed_receiver_ty != receiver_ty.value {
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
    db: &impl HirDatabase,
    impl_id: ImplId,
    self_ty: &Canonical<Ty>,
) -> Option<Substs> {
    let vars = Substs::build_for_def(db, impl_id).fill_with_bound_vars(0).build();
    let self_ty_with_vars = db.impl_self_ty(impl_id).subst(&vars);
    let self_ty_with_vars = Canonical { num_vars: vars.len(), value: self_ty_with_vars };
    super::infer::unify(&self_ty_with_vars, self_ty)
}

fn transform_receiver_ty(
    db: &impl HirDatabase,
    function_id: FunctionId,
    self_ty: &Canonical<Ty>,
) -> Option<Ty> {
    let substs = match function_id.lookup(db).container {
        AssocContainerId::TraitId(_) => Substs::build_for_def(db, function_id)
            .push(self_ty.value.clone())
            .fill_with_unknown()
            .build(),
        AssocContainerId::ImplId(impl_id) => inherent_impl_substs(db, impl_id, &self_ty)?,
        AssocContainerId::ContainerId(_) => unreachable!(),
    };
    let sig = db.callable_item_signature(function_id.into());
    Some(sig.params()[0].clone().subst(&substs))
}

pub fn implements_trait(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    krate: CrateId,
    trait_: TraitId,
) -> bool {
    if ty.value.inherent_trait() == Some(trait_) {
        // FIXME this is a bit of a hack, since Chalk should say the same thing
        // anyway, but currently Chalk doesn't implement `dyn/impl Trait` yet
        return true;
    }
    let env = TraitEnvironment::lower(db, resolver);
    let goal = generic_implements_goal(db, env, trait_, ty.clone());
    let solution = db.trait_solve(krate.into(), goal);

    solution.is_some()
}

/// This creates Substs for a trait with the given Self type and type variables
/// for all other parameters, to query Chalk with it.
fn generic_implements_goal(
    db: &impl HirDatabase,
    env: Arc<TraitEnvironment>,
    trait_: TraitId,
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
