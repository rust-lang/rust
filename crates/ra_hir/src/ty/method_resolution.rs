//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in librustc_typeck/check/method/probe.rs.
use std::sync::Arc;

use arrayvec::ArrayVec;
use hir_def::{
    lang_item::LangItemTarget, resolver::HasResolver, resolver::Resolver, type_ref::Mutability,
    AssocItemId, AstItemDef, HasModule, ImplId, TraitId,
};
use hir_expand::name::Name;
use ra_db::CrateId;
use ra_prof::profile;
use rustc_hash::FxHashMap;

use crate::{
    db::HirDatabase,
    ty::primitive::{FloatBitness, Uncertain},
    ty::{utils::all_super_traits, Ty, TypeCtor},
    AssocItem, Function,
};

use super::{autoderef, Canonical, InEnvironment, TraitEnvironment, TraitRef};

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
            for &impl_id in module_data.impls.iter() {
                let impl_data = db.impl_data(impl_id);
                let resolver = impl_id.resolver(db);

                let target_ty = Ty::from_hir(db, &resolver, &impl_data.target_type);

                match &impl_data.target_trait {
                    Some(trait_ref) => {
                        if let Some(tr) =
                            TraitRef::from_hir(db, &resolver, &trait_ref, Some(target_ty))
                        {
                            res.impls_by_trait.entry(tr.trait_).or_default().push(impl_id);
                        }
                    }
                    None => {
                        if let Some(target_ty_fp) = TyFingerprint::for_impl(&target_ty) {
                            res.impls.entry(target_ty_fp).or_default().push(impl_id);
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
    pub(crate) fn def_crates(
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
            .map(|it| it.module(db).krate)
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
) -> Option<(Ty, Function)> {
    iterate_method_candidates(ty, db, resolver, Some(name), LookupMode::MethodCall, |ty, f| match f
    {
        AssocItem::Function(f) => Some((ty.clone(), f)),
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
pub(crate) fn iterate_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(&Ty, AssocItem) -> Option<T>,
) -> Option<T> {
    let krate = resolver.krate()?;
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
            for derefed_ty in autoderef::autoderef(db, resolver.krate(), ty) {
                if let Some(result) =
                    iterate_inherent_methods(&derefed_ty, db, name, mode, krate, &mut callback)
                {
                    return Some(result);
                }
                if let Some(result) = iterate_trait_method_candidates(
                    &derefed_ty,
                    db,
                    resolver,
                    name,
                    mode,
                    &mut callback,
                ) {
                    return Some(result);
                }
            }
        }
        LookupMode::Path => {
            // No autoderef for path lookups
            if let Some(result) =
                iterate_inherent_methods(&ty, db, name, mode, krate.into(), &mut callback)
            {
                return Some(result);
            }
            if let Some(result) =
                iterate_trait_method_candidates(&ty, db, resolver, name, mode, &mut callback)
            {
                return Some(result);
            }
        }
    }
    None
}

fn iterate_trait_method_candidates<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    resolver: &Resolver,
    name: Option<&Name>,
    mode: LookupMode,
    mut callback: impl FnMut(&Ty, AssocItem) -> Option<T>,
) -> Option<T> {
    let krate = resolver.krate()?;
    // FIXME: maybe put the trait_env behind a query (need to figure out good input parameters for that)
    let env = TraitEnvironment::lower(db, resolver);
    // if ty is `impl Trait` or `dyn Trait`, the trait doesn't need to be in scope
    let inherent_trait = ty.value.inherent_trait().into_iter();
    // if we have `T: Trait` in the param env, the trait doesn't need to be in scope
    let traits_from_env = env
        .trait_predicates_for_self_ty(&ty.value)
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
            if !is_valid_candidate(db, name, mode, (*item).into()) {
                continue;
            }
            if !known_implemented {
                let goal = generic_implements_goal(db, env.clone(), t, ty.clone());
                if db.trait_solve(krate.into(), goal).is_none() {
                    continue 'traits;
                }
            }
            known_implemented = true;
            if let Some(result) = callback(&ty.value, (*item).into()) {
                return Some(result);
            }
        }
    }
    None
}

fn iterate_inherent_methods<T>(
    ty: &Canonical<Ty>,
    db: &impl HirDatabase,
    name: Option<&Name>,
    mode: LookupMode,
    krate: CrateId,
    mut callback: impl FnMut(&Ty, AssocItem) -> Option<T>,
) -> Option<T> {
    for krate in ty.value.def_crates(db, krate)? {
        let impls = db.impls_in_crate(krate);

        for impl_block in impls.lookup_impl_blocks(&ty.value) {
            for &item in db.impl_data(impl_block).items.iter() {
                if !is_valid_candidate(db, name, mode, item) {
                    continue;
                }
                if let Some(result) = callback(&ty.value, item.into()) {
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
    mode: LookupMode,
    item: AssocItemId,
) -> bool {
    match item {
        AssocItemId::FunctionId(m) => {
            let data = db.function_data(m);
            name.map_or(true, |name| &data.name == name)
                && (data.has_self_param || mode == LookupMode::Path)
        }
        AssocItemId::ConstId(c) => {
            let data = db.const_data(c);
            name.map_or(true, |name| data.name.as_ref() == Some(name)) && (mode == LookupMode::Path)
        }
        _ => false,
    }
}

pub(crate) fn implements_trait(
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
