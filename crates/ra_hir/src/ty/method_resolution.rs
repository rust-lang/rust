//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in librustc_typeck/check/method/probe.rs.
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    HirDatabase, module_tree::ModuleId, Module, Crate, Name, Function, Trait,
    ids::TraitId,
    impl_block::{ImplId, ImplBlock, ImplItem},
    ty::{AdtDef, Ty},
};

/// This is used as a key for indexing impls.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TyFingerprint {
    Adt(AdtDef), // we'll also want to index impls for primitive types etc.
}

impl TyFingerprint {
    /// Creates a TyFingerprint for looking up an impl. Only certain types can
    /// have impls: if we have some `struct S`, we can have an `impl S`, but not
    /// `impl &S`. Hence, this will return `None` for reference types and such.
    fn for_impl(ty: &Ty) -> Option<TyFingerprint> {
        match ty {
            Ty::Adt { def_id, .. } => Some(TyFingerprint::Adt(*def_id)),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct CrateImplBlocks {
    /// To make sense of the ModuleIds, we need the source root.
    krate: Crate,
    impls: FxHashMap<TyFingerprint, Vec<(ModuleId, ImplId)>>,
    impls_by_trait: FxHashMap<TraitId, Vec<(ModuleId, ImplId)>>,
}

impl CrateImplBlocks {
    pub fn lookup_impl_blocks<'a>(
        &'a self,
        db: &'a impl HirDatabase,
        ty: &Ty,
    ) -> impl Iterator<Item = (Module, ImplBlock)> + 'a {
        let fingerprint = TyFingerprint::for_impl(ty);
        fingerprint.and_then(|f| self.impls.get(&f)).into_iter().flat_map(|i| i.iter()).map(
            move |(module_id, impl_id)| {
                let module = Module { krate: self.krate, module_id: *module_id };
                let module_impl_blocks = db.impls_in_module(module);
                (module, ImplBlock::from_id(module_impl_blocks, *impl_id))
            },
        )
    }

    pub fn lookup_impl_blocks_for_trait<'a>(
        &'a self,
        db: &'a impl HirDatabase,
        tr: &Trait,
    ) -> impl Iterator<Item = (Module, ImplBlock)> + 'a {
        let id = tr.id;
        self.impls_by_trait.get(&id).into_iter().flat_map(|i| i.iter()).map(
            move |(module_id, impl_id)| {
                let module = Module { krate: self.krate, module_id: *module_id };
                let module_impl_blocks = db.impls_in_module(module);
                (module, ImplBlock::from_id(module_impl_blocks, *impl_id))
            },
        )
    }

    fn collect_recursive(&mut self, db: &impl HirDatabase, module: &Module) {
        let module_impl_blocks = db.impls_in_module(module.clone());

        for (impl_id, _) in module_impl_blocks.impls.iter() {
            let impl_block = ImplBlock::from_id(Arc::clone(&module_impl_blocks), impl_id);

            let target_ty = impl_block.target_ty(db);

            if let Some(target_ty_fp) = TyFingerprint::for_impl(&target_ty) {
                self.impls
                    .entry(target_ty_fp)
                    .or_insert_with(Vec::new)
                    .push((module.module_id, impl_id));
            }

            if let Some(tr) = impl_block.target_trait(db) {
                self.impls_by_trait
                    .entry(tr.id)
                    .or_insert_with(Vec::new)
                    .push((module.module_id, impl_id));
            }
        }

        for child in module.children(db) {
            self.collect_recursive(db, &child);
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
            crate_impl_blocks.collect_recursive(db, &module);
        }
        Arc::new(crate_impl_blocks)
    }
}

fn def_crate(db: &impl HirDatabase, ty: &Ty) -> Option<Crate> {
    match ty {
        Ty::Adt { def_id, .. } => def_id.krate(db),
        _ => None,
    }
}

impl Ty {
    // TODO: cache this as a query?
    // - if so, what signature? (TyFingerprint, Name)?
    // - or maybe cache all names and def_ids of methods per fingerprint?
    pub fn lookup_method(self, db: &impl HirDatabase, name: &Name) -> Option<Function> {
        self.iterate_methods(db, |f| {
            let sig = f.signature(db);
            if sig.name() == name && sig.has_self_param() {
                Some(f)
            } else {
                None
            }
        })
    }

    // This would be nicer if it just returned an iterator, but that runs into
    // lifetime problems, because we need to borrow temp `CrateImplBlocks`.
    pub fn iterate_methods<T>(
        self,
        db: &impl HirDatabase,
        mut callback: impl FnMut(Function) -> Option<T>,
    ) -> Option<T> {
        // For method calls, rust first does any number of autoderef, and then one
        // autoref (i.e. when the method takes &self or &mut self). We just ignore
        // the autoref currently -- when we find a method matching the given name,
        // we assume it fits.

        // Also note that when we've got a receiver like &S, even if the method we
        // find in the end takes &self, we still do the autoderef step (just as
        // rustc does an autoderef and then autoref again).

        for derefed_ty in self.autoderef(db) {
            let krate = match def_crate(db, &derefed_ty) {
                Some(krate) => krate,
                None => continue,
            };
            let impls = db.impls_in_crate(krate);

            for (_, impl_block) in impls.lookup_impl_blocks(db, &derefed_ty) {
                for item in impl_block.items() {
                    match item {
                        ImplItem::Method(f) => {
                            if let Some(result) = callback(f.clone()) {
                                return Some(result);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
}
