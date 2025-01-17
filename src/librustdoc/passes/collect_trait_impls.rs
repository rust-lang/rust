//! Collects trait impls for each item in the crate. For example, if a crate
//! defines a struct that implements a trait, this pass will note that the
//! struct implements that trait.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::{DefId, DefIdMap, DefIdSet, LOCAL_CRATE};
use rustc_middle::ty;
use rustc_span::symbol::sym;
use tracing::debug;

use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::formats::cache::Cache;
use crate::visit::DocVisitor;

pub(crate) const COLLECT_TRAIT_IMPLS: Pass = Pass {
    name: "collect-trait-impls",
    run: Some(collect_trait_impls),
    description: "retrieves trait impls for items in the crate",
};

pub(crate) fn collect_trait_impls(mut krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let tcx = cx.tcx;
    // We need to check if there are errors before running this pass because it would crash when
    // we try to get auto and blanket implementations.
    if tcx.dcx().has_errors().is_some() {
        return krate;
    }

    let synth_impls = cx.sess().time("collect_synthetic_impls", || {
        let mut synth = SyntheticImplCollector { cx, impls: Vec::new() };
        synth.visit_crate(&krate);
        synth.impls
    });

    let local_crate = ExternalCrate { crate_num: LOCAL_CRATE };
    let prims: FxHashSet<PrimitiveType> = local_crate.primitives(tcx).iter().map(|p| p.1).collect();

    let crate_items = {
        let mut coll = ItemAndAliasCollector::new(&cx.cache);
        cx.sess().time("collect_items_for_trait_impls", || coll.visit_crate(&krate));
        coll.items
    };

    let mut new_items_external = Vec::new();
    let mut new_items_local = Vec::new();

    // External trait impls.
    {
        let _prof_timer = tcx.sess.prof.generic_activity("build_extern_trait_impls");
        for &cnum in tcx.crates(()) {
            for &impl_def_id in tcx.trait_impls_in_crate(cnum) {
                cx.with_param_env(impl_def_id, |cx| {
                    inline::build_impl(cx, impl_def_id, None, &mut new_items_external);
                });
            }
        }
    }

    // Local trait impls.
    {
        let _prof_timer = tcx.sess.prof.generic_activity("build_local_trait_impls");
        let mut attr_buf = Vec::new();
        for &impl_def_id in tcx.trait_impls_in_crate(LOCAL_CRATE) {
            let mut parent = Some(tcx.parent(impl_def_id));
            while let Some(did) = parent {
                attr_buf.extend(
                    tcx.get_attrs(did, sym::doc)
                        .filter(|attr| {
                            if let Some([attr]) = attr.meta_item_list().as_deref() {
                                attr.has_name(sym::cfg)
                            } else {
                                false
                            }
                        })
                        .cloned(),
                );
                parent = tcx.opt_parent(did);
            }
            cx.with_param_env(impl_def_id, |cx| {
                inline::build_impl(cx, impl_def_id, Some((&attr_buf, None)), &mut new_items_local);
            });
            attr_buf.clear();
        }
    }

    tcx.sess.prof.generic_activity("build_primitive_trait_impls").run(|| {
        for def_id in PrimitiveType::all_impls(tcx) {
            // Try to inline primitive impls from other crates.
            if !def_id.is_local() {
                cx.with_param_env(def_id, |cx| {
                    inline::build_impl(cx, def_id, None, &mut new_items_external);
                });
            }
        }
        for (prim, did) in PrimitiveType::primitive_locations(tcx) {
            // Do not calculate blanket impl list for docs that are not going to be rendered.
            // While the `impl` blocks themselves are only in `libcore`, the module with `doc`
            // attached is directly included in `libstd` as well.
            if did.is_local() {
                for def_id in prim.impls(tcx).filter(|def_id| {
                    // Avoid including impl blocks with filled-in generics.
                    // https://github.com/rust-lang/rust/issues/94937
                    //
                    // FIXME(notriddle): https://github.com/rust-lang/rust/issues/97129
                    //
                    // This tactic of using inherent impl blocks for getting
                    // auto traits and blanket impls is a hack. What we really
                    // want is to check if `[T]` impls `Send`, which has
                    // nothing to do with the inherent impl.
                    //
                    // Rustdoc currently uses these `impl` block as a source of
                    // the `Ty`, as well as the `ParamEnv`, `GenericArgsRef`, and
                    // `Generics`. To avoid relying on the `impl` block, these
                    // things would need to be created from wholecloth, in a
                    // form that is valid for use in type inference.
                    let ty = tcx.type_of(def_id).instantiate_identity();
                    match ty.kind() {
                        ty::Slice(ty) | ty::Ref(_, ty, _) | ty::RawPtr(ty, _) => {
                            matches!(ty.kind(), ty::Param(..))
                        }
                        ty::Tuple(tys) => tys.iter().all(|ty| matches!(ty.kind(), ty::Param(..))),
                        _ => true,
                    }
                }) {
                    let impls = synthesize_auto_trait_and_blanket_impls(cx, def_id);
                    new_items_external.extend(impls.filter(|i| cx.inlined.insert(i.item_id)));
                }
            }
        }
    });

    let mut cleaner = BadImplStripper { prims, items: crate_items, cache: &cx.cache };
    let mut type_did_to_deref_target: DefIdMap<&Type> = DefIdMap::default();

    // Follow all `Deref` targets of included items and recursively add them as valid
    fn add_deref_target(
        cx: &DocContext<'_>,
        map: &DefIdMap<&Type>,
        cleaner: &mut BadImplStripper<'_>,
        targets: &mut DefIdSet,
        type_did: DefId,
    ) {
        if let Some(target) = map.get(&type_did) {
            debug!("add_deref_target: type {:?}, target {:?}", type_did, target);
            if let Some(target_prim) = target.primitive_type() {
                cleaner.prims.insert(target_prim);
            } else if let Some(target_did) = target.def_id(&cx.cache) {
                // `impl Deref<Target = S> for S`
                if !targets.insert(target_did) {
                    // Avoid infinite cycles
                    return;
                }
                cleaner.items.insert(target_did.into());
                add_deref_target(cx, map, cleaner, targets, target_did);
            }
        }
    }

    // scan through included items ahead of time to splice in Deref targets to the "valid" sets
    for it in new_items_external.iter().chain(new_items_local.iter()) {
        if let ImplItem(box Impl { ref for_, ref trait_, ref items, .. }) = it.kind
            && trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait()
            && cleaner.keep_impl(for_, true)
        {
            let target = items
                .iter()
                .find_map(|item| match item.kind {
                    AssocTypeItem(ref t, _) => Some(&t.type_),
                    _ => None,
                })
                .expect("Deref impl without Target type");

            if let Some(prim) = target.primitive_type() {
                cleaner.prims.insert(prim);
            } else if let Some(did) = target.def_id(&cx.cache) {
                cleaner.items.insert(did.into());
            }
            if let Some(for_did) = for_.def_id(&cx.cache) {
                if type_did_to_deref_target.insert(for_did, target).is_none() {
                    // Since only the `DefId` portion of the `Type` instances is known to be same for both the
                    // `Deref` target type and the impl for type positions, this map of types is keyed by
                    // `DefId` and for convenience uses a special cleaner that accepts `DefId`s directly.
                    if cleaner.keep_impl_with_def_id(for_did.into()) {
                        let mut targets = DefIdSet::default();
                        targets.insert(for_did);
                        add_deref_target(
                            cx,
                            &type_did_to_deref_target,
                            &mut cleaner,
                            &mut targets,
                            for_did,
                        );
                    }
                }
            }
        }
    }

    // Filter out external items that are not needed
    new_items_external.retain(|it| {
        if let ImplItem(box Impl { ref for_, ref trait_, ref kind, .. }) = it.kind {
            cleaner.keep_impl(
                for_,
                trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait(),
            ) || trait_.as_ref().is_some_and(|t| cleaner.keep_impl_with_def_id(t.def_id().into()))
                || kind.is_blanket()
        } else {
            true
        }
    });

    if let ModuleItem(Module { items, .. }) = &mut krate.module.inner.kind {
        items.extend(synth_impls);
        items.extend(new_items_external);
        items.extend(new_items_local);
    } else {
        panic!("collect-trait-impls can't run");
    };

    krate.external_traits.extend(cx.external_traits.drain(..));

    krate
}

struct SyntheticImplCollector<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
    impls: Vec<Item>,
}

impl DocVisitor<'_> for SyntheticImplCollector<'_, '_> {
    fn visit_item(&mut self, i: &Item) {
        if i.is_struct() || i.is_enum() || i.is_union() {
            // FIXME(eddyb) is this `doc(hidden)` check needed?
            if !self.cx.tcx.is_doc_hidden(i.item_id.expect_def_id()) {
                self.impls.extend(synthesize_auto_trait_and_blanket_impls(
                    self.cx,
                    i.item_id.expect_def_id(),
                ));
            }
        }

        self.visit_item_recur(i)
    }
}

struct ItemAndAliasCollector<'cache> {
    items: FxHashSet<ItemId>,
    cache: &'cache Cache,
}

impl<'cache> ItemAndAliasCollector<'cache> {
    fn new(cache: &'cache Cache) -> Self {
        ItemAndAliasCollector { items: FxHashSet::default(), cache }
    }
}

impl DocVisitor<'_> for ItemAndAliasCollector<'_> {
    fn visit_item(&mut self, i: &Item) {
        self.items.insert(i.item_id);

        if let TypeAliasItem(alias) = &i.inner.kind
            && let Some(did) = alias.type_.def_id(self.cache)
        {
            self.items.insert(ItemId::DefId(did));
        }

        self.visit_item_recur(i)
    }
}

struct BadImplStripper<'a> {
    prims: FxHashSet<PrimitiveType>,
    items: FxHashSet<ItemId>,
    cache: &'a Cache,
}

impl BadImplStripper<'_> {
    fn keep_impl(&self, ty: &Type, is_deref: bool) -> bool {
        if let Generic(_) = ty {
            // keep impls made on generics
            true
        } else if let Some(prim) = ty.primitive_type() {
            self.prims.contains(&prim)
        } else if let Some(did) = ty.def_id(self.cache) {
            is_deref || self.keep_impl_with_def_id(did.into())
        } else {
            false
        }
    }

    fn keep_impl_with_def_id(&self, item_id: ItemId) -> bool {
        self.items.contains(&item_id)
    }
}
