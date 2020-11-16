use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_span::symbol::sym;

crate const COLLECT_TRAIT_IMPLS: Pass = Pass {
    name: "collect-trait-impls",
    run: collect_trait_impls,
    description: "retrieves trait impls for items in the crate",
};

crate fn collect_trait_impls(krate: Crate, cx: &DocContext<'_>) -> Crate {
    let mut synth = SyntheticImplCollector::new(cx);
    let mut krate = synth.fold_crate(krate);

    let prims: FxHashSet<PrimitiveType> = krate.primitives.iter().map(|p| p.1).collect();

    let crate_items = {
        let mut coll = ItemCollector::new();
        krate = coll.fold_crate(krate);
        coll.items
    };

    let mut new_items = Vec::new();

    for &cnum in cx.tcx.crates().iter() {
        for &(did, _) in cx.tcx.all_trait_implementations(cnum).iter() {
            cx.tcx.sess.time("build_extern_trait_impl", || {
                inline::build_impl(cx, None, did, None, &mut new_items);
            });
        }
    }

    // Also try to inline primitive impls from other crates.
    for &def_id in PrimitiveType::all_impls(cx.tcx).values().flatten() {
        if !def_id.is_local() {
            inline::build_impl(cx, None, def_id, None, &mut new_items);

            // FIXME(eddyb) is this `doc(hidden)` check needed?
            if !cx.tcx.get_attrs(def_id).lists(sym::doc).has_word(sym::hidden) {
                let self_ty = cx.tcx.type_of(def_id);
                let impls = get_auto_trait_and_blanket_impls(cx, self_ty, def_id);
                let mut renderinfo = cx.renderinfo.borrow_mut();

                new_items.extend(impls.filter(|i| renderinfo.inlined.insert(i.def_id)));
            }
        }
    }

    let mut cleaner = BadImplStripper { prims, items: crate_items };

    // scan through included items ahead of time to splice in Deref targets to the "valid" sets
    for it in &new_items {
        if let ImplItem(Impl { ref for_, ref trait_, ref items, .. }) = it.kind {
            if cleaner.keep_item(for_) && trait_.def_id() == cx.tcx.lang_items().deref_trait() {
                let target = items
                    .iter()
                    .find_map(|item| match item.kind {
                        TypedefItem(ref t, true) => Some(&t.type_),
                        _ => None,
                    })
                    .expect("Deref impl without Target type");

                if let Some(prim) = target.primitive_type() {
                    cleaner.prims.insert(prim);
                } else if let Some(did) = target.def_id() {
                    cleaner.items.insert(did);
                }
            }
        }
    }

    new_items.retain(|it| {
        if let ImplItem(Impl { ref for_, ref trait_, ref blanket_impl, .. }) = it.kind {
            cleaner.keep_item(for_)
                || trait_.as_ref().map_or(false, |t| cleaner.keep_item(t))
                || blanket_impl.is_some()
        } else {
            true
        }
    });

    // `tcx.crates()` doesn't include the local crate, and `tcx.all_trait_implementations`
    // doesn't work with it anyway, so pull them from the HIR map instead
    for &trait_did in cx.tcx.all_traits(LOCAL_CRATE).iter() {
        for &impl_node in cx.tcx.hir().trait_impls(trait_did) {
            let impl_did = cx.tcx.hir().local_def_id(impl_node);
            cx.tcx.sess.time("build_local_trait_impl", || {
                inline::build_impl(cx, None, impl_did.to_def_id(), None, &mut new_items);
            });
        }
    }

    if let Some(ref mut it) = krate.module {
        if let ModuleItem(Module { ref mut items, .. }) = it.kind {
            items.extend(synth.impls);
            items.extend(new_items);
        } else {
            panic!("collect-trait-impls can't run");
        }
    } else {
        panic!("collect-trait-impls can't run");
    }

    krate
}

struct SyntheticImplCollector<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
    impls: Vec<Item>,
}

impl<'a, 'tcx> SyntheticImplCollector<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        SyntheticImplCollector { cx, impls: Vec::new() }
    }
}

impl<'a, 'tcx> DocFolder for SyntheticImplCollector<'a, 'tcx> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if i.is_struct() || i.is_enum() || i.is_union() {
            // FIXME(eddyb) is this `doc(hidden)` check needed?
            if !self.cx.tcx.get_attrs(i.def_id).lists(sym::doc).has_word(sym::hidden) {
                self.impls.extend(get_auto_trait_and_blanket_impls(
                    self.cx,
                    self.cx.tcx.type_of(i.def_id),
                    i.def_id,
                ));
            }
        }

        self.fold_item_recur(i)
    }
}

#[derive(Default)]
struct ItemCollector {
    items: FxHashSet<DefId>,
}

impl ItemCollector {
    fn new() -> Self {
        Self::default()
    }
}

impl DocFolder for ItemCollector {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        self.items.insert(i.def_id);

        self.fold_item_recur(i)
    }
}

struct BadImplStripper {
    prims: FxHashSet<PrimitiveType>,
    items: FxHashSet<DefId>,
}

impl BadImplStripper {
    fn keep_item(&self, ty: &Type) -> bool {
        if let Generic(_) = ty {
            // keep impls made on generics
            true
        } else if let Some(prim) = ty.primitive_type() {
            self.prims.contains(&prim)
        } else if let Some(did) = ty.def_id() {
            self.items.contains(&did)
        } else {
            false
        }
    }
}
