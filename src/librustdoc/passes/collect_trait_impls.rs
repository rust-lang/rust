use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use super::Pass;

use rustc::util::nodemap::FxHashSet;
use rustc::hir::def_id::DefId;
use syntax::symbol::sym;

pub const COLLECT_TRAIT_IMPLS: Pass = Pass {
    name: "collect-trait-impls",
    pass: collect_trait_impls,
    description: "retrieves trait impls for items in the crate",
};

pub fn collect_trait_impls(krate: Crate, cx: &DocContext<'_>) -> Crate {
    let mut synth = SyntheticImplCollector::new(cx);
    let mut krate = synth.fold_crate(krate);

    let prims: FxHashSet<PrimitiveType> =
        krate.primitives.iter().map(|p| p.1).collect();

    let crate_items = {
        let mut coll = ItemCollector::new();
        krate = coll.fold_crate(krate);
        coll.items
    };

    let mut new_items = Vec::new();

    for &cnum in cx.tcx.crates().iter() {
        for &did in cx.tcx.all_trait_implementations(cnum).iter() {
            inline::build_impl(cx, did, &mut new_items);
        }
    }

    // Also try to inline primitive impls from other crates.
    let lang_items = cx.tcx.lang_items();
    let primitive_impls = [
        lang_items.isize_impl(),
        lang_items.i8_impl(),
        lang_items.i16_impl(),
        lang_items.i32_impl(),
        lang_items.i64_impl(),
        lang_items.i128_impl(),
        lang_items.usize_impl(),
        lang_items.u8_impl(),
        lang_items.u16_impl(),
        lang_items.u32_impl(),
        lang_items.u64_impl(),
        lang_items.u128_impl(),
        lang_items.f32_impl(),
        lang_items.f64_impl(),
        lang_items.f32_runtime_impl(),
        lang_items.f64_runtime_impl(),
        lang_items.char_impl(),
        lang_items.str_impl(),
        lang_items.slice_impl(),
        lang_items.slice_u8_impl(),
        lang_items.str_alloc_impl(),
        lang_items.slice_alloc_impl(),
        lang_items.slice_u8_alloc_impl(),
        lang_items.const_ptr_impl(),
        lang_items.mut_ptr_impl(),
    ];

    for def_id in primitive_impls.iter().filter_map(|&def_id| def_id) {
        if !def_id.is_local() {
            inline::build_impl(cx, def_id, &mut new_items);

            // FIXME(eddyb) is this `doc(hidden)` check needed?
            if !cx.tcx.get_attrs(def_id).lists(sym::doc).has_word(sym::hidden) {
                let self_ty = cx.tcx.type_of(def_id);
                let impls = get_auto_trait_and_blanket_impls(cx, self_ty, def_id);
                let mut renderinfo = cx.renderinfo.borrow_mut();

                new_items.extend(impls.filter(|i| renderinfo.inlined.insert(i.def_id)));
            }
        }
    }

    let mut cleaner = BadImplStripper {
        prims,
        items: crate_items,
    };

    // scan through included items ahead of time to splice in Deref targets to the "valid" sets
    for it in &new_items {
        if let ImplItem(Impl { ref for_, ref trait_, ref items, .. }) = it.inner {
            if cleaner.keep_item(for_) && trait_.def_id() == cx.tcx.lang_items().deref_trait() {
                let target = items.iter().filter_map(|item| {
                    match item.inner {
                        TypedefItem(ref t, true) => Some(&t.type_),
                        _ => None,
                    }
                }).next().expect("Deref impl without Target type");

                if let Some(prim) = target.primitive_type() {
                    cleaner.prims.insert(prim);
                } else if let Some(did) = target.def_id() {
                    cleaner.items.insert(did);
                }
            }
        }
    }

    new_items.retain(|it| {
        if let ImplItem(Impl { ref for_, ref trait_, ref blanket_impl, .. }) = it.inner {
            cleaner.keep_item(for_) ||
                trait_.as_ref().map_or(false, |t| cleaner.keep_item(t)) ||
                blanket_impl.is_some()
        } else {
            true
        }
    });

    // `tcx.crates()` doesn't include the local crate, and `tcx.all_trait_implementations`
    // doesn't work with it anyway, so pull them from the HIR map instead
    for &trait_did in cx.all_traits.iter() {
        for &impl_node in cx.tcx.hir().trait_impls(trait_did) {
            let impl_did = cx.tcx.hir().local_def_id_from_hir_id(impl_node);
            inline::build_impl(cx, impl_did, &mut new_items);
        }
    }

    if let Some(ref mut it) = krate.module {
        if let ModuleItem(Module { ref mut items, .. }) = it.inner {
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
        SyntheticImplCollector {
            cx,
            impls: Vec::new(),
        }
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
