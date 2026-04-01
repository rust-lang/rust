//! Propagates [`#[doc(cfg(...))]`](https://github.com/rust-lang/rust/issues/43781) to child items.

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::Attribute;
use rustc_hir::attrs::{AttributeKind, DocAttribute};

use crate::clean::inline::{load_attrs, merge_attrs};
use crate::clean::{CfgInfo, Crate, Item, ItemId, ItemKind};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

pub(crate) const PROPAGATE_DOC_CFG: Pass = Pass {
    name: "propagate-doc-cfg",
    run: Some(propagate_doc_cfg),
    description: "propagates `#[doc(cfg(...))]` to child items",
};

pub(crate) fn propagate_doc_cfg(cr: Crate, cx: &mut DocContext<'_>) -> Crate {
    if cx.tcx.features().doc_cfg() {
        CfgPropagator { cx, cfg_info: CfgInfo::default(), impl_cfg_info: FxHashMap::default() }
            .fold_crate(cr)
    } else {
        cr
    }
}

struct CfgPropagator<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
    cfg_info: CfgInfo,

    /// To ensure the `doc_cfg` feature works with how `rustdoc` handles impls, we need to store
    /// the `cfg` info of `impl`s placeholder to use them later on the "real" impl item.
    impl_cfg_info: FxHashMap<ItemId, CfgInfo>,
}

/// This function goes through the attributes list (`new_attrs`) and extract the `cfg` tokens from
/// it and put them into `attrs`.
fn add_only_cfg_attributes(attrs: &mut Vec<Attribute>, new_attrs: &[Attribute]) {
    for attr in new_attrs {
        if let Attribute::Parsed(AttributeKind::Doc(d)) = attr
            && !d.cfg.is_empty()
        {
            let mut new_attr = DocAttribute::default();
            new_attr.cfg = d.cfg.clone();
            attrs.push(Attribute::Parsed(AttributeKind::Doc(Box::new(new_attr))));
        } else if let Attribute::Parsed(AttributeKind::CfgTrace(..)) = attr {
            // If it's a `cfg()` attribute, we keep it.
            attrs.push(attr.clone());
        }
    }
}

impl CfgPropagator<'_, '_> {
    // Some items need to merge their attributes with their parents' otherwise a few of them
    // (mostly `cfg` ones) will be missing.
    fn merge_with_parent_attributes(&mut self, item: &mut Item) {
        let mut attrs = Vec::new();
        // We only need to merge an item attributes with its parent's in case it's an impl as an
        // impl might not be defined in the same module as the item it implements.
        //
        // Otherwise, `cfg_info` already tracks everything we need so nothing else to do!
        if matches!(item.kind, ItemKind::ImplItem(_))
            && let Some(mut next_def_id) = item.item_id.as_local_def_id()
        {
            while let Some(parent_def_id) = self.cx.tcx.opt_local_parent(next_def_id) {
                let x = load_attrs(self.cx.tcx, parent_def_id.to_def_id());
                add_only_cfg_attributes(&mut attrs, x);
                next_def_id = parent_def_id;
            }
        }

        let (_, cfg) = merge_attrs(
            self.cx,
            item.attrs.other_attrs.as_slice(),
            Some((&attrs, None)),
            &mut self.cfg_info,
        );
        item.inner.cfg = cfg;
    }
}

impl DocFolder for CfgPropagator<'_, '_> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let old_cfg_info = self.cfg_info.clone();

        // If we have an impl, we check if it has an associated `cfg` "context", and if so we will
        // use that context instead of the actual (wrong) one.
        if let ItemKind::ImplItem(_) = item.kind
            && let Some(cfg_info) = self.impl_cfg_info.remove(&item.item_id)
        {
            self.cfg_info = cfg_info;
        }

        if let ItemKind::PlaceholderImplItem = item.kind {
            // If we have a placeholder impl, we store the current `cfg` "context" to be used
            // on the actual impl later on (the impls are generated after we go through the whole
            // AST so they're stored in the `krate` object at the end).
            self.impl_cfg_info.insert(item.item_id, self.cfg_info.clone());
        } else {
            self.merge_with_parent_attributes(&mut item);
        }

        let result = self.fold_item_recur(item);
        self.cfg_info = old_cfg_info;

        Some(result)
    }
}
