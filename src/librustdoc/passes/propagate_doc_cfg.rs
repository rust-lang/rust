//! Propagates [`#[doc(cfg(...))]`](https://github.com/rust-lang/rust/issues/43781) to child items.
use std::sync::Arc;

use crate::clean::cfg::Cfg;
use crate::clean::inline::{load_attrs, merge_attrs};
use crate::clean::{Crate, Item};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

use rustc_hir::def_id::LocalDefId;

pub(crate) const PROPAGATE_DOC_CFG: Pass = Pass {
    name: "propagate-doc-cfg",
    run: propagate_doc_cfg,
    description: "propagates `#[doc(cfg(...))]` to child items",
};

pub(crate) fn propagate_doc_cfg(cr: Crate, cx: &mut DocContext<'_>) -> Crate {
    CfgPropagator { parent_cfg: None, parent: None, cx }.fold_crate(cr)
}

struct CfgPropagator<'a, 'tcx> {
    parent_cfg: Option<Arc<Cfg>>,
    parent: Option<LocalDefId>,
    cx: &'a mut DocContext<'tcx>,
}

impl<'a, 'tcx> DocFolder for CfgPropagator<'a, 'tcx> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let old_parent_cfg = self.parent_cfg.clone();

        if item.kind.is_non_assoc() &&
            let Some(def_id) = item.item_id.as_def_id().and_then(|def_id| def_id.as_local()) {
            let hir = self.cx.tcx.hir();
            let hir_id = hir.local_def_id_to_hir_id(def_id);
            let expected_parent = hir.get_parent_item(hir_id);

            // If parents are different, it means that `item` is a reexport and we need to compute
            // the actual `cfg` by iterating through its "real" parents.
            if self.parent != Some(expected_parent) {
                let mut attrs = Vec::new();
                for (parent_hir_id, _) in hir.parent_iter(hir_id) {
                    if let Some(def_id) = hir.opt_local_def_id(parent_hir_id) {
                        attrs.extend_from_slice(load_attrs(self.cx, def_id.to_def_id()));
                    }
                }
                let (_, cfg) =
                    merge_attrs(self.cx, None, item.attrs.other_attrs.as_slice(), Some(&attrs));
                item.cfg = cfg;
            }
        }
        let new_cfg = match (self.parent_cfg.take(), item.cfg.take()) {
            (None, None) => None,
            (Some(rc), None) | (None, Some(rc)) => Some(rc),
            (Some(mut a), Some(b)) => {
                let b = Arc::try_unwrap(b).unwrap_or_else(|rc| Cfg::clone(&rc));
                *Arc::make_mut(&mut a) &= b;
                Some(a)
            }
        };
        self.parent_cfg = new_cfg.clone();
        item.cfg = new_cfg;

        let old_parent =
            if let Some(def_id) = item.item_id.as_def_id().and_then(|def_id| def_id.as_local()) {
                self.parent.replace(def_id)
            } else {
                self.parent.take()
            };
        let result = self.fold_item_recur(item);
        self.parent_cfg = old_parent_cfg;
        self.parent = old_parent;

        Some(result)
    }
}
