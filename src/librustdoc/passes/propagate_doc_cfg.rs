//! Propagates [`#[doc(cfg(...))]`](https://github.com/rust-lang/rust/issues/43781) to child items.

use std::sync::Arc;

use rustc_hir::def_id::LocalDefId;

use crate::clean::cfg::Cfg;
use crate::clean::inline::{load_attrs, merge_attrs};
use crate::clean::{Crate, Item, ItemKind};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

pub(crate) const PROPAGATE_DOC_CFG: Pass = Pass {
    name: "propagate-doc-cfg",
    run: Some(propagate_doc_cfg),
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

impl CfgPropagator<'_, '_> {
    // Some items need to merge their attributes with their parents' otherwise a few of them
    // (mostly `cfg` ones) will be missing.
    fn merge_with_parent_attributes(&mut self, item: &mut Item) {
        let check_parent = match &item.kind {
            // impl blocks can be in different modules with different cfg and we need to get them
            // as well.
            ItemKind::ImplItem(_) => false,
            kind if kind.is_non_assoc() => true,
            _ => return,
        };

        let Some(def_id) = item.item_id.as_def_id().and_then(|def_id| def_id.as_local()) else {
            return;
        };

        if check_parent {
            let expected_parent = self.cx.tcx.opt_local_parent(def_id);
            // If parents are different, it means that `item` is a reexport and we need
            // to compute the actual `cfg` by iterating through its "real" parents.
            if self.parent.is_some() && self.parent == expected_parent {
                return;
            }
        }

        let mut attrs = Vec::new();
        let mut next_def_id = def_id;
        while let Some(parent_def_id) = self.cx.tcx.opt_local_parent(next_def_id) {
            attrs.extend_from_slice(load_attrs(self.cx, parent_def_id.to_def_id()));
            next_def_id = parent_def_id;
        }

        let (_, cfg) =
            merge_attrs(self.cx, item.attrs.other_attrs.as_slice(), Some((&attrs, None)));
        item.inner.cfg = cfg;
    }
}

impl DocFolder for CfgPropagator<'_, '_> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let old_parent_cfg = self.parent_cfg.clone();

        self.merge_with_parent_attributes(&mut item);

        let new_cfg = match (self.parent_cfg.take(), item.inner.cfg.take()) {
            (None, None) => None,
            (Some(rc), None) | (None, Some(rc)) => Some(rc),
            (Some(mut a), Some(b)) => {
                let b = Arc::try_unwrap(b).unwrap_or_else(|rc| Cfg::clone(&rc));
                *Arc::make_mut(&mut a) &= b;
                Some(a)
            }
        };
        self.parent_cfg = new_cfg.clone();
        item.inner.cfg = new_cfg;

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
