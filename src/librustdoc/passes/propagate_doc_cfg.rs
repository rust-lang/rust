use std::sync::Arc;

use crate::clean::{Crate, Item};
use crate::clean::cfg::Cfg;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

pub const PROPAGATE_DOC_CFG: Pass = Pass {
    name: "propagate-doc-cfg",
    pass: propagate_doc_cfg,
    description: "propagates `#[doc(cfg(...))]` to child items",
};

pub fn propagate_doc_cfg(cr: Crate, _: &DocContext<'_>) -> Crate {
    CfgPropagator { parent_cfg: None }.fold_crate(cr)
}

struct CfgPropagator {
    parent_cfg: Option<Arc<Cfg>>,
}

impl DocFolder for CfgPropagator {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let old_parent_cfg = self.parent_cfg.clone();

        let new_cfg = match (self.parent_cfg.take(), item.attrs.cfg.take()) {
            (None, None) => None,
            (Some(rc), None) | (None, Some(rc)) => Some(rc),
            (Some(mut a), Some(b)) => {
                let b = Arc::try_unwrap(b).unwrap_or_else(|rc| Cfg::clone(&rc));
                *Arc::make_mut(&mut a) &= b;
                Some(a)
            }
        };
        self.parent_cfg = new_cfg.clone();
        item.attrs.cfg = new_cfg;

        let result = self.fold_item_recur(item);
        self.parent_cfg = old_parent_cfg;

        result
    }
}
