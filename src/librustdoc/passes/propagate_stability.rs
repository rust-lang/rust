//! Propagates stability to child items.
//!
//! The purpose of this pass is to make items whose parents are "more unstable"
//! than the item itself inherit the parent's stability.
//! For example, [`core::error::Error`] is marked as stable since 1.0.0, but the
//! [`core::error`] module is marked as stable since 1.81.0, so we want to show
//! [`core::error::Error`] as stable since 1.81.0 as well.

use rustc_attr::{Stability, StabilityLevel};
use rustc_hir::def_id::CRATE_DEF_ID;

use crate::clean::{Crate, Item, ItemId};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

pub(crate) const PROPAGATE_STABILITY: Pass = Pass {
    name: "propagate-stability",
    run: Some(propagate_stability),
    description: "propagates stability to child items",
};

pub(crate) fn propagate_stability(cr: Crate, cx: &mut DocContext<'_>) -> Crate {
    let crate_stability = cx.tcx.lookup_stability(CRATE_DEF_ID);
    StabilityPropagator { parent_stability: crate_stability, cx }.fold_crate(cr)
}

struct StabilityPropagator<'a, 'tcx> {
    parent_stability: Option<Stability>,
    cx: &'a mut DocContext<'tcx>,
}

impl<'a, 'tcx> DocFolder for StabilityPropagator<'a, 'tcx> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let parent_stability = self.parent_stability;

        let stability = match item.item_id {
            ItemId::DefId(def_id) => {
                let own_stability = self.cx.tcx.lookup_stability(def_id);

                // If any of the item's parents was stabilized later or is still unstable,
                // then use the parent's stability instead.
                if let Some(own_stab) = own_stability
                    && let StabilityLevel::Stable {
                        since: own_since,
                        allowed_through_unstable_modules: false,
                    } = own_stab.level
                    && let Some(parent_stab) = parent_stability
                    && (parent_stab.is_unstable()
                        || parent_stab
                            .stable_since()
                            .is_some_and(|parent_since| parent_since > own_since))
                {
                    parent_stability
                } else {
                    own_stability
                }
            }
            ItemId::Auto { .. } | ItemId::Blanket { .. } => {
                // For now, we do now show stability for synthesized impls.
                None
            }
        };

        item.inner.stability = stability;
        self.parent_stability = stability;
        let item = self.fold_item_recur(item);
        self.parent_stability = parent_stability;

        Some(item)
    }
}
