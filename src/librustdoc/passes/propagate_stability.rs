//! Propagates stability to child items.
//!
//! The purpose of this pass is to make items whose parents are "more unstable"
//! than the item itself inherit the parent's stability.
//! For example, [`core::error::Error`] is marked as stable since 1.0.0, but the
//! [`core::error`] module is marked as stable since 1.81.0, so we want to show
//! [`core::error::Error`] as stable since 1.81.0 as well.

use rustc_attr::{Stability, StabilityLevel};
use rustc_hir::def_id::CRATE_DEF_ID;

use crate::clean::{Crate, Item, ItemId, ItemKind};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::Pass;

pub(crate) const PROPAGATE_STABILITY: Pass = Pass {
    name: "propagate-stability",
    run: Some(propagate_stability),
    description: "propagates stability to child items",
};

pub(crate) fn propagate_stability(cr: Crate, cx: &mut DocContext<'_>) -> Crate {
    let crate_stability = cx.tcx.lookup_stability(CRATE_DEF_ID).cloned();
    StabilityPropagator { parent_stability: crate_stability, cx }.fold_crate(cr)
}

struct StabilityPropagator<'a, 'tcx> {
    parent_stability: Option<Stability>,
    cx: &'a mut DocContext<'tcx>,
}

impl DocFolder for StabilityPropagator<'_, '_> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let stability = match item.item_id {
            ItemId::DefId(def_id) => {
                let own_stability = self.cx.tcx.lookup_stability(def_id);

                let (ItemKind::StrippedItem(box kind) | kind) = &item.kind;
                match kind {
                    ItemKind::ExternCrateItem { .. }
                    | ItemKind::ImportItem(..)
                    | ItemKind::StructItem(..)
                    | ItemKind::UnionItem(..)
                    | ItemKind::EnumItem(..)
                    | ItemKind::FunctionItem(..)
                    | ItemKind::ModuleItem(..)
                    | ItemKind::TypeAliasItem(..)
                    | ItemKind::StaticItem(..)
                    | ItemKind::TraitItem(..)
                    | ItemKind::TraitAliasItem(..)
                    | ItemKind::StructFieldItem(..)
                    | ItemKind::VariantItem(..)
                    | ItemKind::ForeignFunctionItem(..)
                    | ItemKind::ForeignStaticItem(..)
                    | ItemKind::ForeignTypeItem
                    | ItemKind::MacroItem(..)
                    | ItemKind::ProcMacroItem(..)
                    | ItemKind::ConstantItem(..) => {
                        merge_stability(own_stability, self.parent_stability.as_ref())
                    }

                    // Don't inherit the parent's stability for these items, because they
                    // are potentially accessible even if the parent is more unstable.
                    ItemKind::ImplItem(..)
                    | ItemKind::TyMethodItem(..)
                    | ItemKind::MethodItem(..)
                    | ItemKind::TyAssocConstItem(..)
                    | ItemKind::AssocConstItem(..)
                    | ItemKind::TyAssocTypeItem(..)
                    | ItemKind::AssocTypeItem(..)
                    | ItemKind::PrimitiveItem(..)
                    | ItemKind::KeywordItem => own_stability.cloned(),

                    ItemKind::StrippedItem(..) => unreachable!(),
                }
            }
            ItemId::Auto { .. } | ItemId::Blanket { .. } => {
                // For now, we do now show stability for synthesized impls.
                None
            }
        };

        item.inner.stability = stability.clone();
        let parent_stability = std::mem::replace(&mut self.parent_stability, stability);
        let item = self.fold_item_recur(item);
        self.parent_stability = parent_stability;

        Some(item)
    }
}

fn merge_stability(
    own_stability: Option<&Stability>,
    parent_stability: Option<&Stability>,
) -> Option<Stability> {
    if let Some(own_stab) = own_stability
        && let Some(parent_stab) = parent_stability
    {
        match own_stab.level {
            // If any of a stable item's parents were stabilized later or are still unstable,
            // then use the parent's stability instead.
            StabilityLevel::Stable {
                since: own_since,
                allowed_through_unstable_modules: false,
                ..
            } if parent_stab.is_unstable()
                || parent_stab
                    .stable_since()
                    .is_some_and(|parent_since| parent_since > own_since) =>
            {
                parent_stability.cloned()
            }

            // If any of an unstable item's parents depend on other unstable features,
            // then use those as well.
            StabilityLevel::Unstable { unstables: ref own_gates, reason, is_soft }
                if let StabilityLevel::Unstable { unstables: parent_gates, .. } =
                    &parent_stab.level =>
            {
                let missing_unstables = parent_gates
                    .iter()
                    .filter(|p| !own_gates.iter().any(|u| u.feature == p.feature));
                let unstables = own_gates.iter().chain(missing_unstables).cloned().collect();
                Some(Stability { level: StabilityLevel::Unstable { unstables, reason, is_soft } })
            }

            _ => own_stability.cloned(),
        }
    } else {
        own_stability.cloned()
    }
}
