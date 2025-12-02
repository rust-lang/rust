use rustc_middle::ty::TyCtxt;
use tracing::debug;

use crate::clean::{self, Item, ItemIdSet};
use crate::core::DocContext;
use crate::fold::{DocFolder, strip_item};
use crate::passes::{ImplStripper, Pass};

/// Strips items that are explicitly marked with `#[deprecated]`.
pub(crate) const STRIP_DEPRECATED: Pass = Pass {
    name: "strip-deprecated",
    run: Some(strip_deprecated),
    description: "strips all items explicitly marked #[deprecated]",
};

pub(crate) fn strip_deprecated(krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    // If the flag isn't set, this pass is a no-op.
    if !cx.exclude_deprecated() {
        return krate;
    }

    let mut retained = ItemIdSet::default();
    let is_json_output = cx.is_json_output();

    // First, strip all explicitly deprecated items.
    let krate = {
        let mut stripper = DeprecatedStripper { retained: &mut retained, tcx: cx.tcx };
        stripper.fold_crate(krate)
    };

    // Then, strip impls referencing stripped items.
    let mut impl_stripper = ImplStripper {
        tcx: cx.tcx,
        retained: &retained,
        cache: &cx.cache,
        is_json_output,
        document_private: cx.document_private(),
        document_hidden: cx.document_hidden(),
    };

    impl_stripper.fold_crate(krate)
}

struct DeprecatedStripper<'a, 'tcx> {
    retained: &'a mut ItemIdSet,
    tcx: TyCtxt<'tcx>,
}

impl DocFolder for DeprecatedStripper<'_, '_> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        // If this is a re-export of a deprecated item, remove the import entirely.
        if let clean::ImportItem(clean::Import { source, .. }) = &i.kind
            && let Some(source_did) = source.did
        {
            if self.tcx.lookup_deprecation(source_did).is_some() {
                debug!("strip_deprecated: stripping re-export of deprecated item {:?}", i.name);
                return None;
            }
        }

        // Determine whether this item itself is explicitly deprecated.
        let is_explicitly_deprecated = i
            .def_id()
            .is_some_and(|did| self.tcx.lookup_deprecation(did).is_some());

        if is_explicitly_deprecated {
            debug!("strip_deprecated: stripping {:?} {:?}", i.type_(), i.name);
            // For certain kinds, keep a stripped placeholder to preserve structure,
            // similar to how strip_hidden handles fields/variants/modules.
            match i.kind {
                clean::StructFieldItem(..) | clean::ModuleItem(..) | clean::VariantItem(..) => {
                    // Recurse to allow nested items to be processed, then strip.
                    let recursed = self.fold_item_recur(i);
                    return Some(strip_item(recursed));
                }
                _ => return None,
            }
        }

        // Keep the item, remember it as retained, and recurse.
        self.retained.insert(i.item_id);
        Some(self.fold_item_recur(i))
    }
}
