//! Strip all doc(hidden) items from the output.

use std::mem;

use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;
use tracing::debug;

use crate::clean::utils::inherits_doc_hidden;
use crate::clean::{self, Item, ItemIdSet, reexport_chain};
use crate::core::DocContext;
use crate::fold::{DocFolder, strip_item};
use crate::passes::{ImplStripper, Pass};

pub(crate) const STRIP_HIDDEN: Pass = Pass {
    name: "strip-hidden",
    run: Some(strip_hidden),
    description: "strips all `#[doc(hidden)]` items from the output",
};

/// Strip items marked `#[doc(hidden)]`
pub(crate) fn strip_hidden(krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    let mut retained = ItemIdSet::default();
    let is_json_output = cx.is_json_output();

    // strip all #[doc(hidden)] items
    let krate = {
        let mut stripper = Stripper {
            retained: &mut retained,
            update_retained: true,
            tcx: cx.tcx,
            is_in_hidden_item: false,
            last_reexport: None,
        };
        stripper.fold_crate(krate)
    };

    // strip all impls referencing stripped items
    let mut stripper = ImplStripper {
        tcx: cx.tcx,
        retained: &retained,
        cache: &cx.cache,
        is_json_output,
        document_private: cx.render_options.document_private,
        document_hidden: cx.render_options.document_hidden,
    };
    stripper.fold_crate(krate)
}

struct Stripper<'a, 'tcx> {
    retained: &'a mut ItemIdSet,
    update_retained: bool,
    tcx: TyCtxt<'tcx>,
    is_in_hidden_item: bool,
    last_reexport: Option<LocalDefId>,
}

impl Stripper<'_, '_> {
    fn set_last_reexport_then_fold_item(&mut self, i: Item) -> Item {
        let prev_from_reexport = self.last_reexport;
        if i.inline_stmt_id.is_some() {
            self.last_reexport = i.item_id.as_def_id().and_then(|def_id| def_id.as_local());
        }
        let ret = self.fold_item_recur(i);
        self.last_reexport = prev_from_reexport;
        ret
    }

    fn set_is_in_hidden_item_and_fold(&mut self, is_in_hidden_item: bool, i: Item) -> Item {
        let prev = self.is_in_hidden_item;
        self.is_in_hidden_item |= is_in_hidden_item;
        let ret = self.set_last_reexport_then_fold_item(i);
        self.is_in_hidden_item = prev;
        ret
    }

    /// In case `i` is a non-hidden impl block, then we special-case it by changing the value
    /// of `is_in_hidden_item` to `true` because the impl children inherit its visibility.
    fn recurse_in_impl_or_exported_macro(&mut self, i: Item) -> Item {
        let prev = mem::replace(&mut self.is_in_hidden_item, false);
        let ret = self.set_last_reexport_then_fold_item(i);
        self.is_in_hidden_item = prev;
        ret
    }
}

impl DocFolder for Stripper<'_, '_> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        let has_doc_hidden = i.is_doc_hidden();

        if let clean::ImportItem(clean::Import { source, .. }) = &i.kind
            && let Some(source_did) = source.did
        {
            if self.tcx.is_doc_hidden(source_did) {
                return None;
            } else if let Some(import_def_id) = i.def_id().and_then(|def_id| def_id.as_local()) {
                let reexports = reexport_chain(self.tcx, import_def_id, source_did);

                // Check if any reexport in the chain has a hidden source
                let has_hidden_source = reexports
                    .iter()
                    .filter_map(|reexport| reexport.id())
                    .any(|reexport_did| self.tcx.is_doc_hidden(reexport_did));

                if has_hidden_source {
                    return None;
                }
            }
        }

        let is_impl_or_exported_macro = match i.kind {
            clean::ImplItem(..) => true,
            // If the macro has the `#[macro_export]` attribute, it means it's accessible at the
            // crate level so it should be handled differently.
            clean::MacroItem(..) => {
                i.attrs.other_attrs.iter().any(|attr| attr.has_name(sym::macro_export))
            }
            _ => false,
        };
        let mut is_hidden = has_doc_hidden;
        if !is_impl_or_exported_macro {
            is_hidden = self.is_in_hidden_item || has_doc_hidden;
            if !is_hidden && i.inline_stmt_id.is_none() {
                // `i.inline_stmt_id` is `Some` if the item is directly reexported. If it is, we
                // don't need to check it, because the reexport itself was already checked.
                //
                // If this item is the child of a reexported module, `self.last_reexport` will be
                // `Some` even though `i.inline_stmt_id` is `None`. Hiddenness inheritance needs to
                // account for the possibility that an item's true parent module is hidden, but it's
                // inlined into a visible module true. This code shouldn't be reachable if the
                // module's reexport is itself hidden, for the same reason it doesn't need to be
                // checked if `i.inline_stmt_id` is Some: hidden reexports are never inlined.
                is_hidden = i
                    .item_id
                    .as_def_id()
                    .and_then(|def_id| def_id.as_local())
                    .map(|def_id| inherits_doc_hidden(self.tcx, def_id, self.last_reexport))
                    .unwrap_or(false);
            }
        }
        if !is_hidden {
            if self.update_retained {
                self.retained.insert(i.item_id);
            }
            return Some(if is_impl_or_exported_macro {
                self.recurse_in_impl_or_exported_macro(i)
            } else {
                self.set_is_in_hidden_item_and_fold(false, i)
            });
        }
        debug!("strip_hidden: stripping {:?} {:?}", i.type_(), i.name);
        // Use a dedicated hidden item for fields, variants, and modules.
        // We need to keep private fields and variants, so that the docs
        // can show a placeholder "// some variants omitted". We need to keep
        // private modules, because they can contain impl blocks, and impl
        // block privacy is inherited from the type and trait, not from the
        // module it's defined in. Both of these are marked "stripped," and
        // not included in the final docs, but since they still have an effect
        // on the final doc, cannot be completely removed from the Clean IR.
        match i.kind {
            clean::StructFieldItem(..) | clean::ModuleItem(..) | clean::VariantItem(..) => {
                // We need to recurse into stripped modules to
                // strip things like impl methods but when doing so
                // we must not add any items to the `retained` set.
                let old = mem::replace(&mut self.update_retained, false);
                let ret = self.set_is_in_hidden_item_and_fold(true, i);
                self.update_retained = old;
                if ret.item_id == clean::ItemId::DefId(CRATE_DEF_ID.into()) {
                    // We don't strip the current crate, even if it has `#[doc(hidden)]`.
                    debug!("strip_hidden: Not stripping local crate");
                    Some(ret)
                } else {
                    Some(strip_item(ret))
                }
            }
            _ => {
                let ret = self.set_is_in_hidden_item_and_fold(true, i);
                if has_doc_hidden {
                    // If the item itself has `#[doc(hidden)]`, then we simply remove it.
                    None
                } else {
                    // However if it's a "descendant" of a `#[doc(hidden)]` item, then we strip it.
                    Some(strip_item(ret))
                }
            }
        }
    }
}
