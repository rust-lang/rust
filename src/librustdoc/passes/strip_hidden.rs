//! Strip all doc(hidden) items from the output.

use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;
use std::mem;

use crate::clean;
use crate::clean::{Item, ItemIdSet, NestedAttributesExt};
use crate::core::DocContext;
use crate::fold::{strip_item, DocFolder};
use crate::passes::{ImplStripper, Pass};
use crate::visit_ast::inherits_doc_hidden;

pub(crate) const STRIP_HIDDEN: Pass = Pass {
    name: "strip-hidden",
    run: strip_hidden,
    description: "strips all `#[doc(hidden)]` items from the output",
};

/// Strip items marked `#[doc(hidden)]`
pub(crate) fn strip_hidden(krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    let mut retained = ItemIdSet::default();
    let is_json_output = cx.output_format.is_json() && !cx.show_coverage;

    // strip all #[doc(hidden)] items
    let krate = {
        let mut stripper = Stripper {
            retained: &mut retained,
            update_retained: true,
            tcx: cx.tcx,
            is_in_hidden_item: false,
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
    };
    stripper.fold_crate(krate)
}

struct Stripper<'a, 'tcx> {
    retained: &'a mut ItemIdSet,
    update_retained: bool,
    tcx: TyCtxt<'tcx>,
    is_in_hidden_item: bool,
}

impl<'a, 'tcx> Stripper<'a, 'tcx> {
    fn set_is_in_hidden_item_and_fold(&mut self, is_in_hidden_item: bool, i: Item) -> Item {
        let prev = self.is_in_hidden_item;
        self.is_in_hidden_item |= is_in_hidden_item;
        let ret = self.fold_item_recur(i);
        self.is_in_hidden_item = prev;
        ret
    }

    /// In case `i` is a non-hidden impl block, then we special-case it by changing the value
    /// of `is_in_hidden_item` to `true` because the impl children inherit its visibility.
    fn recurse_in_impl_or_exported_macro(&mut self, i: Item) -> Item {
        let prev = mem::replace(&mut self.is_in_hidden_item, false);
        let ret = self.fold_item_recur(i);
        self.is_in_hidden_item = prev;
        ret
    }
}

impl<'a, 'tcx> DocFolder for Stripper<'a, 'tcx> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        let has_doc_hidden = i.attrs.lists(sym::doc).has_word(sym::hidden);
        let is_impl_or_exported_macro = match *i.kind {
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
                // We don't need to check if it's coming from a reexport since the reexport itself was
                // already checked.
                is_hidden = i
                    .item_id
                    .as_def_id()
                    .and_then(|def_id| def_id.as_local())
                    .map(|def_id| inherits_doc_hidden(self.tcx, def_id))
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
        match *i.kind {
            clean::StructFieldItem(..) | clean::ModuleItem(..) | clean::VariantItem(..) => {
                // We need to recurse into stripped modules to
                // strip things like impl methods but when doing so
                // we must not add any items to the `retained` set.
                let old = mem::replace(&mut self.update_retained, false);
                let ret = strip_item(self.set_is_in_hidden_item_and_fold(true, i));
                self.update_retained = old;
                Some(ret)
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
