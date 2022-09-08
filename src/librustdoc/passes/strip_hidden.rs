//! Strip all doc(hidden) items from the output.
use rustc_span::symbol::sym;
use std::mem;

use crate::clean;
use crate::clean::{Item, ItemIdSet, NestedAttributesExt};
use crate::core::DocContext;
use crate::fold::{strip_item, DocFolder};
use crate::passes::{ImplStripper, Pass};

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
        let mut stripper = Stripper { retained: &mut retained, update_retained: true };
        stripper.fold_crate(krate)
    };

    // strip all impls referencing stripped items
    let mut stripper = ImplStripper {
        retained: &retained,
        cache: &cx.cache,
        is_json_output,
        document_private: cx.render_options.document_private,
    };
    stripper.fold_crate(krate)
}

struct Stripper<'a> {
    retained: &'a mut ItemIdSet,
    update_retained: bool,
}

impl<'a> DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if i.attrs.lists(sym::doc).has_word(sym::hidden) {
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
                    let ret = strip_item(self.fold_item_recur(i));
                    self.update_retained = old;
                    return Some(ret);
                }
                _ => return None,
            }
        } else {
            if self.update_retained {
                self.retained.insert(i.item_id);
            }
        }
        Some(self.fold_item_recur(i))
    }
}
