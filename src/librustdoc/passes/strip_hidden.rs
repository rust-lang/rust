//! Strip all doc(hidden) items from the output.
use rustc_span::symbol::sym;
use std::mem;

use crate::clean;
use crate::clean::{Item, ItemIdSet, NestedAttributesExt};
use crate::core::DocContext;
use crate::fold::{strip_item, DocFolder};
use crate::passes::{ImplStripper, Pass};

crate const STRIP_HIDDEN: Pass = Pass {
    name: "strip-hidden",
    run: strip_hidden,
    description: "strips all `#[doc(hidden)]` items from the output",
};

/// Strip items marked `#[doc(hidden)]`
crate fn strip_hidden(krate: clean::Crate, _: &mut DocContext<'_>) -> clean::Crate {
    let mut retained = ItemIdSet::default();

    // strip all #[doc(hidden)] items
    let krate = {
        let mut stripper = Stripper { retained: &mut retained, update_retained: true };
        stripper.fold_crate(krate)
    };

    // strip all impls referencing stripped items
    let mut stripper = ImplStripper { retained: &retained };
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
            // use a dedicated hidden item for given item type if any
            match *i.kind {
                clean::StructFieldItem(..) | clean::ModuleItem(..) => {
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
                self.retained.insert(i.def_id);
            }
        }
        Some(self.fold_item_recur(i))
    }
}
