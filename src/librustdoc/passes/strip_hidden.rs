use rustc::util::nodemap::DefIdSet;
use std::mem;

use clean::{self, AttributesExt, NestedAttributesExt};
use clean::Item;
use core::DocContext;
use fold;
use fold::DocFolder;
use fold::StripItem;
use passes::{ImplStripper, Pass};

pub const STRIP_HIDDEN: Pass =
    Pass::early("strip-hidden", strip_hidden,
                "strips all doc(hidden) items from the output");

/// Strip items marked `#[doc(hidden)]`
pub fn strip_hidden(krate: clean::Crate, _: &DocContext) -> clean::Crate {
    let mut retained = DefIdSet::default();

    // strip all #[doc(hidden)] items
    let krate = {
        let mut stripper = Stripper{ retained: &mut retained, update_retained: true };
        stripper.fold_crate(krate)
    };

    // strip all impls referencing stripped items
    let mut stripper = ImplStripper { retained: &retained };
    let krate = stripper.fold_crate(krate);

    krate
}

struct Stripper<'a> {
    retained: &'a mut DefIdSet,
    update_retained: bool,
}

impl<'a> fold::DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if i.attrs.lists("doc").has_word("hidden") {
            debug!("strip_hidden: stripping {} {:?}", i.type_(), i.name);
            // use a dedicated hidden item for given item type if any
            match i.inner {
                clean::StructFieldItem(..) | clean::ModuleItem(..) => {
                    // We need to recurse into stripped modules to
                    // strip things like impl methods but when doing so
                    // we must not add any items to the `retained` set.
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = StripItem(self.fold_item_recur(i).unwrap()).strip();
                    self.update_retained = old;
                    return ret;
                }
                _ => return None,
            }
        } else {
            if self.update_retained {
                self.retained.insert(i.def_id);
            }
        }
        self.fold_item_recur(i)
    }
}
