use std::alloc::Allocator;

use rustc_middle::ty::{TyCtxt, Visibility};

use crate::clean;
use crate::clean::Item;
use crate::core::DocContext;
use crate::fold::{DocFolder, strip_item};
use crate::passes::Pass;

pub(crate) fn strip_aliased_non_local_pass<A: Allocator + Copy>() -> Pass<A> {
    Pass {
        name: "strip-aliased-non-local",
        run: Some(strip_aliased_non_local),
        description: "strips all non-local private aliased items from the output",
    }
}

fn strip_aliased_non_local<A: Allocator + Copy>(
    krate: clean::Crate,
    cx: &mut DocContext<'_, A>,
) -> clean::Crate {
    let mut stripper = AliasedNonLocalStripper { tcx: cx.tcx };
    stripper.fold_crate(krate, *cx.cache.search_index.allocator())
}

struct AliasedNonLocalStripper<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<A: Allocator + Copy> DocFolder<A> for AliasedNonLocalStripper<'_> {
    fn fold_item(&mut self, i: Item, alloc: A) -> Option<Item> {
        Some(match i.kind {
            clean::TypeAliasItem(..) => {
                let mut stripper = NonLocalStripper { tcx: self.tcx };
                // don't call `fold_item` as that could strip the type alias itself
                // which we don't want to strip out
                stripper.fold_item_recur(i, alloc)
            }
            _ => self.fold_item_recur(i, alloc),
        })
    }
}

struct NonLocalStripper<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<A: Allocator + Copy> DocFolder<A> for NonLocalStripper<'_> {
    fn fold_item(&mut self, i: Item, alloc: A) -> Option<Item> {
        // If not local, we want to respect the original visibility of
        // the field and not the one given by the user for the current crate.
        //
        // FIXME(#125009): Not-local should probably consider same Cargo workspace
        if let Some(def_id) = i.def_id()
            && !def_id.is_local()
            && (i.is_doc_hidden()
                // Default to *not* stripping items with inherited visibility.
                || i.visibility(self.tcx).is_some_and(|viz| viz != Visibility::Public))
        {
            return Some(strip_item(i));
        }

        Some(self.fold_item_recur(i, alloc))
    }
}
