//! Validity checking for weak lang items

use crate::ty::TyCtxt;
use rustc_hir::def_id::DefId;
use rustc_lang_items::{lang_items, LangItem};
use rustc_target::spec::PanicStrategy;

pub use rustc_lang_items::weak_lang_items::link_name;

impl<'tcx> TyCtxt<'tcx> {
    pub fn is_weak_lang_item(&self, item_def_id: DefId) -> bool {
        self.lang_items().is_weak_lang_item(item_def_id)
    }
}

/// Returns `true` if the specified `lang_item` doesn't actually need to be
/// present for this compilation.
///
/// Not all lang items are always required for each compilation, particularly in
/// the case of panic=abort. In these situations some lang items are injected by
/// crates and don't actually need to be defined in libstd.
pub fn whitelisted(tcx: TyCtxt<'_>, lang_item: LangItem) -> bool {
    // If we're not compiling with unwinding, we won't actually need these
    // symbols. Other panic runtimes ensure that the relevant symbols are
    // available to link things together, but they're never exercised.
    if tcx.sess.panic_strategy() != PanicStrategy::Unwind {
        return lang_item == lang_items::EhPersonalityLangItem
            || lang_item == lang_items::EhUnwindResumeLangItem;
    }

    false
}
