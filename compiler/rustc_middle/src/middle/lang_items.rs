//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::ty::{self, TyCtxt};

use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_span::Span;
use rustc_target::spec::PanicStrategy;

impl<'tcx> TyCtxt<'tcx> {
    /// Returns the `DefId` for a given `LangItem`.
    /// If not found, fatally aborts compilation.
    pub fn require_lang_item(self, lang_item: LangItem, span: Option<Span>) -> DefId {
        self.lang_items().require(lang_item).unwrap_or_else(|err| {
            if let Some(span) = span {
                self.sess.span_fatal(span, err.to_string())
            } else {
                self.sess.fatal(err.to_string())
            }
        })
    }

    pub fn fn_trait_kind_from_lang_item(self, id: DefId) -> Option<ty::ClosureKind> {
        let items = self.lang_items();
        match Some(id) {
            x if x == items.fn_trait() => Some(ty::ClosureKind::Fn),
            x if x == items.fn_mut_trait() => Some(ty::ClosureKind::FnMut),
            x if x == items.fn_once_trait() => Some(ty::ClosureKind::FnOnce),
            _ => None,
        }
    }

    pub fn is_weak_lang_item(self, item_def_id: DefId) -> bool {
        self.lang_items().is_weak_lang_item(item_def_id)
    }
}

/// Returns `true` if the specified `lang_item` must be present for this
/// compilation.
///
/// Not all lang items are always required for each compilation, particularly in
/// the case of panic=abort. In these situations some lang items are injected by
/// crates and don't actually need to be defined in libstd.
pub fn required(tcx: TyCtxt<'_>, lang_item: LangItem) -> bool {
    // If we're not compiling with unwinding, we won't actually need these
    // symbols. Other panic runtimes ensure that the relevant symbols are
    // available to link things together, but they're never exercised.
    match tcx.sess.panic_strategy() {
        PanicStrategy::Abort => {
            lang_item != LangItem::EhPersonality && lang_item != LangItem::EhCatchTypeinfo
        }
        PanicStrategy::Unwind => true,
    }
}
