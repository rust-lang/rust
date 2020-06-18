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
use rustc_hir::{LangItem, MissingLangItemHandler};
use rustc_span::Span;
use rustc_target::spec::PanicStrategy;

impl<'tcx> TyCtxt<'tcx> {
    pub fn fn_trait_kind_from_lang_item(&self, id: DefId) -> Option<ty::ClosureKind> {
        let items = self.lang_items();
        match id {
            x if items.fn_trait().has_def_id(x) => Some(ty::ClosureKind::Fn),
            x if items.fn_mut_trait().has_def_id(x) => Some(ty::ClosureKind::FnMut),
            x if items.fn_once_trait().has_def_id(x) => Some(ty::ClosureKind::FnOnce),
            _ => None,
        }
    }

    pub fn is_weak_lang_item(&self, item_def_id: DefId) -> bool {
        self.lang_items().is_weak_lang_item(item_def_id)
    }
}

impl<'tcx> MissingLangItemHandler for TyCtxt<'tcx> {
    fn span_fatal(&self, span: Span, msg: &str) -> ! {
        self.sess.span_fatal(span, msg)
    }

    fn fatal(&self, msg: &str) -> ! {
        self.sess.fatal(msg)
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
    if tcx.sess.panic_strategy() != PanicStrategy::Unwind
        && lang_item == LangItem::EhPersonalityLangItem
    {
        return true;
    }

    // If we don't require an allocator then we don't require
    // `#[alloc_error_handler]`.
    if tcx.allocator_kind().is_none() && lang_item == LangItem::OomLangItem {
        return true;
    }

    false
}
