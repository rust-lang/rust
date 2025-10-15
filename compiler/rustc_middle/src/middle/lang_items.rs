//! Detecting lang items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_span::Span;
use rustc_target::spec::PanicStrategy;

use crate::ty::{self, TyCtxt};

impl<'tcx> TyCtxt<'tcx> {
    /// Returns the `DefId` for a given `LangItem`.
    /// If not found, fatally aborts compilation.
    pub fn require_lang_item(self, lang_item: LangItem, span: Span) -> DefId {
        self.lang_items().get(lang_item).unwrap_or_else(|| {
            self.dcx().emit_fatal(crate::error::RequiresLangItem { span, name: lang_item.name() });
        })
    }

    pub fn is_lang_item(self, def_id: DefId, lang_item: LangItem) -> bool {
        self.lang_items().get(lang_item) == Some(def_id)
    }

    pub fn as_lang_item(self, def_id: DefId) -> Option<LangItem> {
        self.lang_items().from_def_id(def_id)
    }

    /// Given a [`DefId`] of one of the [`Fn`], [`FnMut`] or [`FnOnce`] traits,
    /// returns a corresponding [`ty::ClosureKind`].
    /// For any other [`DefId`] return `None`.
    pub fn fn_trait_kind_from_def_id(self, id: DefId) -> Option<ty::ClosureKind> {
        match self.as_lang_item(id)? {
            LangItem::Fn => Some(ty::ClosureKind::Fn),
            LangItem::FnMut => Some(ty::ClosureKind::FnMut),
            LangItem::FnOnce => Some(ty::ClosureKind::FnOnce),
            _ => None,
        }
    }

    /// Given a [`DefId`] of one of the `AsyncFn`, `AsyncFnMut` or `AsyncFnOnce` traits,
    /// returns a corresponding [`ty::ClosureKind`].
    /// For any other [`DefId`] return `None`.
    pub fn async_fn_trait_kind_from_def_id(self, id: DefId) -> Option<ty::ClosureKind> {
        match self.as_lang_item(id)? {
            LangItem::AsyncFn => Some(ty::ClosureKind::Fn),
            LangItem::AsyncFnMut => Some(ty::ClosureKind::FnMut),
            LangItem::AsyncFnOnce => Some(ty::ClosureKind::FnOnce),
            _ => None,
        }
    }

    /// Given a [`ty::ClosureKind`], get the [`DefId`] of its corresponding `Fn`-family
    /// trait, if it is defined.
    pub fn fn_trait_kind_to_def_id(self, kind: ty::ClosureKind) -> Option<DefId> {
        let items = self.lang_items();
        match kind {
            ty::ClosureKind::Fn => items.fn_trait(),
            ty::ClosureKind::FnMut => items.fn_mut_trait(),
            ty::ClosureKind::FnOnce => items.fn_once_trait(),
        }
    }

    /// Given a [`ty::ClosureKind`], get the [`DefId`] of its corresponding `Fn`-family
    /// trait, if it is defined.
    pub fn async_fn_trait_kind_to_def_id(self, kind: ty::ClosureKind) -> Option<DefId> {
        let items = self.lang_items();
        match kind {
            ty::ClosureKind::Fn => items.async_fn_trait(),
            ty::ClosureKind::FnMut => items.async_fn_mut_trait(),
            ty::ClosureKind::FnOnce => items.async_fn_once_trait(),
        }
    }

    /// Returns `true` if `id` is a `DefId` of [`Fn`], [`FnMut`] or [`FnOnce`] traits.
    pub fn is_fn_trait(self, id: DefId) -> bool {
        self.fn_trait_kind_from_def_id(id).is_some()
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
        PanicStrategy::ImmediateAbort => false,
    }
}
