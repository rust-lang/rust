//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

pub use self::LangItem::*;

use crate::ty::{self, TyCtxt};

use rustc_hir::def_id::DefId;
use rustc_span::Span;

pub use rustc_lang_items::{LangItem, LanguageItems};

impl<'tcx> TyCtxt<'tcx> {
    /// Returns the `DefId` for a given `LangItem`.
    /// If not found, fatally aborts compilation.
    pub fn require_lang_item(&self, lang_item: LangItem, span: Option<Span>) -> DefId {
        self.lang_items().require(lang_item).unwrap_or_else(|msg| {
            if let Some(span) = span {
                self.sess.span_fatal(span, &msg)
            } else {
                self.sess.fatal(&msg)
            }
        })
    }

    pub fn fn_trait_lang_item(&self, id: DefId) -> Option<ty::ClosureKind> {
        let items = self.lang_items();
        match Some(id) {
            x if x == items.fn_trait() => Some(ty::ClosureKind::Fn),
            x if x == items.fn_mut_trait() => Some(ty::ClosureKind::FnMut),
            x if x == items.fn_once_trait() => Some(ty::ClosureKind::FnOnce),
            _ => None,
        }
    }
}
