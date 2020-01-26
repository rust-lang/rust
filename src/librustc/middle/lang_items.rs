//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

pub use self::LangItem::*;

use crate::middle::cstore::ExternCrate;
use crate::middle::weak_lang_items;
use crate::ty::{self, TyCtxt};

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_lang_items::lang_items::ITEM_REFS;
use rustc_lang_items::Target;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use syntax::ast;

pub use rustc_lang_items::{LangItem, LanguageItems};

struct LanguageItemCollector<'tcx> {
    items: LanguageItems,
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'v> for LanguageItemCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        if let Some((value, span)) = extract(&item.attrs) {
            let actual_target = Target::from_item(item);
            match ITEM_REFS.get(&*value.as_str()).cloned() {
                // Known lang item with attribute on correct target.
                Some((item_index, expected_target)) if actual_target == expected_target => {
                    let def_id = self.tcx.hir().local_def_id(item.hir_id);
                    self.collect_item(item_index, def_id);
                }
                // Known lang item with attribute on incorrect target.
                Some((_, expected_target)) => {
                    struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0718,
                        "`{}` language item must be applied to a {}",
                        value,
                        expected_target,
                    )
                    .span_label(
                        span,
                        format!(
                            "attribute should be applied to a {}, not a {}",
                            expected_target, actual_target,
                        ),
                    )
                    .emit();
                }
                // Unknown lang item.
                _ => {
                    struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0522,
                        "definition of an unknown language item: `{}`",
                        value
                    )
                    .span_label(span, format!("definition of unknown language item `{}`", value))
                    .emit();
                }
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {
        // At present, lang items are always items, not trait items.
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {
        // At present, lang items are always items, not impl items.
    }
}

impl LanguageItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> LanguageItemCollector<'tcx> {
        LanguageItemCollector { tcx, items: LanguageItems::new() }
    }

    fn collect_item(&mut self, item_index: usize, item_def_id: DefId) {
        // Check for duplicates.
        if let Some(original_def_id) = self.items.items[item_index] {
            if original_def_id != item_def_id {
                let name = LangItem::from_u32(item_index as u32).unwrap().name();
                let mut err = match self.tcx.hir().span_if_local(item_def_id) {
                    Some(span) => struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0152,
                        "found duplicate lang item `{}`",
                        name
                    ),
                    None => match self.tcx.extern_crate(item_def_id) {
                        Some(ExternCrate { dependency_of, .. }) => {
                            self.tcx.sess.struct_err(&format!(
                                "duplicate lang item in crate `{}` (which `{}` depends on): `{}`.",
                                self.tcx.crate_name(item_def_id.krate),
                                self.tcx.crate_name(*dependency_of),
                                name
                            ))
                        }
                        _ => self.tcx.sess.struct_err(&format!(
                            "duplicate lang item in crate `{}`: `{}`.",
                            self.tcx.crate_name(item_def_id.krate),
                            name
                        )),
                    },
                };
                if let Some(span) = self.tcx.hir().span_if_local(original_def_id) {
                    err.span_note(span, "the lang item is first defined here");
                } else {
                    match self.tcx.extern_crate(original_def_id) {
                        Some(ExternCrate { dependency_of, .. }) => {
                            err.note(&format!(
                            "the lang item is first defined in crate `{}` (which `{}` depends on)",
                                      self.tcx.crate_name(original_def_id.krate),
                                      self.tcx.crate_name(*dependency_of)));
                        }
                        _ => {
                            err.note(&format!(
                                "the lang item is first defined in crate `{}`.",
                                self.tcx.crate_name(original_def_id.krate)
                            ));
                        }
                    }
                }
                err.emit();
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
    }
}

/// Extracts the first `lang = "$name"` out of a list of attributes.
/// The attributes `#[panic_handler]` and `#[alloc_error_handler]`
/// are also extracted out when found.
pub fn extract(attrs: &[ast::Attribute]) -> Option<(Symbol, Span)> {
    attrs.iter().find_map(|attr| {
        Some(match attr {
            _ if attr.check_name(sym::lang) => (attr.value_str()?, attr.span),
            _ if attr.check_name(sym::panic_handler) => (sym::panic_impl, attr.span),
            _ if attr.check_name(sym::alloc_error_handler) => (sym::oom, attr.span),
            _ => return None,
        })
    })
}

/// Traverses and collects all the lang items in all crates.
pub fn collect<'tcx>(tcx: TyCtxt<'tcx>) -> LanguageItems {
    // Initialize the collector.
    let mut collector = LanguageItemCollector::new(tcx);

    // Collect lang items in other crates.
    for &cnum in tcx.crates().iter() {
        for &(def_id, item_index) in tcx.defined_lang_items(cnum).iter() {
            collector.collect_item(item_index, def_id);
        }
    }

    // Collect lang items in this crate.
    tcx.hir().krate().visit_all_item_likes(&mut collector);

    // Extract out the found lang items.
    let LanguageItemCollector { mut items, .. } = collector;

    // Find all required but not-yet-defined lang items.
    weak_lang_items::check_crate(tcx, &mut items);

    items
}

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
