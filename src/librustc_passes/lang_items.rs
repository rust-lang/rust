//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::weak_lang_items;

use rustc::middle::cstore::ExternCrate;
use rustc::ty::TyCtxt;

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::lang_items::{extract, ITEM_REFS};
use rustc_hir::{LangItem, LanguageItems, Target};

use rustc::ty::query::Providers;

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
                                self.tcx.crate_name(*dependency_of)
                            ));
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

/// Traverses and collects all the lang items in all crates.
fn collect(tcx: TyCtxt<'_>) -> LanguageItems {
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

pub fn provide(providers: &mut Providers<'_>) {
    providers.get_lang_items = |tcx, id| {
        assert_eq!(id, LOCAL_CRATE);
        tcx.arena.alloc(collect(tcx))
    };
}
