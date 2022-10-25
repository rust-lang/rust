//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::errors::DuplicateLangItem;
use crate::weak_lang_items;

use rustc_hir::def_id::DefId;
use rustc_hir::{LangItem, LanguageItems};
use rustc_middle::ty::TyCtxt;
use rustc_session::cstore::ExternCrate;
use rustc_span::symbol::kw::Empty;

use rustc_middle::ty::query::Providers;

pub(crate) enum Duplicate {
    Plain,
    Crate,
    CrateDepends,
}

fn collect_item(
    tcx: TyCtxt<'_>,
    items: &mut LanguageItems,
    lang_item: LangItem,
    item_def_id: DefId,
) {
    // Check for duplicates.
    if let Some(original_def_id) = items.get(lang_item) {
        if original_def_id != item_def_id {
            let local_span = item_def_id.as_local().map(|id| tcx.source_span(id));
            let lang_item_name = lang_item.name();
            let crate_name = tcx.crate_name(item_def_id.krate);
            let mut dependency_of = Empty;
            let is_local = item_def_id.is_local();
            let path = if is_local {
                String::new()
            } else {
                tcx.crate_extern_paths(item_def_id.krate)
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
                    .into()
            };
            let first_defined_span = original_def_id.as_local().map(|id| tcx.source_span(id));
            let mut orig_crate_name = Empty;
            let mut orig_dependency_of = Empty;
            let orig_is_local = original_def_id.is_local();
            let orig_path = if orig_is_local {
                String::new()
            } else {
                tcx.crate_extern_paths(original_def_id.krate)
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
                    .into()
            };
            if first_defined_span.is_none() {
                orig_crate_name = tcx.crate_name(original_def_id.krate);
                if let Some(ExternCrate { dependency_of: inner_dependency_of, .. }) =
                    tcx.extern_crate(original_def_id)
                {
                    orig_dependency_of = tcx.crate_name(*inner_dependency_of);
                }
            }

            let duplicate = if local_span.is_some() {
                Duplicate::Plain
            } else {
                match tcx.extern_crate(item_def_id) {
                    Some(ExternCrate { dependency_of: inner_dependency_of, .. }) => {
                        dependency_of = tcx.crate_name(*inner_dependency_of);
                        Duplicate::CrateDepends
                    }
                    _ => Duplicate::Crate,
                }
            };

            tcx.sess.emit_err(DuplicateLangItem {
                local_span,
                lang_item_name,
                crate_name,
                dependency_of,
                is_local,
                path,
                first_defined_span,
                orig_crate_name,
                orig_dependency_of,
                orig_is_local,
                orig_path,
                duplicate,
            });
        }
    }

    // Matched.
    items.set(lang_item, item_def_id);
}

/// Traverses and collects all the lang items in all crates.
fn get_lang_items(tcx: TyCtxt<'_>, (): ()) -> LanguageItems {
    let mut items = LanguageItems::new();

    // Collect lang items in other crates.
    for &cnum in tcx.crates(()).iter() {
        for &(def_id, lang_item) in tcx.defined_lang_items(cnum).iter() {
            collect_item(tcx, &mut items, lang_item, def_id);
        }
    }

    // Collect lang items from this crate
    for &(def_id, lang_item) in &tcx.resolutions(()).lang_items {
        collect_item(tcx, &mut items, lang_item, def_id.to_def_id());
    }

    // Find all required but not-yet-defined lang items.
    weak_lang_items::check_crate(tcx, &mut items);

    items
}

pub fn provide(providers: &mut Providers) {
    providers.get_lang_items = get_lang_items;
}
