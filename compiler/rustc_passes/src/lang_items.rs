//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::check_attr::target_from_impl_item;
use crate::errors::{
    DuplicateLangItem, IncorrectTarget, LangItemOnIncorrectTarget, UnknownLangItem,
};
use crate::weak_lang_items;

use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::{extract, GenericRequirement, ITEM_REFS};
use rustc_hir::{HirId, LangItem, LanguageItems, Target};
use rustc_middle::ty::TyCtxt;
use rustc_session::cstore::ExternCrate;
use rustc_span::{symbol::kw::Empty, Span};

use rustc_middle::ty::query::Providers;

pub(crate) enum Duplicate {
    Plain,
    Crate,
    CrateDepends,
}

struct LanguageItemCollector<'tcx> {
    items: LanguageItems,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> LanguageItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> LanguageItemCollector<'tcx> {
        LanguageItemCollector { tcx, items: LanguageItems::new() }
    }

    fn check_for_lang(&mut self, actual_target: Target, hir_id: HirId) {
        let attrs = self.tcx.hir().attrs(hir_id);
        if let Some((name, span)) = extract(&attrs) {
            match ITEM_REFS.get(&name).cloned() {
                // Known lang item with attribute on correct target.
                Some((item_index, expected_target)) if actual_target == expected_target => {
                    self.collect_item_extended(item_index, hir_id, span);
                }
                // Known lang item with attribute on incorrect target.
                Some((_, expected_target)) => {
                    self.tcx.sess.emit_err(LangItemOnIncorrectTarget {
                        span,
                        name,
                        expected_target,
                        actual_target,
                    });
                }
                // Unknown lang item.
                _ => {
                    self.tcx.sess.emit_err(UnknownLangItem { span, name });
                }
            }
        }
    }

    fn collect_item(&mut self, item_index: usize, item_def_id: DefId) {
        // Check for duplicates.
        if let Some(original_def_id) = self.items.items[item_index] {
            if original_def_id != item_def_id {
                let local_span = self.tcx.hir().span_if_local(item_def_id);
                let lang_item_name = LangItem::from_u32(item_index as u32).unwrap().name();
                let crate_name = self.tcx.crate_name(item_def_id.krate);
                let mut dependency_of = Empty;
                let is_local = item_def_id.is_local();
                let path = if is_local {
                    String::new()
                } else {
                    self.tcx
                        .crate_extern_paths(item_def_id.krate)
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                        .into()
                };
                let first_defined_span = self.tcx.hir().span_if_local(original_def_id);
                let mut orig_crate_name = Empty;
                let mut orig_dependency_of = Empty;
                let orig_is_local = original_def_id.is_local();
                let orig_path = if orig_is_local {
                    String::new()
                } else {
                    self.tcx
                        .crate_extern_paths(original_def_id.krate)
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                        .into()
                };
                if first_defined_span.is_none() {
                    orig_crate_name = self.tcx.crate_name(original_def_id.krate);
                    if let Some(ExternCrate { dependency_of: inner_dependency_of, .. }) =
                        self.tcx.extern_crate(original_def_id)
                    {
                        orig_dependency_of = self.tcx.crate_name(*inner_dependency_of);
                    }
                }

                let duplicate = if local_span.is_some() {
                    Duplicate::Plain
                } else {
                    match self.tcx.extern_crate(item_def_id) {
                        Some(ExternCrate { dependency_of: inner_dependency_of, .. }) => {
                            dependency_of = self.tcx.crate_name(*inner_dependency_of);
                            Duplicate::CrateDepends
                        }
                        _ => Duplicate::Crate,
                    }
                };

                self.tcx.sess.emit_err(DuplicateLangItem {
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
        self.items.items[item_index] = Some(item_def_id);
        if let Some(group) = LangItem::from_u32(item_index as u32).unwrap().group() {
            self.items.groups[group as usize].push(item_def_id);
        }
    }

    // Like collect_item() above, but also checks whether the lang item is declared
    // with the right number of generic arguments.
    fn collect_item_extended(&mut self, item_index: usize, hir_id: HirId, span: Span) {
        let item_def_id = self.tcx.hir().local_def_id(hir_id).to_def_id();
        let lang_item = LangItem::from_u32(item_index as u32).unwrap();
        let name = lang_item.name();

        // Now check whether the lang_item has the expected number of generic
        // arguments. Generally speaking, binary and indexing operations have
        // one (for the RHS/index), unary operations have none, the closure
        // traits have one for the argument list, generators have one for the
        // resume argument, and ordering/equality relations have one for the RHS
        // Some other types like Box and various functions like drop_in_place
        // have minimum requirements.

        if let hir::Node::Item(hir::Item { kind, span: item_span, .. }) = self.tcx.hir().get(hir_id)
        {
            let (actual_num, generics_span) = match kind.generics() {
                Some(generics) => (generics.params.len(), generics.span),
                None => (0, *item_span),
            };

            let mut at_least = false;
            let required = match lang_item.required_generics() {
                GenericRequirement::Exact(num) if num != actual_num => Some(num),
                GenericRequirement::Minimum(num) if actual_num < num => {
                    at_least = true;
                    Some(num)}
                ,
                // If the number matches, or there is no requirement, handle it normally
                _ => None,
            };

            if let Some(num) = required {
                // We are issuing E0718 "incorrect target" here, because while the
                // item kind of the target is correct, the target is still wrong
                // because of the wrong number of generic arguments.
                self.tcx.sess.emit_err(IncorrectTarget {
                    span,
                    generics_span,
                    name: name.as_str(),
                    kind: kind.descr(),
                    num,
                    actual_num,
                    at_least,
                });

                // return early to not collect the lang item
                return;
            }
        }

        self.collect_item(item_index, item_def_id);
    }
}

/// Traverses and collects all the lang items in all crates.
fn get_lang_items(tcx: TyCtxt<'_>, (): ()) -> LanguageItems {
    // Initialize the collector.
    let mut collector = LanguageItemCollector::new(tcx);

    // Collect lang items in other crates.
    for &cnum in tcx.crates(()).iter() {
        for &(def_id, item_index) in tcx.defined_lang_items(cnum).iter() {
            collector.collect_item(item_index, def_id);
        }
    }

    // Collect lang items in this crate.
    let crate_items = tcx.hir_crate_items(());

    for id in crate_items.items() {
        collector.check_for_lang(Target::from_def_kind(tcx.def_kind(id.def_id)), id.hir_id());

        if matches!(tcx.def_kind(id.def_id), DefKind::Enum) {
            let item = tcx.hir().item(id);
            if let hir::ItemKind::Enum(def, ..) = &item.kind {
                for variant in def.variants {
                    collector.check_for_lang(Target::Variant, variant.id);
                }
            }
        }
    }

    // FIXME: avoid calling trait_item() when possible
    for id in crate_items.trait_items() {
        let item = tcx.hir().trait_item(id);
        collector.check_for_lang(Target::from_trait_item(item), item.hir_id())
    }

    // FIXME: avoid calling impl_item() when possible
    for id in crate_items.impl_items() {
        let item = tcx.hir().impl_item(id);
        collector.check_for_lang(target_from_impl_item(tcx, item), item.hir_id())
    }

    // Extract out the found lang items.
    let LanguageItemCollector { mut items, .. } = collector;

    // Find all required but not-yet-defined lang items.
    weak_lang_items::check_crate(tcx, &mut items);

    items
}

pub fn provide(providers: &mut Providers) {
    providers.get_lang_items = get_lang_items;
}
