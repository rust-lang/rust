//! Detecting language items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use crate::check_attr::target_from_impl_item;
use crate::weak_lang_items;

use rustc_middle::middle::cstore::ExternCrate;
use rustc_middle::ty::TyCtxt;

use rustc_errors::{pluralize, struct_span_err};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::lang_items::{extract, ITEM_REFS};
use rustc_hir::{HirId, LangItem, LanguageItems, Target};
use rustc_span::Span;

use rustc_middle::ty::query::Providers;

struct LanguageItemCollector<'tcx> {
    items: LanguageItems,
    tcx: TyCtxt<'tcx>,
}

impl ItemLikeVisitor<'v> for LanguageItemCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        self.check_for_lang(Target::from_item(item), item.hir_id());

        if let hir::ItemKind::Enum(def, ..) = &item.kind {
            for variant in def.variants {
                self.check_for_lang(Target::Variant, variant.id);
            }
        }
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem<'_>) {
        self.check_for_lang(Target::from_trait_item(trait_item), trait_item.hir_id())
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) {
        self.check_for_lang(target_from_impl_item(self.tcx, impl_item), impl_item.hir_id())
    }

    fn visit_foreign_item(&mut self, _: &hir::ForeignItem<'_>) {}
}

impl LanguageItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> LanguageItemCollector<'tcx> {
        LanguageItemCollector { tcx, items: LanguageItems::new() }
    }

    fn check_for_lang(&mut self, actual_target: Target, hir_id: HirId) {
        let attrs = self.tcx.hir().attrs(hir_id);
        let check_name = |attr, sym| self.tcx.sess.check_name(attr, sym);
        if let Some((value, span)) = extract(check_name, &attrs) {
            match ITEM_REFS.get(&value).cloned() {
                // Known lang item with attribute on correct target.
                Some((item_index, expected_target)) if actual_target == expected_target => {
                    self.collect_item_extended(item_index, hir_id, span);
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

    fn collect_item(&mut self, item_index: usize, item_def_id: DefId) {
        // Check for duplicates.
        if let Some(original_def_id) = self.items.items[item_index] {
            if original_def_id != item_def_id {
                let lang_item = LangItem::from_u32(item_index as u32).unwrap();
                let name = lang_item.name();
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
                    let mut note_def = |which, def_id: DefId| {
                        let crate_name = self.tcx.crate_name(def_id.krate);
                        let note = if def_id.is_local() {
                            format!("{} definition in the local crate (`{}`)", which, crate_name)
                        } else {
                            let paths: Vec<_> = self
                                .tcx
                                .crate_extern_paths(def_id.krate)
                                .iter()
                                .map(|p| p.display().to_string())
                                .collect();
                            format!(
                                "{} definition in `{}` loaded from {}",
                                which,
                                crate_name,
                                paths.join(", ")
                            )
                        };
                        err.note(&note);
                    };
                    note_def("first", original_def_id);
                    note_def("second", item_def_id);
                }
                err.emit();
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
        if let Some(group) = LangItem::from_u32(item_index as u32).unwrap().group() {
            self.items.groups[group as usize].push(item_def_id);
        }
    }

    // Like collect_item() above, but also checks whether the lang item is declared
    // with the right number of generic arguments if it is a trait.
    fn collect_item_extended(&mut self, item_index: usize, hir_id: HirId, span: Span) {
        let item_def_id = self.tcx.hir().local_def_id(hir_id).to_def_id();
        let lang_item = LangItem::from_u32(item_index as u32).unwrap();
        let name = lang_item.name();

        self.collect_item(item_index, item_def_id);

        // Now check whether the lang_item has the expected number of generic
        // arguments if it is a trait. Generally speaking, binary and indexing
        // operations have one (for the RHS/index), unary operations have none,
        // and the rest also have none except for the closure traits (one for
        // the argument list), generators (one for the resume argument),
        // ordering/equality relations (one for the RHS), and various conversion
        // traits.

        let expected_num = match lang_item {
            // Binary operations
            LangItem::Add
            | LangItem::Sub
            | LangItem::Mul
            | LangItem::Div
            | LangItem::Rem
            | LangItem::BitXor
            | LangItem::BitAnd
            | LangItem::BitOr
            | LangItem::Shl
            | LangItem::Shr
            | LangItem::AddAssign
            | LangItem::SubAssign
            | LangItem::MulAssign
            | LangItem::DivAssign
            | LangItem::RemAssign
            | LangItem::BitXorAssign
            | LangItem::BitAndAssign
            | LangItem::BitOrAssign
            | LangItem::ShlAssign
            | LangItem::ShrAssign
            | LangItem::Index
            | LangItem::IndexMut

            // Miscellaneous
            | LangItem::Unsize
            | LangItem::CoerceUnsized
            | LangItem::DispatchFromDyn
            | LangItem::Fn
            | LangItem::FnMut
            | LangItem::FnOnce
            | LangItem::Generator
            | LangItem::PartialEq
            | LangItem::PartialOrd
                => Some(1),

            // Unary operations
            LangItem::Neg
            | LangItem::Not

            // Miscellaneous
            | LangItem::Deref
            | LangItem::DerefMut
            | LangItem::Sized
            | LangItem::StructuralPeq
            | LangItem::StructuralTeq
            | LangItem::Copy
            | LangItem::Clone
            | LangItem::Sync
            | LangItem::DiscriminantKind
            | LangItem::PointeeTrait
            | LangItem::Freeze
            | LangItem::Drop
            | LangItem::Receiver
            | LangItem::Future
            | LangItem::Unpin
            | LangItem::Termination
            | LangItem::Try
            | LangItem::Send
            | LangItem::UnwindSafe
            | LangItem::RefUnwindSafe
                => Some(0),

            // Not a trait
            _ => None,
        };

        if let Some(expected_num) = expected_num {
            let (actual_num, generics_span) = match self.tcx.hir().get(hir_id) {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Trait(_, _, generics, ..),
                    ..
                }) => (generics.params.len(), generics.span),
                _ => bug!("op/index/deref lang item target is not a trait: {:?}", lang_item),
            };

            if expected_num != actual_num {
                // We are issuing E0718 "incorrect target" here, because while the
                // item kind of the target is correct, the target is still wrong
                // because of the wrong number of generic arguments.
                struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0718,
                    "`{}` language item must be applied to a trait with {} generic argument{}",
                    name,
                    expected_num,
                    pluralize!(expected_num)
                )
                .span_label(
                    generics_span,
                    format!(
                        "this trait has {} generic argument{}, not {}",
                        actual_num,
                        pluralize!(actual_num),
                        expected_num
                    ),
                )
                .emit();
            }
        }
    }
}

/// Traverses and collects all the lang items in all crates.
fn get_lang_items(tcx: TyCtxt<'_>, (): ()) -> LanguageItems {
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

pub fn provide(providers: &mut Providers) {
    providers.get_lang_items = get_lang_items;
}
