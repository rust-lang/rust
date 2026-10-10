//! Detecting lang items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use rustc_ast as ast;
use rustc_attr_ir::target::Target;
use rustc_attr_ir::{GenericRequirement, LangItem, LanguageItems};
use rustc_crate_store::ExternCrate;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::TyCtxt;
use rustc_span::{Span, Symbol, sym};
use rustc_structures::CrateType;

use crate::diagnostics::{DuplicateLangItem, IncorrectCrateType, IncorrectTarget};
use crate::late::{LateResolutionVisitor, RibKind};

pub(crate) enum Duplicate {
    Plain,
    Crate,
    CrateDepends,
}

impl<'tcx> LateResolutionVisitor<'_, '_, '_, 'tcx> {
    pub(super) fn check_for_lang(
        &mut self,
        actual_target: Target,
        def_id: LocalDefId,
        attrs: &[ast::Attribute],
        item_span: Span,
        generics: Option<&ast::Generics>,
    ) {
        if let Some((name, attr_span)) = extract_ast(attrs) {
            match LangItem::from_name(name) {
                // Known lang item
                Some(lang_item) => {
                    if actual_target != lang_item.target() {
                        // `#[panic_handler]` is turned into `#[lang = "panic_impl"]`, but in contrast
                        // to the actual lang item attr, is applied to `Fn` instead of `ForeignFn`.
                        if !(lang_item.is_weak()
                            && actual_target == Target::Fn
                            && lang_item.target() == Target::ForeignFn
                            && matches!(lang_item, LangItem::PanicImpl))
                        {
                            self.r.tcx
                            .dcx()
                            .delayed_bug(format!("lang item target is checked in attribute parser: {:?} has {} but expected {}", def_id, actual_target, lang_item.target()));
                            return;
                        }
                    }
                    // Weak lang items are handled separately
                    if lang_item.is_weak() && actual_target == Target::ForeignFn {
                        self.r.lang_items.missing.push(lang_item);
                    } else {
                        // Weak only lang items are always handled here
                        self.collect_item_extended(
                            lang_item,
                            def_id,
                            item_span,
                            attr_span,
                            generics,
                            actual_target,
                        );
                    }
                }
                // Unknown lang item.
                _ => {
                    self.r.tcx.dcx().delayed_bug("unknown lang item");
                }
            }
        }
    }
}

pub(crate) fn collect_item(
    tcx: TyCtxt<'_>,
    lang_items: &mut LanguageItems,
    lang_item: LangItem,
    item_def_id: DefId,
    item_span: Option<Span>,
) {
    // Check for duplicates.
    if let Some(original_def_id) = lang_items.get(lang_item)
        && original_def_id != item_def_id
    {
        let (item_def_id, original_def_id, item_span) = if let Some(original_def_id) =
            original_def_id.as_local()
            && !item_def_id.is_local()
        {
            // The crate-local definitions are loaded before the upstream crate definitions,
            // because we only know which crates are being loaded after the resolver is fully done.
            // In that case we swap the order of the items to report it meaningfully to the user.
            (original_def_id.to_def_id(), item_def_id, Some(tcx.source_span(original_def_id)))
        } else {
            (item_def_id, original_def_id, item_span)
        };
        let lang_item_name = lang_item.name();
        let crate_name = tcx.crate_name(item_def_id.krate);
        let mut dependency_of = None;
        let is_local = item_def_id.is_local();
        let path = if is_local {
            String::new()
        } else {
            tcx.crate_extern_paths(item_def_id.krate)
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        let mut orig_crate_name = None;
        let mut orig_dependency_of = None;
        let orig_is_local = original_def_id.is_local();
        let orig_path = if orig_is_local {
            String::new()
        } else {
            tcx.crate_extern_paths(original_def_id.krate)
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        if !original_def_id.is_local() {
            orig_crate_name = Some(tcx.crate_name(original_def_id.krate));
            if let Some(ExternCrate { dependency_of: inner_dependency_of, .. }) =
                tcx.extern_crate(original_def_id.krate)
            {
                orig_dependency_of = Some(tcx.crate_name(*inner_dependency_of));
            }
        }

        let duplicate = if item_span.is_some() {
            Duplicate::Plain
        } else {
            match tcx.extern_crate(item_def_id.krate) {
                Some(ExternCrate { dependency_of: inner_dependency_of, .. }) => {
                    dependency_of = Some(tcx.crate_name(*inner_dependency_of));
                    Duplicate::CrateDepends
                }
                _ => Duplicate::Crate,
            }
        };

        // When there's a duplicate lang item, something went very wrong and there's no value
        // in recovering or doing anything. Give the user the one message to let them debug the
        // mess they created and then wish them farewell.
        tcx.dcx().emit_fatal(DuplicateLangItem {
            local_span: item_span,
            lang_item_name,
            crate_name,
            dependency_of,
            is_local,
            path,
            first_defined_span: original_def_id.as_local().map(|did| tcx.source_span(did)),
            orig_crate_name,
            orig_dependency_of,
            orig_is_local,
            orig_path,
            duplicate,
        });
    } else {
        // Matched.
        lang_items.set(lang_item, item_def_id);
    }
}

impl<'tcx> LateResolutionVisitor<'_, '_, '_, 'tcx> {
    // Like collect_item() above, but also checks whether the lang item is declared
    // with the right number of generic arguments.
    fn collect_item_extended(
        &mut self,
        lang_item: LangItem,
        item_def_id: LocalDefId,
        item_span: Span,
        attr_span: Span,
        generics: Option<&ast::Generics>,
        target: Target,
    ) {
        let name = lang_item.name();

        if let Some(generics) = generics {
            // Now check whether the lang_item has the expected number of generic
            // arguments. Generally speaking, binary and indexing operations have
            // one (for the RHS/index), unary operations have none, the closure
            // traits have one for the argument list, coroutines have one for the
            // resume argument, and ordering/equality relations have one for the RHS
            // Some other types like Box and various unsizing-related traits
            // have minimum requirements.

            // FIXME: This still doesn't count, e.g., elided lifetimes and APITs.
            let mut actual_num = generics.params.len();
            if target.is_associated_item() {
                for rib in self.ribs.type_ns.iter().rev() {
                    match rib.kind {
                        RibKind::Item(_, _) => {
                            actual_num += rib.bindings.len();
                            break;
                        }
                        _ => {}
                    }
                }
            }

            let mut at_least = false;
            let required = match lang_item.required_generics() {
                GenericRequirement::Exact(num) if num != actual_num => Some(num),
                GenericRequirement::Minimum(num) if actual_num < num => {
                    at_least = true;
                    Some(num)
                }
                // If the number matches, or there is no requirement, handle it normally
                _ => None,
            };

            if let Some(num) = required {
                // We are issuing E0718 "incorrect target" here, because while the
                // item kind of the target is correct, the target is still wrong
                // because of the wrong number of generic arguments.
                self.r.tcx.dcx().emit_err(IncorrectTarget {
                    span: attr_span,
                    generics_span: generics.span,
                    name: name.as_str(),
                    kind: target.name(),
                    num,
                    actual_num,
                    at_least,
                });

                // return early to not collect the lang item
                return;
            }
        }

        if self.r.tcx.crate_types().contains(&CrateType::Sdylib) {
            self.r.tcx.dcx().emit_err(IncorrectCrateType { span: attr_span });
        }

        collect_item(
            self.r.tcx,
            &mut self.r.lang_items,
            lang_item,
            item_def_id.to_def_id(),
            Some(item_span),
        );
    }
}

/// Extracts the first `lang = "$name"` out of a list of attributes.
/// The `#[panic_handler]` attribute is also extracted out when found.
///
/// This function is used for `ast::Attribute`, for `hir::Attribute` use the `find_attr!` macro with `AttributeKind::Lang`
pub(crate) fn extract_ast(attrs: &[rustc_ast::ast::Attribute]) -> Option<(Symbol, Span)> {
    attrs.iter().find_map(|attr| {
        Some(match attr {
            _ if attr.has_name(sym::lang) => (attr.value_str()?, attr.span()),
            _ if attr.has_name(sym::panic_handler) => (sym::panic_impl, attr.span()),
            _ => return None,
        })
    })
}
