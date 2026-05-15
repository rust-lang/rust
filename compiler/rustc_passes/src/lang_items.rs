//! Detecting lang items.
//!
//! Language items are items that represent concepts intrinsic to the language
//! itself. Examples are:
//!
//! * Traits that specify "kinds"; e.g., `Sync`, `Send`.
//! * Traits that represent operators; e.g., `Add`, `Sub`, `Index`.
//! * Functions called by the compiler itself.

use rustc_ast as ast;
use rustc_ast::visit;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::GenericRequirement;
use rustc_hir::{LangItem, LanguageItems, Target};
use rustc_middle::query::Providers;
use rustc_middle::ty::{ResolverAstLowering, TyCtxt};
use rustc_session::cstore::ExternCrate;
use rustc_span::{Span, Symbol, sym};

use crate::diagnostics::{DuplicateLangItem, IncorrectCrateType, IncorrectTarget};
use crate::weak_lang_items;

pub(crate) enum Duplicate {
    Plain,
    Crate,
    CrateDepends,
}

struct LanguageItemCollector<'ast, 'tcx> {
    items: LanguageItems,
    tcx: TyCtxt<'tcx>,
    resolver: &'ast ResolverAstLowering<'tcx>,
    parent_item: Option<&'ast ast::Item>,
}

impl<'ast, 'tcx> LanguageItemCollector<'ast, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        resolver: &'ast ResolverAstLowering<'tcx>,
    ) -> LanguageItemCollector<'ast, 'tcx> {
        LanguageItemCollector { tcx, resolver, items: LanguageItems::new(), parent_item: None }
    }

    fn check_for_lang(
        &mut self,
        actual_target: Target,
        def_id: LocalDefId,
        attrs: &'ast [ast::Attribute],
        item_span: Span,
        generics: Option<&'ast ast::Generics>,
    ) {
        if let Some((name, attr_span)) = extract_ast(attrs) {
            match LangItem::from_name(name) {
                // Known lang item
                Some(lang_item) => {
                    if actual_target != lang_item.target() {
                        self.tcx
                            .dcx()
                            .delayed_bug(format!("lang item target is checked in attribute parser: {:?} has {} but expected {}", def_id, actual_target, lang_item.target()));
                        return;
                    }
                    // Weak lang items are handled separately
                    if lang_item.is_weak() && actual_target == Target::ForeignFn {
                        self.items.missing.push(lang_item);
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
                    self.tcx.dcx().delayed_bug("unknown lang item");
                }
            }
        }
    }

    fn collect_item(&mut self, lang_item: LangItem, item_def_id: DefId, item_span: Option<Span>) {
        // Check for duplicates.
        if let Some(original_def_id) = self.items.get(lang_item)
            && original_def_id != item_def_id
        {
            let lang_item_name = lang_item.name();
            let crate_name = self.tcx.crate_name(item_def_id.krate);
            let mut dependency_of = None;
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
            };

            let mut orig_crate_name = None;
            let mut orig_dependency_of = None;
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
            };

            if !original_def_id.is_local() {
                orig_crate_name = Some(self.tcx.crate_name(original_def_id.krate));
                if let Some(ExternCrate { dependency_of: inner_dependency_of, .. }) =
                    self.tcx.extern_crate(original_def_id.krate)
                {
                    orig_dependency_of = Some(self.tcx.crate_name(*inner_dependency_of));
                }
            }

            let duplicate = if item_span.is_some() {
                Duplicate::Plain
            } else {
                match self.tcx.extern_crate(item_def_id.krate) {
                    Some(ExternCrate { dependency_of: inner_dependency_of, .. }) => {
                        dependency_of = Some(self.tcx.crate_name(*inner_dependency_of));
                        Duplicate::CrateDepends
                    }
                    _ => Duplicate::Crate,
                }
            };

            // When there's a duplicate lang item, something went very wrong and there's no value
            // in recovering or doing anything. Give the user the one message to let them debug the
            // mess they created and then wish them farewell.
            self.tcx.dcx().emit_fatal(DuplicateLangItem {
                local_span: item_span,
                lang_item_name,
                crate_name,
                dependency_of,
                is_local,
                path,
                first_defined_span: original_def_id.as_local().map(|did| self.tcx.source_span(did)),
                orig_crate_name,
                orig_dependency_of,
                orig_is_local,
                orig_path,
                duplicate,
            });
        } else {
            // Matched.
            self.items.set(lang_item, item_def_id);
        }
    }

    // Like collect_item() above, but also checks whether the lang item is declared
    // with the right number of generic arguments.
    fn collect_item_extended(
        &mut self,
        lang_item: LangItem,
        item_def_id: LocalDefId,
        item_span: Span,
        attr_span: Span,
        generics: Option<&'ast ast::Generics>,
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
                actual_num += self
                    .parent_item
                    .unwrap()
                    .opt_generics()
                    .map_or(0, |generics| generics.params.len());
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
                self.tcx.dcx().emit_err(IncorrectTarget {
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

        if self.tcx.crate_types().contains(&rustc_session::config::CrateType::Sdylib) {
            self.tcx.dcx().emit_err(IncorrectCrateType { span: attr_span });
        }

        self.collect_item(lang_item, item_def_id.to_def_id(), Some(item_span));
    }
}

/// Traverses and collects all the lang items in all crates.
fn get_lang_items(tcx: TyCtxt<'_>, (): ()) -> LanguageItems {
    let (resolver, krate) = tcx.resolver_for_lowering();
    let resolver = &*resolver.borrow();
    let krate = &*krate.borrow();

    // Initialize the collector.
    let mut collector = LanguageItemCollector::new(tcx, resolver);

    // Collect lang items in other crates.
    for &cnum in tcx.used_crates(()).iter() {
        for &(def_id, lang_item) in tcx.defined_lang_items(cnum).iter() {
            collector.collect_item(lang_item, def_id, None);
        }
    }

    // Collect lang items local to this crate.
    visit::Visitor::visit_crate(&mut collector, krate);

    // Find all required but not-yet-defined lang items.
    weak_lang_items::check_crate(tcx, &mut collector.items);

    // Return all the lang items that were found.
    collector.items
}

impl<'ast, 'tcx> visit::Visitor<'ast> for LanguageItemCollector<'ast, 'tcx> {
    fn visit_item(&mut self, i: &'ast ast::Item) {
        let target = Target::from_ast_item(i);

        self.check_for_lang(
            target,
            self.resolver.owners[&i.id].def_id,
            &i.attrs,
            i.span,
            i.opt_generics(),
        );

        let parent_item = self.parent_item.replace(i);
        visit::walk_item(self, i);
        self.parent_item = parent_item;
    }

    fn visit_foreign_item(&mut self, i: &'ast ast::ForeignItem) {
        self.check_for_lang(
            Target::from_foreign_item_kind(&i.kind),
            self.resolver.owners[&i.id].def_id,
            &i.attrs,
            i.span,
            None,
        );
    }

    fn visit_variant(&mut self, variant: &'ast ast::Variant) {
        self.check_for_lang(
            Target::Variant,
            self.resolver.owners[&self.parent_item.unwrap().id].node_id_to_def_id[&variant.id],
            &variant.attrs,
            variant.span,
            None,
        );
    }

    fn visit_assoc_item(&mut self, i: &'ast ast::AssocItem, ctxt: visit::AssocCtxt) {
        let target = Target::from_assoc_item_kind(&i.kind, ctxt);
        let generics = i.opt_generics();

        self.check_for_lang(target, self.resolver.owners[&i.id].def_id, &i.attrs, i.span, generics);

        visit::walk_assoc_item(self, i, ctxt);
    }
}

/// Extracts the first `lang = "$name"` out of a list of attributes.
///
/// This function is used for `ast::Attribute`, for `hir::Attribute` use the `find_attr!` macro with `AttributeKind::Lang`
pub(crate) fn extract_ast(attrs: &[rustc_ast::ast::Attribute]) -> Option<(Symbol, Span)> {
    attrs.iter().find_map(|attr| {
        Some(match attr {
            _ if attr.has_name(sym::lang) => (attr.value_str()?, attr.span()),
            _ => return None,
        })
    })
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.get_lang_items = get_lang_items;
}
