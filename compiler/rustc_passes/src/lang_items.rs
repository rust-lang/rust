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
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::{GenericRequirement, extract};
use rustc_hir::{LangItem, LanguageItems, MethodKind, Target};
use rustc_middle::query::Providers;
use rustc_middle::ty::{ResolverAstLowering, TyCtxt};
use rustc_session::cstore::ExternCrate;
use rustc_span::Span;

use crate::errors::{
    DuplicateLangItem, IncorrectCrateType, IncorrectTarget, LangItemOnIncorrectTarget,
    UnknownLangItem,
};
use crate::weak_lang_items;

pub(crate) enum Duplicate {
    Plain,
    Crate,
    CrateDepends,
}

struct LanguageItemCollector<'ast, 'tcx> {
    items: LanguageItems,
    tcx: TyCtxt<'tcx>,
    resolver: &'ast ResolverAstLowering,
    // FIXME(#118552): We should probably feed def_span eagerly on def-id creation
    // so we can avoid constructing this map for local def-ids.
    item_spans: FxHashMap<DefId, Span>,
    parent_item: Option<&'ast ast::Item>,
}

impl<'ast, 'tcx> LanguageItemCollector<'ast, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        resolver: &'ast ResolverAstLowering,
    ) -> LanguageItemCollector<'ast, 'tcx> {
        LanguageItemCollector {
            tcx,
            resolver,
            items: LanguageItems::new(),
            item_spans: FxHashMap::default(),
            parent_item: None,
        }
    }

    fn check_for_lang(
        &mut self,
        actual_target: Target,
        def_id: LocalDefId,
        attrs: &'ast [ast::Attribute],
        item_span: Span,
        generics: Option<&'ast ast::Generics>,
    ) {
        if let Some((name, attr_span)) = extract(attrs) {
            match LangItem::from_name(name) {
                // Known lang item with attribute on correct target.
                Some(lang_item) if actual_target == lang_item.target() => {
                    self.collect_item_extended(
                        lang_item,
                        def_id,
                        item_span,
                        attr_span,
                        generics,
                        actual_target,
                    );
                }
                // Known lang item with attribute on incorrect target.
                Some(lang_item) => {
                    self.tcx.dcx().emit_err(LangItemOnIncorrectTarget {
                        span: attr_span,
                        name,
                        expected_target: lang_item.target(),
                        actual_target,
                    });
                }
                // Unknown lang item.
                _ => {
                    self.tcx.dcx().emit_err(UnknownLangItem { span: attr_span, name });
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

            let first_defined_span = self.item_spans.get(&original_def_id).copied();
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

            if first_defined_span.is_none() {
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
                first_defined_span,
                orig_crate_name,
                orig_dependency_of,
                orig_is_local,
                orig_path,
                duplicate,
            });
        } else {
            // Matched.
            self.items.set(lang_item, item_def_id);
            // Collect span for error later
            if let Some(item_span) = item_span {
                self.item_spans.insert(item_def_id, item_span);
            }
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
            // Some other types like Box and various functions like drop_in_place
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
    let resolver = tcx.resolver_for_lowering().borrow();
    let (resolver, krate) = &*resolver;

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
    weak_lang_items::check_crate(tcx, &mut collector.items, krate);

    // Return all the lang items that were found.
    collector.items
}

impl<'ast, 'tcx> visit::Visitor<'ast> for LanguageItemCollector<'ast, 'tcx> {
    fn visit_item(&mut self, i: &'ast ast::Item) {
        let target = match &i.kind {
            ast::ItemKind::ExternCrate(..) => Target::ExternCrate,
            ast::ItemKind::Use(_) => Target::Use,
            ast::ItemKind::Static(_) => Target::Static,
            ast::ItemKind::Const(_) => Target::Const,
            ast::ItemKind::Fn(_) | ast::ItemKind::Delegation(..) => Target::Fn,
            ast::ItemKind::Mod(..) => Target::Mod,
            ast::ItemKind::ForeignMod(_) => Target::ForeignFn,
            ast::ItemKind::GlobalAsm(_) => Target::GlobalAsm,
            ast::ItemKind::TyAlias(_) => Target::TyAlias,
            ast::ItemKind::Enum(..) => Target::Enum,
            ast::ItemKind::Struct(..) => Target::Struct,
            ast::ItemKind::Union(..) => Target::Union,
            ast::ItemKind::Trait(_) => Target::Trait,
            ast::ItemKind::TraitAlias(..) => Target::TraitAlias,
            ast::ItemKind::Impl(_) => Target::Impl,
            ast::ItemKind::MacroDef(..) => Target::MacroDef,
            ast::ItemKind::MacCall(_) | ast::ItemKind::DelegationMac(_) => {
                unreachable!("macros should have been expanded")
            }
        };

        self.check_for_lang(
            target,
            self.resolver.node_id_to_def_id[&i.id],
            &i.attrs,
            i.span,
            i.opt_generics(),
        );

        let parent_item = self.parent_item.replace(i);
        visit::walk_item(self, i);
        self.parent_item = parent_item;
    }

    fn visit_variant(&mut self, variant: &'ast ast::Variant) {
        self.check_for_lang(
            Target::Variant,
            self.resolver.node_id_to_def_id[&variant.id],
            &variant.attrs,
            variant.span,
            None,
        );
    }

    fn visit_assoc_item(&mut self, i: &'ast ast::AssocItem, ctxt: visit::AssocCtxt) {
        let (target, generics) = match &i.kind {
            ast::AssocItemKind::Fn(..) | ast::AssocItemKind::Delegation(..) => {
                let (body, generics) = if let ast::AssocItemKind::Fn(fun) = &i.kind {
                    (fun.body.is_some(), Some(&fun.generics))
                } else {
                    (true, None)
                };
                (
                    match &self.parent_item.unwrap().kind {
                        ast::ItemKind::Impl(i) => {
                            if i.of_trait.is_some() {
                                Target::Method(MethodKind::Trait { body })
                            } else {
                                Target::Method(MethodKind::Inherent)
                            }
                        }
                        ast::ItemKind::Trait(_) => Target::Method(MethodKind::Trait { body }),
                        _ => unreachable!(),
                    },
                    generics,
                )
            }
            ast::AssocItemKind::Const(ct) => (Target::AssocConst, Some(&ct.generics)),
            ast::AssocItemKind::Type(ty) => (Target::AssocTy, Some(&ty.generics)),
            ast::AssocItemKind::MacCall(_) | ast::AssocItemKind::DelegationMac(_) => {
                unreachable!("macros should have been expanded")
            }
        };

        self.check_for_lang(
            target,
            self.resolver.node_id_to_def_id[&i.id],
            &i.attrs,
            i.span,
            generics,
        );

        visit::walk_assoc_item(self, i, ctxt);
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.get_lang_items = get_lang_items;
}
