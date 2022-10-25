use crate::errors::{
    IncorrectTarget, LangItemOnIncorrectTarget, UnknownExternLangItem, UnknownLangItem,
};
use crate::Resolver;

use rustc_ast::visit;
use rustc_ast::visit::{AssocCtxt, Visitor};
use rustc_ast::*;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::lang_items::{extract, GenericRequirement};
use rustc_hir::{LangItem, Target};

impl<'a> Resolver<'a> {
    pub(crate) fn resolve_lang_items(&mut self, krate: &Crate) {
        let mut collector =
            LanguageItemCollector { resolver: self, items: Vec::new(), missing: Vec::new() };
        visit::walk_crate(&mut collector, krate);
        let LanguageItemCollector { items, missing, .. } = collector;
        self.lang_items = items;
        self.missing_lang_items = missing;
    }
}
struct LanguageItemCollector<'a, 'b> {
    resolver: &'b Resolver<'a>,
    items: Vec<(LocalDefId, LangItem)>,
    missing: Vec<LangItem>,
}

impl<'ast> Visitor<'ast> for LanguageItemCollector<'_, '_> {
    fn visit_item(&mut self, item: &'ast Item) {
        self.check_for_lang(
            Target::from_ast_item(item),
            &item.attrs,
            item.id,
            item.kind.generics(),
            item.kind.descr(),
        );
        visit::walk_item(self, item);
    }

    fn visit_assoc_item(&mut self, item: &'ast AssocItem, ctxt: AssocCtxt) {
        let target = Target::from_ast_assoc_item(&item.kind, ctxt);
        self.check_for_lang(target, &item.attrs, item.id, item.kind.generics(), target.name());
        visit::walk_assoc_item(self, item, ctxt);
    }

    fn visit_variant(&mut self, variant: &'ast Variant) {
        self.check_for_lang(Target::Variant, &variant.attrs, variant.id, None, "variant");
        visit::walk_variant(self, variant);
    }

    fn visit_foreign_item(&mut self, item: &'ast ForeignItem) {
        let Some((lang_item, _)) = extract(&item.attrs) else { return };
        if let Some(item) = LangItem::from_name(lang_item) && item.is_weak() {
            self.missing.push(item);
        } else {
            self.resolver.session.emit_err(UnknownExternLangItem { span: item.span, lang_item });
        }
        visit::walk_foreign_item(self, item);
    }
}

impl LanguageItemCollector<'_, '_> {
    fn check_for_lang(
        &mut self,
        actual_target: Target,
        attrs: &[Attribute],
        id: NodeId,
        generics: Option<&Generics>,
        descr: &'static str,
    ) {
        let Some((name, span)) = extract(attrs) else { return };
        let Some(lang_item) = LangItem::from_name(name) else {
            self.resolver.session.emit_err(UnknownLangItem { span, name });
            return;
        };
        if actual_target != lang_item.target() {
            self.resolver.session.emit_err(LangItemOnIncorrectTarget {
                span,
                name,
                expected_target: lang_item.target(),
                actual_target,
            });
            return;
        }
        // Now check whether the lang_item has the expected number of generic
        // arguments. Generally speaking, binary and indexing operations have
        // one (for the RHS/index), unary operations have none, the closure
        // traits have one for the argument list, generators have one for the
        // resume argument, and ordering/equality relations have one for the RHS
        // Some other types like Box and various functions like drop_in_place
        // have minimum requirements.
        if let Some(generics) = generics {
            let actual_num = generics.params.len();
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
                self.resolver.session.emit_err(IncorrectTarget {
                    span,
                    generics_span: generics.span,
                    name: lang_item.name().as_str(),
                    kind: descr,
                    num,
                    actual_num,
                    at_least,
                });
                return;
            }
        }

        let item_def_id = self.resolver.local_def_id(id);
        self.items.push((item_def_id, lang_item));
    }
}
