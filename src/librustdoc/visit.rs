use crate::clean::*;

/// Allows a type to traverse the cleaned ast of a crate.
///
/// Note that like [`rustc_ast::visit::Visitor`], but
/// unlike [`rustc_lint::EarlyLintPass`], if you override a
/// `visit_*` method, you will need to manually recurse into
/// its contents.
pub(crate) trait DocVisitor<'a>: Sized {
    fn visit_item(&mut self, item: &'a Item) {
        self.visit_item_recur(item)
    }

    /// Don't override!
    fn visit_inner_recur(&mut self, kind: &'a ItemKind) {
        match kind {
            StrippedItem(..) => unreachable!(),
            ModuleItem(i) => {
                self.visit_mod(i);
            }
            StructItem(i) => i.fields.iter().for_each(|x| self.visit_item(x)),
            UnionItem(i) => i.fields.iter().for_each(|x| self.visit_item(x)),
            EnumItem(i) => i.variants.iter().for_each(|x| self.visit_item(x)),
            TraitItem(i) => i.items.iter().for_each(|x| self.visit_item(x)),
            ImplItem(i) => i.items.iter().for_each(|x| self.visit_item(x)),
            VariantItem(i) => match &i.kind {
                VariantKind::Struct(j) => j.fields.iter().for_each(|x| self.visit_item(x)),
                VariantKind::Tuple(fields) => fields.iter().for_each(|x| self.visit_item(x)),
                VariantKind::CLike => {}
            },
            ExternCrateItem { src: _ }
            | ImportItem(_)
            | FunctionItem(_)
            | TypeAliasItem(_)
            | StaticItem(_)
            | ConstantItem(..)
            | TraitAliasItem(_)
            | RequiredMethodItem(_)
            | MethodItem(_, _)
            | StructFieldItem(_)
            | ForeignFunctionItem(..)
            | ForeignStaticItem(..)
            | ForeignTypeItem
            | MacroItem(_)
            | ProcMacroItem(_)
            | PrimitiveItem(_)
            | RequiredAssocConstItem(..)
            | ProvidedAssocConstItem(..)
            | ImplAssocConstItem(..)
            | RequiredAssocTypeItem(..)
            | AssocTypeItem(..)
            | KeywordItem
            | AttributeItem => {}
        }
    }

    /// Don't override!
    fn visit_item_recur(&mut self, item: &'a Item) {
        match &item.kind {
            StrippedItem(i) => self.visit_inner_recur(i),
            _ => self.visit_inner_recur(&item.kind),
        }
    }

    fn visit_mod(&mut self, m: &'a Module) {
        m.items.iter().for_each(|i| self.visit_item(i))
    }

    /// This is the main entrypoint of [`DocVisitor`].
    fn visit_crate(&mut self, c: &'a Crate) {
        self.visit_item(&c.module);

        for trait_ in c.external_traits.values() {
            trait_.items.iter().for_each(|i| self.visit_item(i));
        }
    }
}
