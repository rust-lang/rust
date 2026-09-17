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
            ItemKind::Stripped(..) => unreachable!(),
            ItemKind::Module(i) => {
                self.visit_mod(i);
            }
            ItemKind::Struct(i) => i.fields.iter().for_each(|x| self.visit_item(x)),
            ItemKind::Union(i) => i.fields.iter().for_each(|x| self.visit_item(x)),
            ItemKind::Enum(i) => i.variants.iter().for_each(|x| self.visit_item(x)),
            ItemKind::Trait(i) => i.items.iter().for_each(|x| self.visit_item(x)),
            ItemKind::Impl(i) => i.items.iter().for_each(|x| self.visit_item(x)),
            ItemKind::Variant(i) => match &i.kind {
                VariantKind::Struct(j) => j.fields.iter().for_each(|x| self.visit_item(x)),
                VariantKind::Tuple(fields) => fields.iter().for_each(|x| self.visit_item(x)),
                VariantKind::CLike => {}
            },
            ItemKind::ExternCrate { src: _ }
            | ItemKind::Import(_)
            | ItemKind::Fn(_)
            | ItemKind::TyAlias(_)
            | ItemKind::Static(_)
            | ItemKind::Const(..)
            | ItemKind::TraitAlias(_)
            | ItemKind::RequiredAssocFn(..)
            | ItemKind::AssocFn(..)
            | ItemKind::StructField(_)
            | ItemKind::ForeignFn(..)
            | ItemKind::ForeignStatic(..)
            | ItemKind::ForeignTy
            | ItemKind::DeclMacro(..)
            | ItemKind::ProcMacro(_)
            | ItemKind::Primitive(_)
            | ItemKind::RequiredAssocConst(..)
            | ItemKind::ProvidedAssocConst(..)
            | ItemKind::ImplAssocConst(..)
            | ItemKind::RequiredAssocTy(..)
            | ItemKind::AssocTy(..)
            | ItemKind::Keyword
            | ItemKind::Attribute
            | ItemKind::PlaceholderImpl => {}
        }
    }

    /// Don't override!
    fn visit_item_recur(&mut self, item: &'a Item) {
        match &item.kind {
            ItemKind::Stripped(i) => self.visit_inner_recur(i),
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
