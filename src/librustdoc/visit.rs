use crate::clean::*;

crate trait DocVisitor: Sized {
    fn visit_item(&mut self, item: &Item) {
        self.visit_item_recur(item)
    }

    /// don't override!
    fn visit_inner_recur(&mut self, kind: &ItemKind) {
        match kind {
            StrippedItem(..) => unreachable!(),
            ModuleItem(i) => {
                self.visit_mod(i);
                return;
            }
            StructItem(i) => i.fields.iter().for_each(|x| self.visit_item(x)),
            UnionItem(i) => i.fields.iter().for_each(|x| self.visit_item(x)),
            EnumItem(i) => i.variants.iter().for_each(|x| self.visit_item(x)),
            TraitItem(i) => i.items.iter().for_each(|x| self.visit_item(x)),
            ImplItem(i) => i.items.iter().for_each(|x| self.visit_item(x)),
            VariantItem(i) => match i {
                Variant::Struct(j) => j.fields.iter().for_each(|x| self.visit_item(x)),
                Variant::Tuple(fields) => fields.iter().for_each(|x| self.visit_item(x)),
                Variant::CLike => {}
            },
            ExternCrateItem { src: _ }
            | ImportItem(_)
            | FunctionItem(_)
            | TypedefItem(_, _)
            | OpaqueTyItem(_)
            | StaticItem(_)
            | ConstantItem(_)
            | TraitAliasItem(_)
            | TyMethodItem(_)
            | MethodItem(_, _)
            | StructFieldItem(_)
            | ForeignFunctionItem(_)
            | ForeignStaticItem(_)
            | ForeignTypeItem
            | MacroItem(_)
            | ProcMacroItem(_)
            | PrimitiveItem(_)
            | AssocConstItem(_, _)
            | AssocTypeItem(_, _)
            | KeywordItem(_) => {}
        }
    }

    /// don't override!
    fn visit_item_recur(&mut self, item: &Item) {
        match &*item.kind {
            StrippedItem(i) => self.visit_inner_recur(i),
            _ => self.visit_inner_recur(&item.kind),
        }
    }

    fn visit_mod(&mut self, m: &Module) {
        m.items.iter().for_each(|i| self.visit_item(i))
    }

    fn visit_crate(&mut self, c: &Crate) {
        self.visit_item(&c.module);

        // FIXME: make this a simple by-ref for loop once external_traits is cleaned up
        let external_traits = { std::mem::take(&mut *c.external_traits.borrow_mut()) };
        for (k, v) in external_traits {
            v.trait_.items.iter().for_each(|i| self.visit_item(i));
            c.external_traits.borrow_mut().insert(k, v);
        }
    }
}
