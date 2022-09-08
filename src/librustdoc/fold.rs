use crate::clean::*;

pub(crate) fn strip_item(mut item: Item) -> Item {
    if !matches!(*item.kind, StrippedItem(..)) {
        item.kind = Box::new(StrippedItem(item.kind));
    }
    item
}

pub(crate) trait DocFolder: Sized {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        Some(self.fold_item_recur(item))
    }

    /// don't override!
    fn fold_inner_recur(&mut self, kind: ItemKind) -> ItemKind {
        match kind {
            StrippedItem(..) => unreachable!(),
            ModuleItem(i) => ModuleItem(self.fold_mod(i)),
            StructItem(mut i) => {
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                StructItem(i)
            }
            UnionItem(mut i) => {
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                UnionItem(i)
            }
            EnumItem(mut i) => {
                i.variants = i.variants.into_iter().filter_map(|x| self.fold_item(x)).collect();
                EnumItem(i)
            }
            TraitItem(mut i) => {
                i.items = i.items.into_iter().filter_map(|x| self.fold_item(x)).collect();
                TraitItem(i)
            }
            ImplItem(mut i) => {
                i.items = i.items.into_iter().filter_map(|x| self.fold_item(x)).collect();
                ImplItem(i)
            }
            VariantItem(i) => match i {
                Variant::Struct(mut j) => {
                    j.fields = j.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                    VariantItem(Variant::Struct(j))
                }
                Variant::Tuple(fields) => {
                    let fields = fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                    VariantItem(Variant::Tuple(fields))
                }
                Variant::CLike(disr) => VariantItem(Variant::CLike(disr)),
            },
            ExternCrateItem { src: _ }
            | ImportItem(_)
            | FunctionItem(_)
            | TypedefItem(_)
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
            | TyAssocConstItem(..)
            | AssocConstItem(..)
            | TyAssocTypeItem(..)
            | AssocTypeItem(..)
            | KeywordItem => kind,
        }
    }

    /// don't override!
    fn fold_item_recur(&mut self, mut item: Item) -> Item {
        item.kind = Box::new(match *item.kind {
            StrippedItem(box i) => StrippedItem(Box::new(self.fold_inner_recur(i))),
            _ => self.fold_inner_recur(*item.kind),
        });
        item
    }

    fn fold_mod(&mut self, m: Module) -> Module {
        Module {
            span: m.span,
            items: m.items.into_iter().filter_map(|i| self.fold_item(i)).collect(),
        }
    }

    fn fold_crate(&mut self, mut c: Crate) -> Crate {
        c.module = self.fold_item(c.module).unwrap();

        let external_traits = { std::mem::take(&mut *c.external_traits.borrow_mut()) };
        for (k, mut v) in external_traits {
            v.trait_.items = v.trait_.items.into_iter().filter_map(|i| self.fold_item(i)).collect();
            c.external_traits.borrow_mut().insert(k, v);
        }

        c
    }
}
