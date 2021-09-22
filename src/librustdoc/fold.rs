use crate::clean::*;

crate fn strip_item(mut item: Item) -> Item {
    if !matches!(*item.kind, StrippedItem(..)) {
        item.kind = box StrippedItem(item.kind);
    }
    item
}

crate trait DocFolder: Sized {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        Some(self.fold_item_recur(item))
    }

    /// don't override!
    fn fold_inner_recur(&mut self, kind: ItemKind) -> ItemKind {
        match kind {
            StrippedItem(..) => unreachable!(),
            ModuleItem(i) => ModuleItem(self.fold_mod(i)),
            StructItem(mut i) => {
                let num_fields = i.fields.len();
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                i.fields_stripped |=
                    num_fields != i.fields.len() || i.fields.iter().any(|f| f.is_stripped());
                StructItem(i)
            }
            UnionItem(mut i) => {
                let num_fields = i.fields.len();
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                i.fields_stripped |=
                    num_fields != i.fields.len() || i.fields.iter().any(|f| f.is_stripped());
                UnionItem(i)
            }
            EnumItem(mut i) => {
                let num_variants = i.variants.len();
                i.variants = i.variants.into_iter().filter_map(|x| self.fold_item(x)).collect();
                i.variants_stripped |=
                    num_variants != i.variants.len() || i.variants.iter().any(|f| f.is_stripped());
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
            VariantItem(i) => {
                let i2 = i.clone(); // this clone is small
                match i {
                    Variant::Struct(mut j) => {
                        let num_fields = j.fields.len();
                        j.fields = j.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        j.fields_stripped |= num_fields != j.fields.len()
                            || j.fields.iter().any(|f| f.is_stripped());
                        VariantItem(Variant::Struct(j))
                    }
                    Variant::Tuple(fields) => {
                        let fields = fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        VariantItem(Variant::Tuple(fields))
                    }
                    _ => VariantItem(i2),
                }
            }
            x => x,
        }
    }

    /// don't override!
    fn fold_item_recur(&mut self, mut item: Item) -> Item {
        item.kind = box match *item.kind {
            StrippedItem(box i) => StrippedItem(box self.fold_inner_recur(i)),
            _ => self.fold_inner_recur(*item.kind),
        };
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

        {
            let external_traits = { std::mem::take(&mut *c.external_traits.borrow_mut()) };
            for (k, mut v) in external_traits {
                v.trait_.items =
                    v.trait_.items.into_iter().filter_map(|i| self.fold_item(i)).collect();
                c.external_traits.borrow_mut().insert(k, v);
            }
        }
        c
    }
}
