use crate::clean::*;

crate struct StripItem(pub Item);

impl StripItem {
    crate fn strip(self) -> Item {
        match self.0 {
            Item { kind: box StrippedItem(..), .. } => self.0,
            mut i => {
                i.kind = box StrippedItem(i.kind);
                i
            }
        }
    }
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
                match i.kind {
                    VariantKind::Struct(mut j) => {
                        let num_fields = j.fields.len();
                        j.fields = j.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        j.fields_stripped |= num_fields != j.fields.len()
                            || j.fields.iter().any(|f| f.is_stripped());
                        VariantItem(Variant { kind: VariantKind::Struct(j), ..i2 })
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
            is_crate: m.is_crate,
            items: m.items.into_iter().filter_map(|i| self.fold_item(i)).collect(),
        }
    }

    fn fold_crate(&mut self, mut c: Crate) -> Crate {
        c.module = c.module.take().and_then(|module| self.fold_item(module));

        {
            let external_traits = { std::mem::take(&mut *c.external_traits.borrow_mut()) };
            for (k, mut v) in external_traits {
                v.items = v.items.into_iter().filter_map(|i| self.fold_item(i)).collect();
                c.external_traits.borrow_mut().insert(k, v);
            }
        }
        c
    }
}
