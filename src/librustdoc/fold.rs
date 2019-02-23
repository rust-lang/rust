use crate::clean::*;

pub struct StripItem(pub Item);

impl StripItem {
    pub fn strip(self) -> Option<Item> {
        match self.0 {
            Item { inner: StrippedItem(..), .. } => Some(self.0),
            mut i => {
                i.inner = StrippedItem(box i.inner);
                Some(i)
            }
        }
    }
}

pub trait DocFolder : Sized {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        self.fold_item_recur(item)
    }

    /// don't override!
    fn fold_inner_recur(&mut self, inner: ItemEnum) -> ItemEnum {
        match inner {
            StrippedItem(..) => unreachable!(),
            ModuleItem(i) => {
                ModuleItem(self.fold_mod(i))
            },
            StructItem(mut i) => {
                let num_fields = i.fields.len();
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                i.fields_stripped |= num_fields != i.fields.len() ||
                                     i.fields.iter().any(|f| f.is_stripped());
                StructItem(i)
            },
            UnionItem(mut i) => {
                let num_fields = i.fields.len();
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                i.fields_stripped |= num_fields != i.fields.len() ||
                                     i.fields.iter().any(|f| f.is_stripped());
                UnionItem(i)
            },
            EnumItem(mut i) => {
                let num_variants = i.variants.len();
                i.variants = i.variants.into_iter().filter_map(|x| self.fold_item(x)).collect();
                i.variants_stripped |= num_variants != i.variants.len() ||
                                       i.variants.iter().any(|f| f.is_stripped());
                EnumItem(i)
            },
            TraitItem(mut i) => {
                i.items = i.items.into_iter().filter_map(|x| self.fold_item(x)).collect();
                TraitItem(i)
            },
            ImplItem(mut i) => {
                i.items = i.items.into_iter().filter_map(|x| self.fold_item(x)).collect();
                ImplItem(i)
            },
            VariantItem(i) => {
                let i2 = i.clone(); // this clone is small
                match i.kind {
                    VariantKind::Struct(mut j) => {
                        let num_fields = j.fields.len();
                        j.fields = j.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        j.fields_stripped |= num_fields != j.fields.len() ||
                                             j.fields.iter().any(|f| f.is_stripped());
                        VariantItem(Variant {kind: VariantKind::Struct(j), ..i2})
                    },
                    _ => VariantItem(i2)
                }
            },
            x => x
        }
    }

    /// don't override!
    fn fold_item_recur(&mut self, item: Item) -> Option<Item> {
        let Item {
            attrs,
            name,
            source,
            visibility,
            def_id,
            inner,
            stability,
            deprecation,
        } = item;

        let inner = match inner {
            StrippedItem(box i) => StrippedItem(box self.fold_inner_recur(i)),
            _ => self.fold_inner_recur(inner),
        };

        Some(Item { attrs, name, source, inner, visibility,
                    stability, deprecation, def_id })
    }

    fn fold_mod(&mut self, m: Module) -> Module {
        Module {
            is_crate: m.is_crate,
            items: m.items.into_iter().filter_map(|i| self.fold_item(i)).collect()
        }
    }

    fn fold_crate(&mut self, mut c: Crate) -> Crate {
        c.module = c.module.take().and_then(|module| self.fold_item(module));

        {
            let guard = c.external_traits.lock();
            let traits = guard.replace(Default::default());
            guard.borrow_mut().extend(traits.into_iter().map(|(k, mut v)| {
                v.items = v.items.into_iter().filter_map(|i| self.fold_item(i)).collect();
                (k, v)
            }));
        }
        c
    }
}
