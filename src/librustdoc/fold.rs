use std::mem;

use crate::clean::*;

pub(crate) fn strip_item(mut item: Item) -> Item {
    if !matches!(item.inner.kind, StrippedItem(..)) {
        item.inner.kind = StrippedItem(Box::new(item.inner.kind));
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
            VariantItem(Variant { kind, discriminant }) => {
                let kind = match kind {
                    VariantKind::Struct(mut j) => {
                        j.fields = j.fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        VariantKind::Struct(j)
                    }
                    VariantKind::Tuple(fields) => {
                        let fields = fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        VariantKind::Tuple(fields)
                    }
                    VariantKind::CLike => VariantKind::CLike,
                };

                VariantItem(Variant { kind, discriminant })
            }
            TypeAliasItem(mut typealias) => {
                typealias.inner_type = typealias.inner_type.map(|inner_type| match inner_type {
                    TypeAliasInnerType::Enum { variants, is_non_exhaustive } => {
                        let variants = variants
                            .into_iter_enumerated()
                            .filter_map(|(_, x)| self.fold_item(x))
                            .collect();

                        TypeAliasInnerType::Enum { variants, is_non_exhaustive }
                    }
                    TypeAliasInnerType::Union { fields } => {
                        let fields = fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        TypeAliasInnerType::Union { fields }
                    }
                    TypeAliasInnerType::Struct { ctor_kind, fields } => {
                        let fields = fields.into_iter().filter_map(|x| self.fold_item(x)).collect();
                        TypeAliasInnerType::Struct { ctor_kind, fields }
                    }
                });

                TypeAliasItem(typealias)
            }
            ExternCrateItem { src: _ }
            | ImportItem(_)
            | FunctionItem(_)
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
            | AttributeItem => kind,
        }
    }

    /// don't override!
    fn fold_item_recur(&mut self, mut item: Item) -> Item {
        item.inner.kind = match item.inner.kind {
            StrippedItem(box i) => StrippedItem(Box::new(self.fold_inner_recur(i))),
            _ => self.fold_inner_recur(item.inner.kind),
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

        for trait_ in c.external_traits.values_mut() {
            trait_.items = mem::take(&mut trait_.items)
                .into_iter()
                .filter_map(|i| self.fold_item(i))
                .collect();
        }

        c
    }
}
