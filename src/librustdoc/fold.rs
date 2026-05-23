use std::alloc::Allocator;
use std::mem;

use crate::clean::*;

pub(crate) fn strip_item(mut item: Item) -> Item {
    if !matches!(item.inner.kind, StrippedItem(..)) {
        item.inner.kind = StrippedItem(Box::new(item.inner.kind));
    }
    item
}

pub(crate) trait DocFolder<A: Allocator + Copy>: Sized {
    fn fold_item(&mut self, item: Item, alloc: A) -> Option<Item> {
        Some(self.fold_item_recur(item, alloc))
    }

    /// don't override!
    fn fold_inner_recur(&mut self, kind: ItemKind, alloc: A) -> ItemKind {
        match kind {
            StrippedItem(..) => unreachable!(),
            ModuleItem(i) => ModuleItem(self.fold_mod(i, alloc)),
            StructItem(mut i) => {
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                StructItem(i)
            }
            UnionItem(mut i) => {
                i.fields = i.fields.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                UnionItem(i)
            }
            EnumItem(mut i) => {
                i.variants =
                    i.variants.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                EnumItem(i)
            }
            TraitItem(mut i) => {
                i.items = i.items.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                TraitItem(i)
            }
            ImplItem(mut i) => {
                i.items = i.items.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                ImplItem(i)
            }
            VariantItem(Variant { kind, discriminant }) => {
                let kind = match kind {
                    VariantKind::Struct(mut j) => {
                        j.fields =
                            j.fields.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                        VariantKind::Struct(j)
                    }
                    VariantKind::Tuple(fields) => {
                        let fields =
                            fields.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
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
                            .filter_map(|(_, x)| self.fold_item(x, alloc))
                            .collect();

                        TypeAliasInnerType::Enum { variants, is_non_exhaustive }
                    }
                    TypeAliasInnerType::Union { fields } => {
                        let fields =
                            fields.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
                        TypeAliasInnerType::Union { fields }
                    }
                    TypeAliasInnerType::Struct { ctor_kind, fields } => {
                        let fields =
                            fields.into_iter().filter_map(|x| self.fold_item(x, alloc)).collect();
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
            | RequiredMethodItem(..)
            | MethodItem(..)
            | StructFieldItem(_)
            | ForeignFunctionItem(..)
            | ForeignStaticItem(..)
            | ForeignTypeItem
            | MacroItem(..)
            | ProcMacroItem(_)
            | PrimitiveItem(_)
            | RequiredAssocConstItem(..)
            | ProvidedAssocConstItem(..)
            | ImplAssocConstItem(..)
            | RequiredAssocTypeItem(..)
            | AssocTypeItem(..)
            | KeywordItem
            | AttributeItem
            | PlaceholderImplItem => kind,
        }
    }

    /// don't override!
    fn fold_item_recur(&mut self, mut item: Item, alloc: A) -> Item {
        item.inner.kind = match item.inner.kind {
            StrippedItem(i) => StrippedItem(Box::new(self.fold_inner_recur(*i, alloc))),
            _ => self.fold_inner_recur(item.inner.kind, alloc),
        };
        item
    }

    fn fold_mod(&mut self, m: Module, alloc: A) -> Module {
        Module {
            span: m.span,
            items: m.items.into_iter().filter_map(|i| self.fold_item(i, alloc)).collect(),
        }
    }

    fn fold_crate(&mut self, mut c: Crate, alloc: A) -> Crate {
        c.module = self.fold_item(c.module, alloc).unwrap();

        for trait_ in c.external_traits.values_mut() {
            trait_.items = mem::take(&mut trait_.items)
                .into_iter()
                .filter_map(|i| self.fold_item(i, alloc))
                .collect();
        }

        c
    }
}
