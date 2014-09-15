// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clean::*;
use std::iter::Extendable;
use std::mem::{replace, swap};

pub trait DocFolder {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        self.fold_item_recur(item)
    }

    /// don't override!
    fn fold_item_recur(&mut self, item: Item) -> Option<Item> {
        let Item { attrs, name, source, visibility, def_id, inner, stability } = item;
        let inner = inner;
        let inner = match inner {
            StructItem(mut i) => {
                let mut foo = Vec::new(); swap(&mut foo, &mut i.fields);
                let num_fields = foo.len();
                i.fields.extend(foo.into_iter().filter_map(|x| self.fold_item(x)));
                i.fields_stripped |= num_fields != i.fields.len();
                StructItem(i)
            },
            ModuleItem(i) => {
                ModuleItem(self.fold_mod(i))
            },
            EnumItem(mut i) => {
                let mut foo = Vec::new(); swap(&mut foo, &mut i.variants);
                let num_variants = foo.len();
                i.variants.extend(foo.into_iter().filter_map(|x| self.fold_item(x)));
                i.variants_stripped |= num_variants != i.variants.len();
                EnumItem(i)
            },
            TraitItem(mut i) => {
                fn vtrm<T: DocFolder>(this: &mut T, trm: TraitItem)
                        -> Option<TraitItem> {
                    match trm {
                        RequiredMethod(it) => {
                            match this.fold_item(it) {
                                Some(x) => return Some(RequiredMethod(x)),
                                None => return None,
                            }
                        },
                        ProvidedMethod(it) => {
                            match this.fold_item(it) {
                                Some(x) => return Some(ProvidedMethod(x)),
                                None => return None,
                            }
                        },
                    }
                }
                let mut foo = Vec::new(); swap(&mut foo, &mut i.items);
                i.items.extend(foo.into_iter().filter_map(|x| vtrm(self, x)));
                TraitItem(i)
            },
            ImplItem(mut i) => {
                let mut foo = Vec::new(); swap(&mut foo, &mut i.items);
                i.items.extend(foo.into_iter()
                                  .filter_map(|x| self.fold_item(x)));
                ImplItem(i)
            },
            VariantItem(i) => {
                let i2 = i.clone(); // this clone is small
                match i.kind {
                    StructVariant(mut j) => {
                        let mut foo = Vec::new(); swap(&mut foo, &mut j.fields);
                        let num_fields = foo.len();
                        let c = |x| self.fold_item(x);
                        j.fields.extend(foo.into_iter().filter_map(c));
                        j.fields_stripped |= num_fields != j.fields.len();
                        VariantItem(Variant {kind: StructVariant(j), ..i2})
                    },
                    _ => VariantItem(i2)
                }
            },
            x => x
        };

        Some(Item { attrs: attrs, name: name, source: source, inner: inner,
                    visibility: visibility, stability: stability, def_id: def_id })
    }

    fn fold_mod(&mut self, m: Module) -> Module {
        Module {
            is_crate: m.is_crate,
            items: m.items.into_iter().filter_map(|i| self.fold_item(i)).collect()
        }
    }

    fn fold_crate(&mut self, mut c: Crate) -> Crate {
        c.module = match replace(&mut c.module, None) {
            Some(module) => self.fold_item(module), None => None
        };
        return c;
    }
}
