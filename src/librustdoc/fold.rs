// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std;
use clean::*;
use std::iter::Extendable;

pub trait DocFolder {
    fn fold_item(&mut self, item: Item) -> Option<Item> {
        self.fold_item_recur(item)
    }

    /// don't override!
    fn fold_item_recur(&mut self, item: Item) -> Option<Item> {
        use std::util::swap;
        let Item { attrs, name, source, visibility, id, inner } = item;
        let inner = inner;
        let c = |x| self.fold_item(x);
        let inner = match inner {
            StructItem(i) => {
                let mut i = i;
                let mut foo = ~[]; swap(&mut foo, &mut i.fields);
                let num_fields = foo.len();
                i.fields.extend(&mut foo.move_iter().filter_map(|x| self.fold_item(x)));
                i.fields_stripped |= num_fields != i.fields.len();
                StructItem(i)
            },
            ModuleItem(i) => {
                ModuleItem(self.fold_mod(i))
            },
            EnumItem(i) => {
                let mut i = i;
                let mut foo = ~[]; swap(&mut foo, &mut i.variants);
                let num_variants = foo.len();
                i.variants.extend(&mut foo.move_iter().filter_map(|x| self.fold_item(x)));
                i.variants_stripped |= num_variants != i.variants.len();
                EnumItem(i)
            },
            TraitItem(i) => {
                fn vtrm<T: DocFolder>(this: &mut T, trm: TraitMethod) -> Option<TraitMethod> {
                    match trm {
                        Required(it) => {
                            match this.fold_item(it) {
                                Some(x) => return Some(Required(x)),
                                None => return None,
                            }
                        },
                        Provided(it) => {
                            match this.fold_item(it) {
                                Some(x) => return Some(Provided(x)),
                                None => return None,
                            }
                        },
                    }
                }
                let mut i = i;
                let mut foo = ~[]; swap(&mut foo, &mut i.methods);
                i.methods.extend(&mut foo.move_iter().filter_map(|x| vtrm(self, x)));
                TraitItem(i)
            },
            ImplItem(i) => {
                let mut i = i;
                let mut foo = ~[]; swap(&mut foo, &mut i.methods);
                i.methods.extend(&mut foo.move_iter().filter_map(|x| self.fold_item(x)));
                ImplItem(i)
            },
            VariantItem(i) => {
                let i2 = i.clone(); // this clone is small
                match i.kind {
                    StructVariant(j) => {
                        let mut j = j;
                        let mut foo = ~[]; swap(&mut foo, &mut j.fields);
                        let num_fields = foo.len();
                        j.fields.extend(&mut foo.move_iter().filter_map(c));
                        j.fields_stripped |= num_fields != j.fields.len();
                        VariantItem(Variant {kind: StructVariant(j), ..i2})
                    },
                    _ => VariantItem(i2)
                }
            },
            x => x
        };

        Some(Item { attrs: attrs, name: name, source: source, inner: inner,
                    visibility: visibility, id: id })
    }

    fn fold_mod(&mut self, m: Module) -> Module {
        Module { items: m.items.move_iter().filter_map(|i| self.fold_item(i)).collect() }
    }

    fn fold_crate(&mut self, mut c: Crate) -> Crate {
        c.module = match std::util::replace(&mut c.module, None) {
            Some(module) => self.fold_item(module), None => None
        };
        return c;
    }
}
