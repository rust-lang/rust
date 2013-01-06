// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Detecting language items.
//
// Language items are items that represent concepts intrinsic to the language
// itself. Examples are:
//
// * Traits that specify "kinds"; e.g. "const", "copy", "owned".
//
// * Traits that represent operators; e.g. "add", "sub", "index".
//
// * Functions called by the compiler itself.

use driver::session::Session;
use metadata::csearch::{each_path, get_item_attrs};
use metadata::cstore::{iter_crate_data};
use metadata::decoder::{dl_def, dl_field, dl_impl};
use syntax::ast::{crate, def_fn, def_id, def_ty, lit_str, meta_item};
use syntax::ast::{meta_list, meta_name_value, meta_word};
use syntax::ast_util::{local_def};
use syntax::visit::{default_simple_visitor, mk_simple_visitor};
use syntax::visit::{visit_crate, visit_item};

use core::ptr;
use std::map::HashMap;
use str_eq = str::eq;

pub enum LangItem {
    ConstTraitLangItem,     // 0
    CopyTraitLangItem,      // 1
    OwnedTraitLangItem,     // 2
    DurableTraitLangItem,   // 3

    DropTraitLangItem,      // 4

    AddTraitLangItem,       // 5
    SubTraitLangItem,       // 6
    MulTraitLangItem,       // 7
    DivTraitLangItem,       // 8
    ModuloTraitLangItem,    // 9
    NegTraitLangItem,       // 10
    BitXorTraitLangItem,    // 11
    BitAndTraitLangItem,    // 12
    BitOrTraitLangItem,     // 13
    ShlTraitLangItem,       // 14
    ShrTraitLangItem,       // 15
    IndexTraitLangItem,     // 16

    EqTraitLangItem,        // 17
    OrdTraitLangItem,       // 18

    StrEqFnLangItem,        // 19
    UniqStrEqFnLangItem,    // 20
    AnnihilateFnLangItem,   // 21
    LogTypeFnLangItem,      // 22
}

struct LanguageItems {
    items: [ Option<def_id> * 23 ]
}

impl LanguageItems {
    static pub fn new() -> LanguageItems {
        LanguageItems {
            items: [ None, ..23 ]
        }
    }

    // XXX: Method macros sure would be nice here.

    pub fn const_trait(&const self) -> def_id {
        self.items[ConstTraitLangItem as uint].get()
    }
    pub fn copy_trait(&const self) -> def_id {
        self.items[CopyTraitLangItem as uint].get()
    }
    pub fn owned_trait(&const self) -> def_id {
        self.items[OwnedTraitLangItem as uint].get()
    }
    pub fn durable_trait(&const self) -> def_id {
        self.items[DurableTraitLangItem as uint].get()
    }

    pub fn drop_trait(&const self) -> def_id {
        self.items[DropTraitLangItem as uint].get()
    }

    pub fn add_trait(&const self) -> def_id {
        self.items[AddTraitLangItem as uint].get()
    }
    pub fn sub_trait(&const self) -> def_id {
        self.items[SubTraitLangItem as uint].get()
    }
    pub fn mul_trait(&const self) -> def_id {
        self.items[MulTraitLangItem as uint].get()
    }
    pub fn div_trait(&const self) -> def_id {
        self.items[DivTraitLangItem as uint].get()
    }
    pub fn modulo_trait(&const self) -> def_id {
        self.items[ModuloTraitLangItem as uint].get()
    }
    pub fn neg_trait(&const self) -> def_id {
        self.items[NegTraitLangItem as uint].get()
    }
    pub fn bitxor_trait(&const self) -> def_id {
        self.items[BitXorTraitLangItem as uint].get()
    }
    pub fn bitand_trait(&const self) -> def_id {
        self.items[BitAndTraitLangItem as uint].get()
    }
    pub fn bitor_trait(&const self) -> def_id {
        self.items[BitOrTraitLangItem as uint].get()
    }
    pub fn shl_trait(&const self) -> def_id {
        self.items[ShlTraitLangItem as uint].get()
    }
    pub fn shr_trait(&const self) -> def_id {
        self.items[ShrTraitLangItem as uint].get()
    }
    pub fn index_trait(&const self) -> def_id {
        self.items[IndexTraitLangItem as uint].get()
    }

    pub fn eq_trait(&const self) -> def_id {
        self.items[EqTraitLangItem as uint].get()
    }
    pub fn ord_trait(&const self) -> def_id {
        self.items[OrdTraitLangItem as uint].get()
    }

    pub fn str_eq_fn(&const self) -> def_id {
        self.items[StrEqFnLangItem as uint].get()
    }
    pub fn uniq_str_eq_fn(&const self) -> def_id {
        self.items[UniqStrEqFnLangItem as uint].get()
    }
    pub fn annihilate_fn(&const self) -> def_id {
        self.items[AnnihilateFnLangItem as uint].get()
    }
    pub fn log_type_fn(&const self) -> def_id {
        self.items[LogTypeFnLangItem as uint].get()
    }
}

fn LanguageItemCollector(crate: @crate,
                         session: Session,
                         items: &r/mut LanguageItems)
                      -> LanguageItemCollector/&r {
    let item_refs = HashMap();

    item_refs.insert(~"const", ConstTraitLangItem as uint);
    item_refs.insert(~"copy", CopyTraitLangItem as uint);
    item_refs.insert(~"owned", OwnedTraitLangItem as uint);
    item_refs.insert(~"durable", DurableTraitLangItem as uint);

    item_refs.insert(~"drop", DropTraitLangItem as uint);

    item_refs.insert(~"add", AddTraitLangItem as uint);
    item_refs.insert(~"sub", SubTraitLangItem as uint);
    item_refs.insert(~"mul", MulTraitLangItem as uint);
    item_refs.insert(~"div", DivTraitLangItem as uint);
    item_refs.insert(~"modulo", ModuloTraitLangItem as uint);
    item_refs.insert(~"neg", NegTraitLangItem as uint);
    item_refs.insert(~"bitxor", BitXorTraitLangItem as uint);
    item_refs.insert(~"bitand", BitAndTraitLangItem as uint);
    item_refs.insert(~"bitor", BitOrTraitLangItem as uint);
    item_refs.insert(~"shl", ShlTraitLangItem as uint);
    item_refs.insert(~"shr", ShrTraitLangItem as uint);
    item_refs.insert(~"index", IndexTraitLangItem as uint);

    item_refs.insert(~"eq", EqTraitLangItem as uint);
    item_refs.insert(~"ord", OrdTraitLangItem as uint);

    item_refs.insert(~"str_eq", StrEqFnLangItem as uint);
    item_refs.insert(~"uniq_str_eq", UniqStrEqFnLangItem as uint);
    item_refs.insert(~"annihilate", AnnihilateFnLangItem as uint);
    item_refs.insert(~"log_type", LogTypeFnLangItem as uint);

    LanguageItemCollector {
        crate: crate,
        session: session,
        items: items,
        item_refs: item_refs
    }
}

struct LanguageItemCollector {
    items: &mut LanguageItems,

    crate: @crate,
    session: Session,

    item_refs: HashMap<~str,uint>,
}

impl LanguageItemCollector {
    fn match_and_collect_meta_item(item_def_id: def_id,
                                   meta_item: meta_item) {
        match meta_item.node {
            meta_name_value(ref key, literal) => {
                match literal.node {
                    lit_str(value) => {
                        self.match_and_collect_item(item_def_id,
                                                    (*key),
                                                    *value);
                    }
                    _ => {} // Skip.
                }
            }
            meta_word(*) | meta_list(*) => {} // Skip.
        }
    }

    fn match_and_collect_item(item_def_id: def_id, key: ~str, value: ~str) {
        if key != ~"lang" {
            return;    // Didn't match.
        }

        match self.item_refs.find(value) {
            None => {
                // Didn't match.
            }
            Some(item_index) => {
                // Check for duplicates.
                match self.items.items[item_index] {
                    Some(original_def_id)
                            if original_def_id != item_def_id => {
                        self.session.err(fmt!("duplicate entry for `%s`",
                                              value));
                    }
                    Some(_) | None => {
                        // OK.
                    }
                }

                // Matched.
                self.items.items[item_index] = Some(item_def_id);
            }
        }
    }

    fn collect_local_language_items() {
        let this = unsafe { ptr::addr_of(&self) };
        visit_crate(*self.crate, (), mk_simple_visitor(@{
            visit_item: |item| {
                for item.attrs.each |attribute| {
                    unsafe {
                        (*this).match_and_collect_meta_item(local_def(item
                                                                      .id),
                                                            attribute.node
                                                                     .value);
                    }
                }
            },
            .. *default_simple_visitor()
        }));
    }

    fn collect_external_language_items() {
        let crate_store = self.session.cstore;
        do iter_crate_data(crate_store) |crate_number, _crate_metadata| {
            for each_path(crate_store, crate_number) |path_entry| {
                let def_id;
                match path_entry.def_like {
                    dl_def(def_ty(did)) | dl_def(def_fn(did, _)) => {
                        def_id = did;
                    }
                    dl_def(_) | dl_impl(_) | dl_field => {
                        // Skip this.
                        loop;
                    }
                }

                do get_item_attrs(crate_store, def_id) |meta_items| {
                    for meta_items.each |meta_item| {
                        self.match_and_collect_meta_item(def_id, **meta_item);
                    }
                }
            }
        }
    }

    fn check_completeness() {
        for self.item_refs.each |key, item_ref| {
            match self.items.items[item_ref] {
                None => {
                    self.session.err(fmt!("no item found for `%s`", key));
                }
                Some(_) => {
                    // OK.
                }
            }
        }
    }

    fn collect() {
        self.collect_local_language_items();
        self.collect_external_language_items();
        self.check_completeness();
    }
}

fn collect_language_items(crate: @crate, session: Session) -> LanguageItems {
    let mut items = LanguageItems::new();
    let collector = LanguageItemCollector(crate, session, &mut items);
    collector.collect();
    copy items
}

