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
// * Traits that specify "kinds"; e.g. "Freeze", "Send".
//
// * Traits that represent operators; e.g. "Add", "Sub", "Index".
//
// * Functions called by the compiler itself.


use driver::session::Session;
use metadata::csearch::each_lang_item;
use metadata::cstore::iter_crate_data;
use middle::ty::{BuiltinBound, BoundFreeze, BoundSend, BoundSized};
use syntax::ast::{Crate, def_id, MetaItem};
use syntax::ast_util::local_def;
use syntax::attr::AttrMetaMethods;
use syntax::ast::{item};
use syntax::visit;
use syntax::visit::Visitor;

use std::hashmap::HashMap;

pub enum LangItem {
    FreezeTraitLangItem,               // 0
    SendTraitLangItem,                 // 1
    SizedTraitLangItem,                // 2

    DropTraitLangItem,                 // 3

    AddTraitLangItem,                  // 4
    SubTraitLangItem,                  // 5
    MulTraitLangItem,                  // 6
    DivTraitLangItem,                  // 7
    RemTraitLangItem,                  // 8
    NegTraitLangItem,                  // 9
    NotTraitLangItem,                  // 10
    BitXorTraitLangItem,               // 11
    BitAndTraitLangItem,               // 12
    BitOrTraitLangItem,                // 13
    ShlTraitLangItem,                  // 14
    ShrTraitLangItem,                  // 15
    IndexTraitLangItem,                // 16

    EqTraitLangItem,                   // 17
    OrdTraitLangItem,                  // 18

    StrEqFnLangItem,                   // 19
    UniqStrEqFnLangItem,               // 20
    LogTypeFnLangItem,                 // 21
    FailFnLangItem,                    // 22
    FailBoundsCheckFnLangItem,         // 23
    ExchangeMallocFnLangItem,          // 24
    ClosureExchangeMallocFnLangItem,   // 25
    ExchangeFreeFnLangItem,            // 26
    MallocFnLangItem,                  // 27
    FreeFnLangItem,                    // 28
    BorrowAsImmFnLangItem,             // 29
    BorrowAsMutFnLangItem,             // 30
    ReturnToMutFnLangItem,             // 31
    CheckNotBorrowedFnLangItem,        // 32
    StrDupUniqFnLangItem,              // 33
    RecordBorrowFnLangItem,            // 34
    UnrecordBorrowFnLangItem,          // 35

    StartFnLangItem,                   // 36

    TyDescStructLangItem,              // 37
    TyVisitorTraitLangItem,            // 38
    OpaqueStructLangItem,              // 39
}

pub struct LanguageItems {
    items: [Option<def_id>, ..41]
}

impl LanguageItems {
    pub fn new() -> LanguageItems {
        LanguageItems {
            items: [ None, ..41 ]
        }
    }

    pub fn each_item(&self, f: &fn(Option<def_id>, uint) -> bool) -> bool {
        self.items.iter().enumerate().advance(|(i, &item)| f(item, i))
    }

    pub fn item_name(index: uint) -> &'static str {
        match index {
            0  => "freeze",
            1  => "send",
            2  => "sized",

            3  => "drop",

            4  => "add",
            5  => "sub",
            6  => "mul",
            7  => "div",
            8  => "rem",
            9  => "neg",
            10 => "not",
            11 => "bitxor",
            12 => "bitand",
            13 => "bitor",
            14 => "shl",
            15 => "shr",
            16 => "index",
            17 => "eq",
            18 => "ord",

            19 => "str_eq",
            20 => "uniq_str_eq",
            21 => "log_type",
            22 => "fail_",
            23 => "fail_bounds_check",
            24 => "exchange_malloc",
            25 => "closure_exchange_malloc",
            26 => "exchange_free",
            27 => "malloc",
            28 => "free",
            29 => "borrow_as_imm",
            30 => "borrow_as_mut",
            31 => "return_to_mut",
            32 => "check_not_borrowed",
            33 => "strdup_uniq",
            34 => "record_borrow",
            35 => "unrecord_borrow",

            36 => "start",

            37 => "ty_desc",
            38 => "ty_visitor",
            39 => "opaque",

            _ => "???"
        }
    }

    // FIXME #4621: Method macros sure would be nice here.

    pub fn require(&self, it: LangItem) -> Result<def_id, ~str> {
        match self.items[it as uint] {
            Some(id) => Ok(id),
            None => Err(fmt!("requires `%s` lang_item",
                             LanguageItems::item_name(it as uint)))
        }
    }

    pub fn to_builtin_kind(&self, id: def_id) -> Option<BuiltinBound> {
        if Some(id) == self.freeze_trait() {
            Some(BoundFreeze)
        } else if Some(id) == self.send_trait() {
            Some(BoundSend)
        } else if Some(id) == self.sized_trait() {
            Some(BoundSized)
        } else {
            None
        }
    }

    pub fn freeze_trait(&self) -> Option<def_id> {
        self.items[FreezeTraitLangItem as uint]
    }
    pub fn send_trait(&self) -> Option<def_id> {
        self.items[SendTraitLangItem as uint]
    }
    pub fn sized_trait(&self) -> Option<def_id> {
        self.items[SizedTraitLangItem as uint]
    }

    pub fn drop_trait(&self) -> Option<def_id> {
        self.items[DropTraitLangItem as uint]
    }

    pub fn add_trait(&self) -> Option<def_id> {
        self.items[AddTraitLangItem as uint]
    }
    pub fn sub_trait(&self) -> Option<def_id> {
        self.items[SubTraitLangItem as uint]
    }
    pub fn mul_trait(&self) -> Option<def_id> {
        self.items[MulTraitLangItem as uint]
    }
    pub fn div_trait(&self) -> Option<def_id> {
        self.items[DivTraitLangItem as uint]
    }
    pub fn rem_trait(&self) -> Option<def_id> {
        self.items[RemTraitLangItem as uint]
    }
    pub fn neg_trait(&self) -> Option<def_id> {
        self.items[NegTraitLangItem as uint]
    }
    pub fn not_trait(&self) -> Option<def_id> {
        self.items[NotTraitLangItem as uint]
    }
    pub fn bitxor_trait(&self) -> Option<def_id> {
        self.items[BitXorTraitLangItem as uint]
    }
    pub fn bitand_trait(&self) -> Option<def_id> {
        self.items[BitAndTraitLangItem as uint]
    }
    pub fn bitor_trait(&self) -> Option<def_id> {
        self.items[BitOrTraitLangItem as uint]
    }
    pub fn shl_trait(&self) -> Option<def_id> {
        self.items[ShlTraitLangItem as uint]
    }
    pub fn shr_trait(&self) -> Option<def_id> {
        self.items[ShrTraitLangItem as uint]
    }
    pub fn index_trait(&self) -> Option<def_id> {
        self.items[IndexTraitLangItem as uint]
    }

    pub fn eq_trait(&self) -> Option<def_id> {
        self.items[EqTraitLangItem as uint]
    }
    pub fn ord_trait(&self) -> Option<def_id> {
        self.items[OrdTraitLangItem as uint]
    }

    pub fn str_eq_fn(&self) -> Option<def_id> {
        self.items[StrEqFnLangItem as uint]
    }
    pub fn uniq_str_eq_fn(&self) -> Option<def_id> {
        self.items[UniqStrEqFnLangItem as uint]
    }
    pub fn log_type_fn(&self) -> Option<def_id> {
        self.items[LogTypeFnLangItem as uint]
    }
    pub fn fail_fn(&self) -> Option<def_id> {
        self.items[FailFnLangItem as uint]
    }
    pub fn fail_bounds_check_fn(&self) -> Option<def_id> {
        self.items[FailBoundsCheckFnLangItem as uint]
    }
    pub fn exchange_malloc_fn(&self) -> Option<def_id> {
        self.items[ExchangeMallocFnLangItem as uint]
    }
    pub fn closure_exchange_malloc_fn(&self) -> Option<def_id> {
        self.items[ClosureExchangeMallocFnLangItem as uint]
    }
    pub fn exchange_free_fn(&self) -> Option<def_id> {
        self.items[ExchangeFreeFnLangItem as uint]
    }
    pub fn malloc_fn(&self) -> Option<def_id> {
        self.items[MallocFnLangItem as uint]
    }
    pub fn free_fn(&self) -> Option<def_id> {
        self.items[FreeFnLangItem as uint]
    }
    pub fn borrow_as_imm_fn(&self) -> Option<def_id> {
        self.items[BorrowAsImmFnLangItem as uint]
    }
    pub fn borrow_as_mut_fn(&self) -> Option<def_id> {
        self.items[BorrowAsMutFnLangItem as uint]
    }
    pub fn return_to_mut_fn(&self) -> Option<def_id> {
        self.items[ReturnToMutFnLangItem as uint]
    }
    pub fn check_not_borrowed_fn(&self) -> Option<def_id> {
        self.items[CheckNotBorrowedFnLangItem as uint]
    }
    pub fn strdup_uniq_fn(&self) -> Option<def_id> {
        self.items[StrDupUniqFnLangItem as uint]
    }
    pub fn record_borrow_fn(&self) -> Option<def_id> {
        self.items[RecordBorrowFnLangItem as uint]
    }
    pub fn unrecord_borrow_fn(&self) -> Option<def_id> {
        self.items[UnrecordBorrowFnLangItem as uint]
    }
    pub fn start_fn(&self) -> Option<def_id> {
        self.items[StartFnLangItem as uint]
    }
    pub fn ty_desc(&self) -> Option<def_id> {
        self.items[TyDescStructLangItem as uint]
    }
    pub fn ty_visitor(&self) -> Option<def_id> {
        self.items[TyVisitorTraitLangItem as uint]
    }
    pub fn opaque(&self) -> Option<def_id> {
        self.items[OpaqueStructLangItem as uint]
    }
}

struct LanguageItemCollector<'self> {
    items: LanguageItems,

    crate: &'self Crate,
    session: Session,

    item_refs: HashMap<@str, uint>,
}

struct LanguageItemVisitor<'self> {
    this: *mut LanguageItemCollector<'self>,
}

impl<'self> Visitor<()> for LanguageItemVisitor<'self> {

    fn visit_item(&mut self, item:@item, _:()) {

                for attribute in item.attrs.iter() {
                    unsafe {
                        (*self.this).match_and_collect_meta_item(
                            local_def(item.id),
                            attribute.node.value
                        );
                    }
                }

        visit::walk_item(self, item, ());
    }
}

impl<'self> LanguageItemCollector<'self> {
    pub fn new<'a>(crate: &'a Crate, session: Session)
                   -> LanguageItemCollector<'a> {
        let mut item_refs = HashMap::new();

        item_refs.insert(@"freeze", FreezeTraitLangItem as uint);
        item_refs.insert(@"send", SendTraitLangItem as uint);
        item_refs.insert(@"sized", SizedTraitLangItem as uint);

        item_refs.insert(@"drop", DropTraitLangItem as uint);

        item_refs.insert(@"add", AddTraitLangItem as uint);
        item_refs.insert(@"sub", SubTraitLangItem as uint);
        item_refs.insert(@"mul", MulTraitLangItem as uint);
        item_refs.insert(@"div", DivTraitLangItem as uint);
        item_refs.insert(@"rem", RemTraitLangItem as uint);
        item_refs.insert(@"neg", NegTraitLangItem as uint);
        item_refs.insert(@"not", NotTraitLangItem as uint);
        item_refs.insert(@"bitxor", BitXorTraitLangItem as uint);
        item_refs.insert(@"bitand", BitAndTraitLangItem as uint);
        item_refs.insert(@"bitor", BitOrTraitLangItem as uint);
        item_refs.insert(@"shl", ShlTraitLangItem as uint);
        item_refs.insert(@"shr", ShrTraitLangItem as uint);
        item_refs.insert(@"index", IndexTraitLangItem as uint);

        item_refs.insert(@"eq", EqTraitLangItem as uint);
        item_refs.insert(@"ord", OrdTraitLangItem as uint);

        item_refs.insert(@"str_eq", StrEqFnLangItem as uint);
        item_refs.insert(@"uniq_str_eq", UniqStrEqFnLangItem as uint);
        item_refs.insert(@"log_type", LogTypeFnLangItem as uint);
        item_refs.insert(@"fail_", FailFnLangItem as uint);
        item_refs.insert(@"fail_bounds_check",
                         FailBoundsCheckFnLangItem as uint);
        item_refs.insert(@"exchange_malloc", ExchangeMallocFnLangItem as uint);
        item_refs.insert(@"closure_exchange_malloc", ClosureExchangeMallocFnLangItem as uint);
        item_refs.insert(@"exchange_free", ExchangeFreeFnLangItem as uint);
        item_refs.insert(@"malloc", MallocFnLangItem as uint);
        item_refs.insert(@"free", FreeFnLangItem as uint);
        item_refs.insert(@"borrow_as_imm", BorrowAsImmFnLangItem as uint);
        item_refs.insert(@"borrow_as_mut", BorrowAsMutFnLangItem as uint);
        item_refs.insert(@"return_to_mut", ReturnToMutFnLangItem as uint);
        item_refs.insert(@"check_not_borrowed",
                         CheckNotBorrowedFnLangItem as uint);
        item_refs.insert(@"strdup_uniq", StrDupUniqFnLangItem as uint);
        item_refs.insert(@"record_borrow", RecordBorrowFnLangItem as uint);
        item_refs.insert(@"unrecord_borrow", UnrecordBorrowFnLangItem as uint);
        item_refs.insert(@"start", StartFnLangItem as uint);
        item_refs.insert(@"ty_desc", TyDescStructLangItem as uint);
        item_refs.insert(@"ty_visitor", TyVisitorTraitLangItem as uint);
        item_refs.insert(@"opaque", OpaqueStructLangItem as uint);

        LanguageItemCollector {
            crate: crate,
            session: session,
            items: LanguageItems::new(),
            item_refs: item_refs
        }
    }

    pub fn match_and_collect_meta_item(&mut self,
                                       item_def_id: def_id,
                                       meta_item: &MetaItem) {
        match meta_item.name_str_pair() {
            Some((key, value)) => {
                self.match_and_collect_item(item_def_id, key, value);
            }
            None => {} // skip
        }
    }

    pub fn collect_item(&mut self, item_index: uint, item_def_id: def_id) {
        // Check for duplicates.
        match self.items.items[item_index] {
            Some(original_def_id) if original_def_id != item_def_id => {
                self.session.err(fmt!("duplicate entry for `%s`",
                                      LanguageItems::item_name(item_index)));
            }
            Some(_) | None => {
                // OK.
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
    }

    pub fn match_and_collect_item(&mut self,
                                  item_def_id: def_id,
                                  key: &str,
                                  value: @str) {
        if "lang" != key {
            return;    // Didn't match.
        }

        let item_index = self.item_refs.find(&value).map_move(|x| *x);
        // prevent borrow checker from considering   ^~~~~~~~~~~
        // self to be borrowed (annoying)

        match item_index {
            Some(item_index) => {
                self.collect_item(item_index, item_def_id);
            }
            None => {
                // Didn't match.
                return;
            }
        }
    }

    pub fn collect_local_language_items(&mut self) {
        let this: *mut LanguageItemCollector = &mut *self;
        let mut v = LanguageItemVisitor { this: this };
        visit::walk_crate(&mut v, self.crate, ());
    }

    pub fn collect_external_language_items(&mut self) {
        let crate_store = self.session.cstore;
        do iter_crate_data(crate_store) |crate_number, _crate_metadata| {
            do each_lang_item(crate_store, crate_number)
                    |node_id, item_index| {
                let def_id = def_id { crate: crate_number, node: node_id };
                self.collect_item(item_index, def_id);
                true
            };
        }
    }

    pub fn collect(&mut self) {
        self.collect_local_language_items();
        self.collect_external_language_items();
    }
}

pub fn collect_language_items(crate: &Crate,
                              session: Session)
                           -> LanguageItems {
    let mut collector = LanguageItemCollector::new(crate, session);
    collector.collect();
    let LanguageItemCollector { items, _ } = collector;
    session.abort_if_errors();
    items
}
