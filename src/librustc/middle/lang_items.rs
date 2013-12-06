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
use syntax::ast;
use syntax::ast_util::local_def;
use syntax::attr::AttrMetaMethods;
use syntax::visit;
use syntax::visit::Visitor;

use std::hashmap::HashMap;
use std::iter::Enumerate;
use std::vec;

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! lets_do_this {
    (
        There are $num_lang_items:expr lang items.
        $( $num:pat, $variant:ident, $name:expr, $method:ident; )*
    ) => {

pub enum LangItem {
    $($variant),*
}

pub struct LanguageItems {
    items: [Option<ast::DefId>, ..$num_lang_items]
}

impl LanguageItems {
    pub fn new() -> LanguageItems {
        LanguageItems {
            items: [ None, ..$num_lang_items ]
        }
    }

    pub fn items<'a>(&'a self) -> Enumerate<vec::VecIterator<'a, Option<ast::DefId>>> {
        self.items.iter().enumerate()
    }

    pub fn item_name(index: uint) -> &'static str {
        match index {
            $( $num => $name, )*
            _ => "???"
        }
    }

    pub fn require(&self, it: LangItem) -> Result<ast::DefId, ~str> {
        match self.items[it as uint] {
            Some(id) => Ok(id),
            None => Err(format!("requires `{}` lang_item",
                             LanguageItems::item_name(it as uint)))
        }
    }

    pub fn to_builtin_kind(&self, id: ast::DefId) -> Option<BuiltinBound> {
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

    $(
        pub fn $method(&self) -> Option<ast::DefId> {
            self.items[$variant as uint]
        }
    )*
}

struct LanguageItemCollector {
    items: LanguageItems,

    session: Session,

    item_refs: HashMap<&'static str, uint>,
}

struct LanguageItemVisitor<'self> {
    this: &'self mut LanguageItemCollector,
}

impl<'self> Visitor<()> for LanguageItemVisitor<'self> {
    fn visit_item(&mut self, item: @ast::item, _: ()) {
        match extract(item.attrs) {
            Some(value) => {
                let item_index = self.this.item_refs.find_equiv(&value).map(|x| *x);

                match item_index {
                    Some(item_index) => {
                        self.this.collect_item(item_index, local_def(item.id))
                    }
                    None => {}
                }
            }
            None => {}
        }

        visit::walk_item(self, item, ());
    }
}

impl LanguageItemCollector {
    pub fn new(session: Session) -> LanguageItemCollector {
        let mut item_refs = HashMap::new();

        $( item_refs.insert($name, $variant as uint); )*

        LanguageItemCollector {
            session: session,
            items: LanguageItems::new(),
            item_refs: item_refs
        }
    }

    pub fn collect_item(&mut self, item_index: uint, item_def_id: ast::DefId) {
        // Check for duplicates.
        match self.items.items[item_index] {
            Some(original_def_id) if original_def_id != item_def_id => {
                self.session.err(format!("duplicate entry for `{}`",
                                      LanguageItems::item_name(item_index)));
            }
            Some(_) | None => {
                // OK.
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
    }

    pub fn collect_local_language_items(&mut self, crate: &ast::Crate) {
        let mut v = LanguageItemVisitor { this: self };
        visit::walk_crate(&mut v, crate, ());
    }

    pub fn collect_external_language_items(&mut self) {
        let crate_store = self.session.cstore;
        iter_crate_data(crate_store, |crate_number, _crate_metadata| {
            each_lang_item(crate_store, crate_number, |node_id, item_index| {
                let def_id = ast::DefId { crate: crate_number, node: node_id };
                self.collect_item(item_index, def_id);
                true
            });
        })
    }

    pub fn collect(&mut self, crate: &ast::Crate) {
        self.collect_local_language_items(crate);
        self.collect_external_language_items();
    }
}

pub fn extract(attrs: &[ast::Attribute]) -> Option<@str> {
    for attribute in attrs.iter() {
        match attribute.name_str_pair() {
            Some((key, value)) if "lang" == key => {
                return Some(value);
            }
            Some(..) | None => {}
        }
    }

    return None;
}

pub fn collect_language_items(crate: &ast::Crate,
                              session: Session)
                           -> LanguageItems {
    let mut collector = LanguageItemCollector::new(session);
    collector.collect(crate);
    let LanguageItemCollector { items, .. } = collector;
    session.abort_if_errors();
    items
}

// End of the macro
    }
}

lets_do_this! {
    There are 41 lang items.

//  ID, Variant name,                    Name,                      Method name;
    0,  FreezeTraitLangItem,             "freeze",                  freeze_trait;
    1,  SendTraitLangItem,               "send",                    send_trait;
    2,  SizedTraitLangItem,              "sized",                   sized_trait;

    3,  DropTraitLangItem,               "drop",                    drop_trait;

    4,  AddTraitLangItem,                "add",                     add_trait;
    5,  SubTraitLangItem,                "sub",                     sub_trait;
    6,  MulTraitLangItem,                "mul",                     mul_trait;
    7,  DivTraitLangItem,                "div",                     div_trait;
    8,  RemTraitLangItem,                "rem",                     rem_trait;
    9,  NegTraitLangItem,                "neg",                     neg_trait;
    10, NotTraitLangItem,                "not",                     not_trait;
    11, BitXorTraitLangItem,             "bitxor",                  bitxor_trait;
    12, BitAndTraitLangItem,             "bitand",                  bitand_trait;
    13, BitOrTraitLangItem,              "bitor",                   bitor_trait;
    14, ShlTraitLangItem,                "shl",                     shl_trait;
    15, ShrTraitLangItem,                "shr",                     shr_trait;
    16, IndexTraitLangItem,              "index",                   index_trait;

    17, EqTraitLangItem,                 "eq",                      eq_trait;
    18, OrdTraitLangItem,                "ord",                     ord_trait;

    19, StrEqFnLangItem,                 "str_eq",                  str_eq_fn;
    20, UniqStrEqFnLangItem,             "uniq_str_eq",             uniq_str_eq_fn;
    21, FailFnLangItem,                  "fail_",                   fail_fn;
    22, FailBoundsCheckFnLangItem,       "fail_bounds_check",       fail_bounds_check_fn;
    23, ExchangeMallocFnLangItem,        "exchange_malloc",         exchange_malloc_fn;
    24, ClosureExchangeMallocFnLangItem, "closure_exchange_malloc", closure_exchange_malloc_fn;
    25, ExchangeFreeFnLangItem,          "exchange_free",           exchange_free_fn;
    26, MallocFnLangItem,                "malloc",                  malloc_fn;
    27, FreeFnLangItem,                  "free",                    free_fn;
    28, BorrowAsImmFnLangItem,           "borrow_as_imm",           borrow_as_imm_fn;
    29, BorrowAsMutFnLangItem,           "borrow_as_mut",           borrow_as_mut_fn;
    30, ReturnToMutFnLangItem,           "return_to_mut",           return_to_mut_fn;
    31, CheckNotBorrowedFnLangItem,      "check_not_borrowed",      check_not_borrowed_fn;
    32, StrDupUniqFnLangItem,            "strdup_uniq",             strdup_uniq_fn;
    33, RecordBorrowFnLangItem,          "record_borrow",           record_borrow_fn;
    34, UnrecordBorrowFnLangItem,        "unrecord_borrow",         unrecord_borrow_fn;

    35, StartFnLangItem,                 "start",                   start_fn;

    36, TyDescStructLangItem,            "ty_desc",                 ty_desc;
    37, TyVisitorTraitLangItem,          "ty_visitor",              ty_visitor;
    38, OpaqueStructLangItem,            "opaque",                  opaque;

    39, EventLoopFactoryLangItem,        "event_loop_factory",      event_loop_factory;

    40, TypeIdLangItem,                  "type_id",                 type_id;
}
