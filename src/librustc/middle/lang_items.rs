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
use middle::ty::{BuiltinBound, BoundFreeze, BoundPod, BoundSend, BoundSized};
use syntax::ast;
use syntax::ast_util::local_def;
use syntax::attr::AttrMetaMethods;
use syntax::parse::token::InternedString;
use syntax::visit::Visitor;
use syntax::visit;

use collections::HashMap;
use std::iter::Enumerate;
use std::vec;
use std::vec_ng::Vec;

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! lets_do_this {
    (
        $( $variant:ident, $name:expr, $method:ident; )*
    ) => {

#[deriving(FromPrimitive)]
pub enum LangItem {
    $($variant),*
}

pub struct LanguageItems {
    items: Vec<Option<ast::DefId>> ,
}

impl LanguageItems {
    pub fn new() -> LanguageItems {
        fn foo(_: LangItem) -> Option<ast::DefId> { None }

        LanguageItems {
            items: vec!($(foo($variant)),*)
        }
    }

    pub fn items<'a>(&'a self) -> Enumerate<vec::Items<'a, Option<ast::DefId>>> {
        self.items.iter().enumerate()
    }

    pub fn item_name(index: uint) -> &'static str {
        let item: Option<LangItem> = FromPrimitive::from_uint(index);
        match item {
            $( Some($variant) => $name, )*
            None => "???"
        }
    }

    pub fn require(&self, it: LangItem) -> Result<ast::DefId, ~str> {
        match self.items.get(it as uint) {
            &Some(id) => Ok(id),
            &None => {
                Err(format!("requires `{}` lang_item",
                            LanguageItems::item_name(it as uint)))
            }
        }
    }

    pub fn to_builtin_kind(&self, id: ast::DefId) -> Option<BuiltinBound> {
        if Some(id) == self.freeze_trait() {
            Some(BoundFreeze)
        } else if Some(id) == self.send_trait() {
            Some(BoundSend)
        } else if Some(id) == self.sized_trait() {
            Some(BoundSized)
        } else if Some(id) == self.pod_trait() {
            Some(BoundPod)
        } else {
            None
        }
    }

    $(
        pub fn $method(&self) -> Option<ast::DefId> {
            *self.items.get($variant as uint)
        }
    )*
}

struct LanguageItemCollector {
    items: LanguageItems,

    session: Session,

    item_refs: HashMap<&'static str, uint>,
}

struct LanguageItemVisitor<'a> {
    this: &'a mut LanguageItemCollector,
}

impl<'a> Visitor<()> for LanguageItemVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        match extract(item.attrs.as_slice()) {
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
        match self.items.items.get(item_index) {
            &Some(original_def_id) if original_def_id != item_def_id => {
                self.session.err(format!("duplicate entry for `{}`",
                                      LanguageItems::item_name(item_index)));
            }
            &Some(_) | &None => {
                // OK.
            }
        }

        // Matched.
        *self.items.items.get_mut(item_index) = Some(item_def_id);
    }

    pub fn collect_local_language_items(&mut self, krate: &ast::Crate) {
        let mut v = LanguageItemVisitor { this: self };
        visit::walk_crate(&mut v, krate, ());
    }

    pub fn collect_external_language_items(&mut self) {
        let crate_store = self.session.cstore;
        crate_store.iter_crate_data(|crate_number, _crate_metadata| {
            each_lang_item(crate_store, crate_number, |node_id, item_index| {
                let def_id = ast::DefId { krate: crate_number, node: node_id };
                self.collect_item(item_index, def_id);
                true
            });
        })
    }

    pub fn collect(&mut self, krate: &ast::Crate) {
        self.collect_local_language_items(krate);
        self.collect_external_language_items();
    }
}

pub fn extract(attrs: &[ast::Attribute]) -> Option<InternedString> {
    for attribute in attrs.iter() {
        match attribute.name_str_pair() {
            Some((ref key, ref value)) if key.equiv(&("lang")) => {
                return Some((*value).clone());
            }
            Some(..) | None => {}
        }
    }

    return None;
}

pub fn collect_language_items(krate: &ast::Crate,
                              session: Session) -> @LanguageItems {
    let mut collector = LanguageItemCollector::new(session);
    collector.collect(krate);
    let LanguageItemCollector { items, .. } = collector;
    session.abort_if_errors();
    @items
}

// End of the macro
    }
}

lets_do_this! {
//  Variant name,                    Name,                      Method name;
    FreezeTraitLangItem,             "freeze",                  freeze_trait;
    SendTraitLangItem,               "send",                    send_trait;
    SizedTraitLangItem,              "sized",                   sized_trait;
    PodTraitLangItem,                "pod",                     pod_trait;

    DropTraitLangItem,               "drop",                    drop_trait;

    AddTraitLangItem,                "add",                     add_trait;
    SubTraitLangItem,                "sub",                     sub_trait;
    MulTraitLangItem,                "mul",                     mul_trait;
    DivTraitLangItem,                "div",                     div_trait;
    RemTraitLangItem,                "rem",                     rem_trait;
    NegTraitLangItem,                "neg",                     neg_trait;
    NotTraitLangItem,                "not",                     not_trait;
    BitXorTraitLangItem,             "bitxor",                  bitxor_trait;
    BitAndTraitLangItem,             "bitand",                  bitand_trait;
    BitOrTraitLangItem,              "bitor",                   bitor_trait;
    ShlTraitLangItem,                "shl",                     shl_trait;
    ShrTraitLangItem,                "shr",                     shr_trait;
    IndexTraitLangItem,              "index",                   index_trait;

    DerefTraitLangItem,              "deref",                   deref_trait;
    DerefMutTraitLangItem,           "deref_mut",               deref_mut_trait;

    EqTraitLangItem,                 "eq",                      eq_trait;
    OrdTraitLangItem,                "ord",                     ord_trait;

    StrEqFnLangItem,                 "str_eq",                  str_eq_fn;
    UniqStrEqFnLangItem,             "uniq_str_eq",             uniq_str_eq_fn;
    FailFnLangItem,                  "fail_",                   fail_fn;
    FailBoundsCheckFnLangItem,       "fail_bounds_check",       fail_bounds_check_fn;
    ExchangeMallocFnLangItem,        "exchange_malloc",         exchange_malloc_fn;
    ClosureExchangeMallocFnLangItem, "closure_exchange_malloc", closure_exchange_malloc_fn;
    ExchangeFreeFnLangItem,          "exchange_free",           exchange_free_fn;
    MallocFnLangItem,                "malloc",                  malloc_fn;
    FreeFnLangItem,                  "free",                    free_fn;
    StrDupUniqFnLangItem,            "strdup_uniq",             strdup_uniq_fn;

    StartFnLangItem,                 "start",                   start_fn;

    TyDescStructLangItem,            "ty_desc",                 ty_desc;
    TyVisitorTraitLangItem,          "ty_visitor",              ty_visitor;
    OpaqueStructLangItem,            "opaque",                  opaque;

    EventLoopFactoryLangItem,        "event_loop_factory",      event_loop_factory;

    TypeIdLangItem,                  "type_id",                 type_id;

    EhPersonalityLangItem,           "eh_personality",          eh_personality_fn;

    ManagedHeapLangItem,             "managed_heap",            managed_heap;
    ExchangeHeapLangItem,            "exchange_heap",           exchange_heap;
    GcLangItem,                      "gc",                      gc;

    CovariantTypeItem,               "covariant_type",          covariant_type;
    ContravariantTypeItem,           "contravariant_type",      contravariant_type;
    InvariantTypeItem,               "invariant_type",          invariant_type;

    CovariantLifetimeItem,           "covariant_lifetime",      covariant_lifetime;
    ContravariantLifetimeItem,       "contravariant_lifetime",  contravariant_lifetime;
    InvariantLifetimeItem,           "invariant_lifetime",      invariant_lifetime;

    NoFreezeItem,                    "no_freeze_bound",         no_freeze_bound;
    NoSendItem,                      "no_send_bound",           no_send_bound;
    NoPodItem,                       "no_pod_bound",            no_pod_bound;
    ManagedItem,                     "managed_bound",           managed_bound;
}
