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
// * Traits that specify "kinds"; e.g. "Sync", "Send".
//
// * Traits that represent operators; e.g. "Add", "Sub", "Index".
//
// * Functions called by the compiler itself.

pub use self::LangItem::*;

use session::Session;
use metadata::csearch::each_lang_item;
use middle::ty;
use middle::weak_lang_items;
use util::nodemap::FnvHashMap;

use syntax::ast;
use syntax::ast_util::local_def;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token::InternedString;
use syntax::visit::Visitor;
use syntax::visit;

use std::iter::Enumerate;
use std::num::FromPrimitive;
use std::slice;

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! lets_do_this {
    (
        $( $variant:ident, $name:expr, $method:ident; )*
    ) => {

#[derive(Copy, FromPrimitive, PartialEq, Eq, Hash)]
pub enum LangItem {
    $($variant),*
}

pub struct LanguageItems {
    pub items: Vec<Option<ast::DefId>>,
    pub missing: Vec<LangItem>,
}

impl LanguageItems {
    pub fn new() -> LanguageItems {
        fn foo(_: LangItem) -> Option<ast::DefId> { None }

        LanguageItems {
            items: vec!($(foo($variant)),*),
            missing: Vec::new(),
        }
    }

    pub fn items<'a>(&'a self) -> Enumerate<slice::Iter<'a, Option<ast::DefId>>> {
        self.items.iter().enumerate()
    }

    pub fn item_name(index: uint) -> &'static str {
        let item: Option<LangItem> = FromPrimitive::from_uint(index);
        match item {
            $( Some($variant) => $name, )*
            None => "???"
        }
    }

    pub fn require(&self, it: LangItem) -> Result<ast::DefId, String> {
        match self.items[it as uint] {
            Some(id) => Ok(id),
            None => {
                Err(format!("requires `{}` lang_item",
                            LanguageItems::item_name(it as uint)))
            }
        }
    }

    pub fn from_builtin_kind(&self, bound: ty::BuiltinBound)
                             -> Result<ast::DefId, String>
    {
        match bound {
            ty::BoundSend => self.require(SendTraitLangItem),
            ty::BoundSized => self.require(SizedTraitLangItem),
            ty::BoundCopy => self.require(CopyTraitLangItem),
            ty::BoundSync => self.require(SyncTraitLangItem),
        }
    }

    pub fn to_builtin_kind(&self, id: ast::DefId) -> Option<ty::BuiltinBound> {
        if Some(id) == self.send_trait() {
            Some(ty::BoundSend)
        } else if Some(id) == self.sized_trait() {
            Some(ty::BoundSized)
        } else if Some(id) == self.copy_trait() {
            Some(ty::BoundCopy)
        } else if Some(id) == self.sync_trait() {
            Some(ty::BoundSync)
        } else {
            None
        }
    }

    pub fn fn_trait_kind(&self, id: ast::DefId) -> Option<ty::ClosureKind> {
        let def_id_kinds = [
            (self.fn_trait(), ty::FnClosureKind),
            (self.fn_mut_trait(), ty::FnMutClosureKind),
            (self.fn_once_trait(), ty::FnOnceClosureKind),
            ];

        for &(opt_def_id, kind) in &def_id_kinds {
            if Some(id) == opt_def_id {
                return Some(kind);
            }
        }

        None
    }

    $(
        #[allow(dead_code)]
        pub fn $method(&self) -> Option<ast::DefId> {
            self.items[$variant as uint]
        }
    )*
}

struct LanguageItemCollector<'a> {
    items: LanguageItems,

    session: &'a Session,

    item_refs: FnvHashMap<&'static str, uint>,
}

impl<'a, 'v> Visitor<'v> for LanguageItemCollector<'a> {
    fn visit_item(&mut self, item: &ast::Item) {
        if let Some(value) = extract(&item.attrs) {
            let item_index = self.item_refs.get(&value[..]).cloned();

            if let Some(item_index) = item_index {
                self.collect_item(item_index, local_def(item.id), item.span)
            }
        }

        visit::walk_item(self, item);
    }
}

impl<'a> LanguageItemCollector<'a> {
    pub fn new(session: &'a Session) -> LanguageItemCollector<'a> {
        let mut item_refs = FnvHashMap();

        $( item_refs.insert($name, $variant as uint); )*

        LanguageItemCollector {
            session: session,
            items: LanguageItems::new(),
            item_refs: item_refs
        }
    }

    pub fn collect_item(&mut self, item_index: uint,
                        item_def_id: ast::DefId, span: Span) {
        // Check for duplicates.
        match self.items.items[item_index] {
            Some(original_def_id) if original_def_id != item_def_id => {
                span_err!(self.session, span, E0152,
                    "duplicate entry for `{}`", LanguageItems::item_name(item_index));
            }
            Some(_) | None => {
                // OK.
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
    }

    pub fn collect_local_language_items(&mut self, krate: &ast::Crate) {
        visit::walk_crate(self, krate);
    }

    pub fn collect_external_language_items(&mut self) {
        let crate_store = &self.session.cstore;
        crate_store.iter_crate_data(|crate_number, _crate_metadata| {
            each_lang_item(crate_store, crate_number, |node_id, item_index| {
                let def_id = ast::DefId { krate: crate_number, node: node_id };
                self.collect_item(item_index, def_id, DUMMY_SP);
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
    for attribute in attrs {
        match attribute.value_str() {
            Some(ref value) if attribute.check_name("lang") => {
                return Some(value.clone());
            }
            _ => {}
        }
    }

    return None;
}

pub fn collect_language_items(krate: &ast::Crate,
                              session: &Session) -> LanguageItems {
    let mut collector = LanguageItemCollector::new(session);
    collector.collect(krate);
    let LanguageItemCollector { mut items, .. } = collector;
    weak_lang_items::check_crate(krate, session, &mut items);
    session.abort_if_errors();
    items
}

// End of the macro
    }
}

lets_do_this! {
//  Variant name,                    Name,                      Method name;
    SendTraitLangItem,               "send",                    send_trait;
    SizedTraitLangItem,              "sized",                   sized_trait;
    CopyTraitLangItem,               "copy",                    copy_trait;
    SyncTraitLangItem,               "sync",                    sync_trait;

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
    IndexMutTraitLangItem,           "index_mut",               index_mut_trait;
    RangeStructLangItem,             "range",                   range_struct;
    RangeFromStructLangItem,         "range_from",              range_from_struct;
    RangeToStructLangItem,           "range_to",                range_to_struct;
    RangeFullStructLangItem,         "range_full",              range_full_struct;

    UnsafeCellTypeLangItem,          "unsafe_cell",             unsafe_cell_type;

    DerefTraitLangItem,              "deref",                   deref_trait;
    DerefMutTraitLangItem,           "deref_mut",               deref_mut_trait;

    FnTraitLangItem,                 "fn",                      fn_trait;
    FnMutTraitLangItem,              "fn_mut",                  fn_mut_trait;
    FnOnceTraitLangItem,             "fn_once",                 fn_once_trait;

    EqTraitLangItem,                 "eq",                      eq_trait;
    OrdTraitLangItem,                "ord",                     ord_trait;

    StrEqFnLangItem,                 "str_eq",                  str_eq_fn;

    // A number of panic-related lang items. The `panic` item corresponds to
    // divide-by-zero and various panic cases with `match`. The
    // `panic_bounds_check` item is for indexing arrays.
    //
    // The `begin_unwind` lang item has a predefined symbol name and is sort of
    // a "weak lang item" in the sense that a crate is not required to have it
    // defined to use it, but a final product is required to define it
    // somewhere. Additionally, there are restrictions on crates that use a weak
    // lang item, but do not have it defined.
    PanicFnLangItem,                 "panic",                   panic_fn;
    PanicBoundsCheckFnLangItem,      "panic_bounds_check",      panic_bounds_check_fn;
    PanicFmtLangItem,                "panic_fmt",               panic_fmt;

    ExchangeMallocFnLangItem,        "exchange_malloc",         exchange_malloc_fn;
    ExchangeFreeFnLangItem,          "exchange_free",           exchange_free_fn;
    StrDupUniqFnLangItem,            "strdup_uniq",             strdup_uniq_fn;

    StartFnLangItem,                 "start",                   start_fn;

    TyDescStructLangItem,            "ty_desc",                 ty_desc;
    OpaqueStructLangItem,            "opaque",                  opaque;

    EhPersonalityLangItem,           "eh_personality",          eh_personality;

    ExchangeHeapLangItem,            "exchange_heap",           exchange_heap;
    OwnedBoxLangItem,                "owned_box",               owned_box;

    PhantomFnItem,                   "phantom_fn",              phantom_fn;
    PhantomDataItem,                 "phantom_data",            phantom_data;

    // Deprecated:
    CovariantTypeItem,               "covariant_type",          covariant_type;
    ContravariantTypeItem,           "contravariant_type",      contravariant_type;
    InvariantTypeItem,               "invariant_type",          invariant_type;
    CovariantLifetimeItem,           "covariant_lifetime",      covariant_lifetime;
    ContravariantLifetimeItem,       "contravariant_lifetime",  contravariant_lifetime;
    InvariantLifetimeItem,           "invariant_lifetime",      invariant_lifetime;

    NoCopyItem,                      "no_copy_bound",           no_copy_bound;
    ManagedItem,                     "managed_bound",           managed_bound;

    NonZeroItem,                     "non_zero",                non_zero;

    StackExhaustedLangItem,          "stack_exhausted",         stack_exhausted;

    DebugTraitLangItem,              "debug_trait",             debug_trait;
}
