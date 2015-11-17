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

use front::map as hir_map;
use session::Session;
use metadata::csearch::each_lang_item;
use middle::def_id::DefId;
use middle::ty;
use middle::weak_lang_items;
use util::nodemap::FnvHashMap;

use syntax::ast;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token::InternedString;
use rustc_front::intravisit::Visitor;
use rustc_front::hir;

use std::iter::Enumerate;
use std::slice;

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! lets_do_this {
    (
        $( $variant:ident, $name:expr, $method:ident; )*
    ) => {


enum_from_u32! {
    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub enum LangItem {
        $($variant,)*
    }
}

pub struct LanguageItems {
    pub items: Vec<Option<DefId>>,
    pub missing: Vec<LangItem>,
}

impl LanguageItems {
    pub fn new() -> LanguageItems {
        fn foo(_: LangItem) -> Option<DefId> { None }

        LanguageItems {
            items: vec!($(foo($variant)),*),
            missing: Vec::new(),
        }
    }

    pub fn items<'a>(&'a self) -> Enumerate<slice::Iter<'a, Option<DefId>>> {
        self.items.iter().enumerate()
    }

    pub fn item_name(index: usize) -> &'static str {
        let item: Option<LangItem> = LangItem::from_u32(index as u32);
        match item {
            $( Some($variant) => $name, )*
            None => "???"
        }
    }

    pub fn require(&self, it: LangItem) -> Result<DefId, String> {
        match self.items[it as usize] {
            Some(id) => Ok(id),
            None => {
                Err(format!("requires `{}` lang_item",
                            LanguageItems::item_name(it as usize)))
            }
        }
    }

    pub fn require_owned_box(&self) -> Result<DefId, String> {
        self.require(OwnedBoxLangItem)
    }

    pub fn from_builtin_kind(&self, bound: ty::BuiltinBound)
                             -> Result<DefId, String>
    {
        match bound {
            ty::BoundSend => self.require(SendTraitLangItem),
            ty::BoundSized => self.require(SizedTraitLangItem),
            ty::BoundCopy => self.require(CopyTraitLangItem),
            ty::BoundSync => self.require(SyncTraitLangItem),
        }
    }

    pub fn to_builtin_kind(&self, id: DefId) -> Option<ty::BuiltinBound> {
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

    pub fn fn_trait_kind(&self, id: DefId) -> Option<ty::ClosureKind> {
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
        pub fn $method(&self) -> Option<DefId> {
            self.items[$variant as usize]
        }
    )*
}

struct LanguageItemCollector<'a, 'tcx: 'a> {
    items: LanguageItems,

    ast_map: &'a hir_map::Map<'tcx>,

    session: &'a Session,

    item_refs: FnvHashMap<&'static str, usize>,
}

impl<'a, 'v, 'tcx> Visitor<'v> for LanguageItemCollector<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let Some(value) = extract(&item.attrs) {
            let item_index = self.item_refs.get(&value[..]).cloned();

            if let Some(item_index) = item_index {
                self.collect_item(item_index, self.ast_map.local_def_id(item.id), item.span)
            }
        }
    }
}

impl<'a, 'tcx> LanguageItemCollector<'a, 'tcx> {
    pub fn new(session: &'a Session, ast_map: &'a hir_map::Map<'tcx>)
               -> LanguageItemCollector<'a, 'tcx> {
        let mut item_refs = FnvHashMap();

        $( item_refs.insert($name, $variant as usize); )*

        LanguageItemCollector {
            session: session,
            ast_map: ast_map,
            items: LanguageItems::new(),
            item_refs: item_refs,
        }
    }

    pub fn collect_item(&mut self, item_index: usize,
                        item_def_id: DefId, span: Span) {
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

    pub fn collect_local_language_items(&mut self, krate: &hir::Crate) {
        krate.visit_all_items(self);
    }

    pub fn collect_external_language_items(&mut self) {
        let crate_store = &self.session.cstore;
        crate_store.iter_crate_data(|crate_number, _crate_metadata| {
            each_lang_item(crate_store, crate_number, |index, item_index| {
                let def_id = DefId { krate: crate_number, index: index };
                self.collect_item(item_index, def_id, DUMMY_SP);
                true
            });
        })
    }

    pub fn collect(&mut self, krate: &hir::Crate) {
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

pub fn collect_language_items(session: &Session,
                              map: &hir_map::Map)
                              -> LanguageItems {
    let krate: &hir::Crate = map.krate();
    let mut collector = LanguageItemCollector::new(session, map);
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
    CharImplItem,                    "char",                    char_impl;
    StrImplItem,                     "str",                     str_impl;
    SliceImplItem,                   "slice",                   slice_impl;
    ConstPtrImplItem,                "const_ptr",               const_ptr_impl;
    MutPtrImplItem,                  "mut_ptr",                 mut_ptr_impl;
    I8ImplItem,                      "i8",                      i8_impl;
    I16ImplItem,                     "i16",                     i16_impl;
    I32ImplItem,                     "i32",                     i32_impl;
    I64ImplItem,                     "i64",                     i64_impl;
    IsizeImplItem,                   "isize",                   isize_impl;
    U8ImplItem,                      "u8",                      u8_impl;
    U16ImplItem,                     "u16",                     u16_impl;
    U32ImplItem,                     "u32",                     u32_impl;
    U64ImplItem,                     "u64",                     u64_impl;
    UsizeImplItem,                   "usize",                   usize_impl;
    F32ImplItem,                     "f32",                     f32_impl;
    F64ImplItem,                     "f64",                     f64_impl;

    SendTraitLangItem,               "send",                    send_trait;
    SizedTraitLangItem,              "sized",                   sized_trait;
    UnsizeTraitLangItem,             "unsize",                  unsize_trait;
    CopyTraitLangItem,               "copy",                    copy_trait;
    SyncTraitLangItem,               "sync",                    sync_trait;

    DropTraitLangItem,               "drop",                    drop_trait;

    CoerceUnsizedTraitLangItem,      "coerce_unsized",          coerce_unsized_trait;

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
    AddAssignTraitLangItem,          "add_assign",              add_assign_trait;
    SubAssignTraitLangItem,          "sub_assign",              sub_assign_trait;
    MulAssignTraitLangItem,          "mul_assign",              mul_assign_trait;
    DivAssignTraitLangItem,          "div_assign",              div_assign_trait;
    RemAssignTraitLangItem,          "rem_assign",              rem_assign_trait;
    BitXorAssignTraitLangItem,       "bitxor_assign",           bitxor_assign_trait;
    BitAndAssignTraitLangItem,       "bitand_assign",           bitand_assign_trait;
    BitOrAssignTraitLangItem,        "bitor_assign",            bitor_assign_trait;
    ShlAssignTraitLangItem,          "shl_assign",              shl_assign_trait;
    ShrAssignTraitLangItem,          "shr_assign",              shr_assign_trait;
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

    EhPersonalityLangItem,           "eh_personality",          eh_personality;
    EhPersonalityCatchLangItem,      "eh_personality_catch",    eh_personality_catch;
    EhUnwindResumeLangItem,          "eh_unwind_resume",        eh_unwind_resume;
    MSVCTryFilterLangItem,           "msvc_try_filter",         msvc_try_filter;

    OwnedBoxLangItem,                "owned_box",               owned_box;

    PhantomDataItem,                 "phantom_data",            phantom_data;

    // Deprecated:
    CovariantTypeItem,               "covariant_type",          covariant_type;
    ContravariantTypeItem,           "contravariant_type",      contravariant_type;
    InvariantTypeItem,               "invariant_type",          invariant_type;
    CovariantLifetimeItem,           "covariant_lifetime",      covariant_lifetime;
    ContravariantLifetimeItem,       "contravariant_lifetime",  contravariant_lifetime;
    InvariantLifetimeItem,           "invariant_lifetime",      invariant_lifetime;

    NoCopyItem,                      "no_copy_bound",           no_copy_bound;

    NonZeroItem,                     "non_zero",                non_zero;

    DebugTraitLangItem,              "debug_trait",             debug_trait;
}
