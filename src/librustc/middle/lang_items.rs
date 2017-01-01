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

use dep_graph::DepNode;
use hir::map as hir_map;
use session::Session;
use hir::def_id::DefId;
use ty;
use middle::weak_lang_items;
use util::nodemap::FxHashMap;

use syntax::ast;
use syntax::symbol::Symbol;
use hir::itemlikevisit::ItemLikeVisitor;
use hir;

// The actual lang items defined come at the end of this file in one handy table.
// So you probably just want to nip down to the end.
macro_rules! language_item_table {
    (
        $( $variant:ident, $name:expr, $method:ident; )*
    ) => {


enum_from_u32! {
    #[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
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
            items: vec![$(foo($variant)),*],
            missing: Vec::new(),
        }
    }

    pub fn items(&self) -> &[Option<DefId>] {
        &*self.items
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

    pub fn fn_trait_kind(&self, id: DefId) -> Option<ty::ClosureKind> {
        let def_id_kinds = [
            (self.fn_trait(), ty::ClosureKind::Fn),
            (self.fn_mut_trait(), ty::ClosureKind::FnMut),
            (self.fn_once_trait(), ty::ClosureKind::FnOnce),
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

    item_refs: FxHashMap<&'static str, usize>,
}

impl<'a, 'v, 'tcx> ItemLikeVisitor<'v> for LanguageItemCollector<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let Some(value) = extract(&item.attrs) {
            let item_index = self.item_refs.get(&*value.as_str()).cloned();

            if let Some(item_index) = item_index {
                self.collect_item(item_index, self.ast_map.local_def_id(item.id))
            } else {
                let span = self.ast_map.span(item.id);
                span_err!(self.session, span, E0522,
                          "definition of an unknown language item: `{}`.",
                          value);
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
        // at present, lang items are always items, not trait items
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
        // at present, lang items are always items, not impl items
    }
}

impl<'a, 'tcx> LanguageItemCollector<'a, 'tcx> {
    pub fn new(session: &'a Session, ast_map: &'a hir_map::Map<'tcx>)
               -> LanguageItemCollector<'a, 'tcx> {
        let mut item_refs = FxHashMap();

        $( item_refs.insert($name, $variant as usize); )*

        LanguageItemCollector {
            session: session,
            ast_map: ast_map,
            items: LanguageItems::new(),
            item_refs: item_refs,
        }
    }

    pub fn collect_item(&mut self, item_index: usize,
                        item_def_id: DefId) {
        // Check for duplicates.
        match self.items.items[item_index] {
            Some(original_def_id) if original_def_id != item_def_id => {
                let cstore = &self.session.cstore;
                let name = LanguageItems::item_name(item_index);
                let mut err = match self.ast_map.span_if_local(item_def_id) {
                    Some(span) => struct_span_err!(
                        self.session,
                        span,
                        E0152,
                        "duplicate lang item found: `{}`.",
                        name),
                    None => self.session.struct_err(&format!(
                            "duplicate lang item in crate `{}`: `{}`.",
                            cstore.crate_name(item_def_id.krate),
                            name)),
                };
                if let Some(span) = self.ast_map.span_if_local(original_def_id) {
                    span_note!(&mut err, span,
                               "first defined here.");
                } else {
                    err.note(&format!("first defined in crate `{}`.",
                                      cstore.crate_name(original_def_id.krate)));
                }
                err.emit();
            }
            _ => {
                // OK.
            }
        }

        // Matched.
        self.items.items[item_index] = Some(item_def_id);
    }

    pub fn collect_local_language_items(&mut self, krate: &hir::Crate) {
        krate.visit_all_item_likes(self);
    }

    pub fn collect_external_language_items(&mut self) {
        let cstore = &self.session.cstore;

        for cnum in cstore.crates() {
            for (index, item_index) in cstore.lang_items(cnum) {
                let def_id = DefId { krate: cnum, index: index };
                self.collect_item(item_index, def_id);
            }
        }
    }

    pub fn collect(&mut self, krate: &hir::Crate) {
        self.collect_external_language_items();
        self.collect_local_language_items(krate);
    }
}

pub fn extract(attrs: &[ast::Attribute]) -> Option<Symbol> {
    for attribute in attrs {
        match attribute.value_str() {
            Some(value) if attribute.check_name("lang") => return Some(value),
            _ => {}
        }
    }

    return None;
}

pub fn collect_language_items(session: &Session,
                              map: &hir_map::Map)
                              -> LanguageItems {
    let _task = map.dep_graph.in_task(DepNode::CollectLanguageItems);
    let krate: &hir::Crate = map.krate();
    let mut collector = LanguageItemCollector::new(session, map);
    collector.collect(krate);
    let LanguageItemCollector { mut items, .. } = collector;
    weak_lang_items::check_crate(krate, session, &mut items);
    items
}

// End of the macro
    }
}

language_item_table! {
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
    I128ImplItem,                     "i128",                   i128_impl;
    IsizeImplItem,                   "isize",                   isize_impl;
    U8ImplItem,                      "u8",                      u8_impl;
    U16ImplItem,                     "u16",                     u16_impl;
    U32ImplItem,                     "u32",                     u32_impl;
    U64ImplItem,                     "u64",                     u64_impl;
    U128ImplItem,                    "u128",                    u128_impl;
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
    BoxFreeFnLangItem,               "box_free",                box_free_fn;
    StrDupUniqFnLangItem,            "strdup_uniq",             strdup_uniq_fn;

    StartFnLangItem,                 "start",                   start_fn;

    EhPersonalityLangItem,           "eh_personality",          eh_personality;
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

impl<'a, 'tcx, 'gcx> ty::TyCtxt<'a, 'tcx, 'gcx> {
    pub fn require_lang_item(&self, lang_item: LangItem) -> DefId {
        self.lang_items.require(lang_item).unwrap_or_else(|msg| {
            self.sess.fatal(&msg)
        })
    }
}
