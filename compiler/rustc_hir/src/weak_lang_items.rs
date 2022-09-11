//! Validity checking for weak lang items

use crate::def_id::DefId;
use crate::{lang_items, LangItem, LanguageItems};

use rustc_ast as ast;
use rustc_data_structures::fx::FxIndexMap;
use rustc_span::symbol::{sym, Symbol};

use std::sync::LazyLock;

macro_rules! weak_lang_items {
    ($($name:ident, $item:ident, $sym:ident;)*) => (

pub static WEAK_ITEMS_REFS: LazyLock<FxIndexMap<Symbol, LangItem>> = LazyLock::new(|| {
    let mut map = FxIndexMap::default();
    $(map.insert(sym::$name, LangItem::$item);)*
    map
});

pub static WEAK_ITEMS_SYMBOLS: LazyLock<FxIndexMap<LangItem, Symbol>> = LazyLock::new(|| {
    let mut map = FxIndexMap::default();
    $(map.insert(LangItem::$item, sym::$sym);)*
    map
});

pub fn link_name(attrs: &[ast::Attribute]) -> Option<Symbol>
{
    lang_items::extract(attrs).and_then(|(name, _)| {
        $(if name == sym::$name {
            Some(sym::$sym)
        } else)* {
            None
        }
    })
}

impl LanguageItems {
    pub fn is_weak_lang_item(&self, item_def_id: DefId) -> bool {
        let did = Some(item_def_id);

        $(self.$name() == did)||*
    }
}

) }

weak_lang_items! {
    panic_impl,         PanicImpl,          rust_begin_unwind;
    eh_personality,     EhPersonality,      rust_eh_personality;
    eh_catch_typeinfo,  EhCatchTypeinfo,    rust_eh_catch_typeinfo;
    oom,                Oom,                rust_oom;
}
