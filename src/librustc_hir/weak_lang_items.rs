//! Validity checking for weak lang items

use crate::def_id::DefId;
use crate::{lang_items, LangItem, LanguageItems};

use rustc_ast::ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::{sym, Symbol};

use lazy_static::lazy_static;

macro_rules! weak_lang_items {
    ($($name:ident, $item:ident, $sym:ident;)*) => (

lazy_static! {
    pub static ref WEAK_ITEMS_REFS: FxHashMap<Symbol, LangItem> = {
        let mut map = FxHashMap::default();
        $(map.insert(sym::$name, lang_items::$item);)*
        map
    };
}

/// The `check_name` argument avoids the need for `librustc_hir` to depend on
/// `librustc_session`.
pub fn link_name<'a, F>(check_name: F, attrs: &'a [ast::Attribute]) -> Option<Symbol>
where
    F: Fn(&'a ast::Attribute, Symbol) -> bool
{
    lang_items::extract(check_name, attrs).and_then(|(name, _)| {
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
    panic_impl,         PanicImplLangItem,          rust_begin_unwind;
    eh_personality,     EhPersonalityLangItem,      rust_eh_personality;
    oom,                OomLangItem,                rust_oom;
}
