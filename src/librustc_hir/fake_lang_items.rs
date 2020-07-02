//! Validity checking for fake lang items

use crate::def_id::DefId;
use crate::{lang_items, LangItem, LanguageItems};

use rustc_data_structures::fx::FxHashMap;
use rustc_span::symbol::{sym, Symbol};

use lazy_static::lazy_static;

macro_rules! fake_lang_items {
    ($($item:ident, $name:ident, $method:ident;)*) => (

lazy_static! {
    pub static ref FAKE_ITEMS_REFS: FxHashMap<Symbol, LangItem> = {
        let mut map = FxHashMap::default();
        $(map.insert(sym::$name, lang_items::$item);)*
        map
    };
}

impl LanguageItems {
    pub fn is_fake_lang_item(&self, item_def_id: DefId) -> bool {
        let did = Some(item_def_id);

        $(self.$method() == did)||*
    }
}

) }

fake_lang_items! {
//  Variant name,                      Symbol,                    Method name,
    CountCodeRegionFnLangItem,         count_code_region,         count_code_region_fn;
    CoverageCounterAddFnLangItem,      coverage_counter_add,      coverage_counter_add_fn;
    CoverageCounterSubtractFnLangItem, coverage_counter_subtract, coverage_counter_subtract_fn;
}
