//! Validity checking for weak lang items

use rustc_span::{Symbol, sym};

use crate::LangItem;

macro_rules! weak_lang_items {
    ($($item:ident, $sym:ident;)*) => {
        pub static WEAK_LANG_ITEMS: &[LangItem] = &[$(LangItem::$item,)*];

        impl LangItem {
            pub fn is_weak(self) -> bool {
                matches!(self, $(LangItem::$item)|*)
            }

            pub fn link_name(self) -> Option<Symbol> {
                match self {
                    $( LangItem::$item => Some(sym::$sym),)*
                    _ => None,
                }
            }
        }
    }
}

macro_rules! decl_lang_items {
    ($($item:ident,)*) => {
        pub static DECL_LANG_ITEMS: &[LangItem] = &[$(LangItem::$item,)*];

        impl LangItem {
            pub fn is_decl(self) -> bool {
                matches!(self, $(LangItem::$item)|*)
            }
        }
    }
}

weak_lang_items! {
    PanicImpl,          rust_begin_unwind;
    EhPersonality,      rust_eh_personality;
    EhCatchTypeinfo,    rust_eh_catch_typeinfo;
}

decl_lang_items! {
    MemCpy,
    MemMove,
    MemSet,
    MemCmp,
    Bcmp,
    StrLen,
}
