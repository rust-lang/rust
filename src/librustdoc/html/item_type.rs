// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Item types.

use std::fmt;
use clean;

/// Item type. Corresponds to `clean::ItemEnum` variants.
///
/// The search index uses item types encoded as smaller numbers which equal to
/// discriminants. JavaScript then is used to decode them into the original value.
/// Consequently, every change to this type should be synchronized to
/// the `itemTypes` mapping table in `static/main.js`.
#[deriving(Eq, Clone)]
pub enum ItemType {
    Module          = 0,
    Struct          = 1,
    Enum            = 2,
    Function        = 3,
    Typedef         = 4,
    Static          = 5,
    Trait           = 6,
    Impl            = 7,
    ViewItem        = 8,
    TyMethod        = 9,
    Method          = 10,
    StructField     = 11,
    Variant         = 12,
    ForeignFunction = 13,
    ForeignStatic   = 14,
    Macro           = 15,
}

impl ItemType {
    pub fn to_static_str(&self) -> &'static str {
        match *self {
            Module          => "mod",
            Struct          => "struct",
            Enum            => "enum",
            Function        => "fn",
            Typedef         => "typedef",
            Static          => "static",
            Trait           => "trait",
            Impl            => "impl",
            ViewItem        => "viewitem",
            TyMethod        => "tymethod",
            Method          => "method",
            StructField     => "structfield",
            Variant         => "variant",
            ForeignFunction => "ffi",
            ForeignStatic   => "ffs",
            Macro           => "macro",
        }
    }
}

impl fmt::Show for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_static_str().fmt(f)
    }
}

impl fmt::Unsigned for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (*self as uint).fmt(f)
    }
}

pub fn shortty(item: &clean::Item) -> ItemType {
    match item.inner {
        clean::ModuleItem(..)          => Module,
        clean::StructItem(..)          => Struct,
        clean::EnumItem(..)            => Enum,
        clean::FunctionItem(..)        => Function,
        clean::TypedefItem(..)         => Typedef,
        clean::StaticItem(..)          => Static,
        clean::TraitItem(..)           => Trait,
        clean::ImplItem(..)            => Impl,
        clean::ViewItemItem(..)        => ViewItem,
        clean::TyMethodItem(..)        => TyMethod,
        clean::MethodItem(..)          => Method,
        clean::StructFieldItem(..)     => StructField,
        clean::VariantItem(..)         => Variant,
        clean::ForeignFunctionItem(..) => ForeignFunction,
        clean::ForeignStaticItem(..)   => ForeignStatic,
        clean::MacroItem(..)           => Macro,
    }
}

