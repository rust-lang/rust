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
#[deriving(PartialEq, Clone)]
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
    // we used to have ForeignFunction and ForeignStatic. they are retired now.
    Macro           = 15,
    Primitive       = 16,
    AssociatedType  = 17,
    Constant        = 18,
}

impl Copy for ItemType {}

impl ItemType {
    pub fn from_item(item: &clean::Item) -> ItemType {
        match item.inner {
            clean::ModuleItem(..)          => ItemType::Module,
            clean::StructItem(..)          => ItemType::Struct,
            clean::EnumItem(..)            => ItemType::Enum,
            clean::FunctionItem(..)        => ItemType::Function,
            clean::TypedefItem(..)         => ItemType::Typedef,
            clean::StaticItem(..)          => ItemType::Static,
            clean::ConstantItem(..)        => ItemType::Constant,
            clean::TraitItem(..)           => ItemType::Trait,
            clean::ImplItem(..)            => ItemType::Impl,
            clean::ViewItemItem(..)        => ItemType::ViewItem,
            clean::TyMethodItem(..)        => ItemType::TyMethod,
            clean::MethodItem(..)          => ItemType::Method,
            clean::StructFieldItem(..)     => ItemType::StructField,
            clean::VariantItem(..)         => ItemType::Variant,
            clean::ForeignFunctionItem(..) => ItemType::Function, // no ForeignFunction
            clean::ForeignStaticItem(..)   => ItemType::Static, // no ForeignStatic
            clean::MacroItem(..)           => ItemType::Macro,
            clean::PrimitiveItem(..)       => ItemType::Primitive,
            clean::AssociatedTypeItem(..)  => ItemType::AssociatedType,
        }
    }

    pub fn from_type_kind(kind: clean::TypeKind) -> ItemType {
        match kind {
            clean::TypeStruct   => ItemType::Struct,
            clean::TypeEnum     => ItemType::Enum,
            clean::TypeFunction => ItemType::Function,
            clean::TypeTrait    => ItemType::Trait,
            clean::TypeModule   => ItemType::Module,
            clean::TypeStatic   => ItemType::Static,
            clean::TypeVariant  => ItemType::Variant,
            clean::TypeTypedef  => ItemType::Typedef,
        }
    }

    pub fn to_static_str(&self) -> &'static str {
        match *self {
            ItemType::Module          => "mod",
            ItemType::Struct          => "struct",
            ItemType::Enum            => "enum",
            ItemType::Function        => "fn",
            ItemType::Typedef         => "type",
            ItemType::Static          => "static",
            ItemType::Trait           => "trait",
            ItemType::Impl            => "impl",
            ItemType::ViewItem        => "viewitem",
            ItemType::TyMethod        => "tymethod",
            ItemType::Method          => "method",
            ItemType::StructField     => "structfield",
            ItemType::Variant         => "variant",
            ItemType::Macro           => "macro",
            ItemType::Primitive       => "primitive",
            ItemType::AssociatedType  => "associatedtype",
            ItemType::Constant        => "constant",
        }
    }
}

impl fmt::Show for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_static_str().fmt(f)
    }
}

