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
#[derive(Copy, PartialEq, Clone)]
pub enum ItemType {
    Module          = 0,
    ExternCrate     = 1,
    Import          = 2,
    Struct          = 3,
    Enum            = 4,
    Function        = 5,
    Typedef         = 6,
    Static          = 7,
    Trait           = 8,
    Impl            = 9,
    TyMethod        = 10,
    Method          = 11,
    StructField     = 12,
    Variant         = 13,
    Macro           = 14,
    Primitive       = 15,
    AssociatedType  = 16,
    Constant        = 17,
}

impl ItemType {
    pub fn from_item(item: &clean::Item) -> ItemType {
        match item.inner {
            clean::ModuleItem(..)          => ItemType::Module,
            clean::ExternCrateItem(..)     => ItemType::ExternCrate,
            clean::ImportItem(..)          => ItemType::Import,
            clean::StructItem(..)          => ItemType::Struct,
            clean::EnumItem(..)            => ItemType::Enum,
            clean::FunctionItem(..)        => ItemType::Function,
            clean::TypedefItem(..)         => ItemType::Typedef,
            clean::StaticItem(..)          => ItemType::Static,
            clean::ConstantItem(..)        => ItemType::Constant,
            clean::TraitItem(..)           => ItemType::Trait,
            clean::ImplItem(..)            => ItemType::Impl,
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
            clean::TypeConst    => ItemType::Constant,
            clean::TypeVariant  => ItemType::Variant,
            clean::TypeTypedef  => ItemType::Typedef,
        }
    }

    pub fn to_static_str(&self) -> &'static str {
        match *self {
            ItemType::Module          => "mod",
            ItemType::ExternCrate     => "externcrate",
            ItemType::Import          => "import",
            ItemType::Struct          => "struct",
            ItemType::Enum            => "enum",
            ItemType::Function        => "fn",
            ItemType::Typedef         => "type",
            ItemType::Static          => "static",
            ItemType::Trait           => "trait",
            ItemType::Impl            => "impl",
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

impl fmt::String for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_static_str().fmt(f)
    }
}

