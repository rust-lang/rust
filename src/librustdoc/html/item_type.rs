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
    AssociatedConst = 18,
    Union           = 19,
}


#[derive(Copy, Eq, PartialEq, Clone)]
pub enum NameSpace {
    Type,
    Value,
    Macro,
}

impl<'a> From<&'a clean::Item> for ItemType {
    fn from(item: &'a clean::Item) -> ItemType {
        let inner = match item.inner {
            clean::StrippedItem(box ref item) => item,
            ref inner@_ => inner,
        };

        match *inner {
            clean::ModuleItem(..)          => ItemType::Module,
            clean::ExternCrateItem(..)     => ItemType::ExternCrate,
            clean::ImportItem(..)          => ItemType::Import,
            clean::StructItem(..)          => ItemType::Struct,
            clean::UnionItem(..)           => ItemType::Union,
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
            clean::AssociatedConstItem(..) => ItemType::AssociatedConst,
            clean::AssociatedTypeItem(..)  => ItemType::AssociatedType,
            clean::DefaultImplItem(..)     => ItemType::Impl,
            clean::StrippedItem(..)        => unreachable!(),
        }
    }
}

impl From<clean::TypeKind> for ItemType {
    fn from(kind: clean::TypeKind) -> ItemType {
        match kind {
            clean::TypeKind::Struct   => ItemType::Struct,
            clean::TypeKind::Union    => ItemType::Union,
            clean::TypeKind::Enum     => ItemType::Enum,
            clean::TypeKind::Function => ItemType::Function,
            clean::TypeKind::Trait    => ItemType::Trait,
            clean::TypeKind::Module   => ItemType::Module,
            clean::TypeKind::Static   => ItemType::Static,
            clean::TypeKind::Const    => ItemType::Constant,
            clean::TypeKind::Variant  => ItemType::Variant,
            clean::TypeKind::Typedef  => ItemType::Typedef,
        }
    }
}

impl ItemType {
    pub fn css_class(&self) -> &'static str {
        match *self {
            ItemType::Module          => "mod",
            ItemType::ExternCrate     => "externcrate",
            ItemType::Import          => "import",
            ItemType::Struct          => "struct",
            ItemType::Union           => "union",
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
            ItemType::AssociatedConst => "associatedconstant",
        }
    }

    pub fn name_space(&self) -> NameSpace {
        match *self {
            ItemType::Struct |
            ItemType::Union |
            ItemType::Enum |
            ItemType::Module |
            ItemType::Typedef |
            ItemType::Trait |
            ItemType::Primitive |
            ItemType::AssociatedType => NameSpace::Type,

            ItemType::ExternCrate |
            ItemType::Import |
            ItemType::Function |
            ItemType::Static |
            ItemType::Impl |
            ItemType::TyMethod |
            ItemType::Method |
            ItemType::StructField |
            ItemType::Variant |
            ItemType::Constant |
            ItemType::AssociatedConst => NameSpace::Value,

            ItemType::Macro => NameSpace::Macro,
        }
    }
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.css_class().fmt(f)
    }
}

pub const NAMESPACE_TYPE: &'static str = "t";
pub const NAMESPACE_VALUE: &'static str = "v";
pub const NAMESPACE_MACRO: &'static str = "m";

impl NameSpace {
    pub fn to_static_str(&self) -> &'static str {
        match *self {
            NameSpace::Type => NAMESPACE_TYPE,
            NameSpace::Value => NAMESPACE_VALUE,
            NameSpace::Macro => NAMESPACE_MACRO,
        }
    }
}

impl fmt::Display for NameSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_static_str().fmt(f)
    }
}
