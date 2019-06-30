//! Item types.

use std::fmt;
use syntax::ext::base::MacroKind;
use crate::clean;

/// Item type. Corresponds to `clean::ItemEnum` variants.
///
/// The search index uses item types encoded as smaller numbers which equal to
/// discriminants. JavaScript then is used to decode them into the original value.
/// Consequently, every change to this type should be synchronized to
/// the `itemTypes` mapping table in `static/main.js`.
///
/// In addition, code in `html::render` uses this enum to generate CSS classes, page prefixes, and
/// module headings. If you are adding to this enum and want to ensure that the sidebar also prints
/// a heading, edit the listing in `html/render.rs`, function `sidebar_module`. This uses an
/// ordering based on a helper function inside `item_module`, in the same file.
#[derive(Copy, PartialEq, Eq, Clone, Debug, PartialOrd, Ord)]
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
    AssocType       = 16,
    Constant        = 17,
    AssocConst      = 18,
    Union           = 19,
    ForeignType     = 20,
    Keyword         = 21,
    Existential     = 22,
    ProcAttribute   = 23,
    ProcDerive      = 24,
    TraitAlias      = 25,
}


#[derive(Copy, Eq, PartialEq, Clone)]
pub enum NameSpace {
    Type,
    Value,
    Macro,
    Keyword,
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
            clean::ExistentialItem(..)     => ItemType::Existential,
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
            clean::AssocConstItem(..)      => ItemType::AssocConst,
            clean::AssocTypeItem(..)       => ItemType::AssocType,
            clean::ForeignTypeItem         => ItemType::ForeignType,
            clean::KeywordItem(..)         => ItemType::Keyword,
            clean::TraitAliasItem(..)      => ItemType::TraitAlias,
            clean::ProcMacroItem(ref mac)  => match mac.kind {
                MacroKind::Bang            => ItemType::Macro,
                MacroKind::Attr            => ItemType::ProcAttribute,
                MacroKind::Derive          => ItemType::ProcDerive,
                MacroKind::ProcMacroStub   => unreachable!(),
            }
            clean::StrippedItem(..)        => unreachable!(),
        }
    }
}

impl From<clean::TypeKind> for ItemType {
    fn from(kind: clean::TypeKind) -> ItemType {
        match kind {
            clean::TypeKind::Struct     => ItemType::Struct,
            clean::TypeKind::Union      => ItemType::Union,
            clean::TypeKind::Enum       => ItemType::Enum,
            clean::TypeKind::Function   => ItemType::Function,
            clean::TypeKind::Trait      => ItemType::Trait,
            clean::TypeKind::Module     => ItemType::Module,
            clean::TypeKind::Static     => ItemType::Static,
            clean::TypeKind::Const      => ItemType::Constant,
            clean::TypeKind::Typedef    => ItemType::Typedef,
            clean::TypeKind::Foreign    => ItemType::ForeignType,
            clean::TypeKind::Macro      => ItemType::Macro,
            clean::TypeKind::Attr       => ItemType::ProcAttribute,
            clean::TypeKind::Derive     => ItemType::ProcDerive,
            clean::TypeKind::TraitAlias => ItemType::TraitAlias,
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
            ItemType::AssocType       => "associatedtype",
            ItemType::Constant        => "constant",
            ItemType::AssocConst      => "associatedconstant",
            ItemType::ForeignType     => "foreigntype",
            ItemType::Keyword         => "keyword",
            ItemType::Existential     => "existential",
            ItemType::ProcAttribute   => "attr",
            ItemType::ProcDerive      => "derive",
            ItemType::TraitAlias      => "traitalias",
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
            ItemType::AssocType |
            ItemType::Existential |
            ItemType::TraitAlias |
            ItemType::ForeignType => NameSpace::Type,

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
            ItemType::AssocConst => NameSpace::Value,

            ItemType::Macro |
            ItemType::ProcAttribute |
            ItemType::ProcDerive => NameSpace::Macro,

            ItemType::Keyword => NameSpace::Keyword,
        }
    }
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.css_class().fmt(f)
    }
}

pub const NAMESPACE_TYPE: &'static str = "t";
pub const NAMESPACE_VALUE: &'static str = "v";
pub const NAMESPACE_MACRO: &'static str = "m";
pub const NAMESPACE_KEYWORD: &'static str = "k";

impl NameSpace {
    pub fn to_static_str(&self) -> &'static str {
        match *self {
            NameSpace::Type => NAMESPACE_TYPE,
            NameSpace::Value => NAMESPACE_VALUE,
            NameSpace::Macro => NAMESPACE_MACRO,
            NameSpace::Keyword => NAMESPACE_KEYWORD,
        }
    }
}

impl fmt::Display for NameSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_static_str().fmt(f)
    }
}
