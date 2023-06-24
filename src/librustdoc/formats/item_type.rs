//! Item types.

use std::fmt;

use serde::{Serialize, Serializer};

use rustc_hir::def::DefKind;
use rustc_span::hygiene::MacroKind;

use crate::clean;

/// Item type. Corresponds to `clean::ItemEnum` variants.
///
/// The search index uses item types encoded as smaller numbers which equal to
/// discriminants. JavaScript then is used to decode them into the original value.
/// Consequently, every change to this type should be synchronized to
/// the `itemTypes` mapping table in `html/static/js/search.js`.
///
/// In addition, code in `html::render` uses this enum to generate CSS classes, page prefixes, and
/// module headings. If you are adding to this enum and want to ensure that the sidebar also prints
/// a heading, edit the listing in `html/render.rs`, function `sidebar_module`. This uses an
/// ordering based on a helper function inside `item_module`, in the same file.
#[derive(Copy, PartialEq, Eq, Hash, Clone, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum ItemType {
    Module = 0,
    ExternCrate = 1,
    Import = 2,
    Struct = 3,
    Enum = 4,
    Function = 5,
    Typedef = 6,
    Static = 7,
    Trait = 8,
    Impl = 9,
    TyMethod = 10,
    Method = 11,
    StructField = 12,
    Variant = 13,
    Macro = 14,
    Primitive = 15,
    AssocType = 16,
    Constant = 17,
    AssocConst = 18,
    Union = 19,
    ForeignType = 20,
    Keyword = 21,
    OpaqueTy = 22,
    ProcAttribute = 23,
    ProcDerive = 24,
    TraitAlias = 25,
}

impl Serialize for ItemType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (*self as u8).serialize(serializer)
    }
}

impl<'a> From<&'a clean::Item> for ItemType {
    fn from(item: &'a clean::Item) -> ItemType {
        let kind = match *item.kind {
            clean::StrippedItem(box ref item) => item,
            ref kind => kind,
        };

        match *kind {
            clean::ModuleItem(..) => ItemType::Module,
            clean::ExternCrateItem { .. } => ItemType::ExternCrate,
            clean::ImportItem(..) => ItemType::Import,
            clean::StructItem(..) => ItemType::Struct,
            clean::UnionItem(..) => ItemType::Union,
            clean::EnumItem(..) => ItemType::Enum,
            clean::FunctionItem(..) => ItemType::Function,
            clean::TypedefItem(..) => ItemType::Typedef,
            clean::OpaqueTyItem(..) => ItemType::OpaqueTy,
            clean::StaticItem(..) => ItemType::Static,
            clean::ConstantItem(..) => ItemType::Constant,
            clean::TraitItem(..) => ItemType::Trait,
            clean::ImplItem(..) => ItemType::Impl,
            clean::TyMethodItem(..) => ItemType::TyMethod,
            clean::MethodItem(..) => ItemType::Method,
            clean::StructFieldItem(..) => ItemType::StructField,
            clean::VariantItem(..) => ItemType::Variant,
            clean::ForeignFunctionItem(..) => ItemType::Function, // no ForeignFunction
            clean::ForeignStaticItem(..) => ItemType::Static,     // no ForeignStatic
            clean::MacroItem(..) => ItemType::Macro,
            clean::PrimitiveItem(..) => ItemType::Primitive,
            clean::TyAssocConstItem(..) | clean::AssocConstItem(..) => ItemType::AssocConst,
            clean::TyAssocTypeItem(..) | clean::AssocTypeItem(..) => ItemType::AssocType,
            clean::ForeignTypeItem => ItemType::ForeignType,
            clean::KeywordItem => ItemType::Keyword,
            clean::TraitAliasItem(..) => ItemType::TraitAlias,
            clean::ProcMacroItem(ref mac) => match mac.kind {
                MacroKind::Bang => ItemType::Macro,
                MacroKind::Attr => ItemType::ProcAttribute,
                MacroKind::Derive => ItemType::ProcDerive,
            },
            clean::StrippedItem(..) => unreachable!(),
        }
    }
}

impl From<DefKind> for ItemType {
    fn from(other: DefKind) -> Self {
        match other {
            DefKind::Enum => Self::Enum,
            DefKind::Fn => Self::Function,
            DefKind::Mod => Self::Module,
            DefKind::Const => Self::Constant,
            DefKind::Static(_) => Self::Static,
            DefKind::Struct => Self::Struct,
            DefKind::Union => Self::Union,
            DefKind::Trait => Self::Trait,
            DefKind::TyAlias => Self::Typedef,
            DefKind::TraitAlias => Self::TraitAlias,
            DefKind::Macro(kind) => match kind {
                MacroKind::Bang => ItemType::Macro,
                MacroKind::Attr => ItemType::ProcAttribute,
                MacroKind::Derive => ItemType::ProcDerive,
            },
            DefKind::ForeignTy
            | DefKind::Variant
            | DefKind::AssocTy
            | DefKind::TyParam
            | DefKind::ConstParam
            | DefKind::Ctor(..)
            | DefKind::AssocFn
            | DefKind::AssocConst
            | DefKind::ExternCrate
            | DefKind::Use
            | DefKind::ForeignMod
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::OpaqueTy
            | DefKind::Field
            | DefKind::LifetimeParam
            | DefKind::GlobalAsm
            | DefKind::Impl { .. }
            | DefKind::Closure
            | DefKind::Generator => Self::ForeignType,
        }
    }
}

impl ItemType {
    pub(crate) fn as_str(&self) -> &'static str {
        match *self {
            ItemType::Module => "mod",
            ItemType::ExternCrate => "externcrate",
            ItemType::Import => "import",
            ItemType::Struct => "struct",
            ItemType::Union => "union",
            ItemType::Enum => "enum",
            ItemType::Function => "fn",
            ItemType::Typedef => "type",
            ItemType::Static => "static",
            ItemType::Trait => "trait",
            ItemType::Impl => "impl",
            ItemType::TyMethod => "tymethod",
            ItemType::Method => "method",
            ItemType::StructField => "structfield",
            ItemType::Variant => "variant",
            ItemType::Macro => "macro",
            ItemType::Primitive => "primitive",
            ItemType::AssocType => "associatedtype",
            ItemType::Constant => "constant",
            ItemType::AssocConst => "associatedconstant",
            ItemType::ForeignType => "foreigntype",
            ItemType::Keyword => "keyword",
            ItemType::OpaqueTy => "opaque",
            ItemType::ProcAttribute => "attr",
            ItemType::ProcDerive => "derive",
            ItemType::TraitAlias => "traitalias",
        }
    }
    pub(crate) fn is_method(&self) -> bool {
        matches!(*self, ItemType::Method | ItemType::TyMethod)
    }
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
