//! Item types.

use std::fmt;

use rustc_hir::def::{CtorOf, DefKind, MacroKinds};
use rustc_span::hygiene::MacroKind;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::clean;

macro_rules! item_type {
    ($($variant:ident = $number:literal,)+) => {

/// Item type. Corresponds to `clean::ItemEnum` variants.
///
/// The search index uses item types encoded as smaller numbers which equal to
/// discriminants. JavaScript then is used to decode them into the original value.
/// Consequently, every change to this type should be synchronized to
/// the `itemTypes` mapping table in `html/static/js/search.js`.
///
/// The search engine in search.js also uses item type numbers as a tie breaker when
/// sorting results. Keywords and primitives are given first because we want them to be easily
/// found by new users who don't know about advanced features like type filters. The rest are
/// mostly in an arbitrary order, but it's easier to test the search engine when
/// it's deterministic, and these are strictly finer-grained than language namespaces, so
/// using the path and the item type together to sort ensures that search sorting is stable.
///
/// In addition, code in `html::render` uses this enum to generate CSS classes, page prefixes, and
/// module headings. If you are adding to this enum and want to ensure that the sidebar also prints
/// a heading, edit the listing in `html/render.rs`, function `sidebar_module`. This uses an
/// ordering based on a helper function inside `item_module`, in the same file.
#[derive(Copy, PartialEq, Eq, Hash, Clone, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum ItemType {
    $($variant = $number,)+
}

impl Serialize for ItemType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (*self as u8).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ItemType {
    fn deserialize<D>(deserializer: D) -> Result<ItemType, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ItemTypeVisitor;
        impl<'de> de::Visitor<'de> for ItemTypeVisitor {
            type Value = ItemType;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "an integer between 0 and 27")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<ItemType, E> {
                Ok(match v {
                    $($number => ItemType::$variant,)+
                    _ => return Err(E::missing_field("unknown number for `ItemType` enum")),
                })
            }
        }
        deserializer.deserialize_any(ItemTypeVisitor)
    }
}

    }
}

item_type! {
    Keyword = 0,
    Primitive = 1,
    Module = 2,
    ExternCrate = 3,
    Import = 4,
    Struct = 5,
    Enum = 6,
    Function = 7,
    TypeAlias = 8,
    Static = 9,
    Trait = 10,
    Impl = 11,
    TyMethod = 12,
    Method = 13,
    StructField = 14,
    Variant = 15,
    Macro = 16,
    AssocType = 17,
    Constant = 18,
    AssocConst = 19,
    Union = 20,
    ForeignType = 21,
    // OpaqueTy used to be here, but it was removed in #127276
    ProcAttribute = 23,
    ProcDerive = 24,
    TraitAlias = 25,
    // This number is reserved for use in JavaScript
    // Generic = 26,
    Attribute = 27,
}

impl<'a> From<&'a clean::Item> for ItemType {
    fn from(item: &'a clean::Item) -> ItemType {
        let kind = match &item.kind {
            clean::StrippedItem(box item) => item,
            kind => kind,
        };

        match kind {
            clean::ModuleItem(..) => ItemType::Module,
            clean::ExternCrateItem { .. } => ItemType::ExternCrate,
            clean::ImportItem(..) => ItemType::Import,
            clean::StructItem(..) => ItemType::Struct,
            clean::UnionItem(..) => ItemType::Union,
            clean::EnumItem(..) => ItemType::Enum,
            clean::FunctionItem(..) => ItemType::Function,
            clean::TypeAliasItem(..) => ItemType::TypeAlias,
            clean::StaticItem(..) => ItemType::Static,
            clean::ConstantItem(..) => ItemType::Constant,
            clean::TraitItem(..) => ItemType::Trait,
            clean::ImplItem(..) => ItemType::Impl,
            clean::RequiredMethodItem(..) => ItemType::TyMethod,
            clean::MethodItem(..) => ItemType::Method,
            clean::StructFieldItem(..) => ItemType::StructField,
            clean::VariantItem(..) => ItemType::Variant,
            clean::ForeignFunctionItem(..) => ItemType::Function, // no ForeignFunction
            clean::ForeignStaticItem(..) => ItemType::Static,     // no ForeignStatic
            clean::MacroItem(..) => ItemType::Macro,
            clean::PrimitiveItem(..) => ItemType::Primitive,
            clean::RequiredAssocConstItem(..)
            | clean::ProvidedAssocConstItem(..)
            | clean::ImplAssocConstItem(..) => ItemType::AssocConst,
            clean::RequiredAssocTypeItem(..) | clean::AssocTypeItem(..) => ItemType::AssocType,
            clean::ForeignTypeItem => ItemType::ForeignType,
            clean::KeywordItem => ItemType::Keyword,
            clean::AttributeItem => ItemType::Attribute,
            clean::TraitAliasItem(..) => ItemType::TraitAlias,
            clean::ProcMacroItem(mac) => match mac.kind {
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
        Self::from_def_kind(other, None)
    }
}

impl ItemType {
    /// Depending on the parent kind, some variants have a different translation (like a `Method`
    /// becoming a `TyMethod`).
    pub(crate) fn from_def_kind(kind: DefKind, parent_kind: Option<DefKind>) -> Self {
        match kind {
            DefKind::Enum => Self::Enum,
            DefKind::Fn => Self::Function,
            DefKind::Mod => Self::Module,
            DefKind::Const => Self::Constant,
            DefKind::Static { .. } => Self::Static,
            DefKind::Struct => Self::Struct,
            DefKind::Union => Self::Union,
            DefKind::Trait => Self::Trait,
            DefKind::TyAlias => Self::TypeAlias,
            DefKind::TraitAlias => Self::TraitAlias,
            DefKind::Macro(MacroKinds::BANG) => ItemType::Macro,
            DefKind::Macro(MacroKinds::ATTR) => ItemType::ProcAttribute,
            DefKind::Macro(MacroKinds::DERIVE) => ItemType::ProcDerive,
            DefKind::Macro(_) => todo!("Handle macros with multiple kinds"),
            DefKind::ForeignTy => Self::ForeignType,
            DefKind::Variant => Self::Variant,
            DefKind::Field => Self::StructField,
            DefKind::AssocTy => Self::AssocType,
            DefKind::AssocFn if let Some(DefKind::Trait) = parent_kind => Self::TyMethod,
            DefKind::AssocFn => Self::Method,
            DefKind::Ctor(CtorOf::Struct, _) => Self::Struct,
            DefKind::Ctor(CtorOf::Variant, _) => Self::Variant,
            DefKind::AssocConst => Self::AssocConst,
            DefKind::TyParam
            | DefKind::ConstParam
            | DefKind::ExternCrate
            | DefKind::Use
            | DefKind::ForeignMod
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::OpaqueTy
            | DefKind::LifetimeParam
            | DefKind::GlobalAsm
            | DefKind::Impl { .. }
            | DefKind::Closure
            | DefKind::SyntheticCoroutineBody => Self::ForeignType,
        }
    }

    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            ItemType::Module => "mod",
            ItemType::ExternCrate => "externcrate",
            ItemType::Import => "import",
            ItemType::Struct => "struct",
            ItemType::Union => "union",
            ItemType::Enum => "enum",
            ItemType::Function => "fn",
            ItemType::TypeAlias => "type",
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
            ItemType::ProcAttribute => "attr",
            ItemType::ProcDerive => "derive",
            ItemType::TraitAlias => "traitalias",
            ItemType::Attribute => "attribute",
        }
    }
    pub(crate) fn is_method(&self) -> bool {
        matches!(self, ItemType::Method | ItemType::TyMethod)
    }
    pub(crate) fn is_adt(&self) -> bool {
        matches!(self, ItemType::Struct | ItemType::Union | ItemType::Enum)
    }
    /// Keep this the same as isFnLikeTy in search.js
    pub(crate) fn is_fn_like(&self) -> bool {
        matches!(self, ItemType::Function | ItemType::Method | ItemType::TyMethod)
    }
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
