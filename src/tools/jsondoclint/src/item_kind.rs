use rustdoc_json_types::{Item, ItemEnum, ItemKind, ItemSummary};

/// A universal way to represent an [`ItemEnum`] or [`ItemKind`]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Kind {
    Module,
    ExternCrate,
    Use,
    Struct,
    StructField,
    Union,
    Enum,
    Variant,
    Function,
    TypeAlias,
    Constant,
    Trait,
    TraitAlias,
    Impl,
    Static,
    ExternType,
    Macro,
    ProcAttribute,
    ProcDerive,
    AssocConst,
    AssocType,
    Primitive,
    Keyword,
    Attribute,
    // Not in ItemKind
    ProcMacro,
}

impl Kind {
    pub fn can_appear_in_mod(self) -> bool {
        use Kind::*;
        match self {
            Module => true,
            ExternCrate => true,
            Use => true,
            Union => true,
            Struct => true,
            Enum => true,
            Function => true,
            Trait => true,
            TraitAlias => true,
            Impl => true,
            TypeAlias => true,
            Constant => true,
            Static => true,
            Macro => true,
            ProcMacro => true,
            Primitive => true,
            ExternType => true,

            // FIXME(adotinthevoid): I'm not sure if these are correct
            Attribute => false,
            Keyword => false,
            ProcAttribute => false,
            ProcDerive => false,

            // Only in traits
            AssocConst => false,
            AssocType => false,

            StructField => false, // Only in structs or variants
            Variant => false,     // Only in enums
        }
    }

    pub fn can_appear_in_import(self) -> bool {
        match self {
            Kind::Variant => true,
            Kind::Use => false,
            other => other.can_appear_in_mod(),
        }
    }

    pub fn can_appear_in_glob_import(self) -> bool {
        match self {
            Kind::Module => true,
            Kind::Enum => true,
            _ => false,
        }
    }

    pub fn can_appear_in_trait(self) -> bool {
        match self {
            Kind::AssocConst => true,
            Kind::AssocType => true,
            Kind::Function => true,

            Kind::Module => false,
            Kind::ExternCrate => false,
            Kind::Use => false,
            Kind::Struct => false,
            Kind::StructField => false,
            Kind::Union => false,
            Kind::Enum => false,
            Kind::Variant => false,
            Kind::TypeAlias => false,
            Kind::Constant => false,
            Kind::Trait => false,
            Kind::TraitAlias => false,
            Kind::Impl => false,
            Kind::Static => false,
            Kind::ExternType => false,
            Kind::Macro => false,
            Kind::ProcAttribute => false,
            Kind::ProcDerive => false,
            Kind::Primitive => false,
            Kind::Keyword => false,
            Kind::ProcMacro => false,
            Kind::Attribute => false,
        }
    }

    pub fn is_struct_field(self) -> bool {
        matches!(self, Kind::StructField)
    }
    pub fn is_module(self) -> bool {
        matches!(self, Kind::Module)
    }
    pub fn is_impl(self) -> bool {
        matches!(self, Kind::Impl)
    }
    pub fn is_variant(self) -> bool {
        matches!(self, Kind::Variant)
    }
    pub fn is_trait_or_alias(self) -> bool {
        matches!(self, Kind::Trait | Kind::TraitAlias)
    }
    pub fn is_type(self) -> bool {
        matches!(self, Kind::Struct | Kind::Enum | Kind::Union | Kind::TypeAlias)
    }

    pub fn from_item(i: &Item) -> Self {
        use Kind::*;
        match i.inner {
            ItemEnum::Module(_) => Module,
            ItemEnum::Use(_) => Use,
            ItemEnum::Union(_) => Union,
            ItemEnum::Struct(_) => Struct,
            ItemEnum::StructField(_) => StructField,
            ItemEnum::Enum(_) => Enum,
            ItemEnum::Variant(_) => Variant,
            ItemEnum::Function(_) => Function,
            ItemEnum::Trait(_) => Trait,
            ItemEnum::TraitAlias(_) => TraitAlias,
            ItemEnum::Impl(_) => Impl,
            ItemEnum::TypeAlias(_) => TypeAlias,
            ItemEnum::Constant { .. } => Constant,
            ItemEnum::Static(_) => Static,
            ItemEnum::Macro(_) => Macro,
            ItemEnum::ProcMacro(_) => ProcMacro,
            ItemEnum::Primitive(_) => Primitive,
            ItemEnum::ExternType => ExternType,
            ItemEnum::ExternCrate { .. } => ExternCrate,
            ItemEnum::AssocConst { .. } => AssocConst,
            ItemEnum::AssocType { .. } => AssocType,
        }
    }

    pub fn from_summary(s: &ItemSummary) -> Self {
        use Kind::*;
        match s.kind {
            ItemKind::AssocConst => AssocConst,
            ItemKind::AssocType => AssocType,
            ItemKind::Attribute => Attribute,
            ItemKind::Constant => Constant,
            ItemKind::Enum => Enum,
            ItemKind::ExternCrate => ExternCrate,
            ItemKind::ExternType => ExternType,
            ItemKind::Function => Function,
            ItemKind::Impl => Impl,
            ItemKind::Use => Use,
            ItemKind::Keyword => Keyword,
            ItemKind::Macro => Macro,
            ItemKind::Module => Module,
            ItemKind::Primitive => Primitive,
            ItemKind::ProcAttribute => ProcAttribute,
            ItemKind::ProcDerive => ProcDerive,
            ItemKind::Static => Static,
            ItemKind::Struct => Struct,
            ItemKind::StructField => StructField,
            ItemKind::Trait => Trait,
            ItemKind::TraitAlias => TraitAlias,
            ItemKind::TypeAlias => TypeAlias,
            ItemKind::Union => Union,
            ItemKind::Variant => Variant,
        }
    }
}
