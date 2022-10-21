use rustdoc_json_types::{Item, ItemEnum, ItemKind, ItemSummary};

/// A univeral way to represent an [`ItemEnum`] or [`ItemKind`]
#[derive(Debug)]
pub(crate) enum Kind {
    Module,
    ExternCrate,
    Import,
    Struct,
    StructField,
    Union,
    Enum,
    Variant,
    Function,
    Typedef,
    OpaqueTy,
    Constant,
    Trait,
    TraitAlias,
    Method,
    Impl,
    Static,
    ForeignType,
    Macro,
    ProcAttribute,
    ProcDerive,
    AssocConst,
    AssocType,
    Primitive,
    Keyword,
    // Not in ItemKind
    ProcMacro,
}

impl Kind {
    pub fn can_appear_in_mod(self) -> bool {
        use Kind::*;
        match self {
            Module => true,
            ExternCrate => true,
            Import => true,
            Union => true,
            Struct => true,
            Enum => true,
            Function => true,
            Trait => true,
            TraitAlias => true,
            Impl => true,
            Typedef => true,
            Constant => true,
            Static => true,
            Macro => true,
            ProcMacro => true,
            Primitive => true,
            ForeignType => true,

            // FIXME(adotinthevoid): I'm not sure if these are corrent
            Keyword => false,
            OpaqueTy => false,
            ProcAttribute => false,
            ProcDerive => false,

            // Only in traits
            AssocConst => false,
            AssocType => false,
            Method => false,

            StructField => false, // Only in structs or variants
            Variant => false,     // Only in enums
        }
    }

    pub fn can_appear_in_trait(self) -> bool {
        match self {
            Kind::AssocConst => true,
            Kind::AssocType => true,
            Kind::Method => true,

            Kind::Module => false,
            Kind::ExternCrate => false,
            Kind::Import => false,
            Kind::Struct => false,
            Kind::StructField => false,
            Kind::Union => false,
            Kind::Enum => false,
            Kind::Variant => false,
            Kind::Function => false,
            Kind::Typedef => false,
            Kind::OpaqueTy => false,
            Kind::Constant => false,
            Kind::Trait => false,
            Kind::TraitAlias => false,
            Kind::Impl => false,
            Kind::Static => false,
            Kind::ForeignType => false,
            Kind::Macro => false,
            Kind::ProcAttribute => false,
            Kind::ProcDerive => false,
            Kind::Primitive => false,
            Kind::Keyword => false,
            Kind::ProcMacro => false,
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
    pub fn is_trait(self) -> bool {
        matches!(self, Kind::Trait)
    }
    pub fn is_struct_enum_union(self) -> bool {
        matches!(self, Kind::Struct | Kind::Enum | Kind::Union)
    }

    pub fn from_item(i: &Item) -> Self {
        use Kind::*;
        match i.inner {
            ItemEnum::Module(_) => Module,
            ItemEnum::Import(_) => Import,
            ItemEnum::Union(_) => Union,
            ItemEnum::Struct(_) => Struct,
            ItemEnum::StructField(_) => StructField,
            ItemEnum::Enum(_) => Enum,
            ItemEnum::Variant(_) => Variant,
            ItemEnum::Function(_) => Function,
            ItemEnum::Trait(_) => Trait,
            ItemEnum::TraitAlias(_) => TraitAlias,
            ItemEnum::Method(_) => Method,
            ItemEnum::Impl(_) => Impl,
            ItemEnum::Typedef(_) => Typedef,
            ItemEnum::OpaqueTy(_) => OpaqueTy,
            ItemEnum::Constant(_) => Constant,
            ItemEnum::Static(_) => Static,
            ItemEnum::Macro(_) => Macro,
            ItemEnum::ProcMacro(_) => ProcMacro,
            ItemEnum::Primitive(_) => Primitive,
            ItemEnum::ForeignType => ForeignType,
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
            ItemKind::Constant => Constant,
            ItemKind::Enum => Enum,
            ItemKind::ExternCrate => ExternCrate,
            ItemKind::ForeignType => ForeignType,
            ItemKind::Function => Function,
            ItemKind::Impl => Impl,
            ItemKind::Import => Import,
            ItemKind::Keyword => Keyword,
            ItemKind::Macro => Macro,
            ItemKind::Method => Method,
            ItemKind::Module => Module,
            ItemKind::OpaqueTy => OpaqueTy,
            ItemKind::Primitive => Primitive,
            ItemKind::ProcAttribute => ProcAttribute,
            ItemKind::ProcDerive => ProcDerive,
            ItemKind::Static => Static,
            ItemKind::Struct => Struct,
            ItemKind::StructField => StructField,
            ItemKind::Trait => Trait,
            ItemKind::TraitAlias => TraitAlias,
            ItemKind::Typedef => Typedef,
            ItemKind::Union => Union,
            ItemKind::Variant => Variant,
        }
    }
}
