//! Utility module for converting between hir_def ids and code_model wrappers.
//!
//! It's unclear if we need this long-term, but it's definitelly useful while we
//! are splitting the hir.

use hir_def::{AdtId, EnumVariantId, ModuleDefId};

use crate::{Adt, EnumVariant, ModuleDef};

macro_rules! from_id {
    ($(($id:path, $ty:path)),*) => {$(
        impl From<$id> for $ty {
            fn from(id: $id) -> $ty {
                $ty { id }
            }
        }
    )*}
}

from_id![
    (hir_def::ModuleId, crate::Module),
    (hir_def::StructId, crate::Struct),
    (hir_def::UnionId, crate::Union),
    (hir_def::EnumId, crate::Enum),
    (hir_def::TypeAliasId, crate::TypeAlias),
    (hir_def::TraitId, crate::Trait),
    (hir_def::StaticId, crate::Static),
    (hir_def::ConstId, crate::Const),
    (hir_def::FunctionId, crate::Function),
    (hir_expand::MacroDefId, crate::MacroDef)
];

impl From<AdtId> for Adt {
    fn from(id: AdtId) -> Self {
        match id {
            AdtId::StructId(it) => Adt::Struct(it.into()),
            AdtId::UnionId(it) => Adt::Union(it.into()),
            AdtId::EnumId(it) => Adt::Enum(it.into()),
        }
    }
}

impl From<EnumVariantId> for EnumVariant {
    fn from(id: EnumVariantId) -> Self {
        EnumVariant { parent: id.parent.into(), id: id.local_id }
    }
}

impl From<ModuleDefId> for ModuleDef {
    fn from(id: ModuleDefId) -> Self {
        match id {
            ModuleDefId::ModuleId(it) => ModuleDef::Module(it.into()),
            ModuleDefId::FunctionId(it) => ModuleDef::Function(it.into()),
            ModuleDefId::AdtId(it) => ModuleDef::Adt(it.into()),
            ModuleDefId::EnumVariantId(it) => ModuleDef::EnumVariant(it.into()),
            ModuleDefId::ConstId(it) => ModuleDef::Const(it.into()),
            ModuleDefId::StaticId(it) => ModuleDef::Static(it.into()),
            ModuleDefId::TraitId(it) => ModuleDef::Trait(it.into()),
            ModuleDefId::TypeAliasId(it) => ModuleDef::TypeAlias(it.into()),
            ModuleDefId::BuiltinType(it) => ModuleDef::BuiltinType(it),
        }
    }
}
