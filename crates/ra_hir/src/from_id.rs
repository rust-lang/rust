//! Utility module for converting between hir_def ids and code_model wrappers.
//!
//! It's unclear if we need this long-term, but it's definitelly useful while we
//! are splitting the hir.

use hir_def::{
    expr::PatId, AdtId, AssocItemId, AttrDefId, DefWithBodyId, EnumVariantId, FieldId,
    GenericDefId, ModuleDefId, VariantId,
};

use crate::{
    code_model::ItemInNs, Adt, AssocItem, AttrDef, DefWithBody, EnumVariant, Field, GenericDef,
    Local, MacroDef, ModuleDef, VariantDef,
};

macro_rules! from_id {
    ($(($id:path, $ty:path)),*) => {$(
        impl From<$id> for $ty {
            fn from(id: $id) -> $ty {
                $ty { id }
            }
        }
        impl From<$ty> for $id {
            fn from(ty: $ty) -> $id {
                ty.id
            }
        }
    )*}
}

from_id![
    (ra_db::CrateId, crate::Crate),
    (hir_def::ModuleId, crate::Module),
    (hir_def::StructId, crate::Struct),
    (hir_def::UnionId, crate::Union),
    (hir_def::EnumId, crate::Enum),
    (hir_def::TypeAliasId, crate::TypeAlias),
    (hir_def::TraitId, crate::Trait),
    (hir_def::StaticId, crate::Static),
    (hir_def::ConstId, crate::Const),
    (hir_def::FunctionId, crate::Function),
    (hir_def::ImplId, crate::ImplDef),
    (hir_def::TypeParamId, crate::TypeParam),
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

impl From<Adt> for AdtId {
    fn from(id: Adt) -> Self {
        match id {
            Adt::Struct(it) => AdtId::StructId(it.id),
            Adt::Union(it) => AdtId::UnionId(it.id),
            Adt::Enum(it) => AdtId::EnumId(it.id),
        }
    }
}

impl From<EnumVariantId> for EnumVariant {
    fn from(id: EnumVariantId) -> Self {
        EnumVariant { parent: id.parent.into(), id: id.local_id }
    }
}

impl From<EnumVariant> for EnumVariantId {
    fn from(def: EnumVariant) -> Self {
        EnumVariantId { parent: def.parent.id, local_id: def.id }
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

impl From<ModuleDef> for ModuleDefId {
    fn from(id: ModuleDef) -> Self {
        match id {
            ModuleDef::Module(it) => ModuleDefId::ModuleId(it.into()),
            ModuleDef::Function(it) => ModuleDefId::FunctionId(it.into()),
            ModuleDef::Adt(it) => ModuleDefId::AdtId(it.into()),
            ModuleDef::EnumVariant(it) => ModuleDefId::EnumVariantId(it.into()),
            ModuleDef::Const(it) => ModuleDefId::ConstId(it.into()),
            ModuleDef::Static(it) => ModuleDefId::StaticId(it.into()),
            ModuleDef::Trait(it) => ModuleDefId::TraitId(it.into()),
            ModuleDef::TypeAlias(it) => ModuleDefId::TypeAliasId(it.into()),
            ModuleDef::BuiltinType(it) => ModuleDefId::BuiltinType(it),
        }
    }
}

impl From<DefWithBody> for DefWithBodyId {
    fn from(def: DefWithBody) -> Self {
        match def {
            DefWithBody::Function(it) => DefWithBodyId::FunctionId(it.id),
            DefWithBody::Static(it) => DefWithBodyId::StaticId(it.id),
            DefWithBody::Const(it) => DefWithBodyId::ConstId(it.id),
        }
    }
}

impl From<DefWithBodyId> for DefWithBody {
    fn from(def: DefWithBodyId) -> Self {
        match def {
            DefWithBodyId::FunctionId(it) => DefWithBody::Function(it.into()),
            DefWithBodyId::StaticId(it) => DefWithBody::Static(it.into()),
            DefWithBodyId::ConstId(it) => DefWithBody::Const(it.into()),
        }
    }
}

impl From<AssocItemId> for AssocItem {
    fn from(def: AssocItemId) -> Self {
        match def {
            AssocItemId::FunctionId(it) => AssocItem::Function(it.into()),
            AssocItemId::TypeAliasId(it) => AssocItem::TypeAlias(it.into()),
            AssocItemId::ConstId(it) => AssocItem::Const(it.into()),
        }
    }
}

impl From<GenericDef> for GenericDefId {
    fn from(def: GenericDef) -> Self {
        match def {
            GenericDef::Function(it) => GenericDefId::FunctionId(it.id),
            GenericDef::Adt(it) => GenericDefId::AdtId(it.into()),
            GenericDef::Trait(it) => GenericDefId::TraitId(it.id),
            GenericDef::TypeAlias(it) => GenericDefId::TypeAliasId(it.id),
            GenericDef::ImplDef(it) => GenericDefId::ImplId(it.id),
            GenericDef::EnumVariant(it) => {
                GenericDefId::EnumVariantId(EnumVariantId { parent: it.parent.id, local_id: it.id })
            }
            GenericDef::Const(it) => GenericDefId::ConstId(it.id),
        }
    }
}

impl From<Adt> for GenericDefId {
    fn from(id: Adt) -> Self {
        match id {
            Adt::Struct(it) => it.id.into(),
            Adt::Union(it) => it.id.into(),
            Adt::Enum(it) => it.id.into(),
        }
    }
}

impl From<VariantId> for VariantDef {
    fn from(def: VariantId) -> Self {
        match def {
            VariantId::StructId(it) => VariantDef::Struct(it.into()),
            VariantId::EnumVariantId(it) => VariantDef::EnumVariant(it.into()),
            VariantId::UnionId(it) => VariantDef::Union(it.into()),
        }
    }
}

impl From<VariantDef> for VariantId {
    fn from(def: VariantDef) -> Self {
        match def {
            VariantDef::Struct(it) => VariantId::StructId(it.id),
            VariantDef::EnumVariant(it) => VariantId::EnumVariantId(it.into()),
            VariantDef::Union(it) => VariantId::UnionId(it.id),
        }
    }
}

impl From<Field> for FieldId {
    fn from(def: Field) -> Self {
        FieldId { parent: def.parent.into(), local_id: def.id }
    }
}

impl From<FieldId> for Field {
    fn from(def: FieldId) -> Self {
        Field { parent: def.parent.into(), id: def.local_id }
    }
}

impl From<AttrDef> for AttrDefId {
    fn from(def: AttrDef) -> Self {
        match def {
            AttrDef::Module(it) => AttrDefId::ModuleId(it.id),
            AttrDef::Field(it) => AttrDefId::FieldId(it.into()),
            AttrDef::Adt(it) => AttrDefId::AdtId(it.into()),
            AttrDef::Function(it) => AttrDefId::FunctionId(it.id),
            AttrDef::EnumVariant(it) => AttrDefId::EnumVariantId(it.into()),
            AttrDef::Static(it) => AttrDefId::StaticId(it.id),
            AttrDef::Const(it) => AttrDefId::ConstId(it.id),
            AttrDef::Trait(it) => AttrDefId::TraitId(it.id),
            AttrDef::TypeAlias(it) => AttrDefId::TypeAliasId(it.id),
            AttrDef::MacroDef(it) => AttrDefId::MacroDefId(it.id),
        }
    }
}

impl From<AssocItem> for GenericDefId {
    fn from(item: AssocItem) -> Self {
        match item {
            AssocItem::Function(f) => f.id.into(),
            AssocItem::Const(c) => c.id.into(),
            AssocItem::TypeAlias(t) => t.id.into(),
        }
    }
}

impl From<(DefWithBodyId, PatId)> for Local {
    fn from((parent, pat_id): (DefWithBodyId, PatId)) -> Self {
        Local { parent, pat_id }
    }
}

impl From<MacroDef> for ItemInNs {
    fn from(macro_def: MacroDef) -> Self {
        ItemInNs::Macros(macro_def.into())
    }
}

impl From<ModuleDef> for ItemInNs {
    fn from(module_def: ModuleDef) -> Self {
        match module_def {
            ModuleDef::Static(_) | ModuleDef::Const(_) | ModuleDef::Function(_) => {
                ItemInNs::Values(module_def.into())
            }
            _ => ItemInNs::Types(module_def.into()),
        }
    }
}
