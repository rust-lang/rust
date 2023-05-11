//! Utility module for converting between hir_def ids and code_model wrappers.
//!
//! It's unclear if we need this long-term, but it's definitely useful while we
//! are splitting the hir.

use hir_def::{
    expr::{BindingId, LabelId},
    AdtId, AssocItemId, DefWithBodyId, EnumVariantId, FieldId, GenericDefId, GenericParamId,
    ModuleDefId, VariantId,
};

use crate::{
    Adt, AssocItem, BuiltinType, DefWithBody, Field, GenericDef, GenericParam, ItemInNs, Label,
    Local, ModuleDef, Variant, VariantDef,
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
    (base_db::CrateId, crate::Crate),
    (hir_def::ModuleId, crate::Module),
    (hir_def::StructId, crate::Struct),
    (hir_def::UnionId, crate::Union),
    (hir_def::EnumId, crate::Enum),
    (hir_def::TypeAliasId, crate::TypeAlias),
    (hir_def::TraitId, crate::Trait),
    (hir_def::TraitAliasId, crate::TraitAlias),
    (hir_def::StaticId, crate::Static),
    (hir_def::ConstId, crate::Const),
    (hir_def::FunctionId, crate::Function),
    (hir_def::ImplId, crate::Impl),
    (hir_def::TypeOrConstParamId, crate::TypeOrConstParam),
    (hir_def::TypeParamId, crate::TypeParam),
    (hir_def::ConstParamId, crate::ConstParam),
    (hir_def::LifetimeParamId, crate::LifetimeParam),
    (hir_def::MacroId, crate::Macro)
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

impl From<GenericParamId> for GenericParam {
    fn from(id: GenericParamId) -> Self {
        match id {
            GenericParamId::TypeParamId(it) => GenericParam::TypeParam(it.into()),
            GenericParamId::ConstParamId(it) => GenericParam::ConstParam(it.into()),
            GenericParamId::LifetimeParamId(it) => GenericParam::LifetimeParam(it.into()),
        }
    }
}

impl From<GenericParam> for GenericParamId {
    fn from(id: GenericParam) -> Self {
        match id {
            GenericParam::LifetimeParam(it) => GenericParamId::LifetimeParamId(it.id),
            GenericParam::ConstParam(it) => GenericParamId::ConstParamId(it.id),
            GenericParam::TypeParam(it) => GenericParamId::TypeParamId(it.id),
        }
    }
}

impl From<EnumVariantId> for Variant {
    fn from(id: EnumVariantId) -> Self {
        Variant { parent: id.parent.into(), id: id.local_id }
    }
}

impl From<Variant> for EnumVariantId {
    fn from(def: Variant) -> Self {
        EnumVariantId { parent: def.parent.id, local_id: def.id }
    }
}

impl From<ModuleDefId> for ModuleDef {
    fn from(id: ModuleDefId) -> Self {
        match id {
            ModuleDefId::ModuleId(it) => ModuleDef::Module(it.into()),
            ModuleDefId::FunctionId(it) => ModuleDef::Function(it.into()),
            ModuleDefId::AdtId(it) => ModuleDef::Adt(it.into()),
            ModuleDefId::EnumVariantId(it) => ModuleDef::Variant(it.into()),
            ModuleDefId::ConstId(it) => ModuleDef::Const(it.into()),
            ModuleDefId::StaticId(it) => ModuleDef::Static(it.into()),
            ModuleDefId::TraitId(it) => ModuleDef::Trait(it.into()),
            ModuleDefId::TraitAliasId(it) => ModuleDef::TraitAlias(it.into()),
            ModuleDefId::TypeAliasId(it) => ModuleDef::TypeAlias(it.into()),
            ModuleDefId::BuiltinType(it) => ModuleDef::BuiltinType(it.into()),
            ModuleDefId::MacroId(it) => ModuleDef::Macro(it.into()),
        }
    }
}

impl From<ModuleDef> for ModuleDefId {
    fn from(id: ModuleDef) -> Self {
        match id {
            ModuleDef::Module(it) => ModuleDefId::ModuleId(it.into()),
            ModuleDef::Function(it) => ModuleDefId::FunctionId(it.into()),
            ModuleDef::Adt(it) => ModuleDefId::AdtId(it.into()),
            ModuleDef::Variant(it) => ModuleDefId::EnumVariantId(it.into()),
            ModuleDef::Const(it) => ModuleDefId::ConstId(it.into()),
            ModuleDef::Static(it) => ModuleDefId::StaticId(it.into()),
            ModuleDef::Trait(it) => ModuleDefId::TraitId(it.into()),
            ModuleDef::TraitAlias(it) => ModuleDefId::TraitAliasId(it.into()),
            ModuleDef::TypeAlias(it) => ModuleDefId::TypeAliasId(it.into()),
            ModuleDef::BuiltinType(it) => ModuleDefId::BuiltinType(it.into()),
            ModuleDef::Macro(it) => ModuleDefId::MacroId(it.into()),
        }
    }
}

impl From<DefWithBody> for DefWithBodyId {
    fn from(def: DefWithBody) -> Self {
        match def {
            DefWithBody::Function(it) => DefWithBodyId::FunctionId(it.id),
            DefWithBody::Static(it) => DefWithBodyId::StaticId(it.id),
            DefWithBody::Const(it) => DefWithBodyId::ConstId(it.id),
            DefWithBody::Variant(it) => DefWithBodyId::VariantId(it.into()),
        }
    }
}

impl From<DefWithBodyId> for DefWithBody {
    fn from(def: DefWithBodyId) -> Self {
        match def {
            DefWithBodyId::FunctionId(it) => DefWithBody::Function(it.into()),
            DefWithBodyId::StaticId(it) => DefWithBody::Static(it.into()),
            DefWithBodyId::ConstId(it) => DefWithBody::Const(it.into()),
            DefWithBodyId::VariantId(it) => DefWithBody::Variant(it.into()),
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
            GenericDef::TraitAlias(it) => GenericDefId::TraitAliasId(it.id),
            GenericDef::TypeAlias(it) => GenericDefId::TypeAliasId(it.id),
            GenericDef::Impl(it) => GenericDefId::ImplId(it.id),
            GenericDef::Variant(it) => GenericDefId::EnumVariantId(it.into()),
            GenericDef::Const(it) => GenericDefId::ConstId(it.id),
        }
    }
}

impl From<GenericDefId> for GenericDef {
    fn from(def: GenericDefId) -> Self {
        match def {
            GenericDefId::FunctionId(it) => GenericDef::Function(it.into()),
            GenericDefId::AdtId(it) => GenericDef::Adt(it.into()),
            GenericDefId::TraitId(it) => GenericDef::Trait(it.into()),
            GenericDefId::TraitAliasId(it) => GenericDef::TraitAlias(it.into()),
            GenericDefId::TypeAliasId(it) => GenericDef::TypeAlias(it.into()),
            GenericDefId::ImplId(it) => GenericDef::Impl(it.into()),
            GenericDefId::EnumVariantId(it) => GenericDef::Variant(it.into()),
            GenericDefId::ConstId(it) => GenericDef::Const(it.into()),
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
            VariantId::EnumVariantId(it) => VariantDef::Variant(it.into()),
            VariantId::UnionId(it) => VariantDef::Union(it.into()),
        }
    }
}

impl From<VariantDef> for VariantId {
    fn from(def: VariantDef) -> Self {
        match def {
            VariantDef::Struct(it) => VariantId::StructId(it.id),
            VariantDef::Variant(it) => VariantId::EnumVariantId(it.into()),
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

impl From<AssocItem> for GenericDefId {
    fn from(item: AssocItem) -> Self {
        match item {
            AssocItem::Function(f) => f.id.into(),
            AssocItem::Const(c) => c.id.into(),
            AssocItem::TypeAlias(t) => t.id.into(),
        }
    }
}

impl From<(DefWithBodyId, BindingId)> for Local {
    fn from((parent, binding_id): (DefWithBodyId, BindingId)) -> Self {
        Local { parent, binding_id }
    }
}

impl From<(DefWithBodyId, LabelId)> for Label {
    fn from((parent, label_id): (DefWithBodyId, LabelId)) -> Self {
        Label { parent, label_id }
    }
}

impl From<hir_def::item_scope::ItemInNs> for ItemInNs {
    fn from(it: hir_def::item_scope::ItemInNs) -> Self {
        match it {
            hir_def::item_scope::ItemInNs::Types(it) => ItemInNs::Types(it.into()),
            hir_def::item_scope::ItemInNs::Values(it) => ItemInNs::Values(it.into()),
            hir_def::item_scope::ItemInNs::Macros(it) => ItemInNs::Macros(it.into()),
        }
    }
}

impl From<ItemInNs> for hir_def::item_scope::ItemInNs {
    fn from(it: ItemInNs) -> Self {
        match it {
            ItemInNs::Types(it) => Self::Types(it.into()),
            ItemInNs::Values(it) => Self::Values(it.into()),
            ItemInNs::Macros(it) => Self::Macros(it.into()),
        }
    }
}

impl From<hir_def::builtin_type::BuiltinType> for BuiltinType {
    fn from(inner: hir_def::builtin_type::BuiltinType) -> Self {
        Self { inner }
    }
}

impl From<BuiltinType> for hir_def::builtin_type::BuiltinType {
    fn from(it: BuiltinType) -> Self {
        it.inner
    }
}
