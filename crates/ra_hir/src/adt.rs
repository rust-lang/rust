//! This module contains the implementation details of the HIR for ADTs, i.e.
//! structs and enums (and unions).

use std::sync::Arc;

use hir_def::adt::VariantData;

use crate::{
    db::{DefDatabase, HirDatabase},
    EnumVariant, Module, Name, Struct, StructField,
};

impl Struct {
    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.struct_data(self.id).variant_data.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VariantDef {
    Struct(Struct),
    EnumVariant(EnumVariant),
}
impl_froms!(VariantDef: Struct, EnumVariant);

impl VariantDef {
    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        match self {
            VariantDef::Struct(it) => it.fields(db),
            VariantDef::EnumVariant(it) => it.fields(db),
        }
    }

    pub fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        match self {
            VariantDef::Struct(it) => it.field(db, name),
            VariantDef::EnumVariant(it) => it.field(db, name),
        }
    }

    pub fn module(self, db: &impl HirDatabase) -> Module {
        match self {
            VariantDef::Struct(it) => it.module(db),
            VariantDef::EnumVariant(it) => it.module(db),
        }
    }

    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        match self {
            VariantDef::Struct(it) => it.variant_data(db),
            VariantDef::EnumVariant(it) => it.variant_data(db),
        }
    }
}
