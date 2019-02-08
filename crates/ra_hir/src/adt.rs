//! This module contains the implementation details of the HIR for ADTs, i.e.
//! structs and enums (and unions).

use std::sync::Arc;

use ra_arena::{RawId, Arena, impl_arena_id};
use ra_syntax::{
    TreeArc,
    ast::{self, NameOwner, StructFlavor}
};

use crate::{
    Name, AsName, Struct, Enum, EnumVariant, Crate,
    HirDatabase, HirFileId, StructField, FieldSource,
    type_ref::TypeRef, PersistentHirDatabase,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdtDef {
    Struct(Struct),
    Enum(Enum),
}
impl_froms!(AdtDef: Struct, Enum);

impl AdtDef {
    pub(crate) fn krate(self, db: &impl HirDatabase) -> Option<Crate> {
        match self {
            AdtDef::Struct(s) => s.module(db),
            AdtDef::Enum(e) => e.module(db),
        }
        .krate(db)
    }
}

impl Struct {
    pub(crate) fn variant_data(&self, db: &impl PersistentHirDatabase) -> Arc<VariantData> {
        db.struct_data((*self).into()).variant_data.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub(crate) name: Option<Name>,
    pub(crate) variant_data: Arc<VariantData>,
}

impl StructData {
    fn new(struct_def: &ast::StructDef) -> StructData {
        let name = struct_def.name().map(|n| n.as_name());
        let variant_data = VariantData::new(struct_def.flavor());
        let variant_data = Arc::new(variant_data);
        StructData { name, variant_data }
    }

    pub(crate) fn struct_data_query(
        db: &impl PersistentHirDatabase,
        struct_: Struct,
    ) -> Arc<StructData> {
        let (_, struct_def) = struct_.source(db);
        Arc::new(StructData::new(&*struct_def))
    }
}

fn variants(enum_def: &ast::EnumDef) -> impl Iterator<Item = &ast::EnumVariant> {
    enum_def.variant_list().into_iter().flat_map(|it| it.variants())
}

impl EnumVariant {
    pub(crate) fn source_impl(
        &self,
        db: &impl PersistentHirDatabase,
    ) -> (HirFileId, TreeArc<ast::EnumVariant>) {
        let (file_id, enum_def) = self.parent.source(db);
        let var = variants(&*enum_def)
            .zip(db.enum_data(self.parent).variants.iter())
            .find(|(_syntax, (id, _))| *id == self.id)
            .unwrap()
            .0
            .to_owned();
        (file_id, var)
    }
    pub(crate) fn variant_data(&self, db: &impl PersistentHirDatabase) -> Arc<VariantData> {
        db.enum_data(self.parent).variants[self.id].variant_data.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub(crate) name: Option<Name>,
    pub(crate) variants: Arena<EnumVariantId, EnumVariantData>,
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &impl PersistentHirDatabase, e: Enum) -> Arc<EnumData> {
        let (_file_id, enum_def) = e.source(db);
        let name = enum_def.name().map(|n| n.as_name());
        let variants = variants(&*enum_def)
            .map(|var| EnumVariantData {
                name: var.name().map(|it| it.as_name()),
                variant_data: Arc::new(VariantData::new(var.flavor())),
            })
            .collect();
        Arc::new(EnumData { name, variants })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EnumVariantId(RawId);
impl_arena_id!(EnumVariantId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EnumVariantData {
    pub(crate) name: Option<Name>,
    variant_data: Arc<VariantData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StructFieldId(RawId);
impl_arena_id!(StructFieldId);

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructFieldData {
    pub(crate) name: Name,
    pub(crate) type_ref: TypeRef,
}

/// Fields of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariantData(VariantDataInner);

#[derive(Debug, Clone, PartialEq, Eq)]
enum VariantDataInner {
    Struct(Arena<StructFieldId, StructFieldData>),
    Tuple(Arena<StructFieldId, StructFieldData>),
    Unit,
}

impl VariantData {
    pub(crate) fn fields(&self) -> Option<&Arena<StructFieldId, StructFieldData>> {
        match &self.0 {
            VariantDataInner::Struct(fields) | VariantDataInner::Tuple(fields) => Some(fields),
            _ => None,
        }
    }
}

impl VariantData {
    fn new(flavor: StructFlavor) -> Self {
        let inner = match flavor {
            ast::StructFlavor::Tuple(fl) => {
                let fields = fl
                    .fields()
                    .enumerate()
                    .map(|(i, fd)| StructFieldData {
                        name: Name::tuple_field_name(i),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantDataInner::Tuple(fields)
            }
            ast::StructFlavor::Named(fl) => {
                let fields = fl
                    .fields()
                    .map(|fd| StructFieldData {
                        name: fd.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantDataInner::Struct(fields)
            }
            ast::StructFlavor::Unit => VariantDataInner::Unit,
        };
        VariantData(inner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VariantDef {
    Struct(Struct),
    EnumVariant(EnumVariant),
}
impl_froms!(VariantDef: Struct, EnumVariant);

impl VariantDef {
    pub(crate) fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        match self {
            VariantDef::Struct(it) => it.field(db, name),
            VariantDef::EnumVariant(it) => it.field(db, name),
        }
    }
    pub(crate) fn variant_data(self, db: &impl PersistentHirDatabase) -> Arc<VariantData> {
        match self {
            VariantDef::Struct(it) => it.variant_data(db),
            VariantDef::EnumVariant(it) => it.variant_data(db),
        }
    }
}

impl StructField {
    pub(crate) fn source_impl(&self, db: &impl PersistentHirDatabase) -> (HirFileId, FieldSource) {
        let var_data = self.parent.variant_data(db);
        let fields = var_data.fields().unwrap();
        let ss;
        let es;
        let (file_id, struct_flavor) = match self.parent {
            VariantDef::Struct(s) => {
                let (file_id, source) = s.source(db);
                ss = source;
                (file_id, ss.flavor())
            }
            VariantDef::EnumVariant(e) => {
                let (file_id, source) = e.source(db);
                es = source;
                (file_id, es.flavor())
            }
        };

        let field_sources = match struct_flavor {
            ast::StructFlavor::Tuple(fl) => {
                fl.fields().map(|it| FieldSource::Pos(it.to_owned())).collect()
            }
            ast::StructFlavor::Named(fl) => {
                fl.fields().map(|it| FieldSource::Named(it.to_owned())).collect()
            }
            ast::StructFlavor::Unit => Vec::new(),
        };
        let field = field_sources
            .into_iter()
            .zip(fields.iter())
            .find(|(_syntax, (id, _))| *id == self.id)
            .unwrap()
            .0;
        (file_id, field)
    }
}
