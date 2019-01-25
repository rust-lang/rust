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
    HirDatabase, HirFileId,
    type_ref::TypeRef,
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
    pub(crate) fn variant_data(&self, db: &impl HirDatabase) -> Arc<VariantData> {
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

    pub(crate) fn struct_data_query(db: &impl HirDatabase, struct_: Struct) -> Arc<StructData> {
        let (_, struct_def) = struct_.source(db);
        Arc::new(StructData::new(&*struct_def))
    }
}

fn variants(enum_def: &ast::EnumDef) -> impl Iterator<Item = &ast::EnumVariant> {
    enum_def
        .variant_list()
        .into_iter()
        .flat_map(|it| it.variants())
}

impl EnumVariant {
    pub(crate) fn source_impl(
        &self,
        db: &impl HirDatabase,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EnumVariantId(RawId);
impl_arena_id!(EnumVariantId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub(crate) name: Option<Name>,
    pub(crate) variants: Arena<EnumVariantId, EnumVariantData>,
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &impl HirDatabase, e: Enum) -> Arc<EnumData> {
        let (_file_id, enum_def) = e.source(db);
        let mut res = EnumData {
            name: enum_def.name().map(|n| n.as_name()),
            variants: Arena::default(),
        };
        for var in variants(&*enum_def) {
            let data = EnumVariantData {
                name: var.name().map(|it| it.as_name()),
                variant_data: Arc::new(VariantData::new(var.flavor())),
            };
            res.variants.alloc(data);
        }

        Arc::new(res)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariantData {
    pub(crate) name: Option<Name>,
    pub(crate) variant_data: Arc<VariantData>,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub(crate) name: Name,
    pub(crate) type_ref: TypeRef,
}

/// Fields of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantData {
    Struct(Vec<StructField>),
    Tuple(Vec<StructField>),
    Unit,
}

impl VariantData {
    pub fn fields(&self) -> &[StructField] {
        match self {
            VariantData::Struct(fields) | VariantData::Tuple(fields) => fields,
            _ => &[],
        }
    }

    pub fn is_struct(&self) -> bool {
        match self {
            VariantData::Struct(..) => true,
            _ => false,
        }
    }

    pub fn is_tuple(&self) -> bool {
        match self {
            VariantData::Tuple(..) => true,
            _ => false,
        }
    }

    pub fn is_unit(&self) -> bool {
        match self {
            VariantData::Unit => true,
            _ => false,
        }
    }
}

impl VariantData {
    fn new(flavor: StructFlavor) -> Self {
        match flavor {
            StructFlavor::Tuple(fl) => {
                let fields = fl
                    .fields()
                    .enumerate()
                    .map(|(i, fd)| StructField {
                        name: Name::tuple_field_name(i),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantData::Tuple(fields)
            }
            StructFlavor::Named(fl) => {
                let fields = fl
                    .fields()
                    .map(|fd| StructField {
                        name: fd.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantData::Struct(fields)
            }
            StructFlavor::Unit => VariantData::Unit,
        }
    }

    pub(crate) fn get_field_type_ref(&self, field_name: &Name) -> Option<&TypeRef> {
        self.fields()
            .iter()
            .find(|f| f.name == *field_name)
            .map(|f| &f.type_ref)
    }
}
