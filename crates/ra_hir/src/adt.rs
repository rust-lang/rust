//! This module contains the implementation details of the HIR for ADTs, i.e.
//! structs and enums (and unions).

use std::sync::Arc;

use ra_arena::{impl_arena_id, Arena, RawId};
use ra_syntax::ast::{self, NameOwner, StructKind, TypeAscriptionOwner};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    type_ref::TypeRef,
    AsName, Enum, EnumVariant, FieldSource, HasSource, Name, Source, Struct, StructField,
};

impl Struct {
    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.struct_data(self).variant_data.clone()
    }
}

/// Note that we use `StructData` for unions as well!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub(crate) name: Option<Name>,
    pub(crate) variant_data: Arc<VariantData>,
}

impl StructData {
    fn new(struct_def: &ast::StructDef) -> StructData {
        let name = struct_def.name().map(|n| n.as_name());
        let variant_data = VariantData::new(struct_def.kind());
        let variant_data = Arc::new(variant_data);
        StructData { name, variant_data }
    }

    pub(crate) fn struct_data_query(
        db: &(impl DefDatabase + AstDatabase),
        struct_: Struct,
    ) -> Arc<StructData> {
        let src = struct_.source(db);
        Arc::new(StructData::new(&src.ast))
    }
}

fn variants(enum_def: &ast::EnumDef) -> impl Iterator<Item = ast::EnumVariant> {
    enum_def.variant_list().into_iter().flat_map(|it| it.variants())
}

impl EnumVariant {
    pub(crate) fn source_impl(
        self,
        db: &(impl DefDatabase + AstDatabase),
    ) -> Source<ast::EnumVariant> {
        let src = self.parent.source(db);
        let ast = variants(&src.ast)
            .zip(db.enum_data(self.parent).variants.iter())
            .find(|(_syntax, (id, _))| *id == self.id)
            .unwrap()
            .0;
        Source { file_id: src.file_id, ast }
    }
    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        db.enum_data(self.parent).variants[self.id].variant_data.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub(crate) name: Option<Name>,
    pub(crate) variants: Arena<EnumVariantId, EnumVariantData>,
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &(impl DefDatabase + AstDatabase), e: Enum) -> Arc<EnumData> {
        let src = e.source(db);
        let name = src.ast.name().map(|n| n.as_name());
        let variants = variants(&src.ast)
            .map(|var| EnumVariantData {
                name: var.name().map(|it| it.as_name()),
                variant_data: Arc::new(VariantData::new(var.kind())),
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
    fn new(flavor: StructKind) -> Self {
        let inner = match flavor {
            ast::StructKind::Tuple(fl) => {
                let fields = fl
                    .fields()
                    .enumerate()
                    .map(|(i, fd)| StructFieldData {
                        name: Name::new_tuple_field(i),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantDataInner::Tuple(fields)
            }
            ast::StructKind::Named(fl) => {
                let fields = fl
                    .fields()
                    .map(|fd| StructFieldData {
                        name: fd.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        type_ref: TypeRef::from_ast_opt(fd.ascribed_type()),
                    })
                    .collect();
                VariantDataInner::Struct(fields)
            }
            ast::StructKind::Unit => VariantDataInner::Unit,
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
    pub fn fields(self, db: &impl HirDatabase) -> Vec<StructField> {
        match self {
            VariantDef::Struct(it) => it.fields(db),
            VariantDef::EnumVariant(it) => it.fields(db),
        }
    }

    pub(crate) fn field(self, db: &impl HirDatabase, name: &Name) -> Option<StructField> {
        match self {
            VariantDef::Struct(it) => it.field(db, name),
            VariantDef::EnumVariant(it) => it.field(db, name),
        }
    }
    pub(crate) fn variant_data(self, db: &impl DefDatabase) -> Arc<VariantData> {
        match self {
            VariantDef::Struct(it) => it.variant_data(db),
            VariantDef::EnumVariant(it) => it.variant_data(db),
        }
    }
}

impl StructField {
    pub(crate) fn source_impl(&self, db: &(impl DefDatabase + AstDatabase)) -> Source<FieldSource> {
        let var_data = self.parent.variant_data(db);
        let fields = var_data.fields().unwrap();
        let ss;
        let es;
        let (file_id, struct_kind) = match self.parent {
            VariantDef::Struct(s) => {
                ss = s.source(db);
                (ss.file_id, ss.ast.kind())
            }
            VariantDef::EnumVariant(e) => {
                es = e.source(db);
                (es.file_id, es.ast.kind())
            }
        };

        let field_sources = match struct_kind {
            ast::StructKind::Tuple(fl) => fl.fields().map(|it| FieldSource::Pos(it)).collect(),
            ast::StructKind::Named(fl) => fl.fields().map(|it| FieldSource::Named(it)).collect(),
            ast::StructKind::Unit => Vec::new(),
        };
        let ast = field_sources
            .into_iter()
            .zip(fields.iter())
            .find(|(_syntax, (id, _))| *id == self.id)
            .unwrap()
            .0;
        Source { file_id, ast }
    }
}
