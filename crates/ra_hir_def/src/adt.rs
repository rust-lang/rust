//! Defines hir-level representation of structs, enums and unions

use std::sync::Arc;

use hir_expand::{
    either::Either,
    name::{AsName, Name},
    Source,
};
use ra_arena::{map::ArenaMap, Arena};
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner};

use crate::{
    db::DefDatabase, trace::Trace, type_ref::TypeRef, AstItemDef, EnumId, HasChildSource,
    LocalEnumVariantId, LocalStructFieldId, StructOrUnionId, VariantId,
};

/// Note that we use `StructData` for unions as well!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub name: Option<Name>,
    pub variant_data: Arc<VariantData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub name: Option<Name>,
    pub variants: Arena<LocalEnumVariantId, EnumVariantData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariantData {
    pub name: Option<Name>,
    pub variant_data: Arc<VariantData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantData {
    Record(Arena<LocalStructFieldId, StructFieldData>),
    Tuple(Arena<LocalStructFieldId, StructFieldData>),
    Unit,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructFieldData {
    pub name: Name,
    pub type_ref: TypeRef,
}

impl StructData {
    pub(crate) fn struct_data_query(db: &impl DefDatabase, id: StructOrUnionId) -> Arc<StructData> {
        let src = id.source(db);
        let name = src.value.name().map(|n| n.as_name());
        let variant_data = VariantData::new(src.value.kind());
        let variant_data = Arc::new(variant_data);
        Arc::new(StructData { name, variant_data })
    }
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &impl DefDatabase, e: EnumId) -> Arc<EnumData> {
        let src = e.source(db);
        let name = src.value.name().map(|n| n.as_name());
        let mut trace = Trace::new_for_arena();
        lower_enum(&mut trace, &src.value);
        Arc::new(EnumData { name, variants: trace.into_arena() })
    }

    pub(crate) fn variant(&self, name: &Name) -> Option<LocalEnumVariantId> {
        let (id, _) = self.variants.iter().find(|(_id, data)| data.name.as_ref() == Some(name))?;
        Some(id)
    }
}

impl HasChildSource for EnumId {
    type ChildId = LocalEnumVariantId;
    type Value = ast::EnumVariant;
    fn child_source(&self, db: &impl DefDatabase) -> Source<ArenaMap<Self::ChildId, Self::Value>> {
        let src = self.source(db);
        let mut trace = Trace::new_for_map();
        lower_enum(&mut trace, &src.value);
        src.with_value(trace.into_map())
    }
}

fn lower_enum(
    trace: &mut Trace<LocalEnumVariantId, EnumVariantData, ast::EnumVariant>,
    ast: &ast::EnumDef,
) {
    for var in ast.variant_list().into_iter().flat_map(|it| it.variants()) {
        trace.alloc(
            || var.clone(),
            || EnumVariantData {
                name: var.name().map(|it| it.as_name()),
                variant_data: Arc::new(VariantData::new(var.kind())),
            },
        );
    }
}

impl VariantData {
    fn new(flavor: ast::StructKind) -> Self {
        let mut trace = Trace::new_for_arena();
        match lower_struct(&mut trace, &flavor) {
            StructKind::Tuple => VariantData::Tuple(trace.into_arena()),
            StructKind::Record => VariantData::Record(trace.into_arena()),
            StructKind::Unit => VariantData::Unit,
        }
    }

    pub fn fields(&self) -> Option<&Arena<LocalStructFieldId, StructFieldData>> {
        match &self {
            VariantData::Record(fields) | VariantData::Tuple(fields) => Some(fields),
            _ => None,
        }
    }
}

impl HasChildSource for VariantId {
    type ChildId = LocalStructFieldId;
    type Value = Either<ast::TupleFieldDef, ast::RecordFieldDef>;

    fn child_source(&self, db: &impl DefDatabase) -> Source<ArenaMap<Self::ChildId, Self::Value>> {
        let src = match self {
            VariantId::EnumVariantId(it) => {
                // I don't really like the fact that we call into parent source
                // here, this might add to more queries then necessary.
                let src = it.parent.child_source(db);
                src.map(|map| map[it.local_id].kind())
            }
            VariantId::StructId(it) => it.0.source(db).map(|it| it.kind()),
        };
        let mut trace = Trace::new_for_map();
        lower_struct(&mut trace, &src.value);
        src.with_value(trace.into_map())
    }
}

enum StructKind {
    Tuple,
    Record,
    Unit,
}

fn lower_struct(
    trace: &mut Trace<
        LocalStructFieldId,
        StructFieldData,
        Either<ast::TupleFieldDef, ast::RecordFieldDef>,
    >,
    ast: &ast::StructKind,
) -> StructKind {
    match ast {
        ast::StructKind::Tuple(fl) => {
            for (i, fd) in fl.fields().enumerate() {
                trace.alloc(
                    || Either::A(fd.clone()),
                    || StructFieldData {
                        name: Name::new_tuple_field(i),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    },
                );
            }
            StructKind::Tuple
        }
        ast::StructKind::Record(fl) => {
            for fd in fl.fields() {
                trace.alloc(
                    || Either::B(fd.clone()),
                    || StructFieldData {
                        name: fd.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        type_ref: TypeRef::from_ast_opt(fd.ascribed_type()),
                    },
                );
            }
            StructKind::Record
        }
        ast::StructKind::Unit => StructKind::Unit,
    }
}
