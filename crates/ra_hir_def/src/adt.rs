//! Defines hir-level representation of structs, enums and unions

use std::sync::Arc;

use hir_expand::name::{AsName, Name};
use ra_arena::Arena;
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner};

use crate::{
    db::DefDatabase2, type_ref::TypeRef, AstItemDef, EnumId, LocalEnumVariantId,
    LocalStructFieldId, StructOrUnionId,
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
    pub(crate) fn struct_data_query(
        db: &impl DefDatabase2,
        id: StructOrUnionId,
    ) -> Arc<StructData> {
        let src = id.source(db);
        let name = src.value.name().map(|n| n.as_name());
        let variant_data = VariantData::new(src.value.kind());
        let variant_data = Arc::new(variant_data);
        Arc::new(StructData { name, variant_data })
    }
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &impl DefDatabase2, e: EnumId) -> Arc<EnumData> {
        let src = e.source(db);
        let name = src.value.name().map(|n| n.as_name());
        let variants = src
            .value
            .variant_list()
            .into_iter()
            .flat_map(|it| it.variants())
            .map(|var| EnumVariantData {
                name: var.name().map(|it| it.as_name()),
                variant_data: Arc::new(VariantData::new(var.kind())),
            })
            .collect();
        Arc::new(EnumData { name, variants })
    }

    pub(crate) fn variant(&self, name: &Name) -> Option<LocalEnumVariantId> {
        let (id, _) = self.variants.iter().find(|(_id, data)| data.name.as_ref() == Some(name))?;
        Some(id)
    }
}

impl VariantData {
    fn new(flavor: ast::StructKind) -> Self {
        match flavor {
            ast::StructKind::Tuple(fl) => {
                let fields = fl
                    .fields()
                    .enumerate()
                    .map(|(i, fd)| StructFieldData {
                        name: Name::new_tuple_field(i),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantData::Tuple(fields)
            }
            ast::StructKind::Record(fl) => {
                let fields = fl
                    .fields()
                    .map(|fd| StructFieldData {
                        name: fd.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        type_ref: TypeRef::from_ast_opt(fd.ascribed_type()),
                    })
                    .collect();
                VariantData::Record(fields)
            }
            ast::StructKind::Unit => VariantData::Unit,
        }
    }

    pub fn fields(&self) -> Option<&Arena<LocalStructFieldId, StructFieldData>> {
        match self {
            VariantData::Record(fields) | VariantData::Tuple(fields) => Some(fields),
            _ => None,
        }
    }
}
