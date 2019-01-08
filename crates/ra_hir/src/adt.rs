use std::sync::Arc;

use ra_syntax::ast::{self, NameOwner, StructFlavor};

use crate::{
    DefId, Name, AsName, Struct, Enum, VariantData, StructField,
    type_ref::TypeRef,
};

impl Struct {
    pub(crate) fn new(def_id: DefId) -> Self {
        Struct { def_id }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub(crate) name: Option<Name>,
    pub(crate) variant_data: Arc<VariantData>,
}

impl StructData {
    pub(crate) fn new(struct_def: &ast::StructDef) -> StructData {
        let name = struct_def.name().map(|n| n.as_name());
        let variant_data = VariantData::new(struct_def.flavor());
        let variant_data = Arc::new(variant_data);
        StructData { name, variant_data }
    }

    pub fn name(&self) -> Option<&Name> {
        self.name.as_ref()
    }

    pub fn variant_data(&self) -> &Arc<VariantData> {
        &self.variant_data
    }
}

impl Enum {
    pub(crate) fn new(def_id: DefId) -> Self {
        Enum { def_id }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub(crate) name: Option<Name>,
    pub(crate) variants: Vec<(Name, Arc<VariantData>)>,
}

impl EnumData {
    pub(crate) fn new(enum_def: &ast::EnumDef) -> Self {
        let name = enum_def.name().map(|n| n.as_name());
        let variants = if let Some(evl) = enum_def.variant_list() {
            evl.variants()
                .map(|v| {
                    (
                        v.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        Arc::new(VariantData::new(v.flavor())),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        EnumData { name, variants }
    }
}

impl VariantData {
    pub fn new(flavor: StructFlavor) -> Self {
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
            .find(|f| f.name() == field_name)
            .map(|f| f.type_ref())
    }

    pub fn fields(&self) -> &[StructField] {
        match *self {
            VariantData::Struct(ref fields) | VariantData::Tuple(ref fields) => fields,
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
