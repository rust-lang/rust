use std::sync::Arc;

use ra_syntax::{SmolStr, ast::{self, NameOwner, StructFlavor}};

use crate::{
    DefId, Cancelable,
    db::{HirDatabase},
    type_ref::TypeRef,
};

pub struct Struct {
    def_id: DefId,
}

impl Struct {
    pub(crate) fn new(def_id: DefId) -> Self {
        Struct { def_id }
    }

    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn variant_data(&self, db: &impl HirDatabase) -> Cancelable<Arc<VariantData>> {
        Ok(db.struct_data(self.def_id)?.variant_data.clone())
    }

    pub fn struct_data(&self, db: &impl HirDatabase) -> Cancelable<Arc<StructData>> {
        Ok(db.struct_data(self.def_id)?)
    }

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<Option<SmolStr>> {
        Ok(db.struct_data(self.def_id)?.name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    name: Option<SmolStr>,
    variant_data: Arc<VariantData>,
}

impl StructData {
    pub(crate) fn new(struct_def: ast::StructDef) -> StructData {
        let name = struct_def.name().map(|n| n.text());
        let variant_data = VariantData::new(struct_def.flavor());
        let variant_data = Arc::new(variant_data);
        StructData { name, variant_data }
    }

    pub fn name(&self) -> Option<&SmolStr> {
        self.name.as_ref()
    }

    pub fn variant_data(&self) -> &Arc<VariantData> {
        &self.variant_data
    }
}

pub struct Enum {
    def_id: DefId,
}

impl Enum {
    pub(crate) fn new(def_id: DefId) -> Self {
        Enum { def_id }
    }

    pub fn def_id(&self) -> DefId {
        self.def_id
    }

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<Option<SmolStr>> {
        Ok(db.enum_data(self.def_id)?.name.clone())
    }

    pub fn variants(&self, db: &impl HirDatabase) -> Cancelable<Vec<(SmolStr, Arc<VariantData>)>> {
        Ok(db.enum_data(self.def_id)?.variants.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    name: Option<SmolStr>,
    variants: Vec<(SmolStr, Arc<VariantData>)>,
}

impl EnumData {
    pub(crate) fn new(enum_def: ast::EnumDef) -> Self {
        let name = enum_def.name().map(|n| n.text());
        let variants = if let Some(evl) = enum_def.variant_list() {
            evl.variants()
                .map(|v| {
                    (
                        v.name()
                            .map(|n| n.text())
                            .unwrap_or_else(|| SmolStr::new("[error]")),
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

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    name: SmolStr,
    type_ref: TypeRef,
}

impl StructField {
    pub fn name(&self) -> SmolStr {
        self.name.clone()
    }
    pub fn type_ref(&self) -> &TypeRef {
        &self.type_ref
    }
}

/// Fields of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantData {
    Struct(Vec<StructField>),
    Tuple(Vec<StructField>),
    Unit,
}

impl VariantData {
    pub fn new(flavor: StructFlavor) -> Self {
        match flavor {
            StructFlavor::Tuple(fl) => {
                let fields = fl
                    .fields()
                    .enumerate()
                    .map(|(i, fd)| StructField {
                        name: SmolStr::new(i.to_string()),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantData::Tuple(fields)
            }
            StructFlavor::Named(fl) => {
                let fields = fl
                    .fields()
                    .map(|fd| StructField {
                        name: fd
                            .name()
                            .map(|n| n.text())
                            .unwrap_or_else(|| SmolStr::new("[error]")),
                        type_ref: TypeRef::from_ast_opt(fd.type_ref()),
                    })
                    .collect();
                VariantData::Struct(fields)
            }
            StructFlavor::Unit => VariantData::Unit,
        }
    }

    pub(crate) fn get_field_type_ref(&self, field_name: &str) -> Option<&TypeRef> {
        self.fields()
            .iter()
            .find(|f| f.name == field_name)
            .map(|f| &f.type_ref)
    }

    pub fn fields(&self) -> &[StructField] {
        match *self {
            VariantData::Struct(ref fields) | VariantData::Tuple(ref fields) => fields,
            _ => &[],
        }
    }
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_tuple(&self) -> bool {
        if let VariantData::Tuple(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_unit(&self) -> bool {
        if let VariantData::Unit = *self {
            true
        } else {
            false
        }
    }
}
