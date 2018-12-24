use ra_syntax::{SmolStr, ast::{self, NameOwner}};

use crate::{
    DefId, Cancelable,
    db::{HirDatabase},
    ty::{Ty},
};

pub struct Struct {
    def_id: DefId,
}

impl Struct {
    pub(crate) fn new(def_id: DefId) -> Self {
        Struct { def_id }
    }

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<SmolStr> {
        Ok(db.struct_data(self.def_id)?.name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    name: SmolStr,
    variant_data: VariantData,
}

impl StructData {
    pub(crate) fn new(struct_def: ast::StructDef) -> StructData {
        let name = struct_def
            .name()
            .map(|n| n.text())
            .unwrap_or(SmolStr::new("[error]"));
        let variant_data = VariantData::Unit; // TODO implement this
        StructData { name, variant_data }
    }
}

pub struct Enum {
    def_id: DefId,
}

impl Enum {
    pub(crate) fn new(def_id: DefId) -> Self {
        Enum { def_id }
    }

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<SmolStr> {
        Ok(db.enum_data(self.def_id)?.name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    name: SmolStr,
    variants: Vec<(SmolStr, VariantData)>,
}

impl EnumData {
    pub(crate) fn new(enum_def: ast::EnumDef) -> Self {
        let name = enum_def
            .name()
            .map(|n| n.text())
            .unwrap_or(SmolStr::new("[error]"));
        let variants = Vec::new(); // TODO implement this
        EnumData { name, variants }
    }
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    name: SmolStr,
    ty: Ty,
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
