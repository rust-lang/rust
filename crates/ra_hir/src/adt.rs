use std::sync::Arc;

use ra_syntax::{SmolStr, ast::{self, NameOwner, StructFlavor}};

use crate::{
    DefId, Cancelable,
    db::{HirDatabase},
    module::Module,
    ty::{Ty},
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

    pub fn struct_data(&self, db: &impl HirDatabase) -> Cancelable<Arc<StructData>> {
        Ok(db.struct_data(self.def_id)?)
    }

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<SmolStr> {
        Ok(db.struct_data(self.def_id)?.name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    name: SmolStr,
    variant_data: Arc<VariantData>,
}

impl StructData {
    pub(crate) fn new(
        db: &impl HirDatabase,
        module: &Module,
        struct_def: ast::StructDef,
    ) -> Cancelable<StructData> {
        let name = struct_def
            .name()
            .map(|n| n.text())
            .unwrap_or(SmolStr::new("[error]"));
        let variant_data = VariantData::new(db, module, struct_def.flavor())?;
        let variant_data = Arc::new(variant_data);
        Ok(StructData { name, variant_data })
    }

    pub fn name(&self) -> &SmolStr {
        &self.name
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

    pub fn name(&self, db: &impl HirDatabase) -> Cancelable<SmolStr> {
        Ok(db.enum_data(self.def_id)?.name.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    name: SmolStr,
    variants: Vec<(SmolStr, Arc<VariantData>)>,
}

impl EnumData {
    pub(crate) fn new(
        db: &impl HirDatabase,
        module: &Module,
        enum_def: ast::EnumDef,
    ) -> Cancelable<Self> {
        let name = enum_def
            .name()
            .map(|n| n.text())
            .unwrap_or(SmolStr::new("[error]"));
        let variants = if let Some(evl) = enum_def.variant_list() {
            evl.variants()
                .map(|v| {
                    Ok((
                        v.name()
                            .map(|n| n.text())
                            .unwrap_or_else(|| SmolStr::new("[error]")),
                        Arc::new(VariantData::new(db, module, v.flavor())?),
                    ))
                })
                .collect::<Cancelable<_>>()?
        } else {
            Vec::new()
        };
        Ok(EnumData { name, variants })
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
    pub fn new(db: &impl HirDatabase, module: &Module, flavor: StructFlavor) -> Cancelable<Self> {
        Ok(match flavor {
            StructFlavor::Tuple(fl) => {
                let fields = fl
                    .fields()
                    .enumerate()
                    .map(|(i, fd)| {
                        Ok(StructField {
                            name: SmolStr::new(i.to_string()),
                            ty: Ty::new_opt(db, &module, fd.type_ref())?,
                        })
                    })
                    .collect::<Cancelable<_>>()?;
                VariantData::Tuple(fields)
            }
            StructFlavor::Named(fl) => {
                let fields = fl
                    .fields()
                    .map(|fd| {
                        Ok(StructField {
                            name: fd
                                .name()
                                .map(|n| n.text())
                                .unwrap_or_else(|| SmolStr::new("[error]")),
                            ty: Ty::new_opt(db, &module, fd.type_ref())?,
                        })
                    })
                    .collect::<Cancelable<_>>()?;
                VariantData::Struct(fields)
            }
            StructFlavor::Unit => VariantData::Unit,
        })
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
