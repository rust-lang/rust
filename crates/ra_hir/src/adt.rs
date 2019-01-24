//! This module contains the implementation details of the HIR for ADTs, i.e.
//! structs and enums (and unions).

use std::sync::Arc;

use ra_syntax::{
    SyntaxNode,
    ast::{self, NameOwner, StructFlavor, AstNode}
};

use crate::{
    DefId, DefLoc, Name, AsName, Struct, Enum, EnumVariant, Module, HirFileId,
    HirDatabase, DefKind,
    SourceItemId,
    type_ref::TypeRef,
    ids::{StructLoc},
};

impl Struct {
    pub(crate) fn from_ast(
        db: &impl HirDatabase,
        module: Module,
        file_id: HirFileId,
        ast: &ast::StructDef,
    ) -> Struct {
        let loc: StructLoc = StructLoc::from_ast(db, module, file_id, ast);
        let id = loc.id(db);
        Struct { id }
    }

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

fn get_def_id(
    db: &impl HirDatabase,
    same_file_loc: &DefLoc,
    node: &SyntaxNode,
    expected_kind: DefKind,
) -> DefId {
    let file_id = same_file_loc.source_item_id.file_id;
    let file_items = db.file_items(file_id);

    let item_id = file_items.id_of(file_id, node);
    let source_item_id = SourceItemId {
        item_id: Some(item_id),
        ..same_file_loc.source_item_id
    };
    let loc = DefLoc {
        module: same_file_loc.module,
        kind: expected_kind,
        source_item_id,
    };
    loc.id(db)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub(crate) name: Option<Name>,
    pub(crate) variants: Vec<(Name, EnumVariant)>,
}

impl EnumData {
    fn new(enum_def: &ast::EnumDef, variants: Vec<(Name, EnumVariant)>) -> Self {
        let name = enum_def.name().map(|n| n.as_name());
        EnumData { name, variants }
    }

    pub(crate) fn enum_data_query(db: &impl HirDatabase, def_id: DefId) -> Arc<EnumData> {
        let def_loc = def_id.loc(db);
        assert!(def_loc.kind == DefKind::Enum);
        let syntax = db.file_item(def_loc.source_item_id);
        let enum_def = ast::EnumDef::cast(&syntax).expect("enum def should point to EnumDef node");
        let variants = if let Some(vl) = enum_def.variant_list() {
            vl.variants()
                .filter_map(|variant_def| {
                    let name = variant_def.name().map(|n| n.as_name());

                    name.map(|n| {
                        let def_id =
                            get_def_id(db, &def_loc, variant_def.syntax(), DefKind::EnumVariant);
                        (n, EnumVariant::new(def_id))
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        Arc::new(EnumData::new(enum_def, variants))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariantData {
    pub(crate) name: Option<Name>,
    pub(crate) variant_data: Arc<VariantData>,
    pub(crate) parent_enum: Enum,
}

impl EnumVariantData {
    fn new(variant_def: &ast::EnumVariant, parent_enum: Enum) -> EnumVariantData {
        let name = variant_def.name().map(|n| n.as_name());
        let variant_data = VariantData::new(variant_def.flavor());
        let variant_data = Arc::new(variant_data);
        EnumVariantData {
            name,
            variant_data,
            parent_enum,
        }
    }

    pub(crate) fn enum_variant_data_query(
        db: &impl HirDatabase,
        def_id: DefId,
    ) -> Arc<EnumVariantData> {
        let def_loc = def_id.loc(db);
        assert!(def_loc.kind == DefKind::EnumVariant);
        let syntax = db.file_item(def_loc.source_item_id);
        let variant_def = ast::EnumVariant::cast(&syntax)
            .expect("enum variant def should point to EnumVariant node");
        let enum_node = syntax
            .parent()
            .expect("enum variant should have enum variant list ancestor")
            .parent()
            .expect("enum variant list should have enum ancestor");
        let enum_def_id = get_def_id(db, &def_loc, enum_node, DefKind::Enum);

        Arc::new(EnumVariantData::new(variant_def, Enum::new(enum_def_id)))
    }
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
