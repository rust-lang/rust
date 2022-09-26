//! Defines hir-level representation of structs, enums and unions

use std::{num::NonZeroU32, sync::Arc};

use base_db::CrateId;
use either::Either;
use hir_expand::{
    name::{AsName, Name},
    HirFileId, InFile,
};
use la_arena::{Arena, ArenaMap};
use syntax::ast::{self, HasName, HasVisibility};
use tt::{Delimiter, DelimiterKind, Leaf, Subtree, TokenTree};

use crate::{
    body::{CfgExpander, LowerCtx},
    builtin_type::{BuiltinInt, BuiltinUint},
    db::DefDatabase,
    intern::Interned,
    item_tree::{AttrOwner, Field, FieldAstId, Fields, ItemTree, ModItem, RawVisibilityId},
    nameres::diagnostics::DefDiagnostic,
    src::HasChildSource,
    src::HasSource,
    trace::Trace,
    type_ref::TypeRef,
    visibility::RawVisibility,
    EnumId, LocalEnumVariantId, LocalFieldId, LocalModuleId, Lookup, ModuleId, StructId, UnionId,
    VariantId,
};
use cfg::CfgOptions;

/// Note that we use `StructData` for unions as well!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub name: Name,
    pub variant_data: Arc<VariantData>,
    pub repr: Option<ReprData>,
    pub visibility: RawVisibility,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub name: Name,
    pub variants: Arena<EnumVariantData>,
    pub repr: Option<ReprData>,
    pub visibility: RawVisibility,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariantData {
    pub name: Name,
    pub variant_data: Arc<VariantData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantData {
    Record(Arena<FieldData>),
    Tuple(Arena<FieldData>),
    Unit,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldData {
    pub name: Name,
    pub type_ref: Interned<TypeRef>,
    pub visibility: RawVisibility,
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub enum ReprKind {
    C,
    BuiltinInt { builtin: Either<BuiltinInt, BuiltinUint>, is_c: bool },
    Transparent,
    Default,
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct ReprData {
    pub kind: ReprKind,
    pub packed: bool,
    pub align: Option<NonZeroU32>,
}

fn repr_from_value(
    db: &dyn DefDatabase,
    krate: CrateId,
    item_tree: &ItemTree,
    of: AttrOwner,
) -> Option<ReprData> {
    item_tree.attrs(db, krate, of).by_key("repr").tt_values().find_map(parse_repr_tt)
}

fn parse_repr_tt(tt: &Subtree) -> Option<ReprData> {
    match tt.delimiter {
        Some(Delimiter { kind: DelimiterKind::Parenthesis, .. }) => {}
        _ => return None,
    }

    let mut data = ReprData { kind: ReprKind::Default, packed: false, align: None };

    let mut tts = tt.token_trees.iter().peekable();
    while let Some(tt) = tts.next() {
        if let TokenTree::Leaf(Leaf::Ident(ident)) = tt {
            match &*ident.text {
                "packed" => {
                    data.packed = true;
                    if let Some(TokenTree::Subtree(_)) = tts.peek() {
                        tts.next();
                    }
                }
                "align" => {
                    if let Some(TokenTree::Subtree(tt)) = tts.peek() {
                        tts.next();
                        if let Some(TokenTree::Leaf(Leaf::Literal(lit))) = tt.token_trees.first() {
                            if let Ok(align) = lit.text.parse() {
                                data.align = Some(align);
                            }
                        }
                    }
                }
                "C" => {
                    if let ReprKind::BuiltinInt { is_c, .. } = &mut data.kind {
                        *is_c = true;
                    } else {
                        data.kind = ReprKind::C;
                    }
                }
                "transparent" => data.kind = ReprKind::Transparent,
                repr => {
                    let is_c = matches!(data.kind, ReprKind::C);
                    if let Some(builtin) = BuiltinInt::from_suffix(repr)
                        .map(Either::Left)
                        .or_else(|| BuiltinUint::from_suffix(repr).map(Either::Right))
                    {
                        data.kind = ReprKind::BuiltinInt { builtin, is_c };
                    }
                }
            }
        }
    }

    Some(data)
}

impl StructData {
    pub(crate) fn struct_data_query(db: &dyn DefDatabase, id: StructId) -> Arc<StructData> {
        db.struct_data_with_diagnostics(id).0
    }

    pub(crate) fn struct_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: StructId,
    ) -> (Arc<StructData>, Arc<[DefDiagnostic]>) {
        let loc = id.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());
        let cfg_options = db.crate_graph()[loc.container.krate].cfg_options.clone();

        let strukt = &item_tree[loc.id.value];
        let (variant_data, diagnostics) = lower_fields(
            db,
            krate,
            loc.id.file_id(),
            loc.container.local_id,
            &item_tree,
            &cfg_options,
            &strukt.fields,
            None,
        );
        (
            Arc::new(StructData {
                name: strukt.name.clone(),
                variant_data: Arc::new(variant_data),
                repr,
                visibility: item_tree[strukt.visibility].clone(),
            }),
            diagnostics.into(),
        )
    }

    pub(crate) fn union_data_query(db: &dyn DefDatabase, id: UnionId) -> Arc<StructData> {
        db.union_data_with_diagnostics(id).0
    }

    pub(crate) fn union_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: UnionId,
    ) -> (Arc<StructData>, Arc<[DefDiagnostic]>) {
        let loc = id.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());
        let cfg_options = db.crate_graph()[loc.container.krate].cfg_options.clone();

        let union = &item_tree[loc.id.value];
        let (variant_data, diagnostics) = lower_fields(
            db,
            krate,
            loc.id.file_id(),
            loc.container.local_id,
            &item_tree,
            &cfg_options,
            &union.fields,
            None,
        );
        (
            Arc::new(StructData {
                name: union.name.clone(),
                variant_data: Arc::new(variant_data),
                repr,
                visibility: item_tree[union.visibility].clone(),
            }),
            diagnostics.into(),
        )
    }
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &dyn DefDatabase, e: EnumId) -> Arc<EnumData> {
        db.enum_data_with_diagnostics(e).0
    }

    pub(crate) fn enum_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        e: EnumId,
    ) -> (Arc<EnumData>, Arc<[DefDiagnostic]>) {
        let loc = e.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let cfg_options = db.crate_graph()[krate].cfg_options.clone();
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());

        let enum_ = &item_tree[loc.id.value];
        let mut variants = Arena::new();
        let mut diagnostics = Vec::new();
        for tree_id in enum_.variants.clone() {
            let attrs = item_tree.attrs(db, krate, tree_id.into());
            let var = &item_tree[tree_id];
            if attrs.is_cfg_enabled(&cfg_options) {
                let (var_data, field_diagnostics) = lower_fields(
                    db,
                    krate,
                    loc.id.file_id(),
                    loc.container.local_id,
                    &item_tree,
                    &cfg_options,
                    &var.fields,
                    Some(enum_.visibility),
                );
                diagnostics.extend(field_diagnostics);

                variants.alloc(EnumVariantData {
                    name: var.name.clone(),
                    variant_data: Arc::new(var_data),
                });
            } else {
                diagnostics.push(DefDiagnostic::unconfigured_code(
                    loc.container.local_id,
                    InFile::new(loc.id.file_id(), var.ast_id.upcast()),
                    attrs.cfg().unwrap(),
                    cfg_options.clone(),
                ))
            }
        }

        (
            Arc::new(EnumData {
                name: enum_.name.clone(),
                variants,
                repr,
                visibility: item_tree[enum_.visibility].clone(),
            }),
            diagnostics.into(),
        )
    }

    pub fn variant(&self, name: &Name) -> Option<LocalEnumVariantId> {
        let (id, _) = self.variants.iter().find(|(_id, data)| &data.name == name)?;
        Some(id)
    }

    pub fn variant_body_type(&self) -> Either<BuiltinInt, BuiltinUint> {
        match self.repr {
            Some(ReprData { kind: ReprKind::BuiltinInt { builtin, .. }, .. }) => builtin,
            _ => Either::Left(BuiltinInt::Isize),
        }
    }
}

impl HasChildSource<LocalEnumVariantId> for EnumId {
    type Value = ast::Variant;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<LocalEnumVariantId, Self::Value>> {
        let src = self.lookup(db).source(db);
        let mut trace = Trace::new_for_map();
        lower_enum(db, &mut trace, &src, self.lookup(db).container);
        src.with_value(trace.into_map())
    }
}

fn lower_enum(
    db: &dyn DefDatabase,
    trace: &mut Trace<EnumVariantData, ast::Variant>,
    ast: &InFile<ast::Enum>,
    module_id: ModuleId,
) {
    let expander = CfgExpander::new(db, ast.file_id, module_id.krate);
    let variants = ast
        .value
        .variant_list()
        .into_iter()
        .flat_map(|it| it.variants())
        .filter(|var| expander.is_cfg_enabled(db, var));
    for var in variants {
        trace.alloc(
            || var.clone(),
            || EnumVariantData {
                name: var.name().map_or_else(Name::missing, |it| it.as_name()),
                variant_data: Arc::new(VariantData::new(db, ast.with_value(var.kind()), module_id)),
            },
        );
    }
}

impl VariantData {
    fn new(db: &dyn DefDatabase, flavor: InFile<ast::StructKind>, module_id: ModuleId) -> Self {
        let mut expander = CfgExpander::new(db, flavor.file_id, module_id.krate);
        let mut trace = Trace::new_for_arena();
        match lower_struct(db, &mut expander, &mut trace, &flavor) {
            StructKind::Tuple => VariantData::Tuple(trace.into_arena()),
            StructKind::Record => VariantData::Record(trace.into_arena()),
            StructKind::Unit => VariantData::Unit,
        }
    }

    pub fn fields(&self) -> &Arena<FieldData> {
        const EMPTY: &Arena<FieldData> = &Arena::new();
        match &self {
            VariantData::Record(fields) | VariantData::Tuple(fields) => fields,
            _ => EMPTY,
        }
    }

    pub fn field(&self, name: &Name) -> Option<LocalFieldId> {
        self.fields().iter().find_map(|(id, data)| if &data.name == name { Some(id) } else { None })
    }

    pub fn kind(&self) -> StructKind {
        match self {
            VariantData::Record(_) => StructKind::Record,
            VariantData::Tuple(_) => StructKind::Tuple,
            VariantData::Unit => StructKind::Unit,
        }
    }
}

impl HasChildSource<LocalFieldId> for VariantId {
    type Value = Either<ast::TupleField, ast::RecordField>;

    fn child_source(&self, db: &dyn DefDatabase) -> InFile<ArenaMap<LocalFieldId, Self::Value>> {
        let (src, module_id) = match self {
            VariantId::EnumVariantId(it) => {
                // I don't really like the fact that we call into parent source
                // here, this might add to more queries then necessary.
                let src = it.parent.child_source(db);
                (src.map(|map| map[it.local_id].kind()), it.parent.lookup(db).container)
            }
            VariantId::StructId(it) => {
                (it.lookup(db).source(db).map(|it| it.kind()), it.lookup(db).container)
            }
            VariantId::UnionId(it) => (
                it.lookup(db).source(db).map(|it| {
                    it.record_field_list()
                        .map(ast::StructKind::Record)
                        .unwrap_or(ast::StructKind::Unit)
                }),
                it.lookup(db).container,
            ),
        };
        let mut expander = CfgExpander::new(db, src.file_id, module_id.krate);
        let mut trace = Trace::new_for_map();
        lower_struct(db, &mut expander, &mut trace, &src);
        src.with_value(trace.into_map())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StructKind {
    Tuple,
    Record,
    Unit,
}

fn lower_struct(
    db: &dyn DefDatabase,
    expander: &mut CfgExpander,
    trace: &mut Trace<FieldData, Either<ast::TupleField, ast::RecordField>>,
    ast: &InFile<ast::StructKind>,
) -> StructKind {
    let ctx = LowerCtx::new(db, ast.file_id);

    match &ast.value {
        ast::StructKind::Tuple(fl) => {
            for (i, fd) in fl.fields().enumerate() {
                if !expander.is_cfg_enabled(db, &fd) {
                    continue;
                }

                trace.alloc(
                    || Either::Left(fd.clone()),
                    || FieldData {
                        name: Name::new_tuple_field(i),
                        type_ref: Interned::new(TypeRef::from_ast_opt(&ctx, fd.ty())),
                        visibility: RawVisibility::from_ast(db, ast.with_value(fd.visibility())),
                    },
                );
            }
            StructKind::Tuple
        }
        ast::StructKind::Record(fl) => {
            for fd in fl.fields() {
                if !expander.is_cfg_enabled(db, &fd) {
                    continue;
                }

                trace.alloc(
                    || Either::Right(fd.clone()),
                    || FieldData {
                        name: fd.name().map(|n| n.as_name()).unwrap_or_else(Name::missing),
                        type_ref: Interned::new(TypeRef::from_ast_opt(&ctx, fd.ty())),
                        visibility: RawVisibility::from_ast(db, ast.with_value(fd.visibility())),
                    },
                );
            }
            StructKind::Record
        }
        ast::StructKind::Unit => StructKind::Unit,
    }
}

fn lower_fields(
    db: &dyn DefDatabase,
    krate: CrateId,
    current_file_id: HirFileId,
    container: LocalModuleId,
    item_tree: &ItemTree,
    cfg_options: &CfgOptions,
    fields: &Fields,
    override_visibility: Option<RawVisibilityId>,
) -> (VariantData, Vec<DefDiagnostic>) {
    let mut diagnostics = Vec::new();
    match fields {
        Fields::Record(flds) => {
            let mut arena = Arena::new();
            for field_id in flds.clone() {
                let attrs = item_tree.attrs(db, krate, field_id.into());
                let field = &item_tree[field_id];
                if attrs.is_cfg_enabled(cfg_options) {
                    arena.alloc(lower_field(item_tree, field, override_visibility));
                } else {
                    diagnostics.push(DefDiagnostic::unconfigured_code(
                        container,
                        InFile::new(
                            current_file_id,
                            match field.ast_id {
                                FieldAstId::Record(it) => it.upcast(),
                                FieldAstId::Tuple(it) => it.upcast(),
                            },
                        ),
                        attrs.cfg().unwrap(),
                        cfg_options.clone(),
                    ))
                }
            }
            (VariantData::Record(arena), diagnostics)
        }
        Fields::Tuple(flds) => {
            let mut arena = Arena::new();
            for field_id in flds.clone() {
                let attrs = item_tree.attrs(db, krate, field_id.into());
                let field = &item_tree[field_id];
                if attrs.is_cfg_enabled(cfg_options) {
                    arena.alloc(lower_field(item_tree, field, override_visibility));
                } else {
                    diagnostics.push(DefDiagnostic::unconfigured_code(
                        container,
                        InFile::new(
                            current_file_id,
                            match field.ast_id {
                                FieldAstId::Record(it) => it.upcast(),
                                FieldAstId::Tuple(it) => it.upcast(),
                            },
                        ),
                        attrs.cfg().unwrap(),
                        cfg_options.clone(),
                    ))
                }
            }
            (VariantData::Tuple(arena), diagnostics)
        }
        Fields::Unit => (VariantData::Unit, diagnostics),
    }
}

fn lower_field(
    item_tree: &ItemTree,
    field: &Field,
    override_visibility: Option<RawVisibilityId>,
) -> FieldData {
    FieldData {
        name: field.name.clone(),
        type_ref: field.type_ref.clone(),
        visibility: item_tree[override_visibility.unwrap_or(field.visibility)].clone(),
    }
}
