//! AST -> `ItemTree` lowering code.

use std::{cell::OnceCell, collections::hash_map::Entry};

use hir_expand::{
    mod_path::path,
    name::AsName,
    span_map::{SpanMap, SpanMapRef},
    HirFileId,
};
use intern::{sym, Symbol};
use la_arena::Arena;
use rustc_hash::FxHashMap;
use span::{AstIdMap, SyntaxContextId};
use stdx::thin_vec::ThinVec;
use syntax::{
    ast::{self, HasModuleItem, HasName, HasTypeBounds, IsString},
    AstNode,
};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    generics::{GenericParams, GenericParamsCollector, TypeParamData, TypeParamProvenance},
    item_tree::{
        AssocItem, AttrOwner, Const, Either, Enum, ExternBlock, ExternCrate, Field, FieldParent,
        FieldsShape, FileItemTreeId, FnFlags, Function, GenericArgs, GenericItemSourceMapBuilder,
        GenericModItem, Idx, Impl, ImportAlias, Interned, ItemTree, ItemTreeData,
        ItemTreeSourceMaps, ItemTreeSourceMapsBuilder, Macro2, MacroCall, MacroRules, Mod, ModItem,
        ModKind, ModPath, Mutability, Name, Param, Path, Range, RawAttrs, RawIdx, RawVisibilityId,
        Static, Struct, StructKind, Trait, TraitAlias, TypeAlias, Union, Use, UseTree, UseTreeKind,
        Variant,
    },
    lower::LowerCtx,
    path::AssociatedTypeBinding,
    type_ref::{
        LifetimeRef, RefType, TraitBoundModifier, TraitRef, TypeBound, TypeRef, TypeRefId,
        TypesMap, TypesSourceMap,
    },
    visibility::RawVisibility,
    LocalLifetimeParamId, LocalTypeOrConstParamId,
};

fn id<N>(index: Idx<N>) -> FileItemTreeId<N> {
    FileItemTreeId(index)
}

pub(super) struct Ctx<'a> {
    db: &'a dyn DefDatabase,
    tree: ItemTree,
    source_ast_id_map: Arc<AstIdMap>,
    generic_param_attr_buffer:
        FxHashMap<Either<LocalTypeOrConstParamId, LocalLifetimeParamId>, RawAttrs>,
    span_map: OnceCell<SpanMap>,
    file: HirFileId,
    source_maps: ItemTreeSourceMapsBuilder,
}

impl<'a> Ctx<'a> {
    pub(super) fn new(db: &'a dyn DefDatabase, file: HirFileId) -> Self {
        Self {
            db,
            tree: ItemTree::default(),
            generic_param_attr_buffer: FxHashMap::default(),
            source_ast_id_map: db.ast_id_map(file),
            file,
            span_map: OnceCell::new(),
            source_maps: ItemTreeSourceMapsBuilder::default(),
        }
    }

    pub(super) fn span_map(&self) -> SpanMapRef<'_> {
        self.span_map.get_or_init(|| self.db.span_map(self.file)).as_ref()
    }

    fn body_ctx<'b, 'c>(
        &self,
        types_map: &'b mut TypesMap,
        types_source_map: &'b mut TypesSourceMap,
    ) -> LowerCtx<'c>
    where
        'a: 'c,
        'b: 'c,
    {
        // FIXME: This seems a bit wasteful that if `LowerCtx` will initialize the span map we won't benefit.
        LowerCtx::with_span_map_cell(
            self.db,
            self.file,
            self.span_map.clone(),
            types_map,
            types_source_map,
        )
    }

    pub(super) fn lower_module_items(
        mut self,
        item_owner: &dyn HasModuleItem,
    ) -> (ItemTree, ItemTreeSourceMaps) {
        self.tree.top_level =
            item_owner.items().flat_map(|item| self.lower_mod_item(&item)).collect();
        assert!(self.generic_param_attr_buffer.is_empty());
        (self.tree, self.source_maps.build())
    }

    pub(super) fn lower_macro_stmts(
        mut self,
        stmts: ast::MacroStmts,
    ) -> (ItemTree, ItemTreeSourceMaps) {
        self.tree.top_level = stmts
            .statements()
            .filter_map(|stmt| {
                match stmt {
                    ast::Stmt::Item(item) => Some(item),
                    // Macro calls can be both items and expressions. The syntax library always treats
                    // them as expressions here, so we undo that.
                    ast::Stmt::ExprStmt(es) => match es.expr()? {
                        ast::Expr::MacroExpr(expr) => {
                            cov_mark::hit!(macro_call_in_macro_stmts_is_added_to_item_tree);
                            Some(expr.macro_call()?.into())
                        }
                        _ => None,
                    },
                    _ => None,
                }
            })
            .flat_map(|item| self.lower_mod_item(&item))
            .collect();

        if let Some(ast::Expr::MacroExpr(tail_macro)) = stmts.expr() {
            if let Some(call) = tail_macro.macro_call() {
                cov_mark::hit!(macro_stmt_with_trailing_macro_expr);
                if let Some(mod_item) = self.lower_mod_item(&call.into()) {
                    self.tree.top_level.push(mod_item);
                }
            }
        }

        assert!(self.generic_param_attr_buffer.is_empty());
        (self.tree, self.source_maps.build())
    }

    pub(super) fn lower_block(mut self, block: &ast::BlockExpr) -> (ItemTree, ItemTreeSourceMaps) {
        self.tree
            .attrs
            .insert(AttrOwner::TopLevel, RawAttrs::new(self.db.upcast(), block, self.span_map()));
        self.tree.top_level = block
            .statements()
            .filter_map(|stmt| match stmt {
                ast::Stmt::Item(item) => self.lower_mod_item(&item),
                // Macro calls can be both items and expressions. The syntax library always treats
                // them as expressions here, so we undo that.
                ast::Stmt::ExprStmt(es) => match es.expr()? {
                    ast::Expr::MacroExpr(expr) => self.lower_mod_item(&expr.macro_call()?.into()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        if let Some(ast::Expr::MacroExpr(expr)) = block.tail_expr() {
            if let Some(call) = expr.macro_call() {
                if let Some(mod_item) = self.lower_mod_item(&call.into()) {
                    self.tree.top_level.push(mod_item);
                }
            }
        }

        assert!(self.generic_param_attr_buffer.is_empty());
        (self.tree, self.source_maps.build())
    }

    fn data(&mut self) -> &mut ItemTreeData {
        self.tree.data_mut()
    }

    fn lower_mod_item(&mut self, item: &ast::Item) -> Option<ModItem> {
        let mod_item: ModItem = match item {
            ast::Item::Struct(ast) => self.lower_struct(ast)?.into(),
            ast::Item::Union(ast) => self.lower_union(ast)?.into(),
            ast::Item::Enum(ast) => self.lower_enum(ast)?.into(),
            ast::Item::Fn(ast) => self.lower_function(ast)?.into(),
            ast::Item::TypeAlias(ast) => self.lower_type_alias(ast)?.into(),
            ast::Item::Static(ast) => self.lower_static(ast)?.into(),
            ast::Item::Const(ast) => self.lower_const(ast).into(),
            ast::Item::Module(ast) => self.lower_module(ast)?.into(),
            ast::Item::Trait(ast) => self.lower_trait(ast)?.into(),
            ast::Item::TraitAlias(ast) => self.lower_trait_alias(ast)?.into(),
            ast::Item::Impl(ast) => self.lower_impl(ast).into(),
            ast::Item::Use(ast) => self.lower_use(ast)?.into(),
            ast::Item::ExternCrate(ast) => self.lower_extern_crate(ast)?.into(),
            ast::Item::MacroCall(ast) => self.lower_macro_call(ast)?.into(),
            ast::Item::MacroRules(ast) => self.lower_macro_rules(ast)?.into(),
            ast::Item::MacroDef(ast) => self.lower_macro_def(ast)?.into(),
            ast::Item::ExternBlock(ast) => self.lower_extern_block(ast).into(),
        };
        let attrs = RawAttrs::new(self.db.upcast(), item, self.span_map());
        self.add_attrs(mod_item.into(), attrs);

        Some(mod_item)
    }

    fn add_attrs(&mut self, item: AttrOwner, attrs: RawAttrs) {
        if !attrs.is_empty() {
            match self.tree.attrs.entry(item) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() = entry.get().merge(attrs);
                }
                Entry::Vacant(entry) => {
                    entry.insert(attrs);
                }
            }
        }
    }

    fn lower_assoc_item(&mut self, item_node: &ast::AssocItem) -> Option<AssocItem> {
        let item: AssocItem = match item_node {
            ast::AssocItem::Fn(ast) => self.lower_function(ast).map(Into::into),
            ast::AssocItem::TypeAlias(ast) => self.lower_type_alias(ast).map(Into::into),
            ast::AssocItem::Const(ast) => Some(self.lower_const(ast).into()),
            ast::AssocItem::MacroCall(ast) => self.lower_macro_call(ast).map(Into::into),
        }?;
        let attrs = RawAttrs::new(self.db.upcast(), item_node, self.span_map());
        self.add_attrs(
            match item {
                AssocItem::Function(it) => AttrOwner::ModItem(ModItem::Function(it)),
                AssocItem::TypeAlias(it) => AttrOwner::ModItem(ModItem::TypeAlias(it)),
                AssocItem::Const(it) => AttrOwner::ModItem(ModItem::Const(it)),
                AssocItem::MacroCall(it) => AttrOwner::ModItem(ModItem::MacroCall(it)),
            },
            attrs,
        );
        Some(item)
    }

    fn lower_struct(&mut self, strukt: &ast::Struct) -> Option<FileItemTreeId<Struct>> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        let visibility = self.lower_visibility(strukt);
        let name = strukt.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(strukt);
        let (fields, kind, attrs) = self.lower_fields(&strukt.kind(), &body_ctx);
        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::No, strukt);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = Struct {
            name,
            visibility,
            generic_params,
            fields,
            shape: kind,
            ast_id,
            types_map: Arc::new(types_map),
        };
        let id = id(self.data().structs.alloc(res));
        self.source_maps.structs.push(GenericItemSourceMapBuilder {
            item: types_source_map,
            generics: generics_source_map,
        });
        for (idx, attr) in attrs {
            self.add_attrs(
                AttrOwner::Field(
                    FieldParent::Struct(id),
                    Idx::from_raw(RawIdx::from_u32(idx as u32)),
                ),
                attr,
            );
        }
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_fields(
        &mut self,
        strukt_kind: &ast::StructKind,
        body_ctx: &LowerCtx<'_>,
    ) -> (Box<[Field]>, FieldsShape, Vec<(usize, RawAttrs)>) {
        match strukt_kind {
            ast::StructKind::Record(it) => {
                let mut fields = vec![];
                let mut attrs = vec![];

                for (i, field) in it.fields().enumerate() {
                    let data = self.lower_record_field(&field, body_ctx);
                    fields.push(data);
                    let attr = RawAttrs::new(self.db.upcast(), &field, self.span_map());
                    if !attr.is_empty() {
                        attrs.push((i, attr))
                    }
                }
                (fields.into(), FieldsShape::Record, attrs)
            }
            ast::StructKind::Tuple(it) => {
                let mut fields = vec![];
                let mut attrs = vec![];

                for (i, field) in it.fields().enumerate() {
                    let data = self.lower_tuple_field(i, &field, body_ctx);
                    fields.push(data);
                    let attr = RawAttrs::new(self.db.upcast(), &field, self.span_map());
                    if !attr.is_empty() {
                        attrs.push((i, attr))
                    }
                }
                (fields.into(), FieldsShape::Tuple, attrs)
            }
            ast::StructKind::Unit => (Box::default(), FieldsShape::Unit, Vec::default()),
        }
    }

    fn lower_record_field(&mut self, field: &ast::RecordField, body_ctx: &LowerCtx<'_>) -> Field {
        let name = match field.name() {
            Some(name) => name.as_name(),
            None => Name::missing(),
        };
        let visibility = self.lower_visibility(field);
        let type_ref = TypeRef::from_ast_opt(body_ctx, field.ty());

        Field { name, type_ref, visibility }
    }

    fn lower_tuple_field(
        &mut self,
        idx: usize,
        field: &ast::TupleField,
        body_ctx: &LowerCtx<'_>,
    ) -> Field {
        let name = Name::new_tuple_field(idx);
        let visibility = self.lower_visibility(field);
        let type_ref = TypeRef::from_ast_opt(body_ctx, field.ty());
        Field { name, type_ref, visibility }
    }

    fn lower_union(&mut self, union: &ast::Union) -> Option<FileItemTreeId<Union>> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        let visibility = self.lower_visibility(union);
        let name = union.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(union);
        let (fields, _, attrs) = match union.record_field_list() {
            Some(record_field_list) => {
                self.lower_fields(&StructKind::Record(record_field_list), &body_ctx)
            }
            None => (Box::default(), FieldsShape::Record, Vec::default()),
        };
        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::No, union);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = Union {
            name,
            visibility,
            generic_params,
            fields,
            ast_id,
            types_map: Arc::new(types_map),
        };
        let id = id(self.data().unions.alloc(res));
        self.source_maps.unions.push(GenericItemSourceMapBuilder {
            item: types_source_map,
            generics: generics_source_map,
        });
        for (idx, attr) in attrs {
            self.add_attrs(
                AttrOwner::Field(
                    FieldParent::Union(id),
                    Idx::from_raw(RawIdx::from_u32(idx as u32)),
                ),
                attr,
            );
        }
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_enum(&mut self, enum_: &ast::Enum) -> Option<FileItemTreeId<Enum>> {
        let visibility = self.lower_visibility(enum_);
        let name = enum_.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(enum_);
        let variants = match &enum_.variant_list() {
            Some(variant_list) => self.lower_variants(variant_list),
            None => {
                FileItemTreeId(self.next_variant_idx())..FileItemTreeId(self.next_variant_idx())
            }
        };
        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::No, enum_);
        let res = Enum { name, visibility, generic_params, variants, ast_id };
        let id = id(self.data().enums.alloc(res));
        self.source_maps.enum_generics.push(generics_source_map);
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_variants(&mut self, variants: &ast::VariantList) -> Range<FileItemTreeId<Variant>> {
        let start = self.next_variant_idx();
        for variant in variants.variants() {
            let idx = self.lower_variant(&variant);
            self.add_attrs(
                id(idx).into(),
                RawAttrs::new(self.db.upcast(), &variant, self.span_map()),
            );
        }
        let end = self.next_variant_idx();
        FileItemTreeId(start)..FileItemTreeId(end)
    }

    fn lower_variant(&mut self, variant: &ast::Variant) -> Idx<Variant> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        let name = match variant.name() {
            Some(name) => name.as_name(),
            None => Name::missing(),
        };
        let (fields, kind, attrs) = self.lower_fields(&variant.kind(), &body_ctx);
        let ast_id = self.source_ast_id_map.ast_id(variant);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = Variant { name, fields, shape: kind, ast_id, types_map: Arc::new(types_map) };
        let id = self.data().variants.alloc(res);
        self.source_maps.variants.push(types_source_map);
        for (idx, attr) in attrs {
            self.add_attrs(
                AttrOwner::Field(
                    FieldParent::Variant(FileItemTreeId(id)),
                    Idx::from_raw(RawIdx::from_u32(idx as u32)),
                ),
                attr,
            );
        }
        id
    }

    fn lower_function(&mut self, func: &ast::Fn) -> Option<FileItemTreeId<Function>> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);

        let visibility = self.lower_visibility(func);
        let name = func.name()?.as_name();

        let mut has_self_param = false;
        let mut has_var_args = false;
        let mut params = vec![];
        let mut attrs = vec![];
        let mut push_attr = |idx, attr: RawAttrs| {
            if !attr.is_empty() {
                attrs.push((idx, attr))
            }
        };
        if let Some(param_list) = func.param_list() {
            if let Some(self_param) = param_list.self_param() {
                push_attr(
                    params.len(),
                    RawAttrs::new(self.db.upcast(), &self_param, self.span_map()),
                );
                let self_type = match self_param.ty() {
                    Some(type_ref) => TypeRef::from_ast(&body_ctx, type_ref),
                    None => {
                        let self_type = body_ctx.alloc_type_ref_desugared(TypeRef::Path(
                            Name::new_symbol_root(sym::Self_.clone()).into(),
                        ));
                        match self_param.kind() {
                            ast::SelfParamKind::Owned => self_type,
                            ast::SelfParamKind::Ref => body_ctx.alloc_type_ref_desugared(
                                TypeRef::Reference(Box::new(RefType {
                                    ty: self_type,
                                    lifetime: self_param.lifetime().as_ref().map(LifetimeRef::new),
                                    mutability: Mutability::Shared,
                                })),
                            ),
                            ast::SelfParamKind::MutRef => body_ctx.alloc_type_ref_desugared(
                                TypeRef::Reference(Box::new(RefType {
                                    ty: self_type,
                                    lifetime: self_param.lifetime().as_ref().map(LifetimeRef::new),
                                    mutability: Mutability::Mut,
                                })),
                            ),
                        }
                    }
                };
                params.push(Param { type_ref: Some(self_type) });
                has_self_param = true;
            }
            for param in param_list.params() {
                push_attr(params.len(), RawAttrs::new(self.db.upcast(), &param, self.span_map()));
                let param = match param.dotdotdot_token() {
                    Some(_) => {
                        has_var_args = true;
                        Param { type_ref: None }
                    }
                    None => {
                        let type_ref = TypeRef::from_ast_opt(&body_ctx, param.ty());
                        Param { type_ref: Some(type_ref) }
                    }
                };
                params.push(param);
            }
        }

        let ret_type = match func.ret_type() {
            Some(rt) => match rt.ty() {
                Some(type_ref) => TypeRef::from_ast(&body_ctx, type_ref),
                None if rt.thin_arrow_token().is_some() => body_ctx.alloc_error_type(),
                None => body_ctx.alloc_type_ref_desugared(TypeRef::unit()),
            },
            None => body_ctx.alloc_type_ref_desugared(TypeRef::unit()),
        };

        let ret_type = if func.async_token().is_some() {
            let future_impl = desugar_future_path(ret_type);
            let ty_bound = TypeBound::Path(future_impl, TraitBoundModifier::None);
            body_ctx.alloc_type_ref_desugared(TypeRef::ImplTrait(ThinVec::from_iter([ty_bound])))
        } else {
            ret_type
        };

        let abi = func.abi().map(lower_abi);

        let ast_id = self.source_ast_id_map.ast_id(func);

        let mut flags = FnFlags::default();
        if func.body().is_some() {
            flags |= FnFlags::HAS_BODY;
        }
        if has_self_param {
            flags |= FnFlags::HAS_SELF_PARAM;
        }
        if func.default_token().is_some() {
            flags |= FnFlags::HAS_DEFAULT_KW;
        }
        if func.const_token().is_some() {
            flags |= FnFlags::HAS_CONST_KW;
        }
        if func.async_token().is_some() {
            flags |= FnFlags::HAS_ASYNC_KW;
        }
        if func.unsafe_token().is_some() {
            flags |= FnFlags::HAS_UNSAFE_KW;
        }
        if func.safe_token().is_some() {
            flags |= FnFlags::HAS_SAFE_KW;
        }
        if has_var_args {
            flags |= FnFlags::IS_VARARGS;
        }

        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::No, func);
        let res = Function {
            name,
            visibility,
            explicit_generic_params: generic_params,
            abi,
            params: params.into_boxed_slice(),
            ret_type,
            ast_id,
            types_map: Arc::new(types_map),
            flags,
        };

        let id = id(self.data().functions.alloc(res));
        self.source_maps.functions.push(GenericItemSourceMapBuilder {
            item: types_source_map,
            generics: generics_source_map,
        });
        for (idx, attr) in attrs {
            self.add_attrs(AttrOwner::Param(id, Idx::from_raw(RawIdx::from_u32(idx as u32))), attr);
        }
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_type_alias(
        &mut self,
        type_alias: &ast::TypeAlias,
    ) -> Option<FileItemTreeId<TypeAlias>> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        let name = type_alias.name()?.as_name();
        let type_ref = type_alias.ty().map(|it| TypeRef::from_ast(&body_ctx, it));
        let visibility = self.lower_visibility(type_alias);
        let bounds = self.lower_type_bounds(type_alias, &body_ctx);
        let ast_id = self.source_ast_id_map.ast_id(type_alias);
        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::No, type_alias);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = TypeAlias {
            name,
            visibility,
            bounds,
            generic_params,
            type_ref,
            ast_id,
            types_map: Arc::new(types_map),
        };
        let id = id(self.data().type_aliases.alloc(res));
        self.source_maps.type_aliases.push(GenericItemSourceMapBuilder {
            item: types_source_map,
            generics: generics_source_map,
        });
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_static(&mut self, static_: &ast::Static) -> Option<FileItemTreeId<Static>> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        let name = static_.name()?.as_name();
        let type_ref = TypeRef::from_ast_opt(&body_ctx, static_.ty());
        let visibility = self.lower_visibility(static_);
        let mutable = static_.mut_token().is_some();
        let has_safe_kw = static_.safe_token().is_some();
        let has_unsafe_kw = static_.unsafe_token().is_some();
        let ast_id = self.source_ast_id_map.ast_id(static_);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = Static {
            name,
            visibility,
            mutable,
            type_ref,
            ast_id,
            has_safe_kw,
            has_unsafe_kw,
            types_map: Arc::new(types_map),
        };
        self.source_maps.statics.push(types_source_map);
        Some(id(self.data().statics.alloc(res)))
    }

    fn lower_const(&mut self, konst: &ast::Const) -> FileItemTreeId<Const> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        let name = konst.name().map(|it| it.as_name());
        let type_ref = TypeRef::from_ast_opt(&body_ctx, konst.ty());
        let visibility = self.lower_visibility(konst);
        let ast_id = self.source_ast_id_map.ast_id(konst);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = Const {
            name,
            visibility,
            type_ref,
            ast_id,
            has_body: konst.body().is_some(),
            types_map: Arc::new(types_map),
        };
        self.source_maps.consts.push(types_source_map);
        id(self.data().consts.alloc(res))
    }

    fn lower_module(&mut self, module: &ast::Module) -> Option<FileItemTreeId<Mod>> {
        let name = module.name()?.as_name();
        let visibility = self.lower_visibility(module);
        let kind = if module.semicolon_token().is_some() {
            ModKind::Outline
        } else {
            ModKind::Inline {
                items: module
                    .item_list()
                    .map(|list| list.items().flat_map(|item| self.lower_mod_item(&item)).collect())
                    .unwrap_or_else(|| {
                        cov_mark::hit!(name_res_works_for_broken_modules);
                        Box::new([]) as Box<[_]>
                    }),
            }
        };
        let ast_id = self.source_ast_id_map.ast_id(module);
        let res = Mod { name, visibility, kind, ast_id };
        Some(id(self.data().mods.alloc(res)))
    }

    fn lower_trait(&mut self, trait_def: &ast::Trait) -> Option<FileItemTreeId<Trait>> {
        let name = trait_def.name()?.as_name();
        let visibility = self.lower_visibility(trait_def);
        let ast_id = self.source_ast_id_map.ast_id(trait_def);
        let is_auto = trait_def.auto_token().is_some();
        let is_unsafe = trait_def.unsafe_token().is_some();

        let items = trait_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|list| list.assoc_items())
            .filter_map(|item_node| self.lower_assoc_item(&item_node))
            .collect();

        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::Yes(trait_def.type_bound_list()), trait_def);
        let def = Trait { name, visibility, generic_params, is_auto, is_unsafe, items, ast_id };
        let id = id(self.data().traits.alloc(def));
        self.source_maps.trait_generics.push(generics_source_map);
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_trait_alias(
        &mut self,
        trait_alias_def: &ast::TraitAlias,
    ) -> Option<FileItemTreeId<TraitAlias>> {
        let name = trait_alias_def.name()?.as_name();
        let visibility = self.lower_visibility(trait_alias_def);
        let ast_id = self.source_ast_id_map.ast_id(trait_alias_def);
        let (generic_params, generics_source_map) = self.lower_generic_params(
            HasImplicitSelf::Yes(trait_alias_def.type_bound_list()),
            trait_alias_def,
        );

        let alias = TraitAlias { name, visibility, generic_params, ast_id };
        let id = id(self.data().trait_aliases.alloc(alias));
        self.source_maps.trait_alias_generics.push(generics_source_map);
        self.write_generic_params_attributes(id.into());
        Some(id)
    }

    fn lower_impl(&mut self, impl_def: &ast::Impl) -> FileItemTreeId<Impl> {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);

        let ast_id = self.source_ast_id_map.ast_id(impl_def);
        // FIXME: If trait lowering fails, due to a non PathType for example, we treat this impl
        // as if it was an non-trait impl. Ideally we want to create a unique missing ref that only
        // equals itself.
        let self_ty = TypeRef::from_ast_opt(&body_ctx, impl_def.self_ty());
        let target_trait = impl_def.trait_().and_then(|tr| TraitRef::from_ast(&body_ctx, tr));
        let is_negative = impl_def.excl_token().is_some();
        let is_unsafe = impl_def.unsafe_token().is_some();

        // We cannot use `assoc_items()` here as that does not include macro calls.
        let items = impl_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|it| it.assoc_items())
            .filter_map(|item| self.lower_assoc_item(&item))
            .collect();
        // Note that trait impls don't get implicit `Self` unlike traits, because here they are a
        // type alias rather than a type parameter, so this is handled by the resolver.
        let (generic_params, generics_source_map) =
            self.lower_generic_params(HasImplicitSelf::No, impl_def);
        types_map.shrink_to_fit();
        types_source_map.shrink_to_fit();
        let res = Impl {
            generic_params,
            target_trait,
            self_ty,
            is_negative,
            is_unsafe,
            items,
            ast_id,
            types_map: Arc::new(types_map),
        };
        let id = id(self.data().impls.alloc(res));
        self.source_maps.impls.push(GenericItemSourceMapBuilder {
            item: types_source_map,
            generics: generics_source_map,
        });
        self.write_generic_params_attributes(id.into());
        id
    }

    fn lower_use(&mut self, use_item: &ast::Use) -> Option<FileItemTreeId<Use>> {
        let visibility = self.lower_visibility(use_item);
        let ast_id = self.source_ast_id_map.ast_id(use_item);
        let (use_tree, _) = lower_use_tree(self.db, use_item.use_tree()?, &mut |range| {
            self.span_map().span_for_range(range).ctx
        })?;

        let res = Use { visibility, ast_id, use_tree };
        Some(id(self.data().uses.alloc(res)))
    }

    fn lower_extern_crate(
        &mut self,
        extern_crate: &ast::ExternCrate,
    ) -> Option<FileItemTreeId<ExternCrate>> {
        let name = extern_crate.name_ref()?.as_name();
        let alias = extern_crate.rename().map(|a| {
            a.name().map(|it| it.as_name()).map_or(ImportAlias::Underscore, ImportAlias::Alias)
        });
        let visibility = self.lower_visibility(extern_crate);
        let ast_id = self.source_ast_id_map.ast_id(extern_crate);

        let res = ExternCrate { name, alias, visibility, ast_id };
        Some(id(self.data().extern_crates.alloc(res)))
    }

    fn lower_macro_call(&mut self, m: &ast::MacroCall) -> Option<FileItemTreeId<MacroCall>> {
        let span_map = self.span_map();
        let path = m.path()?;
        let range = path.syntax().text_range();
        let path = Interned::new(ModPath::from_src(self.db.upcast(), path, &mut |range| {
            span_map.span_for_range(range).ctx
        })?);
        let ast_id = self.source_ast_id_map.ast_id(m);
        let expand_to = hir_expand::ExpandTo::from_call_site(m);
        let res = MacroCall { path, ast_id, expand_to, ctxt: span_map.span_for_range(range).ctx };
        Some(id(self.data().macro_calls.alloc(res)))
    }

    fn lower_macro_rules(&mut self, m: &ast::MacroRules) -> Option<FileItemTreeId<MacroRules>> {
        let name = m.name()?;
        let ast_id = self.source_ast_id_map.ast_id(m);

        let res = MacroRules { name: name.as_name(), ast_id };
        Some(id(self.data().macro_rules.alloc(res)))
    }

    fn lower_macro_def(&mut self, m: &ast::MacroDef) -> Option<FileItemTreeId<Macro2>> {
        let name = m.name()?;

        let ast_id = self.source_ast_id_map.ast_id(m);
        let visibility = self.lower_visibility(m);

        let res = Macro2 { name: name.as_name(), ast_id, visibility };
        Some(id(self.data().macro_defs.alloc(res)))
    }

    fn lower_extern_block(&mut self, block: &ast::ExternBlock) -> FileItemTreeId<ExternBlock> {
        let ast_id = self.source_ast_id_map.ast_id(block);
        let abi = block.abi().map(lower_abi);
        let children: Box<[_]> = block.extern_item_list().map_or(Box::new([]), |list| {
            list.extern_items()
                .filter_map(|item| {
                    // Note: All items in an `extern` block need to be lowered as if they're outside of one
                    // (in other words, the knowledge that they're in an extern block must not be used).
                    // This is because an extern block can contain macros whose ItemTree's top-level items
                    // should be considered to be in an extern block too.
                    let mod_item: ModItem = match &item {
                        ast::ExternItem::Fn(ast) => self.lower_function(ast)?.into(),
                        ast::ExternItem::Static(ast) => self.lower_static(ast)?.into(),
                        ast::ExternItem::TypeAlias(ty) => self.lower_type_alias(ty)?.into(),
                        ast::ExternItem::MacroCall(call) => self.lower_macro_call(call)?.into(),
                    };
                    let attrs = RawAttrs::new(self.db.upcast(), &item, self.span_map());
                    self.add_attrs(mod_item.into(), attrs);
                    Some(mod_item)
                })
                .collect()
        });

        let res = ExternBlock { abi, ast_id, children };
        id(self.data().extern_blocks.alloc(res))
    }

    fn write_generic_params_attributes(&mut self, parent: GenericModItem) {
        self.generic_param_attr_buffer.drain().for_each(|(idx, attrs)| {
            self.tree.attrs.insert(
                match idx {
                    Either::Left(id) => AttrOwner::TypeOrConstParamData(parent, id),
                    Either::Right(id) => AttrOwner::LifetimeParamData(parent, id),
                },
                attrs,
            );
        })
    }

    fn lower_generic_params(
        &mut self,
        has_implicit_self: HasImplicitSelf,
        node: &dyn ast::HasGenericParams,
    ) -> (Arc<GenericParams>, TypesSourceMap) {
        let (mut types_map, mut types_source_map) =
            (TypesMap::default(), TypesSourceMap::default());
        let body_ctx = self.body_ctx(&mut types_map, &mut types_source_map);
        debug_assert!(self.generic_param_attr_buffer.is_empty(),);
        let add_param_attrs = |item: Either<LocalTypeOrConstParamId, LocalLifetimeParamId>,
                               param| {
            let attrs = RawAttrs::new(self.db.upcast(), &param, body_ctx.span_map());
            debug_assert!(self.generic_param_attr_buffer.insert(item, attrs).is_none());
        };
        body_ctx.take_impl_traits_bounds();
        let mut generics = GenericParamsCollector::default();

        if let HasImplicitSelf::Yes(bounds) = has_implicit_self {
            // Traits and trait aliases get the Self type as an implicit first type parameter.
            generics.type_or_consts.alloc(
                TypeParamData {
                    name: Some(Name::new_symbol_root(sym::Self_.clone())),
                    default: None,
                    provenance: TypeParamProvenance::TraitSelf,
                }
                .into(),
            );
            // add super traits as bounds on Self
            // i.e., `trait Foo: Bar` is equivalent to `trait Foo where Self: Bar`
            generics.fill_bounds(
                &body_ctx,
                bounds,
                Either::Left(body_ctx.alloc_type_ref_desugared(TypeRef::Path(
                    Name::new_symbol_root(sym::Self_.clone()).into(),
                ))),
            );
        }

        generics.fill(&body_ctx, node, add_param_attrs);

        let generics = generics.finish(types_map, &mut types_source_map);
        (generics, types_source_map)
    }

    fn lower_type_bounds(
        &mut self,
        node: &dyn ast::HasTypeBounds,
        body_ctx: &LowerCtx<'_>,
    ) -> Box<[TypeBound]> {
        match node.type_bound_list() {
            Some(bound_list) => {
                bound_list.bounds().map(|it| TypeBound::from_ast(body_ctx, it)).collect()
            }
            None => Box::default(),
        }
    }

    fn lower_visibility(&mut self, item: &dyn ast::HasVisibility) -> RawVisibilityId {
        let vis = RawVisibility::from_ast(self.db, item.visibility(), &mut |range| {
            self.span_map().span_for_range(range).ctx
        });
        self.data().vis.alloc(vis)
    }

    fn next_variant_idx(&self) -> Idx<Variant> {
        Idx::from_raw(RawIdx::from(
            self.tree.data.as_ref().map_or(0, |data| data.variants.len() as u32),
        ))
    }
}

fn desugar_future_path(orig: TypeRefId) -> Path {
    let path = path![core::future::Future];
    let mut generic_args: Vec<_> =
        std::iter::repeat(None).take(path.segments().len() - 1).collect();
    let binding = AssociatedTypeBinding {
        name: Name::new_symbol_root(sym::Output.clone()),
        args: None,
        type_ref: Some(orig),
        bounds: Box::default(),
    };
    generic_args.push(Some(GenericArgs { bindings: Box::new([binding]), ..GenericArgs::empty() }));

    Path::from_known_path(path, generic_args)
}

enum HasImplicitSelf {
    /// Inner list is a type bound list for the implicit `Self`.
    Yes(Option<ast::TypeBoundList>),
    No,
}

fn lower_abi(abi: ast::Abi) -> Symbol {
    match abi.abi_string() {
        Some(tok) => Symbol::intern(tok.text_without_quotes()),
        // `extern` default to be `extern "C"`.
        _ => sym::C.clone(),
    }
}

struct UseTreeLowering<'a> {
    db: &'a dyn DefDatabase,
    mapping: Arena<ast::UseTree>,
}

impl UseTreeLowering<'_> {
    fn lower_use_tree(
        &mut self,
        tree: ast::UseTree,
        span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContextId,
    ) -> Option<UseTree> {
        if let Some(use_tree_list) = tree.use_tree_list() {
            let prefix = match tree.path() {
                // E.g. use something::{{{inner}}};
                None => None,
                // E.g. `use something::{inner}` (prefix is `None`, path is `something`)
                // or `use something::{path::{inner::{innerer}}}` (prefix is `something::path`, path is `inner`)
                Some(path) => {
                    match ModPath::from_src(self.db.upcast(), path, span_for_range) {
                        Some(it) => Some(it),
                        None => return None, // FIXME: report errors somewhere
                    }
                }
            };

            let list = use_tree_list
                .use_trees()
                .filter_map(|tree| self.lower_use_tree(tree, span_for_range))
                .collect();

            Some(
                self.use_tree(
                    UseTreeKind::Prefixed { prefix: prefix.map(Interned::new), list },
                    tree,
                ),
            )
        } else {
            let is_glob = tree.star_token().is_some();
            let path = match tree.path() {
                Some(path) => Some(ModPath::from_src(self.db.upcast(), path, span_for_range)?),
                None => None,
            };
            let alias = tree.rename().map(|a| {
                a.name().map(|it| it.as_name()).map_or(ImportAlias::Underscore, ImportAlias::Alias)
            });
            if alias.is_some() && is_glob {
                return None;
            }

            match (path, alias, is_glob) {
                (path, None, true) => {
                    if path.is_none() {
                        cov_mark::hit!(glob_enum_group);
                    }
                    Some(self.use_tree(UseTreeKind::Glob { path: path.map(Interned::new) }, tree))
                }
                // Globs can't be renamed
                (_, Some(_), true) | (None, None, false) => None,
                // `bla::{ as Name}` is invalid
                (None, Some(_), false) => None,
                (Some(path), alias, false) => Some(
                    self.use_tree(UseTreeKind::Single { path: Interned::new(path), alias }, tree),
                ),
            }
        }
    }

    fn use_tree(&mut self, kind: UseTreeKind, ast: ast::UseTree) -> UseTree {
        let index = self.mapping.alloc(ast);
        UseTree { index, kind }
    }
}

pub(crate) fn lower_use_tree(
    db: &dyn DefDatabase,
    tree: ast::UseTree,
    span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContextId,
) -> Option<(UseTree, Arena<ast::UseTree>)> {
    let mut lowering = UseTreeLowering { db, mapping: Arena::new() };
    let tree = lowering.lower_use_tree(tree, span_for_range)?;
    Some((tree, lowering.mapping))
}
