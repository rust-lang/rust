//! AST -> `ItemTree` lowering code.

use std::collections::hash_map::Entry;

use hir_expand::{mod_path::path, name, name::AsName, span_map::SpanMapRef, HirFileId};
use la_arena::Arena;
use span::{AstIdMap, SyntaxContextId};
use syntax::{
    ast::{self, HasModuleItem, HasName, HasTypeBounds, IsString},
    AstNode,
};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    generics::{GenericParams, GenericParamsCollector, TypeParamData, TypeParamProvenance},
    item_tree::{
        AssocItem, AttrOwner, Const, Either, Enum, ExternBlock, ExternCrate, Field, FieldAstId,
        Fields, FileItemTreeId, FnFlags, Function, GenericArgs, Idx, IdxRange, Impl, ImportAlias,
        Interned, ItemTree, ItemTreeData, ItemTreeNode, Macro2, MacroCall, MacroRules, Mod,
        ModItem, ModKind, ModPath, Mutability, Name, Param, ParamAstId, Path, Range, RawAttrs,
        RawIdx, RawVisibilityId, Static, Struct, StructKind, Trait, TraitAlias, TypeAlias, Union,
        Use, UseTree, UseTreeKind, Variant,
    },
    path::AssociatedTypeBinding,
    type_ref::{LifetimeRef, TraitBoundModifier, TraitRef, TypeBound, TypeRef},
    visibility::RawVisibility,
    LocalLifetimeParamId, LocalTypeOrConstParamId,
};

fn id<N: ItemTreeNode>(index: Idx<N>) -> FileItemTreeId<N> {
    FileItemTreeId(index)
}

pub(super) struct Ctx<'a> {
    db: &'a dyn DefDatabase,
    tree: ItemTree,
    source_ast_id_map: Arc<AstIdMap>,
    body_ctx: crate::lower::LowerCtx<'a>,
}

impl<'a> Ctx<'a> {
    pub(super) fn new(db: &'a dyn DefDatabase, file: HirFileId) -> Self {
        Self {
            db,
            tree: ItemTree::default(),
            source_ast_id_map: db.ast_id_map(file),
            body_ctx: crate::lower::LowerCtx::new(db, file),
        }
    }

    pub(super) fn span_map(&self) -> SpanMapRef<'_> {
        self.body_ctx.span_map()
    }

    pub(super) fn lower_module_items(mut self, item_owner: &dyn HasModuleItem) -> ItemTree {
        self.tree.top_level =
            item_owner.items().flat_map(|item| self.lower_mod_item(&item)).collect();
        self.tree
    }

    pub(super) fn lower_macro_stmts(mut self, stmts: ast::MacroStmts) -> ItemTree {
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

        self.tree
    }

    pub(super) fn lower_block(mut self, block: &ast::BlockExpr) -> ItemTree {
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

        self.tree
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
            ast::Item::Impl(ast) => self.lower_impl(ast)?.into(),
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
        match self.tree.attrs.entry(item) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = entry.get().merge(attrs);
            }
            Entry::Vacant(entry) => {
                entry.insert(attrs);
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
        let visibility = self.lower_visibility(strukt);
        let name = strukt.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(strukt);
        let generic_params = self.lower_generic_params(HasImplicitSelf::No, strukt);
        let fields = self.lower_fields(&strukt.kind());
        let res = Struct { name, visibility, generic_params, fields, ast_id };
        Some(id(self.data().structs.alloc(res)))
    }

    fn lower_fields(&mut self, strukt_kind: &ast::StructKind) -> Fields {
        match strukt_kind {
            ast::StructKind::Record(it) => {
                let range = self.lower_record_fields(it);
                Fields::Record(range)
            }
            ast::StructKind::Tuple(it) => {
                let range = self.lower_tuple_fields(it);
                Fields::Tuple(range)
            }
            ast::StructKind::Unit => Fields::Unit,
        }
    }

    fn lower_record_fields(&mut self, fields: &ast::RecordFieldList) -> IdxRange<Field> {
        let start = self.next_field_idx();
        for field in fields.fields() {
            if let Some(data) = self.lower_record_field(&field) {
                let idx = self.data().fields.alloc(data);
                self.add_attrs(
                    idx.into(),
                    RawAttrs::new(self.db.upcast(), &field, self.span_map()),
                );
            }
        }
        let end = self.next_field_idx();
        IdxRange::new(start..end)
    }

    fn lower_record_field(&mut self, field: &ast::RecordField) -> Option<Field> {
        let name = field.name()?.as_name();
        let visibility = self.lower_visibility(field);
        let type_ref = self.lower_type_ref_opt(field.ty());
        let ast_id = FieldAstId::Record(self.source_ast_id_map.ast_id(field));
        let res = Field { name, type_ref, visibility, ast_id };
        Some(res)
    }

    fn lower_tuple_fields(&mut self, fields: &ast::TupleFieldList) -> IdxRange<Field> {
        let start = self.next_field_idx();
        for (i, field) in fields.fields().enumerate() {
            let data = self.lower_tuple_field(i, &field);
            let idx = self.data().fields.alloc(data);
            self.add_attrs(idx.into(), RawAttrs::new(self.db.upcast(), &field, self.span_map()));
        }
        let end = self.next_field_idx();
        IdxRange::new(start..end)
    }

    fn lower_tuple_field(&mut self, idx: usize, field: &ast::TupleField) -> Field {
        let name = Name::new_tuple_field(idx);
        let visibility = self.lower_visibility(field);
        let type_ref = self.lower_type_ref_opt(field.ty());
        let ast_id = FieldAstId::Tuple(self.source_ast_id_map.ast_id(field));
        Field { name, type_ref, visibility, ast_id }
    }

    fn lower_union(&mut self, union: &ast::Union) -> Option<FileItemTreeId<Union>> {
        let visibility = self.lower_visibility(union);
        let name = union.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(union);
        let generic_params = self.lower_generic_params(HasImplicitSelf::No, union);
        let fields = match union.record_field_list() {
            Some(record_field_list) => self.lower_fields(&StructKind::Record(record_field_list)),
            None => Fields::Record(IdxRange::new(self.next_field_idx()..self.next_field_idx())),
        };
        let res = Union { name, visibility, generic_params, fields, ast_id };
        Some(id(self.data().unions.alloc(res)))
    }

    fn lower_enum(&mut self, enum_: &ast::Enum) -> Option<FileItemTreeId<Enum>> {
        let visibility = self.lower_visibility(enum_);
        let name = enum_.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(enum_);
        let generic_params = self.lower_generic_params(HasImplicitSelf::No, enum_);
        let variants = match &enum_.variant_list() {
            Some(variant_list) => self.lower_variants(variant_list),
            None => {
                FileItemTreeId(self.next_variant_idx())..FileItemTreeId(self.next_variant_idx())
            }
        };
        let res = Enum { name, visibility, generic_params, variants, ast_id };
        Some(id(self.data().enums.alloc(res)))
    }

    fn lower_variants(&mut self, variants: &ast::VariantList) -> Range<FileItemTreeId<Variant>> {
        let start = self.next_variant_idx();
        for variant in variants.variants() {
            if let Some(data) = self.lower_variant(&variant) {
                let idx = self.data().variants.alloc(data);
                self.add_attrs(
                    id(idx).into(),
                    RawAttrs::new(self.db.upcast(), &variant, self.span_map()),
                );
            }
        }
        let end = self.next_variant_idx();
        FileItemTreeId(start)..FileItemTreeId(end)
    }

    fn lower_variant(&mut self, variant: &ast::Variant) -> Option<Variant> {
        let name = variant.name()?.as_name();
        let fields = self.lower_fields(&variant.kind());
        let ast_id = self.source_ast_id_map.ast_id(variant);
        let res = Variant { name, fields, ast_id };
        Some(res)
    }

    fn lower_function(&mut self, func: &ast::Fn) -> Option<FileItemTreeId<Function>> {
        let visibility = self.lower_visibility(func);
        let name = func.name()?.as_name();

        let mut has_self_param = false;
        let start_param = self.next_param_idx();
        if let Some(param_list) = func.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = match self_param.ty() {
                    Some(type_ref) => TypeRef::from_ast(&self.body_ctx, type_ref),
                    None => {
                        let self_type = TypeRef::Path(name![Self].into());
                        match self_param.kind() {
                            ast::SelfParamKind::Owned => self_type,
                            ast::SelfParamKind::Ref => TypeRef::Reference(
                                Box::new(self_type),
                                self_param.lifetime().as_ref().map(LifetimeRef::new),
                                Mutability::Shared,
                            ),
                            ast::SelfParamKind::MutRef => TypeRef::Reference(
                                Box::new(self_type),
                                self_param.lifetime().as_ref().map(LifetimeRef::new),
                                Mutability::Mut,
                            ),
                        }
                    }
                };
                let type_ref = Interned::new(self_type);
                let ast_id = self.source_ast_id_map.ast_id(&self_param);
                let idx = self.data().params.alloc(Param {
                    type_ref: Some(type_ref),
                    ast_id: ParamAstId::SelfParam(ast_id),
                });
                self.add_attrs(
                    idx.into(),
                    RawAttrs::new(self.db.upcast(), &self_param, self.span_map()),
                );
                has_self_param = true;
            }
            for param in param_list.params() {
                let ast_id = self.source_ast_id_map.ast_id(&param);
                let idx = match param.dotdotdot_token() {
                    Some(_) => self
                        .data()
                        .params
                        .alloc(Param { type_ref: None, ast_id: ParamAstId::Param(ast_id) }),
                    None => {
                        let type_ref = TypeRef::from_ast_opt(&self.body_ctx, param.ty());
                        let ty = Interned::new(type_ref);
                        self.data()
                            .params
                            .alloc(Param { type_ref: Some(ty), ast_id: ParamAstId::Param(ast_id) })
                    }
                };
                self.add_attrs(
                    idx.into(),
                    RawAttrs::new(self.db.upcast(), &param, self.span_map()),
                );
            }
        }
        let end_param = self.next_param_idx();
        let params = IdxRange::new(start_param..end_param);

        let ret_type = match func.ret_type() {
            Some(rt) => match rt.ty() {
                Some(type_ref) => TypeRef::from_ast(&self.body_ctx, type_ref),
                None if rt.thin_arrow_token().is_some() => TypeRef::Error,
                None => TypeRef::unit(),
            },
            None => TypeRef::unit(),
        };

        let ret_type = if func.async_token().is_some() {
            let future_impl = desugar_future_path(ret_type);
            let ty_bound = Interned::new(TypeBound::Path(future_impl, TraitBoundModifier::None));
            TypeRef::ImplTrait(vec![ty_bound])
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

        let res = Function {
            name,
            visibility,
            explicit_generic_params: self.lower_generic_params(HasImplicitSelf::No, func),
            abi,
            params,
            ret_type: Interned::new(ret_type),
            ast_id,
            flags,
        };

        Some(id(self.data().functions.alloc(res)))
    }

    fn lower_type_alias(
        &mut self,
        type_alias: &ast::TypeAlias,
    ) -> Option<FileItemTreeId<TypeAlias>> {
        let name = type_alias.name()?.as_name();
        let type_ref = type_alias.ty().map(|it| self.lower_type_ref(&it));
        let visibility = self.lower_visibility(type_alias);
        let bounds = self.lower_type_bounds(type_alias);
        let ast_id = self.source_ast_id_map.ast_id(type_alias);
        let generic_params = self.lower_generic_params(HasImplicitSelf::No, type_alias);
        let res = TypeAlias { name, visibility, bounds, generic_params, type_ref, ast_id };
        Some(id(self.data().type_aliases.alloc(res)))
    }

    fn lower_static(&mut self, static_: &ast::Static) -> Option<FileItemTreeId<Static>> {
        let name = static_.name()?.as_name();
        let type_ref = self.lower_type_ref_opt(static_.ty());
        let visibility = self.lower_visibility(static_);
        let mutable = static_.mut_token().is_some();
        let ast_id = self.source_ast_id_map.ast_id(static_);
        let res = Static { name, visibility, mutable, type_ref, ast_id };
        Some(id(self.data().statics.alloc(res)))
    }

    fn lower_const(&mut self, konst: &ast::Const) -> FileItemTreeId<Const> {
        let name = konst.name().map(|it| it.as_name());
        let type_ref = self.lower_type_ref_opt(konst.ty());
        let visibility = self.lower_visibility(konst);
        let ast_id = self.source_ast_id_map.ast_id(konst);
        let res = Const { name, visibility, type_ref, ast_id };
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
        let generic_params =
            self.lower_generic_params(HasImplicitSelf::Yes(trait_def.type_bound_list()), trait_def);
        let is_auto = trait_def.auto_token().is_some();
        let is_unsafe = trait_def.unsafe_token().is_some();

        let items = trait_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|list| list.assoc_items())
            .filter_map(|item_node| self.lower_assoc_item(&item_node))
            .collect();

        let def = Trait { name, visibility, generic_params, is_auto, is_unsafe, items, ast_id };
        Some(id(self.data().traits.alloc(def)))
    }

    fn lower_trait_alias(
        &mut self,
        trait_alias_def: &ast::TraitAlias,
    ) -> Option<FileItemTreeId<TraitAlias>> {
        let name = trait_alias_def.name()?.as_name();
        let visibility = self.lower_visibility(trait_alias_def);
        let ast_id = self.source_ast_id_map.ast_id(trait_alias_def);
        let generic_params = self.lower_generic_params(
            HasImplicitSelf::Yes(trait_alias_def.type_bound_list()),
            trait_alias_def,
        );

        let alias = TraitAlias { name, visibility, generic_params, ast_id };
        Some(id(self.data().trait_aliases.alloc(alias)))
    }

    fn lower_impl(&mut self, impl_def: &ast::Impl) -> Option<FileItemTreeId<Impl>> {
        let ast_id = self.source_ast_id_map.ast_id(impl_def);
        // Note that trait impls don't get implicit `Self` unlike traits, because here they are a
        // type alias rather than a type parameter, so this is handled by the resolver.
        let generic_params = self.lower_generic_params(HasImplicitSelf::No, impl_def);
        // FIXME: If trait lowering fails, due to a non PathType for example, we treat this impl
        // as if it was an non-trait impl. Ideally we want to create a unique missing ref that only
        // equals itself.
        let target_trait = impl_def.trait_().and_then(|tr| self.lower_trait_ref(&tr));
        let self_ty = self.lower_type_ref(&impl_def.self_ty()?);
        let is_negative = impl_def.excl_token().is_some();
        let is_unsafe = impl_def.unsafe_token().is_some();

        // We cannot use `assoc_items()` here as that does not include macro calls.
        let items = impl_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|it| it.assoc_items())
            .filter_map(|item| self.lower_assoc_item(&item))
            .collect();
        let res =
            Impl { generic_params, target_trait, self_ty, is_negative, is_unsafe, items, ast_id };
        Some(id(self.data().impls.alloc(res)))
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
        let path = Interned::new(ModPath::from_src(self.db.upcast(), m.path()?, &mut |range| {
            span_map.span_for_range(range).ctx
        })?);
        let ast_id = self.source_ast_id_map.ast_id(m);
        let expand_to = hir_expand::ExpandTo::from_call_site(m);
        let res = MacroCall {
            path,
            ast_id,
            expand_to,
            call_site: span_map.span_for_range(m.syntax().text_range()),
        };
        Some(id(self.data().macro_calls.alloc(res)))
    }

    fn lower_macro_rules(&mut self, m: &ast::MacroRules) -> Option<FileItemTreeId<MacroRules>> {
        let name = m.name().map(|it| it.as_name())?;
        let ast_id = self.source_ast_id_map.ast_id(m);

        let res = MacroRules { name, ast_id };
        Some(id(self.data().macro_rules.alloc(res)))
    }

    fn lower_macro_def(&mut self, m: &ast::MacroDef) -> Option<FileItemTreeId<Macro2>> {
        let name = m.name().map(|it| it.as_name())?;

        let ast_id = self.source_ast_id_map.ast_id(m);
        let visibility = self.lower_visibility(m);

        let res = Macro2 { name, ast_id, visibility };
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

    fn lower_generic_params(
        &mut self,
        has_implicit_self: HasImplicitSelf,
        node: &dyn ast::HasGenericParams,
    ) -> Interned<GenericParams> {
        let mut generics = GenericParamsCollector::default();

        if let HasImplicitSelf::Yes(bounds) = has_implicit_self {
            // Traits and trait aliases get the Self type as an implicit first type parameter.
            generics.type_or_consts.alloc(
                TypeParamData {
                    name: Some(name![Self]),
                    default: None,
                    provenance: TypeParamProvenance::TraitSelf,
                }
                .into(),
            );
            // add super traits as bounds on Self
            // i.e., `trait Foo: Bar` is equivalent to `trait Foo where Self: Bar`
            let self_param = TypeRef::Path(name![Self].into());
            generics.fill_bounds(&self.body_ctx, bounds, Either::Left(self_param));
        }

        let add_param_attrs = |item: Either<LocalTypeOrConstParamId, LocalLifetimeParamId>,
                               param| {
            let attrs = RawAttrs::new(self.db.upcast(), &param, self.body_ctx.span_map());
            // This is identical to the body of `Ctx::add_attrs()` but we can't call that here
            // because it requires `&mut self` and the call to `generics.fill()` below also
            // references `self`.
            match self.tree.attrs.entry(match item {
                Either::Right(id) => id.into(),
                Either::Left(id) => id.into(),
            }) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() = entry.get().merge(attrs);
                }
                Entry::Vacant(entry) => {
                    entry.insert(attrs);
                }
            }
        };
        generics.fill(&self.body_ctx, node, add_param_attrs);

        Interned::new(generics.finish())
    }

    fn lower_type_bounds(&mut self, node: &dyn ast::HasTypeBounds) -> Box<[Interned<TypeBound>]> {
        match node.type_bound_list() {
            Some(bound_list) => bound_list
                .bounds()
                .map(|it| Interned::new(TypeBound::from_ast(&self.body_ctx, it)))
                .collect(),
            None => Box::default(),
        }
    }

    fn lower_visibility(&mut self, item: &dyn ast::HasVisibility) -> RawVisibilityId {
        let vis = RawVisibility::from_ast(self.db, item.visibility(), &mut |range| {
            self.span_map().span_for_range(range).ctx
        });
        self.data().vis.alloc(vis)
    }

    fn lower_trait_ref(&mut self, trait_ref: &ast::Type) -> Option<Interned<TraitRef>> {
        let trait_ref = TraitRef::from_ast(&self.body_ctx, trait_ref.clone())?;
        Some(Interned::new(trait_ref))
    }

    fn lower_type_ref(&mut self, type_ref: &ast::Type) -> Interned<TypeRef> {
        let tyref = TypeRef::from_ast(&self.body_ctx, type_ref.clone());
        Interned::new(tyref)
    }

    fn lower_type_ref_opt(&mut self, type_ref: Option<ast::Type>) -> Interned<TypeRef> {
        match type_ref.map(|ty| self.lower_type_ref(&ty)) {
            Some(it) => it,
            None => Interned::new(TypeRef::Error),
        }
    }

    fn next_field_idx(&self) -> Idx<Field> {
        Idx::from_raw(RawIdx::from(
            self.tree.data.as_ref().map_or(0, |data| data.fields.len() as u32),
        ))
    }
    fn next_variant_idx(&self) -> Idx<Variant> {
        Idx::from_raw(RawIdx::from(
            self.tree.data.as_ref().map_or(0, |data| data.variants.len() as u32),
        ))
    }
    fn next_param_idx(&self) -> Idx<Param> {
        Idx::from_raw(RawIdx::from(
            self.tree.data.as_ref().map_or(0, |data| data.params.len() as u32),
        ))
    }
}

fn desugar_future_path(orig: TypeRef) -> Path {
    let path = path![core::future::Future];
    let mut generic_args: Vec<_> =
        std::iter::repeat(None).take(path.segments().len() - 1).collect();
    let binding = AssociatedTypeBinding {
        name: name![Output],
        args: None,
        type_ref: Some(orig),
        bounds: Box::default(),
    };
    generic_args.push(Some(Interned::new(GenericArgs {
        bindings: Box::new([binding]),
        ..GenericArgs::empty()
    })));

    Path::from_known_path(path, generic_args)
}

enum HasImplicitSelf {
    /// Inner list is a type bound list for the implicit `Self`.
    Yes(Option<ast::TypeBoundList>),
    No,
}

fn lower_abi(abi: ast::Abi) -> Interned<str> {
    match abi.abi_string() {
        Some(tok) => Interned::new_str(tok.text_without_quotes()),
        // `extern` default to be `extern "C"`.
        _ => Interned::new_str("C"),
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
