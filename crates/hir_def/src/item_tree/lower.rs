//! AST -> `ItemTree` lowering code.

use std::{collections::hash_map::Entry, mem, sync::Arc};

use hir_expand::{ast_id_map::AstIdMap, hygiene::Hygiene, name::known, HirFileId};
use smallvec::SmallVec;
use syntax::{
    ast::{self, ModuleItemOwner},
    SyntaxNode, WalkEvent,
};

use crate::{
    generics::{GenericParams, TypeParamData, TypeParamProvenance},
    type_ref::{LifetimeRef, TraitRef},
};

use super::*;

fn id<N: ItemTreeNode>(index: Idx<N>) -> FileItemTreeId<N> {
    FileItemTreeId { index, _p: PhantomData }
}

struct ModItems(SmallVec<[ModItem; 1]>);

impl<T> From<T> for ModItems
where
    T: Into<ModItem>,
{
    fn from(t: T) -> Self {
        ModItems(SmallVec::from_buf([t.into(); 1]))
    }
}

pub(super) struct Ctx {
    tree: ItemTree,
    hygiene: Hygiene,
    file: HirFileId,
    source_ast_id_map: Arc<AstIdMap>,
    body_ctx: crate::body::LowerCtx,
    forced_visibility: Option<RawVisibilityId>,
}

impl Ctx {
    pub(super) fn new(db: &dyn DefDatabase, hygiene: Hygiene, file: HirFileId) -> Self {
        Self {
            tree: ItemTree::default(),
            hygiene,
            file,
            source_ast_id_map: db.ast_id_map(file),
            body_ctx: crate::body::LowerCtx::new(db, file),
            forced_visibility: None,
        }
    }

    pub(super) fn lower_module_items(mut self, item_owner: &dyn ModuleItemOwner) -> ItemTree {
        self.tree.top_level = item_owner
            .items()
            .flat_map(|item| self.lower_mod_item(&item, false))
            .flat_map(|items| items.0)
            .collect();
        self.tree
    }

    pub(super) fn lower_macro_stmts(mut self, stmts: ast::MacroStmts) -> ItemTree {
        self.tree.top_level = stmts
            .statements()
            .filter_map(|stmt| match stmt {
                ast::Stmt::Item(item) => Some(item),
                _ => None,
            })
            .flat_map(|item| self.lower_mod_item(&item, false))
            .flat_map(|items| items.0)
            .collect();

        // Non-items need to have their inner items collected.
        for stmt in stmts.statements() {
            match stmt {
                ast::Stmt::ExprStmt(_) | ast::Stmt::LetStmt(_) => {
                    self.collect_inner_items(stmt.syntax())
                }
                _ => {}
            }
        }
        if let Some(expr) = stmts.expr() {
            self.collect_inner_items(expr.syntax());
        }
        self.tree
    }

    pub(super) fn lower_inner_items(mut self, within: &SyntaxNode) -> ItemTree {
        self.collect_inner_items(within);
        self.tree
    }

    fn data(&mut self) -> &mut ItemTreeData {
        self.tree.data_mut()
    }

    fn lower_mod_item(&mut self, item: &ast::Item, inner: bool) -> Option<ModItems> {
        // Collect inner items for 1-to-1-lowered items.
        match item {
            ast::Item::Struct(_)
            | ast::Item::Union(_)
            | ast::Item::Enum(_)
            | ast::Item::Fn(_)
            | ast::Item::TypeAlias(_)
            | ast::Item::Const(_)
            | ast::Item::Static(_) => {
                // Skip this if we're already collecting inner items. We'll descend into all nodes
                // already.
                if !inner {
                    self.collect_inner_items(item.syntax());
                }
            }

            // These are handled in their respective `lower_X` method (since we can't just blindly
            // walk them).
            ast::Item::Trait(_) | ast::Item::Impl(_) | ast::Item::ExternBlock(_) => {}

            // These don't have inner items.
            ast::Item::Module(_)
            | ast::Item::ExternCrate(_)
            | ast::Item::Use(_)
            | ast::Item::MacroCall(_)
            | ast::Item::MacroRules(_)
            | ast::Item::MacroDef(_) => {}
        };

        let attrs = RawAttrs::new(item, &self.hygiene);
        let items = match item {
            ast::Item::Struct(ast) => self.lower_struct(ast).map(Into::into),
            ast::Item::Union(ast) => self.lower_union(ast).map(Into::into),
            ast::Item::Enum(ast) => self.lower_enum(ast).map(Into::into),
            ast::Item::Fn(ast) => self.lower_function(ast).map(Into::into),
            ast::Item::TypeAlias(ast) => self.lower_type_alias(ast).map(Into::into),
            ast::Item::Static(ast) => self.lower_static(ast).map(Into::into),
            ast::Item::Const(ast) => Some(self.lower_const(ast).into()),
            ast::Item::Module(ast) => self.lower_module(ast).map(Into::into),
            ast::Item::Trait(ast) => self.lower_trait(ast).map(Into::into),
            ast::Item::Impl(ast) => self.lower_impl(ast).map(Into::into),
            ast::Item::Use(ast) => Some(ModItems(
                self.lower_use(ast).into_iter().map(Into::into).collect::<SmallVec<_>>(),
            )),
            ast::Item::ExternCrate(ast) => self.lower_extern_crate(ast).map(Into::into),
            ast::Item::MacroCall(ast) => self.lower_macro_call(ast).map(Into::into),
            ast::Item::MacroRules(ast) => self.lower_macro_rules(ast).map(Into::into),
            ast::Item::MacroDef(ast) => self.lower_macro_def(ast).map(Into::into),
            ast::Item::ExternBlock(ast) => {
                Some(ModItems(self.lower_extern_block(ast).into_iter().collect::<SmallVec<_>>()))
            }
        };

        if !attrs.is_empty() {
            for item in items.iter().flat_map(|items| &items.0) {
                self.add_attrs((*item).into(), attrs.clone());
            }
        }

        items
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

    fn collect_inner_items(&mut self, container: &SyntaxNode) {
        let forced_vis = self.forced_visibility.take();

        let mut block_stack = Vec::new();

        // if container itself is block, add it to the stack
        if let Some(block) = ast::BlockExpr::cast(container.clone()) {
            block_stack.push(self.source_ast_id_map.ast_id(&block));
        }

        for event in container.preorder().skip(1) {
            match event {
                WalkEvent::Enter(node) => {
                    match_ast! {
                        match node {
                            ast::BlockExpr(block) => {
                                block_stack.push(self.source_ast_id_map.ast_id(&block));
                            },
                            ast::Item(item) => {
                                // FIXME: This triggers for macro calls in expression position
                                let mod_items = self.lower_mod_item(&item, true);
                                let current_block = block_stack.last();
                                if let (Some(mod_items), Some(block)) = (mod_items, current_block) {
                                    if !mod_items.0.is_empty() {
                                        self.data().inner_items.entry(*block).or_default().extend(mod_items.0.iter().copied());
                                    }
                                }
                            },
                            _ => {}
                        }
                    }
                }
                WalkEvent::Leave(node) => {
                    if ast::BlockExpr::cast(node).is_some() {
                        block_stack.pop();
                    }
                }
            }
        }

        self.forced_visibility = forced_vis;
    }

    fn lower_assoc_item(&mut self, item: &ast::AssocItem) -> Option<AssocItem> {
        match item {
            ast::AssocItem::Fn(ast) => self.lower_function(ast).map(Into::into),
            ast::AssocItem::TypeAlias(ast) => self.lower_type_alias(ast).map(Into::into),
            ast::AssocItem::Const(ast) => Some(self.lower_const(ast).into()),
            ast::AssocItem::MacroCall(ast) => self.lower_macro_call(ast).map(Into::into),
        }
    }

    fn lower_struct(&mut self, strukt: &ast::Struct) -> Option<FileItemTreeId<Struct>> {
        let visibility = self.lower_visibility(strukt);
        let name = strukt.name()?.as_name();
        let generic_params = self.lower_generic_params(GenericsOwner::Struct, strukt);
        let fields = self.lower_fields(&strukt.kind());
        let ast_id = self.source_ast_id_map.ast_id(strukt);
        let kind = match strukt.kind() {
            ast::StructKind::Record(_) => StructDefKind::Record,
            ast::StructKind::Tuple(_) => StructDefKind::Tuple,
            ast::StructKind::Unit => StructDefKind::Unit,
        };
        let res = Struct { name, visibility, generic_params, fields, ast_id, kind };
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

    fn lower_record_fields(&mut self, fields: &ast::RecordFieldList) -> IdRange<Field> {
        let start = self.next_field_idx();
        for field in fields.fields() {
            if let Some(data) = self.lower_record_field(&field) {
                let idx = self.data().fields.alloc(data);
                self.add_attrs(idx.into(), RawAttrs::new(&field, &self.hygiene));
            }
        }
        let end = self.next_field_idx();
        IdRange::new(start..end)
    }

    fn lower_record_field(&mut self, field: &ast::RecordField) -> Option<Field> {
        let name = field.name()?.as_name();
        let visibility = self.lower_visibility(field);
        let type_ref = self.lower_type_ref_opt(field.ty());
        let res = Field { name, type_ref, visibility };
        Some(res)
    }

    fn lower_tuple_fields(&mut self, fields: &ast::TupleFieldList) -> IdRange<Field> {
        let start = self.next_field_idx();
        for (i, field) in fields.fields().enumerate() {
            let data = self.lower_tuple_field(i, &field);
            let idx = self.data().fields.alloc(data);
            self.add_attrs(idx.into(), RawAttrs::new(&field, &self.hygiene));
        }
        let end = self.next_field_idx();
        IdRange::new(start..end)
    }

    fn lower_tuple_field(&mut self, idx: usize, field: &ast::TupleField) -> Field {
        let name = Name::new_tuple_field(idx);
        let visibility = self.lower_visibility(field);
        let type_ref = self.lower_type_ref_opt(field.ty());
        let res = Field { name, type_ref, visibility };
        res
    }

    fn lower_union(&mut self, union: &ast::Union) -> Option<FileItemTreeId<Union>> {
        let visibility = self.lower_visibility(union);
        let name = union.name()?.as_name();
        let generic_params = self.lower_generic_params(GenericsOwner::Union, union);
        let fields = match union.record_field_list() {
            Some(record_field_list) => self.lower_fields(&StructKind::Record(record_field_list)),
            None => Fields::Record(IdRange::new(self.next_field_idx()..self.next_field_idx())),
        };
        let ast_id = self.source_ast_id_map.ast_id(union);
        let res = Union { name, visibility, generic_params, fields, ast_id };
        Some(id(self.data().unions.alloc(res)))
    }

    fn lower_enum(&mut self, enum_: &ast::Enum) -> Option<FileItemTreeId<Enum>> {
        let visibility = self.lower_visibility(enum_);
        let name = enum_.name()?.as_name();
        let generic_params = self.lower_generic_params(GenericsOwner::Enum, enum_);
        let variants = match &enum_.variant_list() {
            Some(variant_list) => self.lower_variants(variant_list),
            None => IdRange::new(self.next_variant_idx()..self.next_variant_idx()),
        };
        let ast_id = self.source_ast_id_map.ast_id(enum_);
        let res = Enum { name, visibility, generic_params, variants, ast_id };
        Some(id(self.data().enums.alloc(res)))
    }

    fn lower_variants(&mut self, variants: &ast::VariantList) -> IdRange<Variant> {
        let start = self.next_variant_idx();
        for variant in variants.variants() {
            if let Some(data) = self.lower_variant(&variant) {
                let idx = self.data().variants.alloc(data);
                self.add_attrs(idx.into(), RawAttrs::new(&variant, &self.hygiene));
            }
        }
        let end = self.next_variant_idx();
        IdRange::new(start..end)
    }

    fn lower_variant(&mut self, variant: &ast::Variant) -> Option<Variant> {
        let name = variant.name()?.as_name();
        let fields = self.lower_fields(&variant.kind());
        let res = Variant { name, fields };
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
                let ty = self.data().type_refs.intern(self_type);
                let idx = self.data().params.alloc(Param::Normal(ty));
                self.add_attrs(idx.into(), RawAttrs::new(&self_param, &self.hygiene));
                has_self_param = true;
            }
            for param in param_list.params() {
                let idx = match param.dotdotdot_token() {
                    Some(_) => self.data().params.alloc(Param::Varargs),
                    None => {
                        let type_ref = TypeRef::from_ast_opt(&self.body_ctx, param.ty());
                        let ty = self.data().type_refs.intern(type_ref);
                        self.data().params.alloc(Param::Normal(ty))
                    }
                };
                self.add_attrs(idx.into(), RawAttrs::new(&param, &self.hygiene));
            }
        }
        let end_param = self.next_param_idx();
        let params = IdRange::new(start_param..end_param);

        let ret_type = match func.ret_type().and_then(|rt| rt.ty()) {
            Some(type_ref) => TypeRef::from_ast(&self.body_ctx, type_ref),
            _ => TypeRef::unit(),
        };

        let ret_type = if func.async_token().is_some() {
            let future_impl = desugar_future_path(ret_type);
            let ty_bound = TypeBound::Path(future_impl);
            TypeRef::ImplTrait(vec![ty_bound])
        } else {
            ret_type
        };

        let ret_type = self.data().type_refs.intern(ret_type);

        let has_body = func.body().is_some();

        let ast_id = self.source_ast_id_map.ast_id(func);
        let qualifier = FunctionQualifier {
            is_default: func.default_token().is_some(),
            is_const: func.const_token().is_some(),
            is_async: func.async_token().is_some(),
            is_unsafe: func.unsafe_token().is_some(),
            abi: func.abi().map(|abi| {
                // FIXME: Abi::abi() -> Option<SyntaxToken>?
                match abi.syntax().last_token() {
                    Some(tok) if tok.kind() == SyntaxKind::STRING => {
                        // FIXME: Better way to unescape?
                        tok.text().trim_matches('"').into()
                    }
                    _ => {
                        // `extern` default to be `extern "C"`.
                        "C".into()
                    }
                }
            }),
        };
        let mut res = Function {
            name,
            visibility,
            generic_params: GenericParamsId::EMPTY,
            has_self_param,
            has_body,
            qualifier,
            is_in_extern_block: false,
            params,
            ret_type,
            ast_id,
        };
        res.generic_params = self.lower_generic_params(GenericsOwner::Function(&res), func);

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
        let generic_params = self.lower_generic_params(GenericsOwner::TypeAlias, type_alias);
        let ast_id = self.source_ast_id_map.ast_id(type_alias);
        let res = TypeAlias {
            name,
            visibility,
            bounds: bounds.into_boxed_slice(),
            generic_params,
            type_ref,
            ast_id,
            is_extern: false,
        };
        Some(id(self.data().type_aliases.alloc(res)))
    }

    fn lower_static(&mut self, static_: &ast::Static) -> Option<FileItemTreeId<Static>> {
        let name = static_.name()?.as_name();
        let type_ref = self.lower_type_ref_opt(static_.ty());
        let visibility = self.lower_visibility(static_);
        let mutable = static_.mut_token().is_some();
        let ast_id = self.source_ast_id_map.ast_id(static_);
        let res = Static { name, visibility, mutable, type_ref, ast_id, is_extern: false };
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
            ModKind::Outline {}
        } else {
            ModKind::Inline {
                items: module
                    .item_list()
                    .map(|list| {
                        list.items()
                            .flat_map(|item| self.lower_mod_item(&item, false))
                            .flat_map(|items| items.0)
                            .collect()
                    })
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
        let generic_params =
            self.lower_generic_params_and_inner_items(GenericsOwner::Trait(trait_def), trait_def);
        let is_auto = trait_def.auto_token().is_some();
        let is_unsafe = trait_def.unsafe_token().is_some();
        let bounds = self.lower_type_bounds(trait_def);
        let items = trait_def.assoc_item_list().map(|list| {
            self.with_inherited_visibility(visibility, |this| {
                list.assoc_items()
                    .filter_map(|item| {
                        let attrs = RawAttrs::new(&item, &this.hygiene);
                        this.collect_inner_items(item.syntax());
                        this.lower_assoc_item(&item).map(|item| {
                            this.add_attrs(ModItem::from(item).into(), attrs);
                            item
                        })
                    })
                    .collect()
            })
        });
        let ast_id = self.source_ast_id_map.ast_id(trait_def);
        let res = Trait {
            name,
            visibility,
            generic_params,
            is_auto,
            is_unsafe,
            bounds: bounds.into(),
            items: items.unwrap_or_default(),
            ast_id,
        };
        Some(id(self.data().traits.alloc(res)))
    }

    fn lower_impl(&mut self, impl_def: &ast::Impl) -> Option<FileItemTreeId<Impl>> {
        let generic_params =
            self.lower_generic_params_and_inner_items(GenericsOwner::Impl, impl_def);
        // FIXME: If trait lowering fails, due to a non PathType for example, we treat this impl
        // as if it was an non-trait impl. Ideally we want to create a unique missing ref that only
        // equals itself.
        let target_trait = impl_def.trait_().and_then(|tr| self.lower_trait_ref(&tr));
        let self_ty = self.lower_type_ref(&impl_def.self_ty()?);
        let is_negative = impl_def.excl_token().is_some();

        // We cannot use `assoc_items()` here as that does not include macro calls.
        let items = impl_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|it| it.assoc_items())
            .filter_map(|item| {
                self.collect_inner_items(item.syntax());
                let assoc = self.lower_assoc_item(&item)?;
                let attrs = RawAttrs::new(&item, &self.hygiene);
                self.add_attrs(ModItem::from(assoc).into(), attrs);
                Some(assoc)
            })
            .collect();
        let ast_id = self.source_ast_id_map.ast_id(impl_def);
        let res = Impl { generic_params, target_trait, self_ty, is_negative, items, ast_id };
        Some(id(self.data().impls.alloc(res)))
    }

    fn lower_use(&mut self, use_item: &ast::Use) -> Vec<FileItemTreeId<Import>> {
        let visibility = self.lower_visibility(use_item);
        let ast_id = self.source_ast_id_map.ast_id(use_item);

        // Every use item can expand to many `Import`s.
        let mut imports = Vec::new();
        let tree = self.tree.data_mut();
        ModPath::expand_use_item(
            InFile::new(self.file, use_item.clone()),
            &self.hygiene,
            |path, _use_tree, is_glob, alias| {
                imports.push(id(tree.imports.alloc(Import {
                    path,
                    alias,
                    visibility,
                    is_glob,
                    ast_id,
                    index: imports.len(),
                })));
            },
        );

        imports
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
        let path = ModPath::from_src(m.path()?, &self.hygiene)?;
        let ast_id = self.source_ast_id_map.ast_id(m);
        let res = MacroCall { path, ast_id };
        Some(id(self.data().macro_calls.alloc(res)))
    }

    fn lower_macro_rules(&mut self, m: &ast::MacroRules) -> Option<FileItemTreeId<MacroRules>> {
        let name = m.name().map(|it| it.as_name())?;
        let ast_id = self.source_ast_id_map.ast_id(m);

        let res = MacroRules { name, ast_id };
        Some(id(self.data().macro_rules.alloc(res)))
    }

    fn lower_macro_def(&mut self, m: &ast::MacroDef) -> Option<FileItemTreeId<MacroDef>> {
        let name = m.name().map(|it| it.as_name())?;

        let ast_id = self.source_ast_id_map.ast_id(m);
        let visibility = self.lower_visibility(m);

        let res = MacroDef { name, ast_id, visibility };
        Some(id(self.data().macro_defs.alloc(res)))
    }

    fn lower_extern_block(&mut self, block: &ast::ExternBlock) -> Vec<ModItem> {
        block.extern_item_list().map_or(Vec::new(), |list| {
            list.extern_items()
                .filter_map(|item| {
                    self.collect_inner_items(item.syntax());
                    let attrs = RawAttrs::new(&item, &self.hygiene);
                    let id: ModItem = match item {
                        ast::ExternItem::Fn(ast) => {
                            let func_id = self.lower_function(&ast)?;
                            let func = &mut self.data().functions[func_id.index];
                            func.qualifier.is_unsafe = is_intrinsic_fn_unsafe(&func.name);
                            func.is_in_extern_block = true;
                            func_id.into()
                        }
                        ast::ExternItem::Static(ast) => {
                            let statik = self.lower_static(&ast)?;
                            self.data().statics[statik.index].is_extern = true;
                            statik.into()
                        }
                        ast::ExternItem::TypeAlias(ty) => {
                            let foreign_ty = self.lower_type_alias(&ty)?;
                            self.data().type_aliases[foreign_ty.index].is_extern = true;
                            foreign_ty.into()
                        }
                        ast::ExternItem::MacroCall(_) => return None,
                    };
                    self.add_attrs(id.into(), attrs);
                    Some(id)
                })
                .collect()
        })
    }

    /// Lowers generics defined on `node` and collects inner items defined within.
    fn lower_generic_params_and_inner_items(
        &mut self,
        owner: GenericsOwner<'_>,
        node: &impl ast::GenericParamsOwner,
    ) -> GenericParamsId {
        // Generics are part of item headers and may contain inner items we need to collect.
        if let Some(params) = node.generic_param_list() {
            self.collect_inner_items(params.syntax());
        }
        if let Some(clause) = node.where_clause() {
            self.collect_inner_items(clause.syntax());
        }

        self.lower_generic_params(owner, node)
    }

    fn lower_generic_params(
        &mut self,
        owner: GenericsOwner<'_>,
        node: &impl ast::GenericParamsOwner,
    ) -> GenericParamsId {
        let mut sm = &mut Default::default();
        let mut generics = GenericParams::default();
        match owner {
            GenericsOwner::Function(func) => {
                generics.fill(&self.body_ctx, sm, node);
                // lower `impl Trait` in arguments
                for id in func.params.clone() {
                    if let Param::Normal(ty) = self.data().params[id] {
                        let ty = self.data().type_refs.lookup(ty);
                        generics.fill_implicit_impl_trait_args(ty);
                    }
                }
            }
            GenericsOwner::Struct
            | GenericsOwner::Enum
            | GenericsOwner::Union
            | GenericsOwner::TypeAlias => {
                generics.fill(&self.body_ctx, sm, node);
            }
            GenericsOwner::Trait(trait_def) => {
                // traits get the Self type as an implicit first type parameter
                let self_param_id = generics.types.alloc(TypeParamData {
                    name: Some(name![Self]),
                    default: None,
                    provenance: TypeParamProvenance::TraitSelf,
                });
                sm.type_params.insert(self_param_id, Either::Left(trait_def.clone()));
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name![Self].into());
                generics.fill_bounds(&self.body_ctx, trait_def, Either::Left(self_param));
                generics.fill(&self.body_ctx, &mut sm, node);
            }
            GenericsOwner::Impl => {
                // Note that we don't add `Self` here: in `impl`s, `Self` is not a
                // type-parameter, but rather is a type-alias for impl's target
                // type, so this is handled by the resolver.
                generics.fill(&self.body_ctx, &mut sm, node);
            }
        }

        self.data().generics.alloc(generics)
    }

    fn lower_type_bounds(&mut self, node: &impl ast::TypeBoundsOwner) -> Vec<TypeBound> {
        match node.type_bound_list() {
            Some(bound_list) => {
                bound_list.bounds().map(|it| TypeBound::from_ast(&self.body_ctx, it)).collect()
            }
            None => Vec::new(),
        }
    }

    fn lower_visibility(&mut self, item: &impl ast::VisibilityOwner) -> RawVisibilityId {
        let vis = match self.forced_visibility {
            Some(vis) => return vis,
            None => RawVisibility::from_ast_with_hygiene(item.visibility(), &self.hygiene),
        };

        self.data().vis.alloc(vis)
    }

    fn lower_trait_ref(&mut self, trait_ref: &ast::Type) -> Option<Idx<TraitRef>> {
        let trait_ref = TraitRef::from_ast(&self.body_ctx, trait_ref.clone())?;
        Some(self.data().trait_refs.intern(trait_ref))
    }

    fn lower_type_ref(&mut self, type_ref: &ast::Type) -> Idx<TypeRef> {
        let tyref = TypeRef::from_ast(&self.body_ctx, type_ref.clone());
        self.data().type_refs.intern(tyref)
    }

    fn lower_type_ref_opt(&mut self, type_ref: Option<ast::Type>) -> Idx<TypeRef> {
        match type_ref.map(|ty| self.lower_type_ref(&ty)) {
            Some(it) => it,
            None => self.data().type_refs.intern(TypeRef::Error),
        }
    }

    /// Forces the visibility `vis` to be used for all items lowered during execution of `f`.
    fn with_inherited_visibility<R>(
        &mut self,
        vis: RawVisibilityId,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old = mem::replace(&mut self.forced_visibility, Some(vis));
        let res = f(self);
        self.forced_visibility = old;
        res
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
    let mut last = GenericArgs::empty();
    let binding =
        AssociatedTypeBinding { name: name![Output], type_ref: Some(orig), bounds: Vec::new() };
    last.bindings.push(binding);
    generic_args.push(Some(Arc::new(last)));

    Path::from_known_path(path, generic_args)
}

enum GenericsOwner<'a> {
    /// We need access to the partially-lowered `Function` for lowering `impl Trait` in argument
    /// position.
    Function(&'a Function),
    Struct,
    Enum,
    Union,
    /// The `TraitDef` is needed to fill the source map for the implicit `Self` parameter.
    Trait(&'a ast::Trait),
    TypeAlias,
    Impl,
}

/// Returns `true` if the given intrinsic is unsafe to call, or false otherwise.
fn is_intrinsic_fn_unsafe(name: &Name) -> bool {
    // Should be kept in sync with https://github.com/rust-lang/rust/blob/c6e4db620a7d2f569f11dcab627430921ea8aacf/compiler/rustc_typeck/src/check/intrinsic.rs#L68
    ![
        known::abort,
        known::min_align_of,
        known::needs_drop,
        known::caller_location,
        known::size_of_val,
        known::min_align_of_val,
        known::add_with_overflow,
        known::sub_with_overflow,
        known::mul_with_overflow,
        known::wrapping_add,
        known::wrapping_sub,
        known::wrapping_mul,
        known::saturating_add,
        known::saturating_sub,
        known::rotate_left,
        known::rotate_right,
        known::ctpop,
        known::ctlz,
        known::cttz,
        known::bswap,
        known::bitreverse,
        known::discriminant_value,
        known::type_id,
        known::likely,
        known::unlikely,
        known::ptr_guaranteed_eq,
        known::ptr_guaranteed_ne,
        known::minnumf32,
        known::minnumf64,
        known::maxnumf32,
        known::rustc_peek,
        known::maxnumf64,
        known::type_name,
        known::variant_count,
    ]
    .contains(&name)
}
