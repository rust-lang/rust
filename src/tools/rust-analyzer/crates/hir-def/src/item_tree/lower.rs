//! AST -> `ItemTree` lowering code.

use std::{cell::OnceCell, collections::hash_map::Entry};

use hir_expand::{
    HirFileId,
    mod_path::PathKind,
    name::AsName,
    span_map::{SpanMap, SpanMapRef},
};
use intern::{Symbol, sym};
use la_arena::Arena;
use span::{AstIdMap, SyntaxContext};
use syntax::{
    AstNode,
    ast::{self, HasModuleItem, HasName, IsString},
};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    item_tree::{
        AssocItem, AttrOwner, Const, Enum, ExternBlock, ExternCrate, Field, FieldParent,
        FieldsShape, FileItemTreeId, Function, Idx, Impl, ImportAlias, Interned, ItemTree,
        ItemTreeData, Macro2, MacroCall, MacroRules, Mod, ModItem, ModKind, ModPath, Name, Range,
        RawAttrs, RawIdx, RawVisibility, RawVisibilityId, Static, Struct, StructKind, Trait,
        TraitAlias, TypeAlias, Union, Use, UseTree, UseTreeKind, Variant, VisibilityExplicitness,
    },
};

fn id<N>(index: Idx<N>) -> FileItemTreeId<N> {
    FileItemTreeId(index)
}

pub(super) struct Ctx<'a> {
    db: &'a dyn DefDatabase,
    tree: ItemTree,
    source_ast_id_map: Arc<AstIdMap>,
    span_map: OnceCell<SpanMap>,
    file: HirFileId,
}

impl<'a> Ctx<'a> {
    pub(super) fn new(db: &'a dyn DefDatabase, file: HirFileId) -> Self {
        Self {
            db,
            tree: ItemTree::default(),
            source_ast_id_map: db.ast_id_map(file),
            file,
            span_map: OnceCell::new(),
        }
    }

    pub(super) fn span_map(&self) -> SpanMapRef<'_> {
        self.span_map.get_or_init(|| self.db.span_map(self.file)).as_ref()
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
        self.tree.attrs.insert(AttrOwner::TopLevel, RawAttrs::new(self.db, block, self.span_map()));
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
            ast::Item::Impl(ast) => self.lower_impl(ast).into(),
            ast::Item::Use(ast) => self.lower_use(ast)?.into(),
            ast::Item::ExternCrate(ast) => self.lower_extern_crate(ast)?.into(),
            ast::Item::MacroCall(ast) => self.lower_macro_call(ast)?.into(),
            ast::Item::MacroRules(ast) => self.lower_macro_rules(ast)?.into(),
            ast::Item::MacroDef(ast) => self.lower_macro_def(ast)?.into(),
            ast::Item::ExternBlock(ast) => self.lower_extern_block(ast).into(),
        };
        let attrs = RawAttrs::new(self.db, item, self.span_map());
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
        let attrs = RawAttrs::new(self.db, item_node, self.span_map());
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
        let (fields, kind, attrs) = self.lower_fields(&strukt.kind());
        let res = Struct { name, visibility, fields, shape: kind, ast_id };
        let id = id(self.data().structs.alloc(res));

        for (idx, attr) in attrs {
            self.add_attrs(
                AttrOwner::Field(
                    FieldParent::Struct(id),
                    Idx::from_raw(RawIdx::from_u32(idx as u32)),
                ),
                attr,
            );
        }
        Some(id)
    }

    fn lower_fields(
        &mut self,
        strukt_kind: &ast::StructKind,
    ) -> (Box<[Field]>, FieldsShape, Vec<(usize, RawAttrs)>) {
        match strukt_kind {
            ast::StructKind::Record(it) => {
                let mut fields = vec![];
                let mut attrs = vec![];

                for (i, field) in it.fields().enumerate() {
                    let data = self.lower_record_field(&field);
                    fields.push(data);
                    let attr = RawAttrs::new(self.db, &field, self.span_map());
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
                    let data = self.lower_tuple_field(i, &field);
                    fields.push(data);
                    let attr = RawAttrs::new(self.db, &field, self.span_map());
                    if !attr.is_empty() {
                        attrs.push((i, attr))
                    }
                }
                (fields.into(), FieldsShape::Tuple, attrs)
            }
            ast::StructKind::Unit => (Box::default(), FieldsShape::Unit, Vec::default()),
        }
    }

    fn lower_record_field(&mut self, field: &ast::RecordField) -> Field {
        let name = match field.name() {
            Some(name) => name.as_name(),
            None => Name::missing(),
        };
        let visibility = self.lower_visibility(field);

        Field { name, visibility, is_unsafe: field.unsafe_token().is_some() }
    }

    fn lower_tuple_field(&mut self, idx: usize, field: &ast::TupleField) -> Field {
        let name = Name::new_tuple_field(idx);
        let visibility = self.lower_visibility(field);
        Field { name, visibility, is_unsafe: false }
    }

    fn lower_union(&mut self, union: &ast::Union) -> Option<FileItemTreeId<Union>> {
        let visibility = self.lower_visibility(union);
        let name = union.name()?.as_name();
        let ast_id = self.source_ast_id_map.ast_id(union);
        let (fields, _, attrs) = match union.record_field_list() {
            Some(record_field_list) => self.lower_fields(&StructKind::Record(record_field_list)),
            None => (Box::default(), FieldsShape::Record, Vec::default()),
        };
        let res = Union { name, visibility, fields, ast_id };
        let id = id(self.data().unions.alloc(res));
        for (idx, attr) in attrs {
            self.add_attrs(
                AttrOwner::Field(
                    FieldParent::Union(id),
                    Idx::from_raw(RawIdx::from_u32(idx as u32)),
                ),
                attr,
            );
        }
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
        let res = Enum { name, visibility, variants, ast_id };
        let id = id(self.data().enums.alloc(res));
        Some(id)
    }

    fn lower_variants(&mut self, variants: &ast::VariantList) -> Range<FileItemTreeId<Variant>> {
        let start = self.next_variant_idx();
        for variant in variants.variants() {
            let idx = self.lower_variant(&variant);
            self.add_attrs(id(idx).into(), RawAttrs::new(self.db, &variant, self.span_map()));
        }
        let end = self.next_variant_idx();
        FileItemTreeId(start)..FileItemTreeId(end)
    }

    fn lower_variant(&mut self, variant: &ast::Variant) -> Idx<Variant> {
        let name = match variant.name() {
            Some(name) => name.as_name(),
            None => Name::missing(),
        };
        let (fields, kind, attrs) = self.lower_fields(&variant.kind());
        let ast_id = self.source_ast_id_map.ast_id(variant);
        let res = Variant { name, fields, shape: kind, ast_id };
        let id = self.data().variants.alloc(res);
        for (idx, attr) in attrs {
            self.add_attrs(
                AttrOwner::Field(
                    FieldParent::EnumVariant(FileItemTreeId(id)),
                    Idx::from_raw(RawIdx::from_u32(idx as u32)),
                ),
                attr,
            );
        }
        id
    }

    fn lower_function(&mut self, func: &ast::Fn) -> Option<FileItemTreeId<Function>> {
        let visibility = self.lower_visibility(func);
        let name = func.name()?.as_name();

        let ast_id = self.source_ast_id_map.ast_id(func);

        let res = Function { name, visibility, ast_id };

        let id = id(self.data().functions.alloc(res));
        Some(id)
    }

    fn lower_type_alias(
        &mut self,
        type_alias: &ast::TypeAlias,
    ) -> Option<FileItemTreeId<TypeAlias>> {
        let name = type_alias.name()?.as_name();
        let visibility = self.lower_visibility(type_alias);
        let ast_id = self.source_ast_id_map.ast_id(type_alias);
        let res = TypeAlias { name, visibility, ast_id };
        let id = id(self.data().type_aliases.alloc(res));
        Some(id)
    }

    fn lower_static(&mut self, static_: &ast::Static) -> Option<FileItemTreeId<Static>> {
        let name = static_.name()?.as_name();
        let visibility = self.lower_visibility(static_);
        let ast_id = self.source_ast_id_map.ast_id(static_);
        let res = Static { name, visibility, ast_id };
        Some(id(self.data().statics.alloc(res)))
    }

    fn lower_const(&mut self, konst: &ast::Const) -> FileItemTreeId<Const> {
        let name = konst.name().map(|it| it.as_name());
        let visibility = self.lower_visibility(konst);
        let ast_id = self.source_ast_id_map.ast_id(konst);
        let res = Const { name, visibility, ast_id };
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

        let items = trait_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|list| list.assoc_items())
            .filter_map(|item_node| self.lower_assoc_item(&item_node))
            .collect();

        let def = Trait { name, visibility, items, ast_id };
        let id = id(self.data().traits.alloc(def));
        Some(id)
    }

    fn lower_trait_alias(
        &mut self,
        trait_alias_def: &ast::TraitAlias,
    ) -> Option<FileItemTreeId<TraitAlias>> {
        let name = trait_alias_def.name()?.as_name();
        let visibility = self.lower_visibility(trait_alias_def);
        let ast_id = self.source_ast_id_map.ast_id(trait_alias_def);

        let alias = TraitAlias { name, visibility, ast_id };
        let id = id(self.data().trait_aliases.alloc(alias));
        Some(id)
    }

    fn lower_impl(&mut self, impl_def: &ast::Impl) -> FileItemTreeId<Impl> {
        let ast_id = self.source_ast_id_map.ast_id(impl_def);
        // We cannot use `assoc_items()` here as that does not include macro calls.
        let items = impl_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|it| it.assoc_items())
            .filter_map(|item| self.lower_assoc_item(&item))
            .collect();
        // Note that trait impls don't get implicit `Self` unlike traits, because here they are a
        // type alias rather than a type parameter, so this is handled by the resolver.
        let res = Impl { items, ast_id };
        id(self.data().impls.alloc(res))
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
        let path = Interned::new(ModPath::from_src(self.db, path, &mut |range| {
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
                    let attrs = RawAttrs::new(self.db, &item, self.span_map());
                    self.add_attrs(mod_item.into(), attrs);
                    Some(mod_item)
                })
                .collect()
        });

        let res = ExternBlock { abi, ast_id, children };
        id(self.data().extern_blocks.alloc(res))
    }

    fn lower_visibility(&mut self, item: &dyn ast::HasVisibility) -> RawVisibilityId {
        let vis = visibility_from_ast(self.db, item.visibility(), &mut |range| {
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

fn lower_abi(abi: ast::Abi) -> Symbol {
    match abi.abi_string() {
        Some(tok) => Symbol::intern(tok.text_without_quotes()),
        // `extern` default to be `extern "C"`.
        _ => sym::C,
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
        span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContext,
    ) -> Option<UseTree> {
        if let Some(use_tree_list) = tree.use_tree_list() {
            let prefix = match tree.path() {
                // E.g. use something::{{{inner}}};
                None => None,
                // E.g. `use something::{inner}` (prefix is `None`, path is `something`)
                // or `use something::{path::{inner::{innerer}}}` (prefix is `something::path`, path is `inner`)
                Some(path) => {
                    match ModPath::from_src(self.db, path, span_for_range) {
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
                Some(path) => Some(ModPath::from_src(self.db, path, span_for_range)?),
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
    span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContext,
) -> Option<(UseTree, Arena<ast::UseTree>)> {
    let mut lowering = UseTreeLowering { db, mapping: Arena::new() };
    let tree = lowering.lower_use_tree(tree, span_for_range)?;
    Some((tree, lowering.mapping))
}

fn private_vis() -> RawVisibility {
    RawVisibility::Module(
        Interned::new(ModPath::from_kind(PathKind::SELF)),
        VisibilityExplicitness::Implicit,
    )
}

fn visibility_from_ast(
    db: &dyn DefDatabase,
    node: Option<ast::Visibility>,
    span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContext,
) -> RawVisibility {
    let Some(node) = node else { return private_vis() };
    let path = match node.kind() {
        ast::VisibilityKind::In(path) => {
            let path = ModPath::from_src(db, path, span_for_range);
            match path {
                None => return private_vis(),
                Some(path) => path,
            }
        }
        ast::VisibilityKind::PubCrate => ModPath::from_kind(PathKind::Crate),
        ast::VisibilityKind::PubSuper => ModPath::from_kind(PathKind::Super(1)),
        ast::VisibilityKind::PubSelf => ModPath::from_kind(PathKind::SELF),
        ast::VisibilityKind::Pub => return RawVisibility::Public,
    };
    RawVisibility::Module(Interned::new(path), VisibilityExplicitness::Explicit)
}
