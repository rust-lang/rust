//! AST -> `ItemTree` lowering code.

use std::{collections::hash_map::Entry, mem, sync::Arc};

use arena::map::ArenaMap;
use hir_expand::{ast_id_map::AstIdMap, hygiene::Hygiene, HirFileId};
use smallvec::SmallVec;
use syntax::{
    ast::{self, ModuleItemOwner},
    SyntaxNode,
};

use crate::{
    attr::Attrs,
    generics::{GenericParams, TypeParamData, TypeParamProvenance},
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
    inner_items: Vec<ModItem>,
    forced_visibility: Option<RawVisibilityId>,
}

impl Ctx {
    pub(super) fn new(db: &dyn DefDatabase, hygiene: Hygiene, file: HirFileId) -> Self {
        Self {
            tree: ItemTree::empty(),
            hygiene,
            file,
            source_ast_id_map: db.ast_id_map(file),
            body_ctx: crate::body::LowerCtx::new(db, file),
            inner_items: Vec::new(),
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

    pub(super) fn lower_inner_items(mut self, within: &SyntaxNode) -> ItemTree {
        self.collect_inner_items(within);
        self.tree
    }

    fn data(&mut self) -> &mut ItemTreeData {
        self.tree.data_mut()
    }

    fn lower_mod_item(&mut self, item: &ast::Item, inner: bool) -> Option<ModItems> {
        assert!(inner || self.inner_items.is_empty());

        // Collect inner items for 1-to-1-lowered items.
        match item {
            ast::Item::Struct(_)
            | ast::Item::Union(_)
            | ast::Item::Enum(_)
            | ast::Item::Fn(_)
            | ast::Item::TypeAlias(_)
            | ast::Item::Const(_)
            | ast::Item::Static(_)
            | ast::Item::MacroCall(_) => {
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
            ast::Item::Module(_) | ast::Item::ExternCrate(_) | ast::Item::Use(_) => {}
        };

        let attrs = Attrs::new(item, &self.hygiene);
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

    fn add_attrs(&mut self, item: AttrOwner, attrs: Attrs) {
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
        let mut inner_items = mem::take(&mut self.tree.inner_items);
        inner_items.extend(container.descendants().skip(1).filter_map(ast::Item::cast).filter_map(
            |item| {
                let ast_id = self.source_ast_id_map.ast_id(&item);
                Some((ast_id, self.lower_mod_item(&item, true)?.0))
            },
        ));
        self.tree.inner_items = inner_items;
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
                self.add_attrs(idx.into(), Attrs::new(&field, &self.hygiene));
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
            self.add_attrs(idx.into(), Attrs::new(&field, &self.hygiene));
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
                self.add_attrs(idx.into(), Attrs::new(&variant, &self.hygiene));
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

        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = func.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = match self_param.ty() {
                    Some(type_ref) => TypeRef::from_ast(&self.body_ctx, type_ref),
                    None => {
                        let self_type = TypeRef::Path(name![Self].into());
                        match self_param.kind() {
                            ast::SelfParamKind::Owned => self_type,
                            ast::SelfParamKind::Ref => {
                                TypeRef::Reference(Box::new(self_type), Mutability::Shared)
                            }
                            ast::SelfParamKind::MutRef => {
                                TypeRef::Reference(Box::new(self_type), Mutability::Mut)
                            }
                        }
                    }
                };
                params.push(self_type);
                has_self_param = true;
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(&self.body_ctx, param.ty());
                params.push(type_ref);
            }
        }

        let mut is_varargs = false;
        if let Some(params) = func.param_list() {
            if let Some(last) = params.params().last() {
                is_varargs = last.dotdotdot_token().is_some();
            }
        }

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

        let ast_id = self.source_ast_id_map.ast_id(func);
        let mut res = Function {
            name,
            visibility,
            generic_params: GenericParamsId::EMPTY,
            has_self_param,
            is_unsafe: func.unsafe_token().is_some(),
            params: params.into_boxed_slice(),
            is_varargs,
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
        };
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
                        mark::hit!(name_res_works_for_broken_modules);
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
        let auto = trait_def.auto_token().is_some();
        let items = trait_def.assoc_item_list().map(|list| {
            self.with_inherited_visibility(visibility, |this| {
                list.assoc_items()
                    .filter_map(|item| {
                        let attrs = Attrs::new(&item, &this.hygiene);
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
            auto,
            items: items.unwrap_or_default(),
            ast_id,
        };
        Some(id(self.data().traits.alloc(res)))
    }

    fn lower_impl(&mut self, impl_def: &ast::Impl) -> Option<FileItemTreeId<Impl>> {
        let generic_params =
            self.lower_generic_params_and_inner_items(GenericsOwner::Impl, impl_def);
        let target_trait = impl_def.trait_().map(|tr| self.lower_type_ref(&tr));
        let target_type = self.lower_type_ref(&impl_def.self_ty()?);
        let is_negative = impl_def.excl_token().is_some();

        // We cannot use `assoc_items()` here as that does not include macro calls.
        let items = impl_def
            .assoc_item_list()
            .into_iter()
            .flat_map(|it| it.assoc_items())
            .filter_map(|item| {
                self.collect_inner_items(item.syntax());
                let assoc = self.lower_assoc_item(&item)?;
                let attrs = Attrs::new(&item, &self.hygiene);
                self.add_attrs(ModItem::from(assoc).into(), attrs);
                Some(assoc)
            })
            .collect();
        let ast_id = self.source_ast_id_map.ast_id(impl_def);
        let res = Impl { generic_params, target_trait, target_type, is_negative, items, ast_id };
        Some(id(self.data().impls.alloc(res)))
    }

    fn lower_use(&mut self, use_item: &ast::Use) -> Vec<FileItemTreeId<Import>> {
        // FIXME: cfg_attr
        let is_prelude = use_item.has_atom_attr("prelude_import");
        let visibility = self.lower_visibility(use_item);
        let ast_id = self.source_ast_id_map.ast_id(use_item);

        // Every use item can expand to many `Import`s.
        let mut imports = Vec::new();
        let tree = self.tree.data_mut();
        ModPath::expand_use_item(
            InFile::new(self.file, use_item.clone()),
            &self.hygiene,
            |path, _tree, is_glob, alias| {
                imports.push(id(tree.imports.alloc(Import {
                    path,
                    alias,
                    visibility,
                    is_glob,
                    is_prelude,
                    ast_id,
                })));
            },
        );

        imports
    }

    fn lower_extern_crate(
        &mut self,
        extern_crate: &ast::ExternCrate,
    ) -> Option<FileItemTreeId<ExternCrate>> {
        let path = ModPath::from_name_ref(&extern_crate.name_ref()?);
        let alias = extern_crate.rename().map(|a| {
            a.name().map(|it| it.as_name()).map_or(ImportAlias::Underscore, ImportAlias::Alias)
        });
        let visibility = self.lower_visibility(extern_crate);
        let ast_id = self.source_ast_id_map.ast_id(extern_crate);
        // FIXME: cfg_attr
        let is_macro_use = extern_crate.has_atom_attr("macro_use");

        let res = ExternCrate { path, alias, visibility, is_macro_use, ast_id };
        Some(id(self.data().extern_crates.alloc(res)))
    }

    fn lower_macro_call(&mut self, m: &ast::MacroCall) -> Option<FileItemTreeId<MacroCall>> {
        let name = m.name().map(|it| it.as_name());
        let attrs = Attrs::new(m, &self.hygiene);
        let path = ModPath::from_src(m.path()?, &self.hygiene)?;

        let ast_id = self.source_ast_id_map.ast_id(m);

        // FIXME: cfg_attr
        let export_attr = attrs.by_key("macro_export");

        let is_export = export_attr.exists();
        let is_local_inner = if is_export {
            export_attr.tt_values().map(|it| &it.token_trees).flatten().any(|it| match it {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    ident.text.contains("local_inner_macros")
                }
                _ => false,
            })
        } else {
            false
        };

        let is_builtin = attrs.by_key("rustc_builtin_macro").exists();
        let res = MacroCall { name, path, is_export, is_builtin, is_local_inner, ast_id };
        Some(id(self.data().macro_calls.alloc(res)))
    }

    fn lower_extern_block(&mut self, block: &ast::ExternBlock) -> Vec<ModItem> {
        block.extern_item_list().map_or(Vec::new(), |list| {
            list.extern_items()
                .filter_map(|item| {
                    self.collect_inner_items(item.syntax());
                    let attrs = Attrs::new(&item, &self.hygiene);
                    let id: ModItem = match item {
                        ast::ExternItem::Fn(ast) => {
                            let func = self.lower_function(&ast)?;
                            self.data().functions[func.index].is_unsafe = true;
                            func.into()
                        }
                        ast::ExternItem::Static(ast) => {
                            let statik = self.lower_static(&ast)?;
                            statik.into()
                        }
                        ast::ExternItem::TypeAlias(ty) => {
                            let id = self.lower_type_alias(&ty)?;
                            id.into()
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
        let mut sm = &mut ArenaMap::default();
        let mut generics = GenericParams::default();
        match owner {
            GenericsOwner::Function(func) => {
                generics.fill(&self.body_ctx, sm, node);
                // lower `impl Trait` in arguments
                for param in &*func.params {
                    generics.fill_implicit_impl_trait_args(param);
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
                sm.insert(self_param_id, Either::Left(trait_def.clone()));
                // add super traits as bounds on Self
                // i.e., trait Foo: Bar is equivalent to trait Foo where Self: Bar
                let self_param = TypeRef::Path(name![Self].into());
                generics.fill_bounds(&self.body_ctx, trait_def, self_param);

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

    fn lower_type_ref(&self, type_ref: &ast::Type) -> TypeRef {
        TypeRef::from_ast(&self.body_ctx, type_ref.clone())
    }
    fn lower_type_ref_opt(&self, type_ref: Option<ast::Type>) -> TypeRef {
        type_ref.map(|ty| self.lower_type_ref(&ty)).unwrap_or(TypeRef::Error)
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
        Idx::from_raw(RawId::from(
            self.tree.data.as_ref().map_or(0, |data| data.fields.len() as u32),
        ))
    }
    fn next_variant_idx(&self) -> Idx<Variant> {
        Idx::from_raw(RawId::from(
            self.tree.data.as_ref().map_or(0, |data| data.variants.len() as u32),
        ))
    }
}

fn desugar_future_path(orig: TypeRef) -> Path {
    let path = path![core::future::Future];
    let mut generic_args: Vec<_> = std::iter::repeat(None).take(path.segments.len() - 1).collect();
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
