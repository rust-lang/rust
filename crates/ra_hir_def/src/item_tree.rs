//! A simplified AST that only contains items.

use hir_expand::{
    ast_id_map::{AstIdMap, FileAstId},
    hygiene::Hygiene,
    name::{name, AsName, Name},
    HirFileId,
};
use ra_arena::{Arena, Idx, RawId};
use ra_syntax::{ast, match_ast};

use crate::{
    attr::Attrs,
    db::DefDatabase,
    generics::GenericParams,
    path::{path, AssociatedTypeBinding, GenericArgs, ImportAlias, ModPath, Path},
    type_ref::{Mutability, TypeBound, TypeRef},
    visibility::RawVisibility,
};
use ast::{AstNode, ModuleItemOwner, NameOwner, StructKind, TypeAscriptionOwner};
use std::{
    ops::{Index, Range},
    sync::Arc,
};

#[derive(Debug, Default, Eq, PartialEq)]
pub struct ItemTree {
    top_level: Vec<ModItem>,
    imports: Arena<Import>,
    functions: Arena<Function>,
    structs: Arena<Struct>,
    fields: Arena<Field>,
    unions: Arena<Union>,
    enums: Arena<Enum>,
    variants: Arena<Variant>,
    consts: Arena<Const>,
    statics: Arena<Static>,
    traits: Arena<Trait>,
    impls: Arena<Impl>,
    type_aliases: Arena<TypeAlias>,
    mods: Arena<Mod>,
    macro_calls: Arena<MacroCall>,
    exprs: Arena<Expr>,
}

impl ItemTree {
    pub fn item_tree_query(db: &dyn DefDatabase, file_id: HirFileId) -> Arc<ItemTree> {
        let syntax = if let Some(node) = db.parse_or_expand(file_id) {
            node
        } else {
            return Default::default();
        };

        let (macro_storage, file_storage);
        let item_owner = match_ast! {
            match syntax {
                ast::MacroItems(items) => {
                    macro_storage = items;
                    &macro_storage as &dyn ModuleItemOwner
                },
                ast::SourceFile(file) => {
                    file_storage = file;
                    &file_storage
                },
                _ => return Default::default(),
            }
        };

        let map = db.ast_id_map(file_id);
        let ctx = Ctx {
            tree: ItemTree::default(),
            hygiene: Hygiene::new(db.upcast(), file_id),
            source_ast_id_map: map,
            body_ctx: crate::body::LowerCtx::new(db, file_id),
        };
        Arc::new(ctx.lower(item_owner))
    }

    /// Returns an iterator over all items located at the top level of the `HirFileId` this
    /// `ItemTree` was created from.
    pub fn top_level_items(&self) -> impl Iterator<Item = ModItem> + '_ {
        self.top_level.iter().copied()
    }
}

macro_rules! impl_index {
    ( $($fld:ident: $t:ty),+ $(,)? ) => {
        $(
            impl Index<Idx<$t>> for ItemTree {
                type Output = $t;

                fn index(&self, index: Idx<$t>) -> &Self::Output {
                    &self.$fld[index]
                }
            }
        )+
    };
}

impl_index!(
    imports: Import,
    functions: Function,
    structs: Struct,
    fields: Field,
    unions: Union,
    enums: Enum,
    variants: Variant,
    consts: Const,
    statics: Static,
    traits: Trait,
    impls: Impl,
    type_aliases: TypeAlias,
    mods: Mod,
    macro_calls: MacroCall,
    exprs: Expr,
);

#[derive(Debug, Eq, PartialEq)]
pub struct Import {
    pub path: ModPath,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibility,
    pub is_glob: bool,
    pub is_prelude: bool,
    pub is_extern_crate: bool,
    pub is_macro_use: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Function {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub has_self_param: bool,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    pub ast: FileAstId<ast::FnDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub fields: Fields,
    pub ast: FileAstId<ast::StructDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub fields: Fields,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub variants: Range<Idx<Variant>>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Const {
    /// const _: () = ();
    pub name: Option<Name>,
    pub visibility: RawVisibility,
    pub type_ref: TypeRef,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibility,
    pub type_ref: TypeRef,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub auto: bool,
    pub items: Vec<AssocItem>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Impl {
    pub generic_params: GenericParams,
    pub target_trait: Option<TypeRef>,
    pub target_type: TypeRef,
    pub is_negative: bool,
    pub items: Vec<AssocItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub type_ref: Option<TypeRef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Mod {
    pub name: Name,
    pub visibility: RawVisibility,
    pub items: Vec<ModItem>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct MacroCall {
    pub name: Option<Name>,
    pub path: ModPath,
    pub export: bool,
    pub builtin: bool,
    pub ast_id: FileAstId<ast::MacroCall>,
}

// NB: There's no `FileAstId` for `Expr`. The only case where this would be useful is for array
// lengths, but we don't do much with them yet.
#[derive(Debug, Eq, PartialEq)]
pub struct Expr;

macro_rules! impl_froms {
    ($e:ident { $($v:ident ($t:ty)),* $(,)? }) => {
        $(
            impl From<$t> for $e {
                fn from(it: $t) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ModItem {
    Import(Idx<Import>),
    Function(Idx<Function>),
    Struct(Idx<Struct>),
    Union(Idx<Union>),
    Enum(Idx<Enum>),
    Const(Idx<Const>),
    Static(Idx<Static>),
    Trait(Idx<Trait>),
    Impl(Idx<Impl>),
    TypeAlias(Idx<TypeAlias>),
    Mod(Idx<Mod>),
    MacroCall(Idx<MacroCall>),
}

impl_froms!(ModItem {
    Import(Idx<Import>),
    Function(Idx<Function>),
    Struct(Idx<Struct>),
    Union(Idx<Union>),
    Enum(Idx<Enum>),
    Const(Idx<Const>),
    Static(Idx<Static>),
    Trait(Idx<Trait>),
    Impl(Idx<Impl>),
    TypeAlias(Idx<TypeAlias>),
    Mod(Idx<Mod>),
    MacroCall(Idx<MacroCall>),
});

#[derive(Debug, Eq, PartialEq)]
pub enum AssocItem {
    Function(Idx<Function>),
    TypeAlias(Idx<TypeAlias>),
    Const(Idx<Const>),
    MacroCall(Idx<MacroCall>),
}

impl_froms!(AssocItem {
    Function(Idx<Function>),
    TypeAlias(Idx<TypeAlias>),
    Const(Idx<Const>),
    MacroCall(Idx<MacroCall>),
});

#[derive(Debug, Eq, PartialEq)]
pub struct Variant {
    pub name: Name,
    pub fields: Fields,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fields {
    Record(Range<Idx<Field>>),
    Tuple(Range<Idx<Field>>),
    Unit,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: Name,
    pub type_ref: TypeRef,
    pub visibility: RawVisibility,
}

struct Ctx {
    tree: ItemTree,
    hygiene: Hygiene,
    source_ast_id_map: Arc<AstIdMap>,
    body_ctx: crate::body::LowerCtx,
}

impl Ctx {
    fn lower(mut self, item_owner: &dyn ModuleItemOwner) -> ItemTree {
        self.tree.top_level = item_owner.items().flat_map(|item| self.lower_item(&item)).collect();
        self.tree
    }

    fn lower_item(&mut self, item: &ast::ModuleItem) -> Option<ModItem> {
        match item {
            ast::ModuleItem::StructDef(ast) => {
                self.lower_struct(ast).map(|data| self.tree.structs.alloc(data).into())
            }
            ast::ModuleItem::UnionDef(ast) => {
                self.lower_union(ast).map(|data| self.tree.unions.alloc(data).into())
            }
            ast::ModuleItem::EnumDef(ast) => {
                self.lower_enum(ast).map(|data| self.tree.enums.alloc(data).into())
            }
            ast::ModuleItem::FnDef(ast) => {
                self.lower_function(ast).map(|data| self.tree.functions.alloc(data).into())
            }
            ast::ModuleItem::TypeAliasDef(ast) => {
                self.lower_type_alias(ast).map(|data| self.tree.type_aliases.alloc(data).into())
            }
            ast::ModuleItem::StaticDef(ast) => {
                self.lower_static(ast).map(|data| self.tree.statics.alloc(data).into())
            }
            ast::ModuleItem::ConstDef(ast) => {
                let data = self.lower_const(ast);
                Some(self.tree.consts.alloc(data).into())
            }
            ast::ModuleItem::Module(ast) => {
                self.lower_module(ast).map(|data| self.tree.mods.alloc(data).into())
            }
            ast::ModuleItem::TraitDef(_) => todo!(),
            ast::ModuleItem::ImplDef(_) => todo!(),
            ast::ModuleItem::UseItem(_) => todo!(),
            ast::ModuleItem::ExternCrateItem(_) => todo!(),
            ast::ModuleItem::MacroCall(_) => todo!(),
            ast::ModuleItem::ExternBlock(_) => todo!(),
        }
    }

    fn lower_struct(&mut self, strukt: &ast::StructDef) -> Option<Struct> {
        let attrs = self.lower_attrs(strukt);
        let visibility = self.lower_visibility(strukt);
        let name = strukt.name()?.as_name();
        let generic_params = self.lower_generic_params(strukt);
        let fields = self.lower_fields(&strukt.kind());
        let ast = self.source_ast_id_map.ast_id(strukt);
        let res = Struct { name, attrs, visibility, generic_params, fields, ast };
        Some(res)
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

    fn lower_record_fields(&mut self, fields: &ast::RecordFieldDefList) -> Range<Idx<Field>> {
        let start = self.next_field_idx();
        for field in fields.fields() {
            if let Some(data) = self.lower_record_field(&field) {
                let idx = self.tree.fields.alloc(data);
            }
        }
        let end = self.next_field_idx();
        start..end
    }

    fn lower_record_field(&self, field: &ast::RecordFieldDef) -> Option<Field> {
        let name = field.name()?.as_name();
        let visibility = self.lower_visibility(field);
        let type_ref = self.lower_type_ref(&field.ascribed_type()?);
        let res = Field { name, type_ref, visibility };
        Some(res)
    }

    fn lower_tuple_fields(&mut self, fields: &ast::TupleFieldDefList) -> Range<Idx<Field>> {
        let start = self.next_field_idx();
        for (i, field) in fields.fields().enumerate() {
            if let Some(data) = self.lower_tuple_field(i, &field) {
                let idx = self.tree.fields.alloc(data);
            }
        }
        let end = self.next_field_idx();
        start..end
    }

    fn lower_tuple_field(&self, idx: usize, field: &ast::TupleFieldDef) -> Option<Field> {
        let name = Name::new_tuple_field(idx);
        let visibility = self.lower_visibility(field);
        let type_ref = self.lower_type_ref(&field.type_ref()?);
        let res = Field { name, type_ref, visibility };
        Some(res)
    }

    fn lower_union(&mut self, union: &ast::UnionDef) -> Option<Union> {
        let attrs = self.lower_attrs(union);
        let visibility = self.lower_visibility(union);
        let name = union.name()?.as_name();
        let generic_params = self.lower_generic_params(union);
        let fields = match union.record_field_def_list() {
            Some(record_field_def_list) => {
                self.lower_fields(&StructKind::Record(record_field_def_list))
            }
            None => Fields::Record(self.next_field_idx()..self.next_field_idx()),
        };
        let res = Union { name, attrs, visibility, generic_params, fields };
        Some(res)
    }

    fn lower_enum(&mut self, enum_: &ast::EnumDef) -> Option<Enum> {
        let attrs = self.lower_attrs(enum_);
        let visibility = self.lower_visibility(enum_);
        let name = enum_.name()?.as_name();
        let generic_params = self.lower_generic_params(enum_);
        let variants = match &enum_.variant_list() {
            Some(variant_list) => self.lower_variants(variant_list),
            None => self.next_variant_idx()..self.next_variant_idx(),
        };
        let res = Enum { name, attrs, visibility, generic_params, variants };
        Some(res)
    }

    fn lower_variants(&mut self, variants: &ast::EnumVariantList) -> Range<Idx<Variant>> {
        let start = self.next_variant_idx();
        for variant in variants.variants() {
            if let Some(data) = self.lower_variant(&variant) {
                let idx = self.tree.variants.alloc(data);
            }
        }
        let end = self.next_variant_idx();
        start..end
    }

    fn lower_variant(&mut self, variant: &ast::EnumVariant) -> Option<Variant> {
        let name = variant.name()?.as_name();
        let fields = self.lower_fields(&variant.kind());
        let res = Variant { name, fields };
        Some(res)
    }

    fn lower_function(&mut self, func: &ast::FnDef) -> Option<Function> {
        let attrs = self.lower_attrs(func);
        let visibility = self.lower_visibility(func);
        let name = func.name()?.as_name();
        let generic_params = self.lower_generic_params(func);

        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = func.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.ascribed_type() {
                    TypeRef::from_ast(&self.body_ctx, type_ref)
                } else {
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
                };
                params.push(self_type);
                has_self_param = true;
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(&self.body_ctx, param.ascribed_type());
                params.push(type_ref);
            }
        }
        let ret_type = match func.ret_type().and_then(|rt| rt.type_ref()) {
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

        let ast = self.source_ast_id_map.ast_id(func);
        let res = Function {
            name,
            attrs,
            visibility,
            generic_params,
            has_self_param,
            params,
            ret_type,
            ast,
        };
        Some(res)
    }

    fn lower_type_alias(&mut self, type_alias: &ast::TypeAliasDef) -> Option<TypeAlias> {
        let name = type_alias.name()?.as_name();
        let type_ref = type_alias.type_ref().map(|it| self.lower_type_ref(&it));
        let visibility = self.lower_visibility(type_alias);
        let generic_params = self.lower_generic_params(type_alias);
        let res = TypeAlias { name, visibility, generic_params, type_ref };
        Some(res)
    }

    fn lower_static(&mut self, static_: &ast::StaticDef) -> Option<Static> {
        let name = static_.name()?.as_name();
        let type_ref = self.lower_type_ref_opt(static_.ascribed_type());
        let visibility = self.lower_visibility(static_);
        let res = Static { name, visibility, type_ref };
        Some(res)
    }

    fn lower_const(&mut self, konst: &ast::ConstDef) -> Const {
        let name = konst.name().map(|it| it.as_name());
        let type_ref = self.lower_type_ref_opt(konst.ascribed_type());
        let visibility = self.lower_visibility(konst);
        Const { name, visibility, type_ref }
    }

    fn lower_module(&mut self, module: &ast::Module) -> Option<Mod> {
        let name = module.name()?.as_name();
        let visibility = self.lower_visibility(module);
        let items = module
            .item_list()
            .map(move |list| list.items().flat_map(move |item| self.lower_item(&item)).collect());
        Some(Mod { name, visibility, items: items.unwrap_or_default() })
    }

    fn lower_generic_params(&mut self, _item: &impl ast::TypeParamsOwner) -> GenericParams {
        None.unwrap()
    }

    fn lower_attrs(&self, item: &impl ast::AttrsOwner) -> Attrs {
        Attrs::new(item, &self.hygiene)
    }
    fn lower_visibility(&self, item: &impl ast::VisibilityOwner) -> RawVisibility {
        RawVisibility::from_ast_with_hygiene(item.visibility(), &self.hygiene)
    }
    fn lower_type_ref(&self, type_ref: &ast::TypeRef) -> TypeRef {
        TypeRef::from_ast(&self.body_ctx, type_ref.clone())
    }
    fn lower_type_ref_opt(&self, type_ref: Option<ast::TypeRef>) -> TypeRef {
        TypeRef::from_ast_opt(&self.body_ctx, type_ref)
    }

    fn next_field_idx(&self) -> Idx<Field> {
        Idx::from_raw(RawId::from(self.tree.fields.len() as u32))
    }
    fn next_variant_idx(&self) -> Idx<Variant> {
        Idx::from_raw(RawId::from(self.tree.variants.len() as u32))
    }
}

fn desugar_future_path(orig: TypeRef) -> Path {
    let path = path![std::future::Future];
    let mut generic_args: Vec<_> = std::iter::repeat(None).take(path.segments.len() - 1).collect();
    let mut last = GenericArgs::empty();
    let binding =
        AssociatedTypeBinding { name: name![Output], type_ref: Some(orig), bounds: Vec::new() };
    last.bindings.push(binding);
    generic_args.push(Some(Arc::new(last)));

    Path::from_known_path(path, generic_args)
}
