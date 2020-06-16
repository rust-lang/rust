//! A simplified AST that only contains items.

mod lower;

use std::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Index, Range},
    sync::Arc,
};

use ast::{AstNode, AttrsOwner, ModuleItemOwner, NameOwner, StructKind, TypeAscriptionOwner};
use either::Either;
use hir_expand::{
    ast_id_map::FileAstId,
    hygiene::Hygiene,
    name::{name, AsName, Name},
    HirFileId, InFile,
};
use ra_arena::{Arena, Idx, RawId};
use ra_syntax::{ast, match_ast};
use rustc_hash::FxHashMap;
use test_utils::mark;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    generics,
    path::{path, AssociatedTypeBinding, GenericArgs, ImportAlias, ModPath, Path},
    type_ref::{Mutability, TypeBound, TypeRef},
    visibility::RawVisibility,
};

/// The item tree of a source file.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ItemTree {
    top_level: Vec<ModItem>,
    top_attrs: Attrs,
    attrs: FxHashMap<ModItem, Attrs>,
    empty_attrs: Attrs,

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
        let _p = ra_prof::profile("item_tree_query");
        let syntax = if let Some(node) = db.parse_or_expand(file_id) {
            node
        } else {
            return Default::default();
        };

        let hygiene = Hygiene::new(db.upcast(), file_id);
        let mut top_attrs = None;
        let (macro_storage, file_storage);
        let item_owner = match_ast! {
            match syntax {
                ast::MacroItems(items) => {
                    macro_storage = items;
                    &macro_storage as &dyn ModuleItemOwner
                },
                ast::SourceFile(file) => {
                    top_attrs = Some(Attrs::new(&file, &hygiene));
                    file_storage = file;
                    &file_storage
                },
                _ => return Default::default(),
            }
        };

        let map = db.ast_id_map(file_id);
        let mut ctx = lower::Ctx {
            tree: ItemTree::default(),
            hygiene,
            file: file_id,
            source_ast_id_map: map,
            body_ctx: crate::body::LowerCtx::new(db, file_id),
        };
        ctx.tree.top_attrs = top_attrs.unwrap_or_default();
        Arc::new(ctx.lower(item_owner))
    }

    /// Returns an iterator over all items located at the top level of the `HirFileId` this
    /// `ItemTree` was created from.
    pub fn top_level_items(&self) -> &[ModItem] {
        &self.top_level
    }

    /// Returns the inner attributes of the source file.
    pub fn top_level_attrs(&self) -> &Attrs {
        &self.top_attrs
    }

    pub fn attrs(&self, of: ModItem) -> &Attrs {
        self.attrs.get(&of).unwrap_or(&self.empty_attrs)
    }
}

pub trait ItemTreeNode: Sized {
    fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self;
}

pub struct FileItemTreeId<N: ItemTreeNode> {
    index: Idx<N>,
    _p: PhantomData<N>,
}

impl<N: ItemTreeNode> Clone for FileItemTreeId<N> {
    fn clone(&self) -> Self {
        Self { index: self.index, _p: PhantomData }
    }
}
impl<N: ItemTreeNode> Copy for FileItemTreeId<N> {}

impl<N: ItemTreeNode> PartialEq for FileItemTreeId<N> {
    fn eq(&self, other: &FileItemTreeId<N>) -> bool {
        self.index == other.index
    }
}
impl<N: ItemTreeNode> Eq for FileItemTreeId<N> {}

impl<N: ItemTreeNode> Hash for FileItemTreeId<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state)
    }
}

impl<N: ItemTreeNode> fmt::Debug for FileItemTreeId<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.index.fmt(f)
    }
}

pub type ItemTreeId<N> = InFile<FileItemTreeId<N>>;

macro_rules! nodes {
    ( $($node:ident in $fld:ident),+ $(,)? ) => { $(
        impl ItemTreeNode for $node {
            fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self {
                &tree.$fld[index]
            }
        }
    )+ };
}

nodes!(
    Import in imports,
    Function in functions,
    Struct in structs,
    Union in unions,
    Enum in enums,
    Const in consts,
    Static in statics,
    Trait in traits,
    Impl in impls,
    TypeAlias in type_aliases,
    Mod in mods,
    MacroCall in macro_calls,
);

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

impl<N: ItemTreeNode> Index<FileItemTreeId<N>> for ItemTree {
    type Output = N;
    fn index(&self, id: FileItemTreeId<N>) -> &N {
        N::lookup(self, id.index)
    }
}

/// A desugared `extern crate` or `use` import.
#[derive(Debug, Clone, Eq, PartialEq)]
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
    pub generic_params: generics::GenericParams,
    pub has_self_param: bool,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    pub ast_id: FileAstId<ast::FnDef>,
    // FIXME inner items
}

#[derive(Debug, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: generics::GenericParams,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::StructDef>,
    pub kind: StructDefKind,
}

#[derive(Debug, Eq, PartialEq)]
pub enum StructDefKind {
    /// `struct S { ... }` - type namespace only.
    Record,
    /// `struct S(...);`
    Tuple,
    /// `struct S;`
    Unit,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: generics::GenericParams,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::UnionDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub generic_params: generics::GenericParams,
    pub variants: Range<Idx<Variant>>,
    pub ast_id: FileAstId<ast::EnumDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Const {
    /// const _: () = ();
    pub name: Option<Name>,
    pub visibility: RawVisibility,
    pub type_ref: TypeRef,
    pub ast_id: FileAstId<ast::ConstDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibility,
    pub type_ref: TypeRef,
    pub ast_id: FileAstId<ast::StaticDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: generics::GenericParams,
    pub auto: bool,
    pub items: Vec<AssocItem>,
    pub ast_id: FileAstId<ast::TraitDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Impl {
    pub generic_params: generics::GenericParams,
    pub target_trait: Option<TypeRef>,
    pub target_type: TypeRef,
    pub is_negative: bool,
    pub items: Vec<AssocItem>,
    pub ast_id: FileAstId<ast::ImplDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: generics::GenericParams,
    pub type_ref: Option<TypeRef>,
    pub ast_id: FileAstId<ast::TypeAliasDef>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Mod {
    pub name: Name,
    pub visibility: RawVisibility,
    pub kind: ModKind,
    pub ast_id: FileAstId<ast::Module>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum ModKind {
    /// `mod m { ... }`
    Inline { items: Vec<ModItem> },

    /// `mod m;`
    Outline {},
}

#[derive(Debug, Eq, PartialEq)]
pub struct MacroCall {
    /// For `macro_rules!` declarations, this is the name of the declared macro.
    pub name: Option<Name>,
    /// Path to the called macro.
    pub path: ModPath,
    /// Has `#[macro_export]`.
    pub is_export: bool,
    /// Has `#[macro_export(local_inner_macros)]`.
    pub is_local_inner: bool,
    /// Has `#[rustc_builtin_macro]`.
    pub is_builtin: bool,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ModItem {
    Import(FileItemTreeId<Import>),
    Function(FileItemTreeId<Function>),
    Struct(FileItemTreeId<Struct>),
    Union(FileItemTreeId<Union>),
    Enum(FileItemTreeId<Enum>),
    Const(FileItemTreeId<Const>),
    Static(FileItemTreeId<Static>),
    Trait(FileItemTreeId<Trait>),
    Impl(FileItemTreeId<Impl>),
    TypeAlias(FileItemTreeId<TypeAlias>),
    Mod(FileItemTreeId<Mod>),
    MacroCall(FileItemTreeId<MacroCall>),
}

impl_froms!(ModItem {
    Import(FileItemTreeId<Import>),
    Function(FileItemTreeId<Function>),
    Struct(FileItemTreeId<Struct>),
    Union(FileItemTreeId<Union>),
    Enum(FileItemTreeId<Enum>),
    Const(FileItemTreeId<Const>),
    Static(FileItemTreeId<Static>),
    Trait(FileItemTreeId<Trait>),
    Impl(FileItemTreeId<Impl>),
    TypeAlias(FileItemTreeId<TypeAlias>),
    Mod(FileItemTreeId<Mod>),
    MacroCall(FileItemTreeId<MacroCall>),
});

#[derive(Debug, Eq, PartialEq)]
pub enum AssocItem {
    Function(FileItemTreeId<Function>),
    TypeAlias(FileItemTreeId<TypeAlias>),
    Const(FileItemTreeId<Const>),
    MacroCall(FileItemTreeId<MacroCall>),
}

impl_froms!(AssocItem {
    Function(FileItemTreeId<Function>),
    TypeAlias(FileItemTreeId<TypeAlias>),
    Const(FileItemTreeId<Const>),
    MacroCall(FileItemTreeId<MacroCall>),
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
