//! A simplified AST that only contains items.

mod lower;
#[cfg(test)]
mod tests;

use std::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Index, Range},
    sync::Arc,
};

use ast::{AstNode, AttrsOwner, NameOwner, StructKind, TypeAscriptionOwner};
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
    generics::GenericParams,
    path::{path, AssociatedTypeBinding, GenericArgs, ImportAlias, ModPath, Path},
    type_ref::{Mutability, TypeBound, TypeRef},
    visibility::RawVisibility,
};
use smallvec::SmallVec;

/// The item tree of a source file.
#[derive(Debug, Eq, PartialEq)]
pub struct ItemTree {
    file_id: HirFileId,
    top_level: Vec<ModItem>,
    top_attrs: Attrs,
    attrs: FxHashMap<ModItem, Attrs>,
    empty_attrs: Attrs,
    inner_items: FxHashMap<FileAstId<ast::ModuleItem>, SmallVec<[ModItem; 1]>>,

    imports: Arena<Import>,
    extern_crates: Arena<ExternCrate>,
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
        let _p = ra_prof::profile("item_tree_query").detail(|| format!("{:?}", file_id));
        let syntax = if let Some(node) = db.parse_or_expand(file_id) {
            node
        } else {
            return Arc::new(Self::empty(file_id));
        };

        let hygiene = Hygiene::new(db.upcast(), file_id);
        let ctx = lower::Ctx::new(db, hygiene.clone(), file_id);
        let mut top_attrs = None;
        let mut item_tree = match_ast! {
            match syntax {
                ast::SourceFile(file) => {
                    top_attrs = Some(Attrs::new(&file, &hygiene));
                    ctx.lower_module_items(&file)
                },
                ast::MacroItems(items) => {
                    ctx.lower_module_items(&items)
                },
                // Macros can expand to expressions. We return an empty item tree in this case, but
                // still need to collect inner items.
                ast::Expr(e) => {
                    ctx.lower_inner_items(e.syntax())
                },
                _ => {
                    panic!("cannot create item tree from {:?}", syntax);
                },
            }
        };

        item_tree.top_attrs = top_attrs.unwrap_or_default();
        Arc::new(item_tree)
    }

    fn empty(file_id: HirFileId) -> Self {
        Self {
            file_id,
            top_level: Default::default(),
            top_attrs: Default::default(),
            attrs: Default::default(),
            empty_attrs: Default::default(),
            inner_items: Default::default(),
            imports: Default::default(),
            extern_crates: Default::default(),
            functions: Default::default(),
            structs: Default::default(),
            fields: Default::default(),
            unions: Default::default(),
            enums: Default::default(),
            variants: Default::default(),
            consts: Default::default(),
            statics: Default::default(),
            traits: Default::default(),
            impls: Default::default(),
            type_aliases: Default::default(),
            mods: Default::default(),
            macro_calls: Default::default(),
            exprs: Default::default(),
        }
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

    /// Returns the lowered inner items that `ast` corresponds to.
    ///
    /// Most AST items are lowered to a single `ModItem`, but some (eg. `use` items) may be lowered
    /// to multiple items in the `ItemTree`.
    pub fn inner_items(&self, ast: FileAstId<ast::ModuleItem>) -> &[ModItem] {
        &self.inner_items[&ast]
    }

    pub fn all_inner_items(&self) -> impl Iterator<Item = ModItem> + '_ {
        self.inner_items.values().flatten().copied()
    }

    pub fn source<S: ItemTreeSource>(
        &self,
        db: &dyn DefDatabase,
        of: FileItemTreeId<S>,
    ) -> S::Source {
        // This unwrap cannot fail, since it has either succeeded above, or resulted in an empty
        // ItemTree (in which case there is no valid `FileItemTreeId` to call this method with).
        let root = db
            .parse_or_expand(self.file_id)
            .expect("parse_or_expand failed on constructed ItemTree");

        let id = self[of].ast_id();
        let map = db.ast_id_map(self.file_id);
        let ptr = map.get(id);
        ptr.to_node(&root)
    }
}

/// Trait implemented by all nodes in the item tree.
pub trait ItemTreeNode: Clone {
    /// Looks up an instance of `Self` in an item tree.
    fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self;

    /// Downcasts a `ModItem` to a `FileItemTreeId` specific to this type.
    fn id_from_mod_item(mod_item: ModItem) -> Option<FileItemTreeId<Self>>;

    /// Upcasts a `FileItemTreeId` to a generic `ModItem`.
    fn id_to_mod_item(id: FileItemTreeId<Self>) -> ModItem;
}

/// Trait for item tree nodes that allow accessing the original AST node.
pub trait ItemTreeSource: ItemTreeNode {
    type Source: AstNode + Into<ast::ModuleItem>;

    fn ast_id(&self) -> FileAstId<Self::Source>;
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

macro_rules! mod_items {
    ( $( $typ:ident in $fld:ident -> $ast:ty ),+ $(,)? ) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        pub enum ModItem {
            $(
                $typ(FileItemTreeId<$typ>),
            )+
        }

        $(
            impl From<FileItemTreeId<$typ>> for ModItem {
                fn from(id: FileItemTreeId<$typ>) -> ModItem {
                    ModItem::$typ(id)
                }
            }
        )+

        $(
            impl ItemTreeNode for $typ {
                fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self {
                    &tree.$fld[index]
                }

                fn id_from_mod_item(mod_item: ModItem) -> Option<FileItemTreeId<Self>> {
                    if let ModItem::$typ(id) = mod_item {
                        Some(id)
                    } else {
                        None
                    }
                }

                fn id_to_mod_item(id: FileItemTreeId<Self>) -> ModItem {
                    ModItem::$typ(id)
                }
            }

            impl ItemTreeSource for $typ {
                type Source = $ast;

                fn ast_id(&self) -> FileAstId<Self::Source> {
                    self.ast_id
                }
            }

            impl Index<Idx<$typ>> for ItemTree {
                type Output = $typ;

                fn index(&self, index: Idx<$typ>) -> &Self::Output {
                    &self.$fld[index]
                }
            }
        )+
    };
}

mod_items! {
    Import in imports -> ast::UseItem,
    ExternCrate in extern_crates -> ast::ExternCrateItem,
    Function in functions -> ast::FnDef,
    Struct in structs -> ast::StructDef,
    Union in unions -> ast::UnionDef,
    Enum in enums -> ast::EnumDef,
    Const in consts -> ast::ConstDef,
    Static in statics -> ast::StaticDef,
    Trait in traits -> ast::TraitDef,
    Impl in impls -> ast::ImplDef,
    TypeAlias in type_aliases -> ast::TypeAliasDef,
    Mod in mods -> ast::Module,
    MacroCall in macro_calls -> ast::MacroCall,
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

impl_index!(fields: Field, variants: Variant, exprs: Expr);

impl<N: ItemTreeNode> Index<FileItemTreeId<N>> for ItemTree {
    type Output = N;
    fn index(&self, id: FileItemTreeId<N>) -> &N {
        N::lookup(self, id.index)
    }
}

/// A desugared `use` import.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Import {
    pub path: ModPath,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibility,
    pub is_glob: bool,
    pub is_prelude: bool,
    /// AST ID of the `use` or `extern crate` item this import was derived from. Note that many
    /// `Import`s can map to the same `use` item.
    pub ast_id: FileAstId<ast::UseItem>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternCrate {
    pub path: ModPath,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibility,
    /// Whether this is a `#[macro_use] extern crate ...`.
    pub is_macro_use: bool,
    pub ast_id: FileAstId<ast::ExternCrateItem>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub has_self_param: bool,
    pub is_unsafe: bool,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    pub ast_id: FileAstId<ast::FnDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::StructDef>,
    pub kind: StructDefKind,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StructDefKind {
    /// `struct S { ... }` - type namespace only.
    Record,
    /// `struct S(...);`
    Tuple,
    /// `struct S;`
    Unit,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::UnionDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub variants: Range<Idx<Variant>>,
    pub ast_id: FileAstId<ast::EnumDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Const {
    /// const _: () = ();
    pub name: Option<Name>,
    pub visibility: RawVisibility,
    pub type_ref: TypeRef,
    pub ast_id: FileAstId<ast::ConstDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibility,
    pub mutable: bool,
    pub type_ref: TypeRef,
    pub ast_id: FileAstId<ast::StaticDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibility,
    pub generic_params: GenericParams,
    pub auto: bool,
    pub items: Vec<AssocItem>,
    pub ast_id: FileAstId<ast::TraitDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Impl {
    pub generic_params: GenericParams,
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
    /// Bounds on the type alias itself. Only valid in trait declarations, eg. `type Assoc: Copy;`.
    pub bounds: Vec<TypeBound>,
    pub generic_params: GenericParams,
    pub type_ref: Option<TypeRef>,
    pub ast_id: FileAstId<ast::TypeAliasDef>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Mod {
    pub name: Name,
    pub visibility: RawVisibility,
    pub kind: ModKind,
    pub ast_id: FileAstId<ast::Module>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModKind {
    /// `mod m { ... }`
    Inline { items: Vec<ModItem> },

    /// `mod m;`
    Outline {},
}

#[derive(Debug, Clone, Eq, PartialEq)]
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
#[derive(Debug, Clone, Eq, PartialEq)]
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

impl ModItem {
    pub fn as_assoc_item(&self) -> Option<AssocItem> {
        match self {
            ModItem::Import(_)
            | ModItem::ExternCrate(_)
            | ModItem::Struct(_)
            | ModItem::Union(_)
            | ModItem::Enum(_)
            | ModItem::Static(_)
            | ModItem::Trait(_)
            | ModItem::Impl(_)
            | ModItem::Mod(_) => None,
            ModItem::MacroCall(call) => Some(AssocItem::MacroCall(*call)),
            ModItem::Const(konst) => Some(AssocItem::Const(*konst)),
            ModItem::TypeAlias(alias) => Some(AssocItem::TypeAlias(*alias)),
            ModItem::Function(func) => Some(AssocItem::Function(*func)),
        }
    }

    pub fn downcast<N: ItemTreeNode>(self) -> Option<FileItemTreeId<N>> {
        N::id_from_mod_item(self)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

impl From<AssocItem> for ModItem {
    fn from(item: AssocItem) -> Self {
        match item {
            AssocItem::Function(it) => it.into(),
            AssocItem::TypeAlias(it) => it.into(),
            AssocItem::Const(it) => it.into(),
            AssocItem::MacroCall(it) => it.into(),
        }
    }
}

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
