//! A simplified AST that only contains items.
//!
//! This is the primary IR used throughout `hir_def`. It is the input to the name resolution
//! algorithm, as well as to the queries defined in `adt.rs`, `data.rs`, and most things in
//! `attr.rs`.
//!
//! `ItemTree`s are built per `HirFileId`, from the syntax tree of the parsed file. This means that
//! they are crate-independent: they don't know which `#[cfg]`s are active or which module they
//! belong to, since those concepts don't exist at this level (a single `ItemTree` might be part of
//! multiple crates, or might be included into the same crate twice via `#[path]`).
//!
//! One important purpose of this layer is to provide an "invalidation barrier" for incremental
//! computations: when typing inside an item body, the `ItemTree` of the modified file is typically
//! unaffected, so we don't have to recompute name resolution results or item data (see `data.rs`).
//!
//! The `ItemTree` for the currently open file can be displayed by using the VS Code command
//! "rust-analyzer: Debug ItemTree".
//!
//! Compared to rustc's architecture, `ItemTree` has properties from both rustc's AST and HIR: many
//! syntax-level Rust features are already desugared to simpler forms in the `ItemTree`, but name
//! resolution has not yet been performed. `ItemTree`s are per-file, while rustc's AST and HIR are
//! per-crate, because we are interested in incrementally computing it.
//!
//! The representation of items in the `ItemTree` should generally mirror the surface syntax: it is
//! usually a bad idea to desugar a syntax-level construct to something that is structurally
//! different here. Name resolution needs to be able to process attributes and expand macros
//! (including attribute macros), and having a 1-to-1 mapping between syntax and the `ItemTree`
//! avoids introducing subtle bugs.
//!
//! In general, any item in the `ItemTree` stores its `AstId`, which allows mapping it back to its
//! surface syntax.

mod lower;
mod pretty;
#[cfg(test)]
mod tests;

use std::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Index,
};

use ast::{AstNode, HasName, StructKind};
use base_db::CrateId;
use either::Either;
use hir_expand::{
    ast_id_map::FileAstId,
    attrs::RawAttrs,
    hygiene::Hygiene,
    name::{name, AsName, Name},
    ExpandTo, HirFileId, InFile,
};
use intern::Interned;
use la_arena::{Arena, Idx, IdxRange, RawIdx};
use profile::Count;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use stdx::never;
use syntax::{ast, match_ast, SyntaxKind};
use triomphe::Arc;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    generics::GenericParams,
    path::{path, AssociatedTypeBinding, GenericArgs, ImportAlias, ModPath, Path, PathKind},
    type_ref::{Mutability, TraitRef, TypeBound, TypeRef},
    visibility::RawVisibility,
    BlockId,
};

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct RawVisibilityId(u32);

impl RawVisibilityId {
    pub const PUB: Self = RawVisibilityId(u32::max_value());
    pub const PRIV: Self = RawVisibilityId(u32::max_value() - 1);
    pub const PUB_CRATE: Self = RawVisibilityId(u32::max_value() - 2);
}

impl fmt::Debug for RawVisibilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_tuple("RawVisibilityId");
        match *self {
            Self::PUB => f.field(&"pub"),
            Self::PRIV => f.field(&"pub(self)"),
            Self::PUB_CRATE => f.field(&"pub(crate)"),
            _ => f.field(&self.0),
        };
        f.finish()
    }
}

/// The item tree of a source file.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ItemTree {
    _c: Count<Self>,

    top_level: SmallVec<[ModItem; 1]>,
    attrs: FxHashMap<AttrOwner, RawAttrs>,

    data: Option<Box<ItemTreeData>>,
}

impl ItemTree {
    pub(crate) fn file_item_tree_query(db: &dyn DefDatabase, file_id: HirFileId) -> Arc<ItemTree> {
        let _p = profile::span("file_item_tree_query").detail(|| format!("{file_id:?}"));
        let syntax = db.parse_or_expand(file_id);
        if never!(syntax.kind() == SyntaxKind::ERROR, "{:?} from {:?} {}", file_id, syntax, syntax)
        {
            // FIXME: not 100% sure why these crop up, but return an empty tree to avoid a panic
            return Default::default();
        }

        let ctx = lower::Ctx::new(db, file_id);
        let mut top_attrs = None;
        let mut item_tree = match_ast! {
            match syntax {
                ast::SourceFile(file) => {
                    top_attrs = Some(RawAttrs::new(db.upcast(), &file, ctx.hygiene()));
                    ctx.lower_module_items(&file)
                },
                ast::MacroItems(items) => {
                    ctx.lower_module_items(&items)
                },
                ast::MacroStmts(stmts) => {
                    // The produced statements can include items, which should be added as top-level
                    // items.
                    ctx.lower_macro_stmts(stmts)
                },
                _ => {
                    panic!("cannot create item tree for file {file_id:?} from {syntax:?} {syntax}");
                },
            }
        };

        if let Some(attrs) = top_attrs {
            item_tree.attrs.insert(AttrOwner::TopLevel, attrs);
        }
        item_tree.shrink_to_fit();
        Arc::new(item_tree)
    }

    /// Returns an iterator over all items located at the top level of the `HirFileId` this
    /// `ItemTree` was created from.
    pub fn top_level_items(&self) -> &[ModItem] {
        &self.top_level
    }

    /// Returns the inner attributes of the source file.
    pub fn top_level_attrs(&self, db: &dyn DefDatabase, krate: CrateId) -> Attrs {
        Attrs::filter(
            db,
            krate,
            self.attrs.get(&AttrOwner::TopLevel).unwrap_or(&RawAttrs::EMPTY).clone(),
        )
    }

    pub(crate) fn raw_attrs(&self, of: AttrOwner) -> &RawAttrs {
        self.attrs.get(&of).unwrap_or(&RawAttrs::EMPTY)
    }

    pub(crate) fn attrs(&self, db: &dyn DefDatabase, krate: CrateId, of: AttrOwner) -> Attrs {
        Attrs::filter(db, krate, self.raw_attrs(of).clone())
    }

    pub fn pretty_print(&self, db: &dyn DefDatabase) -> String {
        pretty::print_item_tree(db.upcast(), self)
    }

    fn data(&self) -> &ItemTreeData {
        self.data.as_ref().expect("attempted to access data of empty ItemTree")
    }

    fn data_mut(&mut self) -> &mut ItemTreeData {
        self.data.get_or_insert_with(Box::default)
    }

    fn block_item_tree(db: &dyn DefDatabase, block: BlockId) -> Arc<ItemTree> {
        let loc = db.lookup_intern_block(block);
        let block = loc.ast_id.to_node(db.upcast());
        let ctx = lower::Ctx::new(db, loc.ast_id.file_id);
        Arc::new(ctx.lower_block(&block))
    }

    fn shrink_to_fit(&mut self) {
        if let Some(data) = &mut self.data {
            let ItemTreeData {
                imports,
                extern_crates,
                extern_blocks,
                functions,
                params,
                structs,
                fields,
                unions,
                enums,
                variants,
                consts,
                statics,
                traits,
                trait_aliases,
                impls,
                type_aliases,
                mods,
                macro_calls,
                macro_rules,
                macro_defs,
                vis,
            } = &mut **data;

            imports.shrink_to_fit();
            extern_crates.shrink_to_fit();
            extern_blocks.shrink_to_fit();
            functions.shrink_to_fit();
            params.shrink_to_fit();
            structs.shrink_to_fit();
            fields.shrink_to_fit();
            unions.shrink_to_fit();
            enums.shrink_to_fit();
            variants.shrink_to_fit();
            consts.shrink_to_fit();
            statics.shrink_to_fit();
            traits.shrink_to_fit();
            trait_aliases.shrink_to_fit();
            impls.shrink_to_fit();
            type_aliases.shrink_to_fit();
            mods.shrink_to_fit();
            macro_calls.shrink_to_fit();
            macro_rules.shrink_to_fit();
            macro_defs.shrink_to_fit();

            vis.arena.shrink_to_fit();
        }
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemVisibilities {
    arena: Arena<RawVisibility>,
}

impl ItemVisibilities {
    fn alloc(&mut self, vis: RawVisibility) -> RawVisibilityId {
        match &vis {
            RawVisibility::Public => RawVisibilityId::PUB,
            RawVisibility::Module(path) if path.segments().is_empty() => match &path.kind {
                PathKind::Super(0) => RawVisibilityId::PRIV,
                PathKind::Crate => RawVisibilityId::PUB_CRATE,
                _ => RawVisibilityId(self.arena.alloc(vis).into_raw().into()),
            },
            _ => RawVisibilityId(self.arena.alloc(vis).into_raw().into()),
        }
    }
}

static VIS_PUB: RawVisibility = RawVisibility::Public;
static VIS_PRIV: RawVisibility = RawVisibility::Module(ModPath::from_kind(PathKind::Super(0)));
static VIS_PUB_CRATE: RawVisibility = RawVisibility::Module(ModPath::from_kind(PathKind::Crate));

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemTreeData {
    imports: Arena<Import>,
    extern_crates: Arena<ExternCrate>,
    extern_blocks: Arena<ExternBlock>,
    functions: Arena<Function>,
    params: Arena<Param>,
    structs: Arena<Struct>,
    fields: Arena<Field>,
    unions: Arena<Union>,
    enums: Arena<Enum>,
    variants: Arena<Variant>,
    consts: Arena<Const>,
    statics: Arena<Static>,
    traits: Arena<Trait>,
    trait_aliases: Arena<TraitAlias>,
    impls: Arena<Impl>,
    type_aliases: Arena<TypeAlias>,
    mods: Arena<Mod>,
    macro_calls: Arena<MacroCall>,
    macro_rules: Arena<MacroRules>,
    macro_defs: Arena<MacroDef>,

    vis: ItemVisibilities,
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum AttrOwner {
    /// Attributes on an item.
    ModItem(ModItem),
    /// Inner attributes of the source file.
    TopLevel,

    Variant(Idx<Variant>),
    Field(Idx<Field>),
    Param(Idx<Param>),
}

macro_rules! from_attrs {
    ( $( $var:ident($t:ty) ),+ ) => {
        $(
            impl From<$t> for AttrOwner {
                fn from(t: $t) -> AttrOwner {
                    AttrOwner::$var(t)
                }
            }
        )+
    };
}

from_attrs!(ModItem(ModItem), Variant(Idx<Variant>), Field(Idx<Field>), Param(Idx<Param>));

/// Trait implemented by all item nodes in the item tree.
pub trait ItemTreeNode: Clone {
    type Source: AstNode + Into<ast::Item>;

    fn ast_id(&self) -> FileAstId<Self::Source>;

    /// Looks up an instance of `Self` in an item tree.
    fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self;

    /// Downcasts a `ModItem` to a `FileItemTreeId` specific to this type.
    fn id_from_mod_item(mod_item: ModItem) -> Option<FileItemTreeId<Self>>;

    /// Upcasts a `FileItemTreeId` to a generic `ModItem`.
    fn id_to_mod_item(id: FileItemTreeId<Self>) -> ModItem;
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

/// Identifies a particular [`ItemTree`].
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct TreeId {
    file: HirFileId,
    block: Option<BlockId>,
}

impl TreeId {
    pub(crate) fn new(file: HirFileId, block: Option<BlockId>) -> Self {
        Self { file, block }
    }

    pub(crate) fn item_tree(&self, db: &dyn DefDatabase) -> Arc<ItemTree> {
        match self.block {
            Some(block) => ItemTree::block_item_tree(db, block),
            None => db.file_item_tree(self.file),
        }
    }

    pub(crate) fn file_id(self) -> HirFileId {
        self.file
    }

    pub(crate) fn is_block(self) -> bool {
        self.block.is_some()
    }
}

#[derive(Debug)]
pub struct ItemTreeId<N: ItemTreeNode> {
    tree: TreeId,
    pub value: FileItemTreeId<N>,
}

impl<N: ItemTreeNode> ItemTreeId<N> {
    pub fn new(tree: TreeId, idx: FileItemTreeId<N>) -> Self {
        Self { tree, value: idx }
    }

    pub fn file_id(self) -> HirFileId {
        self.tree.file
    }

    pub fn tree_id(self) -> TreeId {
        self.tree
    }

    pub fn item_tree(self, db: &dyn DefDatabase) -> Arc<ItemTree> {
        self.tree.item_tree(db)
    }
}

impl<N: ItemTreeNode> Copy for ItemTreeId<N> {}
impl<N: ItemTreeNode> Clone for ItemTreeId<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N: ItemTreeNode> PartialEq for ItemTreeId<N> {
    fn eq(&self, other: &Self) -> bool {
        self.tree == other.tree && self.value == other.value
    }
}

impl<N: ItemTreeNode> Eq for ItemTreeId<N> {}

impl<N: ItemTreeNode> Hash for ItemTreeId<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tree.hash(state);
        self.value.hash(state);
    }
}

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
                type Source = $ast;

                fn ast_id(&self) -> FileAstId<Self::Source> {
                    self.ast_id
                }

                fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self {
                    &tree.data().$fld[index]
                }

                fn id_from_mod_item(mod_item: ModItem) -> Option<FileItemTreeId<Self>> {
                    match mod_item {
                        ModItem::$typ(id) => Some(id),
                        _ => None,
                    }
                }

                fn id_to_mod_item(id: FileItemTreeId<Self>) -> ModItem {
                    ModItem::$typ(id)
                }
            }

            impl Index<Idx<$typ>> for ItemTree {
                type Output = $typ;

                fn index(&self, index: Idx<$typ>) -> &Self::Output {
                    &self.data().$fld[index]
                }
            }
        )+
    };
}

mod_items! {
    Import in imports -> ast::Use,
    ExternCrate in extern_crates -> ast::ExternCrate,
    ExternBlock in extern_blocks -> ast::ExternBlock,
    Function in functions -> ast::Fn,
    Struct in structs -> ast::Struct,
    Union in unions -> ast::Union,
    Enum in enums -> ast::Enum,
    Const in consts -> ast::Const,
    Static in statics -> ast::Static,
    Trait in traits -> ast::Trait,
    TraitAlias in trait_aliases -> ast::TraitAlias,
    Impl in impls -> ast::Impl,
    TypeAlias in type_aliases -> ast::TypeAlias,
    Mod in mods -> ast::Module,
    MacroCall in macro_calls -> ast::MacroCall,
    MacroRules in macro_rules -> ast::MacroRules,
    MacroDef in macro_defs -> ast::MacroDef,
}

macro_rules! impl_index {
    ( $($fld:ident: $t:ty),+ $(,)? ) => {
        $(
            impl Index<Idx<$t>> for ItemTree {
                type Output = $t;

                fn index(&self, index: Idx<$t>) -> &Self::Output {
                    &self.data().$fld[index]
                }
            }
        )+
    };
}

impl_index!(fields: Field, variants: Variant, params: Param);

impl Index<RawVisibilityId> for ItemTree {
    type Output = RawVisibility;
    fn index(&self, index: RawVisibilityId) -> &Self::Output {
        match index {
            RawVisibilityId::PRIV => &VIS_PRIV,
            RawVisibilityId::PUB => &VIS_PUB,
            RawVisibilityId::PUB_CRATE => &VIS_PUB_CRATE,
            _ => &self.data().vis.arena[Idx::from_raw(index.0.into())],
        }
    }
}

impl<N: ItemTreeNode> Index<FileItemTreeId<N>> for ItemTree {
    type Output = N;
    fn index(&self, id: FileItemTreeId<N>) -> &N {
        N::lookup(self, id.index)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Import {
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Use>,
    pub use_tree: UseTree,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UseTree {
    pub index: Idx<ast::UseTree>,
    kind: UseTreeKind,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum UseTreeKind {
    /// ```
    /// use path::to::Item;
    /// use path::to::Item as Renamed;
    /// use path::to::Trait as _;
    /// ```
    Single { path: Interned<ModPath>, alias: Option<ImportAlias> },

    /// ```
    /// use *;  // (invalid, but can occur in nested tree)
    /// use path::*;
    /// ```
    Glob { path: Option<Interned<ModPath>> },

    /// ```
    /// use prefix::{self, Item, ...};
    /// ```
    Prefixed { prefix: Option<Interned<ModPath>>, list: Box<[UseTree]> },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternCrate {
    pub name: Name,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::ExternCrate>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExternBlock {
    pub abi: Option<Interned<str>>,
    pub ast_id: FileAstId<ast::ExternBlock>,
    pub children: Box<[ModItem]>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub explicit_generic_params: Interned<GenericParams>,
    pub abi: Option<Interned<str>>,
    pub params: IdxRange<Param>,
    pub ret_type: Interned<TypeRef>,
    pub ast_id: FileAstId<ast::Fn>,
    pub(crate) flags: FnFlags,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Param {
    Normal(Interned<TypeRef>),
    Varargs,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub(crate) struct FnFlags: u8 {
        const HAS_SELF_PARAM = 1 << 0;
        const HAS_BODY = 1 << 1;
        const HAS_DEFAULT_KW = 1 << 2;
        const HAS_CONST_KW = 1 << 3;
        const HAS_ASYNC_KW = 1 << 4;
        const HAS_UNSAFE_KW = 1 << 5;
        const IS_VARARGS = 1 << 6;
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Interned<GenericParams>,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::Struct>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Interned<GenericParams>,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::Union>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Interned<GenericParams>,
    pub variants: IdxRange<Variant>,
    pub ast_id: FileAstId<ast::Enum>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Const {
    /// `None` for `const _: () = ();`
    pub name: Option<Name>,
    pub visibility: RawVisibilityId,
    pub type_ref: Interned<TypeRef>,
    pub ast_id: FileAstId<ast::Const>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub mutable: bool,
    pub type_ref: Interned<TypeRef>,
    pub ast_id: FileAstId<ast::Static>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Interned<GenericParams>,
    pub is_auto: bool,
    pub is_unsafe: bool,
    pub items: Box<[AssocItem]>,
    pub ast_id: FileAstId<ast::Trait>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TraitAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Interned<GenericParams>,
    pub ast_id: FileAstId<ast::TraitAlias>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Impl {
    pub generic_params: Interned<GenericParams>,
    pub target_trait: Option<Interned<TraitRef>>,
    pub self_ty: Interned<TypeRef>,
    pub is_negative: bool,
    pub items: Box<[AssocItem]>,
    pub ast_id: FileAstId<ast::Impl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    /// Bounds on the type alias itself. Only valid in trait declarations, eg. `type Assoc: Copy;`.
    pub bounds: Box<[Interned<TypeBound>]>,
    pub generic_params: Interned<GenericParams>,
    pub type_ref: Option<Interned<TypeRef>>,
    pub ast_id: FileAstId<ast::TypeAlias>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Mod {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub kind: ModKind,
    pub ast_id: FileAstId<ast::Module>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModKind {
    /// `mod m { ... }`
    Inline { items: Box<[ModItem]> },
    /// `mod m;`
    Outline,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroCall {
    /// Path to the called macro.
    pub path: Interned<ModPath>,
    pub ast_id: FileAstId<ast::MacroCall>,
    pub expand_to: ExpandTo,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroRules {
    /// The name of the declared macro.
    pub name: Name,
    pub ast_id: FileAstId<ast::MacroRules>,
}

/// "Macros 2.0" macro definition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroDef {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::MacroDef>,
}

impl Import {
    /// Maps a `UseTree` contained in this import back to its AST node.
    pub fn use_tree_to_ast(
        &self,
        db: &dyn DefDatabase,
        file_id: HirFileId,
        index: Idx<ast::UseTree>,
    ) -> ast::UseTree {
        // Re-lower the AST item and get the source map.
        // Note: The AST unwraps are fine, since if they fail we should have never obtained `index`.
        let ast = InFile::new(file_id, self.ast_id).to_node(db.upcast());
        let ast_use_tree = ast.use_tree().expect("missing `use_tree`");
        let hygiene = Hygiene::new(db.upcast(), file_id);
        let (_, source_map) =
            lower::lower_use_tree(db, &hygiene, ast_use_tree).expect("failed to lower use tree");
        source_map[index].clone()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportKind {
    /// The `ModPath` is imported normally.
    Plain,
    /// This is a glob-import of all names in the `ModPath`.
    Glob,
    /// This is a `some::path::self` import, which imports `some::path` only in type namespace.
    TypeOnly,
}

impl UseTree {
    /// Expands the `UseTree` into individually imported `ModPath`s.
    pub fn expand(
        &self,
        mut cb: impl FnMut(Idx<ast::UseTree>, ModPath, ImportKind, Option<ImportAlias>),
    ) {
        self.expand_impl(None, &mut cb)
    }

    fn expand_impl(
        &self,
        prefix: Option<ModPath>,
        cb: &mut dyn FnMut(Idx<ast::UseTree>, ModPath, ImportKind, Option<ImportAlias>),
    ) {
        fn concat_mod_paths(
            prefix: Option<ModPath>,
            path: &ModPath,
        ) -> Option<(ModPath, ImportKind)> {
            match (prefix, &path.kind) {
                (None, _) => Some((path.clone(), ImportKind::Plain)),
                (Some(mut prefix), PathKind::Plain) => {
                    for segment in path.segments() {
                        prefix.push_segment(segment.clone());
                    }
                    Some((prefix, ImportKind::Plain))
                }
                (Some(mut prefix), PathKind::Super(n))
                    if *n > 0 && prefix.segments().is_empty() =>
                {
                    // `super::super` + `super::rest`
                    match &mut prefix.kind {
                        PathKind::Super(m) => {
                            cov_mark::hit!(concat_super_mod_paths);
                            *m += *n;
                            for segment in path.segments() {
                                prefix.push_segment(segment.clone());
                            }
                            Some((prefix, ImportKind::Plain))
                        }
                        _ => None,
                    }
                }
                (Some(prefix), PathKind::Super(0)) if path.segments().is_empty() => {
                    // `some::path::self` == `some::path`
                    Some((prefix, ImportKind::TypeOnly))
                }
                (Some(_), _) => None,
            }
        }

        match &self.kind {
            UseTreeKind::Single { path, alias } => {
                if let Some((path, kind)) = concat_mod_paths(prefix, path) {
                    cb(self.index, path, kind, alias.clone());
                }
            }
            UseTreeKind::Glob { path: Some(path) } => {
                if let Some((path, _)) = concat_mod_paths(prefix, path) {
                    cb(self.index, path, ImportKind::Glob, None);
                }
            }
            UseTreeKind::Glob { path: None } => {
                if let Some(prefix) = prefix {
                    cb(self.index, prefix, ImportKind::Glob, None);
                }
            }
            UseTreeKind::Prefixed { prefix: additional_prefix, list } => {
                let prefix = match additional_prefix {
                    Some(path) => match concat_mod_paths(prefix, path) {
                        Some((path, ImportKind::Plain)) => Some(path),
                        _ => return,
                    },
                    None => prefix,
                };
                for tree in &**list {
                    tree.expand_impl(prefix.clone(), cb);
                }
            }
        }
    }
}

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
            | ModItem::ExternBlock(_)
            | ModItem::Struct(_)
            | ModItem::Union(_)
            | ModItem::Enum(_)
            | ModItem::Static(_)
            | ModItem::Trait(_)
            | ModItem::TraitAlias(_)
            | ModItem::Impl(_)
            | ModItem::Mod(_)
            | ModItem::MacroRules(_)
            | ModItem::MacroDef(_) => None,
            ModItem::MacroCall(call) => Some(AssocItem::MacroCall(*call)),
            ModItem::Const(konst) => Some(AssocItem::Const(*konst)),
            ModItem::TypeAlias(alias) => Some(AssocItem::TypeAlias(*alias)),
            ModItem::Function(func) => Some(AssocItem::Function(*func)),
        }
    }

    pub fn ast_id(&self, tree: &ItemTree) -> FileAstId<ast::Item> {
        match self {
            ModItem::Import(it) => tree[it.index].ast_id().upcast(),
            ModItem::ExternCrate(it) => tree[it.index].ast_id().upcast(),
            ModItem::ExternBlock(it) => tree[it.index].ast_id().upcast(),
            ModItem::Function(it) => tree[it.index].ast_id().upcast(),
            ModItem::Struct(it) => tree[it.index].ast_id().upcast(),
            ModItem::Union(it) => tree[it.index].ast_id().upcast(),
            ModItem::Enum(it) => tree[it.index].ast_id().upcast(),
            ModItem::Const(it) => tree[it.index].ast_id().upcast(),
            ModItem::Static(it) => tree[it.index].ast_id().upcast(),
            ModItem::Trait(it) => tree[it.index].ast_id().upcast(),
            ModItem::TraitAlias(it) => tree[it.index].ast_id().upcast(),
            ModItem::Impl(it) => tree[it.index].ast_id().upcast(),
            ModItem::TypeAlias(it) => tree[it.index].ast_id().upcast(),
            ModItem::Mod(it) => tree[it.index].ast_id().upcast(),
            ModItem::MacroCall(it) => tree[it.index].ast_id().upcast(),
            ModItem::MacroRules(it) => tree[it.index].ast_id().upcast(),
            ModItem::MacroDef(it) => tree[it.index].ast_id().upcast(),
        }
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

impl AssocItem {
    pub fn ast_id(self, tree: &ItemTree) -> FileAstId<ast::AssocItem> {
        match self {
            AssocItem::Function(id) => tree[id].ast_id.upcast(),
            AssocItem::TypeAlias(id) => tree[id].ast_id.upcast(),
            AssocItem::Const(id) => tree[id].ast_id.upcast(),
            AssocItem::MacroCall(id) => tree[id].ast_id.upcast(),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Variant {
    pub name: Name,
    pub fields: Fields,
    pub ast_id: FileAstId<ast::Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fields {
    Record(IdxRange<Field>),
    Tuple(IdxRange<Field>),
    Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldAstId {
    Record(FileAstId<ast::RecordField>),
    Tuple(FileAstId<ast::TupleField>),
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: Name,
    pub type_ref: Interned<TypeRef>,
    pub visibility: RawVisibilityId,
    pub ast_id: FieldAstId,
}
