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
    hash::Hash,
    ops::Index,
    sync::OnceLock,
};

use ast::{AstNode, StructKind};
use base_db::Crate;
use hir_expand::{
    ExpandTo, HirFileId,
    attrs::RawAttrs,
    mod_path::{ModPath, PathKind},
    name::Name,
};
use intern::Interned;
use la_arena::{Idx, RawIdx};
use rustc_hash::FxHashMap;
use span::{AstIdNode, Edition, FileAstId, SyntaxContext};
use stdx::never;
use syntax::{SyntaxKind, ast, match_ast};
use triomphe::Arc;

use crate::{BlockId, Lookup, attr::Attrs, db::DefDatabase};

pub(crate) use crate::item_tree::lower::{lower_use_tree, visibility_from_ast};

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct RawVisibilityId(u32);

impl RawVisibilityId {
    pub const PUB: Self = RawVisibilityId(u32::MAX);
    pub const PRIV_IMPLICIT: Self = RawVisibilityId(u32::MAX - 1);
    pub const PRIV_EXPLICIT: Self = RawVisibilityId(u32::MAX - 2);
    pub const PUB_CRATE: Self = RawVisibilityId(u32::MAX - 3);
}

impl fmt::Debug for RawVisibilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_tuple("RawVisibilityId");
        match *self {
            Self::PUB => f.field(&"pub"),
            Self::PRIV_IMPLICIT | Self::PRIV_EXPLICIT => f.field(&"pub(self)"),
            Self::PUB_CRATE => f.field(&"pub(crate)"),
            _ => f.field(&self.0),
        };
        f.finish()
    }
}

/// The item tree of a source file.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ItemTree {
    top_level: Box<[ModItemId]>,
    top_attrs: RawAttrs,
    attrs: FxHashMap<FileAstId<ast::Item>, RawAttrs>,
    vis: ItemVisibilities,
    // FIXME: They values store the key, turn this into a FxHashSet<ModItem> instead?
    data: FxHashMap<FileAstId<ast::Item>, ModItem>,
}

impl ItemTree {
    pub(crate) fn file_item_tree_query(db: &dyn DefDatabase, file_id: HirFileId) -> Arc<ItemTree> {
        let _p = tracing::info_span!("file_item_tree_query", ?file_id).entered();
        static EMPTY: OnceLock<Arc<ItemTree>> = OnceLock::new();

        let ctx = lower::Ctx::new(db, file_id);
        let syntax = db.parse_or_expand(file_id);
        let mut item_tree = match_ast! {
            match syntax {
                ast::SourceFile(file) => {
                    let top_attrs = RawAttrs::new(db, &file, ctx.span_map());
                    let mut item_tree = ctx.lower_module_items(&file);
                    item_tree.top_attrs = top_attrs;
                    item_tree
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
                    if never!(syntax.kind() == SyntaxKind::ERROR, "{:?} from {:?} {}", file_id, syntax, syntax) {
                        return Default::default();
                    }
                    panic!("cannot create item tree for file {file_id:?} from {syntax:?} {syntax}");
                },
            }
        };

        if item_tree.data.is_empty()
            && item_tree.top_level.is_empty()
            && item_tree.attrs.is_empty()
            && item_tree.top_attrs.is_empty()
        {
            EMPTY
                .get_or_init(|| {
                    Arc::new(ItemTree {
                        top_level: Box::new([]),
                        attrs: FxHashMap::default(),
                        data: FxHashMap::default(),
                        top_attrs: RawAttrs::EMPTY,
                        vis: ItemVisibilities { arena: Box::new([]) },
                    })
                })
                .clone()
        } else {
            item_tree.shrink_to_fit();
            Arc::new(item_tree)
        }
    }

    pub(crate) fn block_item_tree_query(db: &dyn DefDatabase, block: BlockId) -> Arc<ItemTree> {
        let _p = tracing::info_span!("block_item_tree_query", ?block).entered();
        static EMPTY: OnceLock<Arc<ItemTree>> = OnceLock::new();

        let loc = block.lookup(db);
        let block = loc.ast_id.to_node(db);

        let ctx = lower::Ctx::new(db, loc.ast_id.file_id);
        let mut item_tree = ctx.lower_block(&block);
        if item_tree.data.is_empty()
            && item_tree.top_level.is_empty()
            && item_tree.attrs.is_empty()
            && item_tree.top_attrs.is_empty()
        {
            EMPTY
                .get_or_init(|| {
                    Arc::new(ItemTree {
                        top_level: Box::new([]),
                        attrs: FxHashMap::default(),
                        data: FxHashMap::default(),
                        top_attrs: RawAttrs::EMPTY,
                        vis: ItemVisibilities { arena: Box::new([]) },
                    })
                })
                .clone()
        } else {
            item_tree.shrink_to_fit();
            Arc::new(item_tree)
        }
    }

    /// Returns an iterator over all items located at the top level of the `HirFileId` this
    /// `ItemTree` was created from.
    pub(crate) fn top_level_items(&self) -> &[ModItemId] {
        &self.top_level
    }

    /// Returns the inner attributes of the source file.
    pub fn top_level_raw_attrs(&self) -> &RawAttrs {
        &self.top_attrs
    }

    /// Returns the inner attributes of the source file.
    pub fn top_level_attrs(&self, db: &dyn DefDatabase, krate: Crate) -> Attrs {
        Attrs::expand_cfg_attr(db, krate, self.top_attrs.clone())
    }

    pub(crate) fn raw_attrs(&self, of: FileAstId<ast::Item>) -> &RawAttrs {
        self.attrs.get(&of).unwrap_or(&RawAttrs::EMPTY)
    }

    pub(crate) fn attrs(
        &self,
        db: &dyn DefDatabase,
        krate: Crate,
        of: FileAstId<ast::Item>,
    ) -> Attrs {
        Attrs::expand_cfg_attr(db, krate, self.raw_attrs(of).clone())
    }

    /// Returns a count of a few, expensive items.
    ///
    /// For more detail, see [`ItemTreeDataStats`].
    pub fn item_tree_stats(&self) -> ItemTreeDataStats {
        let mut traits = 0;
        let mut impls = 0;
        let mut mods = 0;
        let mut macro_calls = 0;
        let mut macro_rules = 0;
        for item in self.data.values() {
            match item {
                ModItem::Trait(_) => traits += 1,
                ModItem::Impl(_) => impls += 1,
                ModItem::Mod(_) => mods += 1,
                ModItem::MacroCall(_) => macro_calls += 1,
                ModItem::MacroRules(_) => macro_rules += 1,
                _ => {}
            }
        }
        ItemTreeDataStats { traits, impls, mods, macro_calls, macro_rules }
    }

    pub fn pretty_print(&self, db: &dyn DefDatabase, edition: Edition) -> String {
        pretty::print_item_tree(db, self, edition)
    }

    fn shrink_to_fit(&mut self) {
        let ItemTree { top_level: _, attrs, data, vis: _, top_attrs: _ } = self;
        attrs.shrink_to_fit();
        data.shrink_to_fit();
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemVisibilities {
    arena: Box<[RawVisibility]>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum ModItem {
    Const(Const),
    Enum(Enum),
    ExternBlock(ExternBlock),
    ExternCrate(ExternCrate),
    Function(Function),
    Impl(Impl),
    Macro2(Macro2),
    MacroCall(MacroCall),
    MacroRules(MacroRules),
    Mod(Mod),
    Static(Static),
    Struct(Struct),
    Trait(Trait),
    TraitAlias(TraitAlias),
    TypeAlias(TypeAlias),
    Union(Union),
    Use(Use),
}
#[derive(Default, Debug, Eq, PartialEq)]
pub struct ItemTreeDataStats {
    pub traits: usize,
    pub impls: usize,
    pub mods: usize,
    pub macro_calls: usize,
    pub macro_rules: usize,
}

/// Trait implemented by all nodes in the item tree.
pub trait ItemTreeNode: Clone {
    type Source: AstIdNode;

    fn ast_id(&self) -> FileAstId<Self::Source>;

    /// Looks up an instance of `Self` in an item tree.
    fn lookup(tree: &ItemTree, index: FileAstId<Self::Source>) -> &Self;
}

#[allow(type_alias_bounds)]
pub type ItemTreeAstId<T: ItemTreeNode> = FileAstId<T::Source>;

/// Identifies a particular [`ItemTree`].
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct TreeId {
    file: HirFileId,
    block: Option<BlockId>,
}

impl TreeId {
    pub fn new(file: HirFileId, block: Option<BlockId>) -> Self {
        Self { file, block }
    }

    pub fn item_tree(&self, db: &dyn DefDatabase) -> Arc<ItemTree> {
        match self.block {
            Some(block) => db.block_item_tree(block),
            None => db.file_item_tree(self.file),
        }
    }

    #[inline]
    pub fn file_id(self) -> HirFileId {
        self.file
    }

    pub fn is_block(self) -> bool {
        self.block.is_some()
    }
}

macro_rules! mod_items {
    ($mod_item:ident -> $( $typ:ident in $fld:ident -> $ast:ty ),+ $(,)? ) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        pub(crate) enum $mod_item {
            $(
                $typ(FileAstId<$ast>),
            )+
        }

        impl $mod_item {
            pub(crate) fn ast_id(self) -> FileAstId<ast::Item> {
                match self {
                    $($mod_item::$typ(it) => it.upcast()),+
                }
            }
        }

        $(
            impl From<FileAstId<$ast>> for $mod_item {
                fn from(id: FileAstId<$ast>) -> $mod_item {
                    ModItemId::$typ(id)
                }
            }
        )+

        $(
            impl ItemTreeNode for $typ {
                type Source = $ast;

                fn ast_id(&self) -> FileAstId<$ast> {
                    self.ast_id
                }

                fn lookup(tree: &ItemTree, index: FileAstId<$ast>) -> &Self {
                    match &tree.data[&index.upcast()] {
                        ModItem::$typ(item) => item,
                        _ => panic!("expected item of type `{}` at index `{:?}`", stringify!($typ), index),
                    }
                }
            }

            impl Index<FileAstId<$ast>> for ItemTree {
                type Output = $typ;

                fn index(&self, index: FileAstId<$ast>) -> &Self::Output {
                    match &self.data[&index.upcast()] {
                        ModItem::$typ(item) => item,
                        _ => panic!("expected item of type `{}` at index `{:?}`", stringify!($typ), index),
                    }
                }
            }
        )+
    };
}

mod_items! {
ModItemId ->
    Use in uses -> ast::Use,
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
    Macro2 in macro_defs -> ast::MacroDef,
}

impl Index<RawVisibilityId> for ItemTree {
    type Output = RawVisibility;
    fn index(&self, index: RawVisibilityId) -> &Self::Output {
        static VIS_PUB: RawVisibility = RawVisibility::Public;
        static VIS_PRIV_IMPLICIT: OnceLock<RawVisibility> = OnceLock::new();
        static VIS_PRIV_EXPLICIT: OnceLock<RawVisibility> = OnceLock::new();
        static VIS_PUB_CRATE: OnceLock<RawVisibility> = OnceLock::new();

        match index {
            RawVisibilityId::PRIV_IMPLICIT => VIS_PRIV_IMPLICIT.get_or_init(|| {
                RawVisibility::Module(
                    Interned::new(ModPath::from_kind(PathKind::SELF)),
                    VisibilityExplicitness::Implicit,
                )
            }),
            RawVisibilityId::PRIV_EXPLICIT => VIS_PRIV_EXPLICIT.get_or_init(|| {
                RawVisibility::Module(
                    Interned::new(ModPath::from_kind(PathKind::SELF)),
                    VisibilityExplicitness::Explicit,
                )
            }),
            RawVisibilityId::PUB => &VIS_PUB,
            RawVisibilityId::PUB_CRATE => VIS_PUB_CRATE.get_or_init(|| {
                RawVisibility::Module(
                    Interned::new(ModPath::from_kind(PathKind::Crate)),
                    VisibilityExplicitness::Explicit,
                )
            }),
            _ => &self.vis.arena[index.0 as usize],
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Use {
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Use>,
    pub use_tree: UseTree,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UseTree {
    kind: UseTreeKind,
}

// FIXME: Would be nice to encode `None` into this
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportAlias {
    /// Unnamed alias, as in `use Foo as _;`
    Underscore,
    /// Named alias
    Alias(Name),
}

impl ImportAlias {
    pub fn display(&self, edition: Edition) -> impl fmt::Display + '_ {
        ImportAliasDisplay { value: self, edition }
    }
}

struct ImportAliasDisplay<'a> {
    value: &'a ImportAlias,
    edition: Edition,
}

impl fmt::Display for ImportAliasDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            ImportAlias::Underscore => f.write_str("_"),
            ImportAlias::Alias(name) => fmt::Display::fmt(&name.display_no_db(self.edition), f),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum UseTreeKind {
    /// ```ignore
    /// use path::to::Item;
    /// use path::to::Item as Renamed;
    /// use path::to::Trait as _;
    /// ```
    Single { path: Interned<ModPath>, alias: Option<ImportAlias> },

    /// ```ignore
    /// use *;  // (invalid, but can occur in nested tree)
    /// use path::*;
    /// ```
    Glob { path: Option<Interned<ModPath>> },

    /// ```ignore
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
    pub ast_id: FileAstId<ast::ExternBlock>,
    pub(crate) children: Box<[ModItemId]>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Fn>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub shape: FieldsShape,
    pub ast_id: FileAstId<ast::Struct>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Union>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Enum>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FieldsShape {
    Record,
    Tuple,
    Unit,
}

/// Visibility of an item, not yet resolved.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RawVisibility {
    /// `pub(in module)`, `pub(crate)` or `pub(super)`. Also private, which is
    /// equivalent to `pub(self)`.
    Module(Interned<ModPath>, VisibilityExplicitness),
    /// `pub`.
    Public,
}

/// Whether the item was imported through an explicit `pub(crate) use` or just a `use` without
/// visibility.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum VisibilityExplicitness {
    Explicit,
    Implicit,
}

impl VisibilityExplicitness {
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Const {
    /// `None` for `const _: () = ();`
    pub name: Option<Name>,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Const>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Static>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::Trait>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TraitAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::TraitAlias>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Impl {
    pub ast_id: FileAstId<ast::Impl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::TypeAlias>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Mod {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub(crate) kind: ModKind,
    pub ast_id: FileAstId<ast::Module>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum ModKind {
    /// `mod m { ... }`
    Inline { items: Box<[ModItemId]> },
    /// `mod m;`
    Outline,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroCall {
    /// Path to the called macro.
    pub path: Interned<ModPath>,
    pub ast_id: FileAstId<ast::MacroCall>,
    pub expand_to: ExpandTo,
    pub ctxt: SyntaxContext,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroRules {
    /// The name of the declared macro.
    pub name: Name,
    pub ast_id: FileAstId<ast::MacroRules>,
}

/// "Macros 2.0" macro definition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Macro2 {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub ast_id: FileAstId<ast::MacroDef>,
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

impl Use {
    /// Expands the `UseTree` into individually imported `ModPath`s.
    pub fn expand(
        &self,
        mut cb: impl FnMut(Idx<ast::UseTree>, ModPath, ImportKind, Option<ImportAlias>),
    ) {
        self.use_tree.expand_impl(None, &mut 0, &mut cb)
    }
}

impl UseTree {
    /// The [`UseTreeKind`] of this `UseTree`.
    pub fn kind(&self) -> &UseTreeKind {
        &self.kind
    }

    fn expand_impl(
        &self,
        prefix: Option<ModPath>,
        counting_index: &mut u32,
        cb: &mut impl FnMut(Idx<ast::UseTree>, ModPath, ImportKind, Option<ImportAlias>),
    ) {
        fn concat_mod_paths(
            prefix: Option<ModPath>,
            path: &ModPath,
        ) -> Option<(ModPath, ImportKind)> {
            match (prefix, path.kind) {
                (None, _) => Some((path.clone(), ImportKind::Plain)),
                (Some(mut prefix), PathKind::Plain) => {
                    prefix.extend(path.segments().iter().cloned());
                    Some((prefix, ImportKind::Plain))
                }
                (Some(mut prefix), PathKind::Super(n)) if n > 0 && prefix.segments().is_empty() => {
                    // `super::super` + `super::rest`
                    match &mut prefix.kind {
                        PathKind::Super(m) => {
                            cov_mark::hit!(concat_super_mod_paths);
                            *m += n;
                            prefix.extend(path.segments().iter().cloned());
                            Some((prefix, ImportKind::Plain))
                        }
                        _ => None,
                    }
                }
                (Some(prefix), PathKind::SELF) if path.segments().is_empty() => {
                    // `some::path::self` == `some::path`
                    Some((prefix, ImportKind::TypeOnly))
                }
                (Some(_), _) => None,
            }
        }

        match &self.kind {
            UseTreeKind::Single { path, alias } => {
                if let Some((path, kind)) = concat_mod_paths(prefix, path) {
                    cb(Idx::from_raw(RawIdx::from_u32(*counting_index)), path, kind, alias.clone());
                }
            }
            UseTreeKind::Glob { path: Some(path) } => {
                if let Some((path, _)) = concat_mod_paths(prefix, path) {
                    cb(
                        Idx::from_raw(RawIdx::from_u32(*counting_index)),
                        path,
                        ImportKind::Glob,
                        None,
                    );
                }
            }
            UseTreeKind::Glob { path: None } => {
                if let Some(prefix) = prefix {
                    cb(
                        Idx::from_raw(RawIdx::from_u32(*counting_index)),
                        prefix,
                        ImportKind::Glob,
                        None,
                    );
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
                    *counting_index += 1;
                    tree.expand_impl(prefix.clone(), counting_index, cb);
                }
            }
        }
    }
}
