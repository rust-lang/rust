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
#![allow(unexpected_cfgs)]

mod lower;
mod pretty;
#[cfg(test)]
mod tests;

use std::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    ops::{Index, Range},
    sync::OnceLock,
};

use ast::{AstNode, StructKind};
use base_db::CrateId;
use either::Either;
use hir_expand::{attrs::RawAttrs, name::Name, ExpandTo, HirFileId, InFile};
use intern::{Interned, Symbol};
use la_arena::{Arena, Idx, RawIdx};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use span::{AstIdNode, Edition, FileAstId, SyntaxContextId};
use stdx::never;
use syntax::{ast, match_ast, SyntaxKind};
use triomphe::Arc;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    generics::GenericParams,
    path::{GenericArgs, ImportAlias, ModPath, Path, PathKind},
    type_ref::{Mutability, TraitRef, TypeBound, TypeRefId, TypesMap, TypesSourceMap},
    visibility::{RawVisibility, VisibilityExplicitness},
    BlockId, LocalLifetimeParamId, LocalTypeOrConstParamId, Lookup,
};

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
    top_level: SmallVec<[ModItem; 1]>,
    attrs: FxHashMap<AttrOwner, RawAttrs>,

    data: Option<Box<ItemTreeData>>,
}

impl ItemTree {
    pub(crate) fn file_item_tree_query(db: &dyn DefDatabase, file_id: HirFileId) -> Arc<ItemTree> {
        db.file_item_tree_with_source_map(file_id).0
    }

    pub(crate) fn file_item_tree_with_source_map_query(
        db: &dyn DefDatabase,
        file_id: HirFileId,
    ) -> (Arc<ItemTree>, Arc<ItemTreeSourceMaps>) {
        let _p = tracing::info_span!("file_item_tree_query", ?file_id).entered();
        static EMPTY: OnceLock<(Arc<ItemTree>, Arc<ItemTreeSourceMaps>)> = OnceLock::new();

        let ctx = lower::Ctx::new(db, file_id);
        let syntax = db.parse_or_expand(file_id);
        let mut top_attrs = None;
        let (mut item_tree, source_maps) = match_ast! {
            match syntax {
                ast::SourceFile(file) => {
                    top_attrs = Some(RawAttrs::new(db.upcast(), &file, ctx.span_map()));
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
                    if never!(syntax.kind() == SyntaxKind::ERROR, "{:?} from {:?} {}", file_id, syntax, syntax) {
                        return Default::default();
                    }
                    panic!("cannot create item tree for file {file_id:?} from {syntax:?} {syntax}");
                },
            }
        };

        if let Some(attrs) = top_attrs {
            item_tree.attrs.insert(AttrOwner::TopLevel, attrs);
        }
        if item_tree.data.is_none() && item_tree.top_level.is_empty() && item_tree.attrs.is_empty()
        {
            EMPTY
                .get_or_init(|| {
                    (
                        Arc::new(ItemTree {
                            top_level: SmallVec::new_const(),
                            attrs: FxHashMap::default(),
                            data: None,
                        }),
                        Arc::default(),
                    )
                })
                .clone()
        } else {
            item_tree.shrink_to_fit();
            (Arc::new(item_tree), Arc::new(source_maps))
        }
    }

    pub(crate) fn block_item_tree_query(db: &dyn DefDatabase, block: BlockId) -> Arc<ItemTree> {
        db.block_item_tree_with_source_map(block).0
    }

    pub(crate) fn block_item_tree_with_source_map_query(
        db: &dyn DefDatabase,
        block: BlockId,
    ) -> (Arc<ItemTree>, Arc<ItemTreeSourceMaps>) {
        let _p = tracing::info_span!("block_item_tree_query", ?block).entered();
        static EMPTY: OnceLock<(Arc<ItemTree>, Arc<ItemTreeSourceMaps>)> = OnceLock::new();

        let loc = block.lookup(db);
        let block = loc.ast_id.to_node(db.upcast());

        let ctx = lower::Ctx::new(db, loc.ast_id.file_id);
        let (mut item_tree, source_maps) = ctx.lower_block(&block);
        if item_tree.data.is_none() && item_tree.top_level.is_empty() && item_tree.attrs.is_empty()
        {
            EMPTY
                .get_or_init(|| {
                    (
                        Arc::new(ItemTree {
                            top_level: SmallVec::new_const(),
                            attrs: FxHashMap::default(),
                            data: None,
                        }),
                        Arc::default(),
                    )
                })
                .clone()
        } else {
            item_tree.shrink_to_fit();
            (Arc::new(item_tree), Arc::new(source_maps))
        }
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

    pub fn pretty_print(&self, db: &dyn DefDatabase, edition: Edition) -> String {
        pretty::print_item_tree(db, self, edition)
    }

    fn data(&self) -> &ItemTreeData {
        self.data.as_ref().expect("attempted to access data of empty ItemTree")
    }

    fn data_mut(&mut self) -> &mut ItemTreeData {
        self.data.get_or_insert_with(Box::default)
    }

    fn shrink_to_fit(&mut self) {
        if let Some(data) = &mut self.data {
            let ItemTreeData {
                uses,
                extern_crates,
                extern_blocks,
                functions,
                structs,
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

            uses.shrink_to_fit();
            extern_crates.shrink_to_fit();
            extern_blocks.shrink_to_fit();
            functions.shrink_to_fit();
            structs.shrink_to_fit();
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
            RawVisibility::Module(path, explicitiy) if path.segments().is_empty() => {
                match (path.kind, explicitiy) {
                    (PathKind::SELF, VisibilityExplicitness::Explicit) => {
                        RawVisibilityId::PRIV_EXPLICIT
                    }
                    (PathKind::SELF, VisibilityExplicitness::Implicit) => {
                        RawVisibilityId::PRIV_IMPLICIT
                    }
                    (PathKind::Crate, _) => RawVisibilityId::PUB_CRATE,
                    _ => RawVisibilityId(self.arena.alloc(vis).into_raw().into()),
                }
            }
            _ => RawVisibilityId(self.arena.alloc(vis).into_raw().into()),
        }
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemTreeData {
    uses: Arena<Use>,
    extern_crates: Arena<ExternCrate>,
    extern_blocks: Arena<ExternBlock>,
    functions: Arena<Function>,
    structs: Arena<Struct>,
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
    macro_defs: Arena<Macro2>,

    vis: ItemVisibilities,
}

#[derive(Default, Debug, Eq, PartialEq)]
pub struct ItemTreeSourceMaps {
    all_concatenated: Box<[TypesSourceMap]>,
    structs_offset: u32,
    unions_offset: u32,
    enum_generics_offset: u32,
    variants_offset: u32,
    consts_offset: u32,
    statics_offset: u32,
    trait_generics_offset: u32,
    trait_alias_generics_offset: u32,
    impls_offset: u32,
    type_aliases_offset: u32,
}

#[derive(Clone, Copy)]
pub struct GenericItemSourceMap<'a>(&'a [TypesSourceMap; 2]);

impl<'a> GenericItemSourceMap<'a> {
    #[inline]
    pub fn item(self) -> &'a TypesSourceMap {
        &self.0[0]
    }

    #[inline]
    pub fn generics(self) -> &'a TypesSourceMap {
        &self.0[1]
    }
}

#[derive(Default, Debug, Eq, PartialEq)]
pub struct GenericItemSourceMapBuilder {
    pub item: TypesSourceMap,
    pub generics: TypesSourceMap,
}

#[derive(Default, Debug, Eq, PartialEq)]
struct ItemTreeSourceMapsBuilder {
    functions: Vec<GenericItemSourceMapBuilder>,
    structs: Vec<GenericItemSourceMapBuilder>,
    unions: Vec<GenericItemSourceMapBuilder>,
    enum_generics: Vec<TypesSourceMap>,
    variants: Vec<TypesSourceMap>,
    consts: Vec<TypesSourceMap>,
    statics: Vec<TypesSourceMap>,
    trait_generics: Vec<TypesSourceMap>,
    trait_alias_generics: Vec<TypesSourceMap>,
    impls: Vec<GenericItemSourceMapBuilder>,
    type_aliases: Vec<GenericItemSourceMapBuilder>,
}

impl ItemTreeSourceMapsBuilder {
    fn build(self) -> ItemTreeSourceMaps {
        let ItemTreeSourceMapsBuilder {
            functions,
            structs,
            unions,
            enum_generics,
            variants,
            consts,
            statics,
            trait_generics,
            trait_alias_generics,
            impls,
            type_aliases,
        } = self;
        let structs_offset = functions.len() as u32 * 2;
        let unions_offset = structs_offset + (structs.len() as u32 * 2);
        let enum_generics_offset = unions_offset + (unions.len() as u32 * 2);
        let variants_offset = enum_generics_offset + (enum_generics.len() as u32);
        let consts_offset = variants_offset + (variants.len() as u32);
        let statics_offset = consts_offset + (consts.len() as u32);
        let trait_generics_offset = statics_offset + (statics.len() as u32);
        let trait_alias_generics_offset = trait_generics_offset + (trait_generics.len() as u32);
        let impls_offset = trait_alias_generics_offset + (trait_alias_generics.len() as u32);
        let type_aliases_offset = impls_offset + (impls.len() as u32 * 2);
        let all_concatenated = generics_concat(functions)
            .chain(generics_concat(structs))
            .chain(generics_concat(unions))
            .chain(enum_generics)
            .chain(variants)
            .chain(consts)
            .chain(statics)
            .chain(trait_generics)
            .chain(trait_alias_generics)
            .chain(generics_concat(impls))
            .chain(generics_concat(type_aliases))
            .collect();
        return ItemTreeSourceMaps {
            all_concatenated,
            structs_offset,
            unions_offset,
            enum_generics_offset,
            variants_offset,
            consts_offset,
            statics_offset,
            trait_generics_offset,
            trait_alias_generics_offset,
            impls_offset,
            type_aliases_offset,
        };

        fn generics_concat(
            source_maps: Vec<GenericItemSourceMapBuilder>,
        ) -> impl Iterator<Item = TypesSourceMap> {
            source_maps.into_iter().flat_map(|it| [it.item, it.generics])
        }
    }
}

impl ItemTreeSourceMaps {
    #[inline]
    fn generic_item(&self, offset: u32, index: u32) -> GenericItemSourceMap<'_> {
        GenericItemSourceMap(
            self.all_concatenated[(offset + (index * 2)) as usize..][..2].try_into().unwrap(),
        )
    }

    #[inline]
    fn non_generic_item(&self, offset: u32, index: u32) -> &TypesSourceMap {
        &self.all_concatenated[(offset + index) as usize]
    }

    #[inline]
    pub fn function(&self, index: FileItemTreeId<Function>) -> GenericItemSourceMap<'_> {
        self.generic_item(0, index.0.into_raw().into_u32())
    }
}

macro_rules! index_item_source_maps {
    ( $( $name:ident; $field:ident[$tree_id:ident]; $fn:ident; $ret:ty, )* ) => {
        impl ItemTreeSourceMaps {
            $(
                #[inline]
                pub fn $name(&self, index: FileItemTreeId<$tree_id>) -> $ret {
                    self.$fn(self.$field, index.0.into_raw().into_u32())
                }
            )*
        }
    };
}
index_item_source_maps! {
    strukt; structs_offset[Struct]; generic_item; GenericItemSourceMap<'_>,
    union; unions_offset[Union]; generic_item; GenericItemSourceMap<'_>,
    enum_generic; enum_generics_offset[Enum]; non_generic_item; &TypesSourceMap,
    variant; variants_offset[Variant]; non_generic_item; &TypesSourceMap,
    konst; consts_offset[Const]; non_generic_item; &TypesSourceMap,
    statik; statics_offset[Static]; non_generic_item; &TypesSourceMap,
    trait_generic; trait_generics_offset[Trait]; non_generic_item; &TypesSourceMap,
    trait_alias_generic; trait_alias_generics_offset[TraitAlias]; non_generic_item; &TypesSourceMap,
    impl_; impls_offset[Impl]; generic_item; GenericItemSourceMap<'_>,
    type_alias; type_aliases_offset[TypeAlias]; generic_item; GenericItemSourceMap<'_>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AttrOwner {
    /// Attributes on an item.
    ModItem(ModItem),
    /// Inner attributes of the source file.
    TopLevel,

    Variant(FileItemTreeId<Variant>),
    Field(FieldParent, ItemTreeFieldId),
    Param(FileItemTreeId<Function>, ItemTreeParamId),
    TypeOrConstParamData(GenericModItem, LocalTypeOrConstParamId),
    LifetimeParamData(GenericModItem, LocalLifetimeParamId),
}

impl AttrOwner {
    pub fn make_field_indexed(parent: FieldParent, idx: usize) -> Self {
        AttrOwner::Field(parent, ItemTreeFieldId::from_raw(RawIdx::from_u32(idx as u32)))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum FieldParent {
    Struct(FileItemTreeId<Struct>),
    Union(FileItemTreeId<Union>),
    Variant(FileItemTreeId<Variant>),
}

pub type ItemTreeParamId = Idx<Param>;
pub type ItemTreeFieldId = Idx<Field>;

macro_rules! from_attrs {
    ( $( $var:ident($t:ty) ),+ $(,)? ) => {
        $(
            impl From<$t> for AttrOwner {
                fn from(t: $t) -> AttrOwner {
                    AttrOwner::$var(t)
                }
            }
        )+
    };
}

from_attrs!(ModItem(ModItem), Variant(FileItemTreeId<Variant>));

/// Trait implemented by all nodes in the item tree.
pub trait ItemTreeNode: Clone {
    type Source: AstIdNode;

    fn ast_id(&self) -> FileAstId<Self::Source>;

    /// Looks up an instance of `Self` in an item tree.
    fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self;
    fn attr_owner(id: FileItemTreeId<Self>) -> AttrOwner;
}
pub trait GenericsItemTreeNode: ItemTreeNode {
    fn generic_params(&self) -> &Arc<GenericParams>;
}

pub struct FileItemTreeId<N>(Idx<N>);

impl<N> FileItemTreeId<N> {
    pub fn range_iter(range: Range<Self>) -> impl Iterator<Item = Self> + Clone {
        (range.start.index().into_raw().into_u32()..range.end.index().into_raw().into_u32())
            .map(RawIdx::from_u32)
            .map(Idx::from_raw)
            .map(Self)
    }
}

impl<N> FileItemTreeId<N> {
    pub fn index(&self) -> Idx<N> {
        self.0
    }
}

impl<N> Clone for FileItemTreeId<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<N> Copy for FileItemTreeId<N> {}

impl<N> PartialEq for FileItemTreeId<N> {
    fn eq(&self, other: &FileItemTreeId<N>) -> bool {
        self.0 == other.0
    }
}
impl<N> Eq for FileItemTreeId<N> {}

impl<N> Hash for FileItemTreeId<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<N> fmt::Debug for FileItemTreeId<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
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

    pub fn item_tree(&self, db: &dyn DefDatabase) -> Arc<ItemTree> {
        match self.block {
            Some(block) => db.block_item_tree(block),
            None => db.file_item_tree(self.file),
        }
    }

    pub fn item_tree_with_source_map(
        &self,
        db: &dyn DefDatabase,
    ) -> (Arc<ItemTree>, Arc<ItemTreeSourceMaps>) {
        match self.block {
            Some(block) => db.block_item_tree_with_source_map(block),
            None => db.file_item_tree_with_source_map(self.file),
        }
    }

    pub fn file_id(self) -> HirFileId {
        self.file
    }

    pub fn is_block(self) -> bool {
        self.block.is_some()
    }
}

#[derive(Debug)]
pub struct ItemTreeId<N> {
    tree: TreeId,
    pub value: FileItemTreeId<N>,
}

impl<N> ItemTreeId<N> {
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

    pub fn item_tree_with_source_map(
        self,
        db: &dyn DefDatabase,
    ) -> (Arc<ItemTree>, Arc<ItemTreeSourceMaps>) {
        self.tree.item_tree_with_source_map(db)
    }

    pub fn resolved<R>(self, db: &dyn DefDatabase, cb: impl FnOnce(&N) -> R) -> R
    where
        ItemTree: Index<FileItemTreeId<N>, Output = N>,
    {
        cb(&self.tree.item_tree(db)[self.value])
    }
}

impl<N> Copy for ItemTreeId<N> {}
impl<N> Clone for ItemTreeId<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N> PartialEq for ItemTreeId<N> {
    fn eq(&self, other: &Self) -> bool {
        self.tree == other.tree && self.value == other.value
    }
}

impl<N> Eq for ItemTreeId<N> {}

impl<N> Hash for ItemTreeId<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tree.hash(state);
        self.value.hash(state);
    }
}

macro_rules! mod_items {
    ( $( $typ:ident $(<$generic_params:ident>)? in $fld:ident -> $ast:ty ),+ $(,)? ) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        pub enum ModItem {
            $(
                $typ(FileItemTreeId<$typ>),
            )+
        }

        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        pub enum GenericModItem {
            $(
                $(
                    #[cfg_attr(ignore_fragment, $generic_params)]
                    $typ(FileItemTreeId<$typ>),
                )?
            )+
        }

        impl ModItem {
            pub fn ast_id(&self, tree: &ItemTree) -> FileAstId<ast::Item> {
                match self {
                    $(ModItem::$typ(it) => tree[it.index()].ast_id().upcast()),+
                }
            }
        }

        impl GenericModItem {
            pub fn ast_id(&self, tree: &ItemTree) -> FileAstId<ast::AnyHasGenericParams> {
                match self {
                    $(
                        $(
                            #[cfg_attr(ignore_fragment, $generic_params)]
                            GenericModItem::$typ(it) => tree[it.index()].ast_id().upcast(),
                        )?
                    )+
                }
            }
        }

        impl From<GenericModItem> for ModItem {
            fn from(id: GenericModItem) -> ModItem {
                match id {
                    $(
                        $(
                            #[cfg_attr(ignore_fragment, $generic_params)]
                            GenericModItem::$typ(id) => ModItem::$typ(id),
                        )?
                    )+
                }
            }
        }

        impl From<GenericModItem> for AttrOwner {
            fn from(t: GenericModItem) -> AttrOwner {
                AttrOwner::ModItem(t.into())
            }
        }

        $(
            impl From<FileItemTreeId<$typ>> for ModItem {
                fn from(id: FileItemTreeId<$typ>) -> ModItem {
                    ModItem::$typ(id)
                }
            }
            $(
                #[cfg_attr(ignore_fragment, $generic_params)]
                impl From<FileItemTreeId<$typ>> for GenericModItem {
                    fn from(id: FileItemTreeId<$typ>) -> GenericModItem {
                        GenericModItem::$typ(id)
                    }
                }
            )?
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

                fn attr_owner(id: FileItemTreeId<Self>) -> AttrOwner {
                    AttrOwner::ModItem(ModItem::$typ(id))
                }
            }

            impl Index<Idx<$typ>> for ItemTree {
                type Output = $typ;

                fn index(&self, index: Idx<$typ>) -> &Self::Output {
                    &self.data().$fld[index]
                }
            }

            $(
                impl GenericsItemTreeNode for $typ {
                    fn generic_params(&self) -> &Arc<GenericParams> {
                        &self.$generic_params
                    }
                }
            )?
        )+
    };
}

mod_items! {
    Use in uses -> ast::Use,
    ExternCrate in extern_crates -> ast::ExternCrate,
    ExternBlock in extern_blocks -> ast::ExternBlock,
    Function<explicit_generic_params> in functions -> ast::Fn,
    Struct<generic_params> in structs -> ast::Struct,
    Union<generic_params> in unions -> ast::Union,
    Enum<generic_params> in enums -> ast::Enum,
    Const in consts -> ast::Const,
    Static in statics -> ast::Static,
    Trait<generic_params> in traits -> ast::Trait,
    TraitAlias<generic_params> in trait_aliases -> ast::TraitAlias,
    Impl<generic_params> in impls -> ast::Impl,
    TypeAlias<generic_params> in type_aliases -> ast::TypeAlias,
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
            _ => &self.data().vis.arena[Idx::from_raw(index.0.into())],
        }
    }
}

impl<N: ItemTreeNode> Index<FileItemTreeId<N>> for ItemTree {
    type Output = N;
    fn index(&self, id: FileItemTreeId<N>) -> &N {
        N::lookup(self, id.index())
    }
}

impl ItemTreeNode for Variant {
    type Source = ast::Variant;

    fn ast_id(&self) -> FileAstId<Self::Source> {
        self.ast_id
    }

    fn lookup(tree: &ItemTree, index: Idx<Self>) -> &Self {
        &tree.data().variants[index]
    }

    fn attr_owner(id: FileItemTreeId<Self>) -> AttrOwner {
        AttrOwner::Variant(id)
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
    pub abi: Option<Symbol>,
    pub ast_id: FileAstId<ast::ExternBlock>,
    pub children: Box<[ModItem]>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub explicit_generic_params: Arc<GenericParams>,
    pub abi: Option<Symbol>,
    pub params: Box<[Param]>,
    pub ret_type: TypeRefId,
    pub ast_id: FileAstId<ast::Fn>,
    pub types_map: Arc<TypesMap>,
    pub(crate) flags: FnFlags,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub type_ref: Option<TypeRefId>,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub(crate) struct FnFlags: u16 {
        const HAS_SELF_PARAM = 1 << 0;
        const HAS_BODY = 1 << 1;
        const HAS_DEFAULT_KW = 1 << 2;
        const HAS_CONST_KW = 1 << 3;
        const HAS_ASYNC_KW = 1 << 4;
        const HAS_UNSAFE_KW = 1 << 5;
        const IS_VARARGS = 1 << 6;
        const HAS_SAFE_KW = 1 << 7;
        /// The `#[target_feature]` attribute is necessary to check safety (with RFC 2396),
        /// but keeping it for all functions will consume a lot of memory when there are
        /// only very few functions with it. So we only encode its existence here, and lookup
        /// it if needed.
        const HAS_TARGET_FEATURE = 1 << 8;
        const DEPRECATED_SAFE_2024 = 1 << 9;
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Struct {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Arc<GenericParams>,
    pub fields: Box<[Field]>,
    pub shape: FieldsShape,
    pub ast_id: FileAstId<ast::Struct>,
    pub types_map: Arc<TypesMap>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Union {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Arc<GenericParams>,
    pub fields: Box<[Field]>,
    pub ast_id: FileAstId<ast::Union>,
    pub types_map: Arc<TypesMap>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Enum {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Arc<GenericParams>,
    pub variants: Range<FileItemTreeId<Variant>>,
    pub ast_id: FileAstId<ast::Enum>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub name: Name,
    pub fields: Box<[Field]>,
    pub shape: FieldsShape,
    pub ast_id: FileAstId<ast::Variant>,
    pub types_map: Arc<TypesMap>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FieldsShape {
    Record,
    Tuple,
    Unit,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: Name,
    pub type_ref: TypeRefId,
    pub visibility: RawVisibilityId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Const {
    /// `None` for `const _: () = ();`
    pub name: Option<Name>,
    pub visibility: RawVisibilityId,
    pub type_ref: TypeRefId,
    pub ast_id: FileAstId<ast::Const>,
    pub has_body: bool,
    pub types_map: Arc<TypesMap>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Static {
    pub name: Name,
    pub visibility: RawVisibilityId,
    // TODO: use bitflags when we have more flags
    pub mutable: bool,
    pub has_safe_kw: bool,
    pub has_unsafe_kw: bool,
    pub type_ref: TypeRefId,
    pub ast_id: FileAstId<ast::Static>,
    pub types_map: Arc<TypesMap>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Trait {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Arc<GenericParams>,
    pub is_auto: bool,
    pub is_unsafe: bool,
    pub items: Box<[AssocItem]>,
    pub ast_id: FileAstId<ast::Trait>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TraitAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    pub generic_params: Arc<GenericParams>,
    pub ast_id: FileAstId<ast::TraitAlias>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Impl {
    pub generic_params: Arc<GenericParams>,
    pub target_trait: Option<TraitRef>,
    pub self_ty: TypeRefId,
    pub is_negative: bool,
    pub is_unsafe: bool,
    pub items: Box<[AssocItem]>,
    pub ast_id: FileAstId<ast::Impl>,
    pub types_map: Arc<TypesMap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: Name,
    pub visibility: RawVisibilityId,
    /// Bounds on the type alias itself. Only valid in trait declarations, eg. `type Assoc: Copy;`.
    pub bounds: Box<[TypeBound]>,
    pub generic_params: Arc<GenericParams>,
    pub type_ref: Option<TypeRefId>,
    pub ast_id: FileAstId<ast::TypeAlias>,
    pub types_map: Arc<TypesMap>,
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
    pub ctxt: SyntaxContextId,
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

impl Use {
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
        let (_, source_map) = lower::lower_use_tree(db, ast_use_tree, &mut |range| {
            db.span_map(file_id).span_for_range(range).ctx
        })
        .expect("failed to lower use tree");
        source_map[index].clone()
    }

    /// Maps a `UseTree` contained in this import back to its AST node.
    pub fn use_tree_source_map(
        &self,
        db: &dyn DefDatabase,
        file_id: HirFileId,
    ) -> Arena<ast::UseTree> {
        // Re-lower the AST item and get the source map.
        // Note: The AST unwraps are fine, since if they fail we should have never obtained `index`.
        let ast = InFile::new(file_id, self.ast_id).to_node(db.upcast());
        let ast_use_tree = ast.use_tree().expect("missing `use_tree`");
        lower::lower_use_tree(db, ast_use_tree, &mut |range| {
            db.span_map(file_id).span_for_range(range).ctx
        })
        .expect("failed to lower use tree")
        .1
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

    /// The [`UseTreeKind`] of this `UseTree`.
    pub fn kind(&self) -> &UseTreeKind {
        &self.kind
    }

    fn expand_impl(
        &self,
        prefix: Option<ModPath>,
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
            ModItem::Use(_)
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
            | ModItem::Macro2(_) => None,
            &ModItem::MacroCall(call) => Some(AssocItem::MacroCall(call)),
            &ModItem::Const(konst) => Some(AssocItem::Const(konst)),
            &ModItem::TypeAlias(alias) => Some(AssocItem::TypeAlias(alias)),
            &ModItem::Function(func) => Some(AssocItem::Function(func)),
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
