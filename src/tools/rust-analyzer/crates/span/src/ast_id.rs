//! `AstIdMap` allows to create stable IDs for "large" syntax nodes like items
//! and macro calls.
//!
//! Specifically, it enumerates all items in a file and uses position of a an
//! item as an ID. That way, id's don't change unless the set of items itself
//! changes.
//!
//! These IDs are tricky. If one of them invalidates, its interned ID invalidates,
//! and this can cause *a lot* to be recomputed. For example, if you invalidate the ID
//! of a struct, and that struct has an impl (any impl!) this will cause the `Self`
//! type of the impl to invalidate, which will cause the all impls queries to be
//! invalidated, which will cause every trait solve query in this crate *and* all
//! transitive reverse dependencies to be invalidated, which is pretty much the worst
//! thing that can happen incrementality wise.
//!
//! So we want these IDs to stay as stable as possible. For top-level items, we store
//! their kind and name, which should be unique, but since they can still not be, we
//! also store an index disambiguator. For nested items, we also store the ID of their
//! parent. For macro calls, we store the macro name and an index. There aren't usually
//! a lot of macro calls in item position, and invalidation in bodies is not much of
//! a problem, so this should be enough.

use std::{
    any::type_name,
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
};

use la_arena::{Arena, Idx, RawIdx};
use rustc_hash::{FxBuildHasher, FxHashMap};
use syntax::{
    AstNode, AstPtr, SyntaxKind, SyntaxNode, SyntaxNodePtr,
    ast::{self, HasName},
    match_ast,
};

// The first index is always the root node's AstId
/// The root ast id always points to the encompassing file, using this in spans is discouraged as
/// any range relative to it will be effectively absolute, ruining the entire point of anchored
/// relative text ranges.
pub const ROOT_ERASED_FILE_AST_ID: ErasedFileAstId =
    ErasedFileAstId(pack_hash_index_and_kind(0, 0, ErasedFileAstIdKind::Root as u32));

/// ErasedFileAstId used as the span for syntax node fixups. Any Span containing this file id is to be
/// considered fake.
pub const FIXUP_ERASED_FILE_AST_ID_MARKER: ErasedFileAstId =
    ErasedFileAstId(pack_hash_index_and_kind(0, 0, ErasedFileAstIdKind::Fixup as u32));

/// This is a type erased FileAstId.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErasedFileAstId(u32);

impl fmt::Debug for ErasedFileAstId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = self.kind();
        macro_rules! kind {
            ($($kind:ident),* $(,)?) => {
                if false {
                    // Ensure we covered all variants.
                    match ErasedFileAstIdKind::Root {
                        $( ErasedFileAstIdKind::$kind => {} )*
                    }
                    unreachable!()
                }
                $( else if kind == ErasedFileAstIdKind::$kind as u32 {
                    stringify!($kind)
                } )*
                else {
                    "Unknown"
                }
            };
        }
        let kind = kind!(
            Root,
            Enum,
            Struct,
            Union,
            ExternCrate,
            MacroDef,
            MacroRules,
            Module,
            Static,
            Trait,
            TraitAlias,
            Variant,
            Const,
            Fn,
            MacroCall,
            TypeAlias,
            ExternBlock,
            Use,
            Impl,
            BlockExpr,
            AsmExpr,
            Fixup,
        );
        if f.alternate() {
            write!(f, "{kind}[{:04X}, {}]", self.hash_value(), self.index())
        } else {
            f.debug_struct("ErasedFileAstId")
                .field("kind", &format_args!("{kind}"))
                .field("index", &self.index())
                .field("hash", &format_args!("{:04X}", self.hash_value()))
                .finish()
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
enum ErasedFileAstIdKind {
    /// This needs to not change because it's depended upon by the proc macro server.
    Fixup = 0,
    // The following are associated with `ErasedHasNameFileAstId`.
    Enum,
    Struct,
    Union,
    ExternCrate,
    MacroDef,
    MacroRules,
    Module,
    Static,
    Trait,
    TraitAlias,
    // Until here associated with `ErasedHasNameFileAstId`.
    // The following are associated with `ErasedAssocItemFileAstId`.
    Variant,
    Const,
    Fn,
    MacroCall,
    TypeAlias,
    // Until here associated with `ErasedAssocItemFileAstId`.
    // Extern blocks don't really have any identifying property unfortunately.
    ExternBlock,
    // FIXME: If we store the final `UseTree` instead of the top-level `Use`, we can store its name,
    // and be way more granular for incrementality, at the expense of increased memory usage.
    // Use IDs aren't used a lot. The main thing that stores them is the def map. So everything that
    // uses the def map will be invalidated. That includes infers, and so is pretty bad, but our
    // def map incrementality story is pretty bad anyway and needs to be improved (see
    // https://rust-lang.zulipchat.com/#narrow/channel/185405-t-compiler.2Frust-analyzer/topic/.60infer.60.20queries.20and.20splitting.20.60DefMap.60).
    // So I left this as-is for now, as the def map improvement should also mitigate this.
    Use,
    /// Associated with [`ImplFileAstId`].
    Impl,
    /// Associated with [`BlockExprFileAstId`].
    BlockExpr,
    // `global_asm!()` is an item, so we need to give it an `AstId`. So we give to all inline asm
    // because incrementality is not a problem, they will always be the only item in the macro file,
    // and memory usage also not because they're rare.
    AsmExpr,
    /// Keep this last.
    Root,
}

// First hash, then index, then kind.
const HASH_BITS: u32 = 16;
const INDEX_BITS: u32 = 11;
const KIND_BITS: u32 = 5;
const _: () = assert!(ErasedFileAstIdKind::Fixup as u32 <= ((1 << KIND_BITS) - 1));
const _: () = assert!(HASH_BITS + INDEX_BITS + KIND_BITS == u32::BITS);

#[inline]
const fn u16_hash(hash: u64) -> u16 {
    // We do basically the same as `FxHasher`. We don't use rustc-hash and truncate because the
    // higher bits have more entropy, but unlike rustc-hash we don't rotate because it rotates
    // for hashmaps that just use the low bits, but we compare all bits.
    const K: u16 = 0xecc5;
    let (part1, part2, part3, part4) =
        (hash as u16, (hash >> 16) as u16, (hash >> 32) as u16, (hash >> 48) as u16);
    part1
        .wrapping_add(part2)
        .wrapping_mul(K)
        .wrapping_add(part3)
        .wrapping_mul(K)
        .wrapping_add(part4)
        .wrapping_mul(K)
}

#[inline]
const fn pack_hash_index_and_kind(hash: u16, index: u32, kind: u32) -> u32 {
    (hash as u32) | (index << HASH_BITS) | (kind << (HASH_BITS + INDEX_BITS))
}

impl ErasedFileAstId {
    #[inline]
    fn hash_value(self) -> u16 {
        self.0 as u16
    }

    #[inline]
    fn index(self) -> u32 {
        (self.0 << KIND_BITS) >> (HASH_BITS + KIND_BITS)
    }

    #[inline]
    fn kind(self) -> u32 {
        self.0 >> (HASH_BITS + INDEX_BITS)
    }

    fn ast_id_for(
        node: &SyntaxNode,
        index_map: &mut ErasedAstIdNextIndexMap,
        parent: Option<&ErasedFileAstId>,
    ) -> Option<ErasedFileAstId> {
        // Blocks are deliberately not here - we only want to allocate a block if it contains items.
        has_name_ast_id(node, index_map)
            .or_else(|| assoc_item_ast_id(node, index_map, parent))
            .or_else(|| extern_block_ast_id(node, index_map))
            .or_else(|| use_ast_id(node, index_map))
            .or_else(|| impl_ast_id(node, index_map))
            .or_else(|| asm_expr_ast_id(node, index_map))
    }

    fn should_alloc(node: &SyntaxNode) -> bool {
        let kind = node.kind();
        should_alloc_has_name(kind)
            || should_alloc_assoc_item(kind)
            || ast::ExternBlock::can_cast(kind)
            || ast::Use::can_cast(kind)
            || ast::Impl::can_cast(kind)
            || ast::AsmExpr::can_cast(kind)
    }

    #[inline]
    pub fn into_raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

pub trait AstIdNode: AstNode {}

/// `AstId` points to an AST node in a specific file.
pub struct FileAstId<N> {
    raw: ErasedFileAstId,
    _marker: PhantomData<fn() -> N>,
}

/// Traits are manually implemented because `derive` adds redundant bounds.
impl<N> Clone for FileAstId<N> {
    #[inline]
    fn clone(&self) -> FileAstId<N> {
        *self
    }
}
impl<N> Copy for FileAstId<N> {}

impl<N> PartialEq for FileAstId<N> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<N> Eq for FileAstId<N> {}
impl<N> Hash for FileAstId<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.raw.hash(hasher);
    }
}

impl<N> fmt::Debug for FileAstId<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FileAstId::<{}>({:?})", type_name::<N>(), self.raw)
    }
}

impl<N> FileAstId<N> {
    // Can't make this a From implementation because of coherence
    #[inline]
    pub fn upcast<M: AstIdNode>(self) -> FileAstId<M>
    where
        N: Into<M>,
    {
        FileAstId { raw: self.raw, _marker: PhantomData }
    }

    #[inline]
    pub fn erase(self) -> ErasedFileAstId {
        self.raw
    }
}

#[derive(Hash)]
struct ErasedHasNameFileAstId<'a> {
    name: &'a str,
}

/// This holds the ast ID for variants too (they're a kind of assoc item).
#[derive(Hash)]
struct ErasedAssocItemFileAstId<'a> {
    /// Subtle: items in `extern` blocks **do not** store the ID of the extern block here.
    /// Instead this is left empty. The reason is that `ExternBlockFileAstId` is pretty unstable
    /// (it contains only an index), and extern blocks don't introduce a new scope, so storing
    /// the extern block ID will do more harm to incrementality than help.
    parent: Option<ErasedFileAstId>,
    properties: ErasedHasNameFileAstId<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ImplFileAstId<'a> {
    /// This can be `None` if the `Self` type is not a named type, or if it is inside a macro call.
    self_ty_name: Option<&'a str>,
    /// This can be `None` if this is an inherent impl, or if the trait name is inside a macro call.
    trait_name: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BlockExprFileAstId {
    parent: Option<ErasedFileAstId>,
}

impl AstIdNode for ast::ExternBlock {}

fn extern_block_ast_id(
    node: &SyntaxNode,
    index_map: &mut ErasedAstIdNextIndexMap,
) -> Option<ErasedFileAstId> {
    if ast::ExternBlock::can_cast(node.kind()) {
        Some(index_map.new_id(ErasedFileAstIdKind::ExternBlock, ()))
    } else {
        None
    }
}

impl AstIdNode for ast::Use {}

fn use_ast_id(
    node: &SyntaxNode,
    index_map: &mut ErasedAstIdNextIndexMap,
) -> Option<ErasedFileAstId> {
    if ast::Use::can_cast(node.kind()) {
        Some(index_map.new_id(ErasedFileAstIdKind::Use, ()))
    } else {
        None
    }
}

impl AstIdNode for ast::AsmExpr {}

fn asm_expr_ast_id(
    node: &SyntaxNode,
    index_map: &mut ErasedAstIdNextIndexMap,
) -> Option<ErasedFileAstId> {
    if ast::AsmExpr::can_cast(node.kind()) {
        Some(index_map.new_id(ErasedFileAstIdKind::AsmExpr, ()))
    } else {
        None
    }
}

impl AstIdNode for ast::Impl {}

fn impl_ast_id(
    node: &SyntaxNode,
    index_map: &mut ErasedAstIdNextIndexMap,
) -> Option<ErasedFileAstId> {
    if let Some(node) = ast::Impl::cast(node.clone()) {
        let type_as_name = |ty: Option<ast::Type>| match ty? {
            ast::Type::PathType(it) => Some(it.path()?.segment()?.name_ref()?),
            _ => None,
        };
        let self_ty_name = type_as_name(node.self_ty());
        let trait_name = type_as_name(node.trait_());
        let data = ImplFileAstId {
            self_ty_name: self_ty_name.as_ref().map(|it| it.text_non_mutable()),
            trait_name: trait_name.as_ref().map(|it| it.text_non_mutable()),
        };
        Some(index_map.new_id(ErasedFileAstIdKind::Impl, data))
    } else {
        None
    }
}

// Blocks aren't `AstIdNode`s deliberately, because unlike other nodes, not all blocks get their own
// ast id, only if they have items. To account for that we have a different, fallible, API for blocks.
// impl !AstIdNode for ast::BlockExpr {}

fn block_expr_ast_id(
    node: &SyntaxNode,
    index_map: &mut ErasedAstIdNextIndexMap,
    parent: Option<&ErasedFileAstId>,
) -> Option<ErasedFileAstId> {
    if ast::BlockExpr::can_cast(node.kind()) {
        Some(
            index_map.new_id(
                ErasedFileAstIdKind::BlockExpr,
                BlockExprFileAstId { parent: parent.copied() },
            ),
        )
    } else {
        None
    }
}

#[derive(Default)]
struct ErasedAstIdNextIndexMap(FxHashMap<(ErasedFileAstIdKind, u16), u32>);

impl ErasedAstIdNextIndexMap {
    #[inline]
    fn new_id(&mut self, kind: ErasedFileAstIdKind, data: impl Hash) -> ErasedFileAstId {
        let hash = FxBuildHasher.hash_one(&data);
        let initial_hash = u16_hash(hash);
        // Even though 2^INDEX_BITS=2048 items with the same hash seems like a lot,
        // it could happen with macro calls or `use`s in macro-generated files. So we want
        // to handle it gracefully. We just increment the hash.
        let mut hash = initial_hash;
        let index = loop {
            match self.0.entry((kind, hash)) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let i = entry.get_mut();
                    if *i < ((1 << INDEX_BITS) - 1) {
                        *i += 1;
                        break *i;
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(0);
                    break 0;
                }
            }
            hash = hash.wrapping_add(1);
            if hash == initial_hash {
                // That's 2^27=134,217,728 items!
                panic!("you have way too many items in the same file!");
            }
        };
        let kind = kind as u32;
        ErasedFileAstId(pack_hash_index_and_kind(hash, index, kind))
    }
}

macro_rules! register_enum_ast_id {
    (impl $AstIdNode:ident for $($ident:ident),+ ) => {
        $(
            impl $AstIdNode for ast::$ident {}
        )+
    };
}
register_enum_ast_id! {
    impl AstIdNode for
    Item, AnyHasGenericParams, Adt, Macro,
    AssocItem
}

macro_rules! register_has_name_ast_id {
    (impl $AstIdNode:ident for $($ident:ident = $name_method:ident),+ ) => {
        $(
            impl $AstIdNode for ast::$ident {}
        )+

        fn has_name_ast_id(node: &SyntaxNode, index_map: &mut ErasedAstIdNextIndexMap) -> Option<ErasedFileAstId> {
            match_ast! {
                match node {
                    $(
                        ast::$ident(node) => {
                            let name = node.$name_method();
                            let name = name.as_ref().map_or("", |it| it.text_non_mutable());
                            let result = ErasedHasNameFileAstId {
                                name,
                            };
                            Some(index_map.new_id(ErasedFileAstIdKind::$ident, result))
                        },
                    )*
                    _ => None,
                }
            }
        }

        fn should_alloc_has_name(kind: SyntaxKind) -> bool {
            false $( || ast::$ident::can_cast(kind) )*
        }
    };
}
register_has_name_ast_id! {
    impl AstIdNode for
        Enum = name,
        Struct = name,
        Union = name,
        ExternCrate = name_ref,
        MacroDef = name,
        MacroRules = name,
        Module = name,
        Static = name,
        Trait = name,
        TraitAlias = name
}

macro_rules! register_assoc_item_ast_id {
    (impl $AstIdNode:ident for $($ident:ident = $name_callback:expr),+ ) => {
        $(
            impl $AstIdNode for ast::$ident {}
        )+

        fn assoc_item_ast_id(
            node: &SyntaxNode,
            index_map: &mut ErasedAstIdNextIndexMap,
            parent: Option<&ErasedFileAstId>,
        ) -> Option<ErasedFileAstId> {
            match_ast! {
                match node {
                    $(
                        ast::$ident(node) => {
                            let name = $name_callback(node);
                            let name = name.as_ref().map_or("", |it| it.text_non_mutable());
                            let properties = ErasedHasNameFileAstId {
                                name,
                            };
                            let result = ErasedAssocItemFileAstId {
                                parent: parent.copied(),
                                properties,
                            };
                            Some(index_map.new_id(ErasedFileAstIdKind::$ident, result))
                        },
                    )*
                    _ => None,
                }
            }
        }

        fn should_alloc_assoc_item(kind: SyntaxKind) -> bool {
            false $( || ast::$ident::can_cast(kind) )*
        }
    };
}
register_assoc_item_ast_id! {
    impl AstIdNode for
    Variant = |it: ast::Variant| it.name(),
    Const = |it: ast::Const| it.name(),
    Fn = |it: ast::Fn| it.name(),
    MacroCall = |it: ast::MacroCall| it.path().and_then(|path| path.segment()?.name_ref()),
    TypeAlias = |it: ast::TypeAlias| it.name()
}

/// Maps items' `SyntaxNode`s to `ErasedFileAstId`s and back.
#[derive(Default)]
pub struct AstIdMap {
    /// An arena of the ptrs and their associated ID.
    arena: Arena<(SyntaxNodePtr, ErasedFileAstId)>,
    /// Map ptr to id.
    ptr_map: hashbrown::HashTable<ArenaId>,
    /// Map id to ptr.
    id_map: hashbrown::HashTable<ArenaId>,
}

type ArenaId = Idx<(SyntaxNodePtr, ErasedFileAstId)>;

impl fmt::Debug for AstIdMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstIdMap").field("arena", &self.arena).finish()
    }
}

impl PartialEq for AstIdMap {
    fn eq(&self, other: &Self) -> bool {
        self.arena == other.arena
    }
}
impl Eq for AstIdMap {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContainsItems {
    Yes,
    No,
}

impl AstIdMap {
    pub fn from_source(node: &SyntaxNode) -> AstIdMap {
        assert!(node.parent().is_none());
        let mut res = AstIdMap::default();
        let mut index_map = ErasedAstIdNextIndexMap::default();

        // Ensure we allocate the root.
        res.arena.alloc((SyntaxNodePtr::new(node), ROOT_ERASED_FILE_AST_ID));

        // By walking the tree in breadth-first order we make sure that parents
        // get lower ids then children. That is, adding a new child does not
        // change parent's id. This means that, say, adding a new function to a
        // trait does not change ids of top-level items, which helps caching.

        // This contains the stack of the `BlockExpr`s we are under. We do this
        // so we only allocate `BlockExpr`s if they contain items.
        // The general idea is: when we enter a block we push `(block, false)` here.
        // Items inside the block are attributed to the block's container, not the block.
        // For the first item we find inside a block, we make this `(block, true)`
        // and create an ast id for the block. When exiting the block we pop it,
        // whether or not we created an ast id for it.
        // It may seem that with this setup we will generate an ID for blocks that
        // have no items directly but have items inside other items inside them.
        // This is true, but it doesn't matter, because such blocks can't exist.
        // After all, the block will then contain the *outer* item, so we allocate
        // an ID for it anyway.
        let mut blocks = Vec::new();
        let mut curr_layer = vec![(node.clone(), None)];
        let mut next_layer = vec![];
        while !curr_layer.is_empty() {
            curr_layer.drain(..).for_each(|(node, parent_idx)| {
                let mut preorder = node.preorder();
                while let Some(event) = preorder.next() {
                    match event {
                        syntax::WalkEvent::Enter(node) => {
                            if ast::BlockExpr::can_cast(node.kind()) {
                                blocks.push((node, ContainsItems::No));
                            } else if ErasedFileAstId::should_alloc(&node) {
                                // Allocate blocks on-demand, only if they have items.
                                // We don't associate items with blocks, only with items, since block IDs can be quite unstable.
                                // FIXME: Is this the correct thing to do? Macro calls might actually be more incremental if
                                // associated with blocks (not sure). Either way it's not a big deal.
                                if let Some((
                                    last_block_node,
                                    already_allocated @ ContainsItems::No,
                                )) = blocks.last_mut()
                                {
                                    let block_ast_id = block_expr_ast_id(
                                        last_block_node,
                                        &mut index_map,
                                        parent_of(parent_idx, &res),
                                    )
                                    .expect("not a BlockExpr");
                                    res.arena
                                        .alloc((SyntaxNodePtr::new(last_block_node), block_ast_id));
                                    *already_allocated = ContainsItems::Yes;
                                }

                                let parent = parent_of(parent_idx, &res);
                                let ast_id =
                                    ErasedFileAstId::ast_id_for(&node, &mut index_map, parent)
                                        .expect("this node should have an ast id");
                                let idx = res.arena.alloc((SyntaxNodePtr::new(&node), ast_id));

                                next_layer.extend(node.children().map(|child| (child, Some(idx))));
                                preorder.skip_subtree();
                            }
                        }
                        syntax::WalkEvent::Leave(node) => {
                            if ast::BlockExpr::can_cast(node.kind()) {
                                assert_eq!(
                                    blocks.pop().map(|it| it.0),
                                    Some(node),
                                    "left a BlockExpr we never entered"
                                );
                            }
                        }
                    }
                }
            });
            std::mem::swap(&mut curr_layer, &mut next_layer);
            assert!(blocks.is_empty(), "didn't leave all BlockExprs");
        }

        res.ptr_map = hashbrown::HashTable::with_capacity(res.arena.len());
        res.id_map = hashbrown::HashTable::with_capacity(res.arena.len());
        for (idx, (ptr, ast_id)) in res.arena.iter() {
            let ptr_hash = hash_ptr(ptr);
            let ast_id_hash = hash_ast_id(ast_id);
            match res.ptr_map.entry(
                ptr_hash,
                |idx2| *idx2 == idx,
                |&idx| hash_ptr(&res.arena[idx].0),
            ) {
                hashbrown::hash_table::Entry::Occupied(_) => unreachable!(),
                hashbrown::hash_table::Entry::Vacant(entry) => {
                    entry.insert(idx);
                }
            }
            match res.id_map.entry(
                ast_id_hash,
                |idx2| *idx2 == idx,
                |&idx| hash_ast_id(&res.arena[idx].1),
            ) {
                hashbrown::hash_table::Entry::Occupied(_) => unreachable!(),
                hashbrown::hash_table::Entry::Vacant(entry) => {
                    entry.insert(idx);
                }
            }
        }
        res.arena.shrink_to_fit();
        return res;

        fn parent_of(parent_idx: Option<ArenaId>, res: &AstIdMap) -> Option<&ErasedFileAstId> {
            let mut parent = parent_idx.map(|parent_idx| &res.arena[parent_idx].1);
            if parent.is_some_and(|parent| parent.kind() == ErasedFileAstIdKind::ExternBlock as u32)
            {
                // See the comment on `ErasedAssocItemFileAstId` for why is this.
                // FIXME: Technically there could be an extern block inside another item, e.g.:
                // ```
                // fn foo() {
                //     extern "C" {
                //         fn bar();
                //     }
                // }
                // ```
                // Here we want to make `foo()` the parent of `bar()`, but we make it `None`.
                // Shouldn't be a big deal though.
                parent = None;
            }
            parent
        }
    }

    /// The [`AstId`] of the root node
    pub fn root(&self) -> SyntaxNodePtr {
        self.arena[Idx::from_raw(RawIdx::from_u32(0))].0
    }

    pub fn ast_id<N: AstIdNode>(&self, item: &N) -> FileAstId<N> {
        self.ast_id_for_ptr(AstPtr::new(item))
    }

    /// Blocks may not be allocated (if they have no items), so they have a different API.
    pub fn ast_id_for_block(&self, block: &ast::BlockExpr) -> Option<FileAstId<ast::BlockExpr>> {
        self.ast_id_for_ptr_for_block(AstPtr::new(block))
    }

    pub fn ast_id_for_ptr<N: AstIdNode>(&self, ptr: AstPtr<N>) -> FileAstId<N> {
        let ptr = ptr.syntax_node_ptr();
        FileAstId { raw: self.erased_ast_id(ptr), _marker: PhantomData }
    }

    /// Blocks may not be allocated (if they have no items), so they have a different API.
    pub fn ast_id_for_ptr_for_block(
        &self,
        ptr: AstPtr<ast::BlockExpr>,
    ) -> Option<FileAstId<ast::BlockExpr>> {
        let ptr = ptr.syntax_node_ptr();
        self.try_erased_ast_id(ptr).map(|raw| FileAstId { raw, _marker: PhantomData })
    }

    fn erased_ast_id(&self, ptr: SyntaxNodePtr) -> ErasedFileAstId {
        self.try_erased_ast_id(ptr).unwrap_or_else(|| {
            panic!(
                "Can't find SyntaxNodePtr {:?} in AstIdMap:\n{:?}",
                ptr,
                self.arena.iter().map(|(_id, i)| i).collect::<Vec<_>>(),
            )
        })
    }

    fn try_erased_ast_id(&self, ptr: SyntaxNodePtr) -> Option<ErasedFileAstId> {
        let hash = hash_ptr(&ptr);
        let idx = *self.ptr_map.find(hash, |&idx| self.arena[idx].0 == ptr)?;
        Some(self.arena[idx].1)
    }

    // Don't bound on `AstIdNode` here, because `BlockExpr`s are also valid here (`ast::BlockExpr`
    // doesn't always have a matching `FileAstId`, but a `FileAstId<ast::BlockExpr>` always has
    // a matching node).
    pub fn get<N: AstNode>(&self, id: FileAstId<N>) -> AstPtr<N> {
        let ptr = self.get_erased(id.raw);
        AstPtr::try_from_raw(ptr)
            .unwrap_or_else(|| panic!("AstIdMap node mismatch with node `{ptr:?}`"))
    }

    pub fn get_erased(&self, id: ErasedFileAstId) -> SyntaxNodePtr {
        let hash = hash_ast_id(&id);
        match self.id_map.find(hash, |&idx| self.arena[idx].1 == id) {
            Some(&idx) => self.arena[idx].0,
            None => panic!(
                "Can't find ast id {:?} in AstIdMap:\n{:?}",
                id,
                self.arena.iter().map(|(_id, i)| i).collect::<Vec<_>>(),
            ),
        }
    }
}

#[inline]
fn hash_ptr(ptr: &SyntaxNodePtr) -> u64 {
    FxBuildHasher.hash_one(ptr)
}

#[inline]
fn hash_ast_id(ptr: &ErasedFileAstId) -> u64 {
    FxBuildHasher.hash_one(ptr)
}

#[cfg(test)]
mod tests {
    use syntax::{AstNode, Edition, SourceFile, SyntaxKind, SyntaxNodePtr, WalkEvent, ast};

    use crate::AstIdMap;

    #[test]
    fn check_all_nodes() {
        let syntax = SourceFile::parse(
            r#"
extern crate foo;
fn foo() {
    union U {}
}
struct S;
macro_rules! m {}
macro m2() {}
trait Trait {}
impl Trait for S {}
impl S {}
impl m!() {}
impl m2!() for m!() {}
type T = i32;
enum E {
    V1(),
    V2 {},
    V3,
}
struct S; // duplicate
extern "C" {
    static S: i32;
}
static mut S: i32 = 0;
const FOO: i32 = 0;
        "#,
            Edition::CURRENT,
        )
        .syntax_node();
        let ast_id_map = AstIdMap::from_source(&syntax);
        for node in syntax.preorder() {
            let WalkEvent::Enter(node) = node else { continue };
            if !matches!(
                node.kind(),
                SyntaxKind::EXTERN_CRATE
                    | SyntaxKind::FN
                    | SyntaxKind::UNION
                    | SyntaxKind::STRUCT
                    | SyntaxKind::MACRO_RULES
                    | SyntaxKind::MACRO_DEF
                    | SyntaxKind::MACRO_CALL
                    | SyntaxKind::TRAIT
                    | SyntaxKind::IMPL
                    | SyntaxKind::TYPE_ALIAS
                    | SyntaxKind::ENUM
                    | SyntaxKind::VARIANT
                    | SyntaxKind::EXTERN_BLOCK
                    | SyntaxKind::STATIC
                    | SyntaxKind::CONST
            ) {
                continue;
            }
            let ptr = SyntaxNodePtr::new(&node);
            let ast_id = ast_id_map.erased_ast_id(ptr);
            let turn_back = ast_id_map.get_erased(ast_id);
            assert_eq!(ptr, turn_back);
        }
    }

    #[test]
    fn different_names_get_different_hashes() {
        let syntax = SourceFile::parse(
            r#"
fn foo() {}
fn bar() {}
        "#,
            Edition::CURRENT,
        )
        .syntax_node();
        let ast_id_map = AstIdMap::from_source(&syntax);
        let fns = syntax.descendants().filter_map(ast::Fn::cast).collect::<Vec<_>>();
        let [foo_fn, bar_fn] = fns.as_slice() else {
            panic!("not exactly 2 functions");
        };
        let foo_fn_id = ast_id_map.ast_id(foo_fn);
        let bar_fn_id = ast_id_map.ast_id(bar_fn);
        assert_ne!(foo_fn_id.raw.hash_value(), bar_fn_id.raw.hash_value(), "hashes are equal");
    }

    #[test]
    fn different_parents_get_different_hashes() {
        let syntax = SourceFile::parse(
            r#"
fn foo() {
    m!();
}
fn bar() {
    m!();
}
        "#,
            Edition::CURRENT,
        )
        .syntax_node();
        let ast_id_map = AstIdMap::from_source(&syntax);
        let macro_calls = syntax.descendants().filter_map(ast::MacroCall::cast).collect::<Vec<_>>();
        let [macro_call_foo, macro_call_bar] = macro_calls.as_slice() else {
            panic!("not exactly 2 macro calls");
        };
        let macro_call_foo_id = ast_id_map.ast_id(macro_call_foo);
        let macro_call_bar_id = ast_id_map.ast_id(macro_call_bar);
        assert_ne!(
            macro_call_foo_id.raw.hash_value(),
            macro_call_bar_id.raw.hash_value(),
            "hashes are equal"
        );
    }

    #[test]
    fn blocks_with_no_items_have_no_id() {
        let syntax = SourceFile::parse(
            r#"
fn foo() {
    let foo = 1;
    bar(foo);
}
        "#,
            Edition::CURRENT,
        )
        .syntax_node();
        let ast_id_map = AstIdMap::from_source(&syntax);
        let block = syntax.descendants().find_map(ast::BlockExpr::cast).expect("no block");
        assert!(ast_id_map.ast_id_for_block(&block).is_none());
    }
}
