use std::{
    marker::PhantomData,
    hash::{Hash, Hasher},
    sync::Arc,
};

use ra_db::{LocationIntener, FileId};
use ra_syntax::{TreeArc, SyntaxNode, SourceFile, AstNode, SyntaxNodePtr, ast};
use ra_arena::{Arena, RawId, ArenaId, impl_arena_id};

use crate::{
    Module,
    PersistentHirDatabase,
};

#[derive(Debug, Default)]
pub struct HirInterner {
    macros: LocationIntener<MacroCallLoc, MacroCallId>,
    fns: LocationIntener<ItemLoc<ast::FnDef>, FunctionId>,
    structs: LocationIntener<ItemLoc<ast::StructDef>, StructId>,
    enums: LocationIntener<ItemLoc<ast::EnumDef>, EnumId>,
    consts: LocationIntener<ItemLoc<ast::ConstDef>, ConstId>,
    statics: LocationIntener<ItemLoc<ast::StaticDef>, StaticId>,
    traits: LocationIntener<ItemLoc<ast::TraitDef>, TraitId>,
    types: LocationIntener<ItemLoc<ast::TypeAliasDef>, TypeId>,
}

impl HirInterner {
    pub fn len(&self) -> usize {
        self.macros.len()
            + self.fns.len()
            + self.structs.len()
            + self.enums.len()
            + self.consts.len()
            + self.statics.len()
            + self.traits.len()
            + self.types.len()
    }
}

/// hir makes heavy use of ids: integer (u32) handlers to various things. You
/// can think of id as a pointer (but without a lifetime) or a file descriptor
/// (but for hir objects).
///
/// This module defines a bunch of ids we are using. The most important ones are
/// probably `HirFileId` and `DefId`.

/// Input to the analyzer is a set of files, where each file is identified by
/// `FileId` and contains source code. However, another source of source code in
/// Rust are macros: each macro can be thought of as producing a "temporary
/// file". To assign an id to such a file, we use the id of the macro call that
/// produced the file. So, a `HirFileId` is either a `FileId` (source code
/// written by user), or a `MacroCallId` (source code produced by macro).
///
/// What is a `MacroCallId`? Simplifying, it's a `HirFileId` of a file
/// containing the call plus the offset of the macro call in the file. Note that
/// this is a recursive definition! However, the size_of of `HirFileId` is
/// finite (because everything bottoms out at the real `FileId`) and small
/// (`MacroCallId` uses the location interner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirFileId(HirFileIdRepr);

impl HirFileId {
    /// For macro-expansion files, returns the file original source file the
    /// expansion originated from.
    pub fn original_file(self, db: &impl PersistentHirDatabase) -> FileId {
        match self.0 {
            HirFileIdRepr::File(file_id) => file_id,
            HirFileIdRepr::Macro(macro_call_id) => {
                let loc = macro_call_id.loc(db);
                loc.source_item_id.file_id.original_file(db)
            }
        }
    }

    /// XXX: this is a temporary function, which should go away when we implement the
    /// nameresolution+macro expansion combo. Prefer using `original_file` if
    /// possible.
    pub fn as_original_file(self) -> FileId {
        match self.0 {
            HirFileIdRepr::File(file_id) => file_id,
            HirFileIdRepr::Macro(_r) => panic!("macro generated file: {:?}", self),
        }
    }

    pub(crate) fn as_macro_call_id(self) -> Option<MacroCallId> {
        match self.0 {
            HirFileIdRepr::Macro(it) => Some(it),
            _ => None,
        }
    }

    pub(crate) fn hir_parse(
        db: &impl PersistentHirDatabase,
        file_id: HirFileId,
    ) -> TreeArc<SourceFile> {
        match file_id.0 {
            HirFileIdRepr::File(file_id) => db.parse(file_id),
            HirFileIdRepr::Macro(m) => {
                if let Some(exp) = db.expand_macro_invocation(m) {
                    return exp.file();
                }
                // returning an empty string looks fishy...
                SourceFile::parse("")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HirFileIdRepr {
    File(FileId),
    Macro(MacroCallId),
}

impl From<FileId> for HirFileId {
    fn from(file_id: FileId) -> HirFileId {
        HirFileId(HirFileIdRepr::File(file_id))
    }
}

impl From<MacroCallId> for HirFileId {
    fn from(macro_call_id: MacroCallId) -> HirFileId {
        HirFileId(HirFileIdRepr::Macro(macro_call_id))
    }
}

/// `MacroCallId` identifies a particular macro invocation, like
/// `println!("Hello, {}", world)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroCallId(RawId);
impl_arena_id!(MacroCallId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub(crate) module: Module,
    pub(crate) source_item_id: SourceItemId,
}

impl MacroCallId {
    pub(crate) fn loc(self, db: &impl AsRef<HirInterner>) -> MacroCallLoc {
        db.as_ref().macros.id2loc(self)
    }
}

impl MacroCallLoc {
    #[allow(unused)]
    pub(crate) fn id(&self, db: &impl AsRef<HirInterner>) -> MacroCallId {
        db.as_ref().macros.loc2id(&self)
    }
}

#[derive(Debug)]
pub struct ItemLoc<N: AstNode> {
    pub(crate) module: Module,
    raw: SourceItemId,
    _ty: PhantomData<N>,
}

impl<N: AstNode> PartialEq for ItemLoc<N> {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.raw == other.raw
    }
}
impl<N: AstNode> Eq for ItemLoc<N> {}
impl<N: AstNode> Hash for ItemLoc<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.module.hash(hasher);
        self.raw.hash(hasher);
    }
}

impl<N: AstNode> Clone for ItemLoc<N> {
    fn clone(&self) -> ItemLoc<N> {
        ItemLoc { module: self.module, raw: self.raw, _ty: PhantomData }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LocationCtx<DB> {
    db: DB,
    module: Module,
    file_id: HirFileId,
}

impl<'a, DB: PersistentHirDatabase> LocationCtx<&'a DB> {
    pub(crate) fn new(db: &'a DB, module: Module, file_id: HirFileId) -> LocationCtx<&'a DB> {
        LocationCtx { db, module, file_id }
    }
    pub(crate) fn to_def<N, DEF>(self, ast: &N) -> DEF
    where
        N: AstNode,
        DEF: AstItemDef<N>,
    {
        DEF::from_ast(self, ast)
    }
}

pub(crate) trait AstItemDef<N: AstNode>: ArenaId + Clone {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<N>, Self>;
    fn from_ast(ctx: LocationCtx<&impl PersistentHirDatabase>, ast: &N) -> Self {
        let items = ctx.db.file_items(ctx.file_id);
        let item_id = items.id_of(ctx.file_id, ast.syntax());
        Self::from_source_item_id_unchecked(ctx, item_id)
    }
    fn from_source_item_id_unchecked(
        ctx: LocationCtx<&impl PersistentHirDatabase>,
        item_id: SourceFileItemId,
    ) -> Self {
        let raw = SourceItemId { file_id: ctx.file_id, item_id };
        let loc = ItemLoc { module: ctx.module, raw, _ty: PhantomData };

        Self::interner(ctx.db.as_ref()).loc2id(&loc)
    }
    fn source(self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<N>) {
        let int = Self::interner(db.as_ref());
        let loc = int.id2loc(self);
        let syntax = db.file_item(loc.raw);
        let ast =
            N::cast(&syntax).unwrap_or_else(|| panic!("invalid ItemLoc: {:?}", loc.raw)).to_owned();
        (loc.raw.file_id, ast)
    }
    fn module(self, db: &impl PersistentHirDatabase) -> Module {
        let int = Self::interner(db.as_ref());
        let loc = int.id2loc(self);
        loc.module
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(RawId);
impl_arena_id!(FunctionId);
impl AstItemDef<ast::FnDef> for FunctionId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::FnDef>, Self> {
        &interner.fns
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(RawId);
impl_arena_id!(StructId);
impl AstItemDef<ast::StructDef> for StructId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::StructDef>, Self> {
        &interner.structs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(RawId);
impl_arena_id!(EnumId);
impl AstItemDef<ast::EnumDef> for EnumId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::EnumDef>, Self> {
        &interner.enums
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(RawId);
impl_arena_id!(ConstId);
impl AstItemDef<ast::ConstDef> for ConstId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::ConstDef>, Self> {
        &interner.consts
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(RawId);
impl_arena_id!(StaticId);
impl AstItemDef<ast::StaticDef> for StaticId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::StaticDef>, Self> {
        &interner.statics
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(RawId);
impl_arena_id!(TraitId);
impl AstItemDef<ast::TraitDef> for TraitId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::TraitDef>, Self> {
        &interner.traits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(RawId);
impl_arena_id!(TypeId);
impl AstItemDef<ast::TypeAliasDef> for TypeId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::TypeAliasDef>, Self> {
        &interner.types
    }
}

/// Identifier of item within a specific file. This is stable over reparses, so
/// it's OK to use it as a salsa key/value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceFileItemId(RawId);
impl_arena_id!(SourceFileItemId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceItemId {
    pub(crate) file_id: HirFileId,
    pub(crate) item_id: SourceFileItemId,
}

/// Maps items' `SyntaxNode`s to `SourceFileItemId`s and back.
#[derive(Debug, PartialEq, Eq)]
pub struct SourceFileItems {
    file_id: HirFileId,
    arena: Arena<SourceFileItemId, SyntaxNodePtr>,
}

impl SourceFileItems {
    pub(crate) fn file_items_query(
        db: &impl PersistentHirDatabase,
        file_id: HirFileId,
    ) -> Arc<SourceFileItems> {
        let source_file = db.hir_parse(file_id);
        Arc::new(SourceFileItems::from_source_file(&source_file, file_id))
    }

    pub(crate) fn file_item_query(
        db: &impl PersistentHirDatabase,
        source_item_id: SourceItemId,
    ) -> TreeArc<SyntaxNode> {
        let source_file = db.hir_parse(source_item_id.file_id);
        db.file_items(source_item_id.file_id)[source_item_id.item_id]
            .to_node(&source_file)
            .to_owned()
    }

    pub(crate) fn from_source_file(
        source_file: &SourceFile,
        file_id: HirFileId,
    ) -> SourceFileItems {
        let mut res = SourceFileItems { file_id, arena: Arena::default() };
        // By walking the tree in bread-first order we make sure that parents
        // get lower ids then children. That is, adding a new child does not
        // change parent's id. This means that, say, adding a new function to a
        // trait does not change ids of top-level items, which helps caching.
        bfs(source_file.syntax(), |it| {
            if let Some(module_item) = ast::ModuleItem::cast(it) {
                res.alloc(module_item.syntax());
            } else if let Some(macro_call) = ast::MacroCall::cast(it) {
                res.alloc(macro_call.syntax());
            }
        });
        res
    }

    fn alloc(&mut self, item: &SyntaxNode) -> SourceFileItemId {
        self.arena.alloc(SyntaxNodePtr::new(item))
    }
    pub(crate) fn id_of(&self, file_id: HirFileId, item: &SyntaxNode) -> SourceFileItemId {
        assert_eq!(
            self.file_id, file_id,
            "SourceFileItems: wrong file, expected {:?}, got {:?}",
            self.file_id, file_id
        );
        self.id_of_unchecked(item)
    }
    pub(crate) fn id_of_unchecked(&self, item: &SyntaxNode) -> SourceFileItemId {
        let ptr = SyntaxNodePtr::new(item);
        if let Some((id, _)) = self.arena.iter().find(|(_id, i)| **i == ptr) {
            return id;
        }
        panic!(
            "Can't find {:?} in SourceFileItems:\n{:?}",
            item,
            self.arena.iter().map(|(_id, i)| i).collect::<Vec<_>>(),
        );
    }
}

impl std::ops::Index<SourceFileItemId> for SourceFileItems {
    type Output = SyntaxNodePtr;
    fn index(&self, idx: SourceFileItemId) -> &SyntaxNodePtr {
        &self.arena[idx]
    }
}

/// Walks the subtree in bfs order, calling `f` for each node.
fn bfs(node: &SyntaxNode, mut f: impl FnMut(&SyntaxNode)) {
    let mut curr_layer = vec![node];
    let mut next_layer = vec![];
    while !curr_layer.is_empty() {
        curr_layer.drain(..).for_each(|node| {
            next_layer.extend(node.children());
            f(node);
        });
        std::mem::swap(&mut curr_layer, &mut next_layer);
    }
}
