use std::{
    marker::PhantomData,
    hash::Hash,
};

use ra_db::{LocationIntener, FileId};
use ra_syntax::{TreeArc, SyntaxNode, SourceFile, AstNode, ast};
use ra_arena::{Arena, RawId, ArenaId, impl_arena_id};

use crate::{
    HirDatabase, Def,
    Module, Trait, Type,
};

#[derive(Debug, Default)]
pub struct HirInterner {
    defs: LocationIntener<DefLoc, DefId>,
    macros: LocationIntener<MacroCallLoc, MacroCallId>,
    fns: LocationIntener<ItemLoc<ast::FnDef>, FunctionId>,
    structs: LocationIntener<ItemLoc<ast::StructDef>, StructId>,
    enums: LocationIntener<ItemLoc<ast::EnumDef>, EnumId>,
    enum_variants: LocationIntener<ItemLoc<ast::EnumVariant>, EnumVariantId>,
    consts: LocationIntener<ItemLoc<ast::ConstDef>, ConstId>,
    statics: LocationIntener<ItemLoc<ast::StaticDef>, StaticId>,
}

impl HirInterner {
    pub fn len(&self) -> usize {
        self.defs.len() + self.macros.len()
    }
}

/// hir makes heavy use of ids: integer (u32) handlers to various things. You
/// can think of id as a pointer (but without a lifetime) or a file descriptor
/// (but for hir objects).
///
/// This module defines a bunch of ids we are using. The most important ones are
/// probably `HirFileId` and `DefId`.

/// Input to the analyzer is a set of files, where each file is indentified by
/// `FileId` and contains source code. However, another source of source code in
/// Rust are macros: each macro can be thought of as producing a "temporary
/// file". To assign an id to such a file, we use the id of the macro call that
/// produced the file. So, a `HirFileId` is either a `FileId` (source code
/// written by user), or a `MacroCallId` (source code produced by macro).
///
/// What is a `MacroCallId`? Simplifying, it's a `HirFileId` of a file containin
/// the call plus the offset of the macro call in the file. Note that this is a
/// recursive definition! However, the size_of of `HirFileId` is finite
/// (because everything bottoms out at the real `FileId`) and small
/// (`MacroCallId` uses the location interner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirFileId(HirFileIdRepr);

impl HirFileId {
    /// For macro-expansion files, returns the file original source file the
    /// expansionoriginated from.
    pub fn original_file(self, db: &impl HirDatabase) -> FileId {
        match self.0 {
            HirFileIdRepr::File(file_id) => file_id,
            HirFileIdRepr::Macro(macro_call_id) => {
                let loc = macro_call_id.loc(db);
                loc.source_item_id.file_id.original_file(db)
            }
        }
    }

    pub(crate) fn as_original_file(self) -> FileId {
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

    pub(crate) fn hir_source_file(
        db: &impl HirDatabase,
        file_id: HirFileId,
    ) -> TreeArc<SourceFile> {
        match file_id.0 {
            HirFileIdRepr::File(file_id) => db.source_file(file_id),
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ItemLoc<N: AstNode> {
    pub(crate) module: Module,
    raw: SourceItemId,
    _ty: PhantomData<N>,
}

impl<N: AstNode> Clone for ItemLoc<N> {
    fn clone(&self) -> ItemLoc<N> {
        ItemLoc {
            module: self.module,
            raw: self.raw,
            _ty: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LocationCtx<DB> {
    db: DB,
    module: Module,
    file_id: HirFileId,
}

impl<'a, DB: HirDatabase> LocationCtx<&'a DB> {
    pub(crate) fn new(db: &'a DB, module: Module, file_id: HirFileId) -> LocationCtx<&'a DB> {
        LocationCtx {
            db,
            module,
            file_id,
        }
    }
    pub(crate) fn to_def<N, DEF>(self, ast: &N) -> DEF
    where
        N: AstNode + Eq + Hash,
        DEF: AstItemDef<N>,
    {
        DEF::from_ast(self, ast)
    }
}

pub(crate) trait AstItemDef<N: AstNode + Eq + Hash>: ArenaId + Clone {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<N>, Self>;
    fn from_ast(ctx: LocationCtx<&impl HirDatabase>, ast: &N) -> Self {
        let items = ctx.db.file_items(ctx.file_id);
        let raw = SourceItemId {
            file_id: ctx.file_id,
            item_id: Some(items.id_of(ctx.file_id, ast.syntax())),
        };
        let loc = ItemLoc {
            module: ctx.module,
            raw,
            _ty: PhantomData,
        };

        Self::interner(ctx.db.as_ref()).loc2id(&loc)
    }
    fn source(self, db: &impl HirDatabase) -> (HirFileId, TreeArc<N>) {
        let int = Self::interner(db.as_ref());
        let loc = int.id2loc(self);
        let syntax = db.file_item(loc.raw);
        let ast = N::cast(&syntax)
            .unwrap_or_else(|| panic!("invalid ItemLoc: {:?}", loc.raw))
            .to_owned();
        (loc.raw.file_id, ast)
    }
    fn module(self, db: &impl HirDatabase) -> Module {
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
pub struct EnumVariantId(RawId);
impl_arena_id!(EnumVariantId);
impl AstItemDef<ast::EnumVariant> for EnumVariantId {
    fn interner(interner: &HirInterner) -> &LocationIntener<ItemLoc<ast::EnumVariant>, Self> {
        &interner.enum_variants
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

/// Def's are a core concept of hir. A `Def` is an Item (function, module, etc)
/// in a specific module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(RawId);
impl_arena_id!(DefId);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DefLoc {
    pub(crate) kind: DefKind,
    pub(crate) module: Module,
    pub(crate) source_item_id: SourceItemId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DefKind {
    Trait,
    Type,
    Item,
    // /// The constructor of a struct. E.g. if we have `struct Foo(usize)`, the
    // /// name `Foo` needs to resolve to different types depending on whether we
    // /// are in the types or values namespace: As a type, `Foo` of course refers
    // /// to the struct `Foo`; as a value, `Foo` is a callable type with signature
    // /// `(usize) -> Foo`. The cleanest approach to handle this seems to be to
    // /// have different defs in the two namespaces.
    // ///
    // /// rustc does the same; note that it even creates a struct constructor if
    // /// the struct isn't a tuple struct (see `CtorKind::Fictive` in rustc).
    // StructCtor,
}

impl DefId {
    pub(crate) fn loc(self, db: &impl AsRef<HirInterner>) -> DefLoc {
        db.as_ref().defs.id2loc(self)
    }

    pub fn resolve(self, db: &impl HirDatabase) -> Def {
        let loc = self.loc(db);
        match loc.kind {
            DefKind::Trait => {
                let def = Trait::new(self);
                Def::Trait(def)
            }
            DefKind::Type => {
                let def = Type::new(self);
                Def::Type(def)
            }
            DefKind::Item => Def::Item,
        }
    }

    pub(crate) fn source(self, db: &impl HirDatabase) -> (HirFileId, TreeArc<SyntaxNode>) {
        let loc = self.loc(db);
        let syntax = db.file_item(loc.source_item_id);
        (loc.source_item_id.file_id, syntax)
    }
}

impl DefLoc {
    pub(crate) fn id(&self, db: &impl AsRef<HirInterner>) -> DefId {
        db.as_ref().defs.loc2id(&self)
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
    /// None for the whole file.
    pub(crate) item_id: Option<SourceFileItemId>,
}

/// Maps items' `SyntaxNode`s to `SourceFileItemId`s and back.
#[derive(Debug, PartialEq, Eq)]
pub struct SourceFileItems {
    file_id: HirFileId,
    arena: Arena<SourceFileItemId, TreeArc<SyntaxNode>>,
}

impl SourceFileItems {
    pub(crate) fn new(file_id: HirFileId, source_file: &SourceFile) -> SourceFileItems {
        let mut res = SourceFileItems {
            file_id,
            arena: Arena::default(),
        };
        res.init(source_file);
        res
    }

    fn init(&mut self, source_file: &SourceFile) {
        // By walking the tree in bread-first order we make sure that parents
        // get lower ids then children. That is, addding a new child does not
        // change parent's id. This means that, say, adding a new function to a
        // trait does not chage ids of top-level items, which helps caching.
        bfs(source_file.syntax(), |it| {
            if let Some(enum_variant) = ast::EnumVariant::cast(it) {
                self.alloc(enum_variant.syntax().to_owned());
            } else if let Some(module_item) = ast::ModuleItem::cast(it) {
                self.alloc(module_item.syntax().to_owned());
            } else if let Some(macro_call) = ast::MacroCall::cast(it) {
                self.alloc(macro_call.syntax().to_owned());
            }
        })
    }

    fn alloc(&mut self, item: TreeArc<SyntaxNode>) -> SourceFileItemId {
        self.arena.alloc(item)
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
        if let Some((id, _)) = self.arena.iter().find(|(_id, i)| *i == item) {
            return id;
        }
        // This should not happen. Let's try to give a sensible diagnostics.
        if let Some((id, i)) = self.arena.iter().find(|(_id, i)| i.range() == item.range()) {
            // FIXME(#288): whyyy are we getting here?
            log::error!(
                "unequal syntax nodes with the same range:\n{:?}\n{:?}",
                item,
                i
            );
            return id;
        }
        panic!(
            "Can't find {:?} in SourceFileItems:\n{:?}",
            item,
            self.arena.iter().map(|(_id, i)| i).collect::<Vec<_>>(),
        );
    }
    pub fn id_of_source_file(&self) -> SourceFileItemId {
        let (id, _syntax) = self.arena.iter().next().unwrap();
        id
    }
}

impl std::ops::Index<SourceFileItemId> for SourceFileItems {
    type Output = SyntaxNode;
    fn index(&self, idx: SourceFileItemId) -> &SyntaxNode {
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
