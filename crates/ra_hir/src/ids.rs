//! hir makes heavy use of ids: integer (u32) handlers to various things. You
//! can think of id as a pointer (but without a lifetime) or a file descriptor
//! (but for hir objects).
//!
//! This module defines a bunch of ids we are using. The most important ones are
//! probably `HirFileId` and `DefId`.

use std::hash::{Hash, Hasher};

use ra_db::salsa;
use ra_syntax::{ast, AstNode};

use crate::{
    db::{AstDatabase, InternDatabase},
    AstId, FileAstId, Module, Source,
};

pub use hir_def::expand::{
    HirFileId, MacroCallId, MacroCallLoc, MacroDefId, MacroFile, MacroFileKind,
};

macro_rules! impl_intern_key {
    ($name:ident) => {
        impl salsa::InternKey for $name {
            fn from_intern_id(v: salsa::InternId) -> Self {
                $name(v)
            }
            fn as_intern_id(&self) -> salsa::InternId {
                self.0
            }
        }
    };
}

#[derive(Debug)]
pub struct ItemLoc<N: AstNode> {
    pub(crate) module: Module,
    ast_id: AstId<N>,
}

impl<N: AstNode> PartialEq for ItemLoc<N> {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.ast_id == other.ast_id
    }
}
impl<N: AstNode> Eq for ItemLoc<N> {}
impl<N: AstNode> Hash for ItemLoc<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.module.hash(hasher);
        self.ast_id.hash(hasher);
    }
}

impl<N: AstNode> Clone for ItemLoc<N> {
    fn clone(&self) -> ItemLoc<N> {
        ItemLoc { module: self.module, ast_id: self.ast_id }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LocationCtx<DB> {
    db: DB,
    module: Module,
    file_id: HirFileId,
}

impl<'a, DB> LocationCtx<&'a DB> {
    pub(crate) fn new(db: &'a DB, module: Module, file_id: HirFileId) -> LocationCtx<&'a DB> {
        LocationCtx { db, module, file_id }
    }
}

impl<'a, DB: AstDatabase + InternDatabase> LocationCtx<&'a DB> {
    pub(crate) fn to_def<N, DEF>(self, ast: &N) -> DEF
    where
        N: AstNode,
        DEF: AstItemDef<N>,
    {
        DEF::from_ast(self, ast)
    }
}

pub(crate) trait AstItemDef<N: AstNode>: salsa::InternKey + Clone {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<N>) -> Self;
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<N>;

    fn from_ast(ctx: LocationCtx<&(impl AstDatabase + InternDatabase)>, ast: &N) -> Self {
        let items = ctx.db.ast_id_map(ctx.file_id);
        let item_id = items.ast_id(ast);
        Self::from_ast_id(ctx, item_id)
    }
    fn from_ast_id(ctx: LocationCtx<&impl InternDatabase>, ast_id: FileAstId<N>) -> Self {
        let loc = ItemLoc { module: ctx.module, ast_id: AstId::new(ctx.file_id, ast_id) };
        Self::intern(ctx.db, loc)
    }
    fn source(self, db: &(impl AstDatabase + InternDatabase)) -> Source<N> {
        let loc = self.lookup_intern(db);
        let ast = loc.ast_id.to_node(db);
        Source { file_id: loc.ast_id.file_id(), ast }
    }
    fn module(self, db: &impl InternDatabase) -> Module {
        let loc = self.lookup_intern(db);
        loc.module
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(salsa::InternId);
impl_intern_key!(FunctionId);

impl AstItemDef<ast::FnDef> for FunctionId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::FnDef>) -> Self {
        db.intern_function(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::FnDef> {
        db.lookup_intern_function(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(salsa::InternId);
impl_intern_key!(StructId);
impl AstItemDef<ast::StructDef> for StructId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::StructDef>) -> Self {
        db.intern_struct(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::StructDef> {
        db.lookup_intern_struct(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(salsa::InternId);
impl_intern_key!(EnumId);
impl AstItemDef<ast::EnumDef> for EnumId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::EnumDef>) -> Self {
        db.intern_enum(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::EnumDef> {
        db.lookup_intern_enum(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
impl_intern_key!(ConstId);
impl AstItemDef<ast::ConstDef> for ConstId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::ConstDef>) -> Self {
        db.intern_const(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::ConstDef> {
        db.lookup_intern_const(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
impl_intern_key!(StaticId);
impl AstItemDef<ast::StaticDef> for StaticId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::StaticDef>) -> Self {
        db.intern_static(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::StaticDef> {
        db.lookup_intern_static(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(salsa::InternId);
impl_intern_key!(TraitId);
impl AstItemDef<ast::TraitDef> for TraitId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::TraitDef>) -> Self {
        db.intern_trait(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::TraitDef> {
        db.lookup_intern_trait(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAliasId(salsa::InternId);
impl_intern_key!(TypeAliasId);
impl AstItemDef<ast::TypeAliasDef> for TypeAliasId {
    fn intern(db: &impl InternDatabase, loc: ItemLoc<ast::TypeAliasDef>) -> Self {
        db.intern_type_alias(loc)
    }
    fn lookup_intern(self, db: &impl InternDatabase) -> ItemLoc<ast::TypeAliasDef> {
        db.lookup_intern_type_alias(self)
    }
}

/// This exists just for Chalk, because Chalk just has a single `StructId` where
/// we have different kinds of ADTs, primitive types and special type
/// constructors like tuples and function pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeCtorId(salsa::InternId);
impl_intern_key!(TypeCtorId);

/// This exists just for Chalk, because our ImplIds are only unique per module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalImplId(salsa::InternId);
impl_intern_key!(GlobalImplId);
