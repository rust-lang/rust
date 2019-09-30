//! FIXME: write short doc here

use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use mbe::MacroRules;
use ra_db::{salsa, FileId};
use ra_prof::profile;
use ra_syntax::{ast, AstNode, Parse, SyntaxNode};

use crate::{
    db::{AstDatabase, DefDatabase, InternDatabase},
    AstId, Crate, FileAstId, Module, Source,
};

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
    pub fn original_file(self, db: &impl InternDatabase) -> FileId {
        match self.0 {
            HirFileIdRepr::File(file_id) => file_id,
            HirFileIdRepr::Macro(macro_file) => {
                let loc = macro_file.macro_call_id.loc(db);
                loc.ast_id.file_id().original_file(db)
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

    /// Get the crate which the macro lives in, if it is a macro file.
    pub(crate) fn macro_crate(self, db: &impl AstDatabase) -> Option<Crate> {
        match self.0 {
            HirFileIdRepr::File(_) => None,
            HirFileIdRepr::Macro(macro_file) => {
                let loc = macro_file.macro_call_id.loc(db);
                Some(loc.def.krate)
            }
        }
    }

    pub(crate) fn parse_or_expand_query(
        db: &impl AstDatabase,
        file_id: HirFileId,
    ) -> Option<SyntaxNode> {
        match file_id.0 {
            HirFileIdRepr::File(file_id) => Some(db.parse(file_id).tree().syntax().clone()),
            HirFileIdRepr::Macro(macro_file) => {
                db.parse_macro(macro_file).map(|it| it.syntax_node())
            }
        }
    }

    pub(crate) fn parse_macro_query(
        db: &impl AstDatabase,
        macro_file: MacroFile,
    ) -> Option<Parse<SyntaxNode>> {
        let _p = profile("parse_macro_query");
        let macro_call_id = macro_file.macro_call_id;
        let tt = db
            .macro_expand(macro_call_id)
            .map_err(|err| {
                // Note:
                // The final goal we would like to make all parse_macro success,
                // such that the following log will not call anyway.
                log::warn!(
                    "fail on macro_parse: (reason: {}) {}",
                    err,
                    macro_call_id.debug_dump(db)
                );
            })
            .ok()?;
        match macro_file.macro_file_kind {
            MacroFileKind::Items => mbe::token_tree_to_items(&tt).ok().map(Parse::to_syntax),
            MacroFileKind::Expr => mbe::token_tree_to_expr(&tt).ok().map(Parse::to_syntax),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HirFileIdRepr {
    File(FileId),
    Macro(MacroFile),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroFile {
    macro_call_id: MacroCallId,
    macro_file_kind: MacroFileKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MacroFileKind {
    Items,
    Expr,
}

impl From<FileId> for HirFileId {
    fn from(file_id: FileId) -> HirFileId {
        HirFileId(HirFileIdRepr::File(file_id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDefId {
    pub(crate) ast_id: AstId<ast::MacroCall>,
    pub(crate) krate: Crate,
}

pub(crate) fn macro_def_query(db: &impl AstDatabase, id: MacroDefId) -> Option<Arc<MacroRules>> {
    let macro_call = id.ast_id.to_node(db);
    let arg = macro_call.token_tree()?;
    let (tt, _) = mbe::ast_to_token_tree(&arg).or_else(|| {
        log::warn!("fail on macro_def to token tree: {:#?}", arg);
        None
    })?;
    let rules = MacroRules::parse(&tt).ok().or_else(|| {
        log::warn!("fail on macro_def parse: {:#?}", tt);
        None
    })?;
    Some(Arc::new(rules))
}

pub(crate) fn macro_arg_query(db: &impl AstDatabase, id: MacroCallId) -> Option<Arc<tt::Subtree>> {
    let loc = id.loc(db);
    let macro_call = loc.ast_id.to_node(db);
    let arg = macro_call.token_tree()?;
    let (tt, _) = mbe::ast_to_token_tree(&arg)?;
    Some(Arc::new(tt))
}

pub(crate) fn macro_expand_query(
    db: &impl AstDatabase,
    id: MacroCallId,
) -> Result<Arc<tt::Subtree>, String> {
    let loc = id.loc(db);
    let macro_arg = db.macro_arg(id).ok_or("Fail to args in to tt::TokenTree")?;

    let macro_rules = db.macro_def(loc.def).ok_or("Fail to find macro definition")?;
    let tt = macro_rules.expand(&macro_arg).map_err(|err| format!("{:?}", err))?;
    // Set a hard limit for the expanded tt
    let count = tt.count();
    if count > 65536 {
        return Err(format!("Total tokens count exceed limit : count = {}", count));
    }
    Ok(Arc::new(tt))
}

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

/// `MacroCallId` identifies a particular macro invocation, like
/// `println!("Hello, {}", world)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroCallId(salsa::InternId);
impl_intern_key!(MacroCallId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub(crate) def: MacroDefId,
    pub(crate) ast_id: AstId<ast::MacroCall>,
}

impl MacroCallId {
    pub(crate) fn loc(self, db: &impl InternDatabase) -> MacroCallLoc {
        db.lookup_intern_macro(self)
    }

    pub(crate) fn as_file(self, kind: MacroFileKind) -> HirFileId {
        let macro_file = MacroFile { macro_call_id: self, macro_file_kind: kind };
        HirFileId(HirFileIdRepr::Macro(macro_file))
    }
}

impl MacroCallLoc {
    pub(crate) fn id(self, db: &impl InternDatabase) -> MacroCallId {
        db.intern_macro(self)
    }
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

impl<'a, DB: DefDatabase> LocationCtx<&'a DB> {
    pub(crate) fn new(db: &'a DB, module: Module, file_id: HirFileId) -> LocationCtx<&'a DB> {
        LocationCtx { db, module, file_id }
    }
}

impl<'a, DB: DefDatabase + AstDatabase> LocationCtx<&'a DB> {
    pub(crate) fn to_def<N, DEF>(self, ast: &N) -> DEF
    where
        N: AstNode,
        DEF: AstItemDef<N>,
    {
        DEF::from_ast(self, ast)
    }
}

pub(crate) trait AstItemDef<N: AstNode>: salsa::InternKey + Clone {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<N>) -> Self;
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<N>;

    fn from_ast(ctx: LocationCtx<&(impl AstDatabase + DefDatabase)>, ast: &N) -> Self {
        let items = ctx.db.ast_id_map(ctx.file_id);
        let item_id = items.ast_id(ast);
        Self::from_ast_id(ctx, item_id)
    }
    fn from_ast_id(ctx: LocationCtx<&impl DefDatabase>, ast_id: FileAstId<N>) -> Self {
        let loc = ItemLoc { module: ctx.module, ast_id: ast_id.with_file_id(ctx.file_id) };
        Self::intern(ctx.db, loc)
    }
    fn source(self, db: &(impl AstDatabase + DefDatabase)) -> Source<N> {
        let loc = self.lookup_intern(db);
        let ast = loc.ast_id.to_node(db);
        Source { file_id: loc.ast_id.file_id(), ast }
    }
    fn module(self, db: &impl DefDatabase) -> Module {
        let loc = self.lookup_intern(db);
        loc.module
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(salsa::InternId);
impl_intern_key!(FunctionId);

impl AstItemDef<ast::FnDef> for FunctionId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::FnDef>) -> Self {
        db.intern_function(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::FnDef> {
        db.lookup_intern_function(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(salsa::InternId);
impl_intern_key!(StructId);
impl AstItemDef<ast::StructDef> for StructId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::StructDef>) -> Self {
        db.intern_struct(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::StructDef> {
        db.lookup_intern_struct(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(salsa::InternId);
impl_intern_key!(EnumId);
impl AstItemDef<ast::EnumDef> for EnumId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::EnumDef>) -> Self {
        db.intern_enum(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::EnumDef> {
        db.lookup_intern_enum(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstId(salsa::InternId);
impl_intern_key!(ConstId);
impl AstItemDef<ast::ConstDef> for ConstId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::ConstDef>) -> Self {
        db.intern_const(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::ConstDef> {
        db.lookup_intern_const(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticId(salsa::InternId);
impl_intern_key!(StaticId);
impl AstItemDef<ast::StaticDef> for StaticId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::StaticDef>) -> Self {
        db.intern_static(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::StaticDef> {
        db.lookup_intern_static(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraitId(salsa::InternId);
impl_intern_key!(TraitId);
impl AstItemDef<ast::TraitDef> for TraitId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::TraitDef>) -> Self {
        db.intern_trait(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::TraitDef> {
        db.lookup_intern_trait(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAliasId(salsa::InternId);
impl_intern_key!(TypeAliasId);
impl AstItemDef<ast::TypeAliasDef> for TypeAliasId {
    fn intern(db: &impl DefDatabase, loc: ItemLoc<ast::TypeAliasDef>) -> Self {
        db.intern_type_alias(loc)
    }
    fn lookup_intern(self, db: &impl DefDatabase) -> ItemLoc<ast::TypeAliasDef> {
        db.lookup_intern_type_alias(self)
    }
}

impl MacroCallId {
    pub fn debug_dump(self, db: &impl AstDatabase) -> String {
        let loc = self.loc(db);
        let node = loc.ast_id.to_node(db);
        let syntax_str = {
            let mut res = String::new();
            node.syntax().text().for_each_chunk(|chunk| {
                if !res.is_empty() {
                    res.push(' ')
                }
                res.push_str(chunk)
            });
            res
        };

        // dump the file name
        let file_id: HirFileId = self.loc(db).ast_id.file_id();
        let original = file_id.original_file(db);
        let macro_rules = db.macro_def(loc.def);

        format!(
            "macro call [file: {:?}] : {}\nhas rules: {}",
            db.file_relative_path(original),
            syntax_str,
            macro_rules.is_some()
        )
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
