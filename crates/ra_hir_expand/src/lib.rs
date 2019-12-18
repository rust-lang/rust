//! `ra_hir_expand` deals with macro expansion.
//!
//! Specifically, it implements a concept of `MacroFile` -- a file whose syntax
//! tree originates not from the text of some `FileId`, but from some macro
//! expansion.

pub mod db;
pub mod ast_id_map;
pub mod name;
pub mod hygiene;
pub mod diagnostics;
pub mod builtin_derive;
pub mod builtin_macro;
pub mod quote;

use std::hash::Hash;
use std::sync::Arc;

use ra_db::{salsa, CrateId, FileId};
use ra_syntax::{
    algo,
    ast::{self, AstNode},
    SyntaxNode, SyntaxToken, TextUnit,
};

use crate::ast_id_map::FileAstId;
use crate::builtin_derive::BuiltinDeriveExpander;
use crate::builtin_macro::BuiltinFnLikeExpander;

#[cfg(test)]
mod test_db;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HirFileIdRepr {
    FileId(FileId),
    MacroFile(MacroFile),
}

impl From<FileId> for HirFileId {
    fn from(id: FileId) -> Self {
        HirFileId(HirFileIdRepr::FileId(id))
    }
}

impl From<MacroFile> for HirFileId {
    fn from(id: MacroFile) -> Self {
        HirFileId(HirFileIdRepr::MacroFile(id))
    }
}

impl HirFileId {
    /// For macro-expansion files, returns the file original source file the
    /// expansion originated from.
    pub fn original_file(self, db: &dyn db::AstDatabase) -> FileId {
        match self.0 {
            HirFileIdRepr::FileId(file_id) => file_id,
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc = db.lookup_intern_macro(macro_file.macro_call_id);
                loc.kind.file_id().original_file(db)
            }
        }
    }

    /// If this is a macro call, returns the syntax node of the call.
    pub fn call_node(self, db: &dyn db::AstDatabase) -> Option<InFile<SyntaxNode>> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc = db.lookup_intern_macro(macro_file.macro_call_id);
                Some(loc.kind.node(db))
            }
        }
    }

    /// Return expansion information if it is a macro-expansion file
    pub fn expansion_info(self, db: &dyn db::AstDatabase) -> Option<ExpansionInfo> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);

                let arg_tt = loc.kind.arg(db)?;
                let def_tt = loc.def.ast_id?.to_node(db).token_tree()?;

                let macro_def = db.macro_def(loc.def)?;
                let (parse, exp_map) = db.parse_macro(macro_file)?;
                let macro_arg = db.macro_arg(macro_file.macro_call_id)?;

                Some(ExpansionInfo {
                    expanded: InFile::new(self, parse.syntax_node()),
                    arg: InFile::new(loc.kind.file_id(), arg_tt),
                    def: InFile::new(loc.def.ast_id?.file_id, def_tt),
                    macro_arg,
                    macro_def,
                    exp_map,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroFile {
    macro_call_id: MacroCallId,
}

/// `MacroCallId` identifies a particular macro invocation, like
/// `println!("Hello, {}", world)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroCallId(salsa::InternId);
impl salsa::InternKey for MacroCallId {
    fn from_intern_id(v: salsa::InternId) -> Self {
        MacroCallId(v)
    }
    fn as_intern_id(&self) -> salsa::InternId {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDefId {
    // FIXME: krate and ast_id are currently optional because we don't have a
    // definition location for built-in derives. There is one, though: the
    // standard library defines them. The problem is that it uses the new
    // `macro` syntax for this, which we don't support yet. As soon as we do
    // (which will probably require touching this code), we can instead use
    // that (and also remove the hacks for resolving built-in derives).
    pub krate: Option<CrateId>,
    pub ast_id: Option<AstId<ast::MacroCall>>,
    pub kind: MacroDefKind,
}

impl MacroDefId {
    pub fn as_call_id(self, db: &dyn db::AstDatabase, kind: MacroCallKind) -> MacroCallId {
        db.intern_macro(MacroCallLoc { def: self, kind })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroDefKind {
    Declarative,
    BuiltIn(BuiltinFnLikeExpander),
    // FIXME: maybe just Builtin and rename BuiltinFnLikeExpander to BuiltinExpander
    BuiltInDerive(BuiltinDeriveExpander),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub(crate) def: MacroDefId,
    pub(crate) kind: MacroCallKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroCallKind {
    FnLike(AstId<ast::MacroCall>),
    Attr(AstId<ast::ModuleItem>),
}

impl MacroCallKind {
    pub fn file_id(&self) -> HirFileId {
        match self {
            MacroCallKind::FnLike(ast_id) => ast_id.file_id,
            MacroCallKind::Attr(ast_id) => ast_id.file_id,
        }
    }

    pub fn node(&self, db: &dyn db::AstDatabase) -> InFile<SyntaxNode> {
        match self {
            MacroCallKind::FnLike(ast_id) => ast_id.with_value(ast_id.to_node(db).syntax().clone()),
            MacroCallKind::Attr(ast_id) => ast_id.with_value(ast_id.to_node(db).syntax().clone()),
        }
    }

    pub fn arg(&self, db: &dyn db::AstDatabase) -> Option<SyntaxNode> {
        match self {
            MacroCallKind::FnLike(ast_id) => {
                Some(ast_id.to_node(db).token_tree()?.syntax().clone())
            }
            MacroCallKind::Attr(ast_id) => Some(ast_id.to_node(db).syntax().clone()),
        }
    }
}

impl MacroCallId {
    pub fn as_file(self) -> HirFileId {
        MacroFile { macro_call_id: self }.into()
    }
}

/// ExpansionInfo mainly describes how to map text range between src and expanded macro
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpansionInfo {
    expanded: InFile<SyntaxNode>,
    arg: InFile<SyntaxNode>,
    def: InFile<ast::TokenTree>,

    macro_def: Arc<(db::TokenExpander, mbe::TokenMap)>,
    macro_arg: Arc<(tt::Subtree, mbe::TokenMap)>,
    exp_map: Arc<mbe::TokenMap>,
}

pub use mbe::Origin;

impl ExpansionInfo {
    pub fn call_node(&self) -> Option<InFile<SyntaxNode>> {
        Some(self.arg.with_value(self.arg.value.parent()?))
    }

    pub fn map_token_down(&self, token: InFile<&SyntaxToken>) -> Option<InFile<SyntaxToken>> {
        assert_eq!(token.file_id, self.arg.file_id);
        let range = token.value.text_range().checked_sub(self.arg.value.text_range().start())?;
        let token_id = self.macro_arg.1.token_by_range(range)?;
        let token_id = self.macro_def.0.map_id_down(token_id);

        let range = self.exp_map.range_by_token(token_id)?.by_kind(token.value.kind())?;

        let token = algo::find_covering_element(&self.expanded.value, range).into_token()?;

        Some(self.expanded.with_value(token))
    }

    pub fn map_token_up(
        &self,
        token: InFile<&SyntaxToken>,
    ) -> Option<(InFile<SyntaxToken>, Origin)> {
        let token_id = self.exp_map.token_by_range(token.value.text_range())?;

        let (token_id, origin) = self.macro_def.0.map_id_up(token_id);
        let (token_map, tt) = match origin {
            mbe::Origin::Call => (&self.macro_arg.1, self.arg.clone()),
            mbe::Origin::Def => {
                (&self.macro_def.1, self.def.as_ref().map(|tt| tt.syntax().clone()))
            }
        };

        let range = token_map.range_by_token(token_id)?.by_kind(token.value.kind())?;
        let token = algo::find_covering_element(&tt.value, range + tt.value.text_range().start())
            .into_token()?;
        Some((tt.with_value(token), origin))
    }
}

/// `AstId` points to an AST node in any file.
///
/// It is stable across reparses, and can be used as salsa key/value.
// FIXME: isn't this just a `Source<FileAstId<N>>` ?
pub type AstId<N> = InFile<FileAstId<N>>;

impl<N: AstNode> AstId<N> {
    pub fn to_node(&self, db: &dyn db::AstDatabase) -> N {
        let root = db.parse_or_expand(self.file_id).unwrap();
        db.ast_id_map(self.file_id).get(self.value).to_node(&root)
    }
}

/// `InFile<T>` stores a value of `T` inside a particular file/syntax tree.
///
/// Typical usages are:
///
/// * `InFile<SyntaxNode>` -- syntax node in a file
/// * `InFile<ast::FnDef>` -- ast node in a file
/// * `InFile<TextUnit>` -- offset in a file
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct InFile<T> {
    pub file_id: HirFileId,
    pub value: T,
}

impl<T> InFile<T> {
    pub fn new(file_id: HirFileId, value: T) -> InFile<T> {
        InFile { file_id, value }
    }

    // Similarly, naming here is stupid...
    pub fn with_value<U>(&self, value: U) -> InFile<U> {
        InFile::new(self.file_id, value)
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> InFile<U> {
        InFile::new(self.file_id, f(self.value))
    }
    pub fn as_ref(&self) -> InFile<&T> {
        self.with_value(&self.value)
    }
    pub fn file_syntax(&self, db: &impl db::AstDatabase) -> SyntaxNode {
        db.parse_or_expand(self.file_id).expect("source created from invalid file")
    }
}

impl<T: Clone> InFile<&T> {
    pub fn cloned(&self) -> InFile<T> {
        self.with_value(self.value.clone())
    }
}

impl InFile<SyntaxNode> {
    pub fn ancestors_with_macros<'a>(
        self,
        db: &'a impl crate::db::AstDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + 'a {
        std::iter::successors(Some(self), move |node| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let parent_node = node.file_id.call_node(db)?;
                Some(parent_node)
            }
        })
    }
}
