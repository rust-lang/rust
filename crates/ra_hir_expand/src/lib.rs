//! `ra_hir_expand` deals with macro expansion.
//!
//! Specifically, it implements a concept of `MacroFile` -- a file whose syntax
//! tree originates not from the text of some `FileId`, but from some macro
//! expansion.

pub mod db;
pub mod ast_id_map;
pub mod either;
pub mod name;
pub mod hygiene;
pub mod diagnostics;
pub mod builtin_macro;
pub mod quote;

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ra_db::{salsa, CrateId, FileId};
use ra_syntax::{
    algo,
    ast::{self, AstNode},
    SyntaxNode, SyntaxToken, TextUnit,
};

use crate::ast_id_map::FileAstId;
use crate::builtin_macro::BuiltinExpander;

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
                loc.ast_id.file_id().original_file(db)
            }
        }
    }

    /// Return expansion information if it is a macro-expansion file
    pub fn expansion_info(self, db: &dyn db::AstDatabase) -> Option<ExpansionInfo> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);

                let arg_tt = loc.ast_id.to_node(db).token_tree()?;
                let def_tt = loc.def.ast_id.to_node(db).token_tree()?;

                let macro_def = db.macro_def(loc.def)?;
                let (parse, exp_map) = db.parse_macro(macro_file)?;
                let macro_arg = db.macro_arg(macro_file.macro_call_id)?;

                Some(ExpansionInfo {
                    expanded: Source::new(self, parse.syntax_node()),
                    arg: Source::new(loc.ast_id.file_id, arg_tt),
                    def: Source::new(loc.ast_id.file_id, def_tt),
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
    macro_file_kind: MacroFileKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroFileKind {
    Items,
    Expr,
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
    pub krate: CrateId,
    pub ast_id: AstId<ast::MacroCall>,
    pub kind: MacroDefKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroDefKind {
    Declarative,
    BuiltIn(BuiltinExpander),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub def: MacroDefId,
    pub ast_id: AstId<ast::MacroCall>,
}

impl MacroCallId {
    pub fn as_file(self, kind: MacroFileKind) -> HirFileId {
        let macro_file = MacroFile { macro_call_id: self, macro_file_kind: kind };
        macro_file.into()
    }
}

/// ExpansionInfo mainly describes how to map text range between src and expanded macro
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpansionInfo {
    expanded: Source<SyntaxNode>,
    arg: Source<ast::TokenTree>,
    def: Source<ast::TokenTree>,

    macro_def: Arc<(db::TokenExpander, mbe::TokenMap)>,
    macro_arg: Arc<(tt::Subtree, mbe::TokenMap)>,
    exp_map: Arc<mbe::TokenMap>,
}

impl ExpansionInfo {
    pub fn map_token_down(&self, token: Source<&SyntaxToken>) -> Option<Source<SyntaxToken>> {
        assert_eq!(token.file_id, self.arg.file_id);
        let range =
            token.ast.text_range().checked_sub(self.arg.ast.syntax().text_range().start())?;
        let token_id = self.macro_arg.1.token_by_range(range)?;
        let token_id = self.macro_def.0.map_id_down(token_id);

        let range = self.exp_map.range_by_token(token_id)?;

        let token = algo::find_covering_element(&self.expanded.ast, range).into_token()?;

        Some(self.expanded.with_ast(token))
    }

    pub fn map_token_up(&self, token: Source<&SyntaxToken>) -> Option<Source<SyntaxToken>> {
        let token_id = self.exp_map.token_by_range(token.ast.text_range())?;

        let (token_id, origin) = self.macro_def.0.map_id_up(token_id);
        let (token_map, tt) = match origin {
            mbe::Origin::Call => (&self.macro_arg.1, &self.arg),
            mbe::Origin::Def => (&self.macro_def.1, &self.def),
        };

        let range = token_map.range_by_token(token_id)?;
        let token = algo::find_covering_element(
            tt.ast.syntax(),
            range + tt.ast.syntax().text_range().start(),
        )
        .into_token()?;
        Some(tt.with_ast(token))
    }
}

/// `AstId` points to an AST node in any file.
///
/// It is stable across reparses, and can be used as salsa key/value.
// FIXME: isn't this just a `Source<FileAstId<N>>` ?
#[derive(Debug)]
pub struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> Clone for AstId<N> {
    fn clone(&self) -> AstId<N> {
        *self
    }
}
impl<N: AstNode> Copy for AstId<N> {}

impl<N: AstNode> PartialEq for AstId<N> {
    fn eq(&self, other: &Self) -> bool {
        (self.file_id, self.file_ast_id) == (other.file_id, other.file_ast_id)
    }
}
impl<N: AstNode> Eq for AstId<N> {}
impl<N: AstNode> Hash for AstId<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self.file_id, self.file_ast_id).hash(hasher);
    }
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }

    pub fn file_id(&self) -> HirFileId {
        self.file_id
    }

    pub fn to_node(&self, db: &dyn db::AstDatabase) -> N {
        let root = db.parse_or_expand(self.file_id).unwrap();
        db.ast_id_map(self.file_id).get(self.file_ast_id).to_node(&root)
    }
}

/// FIXME: https://github.com/matklad/with ?
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Source<T> {
    pub file_id: HirFileId,
    // FIXME: this stores all kind of things, not only `ast`.
    // There should be a better name...
    pub ast: T,
}

impl<T> Source<T> {
    pub fn new(file_id: HirFileId, ast: T) -> Source<T> {
        Source { file_id, ast }
    }

    // Similarly, naming here is stupid...
    pub fn with_ast<U>(&self, ast: U) -> Source<U> {
        Source::new(self.file_id, ast)
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source::new(self.file_id, f(self.ast))
    }
    pub fn as_ref(&self) -> Source<&T> {
        self.with_ast(&self.ast)
    }
    pub fn file_syntax(&self, db: &impl db::AstDatabase) -> SyntaxNode {
        db.parse_or_expand(self.file_id).expect("source created from invalid file")
    }
}
