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
    ast::{self, AstNode},
    SyntaxNode, TextRange, TextUnit,
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

                let arg_start = loc.ast_id.to_node(db).token_tree()?.syntax().text_range().start();
                let def_start =
                    loc.def.ast_id.to_node(db).token_tree()?.syntax().text_range().start();

                let macro_def = db.macro_def(loc.def)?;
                let shift = macro_def.0.shift();
                let exp_map = db.parse_macro(macro_file)?.1;
                let macro_arg = db.macro_arg(macro_file.macro_call_id)?;

                let arg_start = (loc.ast_id.file_id, arg_start);
                let def_start = (loc.def.ast_id.file_id, def_start);

                Some(ExpansionInfo { arg_start, def_start, macro_arg, macro_def, exp_map, shift })
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

#[derive(Debug, Clone, PartialEq, Eq)]
/// ExpansionInfo mainly describes how to map text range between src and expanded macro
pub struct ExpansionInfo {
    pub(crate) arg_start: (HirFileId, TextUnit),
    pub(crate) def_start: (HirFileId, TextUnit),
    pub(crate) shift: u32,

    pub(crate) macro_def: Arc<(db::TokenExpander, mbe::TokenMap)>,
    pub(crate) macro_arg: Arc<(tt::Subtree, mbe::TokenMap)>,
    pub(crate) exp_map: Arc<mbe::RevTokenMap>,
}

impl ExpansionInfo {
    pub fn find_range(&self, from: TextRange) -> Option<(HirFileId, TextRange)> {
        let token_id = look_in_rev_map(&self.exp_map, from)?;

        let (token_map, (file_id, start_offset), token_id) = if token_id.0 >= self.shift {
            (&self.macro_arg.1, self.arg_start, tt::TokenId(token_id.0 - self.shift).into())
        } else {
            (&self.macro_def.1, self.def_start, token_id)
        };

        let range = token_map.relative_range_of(token_id)?;

        return Some((file_id, range + start_offset));

        fn look_in_rev_map(exp_map: &mbe::RevTokenMap, from: TextRange) -> Option<tt::TokenId> {
            exp_map.ranges.iter().find(|&it| it.0.is_subrange(&from)).map(|it| it.1)
        }
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Source<T> {
    pub file_id: HirFileId,
    pub ast: T,
}

impl<T> Source<T> {
    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> Source<U> {
        Source { file_id: self.file_id, ast: f(self.ast) }
    }
    pub fn file_syntax(&self, db: &impl db::AstDatabase) -> SyntaxNode {
        db.parse_or_expand(self.file_id).expect("source created from invalid file")
    }
}
