//! `hir_expand` deals with macro expansion.
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
pub mod proc_macro;
pub mod quote;
pub mod eager;

use either::Either;
pub use mbe::{ExpandError, ExpandResult};

use std::hash::Hash;
use std::sync::Arc;

use base_db::{impl_intern_key, salsa, CrateId, FileId, FileRange};
use syntax::{
    algo::skip_trivia_token,
    ast::{self, AstNode},
    Direction, SyntaxNode, SyntaxToken, TextRange, TextSize,
};

use crate::ast_id_map::FileAstId;
use crate::builtin_derive::BuiltinDeriveExpander;
use crate::builtin_macro::{BuiltinFnLikeExpander, EagerExpander};
use crate::proc_macro::ProcMacroExpander;

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
/// (`MacroCallId` uses the location interning. You can check details here:
/// https://en.wikipedia.org/wiki/String_interning).
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
                let file_id = match macro_file.macro_call_id {
                    MacroCallId::LazyMacro(id) => {
                        let loc = db.lookup_intern_macro(id);
                        loc.kind.file_id()
                    }
                    MacroCallId::EagerMacro(id) => {
                        let loc = db.lookup_intern_eager_expansion(id);
                        if let Some(included_file) = loc.included_file {
                            return included_file;
                        } else {
                            loc.call.file_id
                        }
                    }
                };
                file_id.original_file(db)
            }
        }
    }

    pub fn expansion_level(self, db: &dyn db::AstDatabase) -> u32 {
        let mut level = 0;
        let mut curr = self;
        while let HirFileIdRepr::MacroFile(macro_file) = curr.0 {
            level += 1;
            curr = match macro_file.macro_call_id {
                MacroCallId::LazyMacro(id) => {
                    let loc = db.lookup_intern_macro(id);
                    loc.kind.file_id()
                }
                MacroCallId::EagerMacro(id) => {
                    let loc = db.lookup_intern_eager_expansion(id);
                    loc.call.file_id
                }
            };
        }
        level
    }

    /// If this is a macro call, returns the syntax node of the call.
    pub fn call_node(self, db: &dyn db::AstDatabase) -> Option<InFile<SyntaxNode>> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => match macro_file.macro_call_id {
                MacroCallId::LazyMacro(lazy_id) => {
                    let loc: MacroCallLoc = db.lookup_intern_macro(lazy_id);
                    Some(loc.kind.node(db))
                }
                MacroCallId::EagerMacro(id) => {
                    let loc: EagerCallLoc = db.lookup_intern_eager_expansion(id);
                    Some(loc.call.with_value(loc.call.to_node(db).syntax().clone()))
                }
            },
        }
    }

    /// Return expansion information if it is a macro-expansion file
    pub fn expansion_info(self, db: &dyn db::AstDatabase) -> Option<ExpansionInfo> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let lazy_id = match macro_file.macro_call_id {
                    MacroCallId::LazyMacro(id) => id,
                    MacroCallId::EagerMacro(_id) => {
                        // FIXME: handle expansion_info for eager macro
                        return None;
                    }
                };
                let loc: MacroCallLoc = db.lookup_intern_macro(lazy_id);

                let arg_tt = loc.kind.arg(db)?;

                let def = loc.def.ast_id().left().and_then(|id| {
                    let def_tt = match id.to_node(db) {
                        ast::Macro::MacroRules(mac) => mac.token_tree()?,
                        ast::Macro::MacroDef(_) => return None,
                    };
                    Some(InFile::new(id.file_id, def_tt))
                });

                let macro_def = db.macro_def(loc.def)?;
                let (parse, exp_map) = db.parse_macro_expansion(macro_file).value?;
                let macro_arg = db.macro_arg(macro_file.macro_call_id)?;

                Some(ExpansionInfo {
                    expanded: InFile::new(self, parse.syntax_node()),
                    arg: InFile::new(loc.kind.file_id(), arg_tt),
                    def,
                    macro_arg,
                    macro_def,
                    exp_map,
                })
            }
        }
    }

    /// Indicate it is macro file generated for builtin derive
    pub fn is_builtin_derive(&self, db: &dyn db::AstDatabase) -> Option<InFile<ast::Item>> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let lazy_id = match macro_file.macro_call_id {
                    MacroCallId::LazyMacro(id) => id,
                    MacroCallId::EagerMacro(_id) => {
                        return None;
                    }
                };
                let loc: MacroCallLoc = db.lookup_intern_macro(lazy_id);
                let item = match loc.def.kind {
                    MacroDefKind::BuiltInDerive(..) => loc.kind.node(db),
                    _ => return None,
                };
                Some(item.with_value(ast::Item::cast(item.value.clone())?))
            }
        }
    }

    /// Return whether this file is an include macro
    pub fn is_include_macro(&self, db: &dyn db::AstDatabase) -> bool {
        match self.0 {
            HirFileIdRepr::MacroFile(macro_file) => match macro_file.macro_call_id {
                MacroCallId::EagerMacro(id) => {
                    let loc = db.lookup_intern_eager_expansion(id);
                    return loc.included_file.is_some();
                }
                _ => {}
            },
            _ => {}
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroFile {
    macro_call_id: MacroCallId,
}

/// `MacroCallId` identifies a particular macro invocation, like
/// `println!("Hello, {}", world)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroCallId {
    LazyMacro(LazyMacroId),
    EagerMacro(EagerMacroId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LazyMacroId(salsa::InternId);
impl_intern_key!(LazyMacroId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EagerMacroId(salsa::InternId);
impl_intern_key!(EagerMacroId);

impl From<LazyMacroId> for MacroCallId {
    fn from(it: LazyMacroId) -> Self {
        MacroCallId::LazyMacro(it)
    }
}
impl From<EagerMacroId> for MacroCallId {
    fn from(it: EagerMacroId) -> Self {
        MacroCallId::EagerMacro(it)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDefId {
    pub krate: CrateId,
    pub kind: MacroDefKind,

    pub local_inner: bool,
}

impl MacroDefId {
    pub fn as_lazy_macro(
        self,
        db: &dyn db::AstDatabase,
        krate: CrateId,
        kind: MacroCallKind,
    ) -> LazyMacroId {
        db.intern_macro(MacroCallLoc { def: self, krate, kind })
    }

    pub fn ast_id(&self) -> Either<AstId<ast::Macro>, AstId<ast::Fn>> {
        let id = match &self.kind {
            MacroDefKind::Declarative(id) => id,
            MacroDefKind::BuiltIn(_, id) => id,
            MacroDefKind::BuiltInDerive(_, id) => id,
            MacroDefKind::BuiltInEager(_, id) => id,
            MacroDefKind::ProcMacro(_, id) => return Either::Right(*id),
        };
        Either::Left(*id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroDefKind {
    Declarative(AstId<ast::Macro>),
    BuiltIn(BuiltinFnLikeExpander, AstId<ast::Macro>),
    // FIXME: maybe just Builtin and rename BuiltinFnLikeExpander to BuiltinExpander
    BuiltInDerive(BuiltinDeriveExpander, AstId<ast::Macro>),
    BuiltInEager(EagerExpander, AstId<ast::Macro>),
    ProcMacro(ProcMacroExpander, AstId<ast::Fn>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub(crate) def: MacroDefId,
    pub(crate) krate: CrateId,
    pub kind: MacroCallKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroCallKind {
    FnLike(AstId<ast::MacroCall>),
    Derive(AstId<ast::Item>, String),
}

impl MacroCallKind {
    fn file_id(&self) -> HirFileId {
        match self {
            MacroCallKind::FnLike(ast_id) => ast_id.file_id,
            MacroCallKind::Derive(ast_id, _) => ast_id.file_id,
        }
    }

    fn node(&self, db: &dyn db::AstDatabase) -> InFile<SyntaxNode> {
        match self {
            MacroCallKind::FnLike(ast_id) => ast_id.with_value(ast_id.to_node(db).syntax().clone()),
            MacroCallKind::Derive(ast_id, _) => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
        }
    }

    fn arg(&self, db: &dyn db::AstDatabase) -> Option<SyntaxNode> {
        match self {
            MacroCallKind::FnLike(ast_id) => {
                Some(ast_id.to_node(db).token_tree()?.syntax().clone())
            }
            MacroCallKind::Derive(ast_id, _) => Some(ast_id.to_node(db).syntax().clone()),
        }
    }
}

impl MacroCallId {
    pub fn as_file(self) -> HirFileId {
        MacroFile { macro_call_id: self }.into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EagerCallLoc {
    pub(crate) def: MacroDefId,
    pub(crate) fragment: FragmentKind,
    pub(crate) subtree: Arc<tt::Subtree>,
    pub(crate) krate: CrateId,
    pub(crate) call: AstId<ast::MacroCall>,
    // The included file ID of the include macro.
    pub(crate) included_file: Option<FileId>,
}

/// ExpansionInfo mainly describes how to map text range between src and expanded macro
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpansionInfo {
    expanded: InFile<SyntaxNode>,
    arg: InFile<SyntaxNode>,
    /// The `macro_rules!` arguments.
    def: Option<InFile<ast::TokenTree>>,

    macro_def: Arc<(db::TokenExpander, mbe::TokenMap)>,
    macro_arg: Arc<(tt::Subtree, mbe::TokenMap)>,
    exp_map: Arc<mbe::TokenMap>,
}

pub use mbe::Origin;
use parser::FragmentKind;

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

        let token = self.expanded.value.covering_element(range).into_token()?;

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
            mbe::Origin::Def => (
                &self.macro_def.1,
                self.def
                    .as_ref()
                    .expect("`Origin::Def` used with non-`macro_rules!` macro")
                    .as_ref()
                    .map(|tt| tt.syntax().clone()),
            ),
        };

        let range = token_map.range_by_token(token_id)?.by_kind(token.value.kind())?;
        let token =
            tt.value.covering_element(range + tt.value.text_range().start()).into_token()?;
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
/// * `InFile<TextSize>` -- offset in a file
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
    pub fn file_syntax(&self, db: &dyn db::AstDatabase) -> SyntaxNode {
        db.parse_or_expand(self.file_id).expect("source created from invalid file")
    }
}

impl<T: Clone> InFile<&T> {
    pub fn cloned(&self) -> InFile<T> {
        self.with_value(self.value.clone())
    }
}

impl<T> InFile<Option<T>> {
    pub fn transpose(self) -> Option<InFile<T>> {
        let value = self.value?;
        Some(InFile::new(self.file_id, value))
    }
}

impl InFile<SyntaxNode> {
    pub fn ancestors_with_macros(
        self,
        db: &dyn db::AstDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + '_ {
        std::iter::successors(Some(self), move |node| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let parent_node = node.file_id.call_node(db)?;
                Some(parent_node)
            }
        })
    }
}

impl<'a> InFile<&'a SyntaxNode> {
    pub fn original_file_range(self, db: &dyn db::AstDatabase) -> FileRange {
        if let Some(range) = original_range_opt(db, self) {
            let original_file = range.file_id.original_file(db);
            if range.file_id == original_file.into() {
                return FileRange { file_id: original_file, range: range.value };
            }

            log::error!("Fail to mapping up more for {:?}", range);
            return FileRange { file_id: range.file_id.original_file(db), range: range.value };
        }

        // Fall back to whole macro call.
        let mut node = self.cloned();
        while let Some(call_node) = node.file_id.call_node(db) {
            node = call_node;
        }

        let orig_file = node.file_id.original_file(db);
        assert_eq!(node.file_id, orig_file.into());
        FileRange { file_id: orig_file, range: node.value.text_range() }
    }
}

fn original_range_opt(
    db: &dyn db::AstDatabase,
    node: InFile<&SyntaxNode>,
) -> Option<InFile<TextRange>> {
    let expansion = node.file_id.expansion_info(db)?;

    // the input node has only one token ?
    let single = skip_trivia_token(node.value.first_token()?, Direction::Next)?
        == skip_trivia_token(node.value.last_token()?, Direction::Prev)?;

    node.value.descendants().find_map(|it| {
        let first = skip_trivia_token(it.first_token()?, Direction::Next)?;
        let first = ascend_call_token(db, &expansion, node.with_value(first))?;

        let last = skip_trivia_token(it.last_token()?, Direction::Prev)?;
        let last = ascend_call_token(db, &expansion, node.with_value(last))?;

        if (!single && first == last) || (first.file_id != last.file_id) {
            return None;
        }

        Some(first.with_value(first.value.text_range().cover(last.value.text_range())))
    })
}

fn ascend_call_token(
    db: &dyn db::AstDatabase,
    expansion: &ExpansionInfo,
    token: InFile<SyntaxToken>,
) -> Option<InFile<SyntaxToken>> {
    let (mapped, origin) = expansion.map_token_up(token.as_ref())?;
    if origin != Origin::Call {
        return None;
    }
    if let Some(info) = mapped.file_id.expansion_info(db) {
        return ascend_call_token(db, &info, mapped);
    }
    Some(mapped)
}

impl InFile<SyntaxToken> {
    pub fn ancestors_with_macros(
        self,
        db: &dyn db::AstDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + '_ {
        self.value
            .parent()
            .into_iter()
            .flat_map(move |parent| InFile::new(self.file_id, parent).ancestors_with_macros(db))
    }
}

impl<N: AstNode> InFile<N> {
    pub fn descendants<T: AstNode>(self) -> impl Iterator<Item = InFile<T>> {
        self.value.syntax().descendants().filter_map(T::cast).map(move |n| self.with_value(n))
    }

    pub fn syntax(&self) -> InFile<&SyntaxNode> {
        self.with_value(self.value.syntax())
    }
}
