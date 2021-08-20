//! `hir_expand` deals with macro expansion.
//!
//! Specifically, it implements a concept of `MacroFile` -- a file whose syntax
//! tree originates not from the text of some `FileId`, but from some macro
//! expansion.

pub mod db;
pub mod ast_id_map;
pub mod name;
pub mod hygiene;
pub mod builtin_attr;
pub mod builtin_derive;
pub mod builtin_macro;
pub mod proc_macro;
pub mod quote;
pub mod eager;
mod input;

use base_db::ProcMacroKind;
use either::Either;

pub use mbe::{ExpandError, ExpandResult};
pub use parser::FragmentKind;

use std::{hash::Hash, iter, sync::Arc};

use base_db::{impl_intern_key, salsa, CrateId, FileId, FileRange};
use syntax::{
    algo::skip_trivia_token,
    ast::{self, AstNode},
    Direction, SyntaxNode, SyntaxToken, TextRange, TextSize,
};

use crate::{
    ast_id_map::FileAstId,
    builtin_attr::BuiltinAttrExpander,
    builtin_derive::BuiltinDeriveExpander,
    builtin_macro::{BuiltinFnLikeExpander, EagerExpander},
    proc_macro::ProcMacroExpander,
};

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
/// <https://en.wikipedia.org/wiki/String_interning>).
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
                let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);
                let file_id = match &loc.eager {
                    Some(EagerCallInfo { included_file: Some(file), .. }) => (*file).into(),
                    _ => loc.kind.file_id(),
                };
                file_id.original_file(db)
            }
        }
    }

    pub fn expansion_level(self, db: &dyn db::AstDatabase) -> u32 {
        let mut level = 0;
        let mut curr = self;
        while let HirFileIdRepr::MacroFile(macro_file) = curr.0 {
            let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);

            level += 1;
            curr = loc.kind.file_id();
        }
        level
    }

    /// If this is a macro call, returns the syntax node of the call.
    pub fn call_node(self, db: &dyn db::AstDatabase) -> Option<InFile<SyntaxNode>> {
        match self.0 {
            HirFileIdRepr::FileId(_) => None,
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);
                Some(loc.kind.to_node(db))
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

                let def = loc.def.ast_id().left().and_then(|id| {
                    let def_tt = match id.to_node(db) {
                        ast::Macro::MacroRules(mac) => mac.token_tree()?,
                        ast::Macro::MacroDef(mac) => mac.body()?,
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
                let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);
                let item = match loc.def.kind {
                    MacroDefKind::BuiltInDerive(..) => loc.kind.to_node(db),
                    _ => return None,
                };
                Some(item.with_value(ast::Item::cast(item.value.clone())?))
            }
        }
    }

    /// Return whether this file is an include macro
    pub fn is_include_macro(&self, db: &dyn db::AstDatabase) -> bool {
        match self.0 {
            HirFileIdRepr::MacroFile(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);
                matches!(loc.eager, Some(EagerCallInfo { included_file: Some(_), .. }))
            }
            _ => false,
        }
    }

    pub fn is_macro(self) -> bool {
        matches!(self.0, HirFileIdRepr::MacroFile(_))
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
impl_intern_key!(MacroCallId);

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
    ) -> MacroCallId {
        db.intern_macro(MacroCallLoc { def: self, krate, eager: None, kind })
    }

    pub fn ast_id(&self) -> Either<AstId<ast::Macro>, AstId<ast::Fn>> {
        let id = match &self.kind {
            MacroDefKind::ProcMacro(.., id) => return Either::Right(*id),
            MacroDefKind::Declarative(id)
            | MacroDefKind::BuiltIn(_, id)
            | MacroDefKind::BuiltInAttr(_, id)
            | MacroDefKind::BuiltInDerive(_, id)
            | MacroDefKind::BuiltInEager(_, id) => id,
        };
        Either::Left(*id)
    }

    pub fn is_proc_macro(&self) -> bool {
        matches!(self.kind, MacroDefKind::ProcMacro(..))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroDefKind {
    Declarative(AstId<ast::Macro>),
    BuiltIn(BuiltinFnLikeExpander, AstId<ast::Macro>),
    // FIXME: maybe just Builtin and rename BuiltinFnLikeExpander to BuiltinExpander
    BuiltInAttr(BuiltinAttrExpander, AstId<ast::Macro>),
    BuiltInDerive(BuiltinDeriveExpander, AstId<ast::Macro>),
    BuiltInEager(EagerExpander, AstId<ast::Macro>),
    ProcMacro(ProcMacroExpander, ProcMacroKind, AstId<ast::Fn>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EagerCallInfo {
    /// NOTE: This can be *either* the expansion result, *or* the argument to the eager macro!
    arg_or_expansion: Arc<tt::Subtree>,
    included_file: Option<FileId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub def: MacroDefId,
    pub(crate) krate: CrateId,
    eager: Option<EagerCallInfo>,
    pub kind: MacroCallKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroCallKind {
    FnLike {
        ast_id: AstId<ast::MacroCall>,
        fragment: FragmentKind,
    },
    Derive {
        ast_id: AstId<ast::Item>,
        derive_name: String,
        /// Syntactical index of the invoking `#[derive]` attribute.
        ///
        /// Outer attributes are counted first, then inner attributes. This does not support
        /// out-of-line modules, which may have attributes spread across 2 files!
        derive_attr_index: u32,
    },
    Attr {
        ast_id: AstId<ast::Item>,
        attr_name: String,
        attr_args: tt::Subtree,
        /// Syntactical index of the invoking `#[attribute]`.
        ///
        /// Outer attributes are counted first, then inner attributes. This does not support
        /// out-of-line modules, which may have attributes spread across 2 files!
        invoc_attr_index: u32,
    },
}

// FIXME: attribute indices do not account for `cfg_attr`, which means that we'll strip the whole
// `cfg_attr` instead of just one of the attributes it expands to

impl MacroCallKind {
    /// Returns the file containing the macro invocation.
    fn file_id(&self) -> HirFileId {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => ast_id.file_id,
            MacroCallKind::Derive { ast_id, .. } | MacroCallKind::Attr { ast_id, .. } => {
                ast_id.file_id
            }
        }
    }

    pub fn to_node(&self, db: &dyn db::AstDatabase) -> InFile<SyntaxNode> {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
            MacroCallKind::Derive { ast_id, .. } | MacroCallKind::Attr { ast_id, .. } => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
        }
    }

    fn arg(&self, db: &dyn db::AstDatabase) -> Option<SyntaxNode> {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => {
                Some(ast_id.to_node(db).token_tree()?.syntax().clone())
            }
            MacroCallKind::Derive { ast_id, .. } | MacroCallKind::Attr { ast_id, .. } => {
                Some(ast_id.to_node(db).syntax().clone())
            }
        }
    }

    fn fragment_kind(&self) -> FragmentKind {
        match self {
            MacroCallKind::FnLike { fragment, .. } => *fragment,
            MacroCallKind::Derive { .. } => FragmentKind::Items,
            MacroCallKind::Attr { .. } => FragmentKind::Items, // is this always correct?
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
    /// The `macro_rules!` arguments.
    def: Option<InFile<ast::TokenTree>>,

    macro_def: Arc<db::TokenExpander>,
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
        let token_id = self.macro_def.map_id_down(token_id);

        let range = self.exp_map.range_by_token(token_id, token.value.kind())?;

        let token = self.expanded.value.covering_element(range).into_token()?;

        Some(self.expanded.with_value(token))
    }

    pub fn map_token_up(
        &self,
        token: InFile<&SyntaxToken>,
    ) -> Option<(InFile<SyntaxToken>, Origin)> {
        let token_id = self.exp_map.token_by_range(token.value.text_range())?;

        let (token_id, origin) = self.macro_def.map_id_up(token_id);
        let (token_map, tt) = match origin {
            mbe::Origin::Call => (&self.macro_arg.1, self.arg.clone()),
            mbe::Origin::Def => match (&*self.macro_def, self.def.as_ref()) {
                (
                    db::TokenExpander::MacroRules { def_site_token_map, .. }
                    | db::TokenExpander::MacroDef { def_site_token_map, .. },
                    Some(tt),
                ) => (def_site_token_map, tt.syntax().cloned()),
                _ => panic!("`Origin::Def` used with non-`macro_rules!` macro"),
            },
        };

        let range = token_map.range_by_token(token_id, token.value.kind())?;
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
        iter::successors(Some(self), move |node| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let parent_node = node.file_id.call_node(db)?;
                Some(parent_node)
            }
        })
    }
}

impl<'a> InFile<&'a SyntaxNode> {
    /// Falls back to the macro call range if the node cannot be mapped up fully.
    pub fn original_file_range(self, db: &dyn db::AstDatabase) -> FileRange {
        if let Some(res) = self.original_file_range_opt(db) {
            return res;
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

    /// Attempts to map the syntax node back up its macro calls.
    pub fn original_file_range_opt(self, db: &dyn db::AstDatabase) -> Option<FileRange> {
        match original_range_opt(db, self) {
            Some(range) => {
                let original_file = range.file_id.original_file(db);
                if range.file_id != original_file.into() {
                    log::error!("Failed mapping up more for {:?}", range);
                }
                Some(FileRange { file_id: original_file, range: range.value })
            }
            _ if !self.file_id.is_macro() => Some(FileRange {
                file_id: self.file_id.original_file(db),
                range: self.value.text_range(),
            }),
            _ => None,
        }
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
        self.value.parent().into_iter().flat_map({
            let file_id = self.file_id;
            move |parent| InFile::new(file_id, parent).ancestors_with_macros(db)
        })
    }
}

impl<N: AstNode> InFile<N> {
    pub fn descendants<T: AstNode>(self) -> impl Iterator<Item = InFile<T>> {
        self.value.syntax().descendants().filter_map(T::cast).map(move |n| self.with_value(n))
    }

    pub fn syntax(&self) -> InFile<&SyntaxNode> {
        self.with_value(self.value.syntax())
    }

    pub fn nodes_with_attributes<'db>(
        self,
        db: &'db dyn db::AstDatabase,
    ) -> impl Iterator<Item = InFile<N>> + 'db
    where
        N: 'db,
    {
        iter::successors(Some(self), move |node| {
            let InFile { file_id, value } = node.file_id.call_node(db)?;
            N::cast(value).map(|n| InFile::new(file_id, n))
        })
    }

    pub fn node_with_attributes(self, db: &dyn db::AstDatabase) -> InFile<N> {
        self.nodes_with_attributes(db).last().unwrap()
    }
}

/// Given a `MacroCallId`, return what `FragmentKind` it belongs to.
/// FIXME: Not completed
pub fn to_fragment_kind(call: &ast::MacroCall) -> FragmentKind {
    use syntax::SyntaxKind::*;

    let syn = call.syntax();

    let parent = match syn.parent() {
        Some(it) => it,
        None => return FragmentKind::Statements,
    };

    match parent.kind() {
        MACRO_ITEMS | SOURCE_FILE => FragmentKind::Items,
        MACRO_STMTS => FragmentKind::Statements,
        MACRO_PAT => FragmentKind::Pattern,
        MACRO_TYPE => FragmentKind::Type,
        ITEM_LIST => FragmentKind::Items,
        LET_STMT => {
            // FIXME: Handle LHS Pattern
            FragmentKind::Expr
        }
        EXPR_STMT => FragmentKind::Statements,
        BLOCK_EXPR => FragmentKind::Statements,
        ARG_LIST => FragmentKind::Expr,
        TRY_EXPR => FragmentKind::Expr,
        TUPLE_EXPR => FragmentKind::Expr,
        PAREN_EXPR => FragmentKind::Expr,
        ARRAY_EXPR => FragmentKind::Expr,
        FOR_EXPR => FragmentKind::Expr,
        PATH_EXPR => FragmentKind::Expr,
        CLOSURE_EXPR => FragmentKind::Expr,
        CONDITION => FragmentKind::Expr,
        BREAK_EXPR => FragmentKind::Expr,
        RETURN_EXPR => FragmentKind::Expr,
        MATCH_EXPR => FragmentKind::Expr,
        MATCH_ARM => FragmentKind::Expr,
        MATCH_GUARD => FragmentKind::Expr,
        RECORD_EXPR_FIELD => FragmentKind::Expr,
        CALL_EXPR => FragmentKind::Expr,
        INDEX_EXPR => FragmentKind::Expr,
        METHOD_CALL_EXPR => FragmentKind::Expr,
        FIELD_EXPR => FragmentKind::Expr,
        AWAIT_EXPR => FragmentKind::Expr,
        CAST_EXPR => FragmentKind::Expr,
        REF_EXPR => FragmentKind::Expr,
        PREFIX_EXPR => FragmentKind::Expr,
        RANGE_EXPR => FragmentKind::Expr,
        BIN_EXPR => FragmentKind::Expr,
        _ => {
            // Unknown , Just guess it is `Items`
            FragmentKind::Items
        }
    }
}
