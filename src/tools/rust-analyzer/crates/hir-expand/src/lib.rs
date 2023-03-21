//! `hir_expand` deals with macro expansion.
//!
//! Specifically, it implements a concept of `MacroFile` -- a file whose syntax
//! tree originates not from the text of some `FileId`, but from some macro
//! expansion.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

pub mod db;
pub mod ast_id_map;
pub mod name;
pub mod hygiene;
pub mod builtin_attr_macro;
pub mod builtin_derive_macro;
pub mod builtin_fn_macro;
pub mod proc_macro;
pub mod quote;
pub mod eager;
pub mod mod_path;
pub mod attrs;
mod fixup;

pub use mbe::{Origin, ValueResult};

use ::tt::token_id as tt;

use std::{fmt, hash::Hash, iter, sync::Arc};

use base_db::{
    impl_intern_key,
    salsa::{self, InternId},
    CrateId, FileId, FileRange, ProcMacroKind,
};
use either::Either;
use syntax::{
    algo::{self, skip_trivia_token},
    ast::{self, AstNode, HasDocComments},
    Direction, SyntaxNode, SyntaxToken,
};

use crate::{
    ast_id_map::FileAstId,
    attrs::AttrId,
    builtin_attr_macro::BuiltinAttrExpander,
    builtin_derive_macro::BuiltinDeriveExpander,
    builtin_fn_macro::{BuiltinFnLikeExpander, EagerExpander},
    db::TokenExpander,
    mod_path::ModPath,
    proc_macro::ProcMacroExpander,
};

pub type ExpandResult<T> = ValueResult<T, ExpandError>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ExpandError {
    UnresolvedProcMacro(CrateId),
    Mbe(mbe::ExpandError),
    RecursionOverflowPosioned,
    Other(Box<str>),
}

impl From<mbe::ExpandError> for ExpandError {
    fn from(mbe: mbe::ExpandError) -> Self {
        Self::Mbe(mbe)
    }
}

impl fmt::Display for ExpandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpandError::UnresolvedProcMacro(_) => f.write_str("unresolved proc-macro"),
            ExpandError::Mbe(it) => it.fmt(f),
            ExpandError::RecursionOverflowPosioned => {
                f.write_str("overflow expanding the original macro")
            }
            ExpandError::Other(it) => f.write_str(it),
        }
    }
}

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
///
/// The two variants are encoded in a single u32 which are differentiated by the MSB.
/// If the MSB is 0, the value represents a `FileId`, otherwise the remaining 31 bits represent a
/// `MacroCallId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirFileId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroFile {
    pub macro_call_id: MacroCallId,
}

/// `MacroCallId` identifies a particular macro invocation, like
/// `println!("Hello, {}", world)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroCallId(salsa::InternId);
impl_intern_key!(MacroCallId);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub def: MacroDefId,
    pub(crate) krate: CrateId,
    eager: Option<EagerCallInfo>,
    pub kind: MacroCallKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDefId {
    pub krate: CrateId,
    pub kind: MacroDefKind,
    pub local_inner: bool,
    pub allow_internal_unsafe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroDefKind {
    Declarative(AstId<ast::Macro>),
    BuiltIn(BuiltinFnLikeExpander, AstId<ast::Macro>),
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
pub enum MacroCallKind {
    FnLike {
        ast_id: AstId<ast::MacroCall>,
        expand_to: ExpandTo,
    },
    Derive {
        ast_id: AstId<ast::Adt>,
        /// Syntactical index of the invoking `#[derive]` attribute.
        ///
        /// Outer attributes are counted first, then inner attributes. This does not support
        /// out-of-line modules, which may have attributes spread across 2 files!
        derive_attr_index: AttrId,
        /// Index of the derive macro in the derive attribute
        derive_index: u32,
    },
    Attr {
        ast_id: AstId<ast::Item>,
        attr_args: Arc<(tt::Subtree, mbe::TokenMap)>,
        /// Syntactical index of the invoking `#[attribute]`.
        ///
        /// Outer attributes are counted first, then inner attributes. This does not support
        /// out-of-line modules, which may have attributes spread across 2 files!
        invoc_attr_index: AttrId,
        /// Whether this attribute is the `#[derive]` attribute.
        is_derive: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HirFileIdRepr {
    FileId(FileId),
    MacroFile(MacroFile),
}

impl From<FileId> for HirFileId {
    fn from(FileId(id): FileId) -> Self {
        assert!(id < Self::MAX_FILE_ID);
        HirFileId(id)
    }
}

impl From<MacroFile> for HirFileId {
    fn from(MacroFile { macro_call_id: MacroCallId(id) }: MacroFile) -> Self {
        let id = id.as_u32();
        assert!(id < Self::MAX_FILE_ID);
        HirFileId(id | Self::MACRO_FILE_TAG_MASK)
    }
}

impl HirFileId {
    const MAX_FILE_ID: u32 = u32::MAX ^ Self::MACRO_FILE_TAG_MASK;
    const MACRO_FILE_TAG_MASK: u32 = 1 << 31;

    /// For macro-expansion files, returns the file original source file the
    /// expansion originated from.
    pub fn original_file(self, db: &dyn db::ExpandDatabase) -> FileId {
        let mut file_id = self;
        loop {
            match file_id.repr() {
                HirFileIdRepr::FileId(id) => break id,
                HirFileIdRepr::MacroFile(MacroFile { macro_call_id }) => {
                    let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_call_id);
                    file_id = match loc.eager {
                        Some(EagerCallInfo { included_file: Some(file), .. }) => file.into(),
                        _ => loc.kind.file_id(),
                    };
                }
            }
        }
    }

    pub fn expansion_level(self, db: &dyn db::ExpandDatabase) -> u32 {
        let mut level = 0;
        let mut curr = self;
        while let Some(macro_file) = curr.macro_file() {
            let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);

            level += 1;
            curr = loc.kind.file_id();
        }
        level
    }

    /// If this is a macro call, returns the syntax node of the call.
    pub fn call_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxNode>> {
        let macro_file = self.macro_file()?;
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
        Some(loc.kind.to_node(db))
    }

    /// If this is a macro call, returns the syntax node of the very first macro call this file resides in.
    pub fn original_call_node(self, db: &dyn db::ExpandDatabase) -> Option<(FileId, SyntaxNode)> {
        let mut call =
            db.lookup_intern_macro_call(self.macro_file()?.macro_call_id).kind.to_node(db);
        loop {
            match call.file_id.repr() {
                HirFileIdRepr::FileId(file_id) => break Some((file_id, call.value)),
                HirFileIdRepr::MacroFile(MacroFile { macro_call_id }) => {
                    call = db.lookup_intern_macro_call(macro_call_id).kind.to_node(db);
                }
            }
        }
    }

    /// Return expansion information if it is a macro-expansion file
    pub fn expansion_info(self, db: &dyn db::ExpandDatabase) -> Option<ExpansionInfo> {
        let macro_file = self.macro_file()?;
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);

        let arg_tt = loc.kind.arg(db)?;

        let macro_def = db.macro_def(loc.def).ok()?;
        let (parse, exp_map) = db.parse_macro_expansion(macro_file).value?;
        let macro_arg = db.macro_arg(macro_file.macro_call_id)?;

        let def = loc.def.ast_id().left().and_then(|id| {
            let def_tt = match id.to_node(db) {
                ast::Macro::MacroRules(mac) => mac.token_tree()?,
                ast::Macro::MacroDef(_) if matches!(*macro_def, TokenExpander::BuiltinAttr(_)) => {
                    return None
                }
                ast::Macro::MacroDef(mac) => mac.body()?,
            };
            Some(InFile::new(id.file_id, def_tt))
        });
        let attr_input_or_mac_def = def.or_else(|| match loc.kind {
            MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => {
                // FIXME: handle `cfg_attr`
                let tt = ast_id
                    .to_node(db)
                    .doc_comments_and_attrs()
                    .nth(invoc_attr_index.ast_index())
                    .and_then(Either::left)?
                    .token_tree()?;
                Some(InFile::new(ast_id.file_id, tt))
            }
            _ => None,
        });

        Some(ExpansionInfo {
            expanded: InFile::new(self, parse.syntax_node()),
            arg: InFile::new(loc.kind.file_id(), arg_tt),
            attr_input_or_mac_def,
            macro_arg_shift: mbe::Shift::new(&macro_arg.0),
            macro_arg,
            macro_def,
            exp_map,
        })
    }

    /// Indicate it is macro file generated for builtin derive
    pub fn is_builtin_derive(&self, db: &dyn db::ExpandDatabase) -> Option<InFile<ast::Attr>> {
        let macro_file = self.macro_file()?;
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
        let attr = match loc.def.kind {
            MacroDefKind::BuiltInDerive(..) => loc.kind.to_node(db),
            _ => return None,
        };
        Some(attr.with_value(ast::Attr::cast(attr.value.clone())?))
    }

    pub fn is_custom_derive(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                matches!(loc.def.kind, MacroDefKind::ProcMacro(_, ProcMacroKind::CustomDerive, _))
            }
            None => false,
        }
    }

    /// Return whether this file is an include macro
    pub fn is_include_macro(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                matches!(loc.eager, Some(EagerCallInfo { included_file: Some(_), .. }))
            }
            _ => false,
        }
    }

    /// Return whether this file is an attr macro
    pub fn is_attr_macro(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                matches!(loc.kind, MacroCallKind::Attr { .. })
            }
            _ => false,
        }
    }

    /// Return whether this file is the pseudo expansion of the derive attribute.
    /// See [`crate::builtin_attr_macro::derive_attr_expand`].
    pub fn is_derive_attr_pseudo_expansion(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                matches!(loc.kind, MacroCallKind::Attr { is_derive: true, .. })
            }
            None => false,
        }
    }

    #[inline]
    pub fn is_macro(self) -> bool {
        self.0 & Self::MACRO_FILE_TAG_MASK != 0
    }

    #[inline]
    pub fn macro_file(self) -> Option<MacroFile> {
        match self.0 & Self::MACRO_FILE_TAG_MASK {
            0 => None,
            _ => Some(MacroFile {
                macro_call_id: MacroCallId(InternId::from(self.0 ^ Self::MACRO_FILE_TAG_MASK)),
            }),
        }
    }

    #[inline]
    pub fn file_id(self) -> Option<FileId> {
        match self.0 & Self::MACRO_FILE_TAG_MASK {
            0 => Some(FileId(self.0)),
            _ => None,
        }
    }

    fn repr(self) -> HirFileIdRepr {
        match self.0 & Self::MACRO_FILE_TAG_MASK {
            0 => HirFileIdRepr::FileId(FileId(self.0)),
            _ => HirFileIdRepr::MacroFile(MacroFile {
                macro_call_id: MacroCallId(InternId::from(self.0 ^ Self::MACRO_FILE_TAG_MASK)),
            }),
        }
    }
}

impl MacroDefId {
    pub fn as_lazy_macro(
        self,
        db: &dyn db::ExpandDatabase,
        krate: CrateId,
        kind: MacroCallKind,
    ) -> MacroCallId {
        db.intern_macro_call(MacroCallLoc { def: self, krate, eager: None, kind })
    }

    pub fn ast_id(&self) -> Either<AstId<ast::Macro>, AstId<ast::Fn>> {
        let id = match self.kind {
            MacroDefKind::ProcMacro(.., id) => return Either::Right(id),
            MacroDefKind::Declarative(id)
            | MacroDefKind::BuiltIn(_, id)
            | MacroDefKind::BuiltInAttr(_, id)
            | MacroDefKind::BuiltInDerive(_, id)
            | MacroDefKind::BuiltInEager(_, id) => id,
        };
        Either::Left(id)
    }

    pub fn is_proc_macro(&self) -> bool {
        matches!(self.kind, MacroDefKind::ProcMacro(..))
    }

    pub fn is_attribute(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltInAttr(..) | MacroDefKind::ProcMacro(_, ProcMacroKind::Attr, _)
        )
    }
}

// FIXME: attribute indices do not account for nested `cfg_attr`

impl MacroCallKind {
    /// Returns the file containing the macro invocation.
    fn file_id(&self) -> HirFileId {
        match *self {
            MacroCallKind::FnLike { ast_id: InFile { file_id, .. }, .. }
            | MacroCallKind::Derive { ast_id: InFile { file_id, .. }, .. }
            | MacroCallKind::Attr { ast_id: InFile { file_id, .. }, .. } => file_id,
        }
    }

    pub fn to_node(&self, db: &dyn db::ExpandDatabase) -> InFile<SyntaxNode> {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
            MacroCallKind::Derive { ast_id, derive_attr_index, .. } => {
                // FIXME: handle `cfg_attr`
                ast_id.with_value(ast_id.to_node(db)).map(|it| {
                    it.doc_comments_and_attrs()
                        .nth(derive_attr_index.ast_index())
                        .and_then(|it| match it {
                            Either::Left(attr) => Some(attr.syntax().clone()),
                            Either::Right(_) => None,
                        })
                        .unwrap_or_else(|| it.syntax().clone())
                })
            }
            MacroCallKind::Attr { ast_id, is_derive: true, invoc_attr_index, .. } => {
                // FIXME: handle `cfg_attr`
                ast_id.with_value(ast_id.to_node(db)).map(|it| {
                    it.doc_comments_and_attrs()
                        .nth(invoc_attr_index.ast_index())
                        .and_then(|it| match it {
                            Either::Left(attr) => Some(attr.syntax().clone()),
                            Either::Right(_) => None,
                        })
                        .unwrap_or_else(|| it.syntax().clone())
                })
            }
            MacroCallKind::Attr { ast_id, .. } => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
        }
    }

    /// Returns the original file range that best describes the location of this macro call.
    ///
    /// Unlike `MacroCallKind::original_call_range`, this also spans the item of attributes and derives.
    pub fn original_call_range_with_body(self, db: &dyn db::ExpandDatabase) -> FileRange {
        let mut kind = self;
        let file_id = loop {
            match kind.file_id().repr() {
                HirFileIdRepr::MacroFile(file) => {
                    kind = db.lookup_intern_macro_call(file.macro_call_id).kind;
                }
                HirFileIdRepr::FileId(file_id) => break file_id,
            }
        };

        let range = match kind {
            MacroCallKind::FnLike { ast_id, .. } => ast_id.to_node(db).syntax().text_range(),
            MacroCallKind::Derive { ast_id, .. } => ast_id.to_node(db).syntax().text_range(),
            MacroCallKind::Attr { ast_id, .. } => ast_id.to_node(db).syntax().text_range(),
        };

        FileRange { range, file_id }
    }

    /// Returns the original file range that best describes the location of this macro call.
    ///
    /// Here we try to roughly match what rustc does to improve diagnostics: fn-like macros
    /// get the whole `ast::MacroCall`, attribute macros get the attribute's range, and derives
    /// get only the specific derive that is being referred to.
    pub fn original_call_range(self, db: &dyn db::ExpandDatabase) -> FileRange {
        let mut kind = self;
        let file_id = loop {
            match kind.file_id().repr() {
                HirFileIdRepr::MacroFile(file) => {
                    kind = db.lookup_intern_macro_call(file.macro_call_id).kind;
                }
                HirFileIdRepr::FileId(file_id) => break file_id,
            }
        };

        let range = match kind {
            MacroCallKind::FnLike { ast_id, .. } => ast_id.to_node(db).syntax().text_range(),
            MacroCallKind::Derive { ast_id, derive_attr_index, .. } => {
                // FIXME: should be the range of the macro name, not the whole derive
                // FIXME: handle `cfg_attr`
                ast_id
                    .to_node(db)
                    .doc_comments_and_attrs()
                    .nth(derive_attr_index.ast_index())
                    .expect("missing derive")
                    .expect_left("derive is a doc comment?")
                    .syntax()
                    .text_range()
            }
            // FIXME: handle `cfg_attr`
            MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => ast_id
                .to_node(db)
                .doc_comments_and_attrs()
                .nth(invoc_attr_index.ast_index())
                .expect("missing attribute")
                .expect_left("attribute macro is a doc comment?")
                .syntax()
                .text_range(),
        };

        FileRange { range, file_id }
    }

    fn arg(&self, db: &dyn db::ExpandDatabase) -> Option<SyntaxNode> {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => {
                Some(ast_id.to_node(db).token_tree()?.syntax().clone())
            }
            MacroCallKind::Derive { ast_id, .. } => Some(ast_id.to_node(db).syntax().clone()),
            MacroCallKind::Attr { ast_id, .. } => Some(ast_id.to_node(db).syntax().clone()),
        }
    }

    fn expand_to(&self) -> ExpandTo {
        match self {
            MacroCallKind::FnLike { expand_to, .. } => *expand_to,
            MacroCallKind::Derive { .. } => ExpandTo::Items,
            MacroCallKind::Attr { is_derive: true, .. } => ExpandTo::Statements,
            MacroCallKind::Attr { .. } => ExpandTo::Items, // is this always correct?
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
    /// The argument TokenTree or item for attributes
    arg: InFile<SyntaxNode>,
    /// The `macro_rules!` or attribute input.
    attr_input_or_mac_def: Option<InFile<ast::TokenTree>>,

    macro_def: Arc<TokenExpander>,
    macro_arg: Arc<(tt::Subtree, mbe::TokenMap, fixup::SyntaxFixupUndoInfo)>,
    /// A shift built from `macro_arg`'s subtree, relevant for attributes as the item is the macro arg
    /// and as such we need to shift tokens if they are part of an attributes input instead of their item.
    macro_arg_shift: mbe::Shift,
    exp_map: Arc<mbe::TokenMap>,
}

impl ExpansionInfo {
    pub fn expanded(&self) -> InFile<SyntaxNode> {
        self.expanded.clone()
    }

    pub fn call_node(&self) -> Option<InFile<SyntaxNode>> {
        Some(self.arg.with_value(self.arg.value.parent()?))
    }

    /// Map a token down from macro input into the macro expansion.
    ///
    /// The inner workings of this function differ slightly depending on the type of macro we are dealing with:
    /// - declarative:
    ///     For declarative macros, we need to accommodate for the macro definition site(which acts as a second unchanging input)
    ///     , as tokens can mapped in and out of it.
    ///     To do this we shift all ids in the expansion by the maximum id of the definition site giving us an easy
    ///     way to map all the tokens.
    /// - attribute:
    ///     Attributes have two different inputs, the input tokentree in the attribute node and the item
    ///     the attribute is annotating. Similarly as for declarative macros we need to do a shift here
    ///     as well. Currently this is done by shifting the attribute input by the maximum id of the item.
    /// - function-like and derives:
    ///     Both of these only have one simple call site input so no special handling is required here.
    pub fn map_token_down(
        &self,
        db: &dyn db::ExpandDatabase,
        item: Option<ast::Item>,
        token: InFile<&SyntaxToken>,
    ) -> Option<impl Iterator<Item = InFile<SyntaxToken>> + '_> {
        assert_eq!(token.file_id, self.arg.file_id);
        let token_id_in_attr_input = if let Some(item) = item {
            // check if we are mapping down in an attribute input
            // this is a special case as attributes can have two inputs
            let call_id = self.expanded.file_id.macro_file()?.macro_call_id;
            let loc = db.lookup_intern_macro_call(call_id);

            let token_range = token.value.text_range();
            match &loc.kind {
                MacroCallKind::Attr { attr_args, invoc_attr_index, is_derive, .. } => {
                    // FIXME: handle `cfg_attr`
                    let attr = item
                        .doc_comments_and_attrs()
                        .nth(invoc_attr_index.ast_index())
                        .and_then(Either::left)?;
                    match attr.token_tree() {
                        Some(token_tree)
                            if token_tree.syntax().text_range().contains_range(token_range) =>
                        {
                            let attr_input_start =
                                token_tree.left_delimiter_token()?.text_range().start();
                            let relative_range =
                                token.value.text_range().checked_sub(attr_input_start)?;
                            // shift by the item's tree's max id
                            let token_id = attr_args.1.token_by_range(relative_range)?;
                            let token_id = if *is_derive {
                                // we do not shift for `#[derive]`, as we only need to downmap the derive attribute tokens
                                token_id
                            } else {
                                self.macro_arg_shift.shift(token_id)
                            };
                            Some(token_id)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };

        let token_id = match token_id_in_attr_input {
            Some(token_id) => token_id,
            // the token is not inside an attribute's input so do the lookup in the macro_arg as usual
            None => {
                let relative_range =
                    token.value.text_range().checked_sub(self.arg.value.text_range().start())?;
                let token_id = self.macro_arg.1.token_by_range(relative_range)?;
                // conditionally shift the id by a declaratives macro definition
                self.macro_def.map_id_down(token_id)
            }
        };

        let tokens = self
            .exp_map
            .ranges_by_token(token_id, token.value.kind())
            .flat_map(move |range| self.expanded.value.covering_element(range).into_token());

        Some(tokens.map(move |token| self.expanded.with_value(token)))
    }

    /// Map a token up out of the expansion it resides in into the arguments of the macro call of the expansion.
    pub fn map_token_up(
        &self,
        db: &dyn db::ExpandDatabase,
        token: InFile<&SyntaxToken>,
    ) -> Option<(InFile<SyntaxToken>, Origin)> {
        // Fetch the id through its text range,
        let token_id = self.exp_map.token_by_range(token.value.text_range())?;
        // conditionally unshifting the id to accommodate for macro-rules def site
        let (mut token_id, origin) = self.macro_def.map_id_up(token_id);

        let call_id = self.expanded.file_id.macro_file()?.macro_call_id;
        let loc = db.lookup_intern_macro_call(call_id);

        // Attributes are a bit special for us, they have two inputs, the input tokentree and the annotated item.
        let (token_map, tt) = match &loc.kind {
            MacroCallKind::Attr { attr_args, is_derive: true, .. } => {
                (&attr_args.1, self.attr_input_or_mac_def.clone()?.syntax().cloned())
            }
            MacroCallKind::Attr { attr_args, .. } => {
                // try unshifting the the token id, if unshifting fails, the token resides in the non-item attribute input
                // note that the `TokenExpander::map_id_up` earlier only unshifts for declarative macros, so we don't double unshift with this
                match self.macro_arg_shift.unshift(token_id) {
                    Some(unshifted) => {
                        token_id = unshifted;
                        (&attr_args.1, self.attr_input_or_mac_def.clone()?.syntax().cloned())
                    }
                    None => (&self.macro_arg.1, self.arg.clone()),
                }
            }
            _ => match origin {
                mbe::Origin::Call => (&self.macro_arg.1, self.arg.clone()),
                mbe::Origin::Def => match (&*self.macro_def, &self.attr_input_or_mac_def) {
                    (TokenExpander::DeclarativeMacro { def_site_token_map, .. }, Some(tt)) => {
                        (def_site_token_map, tt.syntax().cloned())
                    }
                    _ => panic!("`Origin::Def` used with non-`macro_rules!` macro"),
                },
            },
        };

        let range = token_map.first_range_by_token(token_id, token.value.kind())?;
        let token =
            tt.value.covering_element(range + tt.value.text_range().start()).into_token()?;
        Some((tt.with_value(token), origin))
    }
}

/// `AstId` points to an AST node in any file.
///
/// It is stable across reparses, and can be used as salsa key/value.
pub type AstId<N> = InFile<FileAstId<N>>;

impl<N: AstNode> AstId<N> {
    pub fn to_node(&self, db: &dyn db::ExpandDatabase) -> N {
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

    pub fn with_value<U>(&self, value: U) -> InFile<U> {
        InFile::new(self.file_id, value)
    }

    pub fn map<F: FnOnce(T) -> U, U>(self, f: F) -> InFile<U> {
        InFile::new(self.file_id, f(self.value))
    }

    pub fn as_ref(&self) -> InFile<&T> {
        self.with_value(&self.value)
    }

    pub fn file_syntax(&self, db: &dyn db::ExpandDatabase) -> SyntaxNode {
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

impl<L, R> InFile<Either<L, R>> {
    pub fn transpose(self) -> Either<InFile<L>, InFile<R>> {
        match self.value {
            Either::Left(l) => Either::Left(InFile::new(self.file_id, l)),
            Either::Right(r) => Either::Right(InFile::new(self.file_id, r)),
        }
    }
}

impl<'a> InFile<&'a SyntaxNode> {
    pub fn ancestors_with_macros(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + Clone + '_ {
        iter::successors(Some(self.cloned()), move |node| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => node.file_id.call_node(db),
        })
    }

    /// Skips the attributed item that caused the macro invocation we are climbing up
    pub fn ancestors_with_macros_skip_attr_item(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + '_ {
        let succ = move |node: &InFile<SyntaxNode>| match node.value.parent() {
            Some(parent) => Some(node.with_value(parent)),
            None => {
                let parent_node = node.file_id.call_node(db)?;
                if node.file_id.is_attr_macro(db) {
                    // macro call was an attributed item, skip it
                    // FIXME: does this fail if this is a direct expansion of another macro?
                    parent_node.map(|node| node.parent()).transpose()
                } else {
                    Some(parent_node)
                }
            }
        };
        iter::successors(succ(&self.cloned()), succ)
    }

    /// Falls back to the macro call range if the node cannot be mapped up fully.
    ///
    /// For attributes and derives, this will point back to the attribute only.
    /// For the entire item use [`InFile::original_file_range_full`].
    pub fn original_file_range(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some(res) = self.original_file_range_opt(db) {
                    return res;
                }
                // Fall back to whole macro call.
                let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                loc.kind.original_call_range(db)
            }
        }
    }

    /// Falls back to the macro call range if the node cannot be mapped up fully.
    pub fn original_file_range_full(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some(res) = self.original_file_range_opt(db) {
                    return res;
                }
                // Fall back to whole macro call.
                let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                loc.kind.original_call_range_with_body(db)
            }
        }
    }

    /// Attempts to map the syntax node back up its macro calls.
    pub fn original_file_range_opt(self, db: &dyn db::ExpandDatabase) -> Option<FileRange> {
        match ascend_node_border_tokens(db, self) {
            Some(InFile { file_id, value: (first, last) }) => {
                let original_file = file_id.original_file(db);
                let range = first.text_range().cover(last.text_range());
                if file_id != original_file.into() {
                    tracing::error!("Failed mapping up more for {:?}", range);
                    return None;
                }
                Some(FileRange { file_id: original_file, range })
            }
            _ if !self.file_id.is_macro() => Some(FileRange {
                file_id: self.file_id.original_file(db),
                range: self.value.text_range(),
            }),
            _ => None,
        }
    }

    pub fn original_syntax_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxNode>> {
        // This kind of upmapping can only be achieved in attribute expanded files,
        // as we don't have node inputs otherwise and therefore can't find an `N` node in the input
        if !self.file_id.is_macro() {
            return Some(self.map(Clone::clone));
        } else if !self.file_id.is_attr_macro(db) {
            return None;
        }

        if let Some(InFile { file_id, value: (first, last) }) = ascend_node_border_tokens(db, self)
        {
            if file_id.is_macro() {
                let range = first.text_range().cover(last.text_range());
                tracing::error!("Failed mapping out of macro file for {:?}", range);
                return None;
            }
            // FIXME: This heuristic is brittle and with the right macro may select completely unrelated nodes
            let anc = algo::least_common_ancestor(&first.parent()?, &last.parent()?)?;
            let kind = self.value.kind();
            let value = anc.ancestors().find(|it| it.kind() == kind)?;
            return Some(InFile::new(file_id, value));
        }
        None
    }
}

impl InFile<SyntaxToken> {
    pub fn upmap(self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxToken>> {
        let expansion = self.file_id.expansion_info(db)?;
        expansion.map_token_up(db, self.as_ref()).map(|(it, _)| it)
    }

    /// Falls back to the macro call range if the node cannot be mapped up fully.
    pub fn original_file_range(self, db: &dyn db::ExpandDatabase) -> FileRange {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => FileRange { file_id, range: self.value.text_range() },
            HirFileIdRepr::MacroFile(mac_file) => {
                if let Some(res) = self.original_file_range_opt(db) {
                    return res;
                }
                // Fall back to whole macro call.
                let loc = db.lookup_intern_macro_call(mac_file.macro_call_id);
                loc.kind.original_call_range(db)
            }
        }
    }

    /// Attempts to map the syntax node back up its macro calls.
    pub fn original_file_range_opt(self, db: &dyn db::ExpandDatabase) -> Option<FileRange> {
        match self.file_id.repr() {
            HirFileIdRepr::FileId(file_id) => {
                Some(FileRange { file_id, range: self.value.text_range() })
            }
            HirFileIdRepr::MacroFile(_) => {
                let expansion = self.file_id.expansion_info(db)?;
                let InFile { file_id, value } = ascend_call_token(db, &expansion, self)?;
                let original_file = file_id.original_file(db);
                if file_id != original_file.into() {
                    return None;
                }
                Some(FileRange { file_id: original_file, range: value.text_range() })
            }
        }
    }

    pub fn ancestors_with_macros(
        self,
        db: &dyn db::ExpandDatabase,
    ) -> impl Iterator<Item = InFile<SyntaxNode>> + '_ {
        self.value.parent().into_iter().flat_map({
            let file_id = self.file_id;
            move |parent| InFile::new(file_id, &parent).ancestors_with_macros(db)
        })
    }
}

fn ascend_node_border_tokens(
    db: &dyn db::ExpandDatabase,
    InFile { file_id, value: node }: InFile<&SyntaxNode>,
) -> Option<InFile<(SyntaxToken, SyntaxToken)>> {
    let expansion = file_id.expansion_info(db)?;

    let first_token = |node: &SyntaxNode| skip_trivia_token(node.first_token()?, Direction::Next);
    let last_token = |node: &SyntaxNode| skip_trivia_token(node.last_token()?, Direction::Prev);

    let first = first_token(node)?;
    let last = last_token(node)?;
    let first = ascend_call_token(db, &expansion, InFile::new(file_id, first))?;
    let last = ascend_call_token(db, &expansion, InFile::new(file_id, last))?;
    (first.file_id == last.file_id).then(|| InFile::new(first.file_id, (first.value, last.value)))
}

fn ascend_call_token(
    db: &dyn db::ExpandDatabase,
    expansion: &ExpansionInfo,
    token: InFile<SyntaxToken>,
) -> Option<InFile<SyntaxToken>> {
    let mut mapping = expansion.map_token_up(db, token.as_ref())?;
    while let (mapped, Origin::Call) = mapping {
        match mapped.file_id.expansion_info(db) {
            Some(info) => mapping = info.map_token_up(db, mapped.as_ref())?,
            None => return Some(mapped),
        }
    }
    None
}

impl<N: AstNode> InFile<N> {
    pub fn descendants<T: AstNode>(self) -> impl Iterator<Item = InFile<T>> {
        self.value.syntax().descendants().filter_map(T::cast).map(move |n| self.with_value(n))
    }

    pub fn original_ast_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<N>> {
        // This kind of upmapping can only be achieved in attribute expanded files,
        // as we don't have node inputs otherwise and therefore can't find an `N` node in the input
        if !self.file_id.is_macro() {
            return Some(self);
        } else if !self.file_id.is_attr_macro(db) {
            return None;
        }

        if let Some(InFile { file_id, value: (first, last) }) =
            ascend_node_border_tokens(db, self.syntax())
        {
            if file_id.is_macro() {
                let range = first.text_range().cover(last.text_range());
                tracing::error!("Failed mapping out of macro file for {:?}", range);
                return None;
            }
            // FIXME: This heuristic is brittle and with the right macro may select completely unrelated nodes
            let anc = algo::least_common_ancestor(&first.parent()?, &last.parent()?)?;
            let value = anc.ancestors().find_map(N::cast)?;
            return Some(InFile::new(file_id, value));
        }
        None
    }

    pub fn syntax(&self) -> InFile<&SyntaxNode> {
        self.with_value(self.value.syntax())
    }
}

/// In Rust, macros expand token trees to token trees. When we want to turn a
/// token tree into an AST node, we need to figure out what kind of AST node we
/// want: something like `foo` can be a type, an expression, or a pattern.
///
/// Naively, one would think that "what this expands to" is a property of a
/// particular macro: macro `m1` returns an item, while macro `m2` returns an
/// expression, etc. That's not the case -- macros are polymorphic in the
/// result, and can expand to any type of the AST node.
///
/// What defines the actual AST node is the syntactic context of the macro
/// invocation. As a contrived example, in `let T![*] = T![*];` the first `T`
/// expands to a pattern, while the second one expands to an expression.
///
/// `ExpandTo` captures this bit of information about a particular macro call
/// site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpandTo {
    Statements,
    Items,
    Pattern,
    Type,
    Expr,
}

impl ExpandTo {
    pub fn from_call_site(call: &ast::MacroCall) -> ExpandTo {
        use syntax::SyntaxKind::*;

        let syn = call.syntax();

        let parent = match syn.parent() {
            Some(it) => it,
            None => return ExpandTo::Statements,
        };

        // FIXME: macros in statement position are treated as expression statements, they should
        // probably be their own statement kind. The *grand*parent indicates what's valid.
        if parent.kind() == MACRO_EXPR
            && parent
                .parent()
                .map_or(false, |p| matches!(p.kind(), EXPR_STMT | STMT_LIST | MACRO_STMTS))
        {
            return ExpandTo::Statements;
        }

        match parent.kind() {
            MACRO_ITEMS | SOURCE_FILE | ITEM_LIST => ExpandTo::Items,
            MACRO_STMTS | EXPR_STMT | STMT_LIST => ExpandTo::Statements,
            MACRO_PAT => ExpandTo::Pattern,
            MACRO_TYPE => ExpandTo::Type,

            ARG_LIST | ARRAY_EXPR | AWAIT_EXPR | BIN_EXPR | BREAK_EXPR | CALL_EXPR | CAST_EXPR
            | CLOSURE_EXPR | FIELD_EXPR | FOR_EXPR | IF_EXPR | INDEX_EXPR | LET_EXPR
            | MATCH_ARM | MATCH_EXPR | MATCH_GUARD | METHOD_CALL_EXPR | PAREN_EXPR | PATH_EXPR
            | PREFIX_EXPR | RANGE_EXPR | RECORD_EXPR_FIELD | REF_EXPR | RETURN_EXPR | TRY_EXPR
            | TUPLE_EXPR | WHILE_EXPR | MACRO_EXPR => ExpandTo::Expr,
            _ => {
                // Unknown , Just guess it is `Items`
                ExpandTo::Items
            }
        }
    }
}

#[derive(Debug)]
pub struct UnresolvedMacro {
    pub path: ModPath,
}

intern::impl_internable!(ModPath, attrs::AttrInput);
