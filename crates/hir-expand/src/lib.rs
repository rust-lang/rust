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
// mod fixup;

use triomphe::Arc;

use std::{fmt, hash::Hash, iter};

use base_db::{span::HirFileIdRepr, CrateId, FileId, FileRange, ProcMacroKind};
use either::Either;
use syntax::{
    algo::{self, skip_trivia_token},
    ast::{self, AstNode, HasDocComments},
    AstPtr, Direction, SyntaxNode, SyntaxNodePtr, SyntaxToken, TextSize,
};

use crate::{
    ast_id_map::{AstIdNode, ErasedFileAstId, FileAstId},
    attrs::AttrId,
    builtin_attr_macro::BuiltinAttrExpander,
    builtin_derive_macro::BuiltinDeriveExpander,
    builtin_fn_macro::{BuiltinFnLikeExpander, EagerExpander},
    db::TokenExpander,
    mod_path::ModPath,
    proc_macro::ProcMacroExpander,
};

pub use base_db::span::{HirFileId, MacroCallId, MacroFile};
pub use mbe::ValueResult;

pub type SpanMap = ::mbe::TokenMap<tt::SpanData>;
pub type DeclarativeMacro = ::mbe::DeclarativeMacro<tt::SpanData>;

pub mod tt {
    pub use base_db::span::SpanData;
    pub use tt::{DelimiterKind, Spacing, Span, SpanAnchor};

    pub type Delimiter = ::tt::Delimiter<SpanData>;
    pub type Subtree = ::tt::Subtree<SpanData>;
    pub type Leaf = ::tt::Leaf<SpanData>;
    pub type Literal = ::tt::Literal<SpanData>;
    pub type Punct = ::tt::Punct<SpanData>;
    pub type Ident = ::tt::Ident<SpanData>;
    pub type TokenTree = ::tt::TokenTree<SpanData>;
}

pub type ExpandResult<T> = ValueResult<T, ExpandError>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ExpandError {
    UnresolvedProcMacro(CrateId),
    Mbe(mbe::ExpandError),
    RecursionOverflowPoisoned,
    Other(Box<Box<str>>),
}

impl ExpandError {
    pub fn other(msg: impl Into<Box<str>>) -> Self {
        ExpandError::Other(Box::new(msg.into()))
    }
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
            ExpandError::RecursionOverflowPoisoned => {
                f.write_str("overflow expanding the original macro")
            }
            ExpandError::Other(it) => f.write_str(it),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub def: MacroDefId,
    pub(crate) krate: CrateId,
    /// Some if this is a macro call for an eager macro. Note that this is `None`
    /// for the eager input macro file.
    eager: Option<Box<EagerCallInfo>>,
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
    /// The expanded argument of the eager macro.
    arg: Arc<(tt::Subtree,)>,
    /// Call id of the eager macro's input file (this is the macro file for its fully expanded input).
    arg_id: MacroCallId,
    error: Option<ExpandError>,
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
        attr_args: Arc<tt::Subtree>,
        /// Syntactical index of the invoking `#[attribute]`.
        ///
        /// Outer attributes are counted first, then inner attributes. This does not support
        /// out-of-line modules, which may have attributes spread across 2 files!
        invoc_attr_index: AttrId,
    },
}

pub trait HirFileIdExt {
    /// For macro-expansion files, returns the file original source file the
    /// expansion originated from.
    fn original_file(self, db: &dyn db::ExpandDatabase) -> FileId;
    fn expansion_level(self, db: &dyn db::ExpandDatabase) -> u32;

    /// If this is a macro call, returns the syntax node of the call.
    fn call_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxNode>>;

    /// If this is a macro call, returns the syntax node of the very first macro call this file resides in.
    fn original_call_node(self, db: &dyn db::ExpandDatabase) -> Option<(FileId, SyntaxNode)>;

    /// Return expansion information if it is a macro-expansion file
    fn expansion_info(self, db: &dyn db::ExpandDatabase) -> Option<ExpansionInfo>;

    fn as_builtin_derive_attr_node(&self, db: &dyn db::ExpandDatabase)
        -> Option<InFile<ast::Attr>>;
    fn is_builtin_derive(&self, db: &dyn db::ExpandDatabase) -> bool;
    fn is_custom_derive(&self, db: &dyn db::ExpandDatabase) -> bool;

    /// Return whether this file is an include macro
    fn is_include_macro(&self, db: &dyn db::ExpandDatabase) -> bool;

    fn is_eager(&self, db: &dyn db::ExpandDatabase) -> bool;
    /// Return whether this file is an attr macro
    fn is_attr_macro(&self, db: &dyn db::ExpandDatabase) -> bool;

    /// Return whether this file is the pseudo expansion of the derive attribute.
    /// See [`crate::builtin_attr_macro::derive_attr_expand`].
    fn is_derive_attr_pseudo_expansion(&self, db: &dyn db::ExpandDatabase) -> bool;
}

impl HirFileIdExt for HirFileId {
    fn original_file(self, db: &dyn db::ExpandDatabase) -> FileId {
        let mut file_id = self;
        loop {
            match file_id.repr() {
                HirFileIdRepr::FileId(id) => break id,
                HirFileIdRepr::MacroFile(MacroFile { macro_call_id }) => {
                    let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_call_id);
                    let is_include_expansion = loc.def.is_include() && loc.eager.is_some();
                    file_id = match is_include_expansion.then(|| db.include_expand(macro_call_id)) {
                        Some(Ok((_, file))) => file.into(),
                        _ => loc.kind.file_id(),
                    }
                }
            }
        }
    }

    fn expansion_level(self, db: &dyn db::ExpandDatabase) -> u32 {
        let mut level = 0;
        let mut curr = self;
        while let Some(macro_file) = curr.macro_file() {
            let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);

            level += 1;
            curr = loc.kind.file_id();
        }
        level
    }

    fn call_node(self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxNode>> {
        let macro_file = self.macro_file()?;
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
        Some(loc.to_node(db))
    }

    fn original_call_node(self, db: &dyn db::ExpandDatabase) -> Option<(FileId, SyntaxNode)> {
        let mut call = db.lookup_intern_macro_call(self.macro_file()?.macro_call_id).to_node(db);
        loop {
            match call.file_id.repr() {
                HirFileIdRepr::FileId(file_id) => break Some((file_id, call.value)),
                HirFileIdRepr::MacroFile(MacroFile { macro_call_id }) => {
                    call = db.lookup_intern_macro_call(macro_call_id).to_node(db);
                }
            }
        }
    }

    /// Return expansion information if it is a macro-expansion file
    fn expansion_info(self, db: &dyn db::ExpandDatabase) -> Option<ExpansionInfo> {
        let macro_file = self.macro_file()?;
        ExpansionInfo::new(db, macro_file)
    }

    fn as_builtin_derive_attr_node(
        &self,
        db: &dyn db::ExpandDatabase,
    ) -> Option<InFile<ast::Attr>> {
        let macro_file = self.macro_file()?;
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
        let attr = match loc.def.kind {
            MacroDefKind::BuiltInDerive(..) => loc.to_node(db),
            _ => return None,
        };
        Some(attr.with_value(ast::Attr::cast(attr.value.clone())?))
    }

    fn is_custom_derive(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                matches!(
                    db.lookup_intern_macro_call(macro_file.macro_call_id).def.kind,
                    MacroDefKind::ProcMacro(_, ProcMacroKind::CustomDerive, _)
                )
            }
            None => false,
        }
    }

    fn is_builtin_derive(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                matches!(
                    db.lookup_intern_macro_call(macro_file.macro_call_id).def.kind,
                    MacroDefKind::BuiltInDerive(..)
                )
            }
            None => false,
        }
    }

    fn is_include_macro(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                db.lookup_intern_macro_call(macro_file.macro_call_id).def.is_include()
            }
            _ => false,
        }
    }

    fn is_eager(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                matches!(loc.def.kind, MacroDefKind::BuiltInEager(..))
            }
            _ => false,
        }
    }

    fn is_attr_macro(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                matches!(loc.kind, MacroCallKind::Attr { .. })
            }
            _ => false,
        }
    }

    fn is_derive_attr_pseudo_expansion(&self, db: &dyn db::ExpandDatabase) -> bool {
        match self.macro_file() {
            Some(macro_file) => {
                let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                loc.def.is_attribute_derive()
            }
            None => false,
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

    pub fn is_derive(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltInDerive(..)
                | MacroDefKind::ProcMacro(_, ProcMacroKind::CustomDerive, _)
        )
    }

    pub fn is_fn_like(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltIn(..)
                | MacroDefKind::ProcMacro(_, ProcMacroKind::FuncLike, _)
                | MacroDefKind::BuiltInEager(..)
                | MacroDefKind::Declarative(..)
        )
    }

    pub fn is_attribute_derive(&self) -> bool {
        matches!(self.kind, MacroDefKind::BuiltInAttr(expander, ..) if expander.is_derive())
    }

    pub fn is_include(&self) -> bool {
        matches!(self.kind, MacroDefKind::BuiltInEager(expander, ..) if expander.is_include())
    }
}

impl MacroCallLoc {
    pub fn to_node(&self, db: &dyn db::ExpandDatabase) -> InFile<SyntaxNode> {
        match self.kind {
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
            MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => {
                if self.def.is_attribute_derive() {
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
                } else {
                    ast_id.with_value(ast_id.to_node(db).syntax().clone())
                }
            }
        }
    }

    fn expand_to(&self) -> ExpandTo {
        match self.kind {
            MacroCallKind::FnLike { expand_to, .. } => expand_to,
            MacroCallKind::Derive { .. } => ExpandTo::Items,
            MacroCallKind::Attr { .. } if self.def.is_attribute_derive() => ExpandTo::Statements,
            MacroCallKind::Attr { .. } => {
                // is this always correct?
                ExpandTo::Items
            }
        }
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
            MacroCallKind::FnLike { ast_id, .. } => ast_id.to_ptr(db).text_range(),
            MacroCallKind::Derive { ast_id, .. } => ast_id.to_ptr(db).text_range(),
            MacroCallKind::Attr { ast_id, .. } => ast_id.to_ptr(db).text_range(),
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
            MacroCallKind::FnLike { ast_id, .. } => ast_id.to_ptr(db).text_range(),
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

    fn arg(&self, db: &dyn db::ExpandDatabase) -> Option<InFile<SyntaxNode>> {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => ast_id
                .to_in_file_node(db)
                .map(|it| Some(it.token_tree()?.syntax().clone()))
                .transpose(),
            MacroCallKind::Derive { ast_id, .. } => {
                Some(ast_id.to_in_file_node(db).syntax().cloned())
            }
            MacroCallKind::Attr { ast_id, .. } => {
                Some(ast_id.to_in_file_node(db).syntax().cloned())
            }
        }
    }
}

/// ExpansionInfo mainly describes how to map text range between src and expanded macro
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpansionInfo {
    expanded: InMacroFile<SyntaxNode>,
    /// The argument TokenTree or item for attributes
    arg: InFile<SyntaxNode>,
    /// The `macro_rules!` or attribute input.
    attr_input_or_mac_def: Option<InFile<ast::TokenTree>>,

    macro_def: TokenExpander,
    macro_arg: Arc<tt::Subtree>,
    exp_map: Arc<SpanMap>,
}

impl ExpansionInfo {
    pub fn expanded(&self) -> InFile<SyntaxNode> {
        self.expanded.clone().into()
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
        token: InFile<&SyntaxToken>,
        // FIXME: use this for range mapping, so that we can resolve inline format args
        _relative_token_offset: Option<TextSize>,
    ) -> Option<impl Iterator<Item = InFile<SyntaxToken>> + '_> {
        assert_eq!(token.file_id, self.arg.file_id);
        let span_map = &self.exp_map.span_map;
        let (start, end) = if span_map
            .first()
            .map_or(false, |(_, span)| span.anchor.file_id == token.file_id)
        {
            (0, span_map.partition_point(|a| a.1.anchor.file_id == token.file_id))
        } else {
            let start = span_map.partition_point(|a| a.1.anchor.file_id != token.file_id);
            (
                start,
                start + span_map[start..].partition_point(|a| a.1.anchor.file_id == token.file_id),
            )
        };
        let token_text_range = token.value.text_range();
        let ast_id_map = db.ast_id_map(token.file_id);
        let tokens = span_map[start..end]
            .iter()
            .filter_map(move |(range, span)| {
                let offset = ast_id_map.get_raw(span.anchor.ast_id).text_range().start();
                let abs_range = span.range + offset;
                token_text_range.eq(&abs_range).then_some(*range)
            })
            .flat_map(move |range| self.expanded.value.covering_element(range).into_token());

        Some(tokens.map(move |token| InFile::new(self.expanded.file_id.into(), token)))
    }

    /// Map a token up out of the expansion it resides in into the arguments of the macro call of the expansion.
    pub fn map_token_up(
        &self,
        db: &dyn db::ExpandDatabase,
        token: InFile<&SyntaxToken>,
    ) -> Option<InFile<SyntaxToken>> {
        self.exp_map.span_for_range(token.value.text_range()).and_then(|span| {
            let anchor =
                db.ast_id_map(span.anchor.file_id).get_raw(span.anchor.ast_id).text_range().start();
            InFile::new(
                span.anchor.file_id,
                db.parse_or_expand(span.anchor.file_id)
                    .covering_element(span.range + anchor)
                    .into_token(),
            )
            .transpose()
        })
    }

    fn new(db: &dyn db::ExpandDatabase, macro_file: MacroFile) -> Option<ExpansionInfo> {
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);

        let arg_tt = loc.kind.arg(db)?;

        let macro_def = db.macro_expander(loc.def);
        let (parse, exp_map) = db.parse_macro_expansion(macro_file).value;
        let expanded = InMacroFile { file_id: macro_file, value: parse.syntax_node() };

        let macro_arg = db.macro_arg(macro_file.macro_call_id).value.unwrap_or_else(|| {
            Arc::new(tt::Subtree { delimiter: tt::Delimiter::UNSPECIFIED, token_trees: Vec::new() })
        });

        let def = loc.def.ast_id().left().and_then(|id| {
            let def_tt = match id.to_node(db) {
                ast::Macro::MacroRules(mac) => mac.token_tree()?,
                ast::Macro::MacroDef(_) if matches!(macro_def, TokenExpander::BuiltInAttr(_)) => {
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
            expanded,
            arg: arg_tt,
            attr_input_or_mac_def,
            macro_arg,
            macro_def,
            exp_map,
        })
    }
}

/// `AstId` points to an AST node in any file.
///
/// It is stable across reparses, and can be used as salsa key/value.
pub type AstId<N> = InFile<FileAstId<N>>;

impl<N: AstIdNode> AstId<N> {
    pub fn to_node(&self, db: &dyn db::ExpandDatabase) -> N {
        self.to_ptr(db).to_node(&db.parse_or_expand(self.file_id))
    }
    pub fn to_in_file_node(&self, db: &dyn db::ExpandDatabase) -> InFile<N> {
        InFile::new(self.file_id, self.to_ptr(db).to_node(&db.parse_or_expand(self.file_id)))
    }
    pub fn to_ptr(&self, db: &dyn db::ExpandDatabase) -> AstPtr<N> {
        db.ast_id_map(self.file_id).get(self.value)
    }
}

pub type ErasedAstId = InFile<ErasedFileAstId>;

impl ErasedAstId {
    pub fn to_ptr(&self, db: &dyn db::ExpandDatabase) -> SyntaxNodePtr {
        db.ast_id_map(self.file_id).get_raw(self.value)
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
        db.parse_or_expand(self.file_id)
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

impl InFile<&SyntaxNode> {
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
        expansion.map_token_up(db, self.as_ref())
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
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct InMacroFile<T> {
    pub file_id: MacroFile,
    pub value: T,
}

impl<T> From<InMacroFile<T>> for InFile<T> {
    fn from(macro_file: InMacroFile<T>) -> Self {
        InFile { file_id: macro_file.file_id.into(), value: macro_file.value }
    }
}

// FIXME: Get rid of this
fn ascend_node_border_tokens(
    db: &dyn db::ExpandDatabase,
    InFile { file_id, value: node }: InFile<&SyntaxNode>,
) -> Option<InFile<(SyntaxToken, SyntaxToken)>> {
    let expansion = file_id.expansion_info(db)?;

    let first_token = |node: &SyntaxNode| skip_trivia_token(node.first_token()?, Direction::Next);
    let last_token = |node: &SyntaxNode| skip_trivia_token(node.last_token()?, Direction::Prev);

    // FIXME: Once the token map rewrite is done, this shouldnt need to rely on syntax nodes and tokens anymore
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

    loop {
        match mapping.file_id.expansion_info(db) {
            Some(info) => mapping = info.map_token_up(db, mapping.as_ref())?,
            None => return Some(mapping),
        }
    }
}

impl<N: AstNode> InFile<N> {
    pub fn descendants<T: AstNode>(self) -> impl Iterator<Item = InFile<T>> {
        self.value.syntax().descendants().filter_map(T::cast).map(move |n| self.with_value(n))
    }

    // FIXME: this should return `Option<InFileNotHirFile<N>>`
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
