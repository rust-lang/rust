//! `hir_expand` deals with macro expansion.
//!
//! Specifically, it implements a concept of `MacroFile` -- a file whose syntax
//! tree originates not from the text of some `FileId`, but from some macro
//! expansion.
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
// It's useful to refer to code that is private in doc comments.
#![allow(rustdoc::private_intra_doc_links)]

pub use intern;

pub mod attrs;
pub mod builtin;
pub mod change;
pub mod declarative;
pub mod eager;
pub mod files;
pub mod hygiene;
pub mod inert_attr_macro;
pub mod mod_path;
pub mod name;
pub mod proc_macro;
pub mod span_map;

mod cfg_process;
mod fixup;
mod prettify_macro_expansion_;

use salsa::plumbing::{AsId, FromId};
use thin_vec::ThinVec;
use triomphe::Arc;

use core::fmt;
use std::{borrow::Cow, ops};

use base_db::{Crate, SourceDatabase};
use either::Either;
use mbe::MatchedArmIndex;
use span::{
    AstIdMap, Edition, ErasedFileAstId, FileAstId, NO_DOWNMAP_ERASED_FILE_AST_ID_MARKER, Span,
    SyntaxContext,
};
use syntax::{
    Parse, SyntaxError, SyntaxNode, SyntaxToken, T, TextRange, TextSize,
    ast::{self, AstNode},
};
use syntax_bridge::{DocCommentDesugarMode, syntax_node_to_token_tree};

use crate::{
    attrs::AttrId,
    builtin::{
        BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerExpander,
        include_input_to_file_id, pseudo_derive_attr_expansion,
    },
    cfg_process::attr_macro_input_to_token_tree,
    fixup::SyntaxFixupUndoInfo,
    hygiene::{span_with_call_site_ctxt, span_with_def_site_ctxt, span_with_mixed_site_ctxt},
    proc_macro::{CustomProcMacroExpander, ProcMacroKind, ProcMacros},
    span_map::{ExpansionSpanMap, RealSpanMap, SpanMap},
};

pub use crate::{
    files::{AstId, ErasedAstId, FileRange, InFile, InMacroFile, InRealFile},
    prettify_macro_expansion_::prettify_macro_expansion,
};

pub use base_db::EditionedFileId;
pub use mbe::{DeclarativeMacro, MacroCallStyle, MacroCallStyles, ValueResult};

pub use tt;

/// This is just to ensure the types of [`MacroCallId::macro_arg_considering_derives`]
/// and [`MacroCallId::macro_arg`] are the same.
type MacroArgResult = (tt::TopSubtree, SyntaxFixupUndoInfo, Span);

/// Total limit on the number of tokens produced by any macro invocation.
///
/// If an invocation produces more tokens than this limit, it will not be stored in the database and
/// an error will be emitted.
///
/// Actual max for `analysis-stats .` at some point: 30672.
const TOKEN_LIMIT: usize = 2_097_152;

#[macro_export]
macro_rules! impl_intern_lookup {
    ($id:ident, $loc:ident) => {
        impl $crate::Intern for $loc {
            type ID = $id;
            fn intern(self, db: &dyn ::base_db::SourceDatabase) -> Self::ID {
                $id::new(db, self)
            }
        }

        impl $crate::Lookup for $id {
            type Data = $loc;
            fn lookup<'db>(&self, db: &'db dyn ::base_db::SourceDatabase) -> &'db Self::Data {
                self.loc(db)
            }
        }
    };
}

// ideally these would be defined in base-db, but the orphan rule doesn't let us
pub trait Intern {
    type ID;
    fn intern(self, db: &dyn SourceDatabase) -> Self::ID;
}

pub trait Lookup {
    type Data;
    fn lookup<'db>(&self, db: &'db dyn SourceDatabase) -> &'db Self::Data;
}

impl_intern_lookup!(MacroCallId, MacroCallLoc);

pub type ExpandResult<T> = ValueResult<T, ExpandError>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct ExpandError {
    inner: Arc<(ExpandErrorKind, Span)>,
}

impl ExpandError {
    pub fn new(span: Span, kind: ExpandErrorKind) -> Self {
        ExpandError { inner: Arc::new((kind, span)) }
    }
    pub fn other(span: Span, msg: impl Into<Box<str>>) -> Self {
        ExpandError { inner: Arc::new((ExpandErrorKind::Other(msg.into()), span)) }
    }
    pub fn kind(&self) -> &ExpandErrorKind {
        &self.inner.0
    }
    pub fn span(&self) -> Span {
        self.inner.1
    }

    pub fn render_to_string(&self, db: &dyn SourceDatabase) -> RenderedExpandError {
        self.inner.0.render_to_string(db)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ExpandErrorKind {
    /// Attribute macro expansion is disabled.
    ProcMacroAttrExpansionDisabled,
    MissingProcMacroExpander(Crate),
    /// The macro for this call is disabled.
    MacroDisabled,
    /// The macro definition has errors.
    MacroDefinition,
    Mbe(mbe::ExpandErrorKind),
    RecursionOverflow,
    Other(Box<str>),
    ProcMacroPanic(Box<str>),
}

pub struct RenderedExpandError {
    pub message: String,
    pub error: bool,
    pub kind: &'static str,
}

impl fmt::Display for RenderedExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl RenderedExpandError {
    const GENERAL_KIND: &str = "macro-error";
    const DISABLED: &str = "proc-macro-disabled";
    const ATTR_EXP_DISABLED: &str = "attribute-expansion-disabled";
}

impl ExpandErrorKind {
    pub fn render_to_string(&self, db: &dyn SourceDatabase) -> RenderedExpandError {
        match self {
            ExpandErrorKind::ProcMacroAttrExpansionDisabled => RenderedExpandError {
                message: "procedural attribute macro expansion is disabled".to_owned(),
                error: false,
                kind: RenderedExpandError::ATTR_EXP_DISABLED,
            },
            ExpandErrorKind::MacroDisabled => RenderedExpandError {
                message: "proc-macro is explicitly disabled".to_owned(),
                error: false,
                kind: RenderedExpandError::DISABLED,
            },
            &ExpandErrorKind::MissingProcMacroExpander(def_crate) => {
                match ProcMacros::get_for_crate(db, def_crate).and_then(|it| it.get_error()) {
                    Some(e) => RenderedExpandError {
                        message: e.to_string(),
                        error: e.is_hard_error(),
                        kind: RenderedExpandError::GENERAL_KIND,
                    },
                    None => RenderedExpandError {
                        message: format!(
                            "internal error: proc-macro map is missing error entry for crate {def_crate:?}"
                        ),
                        error: true,
                        kind: RenderedExpandError::GENERAL_KIND,
                    },
                }
            }
            ExpandErrorKind::MacroDefinition => RenderedExpandError {
                message: "macro definition has parse errors".to_owned(),
                error: true,
                kind: RenderedExpandError::GENERAL_KIND,
            },
            ExpandErrorKind::Mbe(e) => RenderedExpandError {
                message: e.to_string(),
                error: true,
                kind: RenderedExpandError::GENERAL_KIND,
            },
            ExpandErrorKind::RecursionOverflow => RenderedExpandError {
                message: "overflow expanding the original macro".to_owned(),
                error: true,
                kind: RenderedExpandError::GENERAL_KIND,
            },
            ExpandErrorKind::Other(e) => RenderedExpandError {
                message: (**e).to_owned(),
                error: true,
                kind: RenderedExpandError::GENERAL_KIND,
            },
            ExpandErrorKind::ProcMacroPanic(e) => RenderedExpandError {
                message: format!("proc-macro panicked: {e}"),
                error: true,
                kind: RenderedExpandError::GENERAL_KIND,
            },
        }
    }
}

impl From<mbe::ExpandError> for ExpandError {
    fn from(mbe: mbe::ExpandError) -> Self {
        ExpandError { inner: Arc::new((ExpandErrorKind::Mbe(mbe.inner.1.clone()), mbe.inner.0)) }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroCallLoc {
    pub def: MacroDefId,
    pub krate: Crate,
    pub kind: MacroCallKind,
    pub ctxt: SyntaxContext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroDefId {
    pub krate: Crate,
    pub edition: Edition,
    pub kind: MacroDefKind,
    pub local_inner: bool,
    pub allow_internal_unsafe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroDefKind {
    Declarative(AstId<ast::Macro>, MacroCallStyles),
    BuiltIn(AstId<ast::Macro>, BuiltinFnLikeExpander),
    BuiltInAttr(AstId<ast::Macro>, BuiltinAttrExpander),
    BuiltInDerive(AstId<ast::Macro>, BuiltinDeriveExpander),
    BuiltInEager(AstId<ast::Macro>, EagerExpander),
    UnimplementedBuiltIn(AstId<ast::Macro>),
    ProcMacro(AstId<ast::Fn>, CustomProcMacroExpander, ProcMacroKind),
}

impl MacroDefKind {
    #[inline]
    pub fn is_declarative(&self) -> bool {
        matches!(self, MacroDefKind::Declarative(..))
    }

    pub fn erased_ast_id(&self) -> ErasedAstId {
        match *self {
            MacroDefKind::ProcMacro(id, ..) => id.erase(),
            MacroDefKind::BuiltIn(id, _)
            | MacroDefKind::BuiltInAttr(id, _)
            | MacroDefKind::BuiltInDerive(id, _)
            | MacroDefKind::BuiltInEager(id, _)
            | MacroDefKind::Declarative(id, ..)
            | MacroDefKind::UnimplementedBuiltIn(id) => id.erase(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EagerCallInfo {
    /// The expanded argument of the eager macro.
    arg: tt::TopSubtree,
    /// Call id of the eager macro's input file (this is the macro file for its fully expanded input).
    arg_id: MacroCallId,
    error: Option<ExpandError>,
    /// The call site span of the eager macro
    span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroCallKind {
    FnLike {
        ast_id: AstId<ast::MacroCall>,
        expand_to: ExpandTo,
        /// Some if this is a macro call for an eager macro. Note that this is `None`
        /// for the eager input macro file.
        // FIXME: This is being interned, subtrees can vary quickly differing just slightly causing
        // leakage problems here
        eager: Option<Box<EagerCallInfo>>,
    },
    Derive {
        ast_id: AstId<ast::Adt>,
        /// Syntactical index of the invoking `#[derive]` attribute.
        derive_attr_index: AttrId,
        /// Index of the derive macro in the derive attribute
        derive_index: u32,
        /// The "parent" macro call.
        /// We will resolve the same token tree for all derive macros in the same derive attribute.
        derive_macro_id: MacroCallId,
    },
    Attr {
        ast_id: AstId<ast::Item>,
        // FIXME: This shouldn't be here, we can derive this from `invoc_attr_index`.
        attr_args: Option<Box<tt::TopSubtree>>,
        /// This contains the list of all *active* attributes (derives and attr macros) preceding this
        /// attribute, including this attribute. You can retrieve the [`AttrId`] of the current attribute
        /// by calling [`invoc_attr()`] on this.
        ///
        /// The macro should not see the attributes here.
        ///
        /// [`invoc_attr()`]: AttrMacroAttrIds::invoc_attr
        censored_attr_ids: AttrMacroAttrIds,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttrMacroAttrIds(AttrMacroAttrIdsRepr);

impl AttrMacroAttrIds {
    #[inline]
    pub fn from_one(id: AttrId) -> Self {
        Self(AttrMacroAttrIdsRepr::One(id))
    }

    #[inline]
    pub fn from_many(ids: &[AttrId]) -> Self {
        if let &[id] = ids {
            Self(AttrMacroAttrIdsRepr::One(id))
        } else {
            Self(AttrMacroAttrIdsRepr::ManyDerives(ids.iter().copied().collect()))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AttrMacroAttrIdsRepr {
    One(AttrId),
    ManyDerives(ThinVec<AttrId>),
}

impl ops::Deref for AttrMacroAttrIds {
    type Target = [AttrId];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match &self.0 {
            AttrMacroAttrIdsRepr::One(one) => std::slice::from_ref(one),
            AttrMacroAttrIdsRepr::ManyDerives(many) => many,
        }
    }
}

impl AttrMacroAttrIds {
    #[inline]
    pub fn invoc_attr(&self) -> AttrId {
        match &self.0 {
            AttrMacroAttrIdsRepr::One(it) => *it,
            AttrMacroAttrIdsRepr::ManyDerives(it) => {
                *it.last().expect("should always have at least one `AttrId`")
            }
        }
    }
}

impl MacroCallKind {
    pub(crate) fn call_style(&self) -> MacroCallStyle {
        match self {
            MacroCallKind::FnLike { .. } => MacroCallStyle::FnLike,
            MacroCallKind::Derive { .. } => MacroCallStyle::Derive,
            MacroCallKind::Attr { .. } => MacroCallStyle::Attr,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroKind {
    /// `macro_rules!` or Macros 2.0 macro.
    Declarative,
    /// A built-in function-like macro.
    DeclarativeBuiltIn,
    /// A custom derive.
    Derive,
    /// A builtin-in derive.
    DeriveBuiltIn,
    /// A procedural attribute macro.
    Attr,
    /// A built-in attribute macro.
    AttrBuiltIn,
    /// A function-like procedural macro.
    ProcMacro,
}

impl MacroCallId {
    pub fn call_node(self, db: &dyn SourceDatabase) -> InFile<SyntaxNode> {
        self.loc(db).to_node(db)
    }
    pub fn expansion_level(self, db: &dyn SourceDatabase) -> u32 {
        let mut level = 0;
        let mut macro_file = self;
        loop {
            let loc = macro_file.loc(db);

            level += 1;
            macro_file = match loc.kind.file_id() {
                HirFileId::FileId(_) => break level,
                HirFileId::MacroFile(it) => it,
            };
        }
    }
    pub fn parent(self, db: &dyn SourceDatabase) -> HirFileId {
        self.loc(db).kind.file_id()
    }

    /// Return expansion information if it is a macro-expansion file
    pub fn expansion_info(self, db: &dyn SourceDatabase) -> ExpansionInfo<'_> {
        ExpansionInfo::new(db, self)
    }

    pub fn kind(self, db: &dyn SourceDatabase) -> MacroKind {
        match self.loc(db).def.kind {
            MacroDefKind::Declarative(..) => MacroKind::Declarative,
            MacroDefKind::BuiltIn(..) | MacroDefKind::BuiltInEager(..) => {
                MacroKind::DeclarativeBuiltIn
            }
            MacroDefKind::BuiltInDerive(..) => MacroKind::DeriveBuiltIn,
            MacroDefKind::ProcMacro(_, _, ProcMacroKind::CustomDerive) => MacroKind::Derive,
            MacroDefKind::ProcMacro(_, _, ProcMacroKind::Attr) => MacroKind::Attr,
            MacroDefKind::ProcMacro(_, _, ProcMacroKind::Bang) => MacroKind::ProcMacro,
            MacroDefKind::BuiltInAttr(..) => MacroKind::AttrBuiltIn,
            MacroDefKind::UnimplementedBuiltIn(..) => MacroKind::Declarative,
        }
    }

    pub fn is_include_macro(self, db: &dyn SourceDatabase) -> bool {
        self.loc(db).def.is_include()
    }

    pub fn is_include_like_macro(self, db: &dyn SourceDatabase) -> bool {
        self.loc(db).def.is_include_like()
    }

    pub fn is_env_or_option_env(self, db: &dyn SourceDatabase) -> bool {
        self.loc(db).def.is_env_or_option_env()
    }

    pub fn is_eager(self, db: &dyn SourceDatabase) -> bool {
        let loc = self.loc(db);
        matches!(loc.def.kind, MacroDefKind::BuiltInEager(..))
    }

    pub fn eager_arg(self, db: &dyn SourceDatabase) -> Option<MacroCallId> {
        let loc = self.loc(db);
        match &loc.kind {
            MacroCallKind::FnLike { eager, .. } => eager.as_ref().map(|it| it.arg_id),
            _ => None,
        }
    }

    pub fn is_derive_attr_pseudo_expansion(self, db: &dyn SourceDatabase) -> bool {
        let loc = self.loc(db);
        loc.def.is_attribute_derive()
    }
}

#[salsa::tracked]
impl MacroCallId {
    /// Implementation of [`HirFileId::parse_or_expand`] for the macro case.
    // FIXME: We should verify that the parsed node is one of the many macro node variants we expect
    // instead of having it be untyped
    #[salsa::tracked(returns(ref), lru = 512)]
    pub fn parse_macro_expansion(
        self,
        db: &dyn SourceDatabase,
    ) -> ExpandResult<(Parse<SyntaxNode>, ExpansionSpanMap)> {
        let _p = tracing::info_span!("parse_macro_expansion").entered();
        let loc = self.loc(db);
        let expand_to = loc.expand_to();
        let mbe::ValueResult { value: (tt, matched_arm), err } = self.macro_expand(db, loc);

        let (parse, mut rev_token_map) = token_tree_to_syntax_node(db, &tt, expand_to);
        rev_token_map.matched_arm = matched_arm;

        ExpandResult { value: (parse, rev_token_map), err }
    }

    pub fn parse_macro_expansion_error(
        self,
        db: &dyn SourceDatabase,
    ) -> Option<ExpandResult<Arc<[SyntaxError]>>> {
        let e: ExpandResult<Arc<[SyntaxError]>> =
            self.parse_macro_expansion(db).as_ref().map(|it| Arc::from(it.0.errors()));
        if e.value.is_empty() && e.err.is_none() { None } else { Some(e) }
    }

    /// This resolves the [MacroCallId] to check if it is a derive macro if so get the [macro_arg] for the derive.
    /// Other wise return the [macro_arg] for the macro_call_id.
    ///
    /// This is not connected to the database so it does not cache the result. However, the inner [macro_arg] query is
    ///
    /// [macro_arg]: Self::macro_arg
    #[allow(deprecated)] // we are macro_arg_considering_derives
    pub fn macro_arg_considering_derives<'db>(
        self,
        db: &'db dyn SourceDatabase,
        kind: &MacroCallKind,
    ) -> &'db MacroArgResult {
        match kind {
            // Get the macro arg for the derive macro
            MacroCallKind::Derive { derive_macro_id, .. } => derive_macro_id.macro_arg(db),
            // Normal macro arg
            _ => self.macro_arg(db),
        }
    }

    /// Lowers syntactic macro call to a token tree representation. That's a firewall
    /// query, only typing in the macro call itself changes the returned
    /// subtree.
    #[salsa::tracked(returns(ref))]
    fn macro_arg(self, db: &dyn SourceDatabase) -> MacroArgResult {
        let loc = self.loc(db);

        if let MacroCallLoc {
            def: MacroDefId { kind: MacroDefKind::BuiltInEager(..), .. },
            kind: MacroCallKind::FnLike { eager: Some(eager), .. },
            ..
        } = &loc
        {
            return (eager.arg.clone(), SyntaxFixupUndoInfo::NONE, eager.span);
        }

        let (parse, map) = loc.kind.file_id().parse_with_map(db);
        let root = parse.syntax_node();

        let (is_derive, censor_item_tree_attr_ids, item_node, span) = match &loc.kind {
            MacroCallKind::FnLike { ast_id, .. } => {
                let node = &ast_id.to_ptr(db).to_node(&root);
                let path_range = node
                    .path()
                    .map_or_else(|| node.syntax().text_range(), |path| path.syntax().text_range());
                let span = map.span_for_range(path_range);

                let dummy_tt = |kind| {
                    (
                        tt::TopSubtree::from_token_trees(
                            tt::Delimiter { open: span, close: span, kind },
                            tt::TokenTreesView::empty(),
                        ),
                        SyntaxFixupUndoInfo::default(),
                        span,
                    )
                };

                let Some(tt) = node.token_tree() else {
                    return dummy_tt(tt::DelimiterKind::Invisible);
                };
                let first = tt.left_delimiter_token().map(|it| it.kind()).unwrap_or(T!['(']);
                let last = tt.right_delimiter_token().map(|it| it.kind()).unwrap_or(T![.]);

                let mismatched_delimiters = !matches!(
                    (first, last),
                    (T!['('], T![')']) | (T!['['], T![']']) | (T!['{'], T!['}'])
                );
                if mismatched_delimiters {
                    // Don't expand malformed (unbalanced) macro invocations. This is
                    // less than ideal, but trying to expand unbalanced  macro calls
                    // sometimes produces pathological, deeply nested code which breaks
                    // all kinds of things.
                    //
                    // So instead, we'll return an empty subtree here
                    cov_mark::hit!(issue9358_bad_macro_stack_overflow);

                    let kind = match first {
                        _ if loc.def.is_proc_macro() => tt::DelimiterKind::Invisible,
                        T!['('] => tt::DelimiterKind::Parenthesis,
                        T!['['] => tt::DelimiterKind::Bracket,
                        T!['{'] => tt::DelimiterKind::Brace,
                        _ => tt::DelimiterKind::Invisible,
                    };
                    return dummy_tt(kind);
                }

                let mut tt = syntax_bridge::syntax_node_to_token_tree(
                    tt.syntax(),
                    map,
                    span,
                    if loc.def.is_proc_macro() {
                        DocCommentDesugarMode::ProcMacro
                    } else {
                        DocCommentDesugarMode::Mbe
                    },
                );
                if loc.def.is_proc_macro() {
                    // proc macros expect their inputs without parentheses, MBEs expect it with them included
                    tt.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
                }
                return (tt, SyntaxFixupUndoInfo::NONE, span);
            }
            // MacroCallKind::Derive should not be here. As we are getting the argument for the derive macro
            MacroCallKind::Derive { .. } => {
                unreachable!("`MacroCallId::macro_arg` called with `MacroCallKind::Derive`")
            }
            MacroCallKind::Attr { ast_id, censored_attr_ids: attr_ids, .. } => {
                let node = ast_id.to_ptr(db).to_node(&root);
                let (_, attr) =
                    attr_ids.invoc_attr().find_attr_range_with_source(db, loc.krate, &node);
                let range = attr
                    .path()
                    .map(|path| path.syntax().text_range())
                    .unwrap_or_else(|| attr.syntax().text_range());
                let span = map.span_for_range(range);

                let is_derive = matches!(loc.def.kind, MacroDefKind::BuiltInAttr(_, expander) if expander.is_derive());
                (is_derive, &**attr_ids, node, span)
            }
        };

        let (mut tt, undo_info) = attr_macro_input_to_token_tree(
            db,
            item_node.syntax(),
            map,
            span,
            is_derive,
            censor_item_tree_attr_ids,
            loc.krate,
        );

        if loc.def.is_proc_macro() {
            // proc macros expect their inputs without parentheses, MBEs expect it with them included
            tt.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
        }

        (tt, undo_info, span)
    }

    fn macro_expand<'db>(
        self,
        db: &'db dyn SourceDatabase,
        loc: &MacroCallLoc,
    ) -> ExpandResult<(Cow<'db, tt::TopSubtree>, MatchedArmIndex)> {
        let _p = tracing::info_span!("macro_expand").entered();

        let (ExpandResult { value: (tt, matched_arm), err }, span) = match loc.def.kind {
            MacroDefKind::ProcMacro(..) => {
                return self.expand_proc_macro(db).as_ref().map(|it| (Cow::Borrowed(it), None));
            }
            _ => {
                let (macro_arg, undo_info, span) =
                    self.macro_arg_considering_derives(db, &loc.kind);
                let span = *span;

                let arg = macro_arg;
                let res = match loc.def.kind {
                    MacroDefKind::Declarative(id, _) => {
                        id.decl_macro_expander(db, loc.def.krate).expand(db, arg, self, span)
                    }
                    MacroDefKind::BuiltIn(_, it) => {
                        it.expand(db, self, arg, span).map_err(Into::into).zip_val(None)
                    }
                    MacroDefKind::BuiltInDerive(_, it) => {
                        it.expand(db, self, arg, span).map_err(Into::into).zip_val(None)
                    }
                    MacroDefKind::UnimplementedBuiltIn(_) => {
                        expand_unimplemented_builtin_macro(span).zip_val(None)
                    }
                    MacroDefKind::BuiltInEager(_, it) => {
                        // This might look a bit odd, but we do not expand the inputs to eager macros here.
                        // Eager macros inputs are expanded, well, eagerly when we collect the macro calls.
                        // That kind of expansion uses the ast id map of an eager macros input though which goes through
                        // the HirFileId machinery. As eager macro inputs are assigned a macro file id that query
                        // will end up going through here again, whereas we want to just want to inspect the raw input.
                        // As such we just return the input subtree here.
                        let eager = match &loc.kind {
                            MacroCallKind::FnLike { eager: None, .. } => {
                                return ExpandResult::ok(Cow::Borrowed(macro_arg)).zip_val(None);
                            }
                            MacroCallKind::FnLike { eager: Some(eager), .. } => Some(&**eager),
                            _ => None,
                        };

                        let mut res = it.expand(db, self, arg, span).map_err(Into::into);

                        if let Some(EagerCallInfo { error, .. }) = eager {
                            // FIXME: We should report both errors!
                            res.err = error.clone().or(res.err);
                        }
                        res.zip_val(None)
                    }
                    MacroDefKind::BuiltInAttr(_, it) => {
                        let mut res = it.expand(db, self, arg, span);
                        fixup::reverse_fixups(&mut res.value, undo_info);
                        res.zip_val(None)
                    }
                    MacroDefKind::ProcMacro(_, _, _) => unreachable!(),
                };
                (res, span)
            }
        };

        // Skip checking token tree limit for include! macro call
        if !loc.def.is_include() {
            // Set a hard limit for the expanded tt
            if let Err(value) = check_tt_count(&tt) {
                return value
                    .map(|()| Cow::Owned(tt::TopSubtree::empty(tt::DelimSpan::from_single(span))))
                    .zip_val(matched_arm);
            }
        }

        ExpandResult { value: (Cow::Owned(tt), matched_arm), err }
    }

    /// Special case of [`Self::macro_expand`] for procedural macros. We can't LRU
    /// proc macros, since they are not deterministic in general, and
    /// non-determinism breaks salsa in a very, very, very bad way.
    /// @edwin0cheng heroically debugged this once! See #4315 for details
    #[salsa::tracked(returns(ref))]
    fn expand_proc_macro(self, db: &dyn SourceDatabase) -> ExpandResult<tt::TopSubtree> {
        let loc = self.loc(db);
        let (macro_arg, undo_info, span) = self.macro_arg_considering_derives(db, &loc.kind);

        let (ast, expander) = match loc.def.kind {
            MacroDefKind::ProcMacro(ast, expander, _) => (ast, expander),
            _ => unreachable!(),
        };

        let attr_arg = match &loc.kind {
            MacroCallKind::Attr { attr_args: Some(attr_args), .. } => Some(&**attr_args),
            _ => None,
        };

        let ExpandResult { value: mut tt, err } = {
            let span = proc_macro_span(db, ast);
            expander.expand(
                db,
                loc.def.krate,
                loc.krate,
                macro_arg,
                attr_arg,
                span_with_def_site_ctxt(db, span, self.into(), loc.def.edition),
                span_with_call_site_ctxt(db, span, self.into(), loc.def.edition),
                span_with_mixed_site_ctxt(db, span, self.into(), loc.def.edition),
            )
        };

        // Set a hard limit for the expanded tt
        if let Err(value) = check_tt_count(&tt) {
            return value.map(|()| tt::TopSubtree::empty(tt::DelimSpan::from_single(*span)));
        }

        fixup::reverse_fixups(&mut tt, undo_info);

        ExpandResult { value: tt, err }
    }
}

impl MacroCallId {
    /// This expands the given macro call, but with different arguments. This is
    /// used for completion, where we want to see what 'would happen' if we insert a
    /// token. The `token_to_map` mapped down into the expansion, with the mapped
    /// token(s) returned with their priority.
    pub fn expand_speculative(
        self,
        db: &dyn SourceDatabase,
        speculative_args: &SyntaxNode,
        token_to_map: SyntaxToken,
    ) -> Option<(SyntaxNode, Vec<(SyntaxToken, u8)>)> {
        let loc = self.loc(db);
        let (_, _, span) = *self.macro_arg_considering_derives(db, &loc.kind);

        let span_map = RealSpanMap::absolute(span.anchor.file_id);
        let span_map = SpanMap::RealSpanMap(&span_map);

        // Build the subtree and token mapping for the speculative args
        let (mut tt, undo_info) = match &loc.kind {
            MacroCallKind::FnLike { .. } => (
                syntax_bridge::syntax_node_to_token_tree(
                    speculative_args,
                    span_map,
                    span,
                    if loc.def.is_proc_macro() {
                        DocCommentDesugarMode::ProcMacro
                    } else {
                        DocCommentDesugarMode::Mbe
                    },
                ),
                SyntaxFixupUndoInfo::NONE,
            ),
            MacroCallKind::Attr { .. } if loc.def.is_attribute_derive() => (
                syntax_bridge::syntax_node_to_token_tree(
                    speculative_args,
                    span_map,
                    span,
                    DocCommentDesugarMode::ProcMacro,
                ),
                SyntaxFixupUndoInfo::NONE,
            ),
            MacroCallKind::Derive { derive_macro_id, .. } => {
                let MacroCallKind::Attr { censored_attr_ids: attr_ids, .. } =
                    &derive_macro_id.loc(db).kind
                else {
                    unreachable!("`derive_macro_id` should be `MacroCallKind::Attr`");
                };
                attr_macro_input_to_token_tree(
                    db,
                    speculative_args,
                    span_map,
                    span,
                    true,
                    attr_ids,
                    loc.krate,
                )
            }
            MacroCallKind::Attr { censored_attr_ids: attr_ids, .. } => {
                attr_macro_input_to_token_tree(
                    db,
                    speculative_args,
                    span_map,
                    span,
                    false,
                    attr_ids,
                    loc.krate,
                )
            }
        };

        let attr_arg = match &loc.kind {
            MacroCallKind::Attr { censored_attr_ids: attr_ids, .. } => {
                if loc.def.is_attribute_derive() {
                    // for pseudo-derive expansion we actually pass the attribute itself only
                    ast::Attr::cast(speculative_args.clone())
                        .and_then(|attr| {
                            if let ast::Meta::TokenTreeMeta(meta) = attr.meta()? {
                                meta.token_tree()
                            } else {
                                None
                            }
                        })
                        .map(|token_tree| {
                            let mut tree = syntax_node_to_token_tree(
                                token_tree.syntax(),
                                span_map,
                                span,
                                DocCommentDesugarMode::ProcMacro,
                            );
                            tree.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
                            tree.set_top_subtree_delimiter_span(tt::DelimSpan::from_single(span));
                            tree
                        })
                } else {
                    // Attributes may have an input token tree, build the subtree and map for this as well
                    // then try finding a token id for our token if it is inside this input subtree.
                    let item = ast::Item::cast(speculative_args.clone())?;
                    let (_, meta) = attr_ids
                        .invoc_attr()
                        .find_attr_range_with_source_opt(db, loc.krate, &item)?;
                    if let ast::Meta::TokenTreeMeta(meta) = meta
                        && let Some(tt) = meta.token_tree()
                    {
                        let mut attr_arg = syntax_bridge::syntax_node_to_token_tree(
                            tt.syntax(),
                            span_map,
                            span,
                            DocCommentDesugarMode::ProcMacro,
                        );
                        attr_arg.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
                        Some(attr_arg)
                    } else {
                        None
                    }
                }
            }
            _ => None,
        };

        // Do the actual expansion, we need to directly expand the proc macro due to the attribute args
        // Otherwise the expand query will fetch the non speculative attribute args and pass those instead.
        let mut speculative_expansion = match loc.def.kind {
            MacroDefKind::ProcMacro(ast, expander, _) => {
                let span = proc_macro_span(db, ast);
                tt.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
                tt.set_top_subtree_delimiter_span(tt::DelimSpan::from_single(span));
                expander.expand(
                    db,
                    loc.def.krate,
                    loc.krate,
                    &tt,
                    attr_arg.as_ref(),
                    span_with_def_site_ctxt(db, span, self.into(), loc.def.edition),
                    span_with_call_site_ctxt(db, span, self.into(), loc.def.edition),
                    span_with_mixed_site_ctxt(db, span, self.into(), loc.def.edition),
                )
            }
            MacroDefKind::BuiltInAttr(_, it) if it.is_derive() => {
                pseudo_derive_attr_expansion(&tt, attr_arg.as_ref()?, span)
            }
            MacroDefKind::Declarative(it, _) => it
                .decl_macro_expander(db, loc.krate)
                .expand_unhygienic(db, &tt, loc.kind.call_style(), span),
            MacroDefKind::BuiltIn(_, it) => it.expand(db, self, &tt, span).map_err(Into::into),
            MacroDefKind::BuiltInDerive(_, it) => {
                it.expand(db, self, &tt, span).map_err(Into::into)
            }
            MacroDefKind::BuiltInEager(_, it) => it.expand(db, self, &tt, span).map_err(Into::into),
            MacroDefKind::BuiltInAttr(_, it) => it.expand(db, self, &tt, span),
            MacroDefKind::UnimplementedBuiltIn(_) => expand_unimplemented_builtin_macro(span),
        };

        let expand_to = loc.expand_to();

        fixup::reverse_fixups(&mut speculative_expansion.value, &undo_info);
        let (node, rev_tmap) =
            token_tree_to_syntax_node(db, &speculative_expansion.value, expand_to);

        let syntax_node = node.syntax_node();
        let token = rev_tmap
            .ranges_with_span(span_map.span_for_range(token_to_map.text_range()))
            .filter_map(|(range, ctx)| {
                syntax_node.covering_element(range).into_token().zip(Some(ctx))
            })
            .map(|(t, ctx)| {
                // prefer tokens of the same kind and text, as well as non opaque marked ones
                // Note the inversion of the score here, as we want to prefer the first token in case
                // of all tokens having the same score
                let ranking = ctx.is_opaque(db) as u8
                    + 2 * (t.kind() != token_to_map.kind()) as u8
                    + 4 * ((t.text() != token_to_map.text()) as u8);
                (t, ranking)
            })
            .collect();
        Some((node.syntax_node(), token))
    }
}

fn expand_unimplemented_builtin_macro(span: Span) -> ExpandResult<tt::TopSubtree> {
    ExpandResult::new(
        tt::TopSubtree::empty(tt::DelimSpan::from_single(span)),
        ExpandError::other(span, "this built-in macro is not implemented"),
    )
}

/// Retrieves the span to be used for a proc-macro expansions spans.
/// This is a firewall query as it requires parsing the file, which we don't want proc-macros to
/// directly depend on as that would cause to frequent invalidations, mainly because of the
/// parse queries being LRU cached. If they weren't the invalidations would only happen if the
/// user wrote in the file that defines the proc-macro.
fn proc_macro_span(db: &dyn SourceDatabase, ast: AstId<ast::Fn>) -> Span {
    #[salsa::tracked]
    fn proc_macro_span(db: &dyn SourceDatabase, ast: AstId<ast::Fn>, _: ()) -> Span {
        let (parse, span_map) = ast.file_id.parse_with_map(db);
        let root = parse.syntax_node();
        let ast_id_map = ast.file_id.ast_id_map(db);

        let node = ast_id_map.get(ast.value).to_node(&root);
        let range = ast::HasName::name(&node)
            .map_or_else(|| node.syntax().text_range(), |name| name.syntax().text_range());
        span_map.span_for_range(range)
    }
    proc_macro_span(db, ast, ())
}

pub(crate) fn token_tree_to_syntax_node(
    db: &dyn SourceDatabase,
    tt: &tt::TopSubtree,
    expand_to: ExpandTo,
) -> (Parse<SyntaxNode>, ExpansionSpanMap) {
    let entry_point = match expand_to {
        ExpandTo::Statements => syntax_bridge::TopEntryPoint::MacroStmts,
        ExpandTo::Items => syntax_bridge::TopEntryPoint::MacroItems,
        ExpandTo::Pattern => syntax_bridge::TopEntryPoint::Pattern,
        ExpandTo::Type => syntax_bridge::TopEntryPoint::Type,
        ExpandTo::Expr => syntax_bridge::TopEntryPoint::Expr,
    };
    syntax_bridge::token_tree_to_syntax_node(tt, entry_point, &mut |ctx| ctx.edition(db))
}

fn check_tt_count(tt: &tt::TopSubtree) -> Result<(), ExpandResult<()>> {
    let tt = tt.top_subtree();
    let count = tt.count();
    if count <= TOKEN_LIMIT {
        Ok(())
    } else {
        Err(ExpandResult {
            value: (),
            err: Some(ExpandError::other(
                tt.delimiter.open,
                format!(
                    "macro invocation exceeds token limit: produced {count} tokens, limit is {TOKEN_LIMIT}",
                ),
            )),
        })
    }
}

impl MacroDefId {
    pub fn make_call(
        self,
        db: &dyn SourceDatabase,
        krate: Crate,
        kind: MacroCallKind,
        ctxt: SyntaxContext,
    ) -> MacroCallId {
        MacroCallId::new(db, MacroCallLoc { def: self, krate, kind, ctxt })
    }

    pub fn definition_range(&self, db: &dyn SourceDatabase) -> InFile<TextRange> {
        match self.kind {
            MacroDefKind::Declarative(id, _)
            | MacroDefKind::BuiltIn(id, _)
            | MacroDefKind::BuiltInAttr(id, _)
            | MacroDefKind::BuiltInDerive(id, _)
            | MacroDefKind::BuiltInEager(id, _)
            | MacroDefKind::UnimplementedBuiltIn(id) => {
                id.with_value(id.file_id.ast_id_map(db).get(id.value).text_range())
            }
            MacroDefKind::ProcMacro(id, _, _) => {
                id.with_value(id.file_id.ast_id_map(db).get(id.value).text_range())
            }
        }
    }

    pub fn ast_id(&self) -> Either<AstId<ast::Macro>, AstId<ast::Fn>> {
        match self.kind {
            MacroDefKind::ProcMacro(id, ..) => Either::Right(id),
            MacroDefKind::Declarative(id, _)
            | MacroDefKind::BuiltIn(id, _)
            | MacroDefKind::BuiltInAttr(id, _)
            | MacroDefKind::BuiltInDerive(id, _)
            | MacroDefKind::BuiltInEager(id, _)
            | MacroDefKind::UnimplementedBuiltIn(id) => Either::Left(id),
        }
    }

    pub fn is_proc_macro(&self) -> bool {
        matches!(self.kind, MacroDefKind::ProcMacro(..))
    }

    pub fn is_attribute(&self) -> bool {
        match self.kind {
            MacroDefKind::BuiltInAttr(..)
            | MacroDefKind::ProcMacro(_, _, ProcMacroKind::Attr)
            | MacroDefKind::UnimplementedBuiltIn(_) => true,
            MacroDefKind::Declarative(_, styles) => styles.contains(MacroCallStyles::ATTR),
            _ => false,
        }
    }

    pub fn is_derive(&self) -> bool {
        match self.kind {
            MacroDefKind::BuiltInDerive(..)
            | MacroDefKind::ProcMacro(_, _, ProcMacroKind::CustomDerive)
            | MacroDefKind::UnimplementedBuiltIn(_) => true,
            MacroDefKind::Declarative(_, styles) => styles.contains(MacroCallStyles::DERIVE),
            _ => false,
        }
    }

    pub fn is_fn_like(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltIn(..)
                | MacroDefKind::ProcMacro(_, _, ProcMacroKind::Bang)
                | MacroDefKind::BuiltInEager(..)
                | MacroDefKind::Declarative(..)
                | MacroDefKind::UnimplementedBuiltIn(_)
        )
    }

    pub fn is_attribute_derive(&self) -> bool {
        matches!(self.kind, MacroDefKind::BuiltInAttr(_, expander) if expander.is_derive())
    }

    pub fn is_include(&self) -> bool {
        matches!(self.kind, MacroDefKind::BuiltInEager(_, expander) if expander.is_include())
    }

    pub fn is_include_like(&self) -> bool {
        matches!(self.kind, MacroDefKind::BuiltInEager(_, expander) if expander.is_include_like())
    }

    pub fn is_env_or_option_env(&self) -> bool {
        matches!(self.kind, MacroDefKind::BuiltInEager(_, expander) if expander.is_env_or_option_env())
    }
}

impl MacroCallLoc {
    pub fn to_node(&self, db: &dyn SourceDatabase) -> InFile<SyntaxNode> {
        match &self.kind {
            MacroCallKind::FnLike { ast_id, .. } => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
            MacroCallKind::Derive { ast_id, derive_attr_index, .. } => {
                let (_, attr) = derive_attr_index.find_attr_range(db, self.krate, *ast_id);
                ast_id.with_value(attr.syntax().clone())
            }
            MacroCallKind::Attr { ast_id, censored_attr_ids: attr_ids, .. } => {
                if self.def.is_attribute_derive() {
                    let (_, attr) = attr_ids.invoc_attr().find_attr_range(db, self.krate, *ast_id);
                    ast_id.with_value(attr.syntax().clone())
                } else {
                    ast_id.with_value(ast_id.to_node(db).syntax().clone())
                }
            }
        }
    }

    pub fn to_node_item(&self, db: &dyn SourceDatabase) -> InFile<ast::Item> {
        match self.kind {
            MacroCallKind::FnLike { ast_id, .. } => {
                InFile::new(ast_id.file_id, ast_id.map(FileAstId::upcast).to_node(db))
            }
            MacroCallKind::Derive { ast_id, .. } => {
                InFile::new(ast_id.file_id, ast_id.map(FileAstId::upcast).to_node(db))
            }
            MacroCallKind::Attr { ast_id, .. } => InFile::new(ast_id.file_id, ast_id.to_node(db)),
        }
    }

    fn expand_to(&self) -> ExpandTo {
        match self.kind {
            MacroCallKind::FnLike { expand_to, .. } => expand_to,
            MacroCallKind::Derive { .. } => ExpandTo::Items,
            MacroCallKind::Attr { .. } if self.def.is_attribute_derive() => ExpandTo::Items,
            MacroCallKind::Attr { .. } => {
                // FIXME(stmt_expr_attributes)
                ExpandTo::Items
            }
        }
    }

    pub fn include_file_id(
        &self,
        db: &dyn SourceDatabase,
        macro_call_id: MacroCallId,
    ) -> Option<EditionedFileId> {
        if self.def.is_include()
            && let MacroCallKind::FnLike { eager: Some(eager), .. } = &self.kind
            && let Ok(it) = include_input_to_file_id(db, macro_call_id, &eager.arg)
        {
            return Some(it);
        }

        None
    }
}

impl MacroCallKind {
    pub fn descr(&self) -> &'static str {
        match self {
            MacroCallKind::FnLike { .. } => "macro call",
            MacroCallKind::Derive { .. } => "derive macro",
            MacroCallKind::Attr { .. } => "attribute macro",
        }
    }

    /// Returns the file containing the macro invocation.
    pub fn file_id(&self) -> HirFileId {
        match *self {
            MacroCallKind::FnLike { ast_id: InFile { file_id, .. }, .. }
            | MacroCallKind::Derive { ast_id: InFile { file_id, .. }, .. }
            | MacroCallKind::Attr { ast_id: InFile { file_id, .. }, .. } => file_id,
        }
    }

    pub fn erased_ast_id(&self) -> ErasedFileAstId {
        match *self {
            MacroCallKind::FnLike { ast_id: InFile { value, .. }, .. } => value.erase(),
            MacroCallKind::Derive { ast_id: InFile { value, .. }, .. } => value.erase(),
            MacroCallKind::Attr { ast_id: InFile { value, .. }, .. } => value.erase(),
        }
    }

    /// Returns the original file range that best describes the location of this macro call.
    ///
    /// This spans the entire macro call, including its input. That is for
    /// - fn_like! {}, it spans the path and token tree
    /// - #\[derive], it spans the `#[derive(...)]` attribute and the annotated item
    /// - #\[attr], it spans the `#[attr(...)]` attribute and the annotated item
    pub fn original_call_range_with_input(&self, db: &dyn SourceDatabase) -> FileRange {
        let get_range = |kind: &_| match kind {
            MacroCallKind::FnLike { ast_id, .. } => ast_id.erase(),
            MacroCallKind::Derive { ast_id, .. } => ast_id.erase(),
            MacroCallKind::Attr { ast_id, .. } => ast_id.erase(),
        };

        let mut ast_id = get_range(self);
        let mut file_id = self.file_id();
        let file_id = loop {
            match file_id {
                HirFileId::MacroFile(file) => {
                    let kind = &file.loc(db).kind;
                    ast_id = get_range(kind);
                    file_id = kind.file_id();
                }
                HirFileId::FileId(file_id) => break file_id,
            }
        };

        FileRange { range: ast_id.to_ptr(db).text_range(), file_id }
    }

    /// Returns the original file range that best describes the location of this macro call.
    ///
    /// Here we try to roughly match what rustc does to improve diagnostics: fn-like macros
    /// get the macro path (rustc shows the whole `ast::MacroCall`), attribute macros get the
    /// attribute's range, and derives get only the specific derive that is being referred to.
    pub fn original_call_range(&self, db: &dyn SourceDatabase, krate: Crate) -> FileRange {
        let get_range = |kind: &_| match kind {
            MacroCallKind::FnLike { ast_id, .. } => {
                let node = ast_id.to_node(db);
                node.path()
                    .unwrap()
                    .syntax()
                    .text_range()
                    .cover(node.excl_token().unwrap().text_range())
            }
            MacroCallKind::Derive { ast_id, derive_attr_index, .. } => {
                // FIXME: should be the range of the macro name, not the whole derive
                derive_attr_index.find_attr_range(db, krate, *ast_id).1.syntax().text_range()
            }
            // FIXME: handle `cfg_attr`
            MacroCallKind::Attr { ast_id, censored_attr_ids: attr_ids, .. } => {
                attr_ids.invoc_attr().find_attr_range(db, krate, *ast_id).1.syntax().text_range()
            }
        };

        let mut range = get_range(self);
        let mut file_id = self.file_id();
        let file_id = loop {
            match file_id {
                HirFileId::MacroFile(file) => {
                    let kind = &file.loc(db).kind;
                    range = get_range(kind);
                    file_id = kind.file_id();
                }
                HirFileId::FileId(file_id) => break file_id,
            }
        };

        FileRange { range, file_id }
    }

    fn arg(&self, db: &dyn SourceDatabase) -> InFile<Option<SyntaxNode>> {
        match self {
            MacroCallKind::FnLike { ast_id, .. } => {
                ast_id.to_in_file_node(db).map(|it| Some(it.token_tree()?.syntax().clone()))
            }
            MacroCallKind::Derive { ast_id, .. } => {
                ast_id.to_in_file_node(db).syntax().cloned().map(Some)
            }
            MacroCallKind::Attr { ast_id, .. } => {
                ast_id.to_in_file_node(db).syntax().cloned().map(Some)
            }
        }
    }
}

/// ExpansionInfo mainly describes how to map text range between src and expanded macro
// FIXME: can be expensive to create, we should check the use sites and maybe replace them with
// simpler function calls if the map is only used once
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpansionInfo<'db> {
    expanded: InMacroFile<SyntaxNode>,
    /// The argument TokenTree or item for attributes
    arg: InFile<Option<SyntaxNode>>,
    exp_map: &'db ExpansionSpanMap,
    arg_map: SpanMap<'db>,
    loc: &'db MacroCallLoc,
}

impl<'db> ExpansionInfo<'db> {
    pub fn expanded(&self) -> InMacroFile<SyntaxNode> {
        self.expanded.clone()
    }

    pub fn arg(&self) -> InFile<Option<&SyntaxNode>> {
        self.arg.as_ref().map(|it| it.as_ref())
    }

    pub fn call_file(&self) -> HirFileId {
        self.arg.file_id
    }

    pub fn is_attr(&self) -> bool {
        matches!(
            self.loc.def.kind,
            MacroDefKind::BuiltInAttr(..) | MacroDefKind::ProcMacro(_, _, ProcMacroKind::Attr)
        )
    }

    /// Maps the passed in file range down into a macro expansion if it is the input to a macro call.
    ///
    /// Note this does a linear search through the entire backing vector of the spanmap.
    // FIXME: Consider adding a reverse map to ExpansionInfo to get rid of the linear search which
    // potentially results in quadratic look ups (notably this might improve semantic highlighting perf)
    pub fn map_range_down_exact(
        &self,
        span: Span,
    ) -> Option<InMacroFile<impl Iterator<Item = (SyntaxToken, SyntaxContext)> + '_>> {
        if span.anchor.ast_id == NO_DOWNMAP_ERASED_FILE_AST_ID_MARKER {
            return None;
        }

        let tokens = self.exp_map.ranges_with_span_exact(span).flat_map(move |(range, ctx)| {
            self.expanded.value.covering_element(range).into_token().zip(Some(ctx))
        });

        Some(InMacroFile::new(self.expanded.file_id, tokens))
    }

    /// Maps the passed in file range down into a macro expansion if it is the input to a macro call.
    /// Unlike [`ExpansionInfo::map_range_down_exact`], this will consider spans that contain the given span.
    ///
    /// Note this does a linear search through the entire backing vector of the spanmap.
    pub fn map_range_down(
        &self,
        span: Span,
    ) -> Option<InMacroFile<impl Iterator<Item = (SyntaxToken, SyntaxContext)> + '_>> {
        if span.anchor.ast_id == NO_DOWNMAP_ERASED_FILE_AST_ID_MARKER {
            return None;
        }

        let tokens = self.exp_map.ranges_with_span(span).flat_map(move |(range, ctx)| {
            self.expanded.value.covering_element(range).into_token().zip(Some(ctx))
        });

        Some(InMacroFile::new(self.expanded.file_id, tokens))
    }

    /// Looks up the span at the given offset.
    pub fn span_for_offset(
        &self,
        db: &dyn SourceDatabase,
        offset: TextSize,
    ) -> (FileRange, SyntaxContext) {
        debug_assert!(self.expanded.value.text_range().contains(offset));
        span_for_offset(db, self.exp_map, offset)
    }

    /// Maps up the text range out of the expansion hierarchy back into the original file its from.
    pub fn map_node_range_up(
        &self,
        db: &dyn SourceDatabase,
        range: TextRange,
    ) -> Option<(FileRange, SyntaxContext)> {
        debug_assert!(self.expanded.value.text_range().contains_range(range));
        map_node_range_up(db, self.exp_map, range)
    }

    /// Maps up the text range out of the expansion into its macro call.
    ///
    /// Note that this may return multiple ranges as we lose the precise association between input to output
    /// and as such we may consider inputs that are unrelated.
    pub fn map_range_up_once(
        &self,
        db: &dyn SourceDatabase,
        token: TextRange,
    ) -> InFile<smallvec::SmallVec<[TextRange; 1]>> {
        debug_assert!(self.expanded.value.text_range().contains_range(token));
        let span = self.exp_map.span_at(token.start());
        match &self.arg_map {
            SpanMap::RealSpanMap(_) => {
                let range = resolve_span(db, span);
                InFile { file_id: range.file_id.into(), value: smallvec::smallvec![range.range] }
            }
            SpanMap::ExpansionSpanMap(arg_map) => {
                let Some(arg_node) = &self.arg.value else {
                    return InFile::new(self.arg.file_id, smallvec::smallvec![]);
                };
                let arg_range = arg_node.text_range();
                InFile::new(
                    self.arg.file_id,
                    arg_map
                        .ranges_with_span_exact(span)
                        .map(|(range, _)| range)
                        .filter(|range| range.intersect(arg_range).is_some())
                        .collect(),
                )
            }
        }
    }

    pub fn new(db: &'db dyn SourceDatabase, macro_file: MacroCallId) -> ExpansionInfo<'db> {
        let _p = tracing::info_span!("ExpansionInfo::new").entered();
        let loc = macro_file.loc(db);

        let arg_tt = loc.kind.arg(db);
        let arg_map = arg_tt.file_id.span_map(db);

        let (parse, exp_map) = &macro_file.parse_macro_expansion(db).value;
        let expanded = InMacroFile { file_id: macro_file, value: parse.syntax_node() };

        ExpansionInfo { expanded, loc, arg: arg_tt, exp_map, arg_map }
    }
}

/// Maps up the text range out of the expansion hierarchy back into the original file its from only
/// considering the root spans contained.
/// Unlike [`map_node_range_up`], this will not return `None` if any anchors or syntax contexts differ.
pub fn map_node_range_up_rooted(
    db: &dyn SourceDatabase,
    exp_map: &ExpansionSpanMap,
    range: TextRange,
) -> Option<FileRange> {
    let mut spans = exp_map.spans_for_range(range).filter(|span| span.ctx.is_root());
    let Span { range, anchor, ctx } = spans.next()?;
    let mut start = range.start();
    let mut end = range.end();

    for span in spans {
        if span.anchor != anchor {
            return None;
        }
        start = start.min(span.range.start());
        end = end.max(span.range.end());
    }
    Some(resolve_span(db, Span { range: TextRange::new(start, end), anchor, ctx }))
}

/// Maps up the text range out of the expansion hierarchy back into the original file its from.
///
/// this will return `None` if any anchors or syntax contexts differ.
pub fn map_node_range_up(
    db: &dyn SourceDatabase,
    exp_map: &ExpansionSpanMap,
    range: TextRange,
) -> Option<(FileRange, SyntaxContext)> {
    let mut spans = exp_map.spans_for_range(range);
    let Span { range, anchor, ctx } = spans.next()?;
    let mut start = range.start();
    let mut end = range.end();

    for span in spans {
        if span.anchor != anchor || span.ctx != ctx {
            return None;
        }
        start = start.min(span.range.start());
        end = end.max(span.range.end());
    }
    Some((resolve_span(db, Span { range: TextRange::new(start, end), anchor, ctx }), ctx))
}

/// Looks up the span at the given offset.
pub fn span_for_offset(
    db: &dyn SourceDatabase,
    exp_map: &ExpansionSpanMap,
    offset: TextSize,
) -> (FileRange, SyntaxContext) {
    let span = exp_map.span_at(offset);
    (resolve_span(db, span), span.ctx)
}

// FIXME: This is only public because of its use in `load_cargo` (which we should consider removing
// by moving the implementations of the subrequests to `hir_expand`, and calling within `load-cargo`).
// Avoid adding any more outside uses.
pub fn resolve_span(db: &dyn SourceDatabase, Span { range, anchor, ctx: _ }: Span) -> FileRange {
    let file_id = EditionedFileId::from_span_file_id(db, anchor.file_id);
    let anchor_offset =
        HirFileId::from(file_id).ast_id_map(db).get_erased(anchor.ast_id).text_range().start();
    FileRange { file_id, range: range + anchor_offset }
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
                .is_some_and(|p| matches!(p.kind(), EXPR_STMT | STMT_LIST | MACRO_STMTS))
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

/// Macro ids. That's probably the tricksiest bit in rust-analyzer, and the
/// reason why we use salsa at all.
///
/// We encode macro definitions into ids of macro calls, this what allows us
/// to be incremental.
#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[doc(alias = "MacroFileId")]
pub struct MacroCallId {
    #[returns(ref)]
    pub loc: MacroCallLoc,
}

impl From<span::MacroCallId> for MacroCallId {
    #[inline]
    fn from(value: span::MacroCallId) -> Self {
        MacroCallId::from_id(value.0)
    }
}

impl From<MacroCallId> for span::MacroCallId {
    #[inline]
    fn from(value: MacroCallId) -> span::MacroCallId {
        span::MacroCallId(value.as_id())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum HirFileId {
    FileId(EditionedFileId),
    MacroFile(MacroCallId),
}

impl From<EditionedFileId> for HirFileId {
    #[inline]
    fn from(file_id: EditionedFileId) -> Self {
        HirFileId::FileId(file_id)
    }
}

impl From<MacroCallId> for HirFileId {
    #[inline]
    fn from(file_id: MacroCallId) -> Self {
        HirFileId::MacroFile(file_id)
    }
}

impl PartialEq<EditionedFileId> for HirFileId {
    fn eq(&self, &other: &EditionedFileId) -> bool {
        *self == HirFileId::from(other)
    }
}
impl PartialEq<HirFileId> for EditionedFileId {
    fn eq(&self, &other: &HirFileId) -> bool {
        other == HirFileId::from(*self)
    }
}

impl HirFileId {
    #[inline]
    pub fn macro_file(self) -> Option<MacroCallId> {
        match self {
            HirFileId::FileId(_) => None,
            HirFileId::MacroFile(it) => Some(it),
        }
    }

    #[inline]
    pub fn is_macro(self) -> bool {
        matches!(self, HirFileId::MacroFile(_))
    }

    #[inline]
    pub fn file_id(self) -> Option<EditionedFileId> {
        match self {
            HirFileId::FileId(it) => Some(it),
            HirFileId::MacroFile(_) => None,
        }
    }

    pub fn syntax_context(self, db: &dyn SourceDatabase, edition: Edition) -> SyntaxContext {
        match self {
            HirFileId::FileId(_) => SyntaxContext::root(edition),
            HirFileId::MacroFile(m) => {
                let kind = &m.loc(db).kind;
                m.macro_arg_considering_derives(db, kind).2.ctx
            }
        }
    }

    pub fn edition(self, db: &dyn SourceDatabase) -> Edition {
        match self {
            HirFileId::FileId(file_id) => file_id.edition(db),
            HirFileId::MacroFile(m) => m.loc(db).def.edition,
        }
    }

    pub fn original_file(self, db: &dyn SourceDatabase) -> EditionedFileId {
        let mut file_id = self;
        loop {
            match file_id {
                HirFileId::FileId(id) => break id,
                HirFileId::MacroFile(macro_call_id) => {
                    file_id = macro_call_id.loc(db).kind.file_id()
                }
            }
        }
    }

    pub fn original_file_respecting_includes(mut self, db: &dyn SourceDatabase) -> EditionedFileId {
        loop {
            match self {
                HirFileId::FileId(id) => break id,
                HirFileId::MacroFile(file) => {
                    let loc = file.loc(db);
                    if loc.def.is_include()
                        && let MacroCallKind::FnLike { eager: Some(eager), .. } = &loc.kind
                        && let Ok(it) = include_input_to_file_id(db, file, &eager.arg)
                    {
                        break it;
                    }
                    self = loc.kind.file_id();
                }
            }
        }
    }

    pub fn original_call_node(self, db: &dyn SourceDatabase) -> Option<InRealFile<SyntaxNode>> {
        let mut call = self.macro_file()?.loc(db).to_node(db);
        loop {
            match call.file_id {
                HirFileId::FileId(file_id) => {
                    break Some(InRealFile { file_id, value: call.value });
                }
                HirFileId::MacroFile(macro_call_id) => {
                    call = macro_call_id.loc(db).to_node(db);
                }
            }
        }
    }

    pub fn call_node(self, db: &dyn SourceDatabase) -> Option<InFile<SyntaxNode>> {
        Some(self.macro_file()?.loc(db).to_node(db))
    }

    pub fn as_builtin_derive_attr_node(
        &self,
        db: &dyn SourceDatabase,
    ) -> Option<InFile<ast::Attr>> {
        let macro_file = self.macro_file()?;
        let loc = macro_file.loc(db);
        let attr = match loc.def.kind {
            MacroDefKind::BuiltInDerive(..) => loc.to_node(db),
            _ => return None,
        };
        Some(attr.with_value(ast::Attr::cast(attr.value.clone())?))
    }

    /// Main public API -- parses a hir file, not caring whether it's a real
    /// file or a macro expansion.
    pub fn parse_or_expand(self, db: &dyn SourceDatabase) -> SyntaxNode {
        match self {
            HirFileId::FileId(file_id) => file_id.parse(db).syntax_node(),
            HirFileId::MacroFile(macro_file) => {
                macro_file.parse_macro_expansion(db).value.0.syntax_node()
            }
        }
    }

    pub(crate) fn parse_with_map(
        self,
        db: &dyn SourceDatabase,
    ) -> (Parse<SyntaxNode>, SpanMap<'_>) {
        match self {
            HirFileId::FileId(file_id) => (
                file_id.parse(db).to_syntax(),
                SpanMap::RealSpanMap(crate::span_map::real_span_map(db, file_id)),
            ),
            HirFileId::MacroFile(macro_file) => {
                let (parse, map) = &macro_file.parse_macro_expansion(db).value;
                (parse.clone(), SpanMap::ExpansionSpanMap(map))
            }
        }
    }
}

#[salsa::tracked]
impl HirFileId {
    #[salsa::tracked(lru = 1024, returns(ref))]
    pub fn ast_id_map(self, db: &dyn SourceDatabase) -> AstIdMap {
        AstIdMap::from_source(&self.parse_or_expand(db))
    }
}
