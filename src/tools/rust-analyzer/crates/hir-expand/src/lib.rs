//! `hir_expand` deals with macro expansion.
//!
//! Specifically, it implements a concept of `MacroFile` -- a file whose syntax
//! tree originates not from the text of some `FileId`, but from some macro
//! expansion.
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

pub use intern;

pub mod attrs;
pub mod builtin;
pub mod change;
pub mod db;
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

use attrs::collect_attrs;
use rustc_hash::FxHashMap;
use salsa::plumbing::{AsId, FromId};
use stdx::TupleExt;
use triomphe::Arc;

use core::fmt;
use std::hash::Hash;

use base_db::Crate;
use either::Either;
use span::{Edition, ErasedFileAstId, FileAstId, Span, SpanAnchor, SyntaxContext};
use syntax::{
    SyntaxNode, SyntaxToken, TextRange, TextSize,
    ast::{self, AstNode},
};

use crate::{
    attrs::AttrId,
    builtin::{
        BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerExpander,
        include_input_to_file_id,
    },
    db::ExpandDatabase,
    mod_path::ModPath,
    proc_macro::{CustomProcMacroExpander, ProcMacroKind},
    span_map::{ExpansionSpanMap, SpanMap},
};

pub use crate::{
    cfg_process::check_cfg_attr_value,
    files::{AstId, ErasedAstId, FileRange, InFile, InMacroFile, InRealFile},
    prettify_macro_expansion_::prettify_macro_expansion,
};

pub use base_db::EditionedFileId;
pub use mbe::{DeclarativeMacro, ValueResult};

pub mod tt {
    pub use span::Span;
    pub use tt::{DelimiterKind, IdentIsRaw, LitKind, Spacing, token_to_literal};

    pub type Delimiter = ::tt::Delimiter<Span>;
    pub type DelimSpan = ::tt::DelimSpan<Span>;
    pub type Subtree = ::tt::Subtree<Span>;
    pub type Leaf = ::tt::Leaf<Span>;
    pub type Literal = ::tt::Literal<Span>;
    pub type Punct = ::tt::Punct<Span>;
    pub type Ident = ::tt::Ident<Span>;
    pub type TokenTree = ::tt::TokenTree<Span>;
    pub type TopSubtree = ::tt::TopSubtree<Span>;
    pub type TopSubtreeBuilder = ::tt::TopSubtreeBuilder<Span>;
    pub type TokenTreesView<'a> = ::tt::TokenTreesView<'a, Span>;
    pub type SubtreeView<'a> = ::tt::SubtreeView<'a, Span>;
    pub type TtElement<'a> = ::tt::iter::TtElement<'a, Span>;
    pub type TtIter<'a> = ::tt::iter::TtIter<'a, Span>;
}

#[macro_export]
macro_rules! impl_intern_lookup {
    ($db:ident, $id:ident, $loc:ident, $intern:ident, $lookup:ident) => {
        impl $crate::Intern for $loc {
            type Database = dyn $db;
            type ID = $id;
            fn intern(self, db: &Self::Database) -> Self::ID {
                db.$intern(self)
            }
        }

        impl $crate::Lookup for $id {
            type Database = dyn $db;
            type Data = $loc;
            fn lookup(&self, db: &Self::Database) -> Self::Data {
                db.$lookup(*self)
            }
        }
    };
}

// ideally these would be defined in base-db, but the orphan rule doesn't let us
pub trait Intern {
    type Database: ?Sized;
    type ID;
    fn intern(self, db: &Self::Database) -> Self::ID;
}

pub trait Lookup {
    type Database: ?Sized;
    type Data;
    fn lookup(&self, db: &Self::Database) -> Self::Data;
}

impl_intern_lookup!(
    ExpandDatabase,
    MacroCallId,
    MacroCallLoc,
    intern_macro_call,
    lookup_intern_macro_call
);

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

    pub fn render_to_string(&self, db: &dyn ExpandDatabase) -> RenderedExpandError {
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
    pub fn render_to_string(&self, db: &dyn ExpandDatabase) -> RenderedExpandError {
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
                match db.proc_macros_for_crate(def_crate).as_ref().and_then(|it| it.get_error()) {
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
    Declarative(AstId<ast::Macro>),
    BuiltIn(AstId<ast::Macro>, BuiltinFnLikeExpander),
    BuiltInAttr(AstId<ast::Macro>, BuiltinAttrExpander),
    BuiltInDerive(AstId<ast::Macro>, BuiltinDeriveExpander),
    BuiltInEager(AstId<ast::Macro>, EagerExpander),
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
            | MacroDefKind::Declarative(id, ..) => id.erase(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EagerCallInfo {
    /// The expanded argument of the eager macro.
    arg: Arc<tt::TopSubtree>,
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
        eager: Option<Arc<EagerCallInfo>>,
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
        /// The "parent" macro call.
        /// We will resolve the same token tree for all derive macros in the same derive attribute.
        derive_macro_id: MacroCallId,
    },
    Attr {
        ast_id: AstId<ast::Item>,
        // FIXME: This shouldn't be here, we can derive this from `invoc_attr_index`
        // but we need to fix the `cfg_attr` handling first.
        attr_args: Option<Arc<tt::TopSubtree>>,
        /// Syntactical index of the invoking `#[attribute]`.
        ///
        /// Outer attributes are counted first, then inner attributes. This does not support
        /// out-of-line modules, which may have attributes spread across 2 files!
        invoc_attr_index: AttrId,
    },
}

impl HirFileId {
    pub fn edition(self, db: &dyn ExpandDatabase) -> Edition {
        match self {
            HirFileId::FileId(file_id) => file_id.editioned_file_id(db).edition(),
            HirFileId::MacroFile(m) => db.lookup_intern_macro_call(m).def.edition,
        }
    }
    pub fn original_file(self, db: &dyn ExpandDatabase) -> EditionedFileId {
        let mut file_id = self;
        loop {
            match file_id {
                HirFileId::FileId(id) => break id,
                HirFileId::MacroFile(macro_call_id) => {
                    file_id = db.lookup_intern_macro_call(macro_call_id).kind.file_id()
                }
            }
        }
    }

    pub fn original_file_respecting_includes(mut self, db: &dyn ExpandDatabase) -> EditionedFileId {
        loop {
            match self {
                HirFileId::FileId(id) => break id,
                HirFileId::MacroFile(file) => {
                    let loc = db.lookup_intern_macro_call(file);
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

    pub fn original_call_node(self, db: &dyn ExpandDatabase) -> Option<InRealFile<SyntaxNode>> {
        let mut call = db.lookup_intern_macro_call(self.macro_file()?).to_node(db);
        loop {
            match call.file_id {
                HirFileId::FileId(file_id) => {
                    break Some(InRealFile { file_id, value: call.value });
                }
                HirFileId::MacroFile(macro_call_id) => {
                    call = db.lookup_intern_macro_call(macro_call_id).to_node(db);
                }
            }
        }
    }

    pub fn call_node(self, db: &dyn ExpandDatabase) -> Option<InFile<SyntaxNode>> {
        Some(db.lookup_intern_macro_call(self.macro_file()?).to_node(db))
    }

    pub fn as_builtin_derive_attr_node(
        &self,
        db: &dyn ExpandDatabase,
    ) -> Option<InFile<ast::Attr>> {
        let macro_file = self.macro_file()?;
        let loc = db.lookup_intern_macro_call(macro_file);
        let attr = match loc.def.kind {
            MacroDefKind::BuiltInDerive(..) => loc.to_node(db),
            _ => return None,
        };
        Some(attr.with_value(ast::Attr::cast(attr.value.clone())?))
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
    pub fn call_node(self, db: &dyn ExpandDatabase) -> InFile<SyntaxNode> {
        db.lookup_intern_macro_call(self).to_node(db)
    }
    pub fn expansion_level(self, db: &dyn ExpandDatabase) -> u32 {
        let mut level = 0;
        let mut macro_file = self;
        loop {
            let loc = db.lookup_intern_macro_call(macro_file);

            level += 1;
            macro_file = match loc.kind.file_id() {
                HirFileId::FileId(_) => break level,
                HirFileId::MacroFile(it) => it,
            };
        }
    }
    pub fn parent(self, db: &dyn ExpandDatabase) -> HirFileId {
        db.lookup_intern_macro_call(self).kind.file_id()
    }

    /// Return expansion information if it is a macro-expansion file
    pub fn expansion_info(self, db: &dyn ExpandDatabase) -> ExpansionInfo {
        ExpansionInfo::new(db, self)
    }

    pub fn kind(self, db: &dyn ExpandDatabase) -> MacroKind {
        match db.lookup_intern_macro_call(self).def.kind {
            MacroDefKind::Declarative(..) => MacroKind::Declarative,
            MacroDefKind::BuiltIn(..) | MacroDefKind::BuiltInEager(..) => {
                MacroKind::DeclarativeBuiltIn
            }
            MacroDefKind::BuiltInDerive(..) => MacroKind::DeriveBuiltIn,
            MacroDefKind::ProcMacro(_, _, ProcMacroKind::CustomDerive) => MacroKind::Derive,
            MacroDefKind::ProcMacro(_, _, ProcMacroKind::Attr) => MacroKind::Attr,
            MacroDefKind::ProcMacro(_, _, ProcMacroKind::Bang) => MacroKind::ProcMacro,
            MacroDefKind::BuiltInAttr(..) => MacroKind::AttrBuiltIn,
        }
    }

    pub fn is_include_macro(self, db: &dyn ExpandDatabase) -> bool {
        db.lookup_intern_macro_call(self).def.is_include()
    }

    pub fn is_include_like_macro(self, db: &dyn ExpandDatabase) -> bool {
        db.lookup_intern_macro_call(self).def.is_include_like()
    }

    pub fn is_env_or_option_env(self, db: &dyn ExpandDatabase) -> bool {
        db.lookup_intern_macro_call(self).def.is_env_or_option_env()
    }

    pub fn is_eager(self, db: &dyn ExpandDatabase) -> bool {
        let loc = db.lookup_intern_macro_call(self);
        matches!(loc.def.kind, MacroDefKind::BuiltInEager(..))
    }

    pub fn eager_arg(self, db: &dyn ExpandDatabase) -> Option<MacroCallId> {
        let loc = db.lookup_intern_macro_call(self);
        match &loc.kind {
            MacroCallKind::FnLike { eager, .. } => eager.as_ref().map(|it| it.arg_id),
            _ => None,
        }
    }

    pub fn is_derive_attr_pseudo_expansion(self, db: &dyn ExpandDatabase) -> bool {
        let loc = db.lookup_intern_macro_call(self);
        loc.def.is_attribute_derive()
    }
}

impl MacroDefId {
    pub fn make_call(
        self,
        db: &dyn ExpandDatabase,
        krate: Crate,
        kind: MacroCallKind,
        ctxt: SyntaxContext,
    ) -> MacroCallId {
        db.intern_macro_call(MacroCallLoc { def: self, krate, kind, ctxt })
    }

    pub fn definition_range(&self, db: &dyn ExpandDatabase) -> InFile<TextRange> {
        match self.kind {
            MacroDefKind::Declarative(id)
            | MacroDefKind::BuiltIn(id, _)
            | MacroDefKind::BuiltInAttr(id, _)
            | MacroDefKind::BuiltInDerive(id, _)
            | MacroDefKind::BuiltInEager(id, _) => {
                id.with_value(db.ast_id_map(id.file_id).get(id.value).text_range())
            }
            MacroDefKind::ProcMacro(id, _, _) => {
                id.with_value(db.ast_id_map(id.file_id).get(id.value).text_range())
            }
        }
    }

    pub fn ast_id(&self) -> Either<AstId<ast::Macro>, AstId<ast::Fn>> {
        match self.kind {
            MacroDefKind::ProcMacro(id, ..) => Either::Right(id),
            MacroDefKind::Declarative(id)
            | MacroDefKind::BuiltIn(id, _)
            | MacroDefKind::BuiltInAttr(id, _)
            | MacroDefKind::BuiltInDerive(id, _)
            | MacroDefKind::BuiltInEager(id, _) => Either::Left(id),
        }
    }

    pub fn is_proc_macro(&self) -> bool {
        matches!(self.kind, MacroDefKind::ProcMacro(..))
    }

    pub fn is_attribute(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltInAttr(..) | MacroDefKind::ProcMacro(_, _, ProcMacroKind::Attr)
        )
    }

    pub fn is_derive(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltInDerive(..)
                | MacroDefKind::ProcMacro(_, _, ProcMacroKind::CustomDerive)
        )
    }

    pub fn is_fn_like(&self) -> bool {
        matches!(
            self.kind,
            MacroDefKind::BuiltIn(..)
                | MacroDefKind::ProcMacro(_, _, ProcMacroKind::Bang)
                | MacroDefKind::BuiltInEager(..)
                | MacroDefKind::Declarative(..)
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
    pub fn to_node(&self, db: &dyn ExpandDatabase) -> InFile<SyntaxNode> {
        match self.kind {
            MacroCallKind::FnLike { ast_id, .. } => {
                ast_id.with_value(ast_id.to_node(db).syntax().clone())
            }
            MacroCallKind::Derive { ast_id, derive_attr_index, .. } => {
                // FIXME: handle `cfg_attr`
                ast_id.with_value(ast_id.to_node(db)).map(|it| {
                    collect_attrs(&it)
                        .nth(derive_attr_index.ast_index())
                        .and_then(|it| match it.1 {
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
                        collect_attrs(&it)
                            .nth(invoc_attr_index.ast_index())
                            .and_then(|it| match it.1 {
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

    pub fn to_node_item(&self, db: &dyn ExpandDatabase) -> InFile<ast::Item> {
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
        db: &dyn ExpandDatabase,
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
    pub fn original_call_range_with_input(self, db: &dyn ExpandDatabase) -> FileRange {
        let mut kind = self;
        let file_id = loop {
            match kind.file_id() {
                HirFileId::MacroFile(file) => {
                    kind = db.lookup_intern_macro_call(file).kind;
                }
                HirFileId::FileId(file_id) => break file_id,
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
    /// get the macro path (rustc shows the whole `ast::MacroCall`), attribute macros get the
    /// attribute's range, and derives get only the specific derive that is being referred to.
    pub fn original_call_range(self, db: &dyn ExpandDatabase) -> FileRange {
        let mut kind = self;
        let file_id = loop {
            match kind.file_id() {
                HirFileId::MacroFile(file) => {
                    kind = db.lookup_intern_macro_call(file).kind;
                }
                HirFileId::FileId(file_id) => break file_id,
            }
        };

        let range = match kind {
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
                // FIXME: handle `cfg_attr`
                collect_attrs(&ast_id.to_node(db))
                    .nth(derive_attr_index.ast_index())
                    .expect("missing derive")
                    .1
                    .expect_left("derive is a doc comment?")
                    .syntax()
                    .text_range()
            }
            // FIXME: handle `cfg_attr`
            MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => {
                collect_attrs(&ast_id.to_node(db))
                    .nth(invoc_attr_index.ast_index())
                    .expect("missing attribute")
                    .1
                    .expect_left("attribute macro is a doc comment?")
                    .syntax()
                    .text_range()
            }
        };

        FileRange { range, file_id }
    }

    fn arg(&self, db: &dyn ExpandDatabase) -> InFile<Option<SyntaxNode>> {
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
pub struct ExpansionInfo {
    expanded: InMacroFile<SyntaxNode>,
    /// The argument TokenTree or item for attributes
    arg: InFile<Option<SyntaxNode>>,
    exp_map: Arc<ExpansionSpanMap>,
    arg_map: SpanMap,
    loc: MacroCallLoc,
}

impl ExpansionInfo {
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
        let tokens = self.exp_map.ranges_with_span_exact(span).flat_map(move |(range, ctx)| {
            self.expanded.value.covering_element(range).into_token().zip(Some(ctx))
        });

        Some(InMacroFile::new(self.expanded.file_id, tokens))
    }

    /// Maps the passed in file range down into a macro expansion if it is the input to a macro call.
    /// Unlike [`map_range_down_exact`], this will consider spans that contain the given span.
    ///
    /// Note this does a linear search through the entire backing vector of the spanmap.
    pub fn map_range_down(
        &self,
        span: Span,
    ) -> Option<InMacroFile<impl Iterator<Item = (SyntaxToken, SyntaxContext)> + '_>> {
        let tokens = self.exp_map.ranges_with_span(span).flat_map(move |(range, ctx)| {
            self.expanded.value.covering_element(range).into_token().zip(Some(ctx))
        });

        Some(InMacroFile::new(self.expanded.file_id, tokens))
    }

    /// Looks up the span at the given offset.
    pub fn span_for_offset(
        &self,
        db: &dyn ExpandDatabase,
        offset: TextSize,
    ) -> (FileRange, SyntaxContext) {
        debug_assert!(self.expanded.value.text_range().contains(offset));
        span_for_offset(db, &self.exp_map, offset)
    }

    /// Maps up the text range out of the expansion hierarchy back into the original file its from.
    pub fn map_node_range_up(
        &self,
        db: &dyn ExpandDatabase,
        range: TextRange,
    ) -> Option<(FileRange, SyntaxContext)> {
        debug_assert!(self.expanded.value.text_range().contains_range(range));
        map_node_range_up(db, &self.exp_map, range)
    }

    /// Maps up the text range out of the expansion into its macro call.
    ///
    /// Note that this may return multiple ranges as we lose the precise association between input to output
    /// and as such we may consider inputs that are unrelated.
    pub fn map_range_up_once(
        &self,
        db: &dyn ExpandDatabase,
        token: TextRange,
    ) -> InFile<smallvec::SmallVec<[TextRange; 1]>> {
        debug_assert!(self.expanded.value.text_range().contains_range(token));
        let span = self.exp_map.span_at(token.start());
        match &self.arg_map {
            SpanMap::RealSpanMap(_) => {
                let file_id = EditionedFileId::from_span(db, span.anchor.file_id).into();
                let anchor_offset =
                    db.ast_id_map(file_id).get_erased(span.anchor.ast_id).text_range().start();
                InFile { file_id, value: smallvec::smallvec![span.range + anchor_offset] }
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
                        .filter(|(range, _)| range.intersect(arg_range).is_some())
                        .map(TupleExt::head)
                        .collect(),
                )
            }
        }
    }

    pub fn new(db: &dyn ExpandDatabase, macro_file: MacroCallId) -> ExpansionInfo {
        let _p = tracing::info_span!("ExpansionInfo::new").entered();
        let loc = db.lookup_intern_macro_call(macro_file);

        let arg_tt = loc.kind.arg(db);
        let arg_map = db.span_map(arg_tt.file_id);

        let (parse, exp_map) = db.parse_macro_expansion(macro_file).value;
        let expanded = InMacroFile { file_id: macro_file, value: parse.syntax_node() };

        ExpansionInfo { expanded, loc, arg: arg_tt, exp_map, arg_map }
    }
}

/// Maps up the text range out of the expansion hierarchy back into the original file its from only
/// considering the root spans contained.
/// Unlike [`map_node_range_up`], this will not return `None` if any anchors or syntax contexts differ.
pub fn map_node_range_up_rooted(
    db: &dyn ExpandDatabase,
    exp_map: &ExpansionSpanMap,
    range: TextRange,
) -> Option<FileRange> {
    let mut spans = exp_map.spans_for_range(range).filter(|span| span.ctx.is_root());
    let Span { range, anchor, ctx: _ } = spans.next()?;
    let mut start = range.start();
    let mut end = range.end();

    for span in spans {
        if span.anchor != anchor {
            return None;
        }
        start = start.min(span.range.start());
        end = end.max(span.range.end());
    }
    let file_id = EditionedFileId::from_span(db, anchor.file_id);
    let anchor_offset =
        db.ast_id_map(file_id.into()).get_erased(anchor.ast_id).text_range().start();
    Some(FileRange { file_id, range: TextRange::new(start, end) + anchor_offset })
}

/// Maps up the text range out of the expansion hierarchy back into the original file its from.
///
/// this will return `None` if any anchors or syntax contexts differ.
pub fn map_node_range_up(
    db: &dyn ExpandDatabase,
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
    let file_id = EditionedFileId::from_span(db, anchor.file_id);
    let anchor_offset =
        db.ast_id_map(file_id.into()).get_erased(anchor.ast_id).text_range().start();
    Some((FileRange { file_id, range: TextRange::new(start, end) + anchor_offset }, ctx))
}

/// Maps up the text range out of the expansion hierarchy back into the original file its from.
/// This version will aggregate the ranges of all spans with the same anchor and syntax context.
pub fn map_node_range_up_aggregated(
    db: &dyn ExpandDatabase,
    exp_map: &ExpansionSpanMap,
    range: TextRange,
) -> FxHashMap<(SpanAnchor, SyntaxContext), TextRange> {
    let mut map = FxHashMap::default();
    for span in exp_map.spans_for_range(range) {
        let range = map.entry((span.anchor, span.ctx)).or_insert_with(|| span.range);
        *range = TextRange::new(
            range.start().min(span.range.start()),
            range.end().max(span.range.end()),
        );
    }
    for ((anchor, _), range) in &mut map {
        let file_id = EditionedFileId::from_span(db, anchor.file_id);
        let anchor_offset =
            db.ast_id_map(file_id.into()).get_erased(anchor.ast_id).text_range().start();
        *range += anchor_offset;
    }
    map
}

/// Looks up the span at the given offset.
pub fn span_for_offset(
    db: &dyn ExpandDatabase,
    exp_map: &ExpansionSpanMap,
    offset: TextSize,
) -> (FileRange, SyntaxContext) {
    let span = exp_map.span_at(offset);
    let file_id = EditionedFileId::from_span(db, span.anchor.file_id);
    let anchor_offset =
        db.ast_id_map(file_id.into()).get_erased(span.anchor.ast_id).text_range().start();
    (FileRange { file_id, range: span.range + anchor_offset }, span.ctx)
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

intern::impl_internable!(ModPath, attrs::AttrInput);

#[salsa_macros::interned(no_lifetime, debug, revisions = usize::MAX)]
#[doc(alias = "MacroFileId")]
pub struct MacroCallId {
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
