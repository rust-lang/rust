use std::fmt;

use rustc_ast::attr::AttributeExt;
use rustc_ast::token::DocFragmentKind;
use rustc_ast::{AttrStyle, DelimArgs, MetaItemLit, ast, join_path_idents};
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_error_messages::{DiagArgValue, IntoDiagArg};
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::{AttrId, DUMMY_SP, Ident, Span, Symbol};
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::AttributeKind;

/// Arguments of a unparsed attribute.
#[derive(Clone, Debug, StableHash, Encodable, Decodable)]
pub enum AttrArgs {
    /// No arguments: `#[attr]`.
    Empty,
    /// Delimited arguments: `#[attr()/[]/{}]`.
    Delimited(DelimArgs),
    /// Arguments of a key-value attribute: `#[attr = "value"]`.
    Eq {
        /// Span of the `=` token.
        eq_span: Span,
        /// The "value".
        expr: MetaItemLit,
    },
}

#[derive(Clone, Debug, StableHash, Encodable, Decodable)]
pub struct AttrPath {
    pub segments: Box<[Symbol]>,
    pub span: Span,
}

impl IntoDiagArg for AttrPath {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(path)
    }
}

impl AttrPath {
    pub fn from_ast(path: &ast::Path, lower_span: impl Copy + Fn(Span) -> Span) -> Self {
        AttrPath {
            segments: path
                .segments
                .iter()
                .map(|i| i.ident.name)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            span: lower_span(path.span),
        }
    }
}

impl fmt::Display for AttrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            join_path_idents(self.segments.iter().map(|i| Ident { name: *i, span: DUMMY_SP }))
        )
    }
}

/// A non built-in attribute.
///
/// This is either a tool attribute or a derive helper attribute.
#[derive(Clone, Debug, StableHash, Encodable, Decodable)]
pub struct AttrItem {
    // Not lowered to hir::Path because we have no NodeId to resolve to.
    pub path: AttrPath,
    pub args: AttrArgs,
    /// Denotes if the attribute decorates the following construct (outer)
    /// or the construct this attribute is contained within (inner).
    pub style: AttrStyle,
    /// Span of the entire attribute
    pub span: Span,
}

/// The derived implementation of [`StableHash`] on [`Attribute`]s shouldn't hash
/// [`AttrId`]s. By wrapping them in this, we make sure we never do.
#[derive(Copy, Debug, Encodable, Decodable, Clone)]
pub struct HashIgnoredAttrId {
    pub attr_id: AttrId,
}

impl StableHash for HashIgnoredAttrId {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _hcx: &mut Hcx, _hasher: &mut StableHasher) {
        /* we don't hash HashIgnoredAttrId, we ignore them */
    }
}
/// A high level representation of an attribute.
///
/// Not to be confused with [`ast::Attribute`], which represents the syntax-level form of
/// an attribute.
///
/// [`ast::Attribute`]: rustc_ast::ast::Attribute
#[derive(Clone, Debug, Encodable, Decodable, StableHash)]
pub enum Attribute {
    /// A parsed built-in attribute.
    ///
    /// Parsed attributes don't generally have a span associated with them. This is because
    /// multiple syntax-level attributes can be represented into a single semantic attribute. For
    /// example, an item can have multiple `repr` attributes on it, which will coalesce into a
    /// single [`AttributeKind::Repr`], and the `stable`, `unstable`, and
    /// `rustc_allowed_through_unstable_modules` all combine into a single
    /// [`AttributeKind::Stability`] attribute. Most `Attribute` variants do have a [`Span`] field,
    /// though, mostly for the purpose of generating diagnostics later in the compilation process.
    Parsed(AttributeKind),

    /// A tool attribute.
    ///
    /// Tool attributes, with the exception of diagnostic attributes, are not converted into a
    /// parsed form.
    Tool(Box<AttrItem>),

    /// A derive helper attribute.
    ///
    /// Derive helper attributes serve no use beyond macro expansion, but they are useful to
    /// consumers of `rustdoc-json`, so we keep them around. See
    /// <https://github.com/rust-lang/rust/issues/157107> for context.
    DeriveHelper(Box<AttrItem>),
}

impl Attribute {
    pub fn value_lit(&self) -> Option<&MetaItemLit> {
        match &self {
            Attribute::Tool(n) | Attribute::DeriveHelper(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Eq { eq_span: _, expr }, .. } => Some(expr),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn is_prefix_attr_for_suggestions(&self) -> bool {
        match self {
            Attribute::Tool(attr) | Attribute::DeriveHelper(attr) => {
                attr.span.desugaring_kind().is_none()
            }
            // Other parsed attributes that can appear on expressions originate from source and
            // should make suggestions treat the expression like a prefixed form.
            Attribute::Parsed(_) => true,
        }
    }
}

impl AttributeExt for Attribute {
    #[inline]
    fn meta_item_list(&self) -> Option<ThinVec<ast::MetaItemInner>> {
        match &self {
            Attribute::Tool(n) | Attribute::DeriveHelper(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Delimited(d), .. } => {
                    ast::MetaItemKind::list_from_tokens(d.tokens.clone())
                }
                _ => None,
            },
            _ => None,
        }
    }

    #[inline]
    fn value_str(&self) -> Option<Symbol> {
        self.value_lit().and_then(|x| x.value_as_str())
    }

    #[inline]
    fn value_span(&self) -> Option<Span> {
        self.value_lit().map(|i| i.span)
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    #[inline]
    fn name(&self) -> Option<Symbol> {
        match &self {
            Attribute::DeriveHelper(n) => {
                if let [ident] = n.path.segments.as_ref() {
                    Some(*ident)
                } else {
                    None
                }
            }
            Attribute::Parsed(..) => None,
            // Tools are never single segment attributes
            Attribute::Tool(..) => None,
        }
    }

    #[inline]
    fn path_matches(&self, name: &[Symbol]) -> bool {
        match &self {
            Attribute::Tool(n) | Attribute::DeriveHelper(n) => n.path.segments.iter().eq(name),
            _ => false,
        }
    }

    #[inline]
    fn is_doc_comment(&self) -> Option<Span> {
        if let Attribute::Parsed(AttributeKind::DocComment { span, .. }) = self {
            Some(*span)
        } else {
            None
        }
    }

    #[inline]
    fn symbol_path(&self) -> Option<SmallVec<[Symbol; 1]>> {
        match &self {
            Attribute::Tool(n) | Attribute::DeriveHelper(n) => {
                Some(n.path.segments.iter().copied().collect())
            }
            _ => None,
        }
    }

    fn path_span(&self) -> Option<Span> {
        match &self {
            Attribute::Tool(attr) | Attribute::DeriveHelper(attr) => Some(attr.path.span),
            Attribute::Parsed(_) => None,
        }
    }

    #[inline]
    fn doc_str(&self) -> Option<Symbol> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { comment, .. }) => Some(*comment),
            _ => None,
        }
    }

    #[inline]
    fn doc_str_and_fragment_kind(&self) -> Option<(Symbol, DocFragmentKind, Span)> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { kind, comment, span, .. }) => {
                Some((*comment, *kind, *span))
            }
            _ => None,
        }
    }

    fn doc_resolution_scope(&self) -> Option<AttrStyle> {
        match self {
            Attribute::Parsed(AttributeKind::DocComment { style, .. }) => Some(*style),
            _ => None,
        }
    }

    fn is_proc_macro_attr(&self) -> bool {
        matches!(
            self,
            Attribute::Parsed(
                AttributeKind::ProcMacro
                    | AttributeKind::ProcMacroAttribute
                    | AttributeKind::ProcMacroDerive { .. }
            )
        )
    }
}
