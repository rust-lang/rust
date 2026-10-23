use std::fmt;

use rustc_ast::attr::AttributeExt;
use rustc_ast::token::DocFragmentKind;
use rustc_ast::{AttrStyle, DelimArgs, MetaItemLit, ast, join_path_idents};
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_error_messages::{DiagArgValue, IntoDiagArg};
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::{AttrId, DUMMY_SP, Ident, Span, Symbol, sym};
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
    pub id: HashIgnoredAttrId,
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
    /// Represents non built-in attributes, which are not converted into a parsed form.
    ///
    /// This is the case for custom tool attributes and derive helper attributes.
    Unparsed(Box<AttrItem>),
}

impl Attribute {
    pub fn value_lit(&self) -> Option<&MetaItemLit> {
        match &self {
            Attribute::Unparsed(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Eq { eq_span: _, expr }, .. } => Some(expr),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn is_prefix_attr_for_suggestions(&self) -> bool {
        match self {
            Attribute::Unparsed(attr) => attr.span.desugaring_kind().is_none(),
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
            Attribute::Unparsed(n) => match n.as_ref() {
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
            Attribute::Unparsed(n) => {
                if let [ident] = n.path.segments.as_ref() {
                    Some(*ident)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn path_matches(&self, name: &[Symbol]) -> bool {
        match &self {
            Attribute::Unparsed(n) => n.path.segments.iter().eq(name),
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
    fn is_word(&self) -> bool {
        match &self {
            Attribute::Unparsed(n) => {
                matches!(n.args, AttrArgs::Empty)
            }
            _ => false,
        }
    }

    #[inline]
    fn symbol_path(&self) -> Option<SmallVec<[Symbol; 1]>> {
        match &self {
            Attribute::Unparsed(n) => Some(n.path.segments.iter().copied().collect()),
            _ => None,
        }
    }

    fn path_span(&self) -> Option<Span> {
        match &self {
            Attribute::Unparsed(attr) => Some(attr.path.span),
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

    fn is_automatically_derived_attr(&self) -> bool {
        matches!(self, Attribute::Parsed(AttributeKind::AutomaticallyDerived))
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
            Attribute::Unparsed(attr) if self.has_name(sym::doc) && self.value_str().is_some() => {
                Some(attr.style)
            }
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

    fn is_doc_hidden(&self) -> bool {
        matches!(self, Attribute::Parsed(AttributeKind::Doc(d)) if d.hidden.is_some())
    }

    fn is_doc_keyword_or_attribute(&self) -> bool {
        matches!(self, Attribute::Parsed(AttributeKind::Doc(d)) if d.attribute.is_some() || d.keyword.is_some())
    }

    fn is_rustc_doc_primitive(&self) -> bool {
        matches!(self, Attribute::Parsed(AttributeKind::RustcDocPrimitive(..)))
    }
}
