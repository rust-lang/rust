use std::fmt;

use rustc_ast::attr::AttributeExt;
use rustc_ast::token::DocFragmentKind;
use rustc_ast::{AttrStyle, DelimArgs, MetaItemInner, MetaItemLit, ast, join_path_idents};
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_error_messages::{DiagArgValue, IntoDiagArg};
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::{AttrId, DUMMY_SP, Ident, Span, Symbol, sym};
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::AttributeKind;
/// Arguments passed to an attribute macro.
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

/// Many functions on this type have their documentation in the [`AttributeExt`] trait,
/// since they defer their implementation directly to that trait.
#[derive(Clone, Debug, Encodable, Decodable, StableHash)]
pub enum Attribute {
    /// A parsed built-in attribute.
    ///
    /// Each attribute has a span connected to it. However, you must be somewhat careful using it.
    /// That's because sometimes we merge multiple attributes together, like when an item has
    /// multiple `repr` attributes. In this case the span might not be very useful.
    Parsed(AttributeKind),

    /// An attribute that could not be parsed, out of a token-like representation.
    /// This is the case for custom tool attributes.
    Unparsed(Box<AttrItem>),
}

impl Attribute {
    pub fn get_normal_item(&self) -> &AttrItem {
        match &self {
            Attribute::Unparsed(normal) => &normal,
            _ => panic!("unexpected parsed attribute"),
        }
    }

    pub fn value_lit(&self) -> Option<&MetaItemLit> {
        match &self {
            Attribute::Unparsed(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Eq { eq_span: _, expr }, .. } => Some(expr),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn is_parsed_attr(&self) -> bool {
        match self {
            Attribute::Parsed(_) => true,
            Attribute::Unparsed(_) => false,
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
    fn id(&self) -> AttrId {
        match &self {
            Attribute::Unparsed(u) => u.id.attr_id,
            _ => panic!(),
        }
    }

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
    fn span(&self) -> Span {
        match &self {
            Attribute::Unparsed(u) => u.span,
            // FIXME: should not be needed anymore when all attrs are parsed
            Attribute::Parsed(AttributeKind::DocComment { span, .. }) => *span,
            Attribute::Parsed(AttributeKind::Deprecated { span, .. }) => *span,
            Attribute::Parsed(AttributeKind::CfgTrace(cfgs)) => cfgs[0].1,
            a => panic!("can't get the span of an arbitrary parsed attribute: {a:?}"),
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
    fn doc_str_and_fragment_kind(&self) -> Option<(Symbol, DocFragmentKind)> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { kind, comment, .. }) => {
                Some((*comment, *kind))
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

// FIXME(fn_delegation): use function delegation instead of manually forwarding
impl Attribute {
    #[inline]
    pub fn id(&self) -> AttrId {
        AttributeExt::id(self)
    }

    #[inline]
    pub fn name(&self) -> Option<Symbol> {
        AttributeExt::name(self)
    }

    #[inline]
    pub fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        AttributeExt::meta_item_list(self)
    }

    #[inline]
    pub fn value_str(&self) -> Option<Symbol> {
        AttributeExt::value_str(self)
    }

    #[inline]
    pub fn value_span(&self) -> Option<Span> {
        AttributeExt::value_span(self)
    }

    #[inline]
    pub fn path_matches(&self, name: &[Symbol]) -> bool {
        AttributeExt::path_matches(self, name)
    }

    #[inline]
    pub fn is_doc_comment(&self) -> Option<Span> {
        AttributeExt::is_doc_comment(self)
    }

    #[inline]
    pub fn has_name(&self, name: Symbol) -> bool {
        AttributeExt::has_name(self, name)
    }

    #[inline]
    pub fn has_any_name(&self, names: &[Symbol]) -> bool {
        AttributeExt::has_any_name(self, names)
    }

    #[inline]
    pub fn span(&self) -> Span {
        AttributeExt::span(self)
    }

    #[inline]
    pub fn is_word(&self) -> bool {
        AttributeExt::is_word(self)
    }

    #[inline]
    pub fn path(&self) -> SmallVec<[Symbol; 1]> {
        AttributeExt::path(self)
    }

    #[inline]
    pub fn doc_str(&self) -> Option<Symbol> {
        AttributeExt::doc_str(self)
    }

    #[inline]
    pub fn is_proc_macro_attr(&self) -> bool {
        AttributeExt::is_proc_macro_attr(self)
    }

    #[inline]
    pub fn doc_str_and_fragment_kind(&self) -> Option<(Symbol, DocFragmentKind)> {
        AttributeExt::doc_str_and_fragment_kind(self)
    }
}
