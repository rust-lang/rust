//! Functions dealing with attributes and meta items.

use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};

use rustc_index::bit_set::GrowableBitSet;
use rustc_span::{Ident, Span, Symbol, sym};
use smallvec::{SmallVec, smallvec};
use thin_vec::{ThinVec, thin_vec};

use crate::ast::{
    AttrArgs, AttrId, AttrItem, AttrKind, AttrStyle, AttrVec, Attribute, DUMMY_NODE_ID, DelimArgs,
    Expr, ExprKind, LitKind, MetaItem, MetaItemInner, MetaItemKind, MetaItemLit, NormalAttr, Path,
    PathSegment, Safety,
};
use crate::ptr::P;
use crate::token::{self, CommentKind, Delimiter, InvisibleOrigin, MetaVarKind, Token};
use crate::tokenstream::{
    DelimSpan, LazyAttrTokenStream, Spacing, TokenStream, TokenStreamIter, TokenTree,
};
use crate::util::comments;
use crate::util::literal::escape_string_symbol;

pub struct MarkedAttrs(GrowableBitSet<AttrId>);

impl MarkedAttrs {
    pub fn new() -> Self {
        // We have no idea how many attributes there will be, so just
        // initiate the vectors with 0 bits. We'll grow them as necessary.
        MarkedAttrs(GrowableBitSet::new_empty())
    }

    pub fn mark(&mut self, attr: &Attribute) {
        self.0.insert(attr.id);
    }

    pub fn is_marked(&self, attr: &Attribute) -> bool {
        self.0.contains(attr.id)
    }
}

pub struct AttrIdGenerator(AtomicU32);

impl AttrIdGenerator {
    pub fn new() -> Self {
        AttrIdGenerator(AtomicU32::new(0))
    }

    pub fn mk_attr_id(&self) -> AttrId {
        let id = self.0.fetch_add(1, Ordering::Relaxed);
        assert!(id != u32::MAX);
        AttrId::from_u32(id)
    }
}

impl Attribute {
    pub fn get_normal_item(&self) -> &AttrItem {
        match &self.kind {
            AttrKind::Normal(normal) => &normal.item,
            AttrKind::DocComment(..) => panic!("unexpected doc comment"),
        }
    }

    pub fn unwrap_normal_item(self) -> AttrItem {
        match self.kind {
            AttrKind::Normal(normal) => normal.into_inner().item,
            AttrKind::DocComment(..) => panic!("unexpected doc comment"),
        }
    }
}

impl AttributeExt for Attribute {
    fn id(&self) -> AttrId {
        self.id
    }

    fn value_span(&self) -> Option<Span> {
        match &self.kind {
            AttrKind::Normal(normal) => match &normal.item.args {
                AttrArgs::Eq { expr, .. } => Some(expr.span),
                _ => None,
            },
            AttrKind::DocComment(..) => None,
        }
    }

    /// Returns `true` if it is a sugared doc comment (`///` or `//!` for example).
    /// So `#[doc = "doc"]` (which is a doc comment) and `#[doc(...)]` (which is not
    /// a doc comment) will return `false`.
    fn is_doc_comment(&self) -> bool {
        match self.kind {
            AttrKind::Normal(..) => false,
            AttrKind::DocComment(..) => true,
        }
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    fn ident(&self) -> Option<Ident> {
        match &self.kind {
            AttrKind::Normal(normal) => {
                if let [ident] = &*normal.item.path.segments {
                    Some(ident.ident)
                } else {
                    None
                }
            }
            AttrKind::DocComment(..) => None,
        }
    }

    fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>> {
        match &self.kind {
            AttrKind::Normal(p) => Some(p.item.path.segments.iter().map(|i| i.ident).collect()),
            AttrKind::DocComment(_, _) => None,
        }
    }

    fn path_matches(&self, name: &[Symbol]) -> bool {
        match &self.kind {
            AttrKind::Normal(normal) => {
                normal.item.path.segments.len() == name.len()
                    && normal
                        .item
                        .path
                        .segments
                        .iter()
                        .zip(name)
                        .all(|(s, n)| s.args.is_none() && s.ident.name == *n)
            }
            AttrKind::DocComment(..) => false,
        }
    }

    fn span(&self) -> Span {
        self.span
    }

    fn is_word(&self) -> bool {
        if let AttrKind::Normal(normal) = &self.kind {
            matches!(normal.item.args, AttrArgs::Empty)
        } else {
            false
        }
    }

    /// Returns a list of meta items if the attribute is delimited with parenthesis:
    ///
    /// ```text
    /// #[attr(a, b = "c")] // Returns `Some()`.
    /// #[attr = ""] // Returns `None`.
    /// #[attr] // Returns `None`.
    /// ```
    fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        match &self.kind {
            AttrKind::Normal(normal) => normal.item.meta_item_list(),
            AttrKind::DocComment(..) => None,
        }
    }

    /// Returns the string value in:
    ///
    /// ```text
    /// #[attribute = "value"]
    ///               ^^^^^^^
    /// ```
    ///
    /// It returns `None` in any other cases, including doc comments if they
    /// are not under the form `#[doc = "..."]`.
    ///
    /// It also returns `None` for:
    ///
    /// ```text
    /// #[attr("value")]
    /// ```
    fn value_str(&self) -> Option<Symbol> {
        match &self.kind {
            AttrKind::Normal(normal) => normal.item.value_str(),
            AttrKind::DocComment(..) => None,
        }
    }

    /// Returns the documentation and its kind if this is a doc comment or a sugared doc comment.
    /// * `///doc` returns `Some(("doc", CommentKind::Line))`.
    /// * `/** doc */` returns `Some(("doc", CommentKind::Block))`.
    /// * `#[doc = "doc"]` returns `Some(("doc", CommentKind::Line))`.
    /// * `#[doc(...)]` returns `None`.
    fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        match &self.kind {
            AttrKind::DocComment(kind, data) => Some((*data, *kind)),
            AttrKind::Normal(normal) if normal.item.path == sym::doc => {
                normal.item.value_str().map(|s| (s, CommentKind::Line))
            }
            _ => None,
        }
    }

    /// Returns the documentation if this is a doc comment or a sugared doc comment.
    /// * `///doc` returns `Some("doc")`.
    /// * `#[doc = "doc"]` returns `Some("doc")`.
    /// * `#[doc(...)]` returns `None`.
    fn doc_str(&self) -> Option<Symbol> {
        match &self.kind {
            AttrKind::DocComment(.., data) => Some(*data),
            AttrKind::Normal(normal) if normal.item.path == sym::doc => normal.item.value_str(),
            _ => None,
        }
    }

    fn style(&self) -> AttrStyle {
        self.style
    }
}

impl Attribute {
    pub fn may_have_doc_links(&self) -> bool {
        self.doc_str().is_some_and(|s| comments::may_have_doc_links(s.as_str()))
    }

    /// Extracts the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        match &self.kind {
            AttrKind::Normal(normal) => normal.item.meta(self.span),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn meta_kind(&self) -> Option<MetaItemKind> {
        match &self.kind {
            AttrKind::Normal(normal) => normal.item.meta_kind(),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn token_trees(&self) -> Vec<TokenTree> {
        match self.kind {
            AttrKind::Normal(ref normal) => normal
                .tokens
                .as_ref()
                .unwrap_or_else(|| panic!("attribute is missing tokens: {self:?}"))
                .to_attr_token_stream()
                .to_token_trees(),
            AttrKind::DocComment(comment_kind, data) => vec![TokenTree::token_alone(
                token::DocComment(comment_kind, self.style, data),
                self.span,
            )],
        }
    }
}

impl AttrItem {
    pub fn span(&self) -> Span {
        self.args.span().map_or(self.path.span, |args_span| self.path.span.to(args_span))
    }

    pub fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        match &self.args {
            AttrArgs::Delimited(args) if args.delim == Delimiter::Parenthesis => {
                MetaItemKind::list_from_tokens(args.tokens.clone())
            }
            AttrArgs::Delimited(_) | AttrArgs::Eq { .. } | AttrArgs::Empty => None,
        }
    }

    /// Returns the string value in:
    ///
    /// ```text
    /// #[attribute = "value"]
    ///               ^^^^^^^
    /// ```
    ///
    /// It returns `None` in any other cases like:
    ///
    /// ```text
    /// #[attr("value")]
    /// ```
    fn value_str(&self) -> Option<Symbol> {
        match &self.args {
            AttrArgs::Eq { expr, .. } => match expr.kind {
                ExprKind::Lit(token_lit) => {
                    LitKind::from_token_lit(token_lit).ok().and_then(|lit| lit.str())
                }
                _ => None,
            },
            AttrArgs::Delimited(_) | AttrArgs::Empty => None,
        }
    }

    pub fn meta(&self, span: Span) -> Option<MetaItem> {
        Some(MetaItem {
            unsafety: Safety::Default,
            path: self.path.clone(),
            kind: self.meta_kind()?,
            span,
        })
    }

    pub fn meta_kind(&self) -> Option<MetaItemKind> {
        MetaItemKind::from_attr_args(&self.args)
    }
}

impl MetaItem {
    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        if let [PathSegment { ident, .. }] = self.path.segments[..] { Some(ident) } else { None }
    }

    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::empty).name
    }

    pub fn has_name(&self, name: Symbol) -> bool {
        self.path == name
    }

    pub fn is_word(&self) -> bool {
        matches!(self.kind, MetaItemKind::Word)
    }

    pub fn meta_item_list(&self) -> Option<&[MetaItemInner]> {
        match &self.kind {
            MetaItemKind::List(l) => Some(&**l),
            _ => None,
        }
    }

    /// ```text
    /// Example:
    ///     #[attribute(name = "value")]
    ///                 ^^^^^^^^^^^^^^
    /// ```
    pub fn name_value_literal(&self) -> Option<&MetaItemLit> {
        match &self.kind {
            MetaItemKind::NameValue(v) => Some(v),
            _ => None,
        }
    }

    /// This is used in case you want the value span instead of the whole attribute. Example:
    ///
    /// ```text
    /// #[doc(alias = "foo")]
    /// ```
    ///
    /// In here, it'll return a span for `"foo"`.
    pub fn name_value_literal_span(&self) -> Option<Span> {
        Some(self.name_value_literal()?.span)
    }

    /// Returns the string value in:
    ///
    /// ```text
    /// #[attribute = "value"]
    ///               ^^^^^^^
    /// ```
    ///
    /// It returns `None` in any other cases like:
    ///
    /// ```text
    /// #[attr("value")]
    /// ```
    pub fn value_str(&self) -> Option<Symbol> {
        match &self.kind {
            MetaItemKind::NameValue(v) => v.kind.str(),
            _ => None,
        }
    }

    fn from_tokens(iter: &mut TokenStreamIter<'_>) -> Option<MetaItem> {
        // FIXME: Share code with `parse_path`.
        let tt = iter.next().map(|tt| TokenTree::uninterpolate(tt));
        let path = match tt.as_deref() {
            Some(&TokenTree::Token(
                Token { kind: ref kind @ (token::Ident(..) | token::PathSep), span },
                _,
            )) => 'arm: {
                let mut segments = if let &token::Ident(name, _) = kind {
                    if let Some(TokenTree::Token(Token { kind: token::PathSep, .. }, _)) =
                        iter.peek()
                    {
                        iter.next();
                        thin_vec![PathSegment::from_ident(Ident::new(name, span))]
                    } else {
                        break 'arm Path::from_ident(Ident::new(name, span));
                    }
                } else {
                    thin_vec![PathSegment::path_root(span)]
                };
                loop {
                    if let Some(&TokenTree::Token(Token { kind: token::Ident(name, _), span }, _)) =
                        iter.next().map(|tt| TokenTree::uninterpolate(tt)).as_deref()
                    {
                        segments.push(PathSegment::from_ident(Ident::new(name, span)));
                    } else {
                        return None;
                    }
                    if let Some(TokenTree::Token(Token { kind: token::PathSep, .. }, _)) =
                        iter.peek()
                    {
                        iter.next();
                    } else {
                        break;
                    }
                }
                let span = span.with_hi(segments.last().unwrap().ident.span.hi());
                Path { span, segments, tokens: None }
            }
            Some(TokenTree::Delimited(
                _span,
                _spacing,
                Delimiter::Invisible(InvisibleOrigin::MetaVar(
                    MetaVarKind::Meta { .. } | MetaVarKind::Path,
                )),
                _stream,
            )) => {
                // This path is currently unreachable in the test suite.
                unreachable!()
            }
            Some(TokenTree::Token(
                Token { kind: token::OpenDelim(_) | token::CloseDelim(_), .. },
                _,
            )) => {
                panic!("Should be `AttrTokenTree::Delimited`, not delim tokens: {:?}", tt);
            }
            _ => return None,
        };
        let list_closing_paren_pos = iter.peek().map(|tt| tt.span().hi());
        let kind = MetaItemKind::from_tokens(iter)?;
        let hi = match &kind {
            MetaItemKind::NameValue(lit) => lit.span.hi(),
            MetaItemKind::List(..) => list_closing_paren_pos.unwrap_or(path.span.hi()),
            _ => path.span.hi(),
        };
        let span = path.span.with_hi(hi);
        // FIXME: This parses `unsafe()` not as unsafe attribute syntax in `MetaItem`,
        // but as a parenthesized list. This (and likely `MetaItem`) should be changed in
        // such a way that builtin macros don't accept extraneous `unsafe()`.
        Some(MetaItem { unsafety: Safety::Default, path, kind, span })
    }
}

impl MetaItemKind {
    // public because it can be called in the hir
    pub fn list_from_tokens(tokens: TokenStream) -> Option<ThinVec<MetaItemInner>> {
        let mut iter = tokens.iter();
        let mut result = ThinVec::new();
        while iter.peek().is_some() {
            let item = MetaItemInner::from_tokens(&mut iter)?;
            result.push(item);
            match iter.next() {
                None | Some(TokenTree::Token(Token { kind: token::Comma, .. }, _)) => {}
                _ => return None,
            }
        }
        Some(result)
    }

    fn name_value_from_tokens(iter: &mut TokenStreamIter<'_>) -> Option<MetaItemKind> {
        match iter.next() {
            Some(TokenTree::Delimited(.., Delimiter::Invisible(_), inner_tokens)) => {
                MetaItemKind::name_value_from_tokens(&mut inner_tokens.iter())
            }
            Some(TokenTree::Token(token, _)) => {
                MetaItemLit::from_token(token).map(MetaItemKind::NameValue)
            }
            _ => None,
        }
    }

    fn from_tokens(iter: &mut TokenStreamIter<'_>) -> Option<MetaItemKind> {
        match iter.peek() {
            Some(TokenTree::Delimited(.., Delimiter::Parenthesis, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                iter.next();
                MetaItemKind::list_from_tokens(inner_tokens).map(MetaItemKind::List)
            }
            Some(TokenTree::Delimited(..)) => None,
            Some(TokenTree::Token(Token { kind: token::Eq, .. }, _)) => {
                iter.next();
                MetaItemKind::name_value_from_tokens(iter)
            }
            _ => Some(MetaItemKind::Word),
        }
    }

    fn from_attr_args(args: &AttrArgs) -> Option<MetaItemKind> {
        match args {
            AttrArgs::Empty => Some(MetaItemKind::Word),
            AttrArgs::Delimited(DelimArgs { dspan: _, delim: Delimiter::Parenthesis, tokens }) => {
                MetaItemKind::list_from_tokens(tokens.clone()).map(MetaItemKind::List)
            }
            AttrArgs::Delimited(..) => None,
            AttrArgs::Eq { expr, .. } => match expr.kind {
                ExprKind::Lit(token_lit) => {
                    // Turn failures to `None`, we'll get parse errors elsewhere.
                    MetaItemLit::from_token_lit(token_lit, expr.span)
                        .ok()
                        .map(|lit| MetaItemKind::NameValue(lit))
                }
                _ => None,
            },
        }
    }
}

impl MetaItemInner {
    pub fn span(&self) -> Span {
        match self {
            MetaItemInner::MetaItem(item) => item.span,
            MetaItemInner::Lit(lit) => lit.span,
        }
    }

    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        self.meta_item().and_then(|meta_item| meta_item.ident())
    }

    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::empty).name
    }

    /// Returns `true` if this list item is a MetaItem with a name of `name`.
    pub fn has_name(&self, name: Symbol) -> bool {
        self.meta_item().is_some_and(|meta_item| meta_item.has_name(name))
    }

    /// Returns `true` if `self` is a `MetaItem` and the meta item is a word.
    pub fn is_word(&self) -> bool {
        self.meta_item().is_some_and(|meta_item| meta_item.is_word())
    }

    /// Gets a list of inner meta items from a list `MetaItem` type.
    pub fn meta_item_list(&self) -> Option<&[MetaItemInner]> {
        self.meta_item().and_then(|meta_item| meta_item.meta_item_list())
    }

    /// If it's a singleton list of the form `foo(lit)`, returns the `foo` and
    /// the `lit`.
    pub fn singleton_lit_list(&self) -> Option<(Symbol, &MetaItemLit)> {
        self.meta_item().and_then(|meta_item| {
            meta_item.meta_item_list().and_then(|meta_item_list| {
                if meta_item_list.len() == 1
                    && let Some(ident) = meta_item.ident()
                    && let Some(lit) = meta_item_list[0].lit()
                {
                    return Some((ident.name, lit));
                }
                None
            })
        })
    }

    /// See [`MetaItem::name_value_literal_span`].
    pub fn name_value_literal_span(&self) -> Option<Span> {
        self.meta_item()?.name_value_literal_span()
    }

    /// Gets the string value if `self` is a `MetaItem` and the `MetaItem` is a
    /// `MetaItemKind::NameValue` variant containing a string, otherwise `None`.
    pub fn value_str(&self) -> Option<Symbol> {
        self.meta_item().and_then(|meta_item| meta_item.value_str())
    }

    /// Returns the `MetaItemLit` if `self` is a `MetaItemInner::Literal`s.
    pub fn lit(&self) -> Option<&MetaItemLit> {
        match self {
            MetaItemInner::Lit(lit) => Some(lit),
            _ => None,
        }
    }

    /// Returns the bool if `self` is a boolean `MetaItemInner::Literal`.
    pub fn boolean_literal(&self) -> Option<bool> {
        match self {
            MetaItemInner::Lit(MetaItemLit { kind: LitKind::Bool(b), .. }) => Some(*b),
            _ => None,
        }
    }

    /// Returns the `MetaItem` if `self` is a `MetaItemInner::MetaItem` or if it's
    /// `MetaItemInner::Lit(MetaItemLit { kind: LitKind::Bool(_), .. })`.
    pub fn meta_item_or_bool(&self) -> Option<&MetaItemInner> {
        match self {
            MetaItemInner::MetaItem(_item) => Some(self),
            MetaItemInner::Lit(MetaItemLit { kind: LitKind::Bool(_), .. }) => Some(self),
            _ => None,
        }
    }

    /// Returns the `MetaItem` if `self` is a `MetaItemInner::MetaItem`.
    pub fn meta_item(&self) -> Option<&MetaItem> {
        match self {
            MetaItemInner::MetaItem(item) => Some(item),
            _ => None,
        }
    }

    /// Returns `true` if the variant is `MetaItem`.
    pub fn is_meta_item(&self) -> bool {
        self.meta_item().is_some()
    }

    fn from_tokens(iter: &mut TokenStreamIter<'_>) -> Option<MetaItemInner> {
        match iter.peek() {
            Some(TokenTree::Token(token, _)) if let Some(lit) = MetaItemLit::from_token(token) => {
                iter.next();
                return Some(MetaItemInner::Lit(lit));
            }
            Some(TokenTree::Delimited(.., Delimiter::Invisible(_), inner_tokens)) => {
                iter.next();
                return MetaItemInner::from_tokens(&mut inner_tokens.iter());
            }
            _ => {}
        }
        MetaItem::from_tokens(iter).map(MetaItemInner::MetaItem)
    }
}

pub fn mk_doc_comment(
    g: &AttrIdGenerator,
    comment_kind: CommentKind,
    style: AttrStyle,
    data: Symbol,
    span: Span,
) -> Attribute {
    Attribute { kind: AttrKind::DocComment(comment_kind, data), id: g.mk_attr_id(), style, span }
}

fn mk_attr(
    g: &AttrIdGenerator,
    style: AttrStyle,
    unsafety: Safety,
    path: Path,
    args: AttrArgs,
    span: Span,
) -> Attribute {
    mk_attr_from_item(g, AttrItem { unsafety, path, args, tokens: None }, None, style, span)
}

pub fn mk_attr_from_item(
    g: &AttrIdGenerator,
    item: AttrItem,
    tokens: Option<LazyAttrTokenStream>,
    style: AttrStyle,
    span: Span,
) -> Attribute {
    Attribute {
        kind: AttrKind::Normal(P(NormalAttr { item, tokens })),
        id: g.mk_attr_id(),
        style,
        span,
    }
}

pub fn mk_attr_word(
    g: &AttrIdGenerator,
    style: AttrStyle,
    unsafety: Safety,
    name: Symbol,
    span: Span,
) -> Attribute {
    let path = Path::from_ident(Ident::new(name, span));
    let args = AttrArgs::Empty;
    mk_attr(g, style, unsafety, path, args, span)
}

pub fn mk_attr_nested_word(
    g: &AttrIdGenerator,
    style: AttrStyle,
    unsafety: Safety,
    outer: Symbol,
    inner: Symbol,
    span: Span,
) -> Attribute {
    let inner_tokens = TokenStream::new(vec![TokenTree::Token(
        Token::from_ast_ident(Ident::new(inner, span)),
        Spacing::Alone,
    )]);
    let outer_ident = Ident::new(outer, span);
    let path = Path::from_ident(outer_ident);
    let attr_args = AttrArgs::Delimited(DelimArgs {
        dspan: DelimSpan::from_single(span),
        delim: Delimiter::Parenthesis,
        tokens: inner_tokens,
    });
    mk_attr(g, style, unsafety, path, attr_args, span)
}

pub fn mk_attr_name_value_str(
    g: &AttrIdGenerator,
    style: AttrStyle,
    unsafety: Safety,
    name: Symbol,
    val: Symbol,
    span: Span,
) -> Attribute {
    let lit = token::Lit::new(token::Str, escape_string_symbol(val), None);
    let expr = P(Expr {
        id: DUMMY_NODE_ID,
        kind: ExprKind::Lit(lit),
        span,
        attrs: AttrVec::new(),
        tokens: None,
    });
    let path = Path::from_ident(Ident::new(name, span));
    let args = AttrArgs::Eq { eq_span: span, expr };
    mk_attr(g, style, unsafety, path, args, span)
}

pub fn filter_by_name<A: AttributeExt>(attrs: &[A], name: Symbol) -> impl Iterator<Item = &A> {
    attrs.iter().filter(move |attr| attr.has_name(name))
}

pub fn find_by_name<A: AttributeExt>(attrs: &[A], name: Symbol) -> Option<&A> {
    filter_by_name(attrs, name).next()
}

pub fn first_attr_value_str_by_name(attrs: &[impl AttributeExt], name: Symbol) -> Option<Symbol> {
    find_by_name(attrs, name).and_then(|attr| attr.value_str())
}

pub fn contains_name(attrs: &[impl AttributeExt], name: Symbol) -> bool {
    find_by_name(attrs, name).is_some()
}

pub fn list_contains_name(items: &[MetaItemInner], name: Symbol) -> bool {
    items.iter().any(|item| item.has_name(name))
}

impl MetaItemLit {
    pub fn value_str(&self) -> Option<Symbol> {
        LitKind::from_token_lit(self.as_token_lit()).ok().and_then(|lit| lit.str())
    }
}

pub trait AttributeExt: Debug {
    fn id(&self) -> AttrId;

    /// For a single-segment attribute (i.e., `#[attr]` and not `#[path::atrr]`),
    /// return the name of the attribute, else return the empty identifier.
    fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::empty).name
    }

    /// Get the meta item list, `#[attr(meta item list)]`
    fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>>;

    /// Gets the value literal, as string, when using `#[attr = value]`
    fn value_str(&self) -> Option<Symbol>;

    /// Gets the span of the value literal, as string, when using `#[attr = value]`
    fn value_span(&self) -> Option<Span>;

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    fn ident(&self) -> Option<Ident>;

    /// Checks whether the path of this attribute matches the name.
    ///
    /// Matches one segment of the path to each element in `name`
    fn path_matches(&self, name: &[Symbol]) -> bool;

    /// Returns `true` if it is a sugared doc comment (`///` or `//!` for example).
    /// So `#[doc = "doc"]` (which is a doc comment) and `#[doc(...)]` (which is not
    /// a doc comment) will return `false`.
    fn is_doc_comment(&self) -> bool;

    #[inline]
    fn has_name(&self, name: Symbol) -> bool {
        self.ident().map(|x| x.name == name).unwrap_or(false)
    }

    /// get the span of the entire attribute
    fn span(&self) -> Span;

    fn is_word(&self) -> bool;

    fn path(&self) -> SmallVec<[Symbol; 1]> {
        self.ident_path()
            .map(|i| i.into_iter().map(|i| i.name).collect())
            .unwrap_or(smallvec![sym::doc])
    }

    /// Returns None for doc comments
    fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>>;

    /// Returns the documentation if this is a doc comment or a sugared doc comment.
    /// * `///doc` returns `Some("doc")`.
    /// * `#[doc = "doc"]` returns `Some("doc")`.
    /// * `#[doc(...)]` returns `None`.
    fn doc_str(&self) -> Option<Symbol>;

    fn is_proc_macro_attr(&self) -> bool {
        [sym::proc_macro, sym::proc_macro_attribute, sym::proc_macro_derive]
            .iter()
            .any(|kind| self.has_name(*kind))
    }

    /// Returns the documentation and its kind if this is a doc comment or a sugared doc comment.
    /// * `///doc` returns `Some(("doc", CommentKind::Line))`.
    /// * `/** doc */` returns `Some(("doc", CommentKind::Block))`.
    /// * `#[doc = "doc"]` returns `Some(("doc", CommentKind::Line))`.
    /// * `#[doc(...)]` returns `None`.
    fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)>;

    fn style(&self) -> AttrStyle;
}

// FIXME(fn_delegation): use function delegation instead of manually forwarding

impl Attribute {
    pub fn id(&self) -> AttrId {
        AttributeExt::id(self)
    }

    pub fn name_or_empty(&self) -> Symbol {
        AttributeExt::name_or_empty(self)
    }

    pub fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        AttributeExt::meta_item_list(self)
    }

    pub fn value_str(&self) -> Option<Symbol> {
        AttributeExt::value_str(self)
    }

    pub fn value_span(&self) -> Option<Span> {
        AttributeExt::value_span(self)
    }

    pub fn ident(&self) -> Option<Ident> {
        AttributeExt::ident(self)
    }

    pub fn path_matches(&self, name: &[Symbol]) -> bool {
        AttributeExt::path_matches(self, name)
    }

    pub fn is_doc_comment(&self) -> bool {
        AttributeExt::is_doc_comment(self)
    }

    #[inline]
    pub fn has_name(&self, name: Symbol) -> bool {
        AttributeExt::has_name(self, name)
    }

    pub fn span(&self) -> Span {
        AttributeExt::span(self)
    }

    pub fn is_word(&self) -> bool {
        AttributeExt::is_word(self)
    }

    pub fn path(&self) -> SmallVec<[Symbol; 1]> {
        AttributeExt::path(self)
    }

    pub fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>> {
        AttributeExt::ident_path(self)
    }

    pub fn doc_str(&self) -> Option<Symbol> {
        AttributeExt::doc_str(self)
    }

    pub fn is_proc_macro_attr(&self) -> bool {
        AttributeExt::is_proc_macro_attr(self)
    }

    pub fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        AttributeExt::doc_str_and_comment_kind(self)
    }

    pub fn style(&self) -> AttrStyle {
        AttributeExt::style(self)
    }
}
