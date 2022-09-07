//! Functions dealing with attributes and meta items.

use crate::ast;
use crate::ast::{AttrId, AttrItem, AttrKind, AttrStyle, Attribute};
use crate::ast::{Lit, LitKind};
use crate::ast::{MacArgs, MacArgsEq, MacDelimiter, MetaItem, MetaItemKind, NestedMetaItem};
use crate::ast::{Path, PathSegment};
use crate::ptr::P;
use crate::token::{self, CommentKind, Delimiter, Token};
use crate::tokenstream::{AttrAnnotatedTokenStream, AttrAnnotatedTokenTree};
use crate::tokenstream::{DelimSpan, Spacing, TokenTree};
use crate::tokenstream::{LazyTokenStream, TokenStream};
use crate::util::comments;

use rustc_index::bit_set::GrowableBitSet;
use rustc_span::source_map::BytePos;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;

use std::iter;

pub struct MarkedAttrs(GrowableBitSet<AttrId>);

impl MarkedAttrs {
    // We have no idea how many attributes there will be, so just
    // initiate the vectors with 0 bits. We'll grow them as necessary.
    pub fn new() -> Self {
        MarkedAttrs(GrowableBitSet::new_empty())
    }

    pub fn mark(&mut self, attr: &Attribute) {
        self.0.insert(attr.id);
    }

    pub fn is_marked(&self, attr: &Attribute) -> bool {
        self.0.contains(attr.id)
    }
}

impl NestedMetaItem {
    /// Returns the `MetaItem` if `self` is a `NestedMetaItem::MetaItem`.
    pub fn meta_item(&self) -> Option<&MetaItem> {
        match *self {
            NestedMetaItem::MetaItem(ref item) => Some(item),
            _ => None,
        }
    }

    /// Returns the `Lit` if `self` is a `NestedMetaItem::Literal`s.
    pub fn literal(&self) -> Option<&Lit> {
        match *self {
            NestedMetaItem::Literal(ref lit) => Some(lit),
            _ => None,
        }
    }

    /// Returns `true` if this list item is a MetaItem with a name of `name`.
    pub fn has_name(&self, name: Symbol) -> bool {
        self.meta_item().map_or(false, |meta_item| meta_item.has_name(name))
    }

    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        self.meta_item().and_then(|meta_item| meta_item.ident())
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::empty).name
    }

    /// Gets the string value if `self` is a `MetaItem` and the `MetaItem` is a
    /// `MetaItemKind::NameValue` variant containing a string, otherwise `None`.
    pub fn value_str(&self) -> Option<Symbol> {
        self.meta_item().and_then(|meta_item| meta_item.value_str())
    }

    /// Returns a name and single literal value tuple of the `MetaItem`.
    pub fn name_value_literal(&self) -> Option<(Symbol, &Lit)> {
        self.meta_item().and_then(|meta_item| {
            meta_item.meta_item_list().and_then(|meta_item_list| {
                if meta_item_list.len() == 1
                    && let Some(ident) = meta_item.ident()
                    && let Some(lit) = meta_item_list[0].literal()
                {
                    return Some((ident.name, lit));
                }
                None
            })
        })
    }

    /// Gets a list of inner meta items from a list `MetaItem` type.
    pub fn meta_item_list(&self) -> Option<&[NestedMetaItem]> {
        self.meta_item().and_then(|meta_item| meta_item.meta_item_list())
    }

    /// Returns `true` if the variant is `MetaItem`.
    pub fn is_meta_item(&self) -> bool {
        self.meta_item().is_some()
    }

    /// Returns `true` if `self` is a `MetaItem` and the meta item is a word.
    pub fn is_word(&self) -> bool {
        self.meta_item().map_or(false, |meta_item| meta_item.is_word())
    }

    /// See [`MetaItem::name_value_literal_span`].
    pub fn name_value_literal_span(&self) -> Option<Span> {
        self.meta_item()?.name_value_literal_span()
    }
}

impl Attribute {
    #[inline]
    pub fn has_name(&self, name: Symbol) -> bool {
        match self.kind {
            AttrKind::Normal(ref normal) => normal.item.path == name,
            AttrKind::DocComment(..) => false,
        }
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        match self.kind {
            AttrKind::Normal(ref normal) => {
                if normal.item.path.segments.len() == 1 {
                    Some(normal.item.path.segments[0].ident)
                } else {
                    None
                }
            }
            AttrKind::DocComment(..) => None,
        }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::empty).name
    }

    pub fn value_str(&self) -> Option<Symbol> {
        match self.kind {
            AttrKind::Normal(ref normal) => {
                normal.item.meta_kind().and_then(|kind| kind.value_str())
            }
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn meta_item_list(&self) -> Option<Vec<NestedMetaItem>> {
        match self.kind {
            AttrKind::Normal(ref normal) => match normal.item.meta_kind() {
                Some(MetaItemKind::List(list)) => Some(list),
                _ => None,
            },
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn is_word(&self) -> bool {
        if let AttrKind::Normal(normal) = &self.kind {
            matches!(normal.item.args, MacArgs::Empty)
        } else {
            false
        }
    }
}

impl MetaItem {
    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        if self.path.segments.len() == 1 { Some(self.path.segments[0].ident) } else { None }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::empty).name
    }

    // Example:
    //     #[attribute(name = "value")]
    //                 ^^^^^^^^^^^^^^
    pub fn name_value_literal(&self) -> Option<&Lit> {
        match &self.kind {
            MetaItemKind::NameValue(v) => Some(v),
            _ => None,
        }
    }

    pub fn value_str(&self) -> Option<Symbol> {
        self.kind.value_str()
    }

    pub fn meta_item_list(&self) -> Option<&[NestedMetaItem]> {
        match self.kind {
            MetaItemKind::List(ref l) => Some(&l[..]),
            _ => None,
        }
    }

    pub fn is_word(&self) -> bool {
        matches!(self.kind, MetaItemKind::Word)
    }

    pub fn has_name(&self, name: Symbol) -> bool {
        self.path == name
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
}

impl AttrItem {
    pub fn span(&self) -> Span {
        self.args.span().map_or(self.path.span, |args_span| self.path.span.to(args_span))
    }

    pub fn meta(&self, span: Span) -> Option<MetaItem> {
        Some(MetaItem {
            path: self.path.clone(),
            kind: MetaItemKind::from_mac_args(&self.args)?,
            span,
        })
    }

    pub fn meta_kind(&self) -> Option<MetaItemKind> {
        MetaItemKind::from_mac_args(&self.args)
    }
}

impl Attribute {
    /// Returns `true` if it is a sugared doc comment (`///` or `//!` for example).
    /// So `#[doc = "doc"]` will return `false`.
    pub fn is_doc_comment(&self) -> bool {
        match self.kind {
            AttrKind::Normal(..) => false,
            AttrKind::DocComment(..) => true,
        }
    }

    pub fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        match self.kind {
            AttrKind::DocComment(kind, data) => Some((data, kind)),
            AttrKind::Normal(ref normal) if normal.item.path == sym::doc => normal
                .item
                .meta_kind()
                .and_then(|kind| kind.value_str())
                .map(|data| (data, CommentKind::Line)),
            _ => None,
        }
    }

    pub fn doc_str(&self) -> Option<Symbol> {
        match self.kind {
            AttrKind::DocComment(.., data) => Some(data),
            AttrKind::Normal(ref normal) if normal.item.path == sym::doc => {
                normal.item.meta_kind().and_then(|kind| kind.value_str())
            }
            _ => None,
        }
    }

    pub fn may_have_doc_links(&self) -> bool {
        self.doc_str().map_or(false, |s| comments::may_have_doc_links(s.as_str()))
    }

    pub fn get_normal_item(&self) -> &AttrItem {
        match self.kind {
            AttrKind::Normal(ref normal) => &normal.item,
            AttrKind::DocComment(..) => panic!("unexpected doc comment"),
        }
    }

    pub fn unwrap_normal_item(self) -> AttrItem {
        match self.kind {
            AttrKind::Normal(normal) => normal.into_inner().item,
            AttrKind::DocComment(..) => panic!("unexpected doc comment"),
        }
    }

    /// Extracts the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        match self.kind {
            AttrKind::Normal(ref normal) => normal.item.meta(self.span),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn meta_kind(&self) -> Option<MetaItemKind> {
        match self.kind {
            AttrKind::Normal(ref normal) => normal.item.meta_kind(),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn tokens(&self) -> AttrAnnotatedTokenStream {
        match self.kind {
            AttrKind::Normal(ref normal) => normal
                .tokens
                .as_ref()
                .unwrap_or_else(|| panic!("attribute is missing tokens: {:?}", self))
                .create_token_stream(),
            AttrKind::DocComment(comment_kind, data) => AttrAnnotatedTokenStream::from((
                AttrAnnotatedTokenTree::Token(Token::new(
                    token::DocComment(comment_kind, self.style, data),
                    self.span,
                )),
                Spacing::Alone,
            )),
        }
    }
}

/* Constructors */

pub fn mk_name_value_item_str(ident: Ident, str: Symbol, str_span: Span) -> MetaItem {
    let lit_kind = LitKind::Str(str, ast::StrStyle::Cooked);
    mk_name_value_item(ident, lit_kind, str_span)
}

pub fn mk_name_value_item(ident: Ident, lit_kind: LitKind, lit_span: Span) -> MetaItem {
    let lit = Lit::from_lit_kind(lit_kind, lit_span);
    let span = ident.span.to(lit_span);
    MetaItem { path: Path::from_ident(ident), span, kind: MetaItemKind::NameValue(lit) }
}

pub fn mk_list_item(ident: Ident, items: Vec<NestedMetaItem>) -> MetaItem {
    MetaItem { path: Path::from_ident(ident), span: ident.span, kind: MetaItemKind::List(items) }
}

pub fn mk_word_item(ident: Ident) -> MetaItem {
    MetaItem { path: Path::from_ident(ident), span: ident.span, kind: MetaItemKind::Word }
}

pub fn mk_nested_word_item(ident: Ident) -> NestedMetaItem {
    NestedMetaItem::MetaItem(mk_word_item(ident))
}

pub(crate) fn mk_attr_id() -> AttrId {
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;

    static NEXT_ATTR_ID: AtomicU32 = AtomicU32::new(0);

    let id = NEXT_ATTR_ID.fetch_add(1, Ordering::SeqCst);
    assert!(id != u32::MAX);
    AttrId::from_u32(id)
}

pub fn mk_attr(style: AttrStyle, path: Path, args: MacArgs, span: Span) -> Attribute {
    mk_attr_from_item(AttrItem { path, args, tokens: None }, None, style, span)
}

pub fn mk_attr_from_item(
    item: AttrItem,
    tokens: Option<LazyTokenStream>,
    style: AttrStyle,
    span: Span,
) -> Attribute {
    Attribute {
        kind: AttrKind::Normal(P(ast::NormalAttr { item, tokens })),
        id: mk_attr_id(),
        style,
        span,
    }
}

/// Returns an inner attribute with the given value and span.
pub fn mk_attr_inner(item: MetaItem) -> Attribute {
    mk_attr(AttrStyle::Inner, item.path, item.kind.mac_args(item.span), item.span)
}

/// Returns an outer attribute with the given value and span.
pub fn mk_attr_outer(item: MetaItem) -> Attribute {
    mk_attr(AttrStyle::Outer, item.path, item.kind.mac_args(item.span), item.span)
}

pub fn mk_doc_comment(
    comment_kind: CommentKind,
    style: AttrStyle,
    data: Symbol,
    span: Span,
) -> Attribute {
    Attribute { kind: AttrKind::DocComment(comment_kind, data), id: mk_attr_id(), style, span }
}

pub fn list_contains_name(items: &[NestedMetaItem], name: Symbol) -> bool {
    items.iter().any(|item| item.has_name(name))
}

impl MetaItem {
    fn token_trees(&self) -> Vec<TokenTree> {
        let mut idents = vec![];
        let mut last_pos = BytePos(0_u32);
        for (i, segment) in self.path.segments.iter().enumerate() {
            let is_first = i == 0;
            if !is_first {
                let mod_sep_span =
                    Span::new(last_pos, segment.ident.span.lo(), segment.ident.span.ctxt(), None);
                idents.push(TokenTree::token_alone(token::ModSep, mod_sep_span));
            }
            idents.push(TokenTree::Token(Token::from_ast_ident(segment.ident), Spacing::Alone));
            last_pos = segment.ident.span.hi();
        }
        idents.extend(self.kind.token_trees(self.span));
        idents
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItem>
    where
        I: Iterator<Item = TokenTree>,
    {
        // FIXME: Share code with `parse_path`.
        let path = match tokens.next().map(TokenTree::uninterpolate) {
            Some(TokenTree::Token(
                Token { kind: kind @ (token::Ident(..) | token::ModSep), span },
                _,
            )) => 'arm: {
                let mut segments = if let token::Ident(name, _) = kind {
                    if let Some(TokenTree::Token(Token { kind: token::ModSep, .. }, _)) =
                        tokens.peek()
                    {
                        tokens.next();
                        vec![PathSegment::from_ident(Ident::new(name, span))]
                    } else {
                        break 'arm Path::from_ident(Ident::new(name, span));
                    }
                } else {
                    vec![PathSegment::path_root(span)]
                };
                loop {
                    if let Some(TokenTree::Token(Token { kind: token::Ident(name, _), span }, _)) =
                        tokens.next().map(TokenTree::uninterpolate)
                    {
                        segments.push(PathSegment::from_ident(Ident::new(name, span)));
                    } else {
                        return None;
                    }
                    if let Some(TokenTree::Token(Token { kind: token::ModSep, .. }, _)) =
                        tokens.peek()
                    {
                        tokens.next();
                    } else {
                        break;
                    }
                }
                let span = span.with_hi(segments.last().unwrap().ident.span.hi());
                Path { span, segments, tokens: None }
            }
            Some(TokenTree::Token(Token { kind: token::Interpolated(nt), .. }, _)) => match *nt {
                token::Nonterminal::NtMeta(ref item) => return item.meta(item.path.span),
                token::Nonterminal::NtPath(ref path) => (**path).clone(),
                _ => return None,
            },
            _ => return None,
        };
        let list_closing_paren_pos = tokens.peek().map(|tt| tt.span().hi());
        let kind = MetaItemKind::from_tokens(tokens)?;
        let hi = match kind {
            MetaItemKind::NameValue(ref lit) => lit.span.hi(),
            MetaItemKind::List(..) => list_closing_paren_pos.unwrap_or(path.span.hi()),
            _ => path.span.hi(),
        };
        let span = path.span.with_hi(hi);
        Some(MetaItem { path, kind, span })
    }
}

impl MetaItemKind {
    pub fn value_str(&self) -> Option<Symbol> {
        match self {
            MetaItemKind::NameValue(ref v) => match v.kind {
                LitKind::Str(ref s, _) => Some(*s),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn mac_args(&self, span: Span) -> MacArgs {
        match self {
            MetaItemKind::Word => MacArgs::Empty,
            MetaItemKind::NameValue(lit) => {
                let expr = P(ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    kind: ast::ExprKind::Lit(lit.clone()),
                    span: lit.span,
                    attrs: ast::AttrVec::new(),
                    tokens: None,
                });
                MacArgs::Eq(span, MacArgsEq::Ast(expr))
            }
            MetaItemKind::List(list) => {
                let mut tts = Vec::new();
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        tts.push(TokenTree::token_alone(token::Comma, span));
                    }
                    tts.extend(item.token_trees())
                }
                MacArgs::Delimited(
                    DelimSpan::from_single(span),
                    MacDelimiter::Parenthesis,
                    TokenStream::new(tts),
                )
            }
        }
    }

    fn token_trees(&self, span: Span) -> Vec<TokenTree> {
        match *self {
            MetaItemKind::Word => vec![],
            MetaItemKind::NameValue(ref lit) => {
                vec![
                    TokenTree::token_alone(token::Eq, span),
                    TokenTree::Token(lit.to_token(), Spacing::Alone),
                ]
            }
            MetaItemKind::List(ref list) => {
                let mut tokens = Vec::new();
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        tokens.push(TokenTree::token_alone(token::Comma, span));
                    }
                    tokens.extend(item.token_trees())
                }
                vec![TokenTree::Delimited(
                    DelimSpan::from_single(span),
                    Delimiter::Parenthesis,
                    TokenStream::new(tokens),
                )]
            }
        }
    }

    fn list_from_tokens(tokens: TokenStream) -> Option<MetaItemKind> {
        let mut tokens = tokens.into_trees().peekable();
        let mut result = Vec::new();
        while tokens.peek().is_some() {
            let item = NestedMetaItem::from_tokens(&mut tokens)?;
            result.push(item);
            match tokens.next() {
                None | Some(TokenTree::Token(Token { kind: token::Comma, .. }, _)) => {}
                _ => return None,
            }
        }
        Some(MetaItemKind::List(result))
    }

    fn name_value_from_tokens(
        tokens: &mut impl Iterator<Item = TokenTree>,
    ) -> Option<MetaItemKind> {
        match tokens.next() {
            Some(TokenTree::Delimited(_, Delimiter::Invisible, inner_tokens)) => {
                MetaItemKind::name_value_from_tokens(&mut inner_tokens.into_trees())
            }
            Some(TokenTree::Token(token, _)) => {
                Lit::from_token(&token).ok().map(MetaItemKind::NameValue)
            }
            _ => None,
        }
    }

    fn from_mac_args(args: &MacArgs) -> Option<MetaItemKind> {
        match args {
            MacArgs::Empty => Some(MetaItemKind::Word),
            MacArgs::Delimited(_, MacDelimiter::Parenthesis, tokens) => {
                MetaItemKind::list_from_tokens(tokens.clone())
            }
            MacArgs::Delimited(..) => None,
            MacArgs::Eq(_, MacArgsEq::Ast(expr)) => match &expr.kind {
                ast::ExprKind::Lit(lit) => Some(MetaItemKind::NameValue(lit.clone())),
                _ => None,
            },
            MacArgs::Eq(_, MacArgsEq::Hir(lit)) => Some(MetaItemKind::NameValue(lit.clone())),
        }
    }

    fn from_tokens(
        tokens: &mut iter::Peekable<impl Iterator<Item = TokenTree>>,
    ) -> Option<MetaItemKind> {
        match tokens.peek() {
            Some(TokenTree::Delimited(_, Delimiter::Parenthesis, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                tokens.next();
                MetaItemKind::list_from_tokens(inner_tokens)
            }
            Some(TokenTree::Delimited(..)) => None,
            Some(TokenTree::Token(Token { kind: token::Eq, .. }, _)) => {
                tokens.next();
                MetaItemKind::name_value_from_tokens(tokens)
            }
            _ => Some(MetaItemKind::Word),
        }
    }
}

impl NestedMetaItem {
    pub fn span(&self) -> Span {
        match *self {
            NestedMetaItem::MetaItem(ref item) => item.span,
            NestedMetaItem::Literal(ref lit) => lit.span,
        }
    }

    fn token_trees(&self) -> Vec<TokenTree> {
        match *self {
            NestedMetaItem::MetaItem(ref item) => item.token_trees(),
            NestedMetaItem::Literal(ref lit) => {
                vec![TokenTree::Token(lit.to_token(), Spacing::Alone)]
            }
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<NestedMetaItem>
    where
        I: Iterator<Item = TokenTree>,
    {
        match tokens.peek() {
            Some(TokenTree::Token(token, _))
                if let Ok(lit) = Lit::from_token(token) =>
            {
                tokens.next();
                return Some(NestedMetaItem::Literal(lit));
            }
            Some(TokenTree::Delimited(_, Delimiter::Invisible, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                tokens.next();
                return NestedMetaItem::from_tokens(&mut inner_tokens.into_trees().peekable());
            }
            _ => {}
        }
        MetaItem::from_tokens(tokens).map(NestedMetaItem::MetaItem)
    }
}
