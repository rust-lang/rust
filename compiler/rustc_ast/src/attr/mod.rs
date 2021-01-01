//! Functions dealing with attributes and meta items.

use crate::ast;
use crate::ast::{AttrId, AttrItem, AttrKind, AttrStyle, AttrVec, Attribute};
use crate::ast::{Expr, GenericParam, Item, Lit, LitKind, Local, Stmt, StmtKind};
use crate::ast::{MacArgs, MacDelimiter, MetaItem, MetaItemKind, NestedMetaItem};
use crate::ast::{Path, PathSegment};
use crate::mut_visit::visit_clobber;
use crate::ptr::P;
use crate::token::{self, CommentKind, Token};
use crate::tokenstream::{DelimSpan, LazyTokenStream, TokenStream, TokenTree, TreeAndSpacing};

use rustc_index::bit_set::GrowableBitSet;
use rustc_span::source_map::{BytePos, Spanned};
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

pub fn is_known_lint_tool(m_item: Ident) -> bool {
    [sym::clippy, sym::rustc].contains(&m_item.name)
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
        self.ident().unwrap_or_else(Ident::invalid).name
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
                if meta_item_list.len() == 1 {
                    if let Some(ident) = meta_item.ident() {
                        if let Some(lit) = meta_item_list[0].literal() {
                            return Some((ident.name, lit));
                        }
                    }
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

    /// Returns `true` if `self` is a `MetaItem` and the meta item is a `ValueString`.
    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }

    /// Returns `true` if `self` is a `MetaItem` and the meta item is a list.
    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }

    pub fn name_value_literal_span(&self) -> Option<Span> {
        self.meta_item()?.name_value_literal_span()
    }
}

impl Attribute {
    pub fn has_name(&self, name: Symbol) -> bool {
        match self.kind {
            AttrKind::Normal(ref item, _) => item.path == name,
            AttrKind::DocComment(..) => false,
        }
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        match self.kind {
            AttrKind::Normal(ref item, _) => {
                if item.path.segments.len() == 1 {
                    Some(item.path.segments[0].ident)
                } else {
                    None
                }
            }
            AttrKind::DocComment(..) => None,
        }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::invalid).name
    }

    pub fn value_str(&self) -> Option<Symbol> {
        match self.kind {
            AttrKind::Normal(ref item, _) => item.meta(self.span).and_then(|meta| meta.value_str()),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn meta_item_list(&self) -> Option<Vec<NestedMetaItem>> {
        match self.kind {
            AttrKind::Normal(ref item, _) => match item.meta(self.span) {
                Some(MetaItem { kind: MetaItemKind::List(list), .. }) => Some(list),
                _ => None,
            },
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn is_word(&self) -> bool {
        if let AttrKind::Normal(item, _) = &self.kind {
            matches!(item.args, MacArgs::Empty)
        } else {
            false
        }
    }

    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }

    /// Indicates if the attribute is a `ValueString`.
    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }

    /// This is used in case you want the value span instead of the whole attribute. Example:
    ///
    /// ```text
    /// #[doc(alias = "foo")]
    /// ```
    ///
    /// In here, it'll return a span for `"foo"`.
    pub fn name_value_literal_span(&self) -> Option<Span> {
        match self.kind {
            AttrKind::Normal(ref item, _) => {
                item.meta(self.span).and_then(|meta| meta.name_value_literal_span())
            }
            AttrKind::DocComment(..) => None,
        }
    }
}

impl MetaItem {
    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        if self.path.segments.len() == 1 { Some(self.path.segments[0].ident) } else { None }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or_else(Ident::invalid).name
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
        match self.kind {
            MetaItemKind::NameValue(ref v) => match v.kind {
                LitKind::Str(ref s, _) => Some(*s),
                _ => None,
            },
            _ => None,
        }
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

    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
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
}

impl Attribute {
    pub fn is_doc_comment(&self) -> bool {
        match self.kind {
            AttrKind::Normal(..) => false,
            AttrKind::DocComment(..) => true,
        }
    }

    pub fn doc_str(&self) -> Option<Symbol> {
        match self.kind {
            AttrKind::DocComment(.., data) => Some(data),
            AttrKind::Normal(ref item, _) if item.path == sym::doc => {
                item.meta(self.span).and_then(|meta| meta.value_str())
            }
            _ => None,
        }
    }

    pub fn get_normal_item(&self) -> &AttrItem {
        match self.kind {
            AttrKind::Normal(ref item, _) => item,
            AttrKind::DocComment(..) => panic!("unexpected doc comment"),
        }
    }

    pub fn unwrap_normal_item(self) -> AttrItem {
        match self.kind {
            AttrKind::Normal(item, _) => item,
            AttrKind::DocComment(..) => panic!("unexpected doc comment"),
        }
    }

    /// Extracts the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        match self.kind {
            AttrKind::Normal(ref item, _) => item.meta(self.span),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn tokens(&self) -> TokenStream {
        match self.kind {
            AttrKind::Normal(_, ref tokens) => tokens
                .as_ref()
                .unwrap_or_else(|| panic!("attribute is missing tokens: {:?}", self))
                .create_token_stream(),
            AttrKind::DocComment(comment_kind, data) => TokenStream::from(TokenTree::Token(
                Token::new(token::DocComment(comment_kind, self.style, data), self.span),
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

crate fn mk_attr_id() -> AttrId {
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
    Attribute { kind: AttrKind::Normal(item, tokens), id: mk_attr_id(), style, span }
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
    fn token_trees_and_spacings(&self) -> Vec<TreeAndSpacing> {
        let mut idents = vec![];
        let mut last_pos = BytePos(0_u32);
        for (i, segment) in self.path.segments.iter().enumerate() {
            let is_first = i == 0;
            if !is_first {
                let mod_sep_span =
                    Span::new(last_pos, segment.ident.span.lo(), segment.ident.span.ctxt());
                idents.push(TokenTree::token(token::ModSep, mod_sep_span).into());
            }
            idents.push(TokenTree::Token(Token::from_ast_ident(segment.ident)).into());
            last_pos = segment.ident.span.hi();
        }
        idents.extend(self.kind.token_trees_and_spacings(self.span));
        idents
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItem>
    where
        I: Iterator<Item = TokenTree>,
    {
        // FIXME: Share code with `parse_path`.
        let path = match tokens.next().map(TokenTree::uninterpolate) {
            Some(TokenTree::Token(Token {
                kind: kind @ (token::Ident(..) | token::ModSep),
                span,
            })) => 'arm: {
                let mut segments = if let token::Ident(name, _) = kind {
                    if let Some(TokenTree::Token(Token { kind: token::ModSep, .. })) = tokens.peek()
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
                    if let Some(TokenTree::Token(Token { kind: token::Ident(name, _), span })) =
                        tokens.next().map(TokenTree::uninterpolate)
                    {
                        segments.push(PathSegment::from_ident(Ident::new(name, span)));
                    } else {
                        return None;
                    }
                    if let Some(TokenTree::Token(Token { kind: token::ModSep, .. })) = tokens.peek()
                    {
                        tokens.next();
                    } else {
                        break;
                    }
                }
                let span = span.with_hi(segments.last().unwrap().ident.span.hi());
                Path { span, segments, tokens: None }
            }
            Some(TokenTree::Token(Token { kind: token::Interpolated(nt), .. })) => match *nt {
                token::Nonterminal::NtMeta(ref item) => return item.meta(item.path.span),
                token::Nonterminal::NtPath(ref path) => path.clone(),
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
    pub fn mac_args(&self, span: Span) -> MacArgs {
        match self {
            MetaItemKind::Word => MacArgs::Empty,
            MetaItemKind::NameValue(lit) => MacArgs::Eq(span, lit.token_tree().into()),
            MetaItemKind::List(list) => {
                let mut tts = Vec::new();
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        tts.push(TokenTree::token(token::Comma, span).into());
                    }
                    tts.extend(item.token_trees_and_spacings())
                }
                MacArgs::Delimited(
                    DelimSpan::from_single(span),
                    MacDelimiter::Parenthesis,
                    TokenStream::new(tts),
                )
            }
        }
    }

    fn token_trees_and_spacings(&self, span: Span) -> Vec<TreeAndSpacing> {
        match *self {
            MetaItemKind::Word => vec![],
            MetaItemKind::NameValue(ref lit) => {
                vec![TokenTree::token(token::Eq, span).into(), lit.token_tree().into()]
            }
            MetaItemKind::List(ref list) => {
                let mut tokens = Vec::new();
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        tokens.push(TokenTree::token(token::Comma, span).into());
                    }
                    tokens.extend(item.token_trees_and_spacings())
                }
                vec![
                    TokenTree::Delimited(
                        DelimSpan::from_single(span),
                        token::Paren,
                        TokenStream::new(tokens),
                    )
                    .into(),
                ]
            }
        }
    }

    fn list_from_tokens(tokens: TokenStream) -> Option<MetaItemKind> {
        let mut tokens = tokens.into_trees().peekable();
        let mut result = Vec::new();
        while let Some(..) = tokens.peek() {
            let item = NestedMetaItem::from_tokens(&mut tokens)?;
            result.push(item);
            match tokens.next() {
                None | Some(TokenTree::Token(Token { kind: token::Comma, .. })) => {}
                _ => return None,
            }
        }
        Some(MetaItemKind::List(result))
    }

    fn name_value_from_tokens(
        tokens: &mut impl Iterator<Item = TokenTree>,
    ) -> Option<MetaItemKind> {
        match tokens.next() {
            Some(TokenTree::Delimited(_, token::NoDelim, inner_tokens)) => {
                MetaItemKind::name_value_from_tokens(&mut inner_tokens.trees())
            }
            Some(TokenTree::Token(token)) => {
                Lit::from_token(&token).ok().map(MetaItemKind::NameValue)
            }
            _ => None,
        }
    }

    fn from_mac_args(args: &MacArgs) -> Option<MetaItemKind> {
        match args {
            MacArgs::Delimited(_, MacDelimiter::Parenthesis, tokens) => {
                MetaItemKind::list_from_tokens(tokens.clone())
            }
            MacArgs::Delimited(..) => None,
            MacArgs::Eq(_, tokens) => {
                assert!(tokens.len() == 1);
                MetaItemKind::name_value_from_tokens(&mut tokens.trees())
            }
            MacArgs::Empty => Some(MetaItemKind::Word),
        }
    }

    fn from_tokens(
        tokens: &mut iter::Peekable<impl Iterator<Item = TokenTree>>,
    ) -> Option<MetaItemKind> {
        match tokens.peek() {
            Some(TokenTree::Delimited(_, token::Paren, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                tokens.next();
                MetaItemKind::list_from_tokens(inner_tokens)
            }
            Some(TokenTree::Delimited(..)) => None,
            Some(TokenTree::Token(Token { kind: token::Eq, .. })) => {
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

    fn token_trees_and_spacings(&self) -> Vec<TreeAndSpacing> {
        match *self {
            NestedMetaItem::MetaItem(ref item) => item.token_trees_and_spacings(),
            NestedMetaItem::Literal(ref lit) => vec![lit.token_tree().into()],
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<NestedMetaItem>
    where
        I: Iterator<Item = TokenTree>,
    {
        match tokens.peek() {
            Some(TokenTree::Token(token)) => {
                if let Ok(lit) = Lit::from_token(token) {
                    tokens.next();
                    return Some(NestedMetaItem::Literal(lit));
                }
            }
            Some(TokenTree::Delimited(_, token::NoDelim, inner_tokens)) => {
                let inner_tokens = inner_tokens.clone();
                tokens.next();
                return NestedMetaItem::from_tokens(&mut inner_tokens.into_trees().peekable());
            }
            _ => {}
        }
        MetaItem::from_tokens(tokens).map(NestedMetaItem::MetaItem)
    }
}

pub trait HasAttrs: Sized {
    fn attrs(&self) -> &[Attribute];
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>));
}

impl<T: HasAttrs> HasAttrs for Spanned<T> {
    fn attrs(&self) -> &[Attribute] {
        self.node.attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        self.node.visit_attrs(f);
    }
}

impl HasAttrs for Vec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        self
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        f(self)
    }
}

impl HasAttrs for AttrVec {
    fn attrs(&self) -> &[Attribute] {
        self
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        visit_clobber(self, |this| {
            let mut vec = this.into();
            f(&mut vec);
            vec.into()
        });
    }
}

impl<T: HasAttrs + 'static> HasAttrs for P<T> {
    fn attrs(&self) -> &[Attribute] {
        (**self).attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        (**self).visit_attrs(f);
    }
}

impl HasAttrs for StmtKind {
    fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtKind::Local(ref local) => local.attrs(),
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => expr.attrs(),
            StmtKind::Item(ref item) => item.attrs(),
            StmtKind::Empty => &[],
            StmtKind::MacCall(ref mac) => mac.attrs.attrs(),
        }
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        match self {
            StmtKind::Local(local) => local.visit_attrs(f),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.visit_attrs(f),
            StmtKind::Item(item) => item.visit_attrs(f),
            StmtKind::Empty => {}
            StmtKind::MacCall(mac) => {
                mac.attrs.visit_attrs(f);
            }
        }
    }
}

impl HasAttrs for Stmt {
    fn attrs(&self) -> &[ast::Attribute] {
        self.kind.attrs()
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        self.kind.visit_attrs(f);
    }
}

macro_rules! derive_has_attrs {
    ($($ty:path),*) => { $(
        impl HasAttrs for $ty {
            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                self.attrs.visit_attrs(f);
            }
        }
    )* }
}

derive_has_attrs! {
    Item, Expr, Local, ast::AssocItem, ast::ForeignItem, ast::StructField, ast::Arm,
    ast::Field, ast::FieldPat, ast::Variant, ast::Param, GenericParam
}
