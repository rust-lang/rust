// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functions dealing with attributes and meta items

mod builtin;

pub use self::builtin::{
    cfg_matches, contains_feature_attr, eval_condition, find_crate_name, find_deprecation,
    find_repr_attrs, find_stability, find_unwind_attr, Deprecation, InlineAttr, IntType, ReprAttr,
    RustcConstUnstable, RustcDeprecation, Stability, StabilityLevel, UnwindAttr,
};
pub use self::IntType::*;
pub use self::ReprAttr::*;
pub use self::StabilityLevel::*;

use ast;
use ast::{AttrId, Attribute, AttrStyle, Name, Ident, Path, PathSegment};
use ast::{MetaItem, MetaItemKind, NestedMetaItem, NestedMetaItemKind};
use ast::{Lit, LitKind, Expr, ExprKind, Item, Local, Stmt, StmtKind, GenericParam};
use source_map::{BytePos, Spanned, respan, dummy_spanned};
use syntax_pos::{FileName, Span};
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::parser::Parser;
use parse::{self, ParseSess, PResult};
use parse::token::{self, Token};
use ptr::P;
use symbol::Symbol;
use ThinVec;
use tokenstream::{TokenStream, TokenTree, Delimited};
use GLOBALS;

use std::iter;

pub fn mark_used(attr: &Attribute) {
    debug!("Marking {:?} as used.", attr);
    GLOBALS.with(|globals| {
        globals.used_attrs.lock().insert(attr.id);
    });
}

pub fn is_used(attr: &Attribute) -> bool {
    GLOBALS.with(|globals| {
        globals.used_attrs.lock().contains(attr.id)
    })
}

pub fn mark_known(attr: &Attribute) {
    debug!("Marking {:?} as known.", attr);
    GLOBALS.with(|globals| {
        globals.known_attrs.lock().insert(attr.id);
    });
}

pub fn is_known(attr: &Attribute) -> bool {
    GLOBALS.with(|globals| {
        globals.known_attrs.lock().contains(attr.id)
    })
}

pub fn is_known_lint_tool(m_item: Ident) -> bool {
    ["clippy"].contains(&m_item.as_str().as_ref())
}

impl NestedMetaItem {
    /// Returns the MetaItem if self is a NestedMetaItemKind::MetaItem.
    pub fn meta_item(&self) -> Option<&MetaItem> {
        match self.node {
            NestedMetaItemKind::MetaItem(ref item) => Some(item),
            _ => None
        }
    }

    /// Returns the Lit if self is a NestedMetaItemKind::Literal.
    pub fn literal(&self) -> Option<&Lit> {
        match self.node {
            NestedMetaItemKind::Literal(ref lit) => Some(lit),
            _ => None
        }
    }

    /// Returns the Span for `self`.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Returns true if this list item is a MetaItem with a name of `name`.
    pub fn check_name(&self, name: &str) -> bool {
        self.meta_item().map_or(false, |meta_item| meta_item.check_name(name))
    }

    /// Returns the name of the meta item, e.g. `foo` in `#[foo]`,
    /// `#[foo="bar"]` and `#[foo(bar)]`, if self is a MetaItem
    pub fn name(&self) -> Option<Name> {
        self.meta_item().and_then(|meta_item| Some(meta_item.name()))
    }

    /// Gets the string value if self is a MetaItem and the MetaItem is a
    /// MetaItemKind::NameValue variant containing a string, otherwise None.
    pub fn value_str(&self) -> Option<Symbol> {
        self.meta_item().and_then(|meta_item| meta_item.value_str())
    }

    /// Returns a name and single literal value tuple of the MetaItem.
    pub fn name_value_literal(&self) -> Option<(Name, &Lit)> {
        self.meta_item().and_then(
            |meta_item| meta_item.meta_item_list().and_then(
                |meta_item_list| {
                    if meta_item_list.len() == 1 {
                        let nested_item = &meta_item_list[0];
                        if nested_item.is_literal() {
                            Some((meta_item.name(), nested_item.literal().unwrap()))
                        } else {
                            None
                        }
                    }
                    else {
                        None
                    }}))
    }

    /// Returns a MetaItem if self is a MetaItem with Kind Word.
    pub fn word(&self) -> Option<&MetaItem> {
        self.meta_item().and_then(|meta_item| if meta_item.is_word() {
            Some(meta_item)
        } else {
            None
        })
    }

    /// Gets a list of inner meta items from a list MetaItem type.
    pub fn meta_item_list(&self) -> Option<&[NestedMetaItem]> {
        self.meta_item().and_then(|meta_item| meta_item.meta_item_list())
    }

    /// Returns `true` if the variant is MetaItem.
    pub fn is_meta_item(&self) -> bool {
        self.meta_item().is_some()
    }

    /// Returns `true` if the variant is Literal.
    pub fn is_literal(&self) -> bool {
        self.literal().is_some()
    }

    /// Returns `true` if self is a MetaItem and the meta item is a word.
    pub fn is_word(&self) -> bool {
        self.word().is_some()
    }

    /// Returns `true` if self is a MetaItem and the meta item is a ValueString.
    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }

    /// Returns `true` if self is a MetaItem and the meta item is a list.
    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }
}

fn name_from_path(path: &Path) -> Name {
    path.segments.last().expect("empty path in attribute").ident.name
}

impl Attribute {
    pub fn check_name(&self, name: &str) -> bool {
        let matches = self.path == name;
        if matches {
            mark_used(self);
        }
        matches
    }

    /// Returns the **last** segment of the name of this attribute.
    /// E.g. `foo` for `#[foo]`, `skip` for `#[rustfmt::skip]`.
    pub fn name(&self) -> Name {
        name_from_path(&self.path)
    }

    pub fn value_str(&self) -> Option<Symbol> {
        self.meta().and_then(|meta| meta.value_str())
    }

    pub fn meta_item_list(&self) -> Option<Vec<NestedMetaItem>> {
        match self.meta() {
            Some(MetaItem { node: MetaItemKind::List(list), .. }) => Some(list),
            _ => None
        }
    }

    pub fn is_word(&self) -> bool {
        self.path.segments.len() == 1 && self.tokens.is_empty()
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }

    /// Indicates if the attribute is a Value String.
    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }
}

impl MetaItem {
    pub fn name(&self) -> Name {
        name_from_path(&self.ident)
    }

    pub fn value_str(&self) -> Option<Symbol> {
        match self.node {
            MetaItemKind::NameValue(ref v) => {
                match v.node {
                    LitKind::Str(ref s, _) => Some(*s),
                    _ => None,
                }
            },
            _ => None
        }
    }

    pub fn meta_item_list(&self) -> Option<&[NestedMetaItem]> {
        match self.node {
            MetaItemKind::List(ref l) => Some(&l[..]),
            _ => None
        }
    }

    pub fn is_word(&self) -> bool {
        match self.node {
            MetaItemKind::Word => true,
            _ => false,
        }
    }

    pub fn span(&self) -> Span { self.span }

    pub fn check_name(&self, name: &str) -> bool {
        self.name() == name
    }

    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }

    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }

    pub fn is_scoped(&self) -> Option<Ident> {
        if self.ident.segments.len() > 1 {
            Some(self.ident.segments[0].ident)
        } else {
            None
        }
    }
}

impl Attribute {
    /// Extract the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        let mut tokens = self.tokens.trees().peekable();
        Some(MetaItem {
            ident: self.path.clone(),
            node: if let Some(node) = MetaItemKind::from_tokens(&mut tokens) {
                if tokens.peek().is_some() {
                    return None;
                }
                node
            } else {
                return None;
            },
            span: self.span,
        })
    }

    pub fn parse<'a, T, F>(&self, sess: &'a ParseSess, mut f: F) -> PResult<'a, T>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        let mut parser = Parser::new(sess, self.tokens.clone(), None, false, false);
        let result = f(&mut parser)?;
        if parser.token != token::Eof {
            parser.unexpected()?;
        }
        Ok(result)
    }

    pub fn parse_list<'a, T, F>(&self, sess: &'a ParseSess, mut f: F) -> PResult<'a, Vec<T>>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        if self.tokens.is_empty() {
            return Ok(Vec::new());
        }
        self.parse(sess, |parser| {
            parser.expect(&token::OpenDelim(token::Paren))?;
            let mut list = Vec::new();
            while !parser.eat(&token::CloseDelim(token::Paren)) {
                list.push(f(parser)?);
                if !parser.eat(&token::Comma) {
                   parser.expect(&token::CloseDelim(token::Paren))?;
                    break
                }
            }
            Ok(list)
        })
    }

    pub fn parse_meta<'a>(&self, sess: &'a ParseSess) -> PResult<'a, MetaItem> {
        Ok(MetaItem {
            ident: self.path.clone(),
            node: self.parse(sess, |parser| parser.parse_meta_item_kind())?,
            span: self.span,
        })
    }

    /// Convert self to a normal #[doc="foo"] comment, if it is a
    /// comment like `///` or `/** */`. (Returns self unchanged for
    /// non-sugared doc attributes.)
    pub fn with_desugared_doc<T, F>(&self, f: F) -> T where
        F: FnOnce(&Attribute) -> T,
    {
        if self.is_sugared_doc {
            let comment = self.value_str().unwrap();
            let meta = mk_name_value_item_str(
                Ident::from_str("doc"),
                dummy_spanned(Symbol::intern(&strip_doc_comment_decoration(&comment.as_str()))));
            let mut attr = if self.style == ast::AttrStyle::Outer {
                mk_attr_outer(self.span, self.id, meta)
            } else {
                mk_attr_inner(self.span, self.id, meta)
            };
            attr.is_sugared_doc = true;
            f(&attr)
        } else {
            f(self)
        }
    }
}

/* Constructors */

pub fn mk_name_value_item_str(ident: Ident, value: Spanned<Symbol>) -> MetaItem {
    let value = respan(value.span, LitKind::Str(value.node, ast::StrStyle::Cooked));
    mk_name_value_item(ident.span.to(value.span), ident, value)
}

pub fn mk_name_value_item(span: Span, ident: Ident, value: ast::Lit) -> MetaItem {
    MetaItem { ident: Path::from_ident(ident), span, node: MetaItemKind::NameValue(value) }
}

pub fn mk_list_item(span: Span, ident: Ident, items: Vec<NestedMetaItem>) -> MetaItem {
    MetaItem { ident: Path::from_ident(ident), span, node: MetaItemKind::List(items) }
}

pub fn mk_word_item(ident: Ident) -> MetaItem {
    MetaItem { ident: Path::from_ident(ident), span: ident.span, node: MetaItemKind::Word }
}

pub fn mk_nested_word_item(ident: Ident) -> NestedMetaItem {
    respan(ident.span, NestedMetaItemKind::MetaItem(mk_word_item(ident)))
}

pub fn mk_attr_id() -> AttrId {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    static NEXT_ATTR_ID: AtomicUsize = AtomicUsize::new(0);

    let id = NEXT_ATTR_ID.fetch_add(1, Ordering::SeqCst);
    assert!(id != ::std::usize::MAX);
    AttrId(id)
}

/// Returns an inner attribute with the given value.
pub fn mk_attr_inner(span: Span, id: AttrId, item: MetaItem) -> Attribute {
    mk_spanned_attr_inner(span, id, item)
}

/// Returns an inner attribute with the given value and span.
pub fn mk_spanned_attr_inner(sp: Span, id: AttrId, item: MetaItem) -> Attribute {
    Attribute {
        id,
        style: ast::AttrStyle::Inner,
        path: item.ident,
        tokens: item.node.tokens(item.span),
        is_sugared_doc: false,
        span: sp,
    }
}

/// Returns an outer attribute with the given value.
pub fn mk_attr_outer(span: Span, id: AttrId, item: MetaItem) -> Attribute {
    mk_spanned_attr_outer(span, id, item)
}

/// Returns an outer attribute with the given value and span.
pub fn mk_spanned_attr_outer(sp: Span, id: AttrId, item: MetaItem) -> Attribute {
    Attribute {
        id,
        style: ast::AttrStyle::Outer,
        path: item.ident,
        tokens: item.node.tokens(item.span),
        is_sugared_doc: false,
        span: sp,
    }
}

pub fn mk_sugared_doc_attr(id: AttrId, text: Symbol, span: Span) -> Attribute {
    let style = doc_comment_style(&text.as_str());
    let lit = respan(span, LitKind::Str(text, ast::StrStyle::Cooked));
    Attribute {
        id,
        style,
        path: Path::from_ident(Ident::from_str("doc").with_span_pos(span)),
        tokens: MetaItemKind::NameValue(lit).tokens(span),
        is_sugared_doc: true,
        span,
    }
}

pub fn list_contains_name(items: &[NestedMetaItem], name: &str) -> bool {
    items.iter().any(|item| {
        item.check_name(name)
    })
}

pub fn contains_name(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|item| {
        item.check_name(name)
    })
}

pub fn find_by_name<'a>(attrs: &'a [Attribute], name: &str) -> Option<&'a Attribute> {
    attrs.iter().find(|attr| attr.check_name(name))
}

pub fn first_attr_value_str_by_name(attrs: &[Attribute], name: &str) -> Option<Symbol> {
    attrs.iter()
        .find(|at| at.check_name(name))
        .and_then(|at| at.value_str())
}

impl MetaItem {
    fn tokens(&self) -> TokenStream {
        let mut idents = vec![];
        let mut last_pos = BytePos(0 as u32);
        for (i, segment) in self.ident.segments.iter().enumerate() {
            let is_first = i == 0;
            if !is_first {
                let mod_sep_span = Span::new(last_pos,
                                             segment.ident.span.lo(),
                                             segment.ident.span.ctxt());
                idents.push(TokenTree::Token(mod_sep_span, Token::ModSep).into());
            }
            idents.push(TokenTree::Token(segment.ident.span,
                                         Token::from_ast_ident(segment.ident)).into());
            last_pos = segment.ident.span.hi();
        }
        idents.push(self.node.tokens(self.span));
        TokenStream::concat(idents)
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItem>
        where I: Iterator<Item = TokenTree>,
    {
        // FIXME: Share code with `parse_path`.
        let ident = match tokens.next() {
            Some(TokenTree::Token(span, Token::Ident(ident, _))) => {
                if let Some(TokenTree::Token(_, Token::ModSep)) = tokens.peek() {
                    let mut segments = vec![PathSegment::from_ident(ident.with_span_pos(span))];
                    tokens.next();
                    loop {
                        if let Some(TokenTree::Token(span,
                                                     Token::Ident(ident, _))) = tokens.next() {
                            segments.push(PathSegment::from_ident(ident.with_span_pos(span)));
                        } else {
                            return None;
                        }
                        if let Some(TokenTree::Token(_, Token::ModSep)) = tokens.peek() {
                            tokens.next();
                        } else {
                            break;
                        }
                    }
                    let span = span.with_hi(segments.last().unwrap().ident.span.hi());
                    Path { span, segments }
                } else {
                    Path::from_ident(ident.with_span_pos(span))
                }
            }
            Some(TokenTree::Token(_, Token::Interpolated(ref nt))) => match nt.0 {
                token::Nonterminal::NtIdent(ident, _) => Path::from_ident(ident),
                token::Nonterminal::NtMeta(ref meta) => return Some(meta.clone()),
                token::Nonterminal::NtPath(ref path) => path.clone(),
                _ => return None,
            },
            _ => return None,
        };
        let list_closing_paren_pos = tokens.peek().map(|tt| tt.span().hi());
        let node = MetaItemKind::from_tokens(tokens)?;
        let hi = match node {
            MetaItemKind::NameValue(ref lit) => lit.span.hi(),
            MetaItemKind::List(..) => list_closing_paren_pos.unwrap_or(ident.span.hi()),
            _ => ident.span.hi(),
        };
        let span = ident.span.with_hi(hi);
        Some(MetaItem { ident, node, span })
    }
}

impl MetaItemKind {
    pub fn tokens(&self, span: Span) -> TokenStream {
        match *self {
            MetaItemKind::Word => TokenStream::empty(),
            MetaItemKind::NameValue(ref lit) => {
                TokenStream::concat(vec![TokenTree::Token(span, Token::Eq).into(), lit.tokens()])
            }
            MetaItemKind::List(ref list) => {
                let mut tokens = Vec::new();
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        tokens.push(TokenTree::Token(span, Token::Comma).into());
                    }
                    tokens.push(item.node.tokens());
                }
                TokenTree::Delimited(span, Delimited {
                    delim: token::Paren,
                    tts: TokenStream::concat(tokens).into(),
                }).into()
            }
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItemKind>
        where I: Iterator<Item = TokenTree>,
    {
        let delimited = match tokens.peek().cloned() {
            Some(TokenTree::Token(_, token::Eq)) => {
                tokens.next();
                return if let Some(TokenTree::Token(span, token)) = tokens.next() {
                    LitKind::from_token(token)
                        .map(|lit| MetaItemKind::NameValue(Spanned { node: lit, span: span }))
                } else {
                    None
                };
            }
            Some(TokenTree::Delimited(_, ref delimited)) if delimited.delim == token::Paren => {
                tokens.next();
                delimited.stream()
            }
            _ => return Some(MetaItemKind::Word),
        };

        let mut tokens = delimited.into_trees().peekable();
        let mut result = Vec::new();
        while let Some(..) = tokens.peek() {
            let item = NestedMetaItemKind::from_tokens(&mut tokens)?;
            result.push(respan(item.span(), item));
            match tokens.next() {
                None | Some(TokenTree::Token(_, Token::Comma)) => {}
                _ => return None,
            }
        }
        Some(MetaItemKind::List(result))
    }
}

impl NestedMetaItemKind {
    fn span(&self) -> Span {
        match *self {
            NestedMetaItemKind::MetaItem(ref item) => item.span,
            NestedMetaItemKind::Literal(ref lit) => lit.span,
        }
    }

    fn tokens(&self) -> TokenStream {
        match *self {
            NestedMetaItemKind::MetaItem(ref item) => item.tokens(),
            NestedMetaItemKind::Literal(ref lit) => lit.tokens(),
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<NestedMetaItemKind>
        where I: Iterator<Item = TokenTree>,
    {
        if let Some(TokenTree::Token(span, token)) = tokens.peek().cloned() {
            if let Some(node) = LitKind::from_token(token) {
                tokens.next();
                return Some(NestedMetaItemKind::Literal(respan(span, node)));
            }
        }

        MetaItem::from_tokens(tokens).map(NestedMetaItemKind::MetaItem)
    }
}

impl Lit {
    crate fn tokens(&self) -> TokenStream {
        TokenTree::Token(self.span, self.node.token()).into()
    }
}

impl LitKind {
    fn token(&self) -> Token {
        use std::ascii;

        match *self {
            LitKind::Str(string, ast::StrStyle::Cooked) => {
                let escaped = string.as_str().escape_default();
                Token::Literal(token::Lit::Str_(Symbol::intern(&escaped)), None)
            }
            LitKind::Str(string, ast::StrStyle::Raw(n)) => {
                Token::Literal(token::Lit::StrRaw(string, n), None)
            }
            LitKind::ByteStr(ref bytes) => {
                let string = bytes.iter().cloned().flat_map(ascii::escape_default)
                    .map(Into::<char>::into).collect::<String>();
                Token::Literal(token::Lit::ByteStr(Symbol::intern(&string)), None)
            }
            LitKind::Byte(byte) => {
                let string: String = ascii::escape_default(byte).map(Into::<char>::into).collect();
                Token::Literal(token::Lit::Byte(Symbol::intern(&string)), None)
            }
            LitKind::Char(ch) => {
                let string: String = ch.escape_default().map(Into::<char>::into).collect();
                Token::Literal(token::Lit::Char(Symbol::intern(&string)), None)
            }
            LitKind::Int(n, ty) => {
                let suffix = match ty {
                    ast::LitIntType::Unsigned(ty) => Some(Symbol::intern(ty.ty_to_string())),
                    ast::LitIntType::Signed(ty) => Some(Symbol::intern(ty.ty_to_string())),
                    ast::LitIntType::Unsuffixed => None,
                };
                Token::Literal(token::Lit::Integer(Symbol::intern(&n.to_string())), suffix)
            }
            LitKind::Float(symbol, ty) => {
                Token::Literal(token::Lit::Float(symbol), Some(Symbol::intern(ty.ty_to_string())))
            }
            LitKind::FloatUnsuffixed(symbol) => Token::Literal(token::Lit::Float(symbol), None),
            LitKind::Bool(value) => Token::Ident(Ident::with_empty_ctxt(Symbol::intern(if value {
                "true"
            } else {
                "false"
            })), false),
        }
    }

    fn from_token(token: Token) -> Option<LitKind> {
        match token {
            Token::Ident(ident, false) if ident.name == "true" => Some(LitKind::Bool(true)),
            Token::Ident(ident, false) if ident.name == "false" => Some(LitKind::Bool(false)),
            Token::Interpolated(ref nt) => match nt.0 {
                token::NtExpr(ref v) | token::NtLiteral(ref v) => match v.node {
                    ExprKind::Lit(ref lit) => Some(lit.node.clone()),
                    _ => None,
                },
                _ => None,
            },
            Token::Literal(lit, suf) => {
                let (suffix_illegal, result) = parse::lit_token(lit, suf, None);
                if suffix_illegal && suf.is_some() {
                    return None;
                }
                result
            }
            _ => None,
        }
    }
}

pub trait HasAttrs: Sized {
    fn attrs(&self) -> &[ast::Attribute];
    fn map_attrs<F: FnOnce(Vec<ast::Attribute>) -> Vec<ast::Attribute>>(self, f: F) -> Self;
}

impl<T: HasAttrs> HasAttrs for Spanned<T> {
    fn attrs(&self) -> &[ast::Attribute] { self.node.attrs() }
    fn map_attrs<F: FnOnce(Vec<ast::Attribute>) -> Vec<ast::Attribute>>(self, f: F) -> Self {
        respan(self.span, self.node.map_attrs(f))
    }
}

impl HasAttrs for Vec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        self
    }
    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(self, f: F) -> Self {
        f(self)
    }
}

impl HasAttrs for ThinVec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        self
    }
    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(self, f: F) -> Self {
        f(self.into()).into()
    }
}

impl<T: HasAttrs + 'static> HasAttrs for P<T> {
    fn attrs(&self) -> &[Attribute] {
        (**self).attrs()
    }
    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(self, f: F) -> Self {
        self.map(|t| t.map_attrs(f))
    }
}

impl HasAttrs for StmtKind {
    fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtKind::Local(ref local) => local.attrs(),
            StmtKind::Item(..) => &[],
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => expr.attrs(),
            StmtKind::Mac(ref mac) => {
                let (_, _, ref attrs) = **mac;
                attrs.attrs()
            }
        }
    }

    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(self, f: F) -> Self {
        match self {
            StmtKind::Local(local) => StmtKind::Local(local.map_attrs(f)),
            StmtKind::Item(..) => self,
            StmtKind::Expr(expr) => StmtKind::Expr(expr.map_attrs(f)),
            StmtKind::Semi(expr) => StmtKind::Semi(expr.map_attrs(f)),
            StmtKind::Mac(mac) => StmtKind::Mac(mac.map(|(mac, style, attrs)| {
                (mac, style, attrs.map_attrs(f))
            })),
        }
    }
}

impl HasAttrs for Stmt {
    fn attrs(&self) -> &[ast::Attribute] { self.node.attrs() }
    fn map_attrs<F: FnOnce(Vec<ast::Attribute>) -> Vec<ast::Attribute>>(self, f: F) -> Self {
        Stmt { id: self.id, node: self.node.map_attrs(f), span: self.span }
    }
}

impl HasAttrs for GenericParam {
    fn attrs(&self) -> &[ast::Attribute] {
        &self.attrs
    }

    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(mut self, f: F) -> Self {
        self.attrs = self.attrs.map_attrs(f);
        self
    }
}

macro_rules! derive_has_attrs {
    ($($ty:path),*) => { $(
        impl HasAttrs for $ty {
            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn map_attrs<F>(mut self, f: F) -> Self
                where F: FnOnce(Vec<Attribute>) -> Vec<Attribute>,
            {
                self.attrs = self.attrs.map_attrs(f);
                self
            }
        }
    )* }
}

derive_has_attrs! {
    Item, Expr, Local, ast::ForeignItem, ast::StructField, ast::ImplItem, ast::TraitItem, ast::Arm,
    ast::Field, ast::FieldPat, ast::Variant_
}

pub fn inject(mut krate: ast::Crate, parse_sess: &ParseSess, attrs: &[String]) -> ast::Crate {
    for raw_attr in attrs {
        let mut parser = parse::new_parser_from_source_str(
            parse_sess,
            FileName::CliCrateAttr,
            raw_attr.clone(),
        );

        let start_span = parser.span;
        let (path, tokens) = panictry!(parser.parse_meta_item_unrestricted());
        let end_span = parser.span;
        if parser.token != token::Eof {
            parse_sess.span_diagnostic
                .span_err(start_span.to(end_span), "invalid crate attribute");
            continue;
        }

        krate.attrs.push(Attribute {
            id: mk_attr_id(),
            style: AttrStyle::Inner,
            path,
            tokens,
            is_sugared_doc: false,
            span: start_span.to(end_span),
        });
    }

    krate
}
