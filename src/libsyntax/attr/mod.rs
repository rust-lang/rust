//! Functions dealing with attributes and meta items

mod builtin;

pub use builtin::{
    cfg_matches, contains_feature_attr, eval_condition, find_crate_name, find_deprecation,
    find_repr_attrs, find_stability, find_unwind_attr, Deprecation, InlineAttr, OptimizeAttr,
    IntType, ReprAttr, RustcDeprecation, Stability, StabilityLevel, UnwindAttr,
};
pub use IntType::*;
pub use ReprAttr::*;
pub use StabilityLevel::*;

use crate::ast;
use crate::ast::{AttrId, Attribute, AttrStyle, Name, Ident, Path, PathSegment};
use crate::ast::{MetaItem, MetaItemKind, NestedMetaItem};
use crate::ast::{Lit, LitKind, Expr, Item, Local, Stmt, StmtKind, GenericParam};
use crate::mut_visit::visit_clobber;
use crate::source_map::{BytePos, Spanned, dummy_spanned};
use crate::parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use crate::parse::parser::Parser;
use crate::parse::{self, ParseSess, PResult};
use crate::parse::token::{self, Token};
use crate::ptr::P;
use crate::symbol::{sym, Symbol};
use crate::ThinVec;
use crate::tokenstream::{TokenStream, TokenTree, DelimSpan};
use crate::GLOBALS;

use log::debug;
use syntax_pos::{FileName, Span};

use std::iter;
use std::ops::DerefMut;

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
    /// Returns the MetaItem if self is a NestedMetaItem::MetaItem.
    pub fn meta_item(&self) -> Option<&MetaItem> {
        match *self {
            NestedMetaItem::MetaItem(ref item) => Some(item),
            _ => None
        }
    }

    /// Returns the Lit if self is a NestedMetaItem::Literal.
    pub fn literal(&self) -> Option<&Lit> {
        match *self {
            NestedMetaItem::Literal(ref lit) => Some(lit),
            _ => None
        }
    }

    /// Returns `true` if this list item is a MetaItem with a name of `name`.
    pub fn check_name(&self, name: Symbol) -> bool {
        self.meta_item().map_or(false, |meta_item| meta_item.check_name(name))
    }

    /// For a single-segment meta-item returns its name, otherwise returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        self.meta_item().and_then(|meta_item| meta_item.ident())
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or(Ident::invalid()).name
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
                        if let Some(ident) = meta_item.ident() {
                            if let Some(lit) = meta_item_list[0].literal() {
                                return Some((ident.name, lit));
                            }
                        }
                    }
                    None
                }))
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
        self.meta_item().map_or(false, |meta_item| meta_item.is_word())
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

impl Attribute {
    /// Returns `true` if the attribute's path matches the argument. If it matches, then the
    /// attribute is marked as used.
    ///
    /// To check the attribute name without marking it used, use the `path` field directly.
    pub fn check_name(&self, name: Symbol) -> bool {
        let matches = self.path == name;
        if matches {
            mark_used(self);
        }
        matches
    }

    /// For a single-segment attribute returns its name, otherwise returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        if self.path.segments.len() == 1 {
            Some(self.path.segments[0].ident)
        } else {
            None
        }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or(Ident::invalid()).name
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
        self.tokens.is_empty()
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
    /// For a single-segment meta-item returns its name, otherwise returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        if self.path.segments.len() == 1 {
            Some(self.path.segments[0].ident)
        } else {
            None
        }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or(Ident::invalid()).name
    }

    // #[attribute(name = "value")]
    //             ^^^^^^^^^^^^^^
    pub fn name_value_literal(&self) -> Option<&Lit> {
        match &self.node {
            MetaItemKind::NameValue(v) => Some(v),
            _ => None,
        }
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

    pub fn check_name(&self, name: Symbol) -> bool {
        self.path == name
    }

    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }

    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }
}

impl Attribute {
    /// Extracts the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        let mut tokens = self.tokens.trees().peekable();
        Some(MetaItem {
            path: self.path.clone(),
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
        let mut parser = Parser::new(
            sess,
            self.tokens.clone(),
            None,
            false,
            false,
            Some("attribute"),
        );
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
            path: self.path.clone(),
            node: self.parse(sess, |parser| parser.parse_meta_item_kind())?,
            span: self.span,
        })
    }

    /// Converts self to a normal #[doc="foo"] comment, if it is a
    /// comment like `///` or `/** */`. (Returns self unchanged for
    /// non-sugared doc attributes.)
    pub fn with_desugared_doc<T, F>(&self, f: F) -> T where
        F: FnOnce(&Attribute) -> T,
    {
        if self.is_sugared_doc {
            let comment = self.value_str().unwrap();
            let meta = mk_name_value_item_str(
                Ident::with_empty_ctxt(sym::doc),
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
    let lit_kind = LitKind::Str(value.node, ast::StrStyle::Cooked);
    mk_name_value_item(ident.span.to(value.span), ident, lit_kind, value.span)
}

pub fn mk_name_value_item(span: Span, ident: Ident, lit_kind: LitKind, lit_span: Span) -> MetaItem {
    let lit = Lit::from_lit_kind(lit_kind, lit_span);
    MetaItem { path: Path::from_ident(ident), span, node: MetaItemKind::NameValue(lit) }
}

pub fn mk_list_item(span: Span, ident: Ident, items: Vec<NestedMetaItem>) -> MetaItem {
    MetaItem { path: Path::from_ident(ident), span, node: MetaItemKind::List(items) }
}

pub fn mk_word_item(ident: Ident) -> MetaItem {
    MetaItem { path: Path::from_ident(ident), span: ident.span, node: MetaItemKind::Word }
}

pub fn mk_nested_word_item(ident: Ident) -> NestedMetaItem {
    NestedMetaItem::MetaItem(mk_word_item(ident))
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
        path: item.path,
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
        path: item.path,
        tokens: item.node.tokens(item.span),
        is_sugared_doc: false,
        span: sp,
    }
}

pub fn mk_sugared_doc_attr(id: AttrId, text: Symbol, span: Span) -> Attribute {
    let style = doc_comment_style(&text.as_str());
    let lit_kind = LitKind::Str(text, ast::StrStyle::Cooked);
    let lit = Lit::from_lit_kind(lit_kind, span);
    Attribute {
        id,
        style,
        path: Path::from_ident(Ident::with_empty_ctxt(sym::doc).with_span_pos(span)),
        tokens: MetaItemKind::NameValue(lit).tokens(span),
        is_sugared_doc: true,
        span,
    }
}

pub fn list_contains_name(items: &[NestedMetaItem], name: Symbol) -> bool {
    items.iter().any(|item| {
        item.check_name(name)
    })
}

pub fn contains_name(attrs: &[Attribute], name: Symbol) -> bool {
    attrs.iter().any(|item| {
        item.check_name(name)
    })
}

pub fn find_by_name(attrs: &[Attribute], name: Symbol) -> Option<&Attribute> {
    attrs.iter().find(|attr| attr.check_name(name))
}

pub fn filter_by_name(attrs: &[Attribute], name: Symbol)
                      -> impl Iterator<Item=&Attribute> {
    attrs.iter().filter(move |attr| attr.check_name(name))
}

pub fn first_attr_value_str_by_name(attrs: &[Attribute], name: Symbol) -> Option<Symbol> {
    attrs.iter()
        .find(|at| at.check_name(name))
        .and_then(|at| at.value_str())
}

impl MetaItem {
    fn tokens(&self) -> TokenStream {
        let mut idents = vec![];
        let mut last_pos = BytePos(0 as u32);
        for (i, segment) in self.path.segments.iter().enumerate() {
            let is_first = i == 0;
            if !is_first {
                let mod_sep_span = Span::new(last_pos,
                                             segment.ident.span.lo(),
                                             segment.ident.span.ctxt());
                idents.push(TokenTree::token(token::ModSep, mod_sep_span).into());
            }
            idents.push(TokenTree::Token(Token::from_ast_ident(segment.ident)).into());
            last_pos = segment.ident.span.hi();
        }
        self.node.tokens(self.span).append_to_tree_and_joint_vec(&mut idents);
        TokenStream::new(idents)
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItem>
        where I: Iterator<Item = TokenTree>,
    {
        // FIXME: Share code with `parse_path`.
        let path = match tokens.next() {
            Some(TokenTree::Token(Token { kind: kind @ token::Ident(..), span })) |
            Some(TokenTree::Token(Token { kind: kind @ token::ModSep, span })) => 'arm: {
                let mut segments = if let token::Ident(name, _) = kind {
                    if let Some(TokenTree::Token(Token { kind: token::ModSep, .. }))
                            = tokens.peek() {
                        tokens.next();
                        vec![PathSegment::from_ident(Ident::new(name, span))]
                    } else {
                        break 'arm Path::from_ident(Ident::new(name, span));
                    }
                } else {
                    vec![PathSegment::path_root(span)]
                };
                loop {
                    if let Some(TokenTree::Token(Token { kind: token::Ident(name, _), span }))
                            = tokens.next() {
                        segments.push(PathSegment::from_ident(Ident::new(name, span)));
                    } else {
                        return None;
                    }
                    if let Some(TokenTree::Token(Token { kind: token::ModSep, .. }))
                            = tokens.peek() {
                        tokens.next();
                    } else {
                        break;
                    }
                }
                let span = span.with_hi(segments.last().unwrap().ident.span.hi());
                Path { span, segments }
            }
            Some(TokenTree::Token(Token { kind: token::Interpolated(nt), .. })) => match *nt {
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
            MetaItemKind::List(..) => list_closing_paren_pos.unwrap_or(path.span.hi()),
            _ => path.span.hi(),
        };
        let span = path.span.with_hi(hi);
        Some(MetaItem { path, node, span })
    }
}

impl MetaItemKind {
    pub fn tokens(&self, span: Span) -> TokenStream {
        match *self {
            MetaItemKind::Word => TokenStream::empty(),
            MetaItemKind::NameValue(ref lit) => {
                let mut vec = vec![TokenTree::token(token::Eq, span).into()];
                lit.tokens().append_to_tree_and_joint_vec(&mut vec);
                TokenStream::new(vec)
            }
            MetaItemKind::List(ref list) => {
                let mut tokens = Vec::new();
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        tokens.push(TokenTree::token(token::Comma, span).into());
                    }
                    item.tokens().append_to_tree_and_joint_vec(&mut tokens);
                }
                TokenTree::Delimited(
                    DelimSpan::from_single(span),
                    token::Paren,
                    TokenStream::new(tokens).into(),
                ).into()
            }
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItemKind>
        where I: Iterator<Item = TokenTree>,
    {
        let delimited = match tokens.peek().cloned() {
            Some(TokenTree::Token(token)) if token == token::Eq => {
                tokens.next();
                return if let Some(TokenTree::Token(token)) = tokens.next() {
                    Lit::from_token(&token).ok().map(MetaItemKind::NameValue)
                } else {
                    None
                };
            }
            Some(TokenTree::Delimited(_, delim, ref tts)) if delim == token::Paren => {
                tokens.next();
                tts.clone()
            }
            _ => return Some(MetaItemKind::Word),
        };

        let mut tokens = delimited.into_trees().peekable();
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
}

impl NestedMetaItem {
    pub fn span(&self) -> Span {
        match *self {
            NestedMetaItem::MetaItem(ref item) => item.span,
            NestedMetaItem::Literal(ref lit) => lit.span,
        }
    }

    fn tokens(&self) -> TokenStream {
        match *self {
            NestedMetaItem::MetaItem(ref item) => item.tokens(),
            NestedMetaItem::Literal(ref lit) => lit.tokens(),
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<NestedMetaItem>
        where I: Iterator<Item = TokenTree>,
    {
        if let Some(TokenTree::Token(token)) = tokens.peek() {
            if let Ok(lit) = Lit::from_token(token) {
                tokens.next();
                return Some(NestedMetaItem::Literal(lit));
            }
        }

        MetaItem::from_tokens(tokens).map(NestedMetaItem::MetaItem)
    }
}

pub trait HasAttrs: Sized {
    fn attrs(&self) -> &[ast::Attribute];
    fn visit_attrs<F: FnOnce(&mut Vec<ast::Attribute>)>(&mut self, f: F);
}

impl<T: HasAttrs> HasAttrs for Spanned<T> {
    fn attrs(&self) -> &[ast::Attribute] { self.node.attrs() }
    fn visit_attrs<F: FnOnce(&mut Vec<ast::Attribute>)>(&mut self, f: F) {
        self.node.visit_attrs(f);
    }
}

impl HasAttrs for Vec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        self
    }
    fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
        f(self)
    }
}

impl HasAttrs for ThinVec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        self
    }
    fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
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
    fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
        (**self).visit_attrs(f);
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

    fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
        match self {
            StmtKind::Local(local) => local.visit_attrs(f),
            StmtKind::Item(..) => {}
            StmtKind::Expr(expr) => expr.visit_attrs(f),
            StmtKind::Semi(expr) => expr.visit_attrs(f),
            StmtKind::Mac(mac) => {
                let (_mac, _style, attrs) = mac.deref_mut();
                attrs.visit_attrs(f);
            }
        }
    }
}

impl HasAttrs for Stmt {
    fn attrs(&self) -> &[ast::Attribute] {
        self.node.attrs()
    }

    fn visit_attrs<F: FnOnce(&mut Vec<ast::Attribute>)>(&mut self, f: F) {
        self.node.visit_attrs(f);
    }
}

impl HasAttrs for GenericParam {
    fn attrs(&self) -> &[ast::Attribute] {
        &self.attrs
    }

    fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
        self.attrs.visit_attrs(f);
    }
}

macro_rules! derive_has_attrs {
    ($($ty:path),*) => { $(
        impl HasAttrs for $ty {
            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
                self.attrs.visit_attrs(f);
            }
        }
    )* }
}

derive_has_attrs! {
    Item, Expr, Local, ast::ForeignItem, ast::StructField, ast::ImplItem, ast::TraitItem, ast::Arm,
    ast::Field, ast::FieldPat, ast::Variant_, ast::Arg
}

pub fn inject(mut krate: ast::Crate, parse_sess: &ParseSess, attrs: &[String]) -> ast::Crate {
    for raw_attr in attrs {
        let mut parser = parse::new_parser_from_source_str(
            parse_sess,
            FileName::cli_crate_attr_source_code(&raw_attr),
            raw_attr.clone(),
        );

        let start_span = parser.token.span;
        let (path, tokens) = panictry!(parser.parse_meta_item_unrestricted());
        let end_span = parser.token.span;
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
