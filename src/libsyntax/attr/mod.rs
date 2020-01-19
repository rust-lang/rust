//! Functions dealing with attributes and meta items.

use crate::ast;
use crate::ast::{AttrId, AttrItem, AttrKind, AttrStyle, AttrVec, Attribute};
use crate::ast::{Expr, GenericParam, Item, Lit, LitKind, Local, Stmt, StmtKind};
use crate::ast::{Ident, Name, Path, PathSegment};
use crate::ast::{MacArgs, MacDelimiter, MetaItem, MetaItemKind, NestedMetaItem};
use crate::mut_visit::visit_clobber;
use crate::ptr::P;
use crate::token::{self, Token};
use crate::tokenstream::{DelimSpan, TokenStream, TokenTree, TreeAndJoint};

use rustc_data_structures::sync::Lock;
use rustc_index::bit_set::GrowableBitSet;
use rustc_span::edition::{Edition, DEFAULT_EDITION};
use rustc_span::source_map::{BytePos, Spanned};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

use log::debug;
use std::iter;
use std::ops::DerefMut;

pub struct Globals {
    used_attrs: Lock<GrowableBitSet<AttrId>>,
    known_attrs: Lock<GrowableBitSet<AttrId>>,
    rustc_span_globals: rustc_span::Globals,
}

impl Globals {
    fn new(edition: Edition) -> Globals {
        Globals {
            // We have no idea how many attributes there will be, so just
            // initiate the vectors with 0 bits. We'll grow them as necessary.
            used_attrs: Lock::new(GrowableBitSet::new_empty()),
            known_attrs: Lock::new(GrowableBitSet::new_empty()),
            rustc_span_globals: rustc_span::Globals::new(edition),
        }
    }
}

pub fn with_globals<R>(edition: Edition, f: impl FnOnce() -> R) -> R {
    let globals = Globals::new(edition);
    GLOBALS.set(&globals, || rustc_span::GLOBALS.set(&globals.rustc_span_globals, f))
}

pub fn with_default_globals<R>(f: impl FnOnce() -> R) -> R {
    with_globals(DEFAULT_EDITION, f)
}

scoped_tls::scoped_thread_local!(pub static GLOBALS: Globals);

pub fn mark_used(attr: &Attribute) {
    debug!("marking {:?} as used", attr);
    GLOBALS.with(|globals| {
        globals.used_attrs.lock().insert(attr.id);
    });
}

pub fn is_used(attr: &Attribute) -> bool {
    GLOBALS.with(|globals| globals.used_attrs.lock().contains(attr.id))
}

pub fn mark_known(attr: &Attribute) {
    debug!("marking {:?} as known", attr);
    GLOBALS.with(|globals| {
        globals.known_attrs.lock().insert(attr.id);
    });
}

pub fn is_known(attr: &Attribute) -> bool {
    GLOBALS.with(|globals| globals.known_attrs.lock().contains(attr.id))
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
    pub fn check_name(&self, name: Symbol) -> bool {
        self.meta_item().map_or(false, |meta_item| meta_item.check_name(name))
    }

    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        self.meta_item().and_then(|meta_item| meta_item.ident())
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or(Ident::invalid()).name
    }

    /// Gets the string value if `self` is a `MetaItem` and the `MetaItem` is a
    /// `MetaItemKind::NameValue` variant containing a string, otherwise `None`.
    pub fn value_str(&self) -> Option<Symbol> {
        self.meta_item().and_then(|meta_item| meta_item.value_str())
    }

    /// Returns a name and single literal value tuple of the `MetaItem`.
    pub fn name_value_literal(&self) -> Option<(Name, &Lit)> {
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

    /// Returns `true` if the variant is `Literal`.
    pub fn is_literal(&self) -> bool {
        self.literal().is_some()
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
}

impl Attribute {
    pub fn has_name(&self, name: Symbol) -> bool {
        match self.kind {
            AttrKind::Normal(ref item) => item.path == name,
            AttrKind::DocComment(_) => false,
        }
    }

    /// Returns `true` if the attribute's path matches the argument. If it matches, then the
    /// attribute is marked as used.
    pub fn check_name(&self, name: Symbol) -> bool {
        let matches = self.has_name(name);
        if matches {
            mark_used(self);
        }
        matches
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        match self.kind {
            AttrKind::Normal(ref item) => {
                if item.path.segments.len() == 1 {
                    Some(item.path.segments[0].ident)
                } else {
                    None
                }
            }
            AttrKind::DocComment(_) => None,
        }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or(Ident::invalid()).name
    }

    pub fn value_str(&self) -> Option<Symbol> {
        match self.kind {
            AttrKind::Normal(ref item) => item.meta(self.span).and_then(|meta| meta.value_str()),
            AttrKind::DocComment(..) => None,
        }
    }

    pub fn meta_item_list(&self) -> Option<Vec<NestedMetaItem>> {
        match self.kind {
            AttrKind::Normal(ref item) => match item.meta(self.span) {
                Some(MetaItem { kind: MetaItemKind::List(list), .. }) => Some(list),
                _ => None,
            },
            AttrKind::DocComment(_) => None,
        }
    }

    pub fn is_word(&self) -> bool {
        if let AttrKind::Normal(item) = &self.kind {
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
}

impl MetaItem {
    /// For a single-segment meta item, returns its name; otherwise, returns `None`.
    pub fn ident(&self) -> Option<Ident> {
        if self.path.segments.len() == 1 { Some(self.path.segments[0].ident) } else { None }
    }
    pub fn name_or_empty(&self) -> Symbol {
        self.ident().unwrap_or(Ident::invalid()).name
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
        match self.kind {
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

impl AttrItem {
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
            AttrKind::Normal(_) => false,
            AttrKind::DocComment(_) => true,
        }
    }

    pub fn doc_str(&self) -> Option<Symbol> {
        match self.kind {
            AttrKind::DocComment(symbol) => Some(symbol),
            AttrKind::Normal(ref item) if item.path == sym::doc => {
                item.meta(self.span).and_then(|meta| meta.value_str())
            }
            _ => None,
        }
    }

    pub fn get_normal_item(&self) -> &AttrItem {
        match self.kind {
            AttrKind::Normal(ref item) => item,
            AttrKind::DocComment(_) => panic!("unexpected doc comment"),
        }
    }

    pub fn unwrap_normal_item(self) -> AttrItem {
        match self.kind {
            AttrKind::Normal(item) => item,
            AttrKind::DocComment(_) => panic!("unexpected doc comment"),
        }
    }

    /// Extracts the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        match self.kind {
            AttrKind::Normal(ref item) => item.meta(self.span),
            AttrKind::DocComment(..) => None,
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    static NEXT_ATTR_ID: AtomicUsize = AtomicUsize::new(0);

    let id = NEXT_ATTR_ID.fetch_add(1, Ordering::SeqCst);
    assert!(id != ::std::usize::MAX);
    AttrId(id)
}

pub fn mk_attr(style: AttrStyle, path: Path, args: MacArgs, span: Span) -> Attribute {
    mk_attr_from_item(style, AttrItem { path, args }, span)
}

pub fn mk_attr_from_item(style: AttrStyle, item: AttrItem, span: Span) -> Attribute {
    Attribute { kind: AttrKind::Normal(item), id: mk_attr_id(), style, span }
}

/// Returns an inner attribute with the given value and span.
pub fn mk_attr_inner(item: MetaItem) -> Attribute {
    mk_attr(AttrStyle::Inner, item.path, item.kind.mac_args(item.span), item.span)
}

/// Returns an outer attribute with the given value and span.
pub fn mk_attr_outer(item: MetaItem) -> Attribute {
    mk_attr(AttrStyle::Outer, item.path, item.kind.mac_args(item.span), item.span)
}

pub fn mk_doc_comment(style: AttrStyle, comment: Symbol, span: Span) -> Attribute {
    Attribute { kind: AttrKind::DocComment(comment), id: mk_attr_id(), style, span }
}

pub fn list_contains_name(items: &[NestedMetaItem], name: Symbol) -> bool {
    items.iter().any(|item| item.check_name(name))
}

pub fn contains_name(attrs: &[Attribute], name: Symbol) -> bool {
    attrs.iter().any(|item| item.check_name(name))
}

pub fn find_by_name(attrs: &[Attribute], name: Symbol) -> Option<&Attribute> {
    attrs.iter().find(|attr| attr.check_name(name))
}

pub fn filter_by_name(attrs: &[Attribute], name: Symbol) -> impl Iterator<Item = &Attribute> {
    attrs.iter().filter(move |attr| attr.check_name(name))
}

pub fn first_attr_value_str_by_name(attrs: &[Attribute], name: Symbol) -> Option<Symbol> {
    attrs.iter().find(|at| at.check_name(name)).and_then(|at| at.value_str())
}

impl MetaItem {
    fn token_trees_and_joints(&self) -> Vec<TreeAndJoint> {
        let mut idents = vec![];
        let mut last_pos = BytePos(0 as u32);
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
        idents.extend(self.kind.token_trees_and_joints(self.span));
        idents
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItem>
    where
        I: Iterator<Item = TokenTree>,
    {
        // FIXME: Share code with `parse_path`.
        let path = match tokens.next() {
            Some(TokenTree::Token(Token { kind: kind @ token::Ident(..), span }))
            | Some(TokenTree::Token(Token { kind: kind @ token::ModSep, span })) => 'arm: {
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
                        tokens.next()
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
                Path { span, segments }
            }
            Some(TokenTree::Token(Token { kind: token::Interpolated(nt), .. })) => match *nt {
                token::Nonterminal::NtIdent(ident, _) => Path::from_ident(ident),
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
                    tts.extend(item.token_trees_and_joints())
                }
                MacArgs::Delimited(
                    DelimSpan::from_single(span),
                    MacDelimiter::Parenthesis,
                    TokenStream::new(tts),
                )
            }
        }
    }

    fn token_trees_and_joints(&self, span: Span) -> Vec<TreeAndJoint> {
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
                    tokens.extend(item.token_trees_and_joints())
                }
                vec![
                    TokenTree::Delimited(
                        DelimSpan::from_single(span),
                        token::Paren,
                        TokenStream::new(tokens).into(),
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

    fn token_trees_and_joints(&self) -> Vec<TreeAndJoint> {
        match *self {
            NestedMetaItem::MetaItem(ref item) => item.token_trees_and_joints(),
            NestedMetaItem::Literal(ref lit) => vec![lit.token_tree().into()],
        }
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<NestedMetaItem>
    where
        I: Iterator<Item = TokenTree>,
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
            StmtKind::Item(..) => &[],
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => expr.attrs(),
            StmtKind::Mac(ref mac) => {
                let (_, _, ref attrs) = **mac;
                attrs.attrs()
            }
        }
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
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
    Item, Expr, Local, ast::ForeignItem, ast::StructField, ast::AssocItem, ast::Arm,
    ast::Field, ast::FieldPat, ast::Variant, ast::Param, GenericParam
}
