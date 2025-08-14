//! A higher level attributes based on TokenTree, with also some shortcuts.
use std::iter;
use std::{borrow::Cow, fmt, ops};

use base_db::Crate;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use intern::{Interned, Symbol, sym};

use mbe::{DelimiterKind, Punct};
use smallvec::{SmallVec, smallvec};
use span::{Span, SyntaxContext};
use syntax::unescape;
use syntax::{AstNode, AstToken, SyntaxNode, ast, match_ast};
use syntax_bridge::{DocCommentDesugarMode, desugar_doc_comment_text, syntax_node_to_token_tree};
use triomphe::ThinArc;

use crate::{
    db::ExpandDatabase,
    mod_path::ModPath,
    name::Name,
    span_map::SpanMapRef,
    tt::{self, TopSubtree, token_to_literal},
};

/// Syntactical attributes, without filtering of `cfg_attr`s.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct RawAttrs {
    // FIXME: This can become `Box<[Attr]>` if https://internals.rust-lang.org/t/layout-of-dst-box/21728?u=chrefr is accepted.
    entries: Option<ThinArc<(), Attr>>,
}

impl ops::Deref for RawAttrs {
    type Target = [Attr];

    fn deref(&self) -> &[Attr] {
        match &self.entries {
            Some(it) => &it.slice,
            None => &[],
        }
    }
}

impl RawAttrs {
    pub const EMPTY: Self = Self { entries: None };

    pub fn new(
        db: &dyn ExpandDatabase,
        owner: &dyn ast::HasAttrs,
        span_map: SpanMapRef<'_>,
    ) -> Self {
        let entries: Vec<_> = Self::attrs_iter::<true>(db, owner, span_map).collect();

        let entries = if entries.is_empty() {
            None
        } else {
            Some(ThinArc::from_header_and_iter((), entries.into_iter()))
        };

        RawAttrs { entries }
    }

    /// A [`RawAttrs`] that has its `#[cfg_attr(...)]` attributes expanded.
    pub fn new_expanded(
        db: &dyn ExpandDatabase,
        owner: &dyn ast::HasAttrs,
        span_map: SpanMapRef<'_>,
        cfg_options: &CfgOptions,
    ) -> Self {
        let entries: Vec<_> =
            Self::attrs_iter_expanded::<true>(db, owner, span_map, cfg_options).collect();

        let entries = if entries.is_empty() {
            None
        } else {
            Some(ThinArc::from_header_and_iter((), entries.into_iter()))
        };

        RawAttrs { entries }
    }

    pub fn attrs_iter<const DESUGAR_COMMENTS: bool>(
        db: &dyn ExpandDatabase,
        owner: &dyn ast::HasAttrs,
        span_map: SpanMapRef<'_>,
    ) -> impl Iterator<Item = Attr> {
        collect_attrs(owner).filter_map(move |(id, attr)| match attr {
            Either::Left(attr) => {
                attr.meta().and_then(|meta| Attr::from_src(db, meta, span_map, id))
            }
            Either::Right(comment) if DESUGAR_COMMENTS => comment.doc_comment().map(|doc| {
                let span = span_map.span_for_range(comment.syntax().text_range());
                let (text, kind) = desugar_doc_comment_text(doc, DocCommentDesugarMode::ProcMacro);
                Attr {
                    id,
                    input: Some(Box::new(AttrInput::Literal(tt::Literal {
                        symbol: text,
                        span,
                        kind,
                        suffix: None,
                    }))),
                    path: Interned::new(ModPath::from(Name::new_symbol(sym::doc, span.ctx))),
                    ctxt: span.ctx,
                }
            }),
            Either::Right(_) => None,
        })
    }

    pub fn attrs_iter_expanded<const DESUGAR_COMMENTS: bool>(
        db: &dyn ExpandDatabase,
        owner: &dyn ast::HasAttrs,
        span_map: SpanMapRef<'_>,
        cfg_options: &CfgOptions,
    ) -> impl Iterator<Item = Attr> {
        Self::attrs_iter::<DESUGAR_COMMENTS>(db, owner, span_map)
            .flat_map(|attr| attr.expand_cfg_attr(db, cfg_options))
    }

    pub fn merge(&self, other: Self) -> Self {
        match (&self.entries, other.entries) {
            (None, None) => Self::EMPTY,
            (None, entries @ Some(_)) => Self { entries },
            (Some(entries), None) => Self { entries: Some(entries.clone()) },
            (Some(a), Some(b)) => {
                let last_ast_index = a.slice.last().map_or(0, |it| it.id.ast_index() + 1);
                let items = a
                    .slice
                    .iter()
                    .cloned()
                    .chain(b.slice.iter().map(|it| {
                        let mut it = it.clone();
                        let id = it.id.ast_index() + last_ast_index;
                        it.id = AttrId::new(id, it.id.is_inner_attr());
                        it
                    }))
                    .collect::<Vec<_>>();
                Self { entries: Some(ThinArc::from_header_and_iter((), items.into_iter())) }
            }
        }
    }

    /// Processes `cfg_attr`s
    pub fn expand_cfg_attr(self, db: &dyn ExpandDatabase, krate: Crate) -> RawAttrs {
        let has_cfg_attrs =
            self.iter().any(|attr| attr.path.as_ident().is_some_and(|name| *name == sym::cfg_attr));
        if !has_cfg_attrs {
            return self;
        }

        let cfg_options = krate.cfg_options(db);
        let new_attrs = self
            .iter()
            .cloned()
            .flat_map(|attr| attr.expand_cfg_attr(db, cfg_options))
            .collect::<Vec<_>>();
        let entries = if new_attrs.is_empty() {
            None
        } else {
            Some(ThinArc::from_header_and_iter((), new_attrs.into_iter()))
        };
        RawAttrs { entries }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttrId {
    id: u32,
}

// FIXME: This only handles a single level of cfg_attr nesting
// that is `#[cfg_attr(all(), cfg_attr(all(), cfg(any())))]` breaks again
impl AttrId {
    const INNER_ATTR_SET_BIT: u32 = 1 << 31;

    pub fn new(id: usize, is_inner: bool) -> Self {
        assert!(id <= !Self::INNER_ATTR_SET_BIT as usize);
        let id = id as u32;
        Self { id: if is_inner { id | Self::INNER_ATTR_SET_BIT } else { id } }
    }

    pub fn ast_index(&self) -> usize {
        (self.id & !Self::INNER_ATTR_SET_BIT) as usize
    }

    pub fn is_inner_attr(&self) -> bool {
        self.id & Self::INNER_ATTR_SET_BIT != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub id: AttrId,
    pub path: Interned<ModPath>,
    pub input: Option<Box<AttrInput>>,
    pub ctxt: SyntaxContext,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttrInput {
    /// `#[attr = "string"]`
    Literal(tt::Literal),
    /// `#[attr(subtree)]`
    TokenTree(tt::TopSubtree),
}

impl fmt::Display for AttrInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttrInput::Literal(lit) => write!(f, " = {lit}"),
            AttrInput::TokenTree(tt) => tt.fmt(f),
        }
    }
}

impl Attr {
    fn from_src(
        db: &dyn ExpandDatabase,
        ast: ast::Meta,
        span_map: SpanMapRef<'_>,
        id: AttrId,
    ) -> Option<Attr> {
        let path = ast.path()?;
        let range = path.syntax().text_range();
        let path = Interned::new(ModPath::from_src(db, path, &mut |range| {
            span_map.span_for_range(range).ctx
        })?);
        let span = span_map.span_for_range(range);
        let input = if let Some(ast::Expr::Literal(lit)) = ast.expr() {
            let token = lit.token();
            Some(Box::new(AttrInput::Literal(token_to_literal(token.text(), span))))
        } else if let Some(tt) = ast.token_tree() {
            let tree = syntax_node_to_token_tree(
                tt.syntax(),
                span_map,
                span,
                DocCommentDesugarMode::ProcMacro,
            );
            Some(Box::new(AttrInput::TokenTree(tree)))
        } else {
            None
        };
        Some(Attr { id, path, input, ctxt: span.ctx })
    }

    fn from_tt(
        db: &dyn ExpandDatabase,
        mut tt: tt::TokenTreesView<'_>,
        id: AttrId,
    ) -> Option<Attr> {
        if matches!(tt.flat_tokens(),
            [tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident { sym, .. })), ..]
            if *sym == sym::unsafe_
        ) {
            match tt.iter().nth(1) {
                Some(tt::TtElement::Subtree(_, iter)) => tt = iter.remaining(),
                _ => return None,
            }
        }
        let first = tt.flat_tokens().first()?;
        let ctxt = first.first_span().ctx;
        let (path, input) = {
            let mut iter = tt.iter();
            let start = iter.savepoint();
            let mut input = tt::TokenTreesView::new(&[]);
            let mut path = iter.from_savepoint(start);
            let mut path_split_savepoint = iter.savepoint();
            while let Some(tt) = iter.next() {
                path = iter.from_savepoint(start);
                if !matches!(
                    tt,
                    tt::TtElement::Leaf(
                        tt::Leaf::Punct(tt::Punct { char: ':' | '$', .. }) | tt::Leaf::Ident(_),
                    )
                ) {
                    input = path_split_savepoint.remaining();
                    break;
                }
                path_split_savepoint = iter.savepoint();
            }
            (path, input)
        };

        let path = Interned::new(ModPath::from_tt(db, path)?);

        let input = match (input.flat_tokens().first(), input.try_into_subtree()) {
            (_, Some(tree)) => {
                Some(Box::new(AttrInput::TokenTree(tt::TopSubtree::from_subtree(tree))))
            }
            (Some(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: '=', .. }))), _) => {
                match input.flat_tokens().get(1) {
                    Some(tt::TokenTree::Leaf(tt::Leaf::Literal(lit))) => {
                        Some(Box::new(AttrInput::Literal(lit.clone())))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        Some(Attr { id, path, input, ctxt })
    }

    pub fn path(&self) -> &ModPath {
        &self.path
    }

    pub fn expand_cfg_attr(
        self,
        db: &dyn ExpandDatabase,
        cfg_options: &CfgOptions,
    ) -> impl IntoIterator<Item = Self> {
        let is_cfg_attr = self.path.as_ident().is_some_and(|name| *name == sym::cfg_attr);
        if !is_cfg_attr {
            return smallvec![self];
        }

        let subtree = match self.token_tree_value() {
            Some(it) => it,
            _ => return smallvec![self.clone()],
        };

        let (cfg, parts) = match parse_cfg_attr_input(subtree) {
            Some(it) => it,
            None => return smallvec![self.clone()],
        };
        let index = self.id;
        let attrs = parts.filter_map(|attr| Attr::from_tt(db, attr, index));

        let cfg = TopSubtree::from_token_trees(subtree.top_subtree().delimiter, cfg);
        let cfg = CfgExpr::parse(&cfg);
        if cfg_options.check(&cfg) == Some(false) {
            smallvec![]
        } else {
            cov_mark::hit!(cfg_attr_active);

            attrs.collect::<SmallVec<[_; 1]>>()
        }
    }
}

impl Attr {
    /// #[path = "string"]
    pub fn string_value(&self) -> Option<&Symbol> {
        match self.input.as_deref()? {
            AttrInput::Literal(tt::Literal {
                symbol: text,
                kind: tt::LitKind::Str | tt::LitKind::StrRaw(_),
                ..
            }) => Some(text),
            _ => None,
        }
    }

    /// #[path = "string"]
    pub fn string_value_with_span(&self) -> Option<(&Symbol, span::Span)> {
        match self.input.as_deref()? {
            AttrInput::Literal(tt::Literal {
                symbol: text,
                kind: tt::LitKind::Str | tt::LitKind::StrRaw(_),
                span,
                suffix: _,
            }) => Some((text, *span)),
            _ => None,
        }
    }

    pub fn string_value_unescape(&self) -> Option<Cow<'_, str>> {
        match self.input.as_deref()? {
            AttrInput::Literal(tt::Literal {
                symbol: text, kind: tt::LitKind::StrRaw(_), ..
            }) => Some(Cow::Borrowed(text.as_str())),
            AttrInput::Literal(tt::Literal { symbol: text, kind: tt::LitKind::Str, .. }) => {
                unescape(text.as_str())
            }
            _ => None,
        }
    }

    /// #[path(ident)]
    pub fn single_ident_value(&self) -> Option<&tt::Ident> {
        match self.input.as_deref()? {
            AttrInput::TokenTree(tt) => match tt.token_trees().flat_tokens() {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] => Some(ident),
                _ => None,
            },
            _ => None,
        }
    }

    /// #[path TokenTree]
    pub fn token_tree_value(&self) -> Option<&TopSubtree> {
        match self.input.as_deref()? {
            AttrInput::TokenTree(tt) => Some(tt),
            _ => None,
        }
    }

    /// Parses this attribute as a token tree consisting of comma separated paths.
    pub fn parse_path_comma_token_tree<'a>(
        &'a self,
        db: &'a dyn ExpandDatabase,
    ) -> Option<impl Iterator<Item = (ModPath, Span)> + 'a> {
        let args = self.token_tree_value()?;

        if args.top_subtree().delimiter.kind != DelimiterKind::Parenthesis {
            return None;
        }
        let paths = args
            .token_trees()
            .split(|tt| matches!(tt, tt::TtElement::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))))
            .filter_map(move |tts| {
                let span = tts.flat_tokens().first()?.first_span();
                Some((ModPath::from_tt(db, tts)?, span))
            });

        Some(paths)
    }

    pub fn cfg(&self) -> Option<CfgExpr> {
        if *self.path.as_ident()? == sym::cfg {
            self.token_tree_value().map(CfgExpr::parse)
        } else {
            None
        }
    }
}

fn unescape(s: &str) -> Option<Cow<'_, str>> {
    let mut buf = String::new();
    let mut prev_end = 0;
    let mut has_error = false;
    unescape::unescape_str(s, |char_range, unescaped_char| {
        match (unescaped_char, buf.capacity() == 0) {
            (Ok(c), false) => buf.push(c),
            (Ok(_), true) if char_range.len() == 1 && char_range.start == prev_end => {
                prev_end = char_range.end
            }
            (Ok(c), true) => {
                buf.reserve_exact(s.len());
                buf.push_str(&s[..prev_end]);
                buf.push(c);
            }
            (Err(_), _) => has_error = true,
        }
    });

    match (has_error, buf.capacity() == 0) {
        (true, _) => None,
        (false, false) => Some(Cow::Owned(buf)),
        (false, true) => Some(Cow::Borrowed(s)),
    }
}

pub fn collect_attrs(
    owner: &dyn ast::HasAttrs,
) -> impl Iterator<Item = (AttrId, Either<ast::Attr, ast::Comment>)> {
    let inner_attrs =
        inner_attributes(owner.syntax()).into_iter().flatten().zip(iter::repeat(true));
    let outer_attrs = ast::AttrDocCommentIter::from_syntax_node(owner.syntax())
        .filter(|el| match el {
            Either::Left(attr) => attr.kind().is_outer(),
            Either::Right(comment) => comment.is_outer(),
        })
        .zip(iter::repeat(false));
    outer_attrs
        .chain(inner_attrs)
        .enumerate()
        .map(|(id, (attr, is_inner))| (AttrId::new(id, is_inner), attr))
}

fn inner_attributes(
    syntax: &SyntaxNode,
) -> Option<impl Iterator<Item = Either<ast::Attr, ast::Comment>>> {
    let node = match_ast! {
        match syntax {
            ast::SourceFile(_) => syntax.clone(),
            ast::ExternBlock(it) => it.extern_item_list()?.syntax().clone(),
            ast::Fn(it) => it.body()?.stmt_list()?.syntax().clone(),
            ast::Impl(it) => it.assoc_item_list()?.syntax().clone(),
            ast::Module(it) => it.item_list()?.syntax().clone(),
            ast::BlockExpr(it) => {
                if !it.may_carry_attributes() {
                    return None
                }
                syntax.clone()
            },
            _ => return None,
        }
    };

    let attrs = ast::AttrDocCommentIter::from_syntax_node(&node).filter(|el| match el {
        Either::Left(attr) => attr.kind().is_inner(),
        Either::Right(comment) => comment.is_inner(),
    });
    Some(attrs)
}

// Input subtree is: `(cfg, $(attr),+)`
// Split it up into a `cfg` subtree and the `attr` subtrees.
fn parse_cfg_attr_input(
    subtree: &TopSubtree,
) -> Option<(tt::TokenTreesView<'_>, impl Iterator<Item = tt::TokenTreesView<'_>>)> {
    let mut parts = subtree
        .token_trees()
        .split(|tt| matches!(tt, tt::TtElement::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))));
    let cfg = parts.next()?;
    Some((cfg, parts.filter(|it| !it.is_empty())))
}
