//! A higher level attributes based on TokenTree, with also some shortcuts.
use std::{borrow::Cow, fmt, ops};

use base_db::CrateId;
use cfg::CfgExpr;
use either::Either;
use intern::{sym, Interned, Symbol};

use mbe::{DelimiterKind, Punct};
use smallvec::{smallvec, SmallVec};
use span::{Span, SyntaxContextId};
use syntax::unescape;
use syntax::{ast, match_ast, AstNode, AstToken, SyntaxNode};
use syntax_bridge::{desugar_doc_comment_text, syntax_node_to_token_tree, DocCommentDesugarMode};
use triomphe::ThinArc;

use crate::name::Name;
use crate::{
    db::ExpandDatabase,
    mod_path::ModPath,
    span_map::SpanMapRef,
    tt::{self, token_to_literal, Subtree},
    InFile,
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
        let entries: Vec<_> = collect_attrs(owner)
            .filter_map(|(id, attr)| match attr {
                Either::Left(attr) => {
                    attr.meta().and_then(|meta| Attr::from_src(db, meta, span_map, id))
                }
                Either::Right(comment) => comment.doc_comment().map(|doc| {
                    let span = span_map.span_for_range(comment.syntax().text_range());
                    let (text, kind) =
                        desugar_doc_comment_text(doc, DocCommentDesugarMode::ProcMacro);
                    Attr {
                        id,
                        input: Some(Box::new(AttrInput::Literal(tt::Literal {
                            symbol: text,
                            span,
                            kind,
                            suffix: None,
                        }))),
                        path: Interned::new(ModPath::from(Name::new_symbol(
                            sym::doc.clone(),
                            span.ctx,
                        ))),
                        ctxt: span.ctx,
                    }
                }),
            })
            .collect();

        let entries = if entries.is_empty() {
            None
        } else {
            Some(ThinArc::from_header_and_iter((), entries.into_iter()))
        };

        RawAttrs { entries }
    }

    pub fn from_attrs_owner(
        db: &dyn ExpandDatabase,
        owner: InFile<&dyn ast::HasAttrs>,
        span_map: SpanMapRef<'_>,
    ) -> Self {
        Self::new(db, owner.value, span_map)
    }

    pub fn merge(&self, other: Self) -> Self {
        match (&self.entries, other.entries) {
            (None, None) => Self::EMPTY,
            (None, entries @ Some(_)) => Self { entries },
            (Some(entries), None) => Self { entries: Some(entries.clone()) },
            (Some(a), Some(b)) => {
                let last_ast_index = a.slice.last().map_or(0, |it| it.id.ast_index() + 1) as u32;
                let items = a
                    .slice
                    .iter()
                    .cloned()
                    .chain(b.slice.iter().map(|it| {
                        let mut it = it.clone();
                        it.id.id = (it.id.ast_index() as u32 + last_ast_index)
                            | (it.id.cfg_attr_index().unwrap_or(0) as u32)
                                << AttrId::AST_INDEX_BITS;
                        it
                    }))
                    .collect::<Vec<_>>();
                Self { entries: Some(ThinArc::from_header_and_iter((), items.into_iter())) }
            }
        }
    }

    /// Processes `cfg_attr`s, returning the resulting semantic `Attrs`.
    // FIXME: This should return a different type, signaling it was filtered?
    pub fn filter(self, db: &dyn ExpandDatabase, krate: CrateId) -> RawAttrs {
        let has_cfg_attrs = self
            .iter()
            .any(|attr| attr.path.as_ident().map_or(false, |name| *name == sym::cfg_attr.clone()));
        if !has_cfg_attrs {
            return self;
        }

        let crate_graph = db.crate_graph();
        let new_attrs =
            self.iter()
                .flat_map(|attr| -> SmallVec<[_; 1]> {
                    let is_cfg_attr =
                        attr.path.as_ident().map_or(false, |name| *name == sym::cfg_attr.clone());
                    if !is_cfg_attr {
                        return smallvec![attr.clone()];
                    }

                    let subtree = match attr.token_tree_value() {
                        Some(it) => it,
                        _ => return smallvec![attr.clone()],
                    };

                    let (cfg, parts) = match parse_cfg_attr_input(subtree) {
                        Some(it) => it,
                        None => return smallvec![attr.clone()],
                    };
                    let index = attr.id;
                    let attrs = parts.enumerate().take(1 << AttrId::CFG_ATTR_BITS).filter_map(
                        |(idx, attr)| Attr::from_tt(db, attr, index.with_cfg_attr(idx)),
                    );

                    let cfg_options = &crate_graph[krate].cfg_options;
                    let cfg = Subtree { delimiter: subtree.delimiter, token_trees: Box::from(cfg) };
                    let cfg = CfgExpr::parse(&cfg);
                    if cfg_options.check(&cfg) == Some(false) {
                        smallvec![]
                    } else {
                        cov_mark::hit!(cfg_attr_active);

                        attrs.collect()
                    }
                })
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
    const CFG_ATTR_BITS: usize = 7;
    const AST_INDEX_MASK: usize = 0x00FF_FFFF;
    const AST_INDEX_BITS: usize = Self::AST_INDEX_MASK.count_ones() as usize;
    const CFG_ATTR_SET_BITS: u32 = 1 << 31;

    pub fn ast_index(&self) -> usize {
        self.id as usize & Self::AST_INDEX_MASK
    }

    pub fn cfg_attr_index(&self) -> Option<usize> {
        if self.id & Self::CFG_ATTR_SET_BITS == 0 {
            None
        } else {
            Some(self.id as usize >> Self::AST_INDEX_BITS)
        }
    }

    pub fn with_cfg_attr(self, idx: usize) -> AttrId {
        AttrId { id: self.id | (idx as u32) << Self::AST_INDEX_BITS | Self::CFG_ATTR_SET_BITS }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub id: AttrId,
    pub path: Interned<ModPath>,
    pub input: Option<Box<AttrInput>>,
    pub ctxt: SyntaxContextId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttrInput {
    /// `#[attr = "string"]`
    Literal(tt::Literal),
    /// `#[attr(subtree)]`
    TokenTree(Box<tt::Subtree>),
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
            Some(Box::new(AttrInput::TokenTree(Box::new(tree))))
        } else {
            None
        };
        Some(Attr { id, path, input, ctxt: span.ctx })
    }

    fn from_tt(db: &dyn ExpandDatabase, mut tt: &[tt::TokenTree], id: AttrId) -> Option<Attr> {
        if matches!(tt,
            [tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident { sym, .. })), ..]
            if *sym == sym::unsafe_
        ) {
            match tt.get(1) {
                Some(tt::TokenTree::Subtree(subtree)) => tt = &subtree.token_trees,
                _ => return None,
            }
        }
        let first = &tt.first()?;
        let ctxt = first.first_span().ctx;
        let path_end = tt
            .iter()
            .position(|tt| {
                !matches!(
                    tt,
                    tt::TokenTree::Leaf(
                        tt::Leaf::Punct(tt::Punct { char: ':' | '$', .. }) | tt::Leaf::Ident(_),
                    )
                )
            })
            .unwrap_or(tt.len());

        let (path, input) = tt.split_at(path_end);
        let path = Interned::new(ModPath::from_tt(db, path)?);

        let input = match input.first() {
            Some(tt::TokenTree::Subtree(tree)) => {
                Some(Box::new(AttrInput::TokenTree(Box::new(tree.clone()))))
            }
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: '=', .. }))) => {
                let input = match input.get(1) {
                    Some(tt::TokenTree::Leaf(tt::Leaf::Literal(lit))) => {
                        Some(Box::new(AttrInput::Literal(lit.clone())))
                    }
                    _ => None,
                };
                input
            }
            _ => None,
        };
        Some(Attr { id, path, input, ctxt })
    }

    pub fn path(&self) -> &ModPath {
        &self.path
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
            AttrInput::TokenTree(tt) => match &*tt.token_trees {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(ident))] => Some(ident),
                _ => None,
            },
            _ => None,
        }
    }

    /// #[path TokenTree]
    pub fn token_tree_value(&self) -> Option<&Subtree> {
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

        if args.delimiter.kind != DelimiterKind::Parenthesis {
            return None;
        }
        let paths = args
            .token_trees
            .split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))))
            .filter_map(move |tts| {
                let span = tts.first()?.first_span();
                Some((ModPath::from_tt(db, tts)?, span))
            });

        Some(paths)
    }

    pub fn cfg(&self) -> Option<CfgExpr> {
        if *self.path.as_ident()? == sym::cfg.clone() {
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
    unescape::unescape_unicode(s, unescape::Mode::Str, &mut |char_range, unescaped_char| match (
        unescaped_char,
        buf.capacity() == 0,
    ) {
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
    let inner_attrs = inner_attributes(owner.syntax()).into_iter().flatten();
    let outer_attrs =
        ast::AttrDocCommentIter::from_syntax_node(owner.syntax()).filter(|el| match el {
            Either::Left(attr) => attr.kind().is_outer(),
            Either::Right(comment) => comment.is_outer(),
        });
    outer_attrs.chain(inner_attrs).enumerate().map(|(id, attr)| (AttrId { id: id as u32 }, attr))
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
    subtree: &Subtree,
) -> Option<(&[tt::TokenTree], impl Iterator<Item = &[tt::TokenTree]>)> {
    let mut parts = subtree
        .token_trees
        .split(|tt| matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))));
    let cfg = parts.next()?;
    Some((cfg, parts.filter(|it| !it.is_empty())))
}
