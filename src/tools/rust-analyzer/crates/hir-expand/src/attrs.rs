//! Defines the basics of attributes lowering.
//!
//! The heart and soul of this module is [`expand_cfg_attr()`], alongside its sibling
//! [`expand_cfg_attr_with_doc_comments()`]. It is used to implement all attribute lowering
//! in r-a. Its basic job is to list attributes; however, attributes do not necessarily map
//! into [`ast::Attr`], because `cfg_attr` can map to zero, one, or more attributes
//! (`#[cfg_attr(predicate, attr1, attr2, ...)]`). To bridge this gap, this module defines
//! [`Meta`], which represents a desugared attribute. Various bits of r-a need different
//! things from [`Meta`], therefore it contains many parts. The basic idea is:
//!
//!  - There are three kinds of attributes, `path = value`, `path`, and `path(token_tree)`.
//!  - Most bits of rust-analyzer only need to deal with some paths. Therefore, we keep
//!    the path only if it has up to 2 segments, or one segment for `path = value`.
//!    We also only keep the value in `path = value` if it is a literal. However, we always
//!    save the all relevant ranges of attributes (the path range, and the full attribute range)
//!    for parts of r-a (e.g. name resolution) that need a faithful representation of the
//!    attribute.
//!
//! [`expand_cfg_attr()`] expands `cfg_attr`s as it goes (as its name implies), to list
//! all attributes.
//!
//! Another thing to note is that we need to be able to map an attribute back to a range
//! (for diagnostic purposes etc.). This is only ever needed for attributes that participate
//! in name resolution. An attribute is mapped back by its [`AttrId`], which is just an
//! index into the item tree attributes list. To minimize the risk of bugs, we have one
//! place (here) and one function ([`is_item_tree_filtered_attr()`]) that decides whether
//! an attribute participate in name resolution.

use std::{
    borrow::Cow, cell::OnceCell, convert::Infallible, fmt, iter::Peekable, ops::ControlFlow,
};

use ::tt::{TextRange, TextSize};
use arrayvec::ArrayVec;
use base_db::Crate;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use intern::{Interned, Symbol};
use mbe::{DelimiterKind, Punct};
use parser::T;
use smallvec::SmallVec;
use span::{RealSpanMap, Span, SyntaxContext};
use syntax::{
    AstNode, NodeOrToken, SyntaxNode, SyntaxToken,
    ast::{self, TokenTreeChildren},
    unescape,
};
use syntax_bridge::DocCommentDesugarMode;

use crate::{
    AstId,
    db::ExpandDatabase,
    mod_path::ModPath,
    span_map::SpanMapRef,
    tt::{self, TopSubtree},
};

#[derive(Debug)]
pub struct AttrPath {
    /// This can be empty if the path is not of 1 or 2 segments exactly.
    pub segments: ArrayVec<SyntaxToken, 2>,
    pub range: TextRange,
    // FIXME: This shouldn't be textual, `#[test]` needs name resolution.
    // And if textual, it shouldn't be here, it should be in hir-def/src/attrs.rs. But some macros
    // fully qualify `test` as `core::prelude::vX::test`, and this is more than 2 segments, so hir-def
    // attrs can't find it. But this will mean we have to push every up-to-4-segments path, which
    // may impact perf. So it was easier to just hack it here.
    pub is_test: bool,
}

impl AttrPath {
    #[inline]
    fn extract(path: &ast::Path) -> Self {
        let mut is_test = false;
        let segments = (|| {
            let mut segments = ArrayVec::new();
            let segment2 = path.segment()?.name_ref()?.syntax().first_token()?;
            if segment2.text() == "test" {
                // `#[test]` or `#[core::prelude::vX::test]`.
                is_test = true;
            }
            let segment1 = path.qualifier();
            if let Some(segment1) = segment1 {
                if segment1.qualifier().is_some() {
                    None
                } else {
                    let segment1 = segment1.segment()?.name_ref()?.syntax().first_token()?;
                    segments.push(segment1);
                    segments.push(segment2);
                    Some(segments)
                }
            } else {
                segments.push(segment2);
                Some(segments)
            }
        })();
        AttrPath {
            segments: segments.unwrap_or(ArrayVec::new()),
            range: path.syntax().text_range(),
            is_test,
        }
    }

    #[inline]
    pub fn is1(&self, segment: &str) -> bool {
        self.segments.len() == 1 && self.segments[0].text() == segment
    }
}

#[derive(Debug)]
pub enum Meta {
    /// `name` is `None` if not a single token. `value` is a literal or `None`.
    NamedKeyValue {
        path_range: TextRange,
        name: Option<SyntaxToken>,
        value: Option<SyntaxToken>,
    },
    TokenTree {
        path: AttrPath,
        tt: ast::TokenTree,
    },
    Path {
        path: AttrPath,
    },
}

impl Meta {
    #[inline]
    pub fn path_range(&self) -> TextRange {
        match self {
            Meta::NamedKeyValue { path_range, .. } => *path_range,
            Meta::TokenTree { path, .. } | Meta::Path { path } => path.range,
        }
    }

    fn extract(iter: &mut Peekable<TokenTreeChildren>) -> Option<(Self, TextSize)> {
        let mut start_offset = None;
        if let Some(NodeOrToken::Token(colon1)) = iter.peek()
            && colon1.kind() == T![:]
        {
            start_offset = Some(colon1.text_range().start());
            iter.next();
            iter.next_if(|it| it.as_token().is_some_and(|it| it.kind() == T![:]));
        }
        let first_segment = iter
            .next_if(|it| it.as_token().is_some_and(|it| it.kind().is_any_identifier()))?
            .into_token()?;
        let mut is_test = first_segment.text() == "test";
        let start_offset = start_offset.unwrap_or_else(|| first_segment.text_range().start());

        let mut segments_len = 1;
        let mut second_segment = None;
        let mut path_range = first_segment.text_range();
        while iter.peek().and_then(NodeOrToken::as_token).is_some_and(|it| it.kind() == T![:])
            && let _ = iter.next()
            && iter.peek().and_then(NodeOrToken::as_token).is_some_and(|it| it.kind() == T![:])
            && let _ = iter.next()
            && let Some(NodeOrToken::Token(segment)) = iter.peek()
            && segment.kind().is_any_identifier()
        {
            segments_len += 1;
            is_test = segment.text() == "test";
            second_segment = Some(segment.clone());
            path_range = TextRange::new(path_range.start(), segment.text_range().end());
            iter.next();
        }

        let segments = |first, second| {
            let mut segments = ArrayVec::new();
            if segments_len <= 2 {
                segments.push(first);
                if let Some(second) = second {
                    segments.push(second);
                }
            }
            segments
        };
        let meta = match iter.peek() {
            Some(NodeOrToken::Token(eq)) if eq.kind() == T![=] => {
                iter.next();
                let value = match iter.peek() {
                    Some(NodeOrToken::Token(token)) if token.kind().is_literal() => {
                        // No need to consume it, it will be consumed by `extract_and_eat_comma()`.
                        Some(token.clone())
                    }
                    _ => None,
                };
                let name = if second_segment.is_none() { Some(first_segment) } else { None };
                Meta::NamedKeyValue { path_range, name, value }
            }
            Some(NodeOrToken::Node(tt)) => Meta::TokenTree {
                path: AttrPath {
                    segments: segments(first_segment, second_segment),
                    range: path_range,
                    is_test,
                },
                tt: tt.clone(),
            },
            _ => Meta::Path {
                path: AttrPath {
                    segments: segments(first_segment, second_segment),
                    range: path_range,
                    is_test,
                },
            },
        };
        Some((meta, start_offset))
    }

    fn extract_possibly_unsafe(
        iter: &mut Peekable<TokenTreeChildren>,
        container: &ast::TokenTree,
    ) -> Option<(Self, TextRange)> {
        if iter.peek().is_some_and(|it| it.as_token().is_some_and(|it| it.kind() == T![unsafe])) {
            iter.next();
            let tt = iter.next()?.into_node()?;
            let result = Self::extract(&mut TokenTreeChildren::new(&tt).peekable()).map(
                |(meta, start_offset)| (meta, TextRange::new(start_offset, tt_end_offset(&tt))),
            );
            while iter.next().is_some_and(|it| it.as_token().is_none_or(|it| it.kind() != T![,])) {}
            result
        } else {
            Self::extract(iter).map(|(meta, start_offset)| {
                let end_offset = 'find_end_offset: {
                    for it in iter {
                        if let NodeOrToken::Token(it) = it
                            && it.kind() == T![,]
                        {
                            break 'find_end_offset it.text_range().start();
                        }
                    }
                    tt_end_offset(container)
                };
                (meta, TextRange::new(start_offset, end_offset))
            })
        }
    }
}

fn tt_end_offset(tt: &ast::TokenTree) -> TextSize {
    tt.syntax().last_token().unwrap().text_range().start()
}

/// The callback is passed a desugared form of the attribute ([`Meta`]), a [`SyntaxNode`] fully containing it
/// (note: it may not be the direct parent), the range within the [`SyntaxNode`] bounding the attribute,
/// and the outermost `ast::Attr`. Note that one node may map to multiple [`Meta`]s due to `cfg_attr`.
#[inline]
pub fn expand_cfg_attr<'a, BreakValue>(
    attrs: impl Iterator<Item = ast::Attr>,
    cfg_options: impl FnMut() -> &'a CfgOptions,
    mut callback: impl FnMut(Meta, &SyntaxNode, TextRange, &ast::Attr) -> ControlFlow<BreakValue>,
) -> Option<BreakValue> {
    expand_cfg_attr_with_doc_comments::<Infallible, _>(
        attrs.map(Either::Left),
        cfg_options,
        move |Either::Left((meta, container, range, top_attr))| {
            callback(meta, container, range, top_attr)
        },
    )
}

#[inline]
pub fn expand_cfg_attr_with_doc_comments<'a, DocComment, BreakValue>(
    mut attrs: impl Iterator<Item = Either<ast::Attr, DocComment>>,
    mut cfg_options: impl FnMut() -> &'a CfgOptions,
    mut callback: impl FnMut(
        Either<(Meta, &SyntaxNode, TextRange, &ast::Attr), DocComment>,
    ) -> ControlFlow<BreakValue>,
) -> Option<BreakValue> {
    let mut stack = SmallVec::<[_; 1]>::new();
    let result = attrs.try_for_each(|top_attr| {
        let top_attr = match top_attr {
            Either::Left(it) => it,
            Either::Right(comment) => return callback(Either::Right(comment)),
        };
        if let Some((attr_name, tt)) = top_attr.as_simple_call()
            && attr_name == "cfg_attr"
        {
            let mut tt_iter = TokenTreeChildren::new(&tt).peekable();
            let cfg = cfg::CfgExpr::parse_from_ast(&mut tt_iter);
            if cfg_options().check(&cfg) != Some(false) {
                stack.push((tt_iter, tt));
                while let Some((tt_iter, tt)) = stack.last_mut() {
                    let Some((attr, range)) = Meta::extract_possibly_unsafe(tt_iter, tt) else {
                        stack.pop();
                        continue;
                    };
                    if let Meta::TokenTree { path, tt: nested_tt } = &attr
                        && path.is1("cfg_attr")
                    {
                        let mut nested_tt_iter = TokenTreeChildren::new(nested_tt).peekable();
                        let cfg = cfg::CfgExpr::parse_from_ast(&mut nested_tt_iter);
                        if cfg_options().check(&cfg) != Some(false) {
                            stack.push((nested_tt_iter, nested_tt.clone()));
                        }
                    } else {
                        callback(Either::Left((attr, tt.syntax(), range, &top_attr)))?;
                    }
                }
            }
        } else if let Some(ast_meta) = top_attr.meta()
            && let Some(path) = ast_meta.path()
        {
            let path = AttrPath::extract(&path);
            let meta = if let Some(tt) = ast_meta.token_tree() {
                Meta::TokenTree { path, tt }
            } else if let Some(value) = ast_meta.expr() {
                let value =
                    if let ast::Expr::Literal(value) = value { Some(value.token()) } else { None };
                let name =
                    if path.segments.len() == 1 { Some(path.segments[0].clone()) } else { None };
                Meta::NamedKeyValue { name, value, path_range: path.range }
            } else {
                Meta::Path { path }
            };
            callback(Either::Left((
                meta,
                ast_meta.syntax(),
                ast_meta.syntax().text_range(),
                &top_attr,
            )))?;
        }
        ControlFlow::Continue(())
    });
    result.break_value()
}

#[inline]
pub(crate) fn is_item_tree_filtered_attr(name: &str) -> bool {
    matches!(
        name,
        "doc"
            | "stable"
            | "unstable"
            | "target_feature"
            | "allow"
            | "expect"
            | "warn"
            | "deny"
            | "forbid"
            | "repr"
            | "inline"
            | "track_caller"
            | "must_use"
    )
}

/// This collects attributes exactly as the item tree needs them. This is used for the item tree,
/// as well as for resolving [`AttrId`]s.
pub fn collect_item_tree_attrs<'a, BreakValue>(
    owner: &dyn ast::HasAttrs,
    cfg_options: impl Fn() -> &'a CfgOptions,
    mut on_attr: impl FnMut(Meta, &SyntaxNode, &ast::Attr, TextRange) -> ControlFlow<BreakValue>,
) -> Option<Either<BreakValue, CfgExpr>> {
    let attrs = ast::attrs_including_inner(owner);
    expand_cfg_attr(
        attrs,
        || cfg_options(),
        |attr, container, range, top_attr| {
            // We filter builtin attributes that we don't need for nameres, because this saves memory.
            // I only put the most common attributes, but if some attribute becomes common feel free to add it.
            // Notice, however: for an attribute to be filtered out, it *must* not be shadowable with a macro!
            let filter = match &attr {
                Meta::NamedKeyValue { name: Some(name), .. } => {
                    is_item_tree_filtered_attr(name.text())
                }
                Meta::TokenTree { path, tt } if path.segments.len() == 1 => {
                    let name = path.segments[0].text();
                    if name == "cfg" {
                        let cfg =
                            CfgExpr::parse_from_ast(&mut TokenTreeChildren::new(tt).peekable());
                        if cfg_options().check(&cfg) == Some(false) {
                            return ControlFlow::Break(Either::Right(cfg));
                        }
                        true
                    } else {
                        is_item_tree_filtered_attr(name)
                    }
                }
                Meta::Path { path } => {
                    path.segments.len() == 1 && is_item_tree_filtered_attr(path.segments[0].text())
                }
                _ => false,
            };
            if !filter && let ControlFlow::Break(v) = on_attr(attr, container, top_attr, range) {
                return ControlFlow::Break(Either::Left(v));
            }
            ControlFlow::Continue(())
        },
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
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
    ) -> Option<impl Iterator<Item = (ModPath, Span, tt::TokenTreesView<'a>)> + 'a> {
        let args = self.token_tree_value()?;

        if args.top_subtree().delimiter.kind != DelimiterKind::Parenthesis {
            return None;
        }
        Some(parse_path_comma_token_tree(db, args))
    }
}

fn parse_path_comma_token_tree<'a>(
    db: &'a dyn ExpandDatabase,
    args: &'a tt::TopSubtree,
) -> impl Iterator<Item = (ModPath, Span, tt::TokenTreesView<'a>)> {
    args.token_trees()
        .split(|tt| matches!(tt, tt::TtElement::Leaf(tt::Leaf::Punct(Punct { char: ',', .. }))))
        .filter_map(move |tts| {
            let span = tts.flat_tokens().first()?.first_span();
            Some((ModPath::from_tt(db, tts)?, span, tts))
        })
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

/// This is an index of an attribute *that always points to the item tree attributes*.
///
/// Outer attributes are counted first, then inner attributes. This does not support
/// out-of-line modules, which may have attributes spread across 2 files!
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttrId {
    id: u32,
}

impl AttrId {
    #[inline]
    pub fn from_item_tree_index(id: u32) -> Self {
        Self { id }
    }

    #[inline]
    pub fn item_tree_index(self) -> u32 {
        self.id
    }

    /// Returns the containing `ast::Attr` (note that it may contain other attributes as well due
    /// to `cfg_attr`), a `SyntaxNode` guaranteed to contain the attribute, the full range of the
    /// attribute, and its desugared [`Meta`].
    pub fn find_attr_range<N: ast::HasAttrs>(
        self,
        db: &dyn ExpandDatabase,
        krate: Crate,
        owner: AstId<N>,
    ) -> (ast::Attr, SyntaxNode, TextRange, Meta) {
        self.find_attr_range_with_source(db, krate, &owner.to_node(db))
    }

    /// Returns the containing `ast::Attr` (note that it may contain other attributes as well due
    /// to `cfg_attr`), a `SyntaxNode` guaranteed to contain the attribute, the full range of the
    /// attribute, and its desugared [`Meta`].
    pub fn find_attr_range_with_source(
        self,
        db: &dyn ExpandDatabase,
        krate: Crate,
        owner: &dyn ast::HasAttrs,
    ) -> (ast::Attr, SyntaxNode, TextRange, Meta) {
        let cfg_options = OnceCell::new();
        let mut index = 0;
        let result = collect_item_tree_attrs(
            owner,
            || cfg_options.get_or_init(|| krate.cfg_options(db)),
            |meta, container, top_attr, range| {
                if index == self.id {
                    return ControlFlow::Break((top_attr.clone(), container.clone(), range, meta));
                }
                index += 1;
                ControlFlow::Continue(())
            },
        );
        match result {
            Some(Either::Left(it)) => it,
            _ => {
                panic!("used an incorrect `AttrId`; crate={krate:?}, attr_id={self:?}");
            }
        }
    }

    pub fn find_derive_range(
        self,
        db: &dyn ExpandDatabase,
        krate: Crate,
        owner: AstId<ast::Adt>,
        derive_index: u32,
    ) -> TextRange {
        let (_, _, derive_attr_range, derive_attr) = self.find_attr_range(db, krate, owner);
        let Meta::TokenTree { tt, .. } = derive_attr else {
            return derive_attr_range;
        };
        // Fake the span map, as we don't really need spans here, just the offsets of the node in the file.
        let span_map = RealSpanMap::absolute(span::EditionedFileId::current_edition(
            span::FileId::from_raw(0),
        ));
        let tt = syntax_bridge::syntax_node_to_token_tree(
            tt.syntax(),
            SpanMapRef::RealSpanMap(&span_map),
            span_map.span_for_range(tt.syntax().text_range()),
            DocCommentDesugarMode::ProcMacro,
        );
        let Some((_, _, derive_tts)) =
            parse_path_comma_token_tree(db, &tt).nth(derive_index as usize)
        else {
            return derive_attr_range;
        };
        let (Some(first_tt), Some(last_tt)) =
            (derive_tts.flat_tokens().first(), derive_tts.flat_tokens().last())
        else {
            return derive_attr_range;
        };
        let start = first_tt.first_span().range.start();
        let end = match last_tt {
            tt::TokenTree::Leaf(it) => it.span().range.end(),
            tt::TokenTree::Subtree(it) => it.delimiter.close.range.end(),
        };
        TextRange::new(start, end)
    }
}
