//! Defines the basics of attributes lowering.
//!
//! The heart and soul of this module is [`expand_cfg_attr()`], alongside its sibling
//! [`expand_cfg_attr_with_doc_comments()`]. It is used to implement all attribute lowering
//! in r-a. Its basic job is to list attributes; however, attributes do not necessarily map
//! into [`ast::Attr`], because `cfg_attr` can map to zero, one, or more attributes
//! (`#[cfg_attr(predicate, attr1, attr2, ...)]`). [`expand_cfg_attr()`] expands `cfg_attr`s
//! as it goes (as its name implies), to list all attributes.
//!
//! Another thing to note is that we need to be able to map an attribute back to a range
//! (for diagnostic purposes etc.). This is only ever needed for attributes that participate
//! in name resolution. An attribute is mapped back by its [`AttrId`], which is just an
//! index into the item tree attributes list. To minimize the risk of bugs, we have one
//! place (here) and one function ([`is_item_tree_filtered_attr()`]) that decides whether
//! an attribute participate in name resolution.

use std::{borrow::Cow, cell::OnceCell, convert::Infallible, fmt, ops::ControlFlow};

use ::tt::TextRange;
use base_db::Crate;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use intern::Interned;
use itertools::Itertools;
use mbe::{DelimiterKind, Punct};
use smallvec::SmallVec;
use span::{RealSpanMap, Span, SyntaxContext};
use syntax::{AstNode, SmolStr, ast, unescape};
use syntax_bridge::DocCommentDesugarMode;

use crate::{
    AstId,
    db::ExpandDatabase,
    mod_path::ModPath,
    span_map::SpanMapRef,
    tt::{self, TopSubtree},
};

pub trait AstPathExt {
    fn is1(&self, segment: &str) -> bool;

    fn as_one_segment(&self) -> Option<SmolStr>;

    fn as_up_to_two_segment(&self) -> Option<(SmolStr, Option<SmolStr>)>;
}

impl AstPathExt for ast::Path {
    fn is1(&self, segment: &str) -> bool {
        self.as_one_segment().is_some_and(|it| it == segment)
    }

    fn as_one_segment(&self) -> Option<SmolStr> {
        Some(self.as_single_name_ref()?.text().into())
    }

    fn as_up_to_two_segment(&self) -> Option<(SmolStr, Option<SmolStr>)> {
        let parent = self.qualifier().as_one_segment();
        let this = self.segment()?.name_ref()?.text().into();
        if let Some(parent) = parent { Some((parent, Some(this))) } else { Some((this, None)) }
    }
}

impl AstPathExt for Option<ast::Path> {
    fn is1(&self, segment: &str) -> bool {
        self.as_ref().is_some_and(|it| it.is1(segment))
    }

    fn as_one_segment(&self) -> Option<SmolStr> {
        self.as_ref().and_then(|it| it.as_one_segment())
    }

    fn as_up_to_two_segment(&self) -> Option<(SmolStr, Option<SmolStr>)> {
        self.as_ref().and_then(|it| it.as_up_to_two_segment())
    }
}

pub trait AstKeyValueMetaExt {
    fn value_string(&self) -> Option<SmolStr>;
}

impl AstKeyValueMetaExt for ast::KeyValueMeta {
    fn value_string(&self) -> Option<SmolStr> {
        if let Some(ast::Expr::Literal(value)) = self.expr()
            && let ast::LiteralKind::String(value) = value.kind()
            && let Ok(value) = value.value()
        {
            Some((*value).into())
        } else {
            None
        }
    }
}

/// The callback is passed the attribute and the outermost `ast::Attr`.
/// Note that one node may map to multiple [`ast::Meta`]s due to `cfg_attr`.
///
/// `unsafe(attr)` are passed the inner attribute for now.
#[inline]
pub fn expand_cfg_attr<'a, BreakValue>(
    attrs: impl Iterator<Item = ast::Attr>,
    cfg_options: impl FnMut() -> &'a CfgOptions,
    mut callback: impl FnMut(ast::Meta, ast::Attr) -> ControlFlow<BreakValue>,
) -> Option<BreakValue> {
    expand_cfg_attr_with_doc_comments::<Infallible, _>(
        attrs.map(Either::Left),
        cfg_options,
        move |Either::Left((meta, top_attr))| callback(meta, top_attr),
    )
}

#[inline]
pub fn expand_cfg_attr_with_doc_comments<'a, DocComment, BreakValue>(
    mut attrs: impl Iterator<Item = Either<ast::Attr, DocComment>>,
    mut cfg_options: impl FnMut() -> &'a CfgOptions,
    mut callback: impl FnMut(Either<(ast::Meta, ast::Attr), DocComment>) -> ControlFlow<BreakValue>,
) -> Option<BreakValue> {
    let mut stack = SmallVec::<[_; 1]>::new();
    loop {
        let (mut meta, top_attr) = if let Some(it) = stack.pop() {
            it
        } else {
            let attr = attrs.next()?;
            match attr {
                Either::Left(attr) => {
                    let Some(meta) = attr.meta() else { continue };
                    stack.push((meta, attr));
                }
                Either::Right(doc_comment) => {
                    if let ControlFlow::Break(break_value) = callback(Either::Right(doc_comment)) {
                        return Some(break_value);
                    }
                }
            }
            continue;
        };

        while let ast::Meta::UnsafeMeta(unsafe_meta) = &meta {
            let Some(inner) = unsafe_meta.meta() else { continue };
            meta = inner;
        }

        if let ast::Meta::CfgAttrMeta(meta) = meta {
            let Some(cfg_predicate) = meta.cfg_predicate() else { continue };
            let cfg_predicate = CfgExpr::parse_from_ast(cfg_predicate);
            if cfg_options().check(&cfg_predicate) != Some(false) {
                let prev_stack_len = stack.len();
                stack.extend(meta.metas().map(|meta| (meta, top_attr.clone())));
                stack[prev_stack_len..].reverse();
            }
        } else {
            if let ControlFlow::Break(break_value) = callback(Either::Left((meta, top_attr))) {
                return Some(break_value);
            }
        }
    }
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
    mut on_attr: impl FnMut(ast::Meta, ast::Attr) -> ControlFlow<BreakValue>,
) -> Option<Either<BreakValue, CfgExpr>> {
    let attrs = ast::attrs_including_inner(owner);
    expand_cfg_attr(
        attrs,
        || cfg_options(),
        |attr, top_attr| {
            // We filter builtin attributes that we don't need for nameres, because this saves memory.
            // I only put the most common attributes, but if some attribute becomes common feel free to add it.
            // Notice, however: for an attribute to be filtered out, it *must* not be shadowable with a macro!
            let filter = match &attr {
                ast::Meta::CfgMeta(attr) => {
                    let Some(cfg_predicate) = attr.cfg_predicate() else {
                        return ControlFlow::Continue(());
                    };
                    let cfg = CfgExpr::parse_from_ast(cfg_predicate);
                    if cfg_options().check(&cfg) == Some(false) {
                        return ControlFlow::Break(Either::Right(cfg));
                    }
                    true
                }
                _ => attr
                    .path()
                    .and_then(|path| path.as_one_segment())
                    .is_some_and(|segment| is_item_tree_filtered_attr(&segment)),
            };
            if !filter && let ControlFlow::Break(v) = on_attr(attr, top_attr) {
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
    pub fn string_value(&self) -> Option<&str> {
        match self.input.as_deref()? {
            AttrInput::Literal(
                lit @ tt::Literal { kind: tt::LitKind::Str | tt::LitKind::StrRaw(_), .. },
            ) => Some(lit.text()),
            _ => None,
        }
    }

    /// #[path = "string"]
    pub fn string_value_with_span(&self) -> Option<(&str, span::Span)> {
        match self.input.as_deref()? {
            AttrInput::Literal(
                lit @ tt::Literal { kind: tt::LitKind::Str | tt::LitKind::StrRaw(_), span, .. },
            ) => Some((lit.text(), *span)),
            _ => None,
        }
    }

    pub fn string_value_unescape(&self) -> Option<Cow<'_, str>> {
        match self.input.as_deref()? {
            AttrInput::Literal(lit @ tt::Literal { kind: tt::LitKind::StrRaw(_), .. }) => {
                Some(Cow::Borrowed(lit.text()))
            }
            AttrInput::Literal(lit @ tt::Literal { kind: tt::LitKind::Str, .. }) => {
                unescape(lit.text())
            }
            _ => None,
        }
    }

    /// #[path(ident)]
    pub fn single_ident_value(&self) -> Option<tt::Ident> {
        match self.input.as_deref()? {
            AttrInput::TokenTree(tt) => match tt.token_trees().iter().collect_array() {
                Some([tt::TtElement::Leaf(tt::Leaf::Ident(ident))]) => Some(ident),
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
            let span = tts.first_span()?;
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
    /// to `cfg_attr`) and its [`ast::Meta`].
    pub fn find_attr_range<N: ast::HasAttrs>(
        self,
        db: &dyn ExpandDatabase,
        krate: Crate,
        owner: AstId<N>,
    ) -> (ast::Attr, ast::Meta) {
        self.find_attr_range_with_source(db, krate, &owner.to_node(db))
    }

    /// Returns the containing `ast::Attr` (note that it may contain other attributes as well due
    /// to `cfg_attr`) and its [`ast::Meta`].
    pub fn find_attr_range_with_source(
        self,
        db: &dyn ExpandDatabase,
        krate: Crate,
        owner: &dyn ast::HasAttrs,
    ) -> (ast::Attr, ast::Meta) {
        let cfg_options = OnceCell::new();
        let mut index = 0;
        let result = collect_item_tree_attrs(
            owner,
            || cfg_options.get_or_init(|| krate.cfg_options(db)),
            |meta, top_attr| {
                if index == self.id {
                    return ControlFlow::Break((top_attr, meta));
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
        let (_, derive_attr) = self.find_attr_range(db, krate, owner);
        let ast::Meta::TokenTreeMeta(derive_attr) = derive_attr else {
            return derive_attr.syntax().text_range();
        };
        let Some(tt) = derive_attr.token_tree() else {
            return derive_attr.syntax().text_range();
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
            return derive_attr.syntax().text_range();
        };
        let (Some(first_span), Some(last_span)) = (derive_tts.first_span(), derive_tts.last_span())
        else {
            return derive_attr.syntax().text_range();
        };
        let start = first_span.range.start();
        let end = last_span.range.end();
        TextRange::new(start, end)
    }
}
