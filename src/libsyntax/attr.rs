// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Functions dealing with attributes and meta items

pub use self::StabilityLevel::*;
pub use self::ReprAttr::*;
pub use self::IntType::*;

use ast;
use ast::{AttrId, Attribute, Name, Ident};
use ast::{MetaItem, MetaItemKind, NestedMetaItem, NestedMetaItemKind};
use ast::{Lit, LitKind, Expr, ExprKind, Item, Local, Stmt, StmtKind};
use codemap::{Spanned, respan, dummy_spanned};
use syntax_pos::{Span, DUMMY_SP};
use errors::Handler;
use feature_gate::{Features, GatedCfg};
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::parser::Parser;
use parse::{self, ParseSess, PResult};
use parse::token::{self, Token};
use ptr::P;
use symbol::Symbol;
use tokenstream::{TokenStream, TokenTree, Delimited};
use util::ThinVec;

use std::cell::RefCell;
use std::iter;

thread_local! {
    static USED_ATTRS: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    static KNOWN_ATTRS: RefCell<Vec<u64>> = RefCell::new(Vec::new());
}

enum AttrError {
    MultipleItem(Name),
    UnknownMetaItem(Name),
    MissingSince,
    MissingFeature,
    MultipleStabilityLevels,
    UnsupportedLiteral
}

fn handle_errors(diag: &Handler, span: Span, error: AttrError) {
    match error {
        AttrError::MultipleItem(item) => span_err!(diag, span, E0538,
                                                   "multiple '{}' items", item),
        AttrError::UnknownMetaItem(item) => span_err!(diag, span, E0541,
                                                      "unknown meta item '{}'", item),
        AttrError::MissingSince => span_err!(diag, span, E0542, "missing 'since'"),
        AttrError::MissingFeature => span_err!(diag, span, E0546, "missing 'feature'"),
        AttrError::MultipleStabilityLevels => span_err!(diag, span, E0544,
                                                        "multiple stability levels"),
        AttrError::UnsupportedLiteral => span_err!(diag, span, E0565, "unsupported literal"),
    }
}

pub fn mark_used(attr: &Attribute) {
    debug!("Marking {:?} as used.", attr);
    let AttrId(id) = attr.id;
    USED_ATTRS.with(|slot| {
        let idx = (id / 64) as usize;
        let shift = id % 64;
        if slot.borrow().len() <= idx {
            slot.borrow_mut().resize(idx + 1, 0);
        }
        slot.borrow_mut()[idx] |= 1 << shift;
    });
}

pub fn is_used(attr: &Attribute) -> bool {
    let AttrId(id) = attr.id;
    USED_ATTRS.with(|slot| {
        let idx = (id / 64) as usize;
        let shift = id % 64;
        slot.borrow().get(idx).map(|bits| bits & (1 << shift) != 0)
            .unwrap_or(false)
    })
}

pub fn mark_known(attr: &Attribute) {
    debug!("Marking {:?} as known.", attr);
    let AttrId(id) = attr.id;
    KNOWN_ATTRS.with(|slot| {
        let idx = (id / 64) as usize;
        let shift = id % 64;
        if slot.borrow().len() <= idx {
            slot.borrow_mut().resize(idx + 1, 0);
        }
        slot.borrow_mut()[idx] |= 1 << shift;
    });
}

pub fn is_known(attr: &Attribute) -> bool {
    let AttrId(id) = attr.id;
    KNOWN_ATTRS.with(|slot| {
        let idx = (id / 64) as usize;
        let shift = id % 64;
        slot.borrow().get(idx).map(|bits| bits & (1 << shift) != 0)
            .unwrap_or(false)
    })
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

impl Attribute {
    pub fn check_name(&self, name: &str) -> bool {
        let matches = self.path == name;
        if matches {
            mark_used(self);
        }
        matches
    }

    pub fn name(&self) -> Option<Name> {
        match self.path.segments.len() {
            1 => Some(self.path.segments[0].identifier.name),
            _ => None,
        }
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
        self.name
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
}

impl Attribute {
    /// Extract the MetaItem from inside this Attribute.
    pub fn meta(&self) -> Option<MetaItem> {
        let mut tokens = self.tokens.trees().peekable();
        Some(MetaItem {
            name: match self.path.segments.len() {
                1 => self.path.segments[0].identifier.name,
                _ => return None,
            },
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
        if self.path.segments.len() > 1 {
            sess.span_diagnostic.span_err(self.path.span, "expected ident, found path");
        }

        Ok(MetaItem {
            name: self.path.segments.last().unwrap().identifier.name,
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
                Symbol::intern("doc"),
                Symbol::intern(&strip_doc_comment_decoration(&comment.as_str())));
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

pub fn mk_name_value_item_str(name: Name, value: Symbol) -> MetaItem {
    let value_lit = dummy_spanned(LitKind::Str(value, ast::StrStyle::Cooked));
    mk_spanned_name_value_item(DUMMY_SP, name, value_lit)
}

pub fn mk_name_value_item(name: Name, value: ast::Lit) -> MetaItem {
    mk_spanned_name_value_item(DUMMY_SP, name, value)
}

pub fn mk_list_item(name: Name, items: Vec<NestedMetaItem>) -> MetaItem {
    mk_spanned_list_item(DUMMY_SP, name, items)
}

pub fn mk_list_word_item(name: Name) -> ast::NestedMetaItem {
    dummy_spanned(NestedMetaItemKind::MetaItem(mk_spanned_word_item(DUMMY_SP, name)))
}

pub fn mk_word_item(name: Name) -> MetaItem {
    mk_spanned_word_item(DUMMY_SP, name)
}

pub fn mk_spanned_name_value_item(sp: Span, name: Name, value: ast::Lit) -> MetaItem {
    MetaItem { span: sp, name: name, node: MetaItemKind::NameValue(value) }
}

pub fn mk_spanned_list_item(sp: Span, name: Name, items: Vec<NestedMetaItem>) -> MetaItem {
    MetaItem { span: sp, name: name, node: MetaItemKind::List(items) }
}

pub fn mk_spanned_word_item(sp: Span, name: Name) -> MetaItem {
    MetaItem { span: sp, name: name, node: MetaItemKind::Word }
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
        path: ast::Path::from_ident(item.span, ast::Ident::with_empty_ctxt(item.name)),
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
        path: ast::Path::from_ident(item.span, ast::Ident::with_empty_ctxt(item.name)),
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
        path: ast::Path::from_ident(span, ast::Ident::from_str("doc")),
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

/// Check if `attrs` contains an attribute like `#![feature(feature_name)]`.
/// This will not perform any "sanity checks" on the form of the attributes.
pub fn contains_feature_attr(attrs: &[Attribute], feature_name: &str) -> bool {
    attrs.iter().any(|item| {
        item.check_name("feature") &&
        item.meta_item_list().map(|list| {
            list.iter().any(|mi| {
                mi.word().map(|w| w.name() == feature_name)
                         .unwrap_or(false)
            })
        }).unwrap_or(false)
    })
}

/* Higher-level applications */

pub fn find_crate_name(attrs: &[Attribute]) -> Option<Symbol> {
    first_attr_value_str_by_name(attrs, "crate_name")
}

#[derive(Copy, Clone, PartialEq)]
pub enum InlineAttr {
    None,
    Hint,
    Always,
    Never,
}

/// Determine what `#[inline]` attribute is present in `attrs`, if any.
pub fn find_inline_attr(diagnostic: Option<&Handler>, attrs: &[Attribute]) -> InlineAttr {
    attrs.iter().fold(InlineAttr::None, |ia, attr| {
        if attr.path != "inline" {
            return ia;
        }
        let meta = match attr.meta() {
            Some(meta) => meta.node,
            None => return ia,
        };
        match meta {
            MetaItemKind::Word => {
                mark_used(attr);
                InlineAttr::Hint
            }
            MetaItemKind::List(ref items) => {
                mark_used(attr);
                if items.len() != 1 {
                    diagnostic.map(|d|{ span_err!(d, attr.span, E0534, "expected one argument"); });
                    InlineAttr::None
                } else if list_contains_name(&items[..], "always") {
                    InlineAttr::Always
                } else if list_contains_name(&items[..], "never") {
                    InlineAttr::Never
                } else {
                    diagnostic.map(|d| {
                        span_err!(d, items[0].span, E0535, "invalid argument");
                    });

                    InlineAttr::None
                }
            }
            _ => ia,
        }
    })
}

/// True if `#[inline]` or `#[inline(always)]` is present in `attrs`.
pub fn requests_inline(attrs: &[Attribute]) -> bool {
    match find_inline_attr(None, attrs) {
        InlineAttr::Hint | InlineAttr::Always => true,
        InlineAttr::None | InlineAttr::Never => false,
    }
}

/// Tests if a cfg-pattern matches the cfg set
pub fn cfg_matches(cfg: &ast::MetaItem, sess: &ParseSess, features: Option<&Features>) -> bool {
    eval_condition(cfg, sess, &mut |cfg| {
        if let (Some(feats), Some(gated_cfg)) = (features, GatedCfg::gate(cfg)) {
            gated_cfg.check_and_emit(sess, feats);
        }
        sess.config.contains(&(cfg.name(), cfg.value_str()))
    })
}

/// Evaluate a cfg-like condition (with `any` and `all`), using `eval` to
/// evaluate individual items.
pub fn eval_condition<F>(cfg: &ast::MetaItem, sess: &ParseSess, eval: &mut F)
                         -> bool
    where F: FnMut(&ast::MetaItem) -> bool
{
    match cfg.node {
        ast::MetaItemKind::List(ref mis) => {
            for mi in mis.iter() {
                if !mi.is_meta_item() {
                    handle_errors(&sess.span_diagnostic, mi.span, AttrError::UnsupportedLiteral);
                    return false;
                }
            }

            // The unwraps below may look dangerous, but we've already asserted
            // that they won't fail with the loop above.
            match &*cfg.name.as_str() {
                "any" => mis.iter().any(|mi| {
                    eval_condition(mi.meta_item().unwrap(), sess, eval)
                }),
                "all" => mis.iter().all(|mi| {
                    eval_condition(mi.meta_item().unwrap(), sess, eval)
                }),
                "not" => {
                    if mis.len() != 1 {
                        span_err!(sess.span_diagnostic, cfg.span, E0536, "expected 1 cfg-pattern");
                        return false;
                    }

                    !eval_condition(mis[0].meta_item().unwrap(), sess, eval)
                },
                p => {
                    span_err!(sess.span_diagnostic, cfg.span, E0537, "invalid predicate `{}`", p);
                    false
                }
            }
        },
        ast::MetaItemKind::Word | ast::MetaItemKind::NameValue(..) => {
            eval(cfg)
        }
    }
}

/// Represents the #[stable], #[unstable], #[rustc_{deprecated,const_unstable}] attributes.
#[derive(RustcEncodable, RustcDecodable, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    pub rustc_depr: Option<RustcDeprecation>,
    pub rustc_const_unstable: Option<RustcConstUnstable>,
}

/// The available stability levels.
#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub enum StabilityLevel {
    // Reason for the current stability level and the relevant rust-lang issue
    Unstable { reason: Option<Symbol>, issue: u32 },
    Stable { since: Symbol },
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub struct RustcDeprecation {
    pub since: Symbol,
    pub reason: Symbol,
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub struct RustcConstUnstable {
    pub feature: Symbol,
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub struct Deprecation {
    pub since: Option<Symbol>,
    pub note: Option<Symbol>,
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool { if let Unstable {..} = *self { true } else { false }}
    pub fn is_stable(&self) -> bool { if let Stable {..} = *self { true } else { false }}
}

fn find_stability_generic<'a, I>(diagnostic: &Handler,
                                 attrs_iter: I,
                                 item_sp: Span)
                                 -> Option<Stability>
    where I: Iterator<Item = &'a Attribute>
{
    let mut stab: Option<Stability> = None;
    let mut rustc_depr: Option<RustcDeprecation> = None;
    let mut rustc_const_unstable: Option<RustcConstUnstable> = None;

    'outer: for attr in attrs_iter {
        if ![
            "rustc_deprecated",
            "rustc_const_unstable",
            "unstable",
            "stable",
        ].iter().any(|&s| attr.path == s) {
            continue // not a stability level
        }

        mark_used(attr);

        let meta = attr.meta();
        if let Some(MetaItem { node: MetaItemKind::List(ref metas), .. }) = meta {
            let meta = meta.as_ref().unwrap();
            let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                if item.is_some() {
                    handle_errors(diagnostic, meta.span, AttrError::MultipleItem(meta.name()));
                    return false
                }
                if let Some(v) = meta.value_str() {
                    *item = Some(v);
                    true
                } else {
                    span_err!(diagnostic, meta.span, E0539, "incorrect meta item");
                    false
                }
            };

            macro_rules! get_meta {
                ($($name:ident),+) => {
                    $(
                        let mut $name = None;
                    )+
                    for meta in metas {
                        if let Some(mi) = meta.meta_item() {
                            match &*mi.name().as_str() {
                                $(
                                    stringify!($name)
                                        => if !get(mi, &mut $name) { continue 'outer },
                                )+
                                _ => {
                                    handle_errors(diagnostic, mi.span,
                                                  AttrError::UnknownMetaItem(mi.name()));
                                    continue 'outer
                                }
                            }
                        } else {
                            handle_errors(diagnostic, meta.span, AttrError::UnsupportedLiteral);
                            continue 'outer
                        }
                    }
                }
            }

            match &*meta.name.as_str() {
                "rustc_deprecated" => {
                    if rustc_depr.is_some() {
                        span_err!(diagnostic, item_sp, E0540,
                                  "multiple rustc_deprecated attributes");
                        continue 'outer
                    }

                    get_meta!(since, reason);

                    match (since, reason) {
                        (Some(since), Some(reason)) => {
                            rustc_depr = Some(RustcDeprecation {
                                since,
                                reason,
                            })
                        }
                        (None, _) => {
                            handle_errors(diagnostic, attr.span(), AttrError::MissingSince);
                            continue
                        }
                        _ => {
                            span_err!(diagnostic, attr.span(), E0543, "missing 'reason'");
                            continue
                        }
                    }
                }
                "rustc_const_unstable" => {
                    if rustc_const_unstable.is_some() {
                        span_err!(diagnostic, item_sp, E0553,
                                  "multiple rustc_const_unstable attributes");
                        continue 'outer
                    }

                    get_meta!(feature);
                    if let Some(feature) = feature {
                        rustc_const_unstable = Some(RustcConstUnstable {
                            feature
                        });
                    } else {
                        span_err!(diagnostic, attr.span(), E0629, "missing 'feature'");
                        continue
                    }
                }
                "unstable" => {
                    if stab.is_some() {
                        handle_errors(diagnostic, attr.span(), AttrError::MultipleStabilityLevels);
                        break
                    }

                    let mut feature = None;
                    let mut reason = None;
                    let mut issue = None;
                    for meta in metas {
                        if let Some(mi) = meta.meta_item() {
                            match &*mi.name().as_str() {
                                "feature" => if !get(mi, &mut feature) { continue 'outer },
                                "reason" => if !get(mi, &mut reason) { continue 'outer },
                                "issue" => if !get(mi, &mut issue) { continue 'outer },
                                _ => {
                                    handle_errors(diagnostic, meta.span,
                                                  AttrError::UnknownMetaItem(mi.name()));
                                    continue 'outer
                                }
                            }
                        } else {
                            handle_errors(diagnostic, meta.span, AttrError::UnsupportedLiteral);
                            continue 'outer
                        }
                    }

                    match (feature, reason, issue) {
                        (Some(feature), reason, Some(issue)) => {
                            stab = Some(Stability {
                                level: Unstable {
                                    reason,
                                    issue: {
                                        if let Ok(issue) = issue.as_str().parse() {
                                            issue
                                        } else {
                                            span_err!(diagnostic, attr.span(), E0545,
                                                      "incorrect 'issue'");
                                            continue
                                        }
                                    }
                                },
                                feature,
                                rustc_depr: None,
                                rustc_const_unstable: None,
                            })
                        }
                        (None, _, _) => {
                            handle_errors(diagnostic, attr.span(), AttrError::MissingFeature);
                            continue
                        }
                        _ => {
                            span_err!(diagnostic, attr.span(), E0547, "missing 'issue'");
                            continue
                        }
                    }
                }
                "stable" => {
                    if stab.is_some() {
                        handle_errors(diagnostic, attr.span(), AttrError::MultipleStabilityLevels);
                        break
                    }

                    let mut feature = None;
                    let mut since = None;
                    for meta in metas {
                        if let NestedMetaItemKind::MetaItem(ref mi) = meta.node {
                            match &*mi.name().as_str() {
                                "feature" => if !get(mi, &mut feature) { continue 'outer },
                                "since" => if !get(mi, &mut since) { continue 'outer },
                                _ => {
                                    handle_errors(diagnostic, meta.span,
                                                  AttrError::UnknownMetaItem(mi.name()));
                                    continue 'outer
                                }
                            }
                        } else {
                            handle_errors(diagnostic, meta.span, AttrError::UnsupportedLiteral);
                            continue 'outer
                        }
                    }

                    match (feature, since) {
                        (Some(feature), Some(since)) => {
                            stab = Some(Stability {
                                level: Stable {
                                    since,
                                },
                                feature,
                                rustc_depr: None,
                                rustc_const_unstable: None,
                            })
                        }
                        (None, _) => {
                            handle_errors(diagnostic, attr.span(), AttrError::MissingFeature);
                            continue
                        }
                        _ => {
                            handle_errors(diagnostic, attr.span(), AttrError::MissingSince);
                            continue
                        }
                    }
                }
                _ => unreachable!()
            }
        } else {
            span_err!(diagnostic, attr.span(), E0548, "incorrect stability attribute type");
            continue
        }
    }

    // Merge the deprecation info into the stability info
    if let Some(rustc_depr) = rustc_depr {
        if let Some(ref mut stab) = stab {
            stab.rustc_depr = Some(rustc_depr);
        } else {
            span_err!(diagnostic, item_sp, E0549,
                      "rustc_deprecated attribute must be paired with \
                       either stable or unstable attribute");
        }
    }

    // Merge the const-unstable info into the stability info
    if let Some(rustc_const_unstable) = rustc_const_unstable {
        if let Some(ref mut stab) = stab {
            stab.rustc_const_unstable = Some(rustc_const_unstable);
        } else {
            span_err!(diagnostic, item_sp, E0630,
                      "rustc_const_unstable attribute must be paired with \
                       either stable or unstable attribute");
        }
    }

    stab
}

fn find_deprecation_generic<'a, I>(diagnostic: &Handler,
                                   attrs_iter: I,
                                   item_sp: Span)
                                   -> Option<Deprecation>
    where I: Iterator<Item = &'a Attribute>
{
    let mut depr: Option<Deprecation> = None;

    'outer: for attr in attrs_iter {
        if attr.path != "deprecated" {
            continue
        }

        mark_used(attr);

        if depr.is_some() {
            span_err!(diagnostic, item_sp, E0550, "multiple deprecated attributes");
            break
        }

        depr = if let Some(metas) = attr.meta_item_list() {
            let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                if item.is_some() {
                    handle_errors(diagnostic, meta.span, AttrError::MultipleItem(meta.name()));
                    return false
                }
                if let Some(v) = meta.value_str() {
                    *item = Some(v);
                    true
                } else {
                    span_err!(diagnostic, meta.span, E0551, "incorrect meta item");
                    false
                }
            };

            let mut since = None;
            let mut note = None;
            for meta in metas {
                if let NestedMetaItemKind::MetaItem(ref mi) = meta.node {
                    match &*mi.name().as_str() {
                        "since" => if !get(mi, &mut since) { continue 'outer },
                        "note" => if !get(mi, &mut note) { continue 'outer },
                        _ => {
                            handle_errors(diagnostic, meta.span,
                                          AttrError::UnknownMetaItem(mi.name()));
                            continue 'outer
                        }
                    }
                } else {
                    handle_errors(diagnostic, meta.span, AttrError::UnsupportedLiteral);
                    continue 'outer
                }
            }

            Some(Deprecation {since: since, note: note})
        } else {
            Some(Deprecation{since: None, note: None})
        }
    }

    depr
}

/// Find the first stability attribute. `None` if none exists.
pub fn find_stability(diagnostic: &Handler, attrs: &[Attribute],
                      item_sp: Span) -> Option<Stability> {
    find_stability_generic(diagnostic, attrs.iter(), item_sp)
}

/// Find the deprecation attribute. `None` if none exists.
pub fn find_deprecation(diagnostic: &Handler, attrs: &[Attribute],
                        item_sp: Span) -> Option<Deprecation> {
    find_deprecation_generic(diagnostic, attrs.iter(), item_sp)
}


/// Parse #[repr(...)] forms.
///
/// Valid repr contents: any of the primitive integral type names (see
/// `int_type_of_word`, below) to specify enum discriminant type; `C`, to use
/// the same discriminant size that the corresponding C enum would or C
/// structure layout, and `packed` to remove padding.
pub fn find_repr_attrs(diagnostic: &Handler, attr: &Attribute) -> Vec<ReprAttr> {
    let mut acc = Vec::new();
    if attr.path == "repr" {
        if let Some(items) = attr.meta_item_list() {
            mark_used(attr);
            for item in items {
                if !item.is_meta_item() {
                    handle_errors(diagnostic, item.span, AttrError::UnsupportedLiteral);
                    continue
                }

                let mut recognised = false;
                if let Some(mi) = item.word() {
                    let word = &*mi.name().as_str();
                    let hint = match word {
                        "C" => Some(ReprC),
                        "packed" => Some(ReprPacked),
                        "simd" => Some(ReprSimd),
                        _ => match int_type_of_word(word) {
                            Some(ity) => Some(ReprInt(ity)),
                            None => {
                                None
                            }
                        }
                    };

                    if let Some(h) = hint {
                        recognised = true;
                        acc.push(h);
                    }
                } else if let Some((name, value)) = item.name_value_literal() {
                    if name == "align" {
                        recognised = true;
                        let mut align_error = None;
                        if let ast::LitKind::Int(align, ast::LitIntType::Unsuffixed) = value.node {
                            if align.is_power_of_two() {
                                // rustc::ty::layout::Align restricts align to <= 2147483647
                                if align <= 2147483647 {
                                    acc.push(ReprAlign(align as u32));
                                } else {
                                    align_error = Some("larger than 2147483647");
                                }
                            } else {
                                align_error = Some("not a power of two");
                            }
                        } else {
                            align_error = Some("not an unsuffixed integer");
                        }
                        if let Some(align_error) = align_error {
                            span_err!(diagnostic, item.span, E0589,
                                      "invalid `repr(align)` attribute: {}", align_error);
                        }
                    }
                }
                if !recognised {
                    // Not a word we recognize
                    span_err!(diagnostic, item.span, E0552,
                              "unrecognized representation hint");
                }
            }
        }
    }
    acc
}

fn int_type_of_word(s: &str) -> Option<IntType> {
    match s {
        "i8" => Some(SignedInt(ast::IntTy::I8)),
        "u8" => Some(UnsignedInt(ast::UintTy::U8)),
        "i16" => Some(SignedInt(ast::IntTy::I16)),
        "u16" => Some(UnsignedInt(ast::UintTy::U16)),
        "i32" => Some(SignedInt(ast::IntTy::I32)),
        "u32" => Some(UnsignedInt(ast::UintTy::U32)),
        "i64" => Some(SignedInt(ast::IntTy::I64)),
        "u64" => Some(UnsignedInt(ast::UintTy::U64)),
        "i128" => Some(SignedInt(ast::IntTy::I128)),
        "u128" => Some(UnsignedInt(ast::UintTy::U128)),
        "isize" => Some(SignedInt(ast::IntTy::Isize)),
        "usize" => Some(UnsignedInt(ast::UintTy::Usize)),
        _ => None
    }
}

#[derive(PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone)]
pub enum ReprAttr {
    ReprInt(IntType),
    ReprC,
    ReprPacked,
    ReprSimd,
    ReprAlign(u32),
}

#[derive(Eq, Hash, PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone)]
pub enum IntType {
    SignedInt(ast::IntTy),
    UnsignedInt(ast::UintTy)
}

impl IntType {
    #[inline]
    pub fn is_signed(self) -> bool {
        match self {
            SignedInt(..) => true,
            UnsignedInt(..) => false
        }
    }
}

impl MetaItem {
    fn tokens(&self) -> TokenStream {
        let ident = TokenTree::Token(self.span, Token::Ident(Ident::with_empty_ctxt(self.name)));
        TokenStream::concat(vec![ident.into(), self.node.tokens(self.span)])
    }

    fn from_tokens<I>(tokens: &mut iter::Peekable<I>) -> Option<MetaItem>
        where I: Iterator<Item = TokenTree>,
    {
        let (span, name) = match tokens.next() {
            Some(TokenTree::Token(span, Token::Ident(ident))) => (span, ident.name),
            Some(TokenTree::Token(_, Token::Interpolated(ref nt))) => match nt.0 {
                token::Nonterminal::NtIdent(ident) => (ident.span, ident.node.name),
                token::Nonterminal::NtMeta(ref meta) => return Some(meta.clone()),
                _ => return None,
            },
            _ => return None,
        };
        let list_closing_paren_pos = tokens.peek().map(|tt| tt.span().hi());
        let node = MetaItemKind::from_tokens(tokens)?;
        let hi = match node {
            MetaItemKind::NameValue(ref lit) => lit.span.hi(),
            MetaItemKind::List(..) => list_closing_paren_pos.unwrap_or(span.hi()),
            _ => span.hi(),
        };
        Some(MetaItem { name, node, span: span.with_hi(hi) })
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
    fn tokens(&self) -> TokenStream {
        TokenTree::Token(self.span, self.node.token()).into()
    }
}

impl LitKind {
    fn token(&self) -> Token {
        use std::ascii;

        match *self {
            LitKind::Str(string, ast::StrStyle::Cooked) => {
                let mut escaped = String::new();
                for ch in string.as_str().chars() {
                    escaped.extend(ch.escape_unicode());
                }
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
            }))),
        }
    }

    fn from_token(token: Token) -> Option<LitKind> {
        match token {
            Token::Ident(ident) if ident.name == "true" => Some(LitKind::Bool(true)),
            Token::Ident(ident) if ident.name == "false" => Some(LitKind::Bool(false)),
            Token::Interpolated(ref nt) => match nt.0 {
                token::NtExpr(ref v) => match v.node {
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
