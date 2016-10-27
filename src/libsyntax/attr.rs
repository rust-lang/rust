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
use ast::{AttrId, Attribute, Attribute_};
use ast::{MetaItem, MetaItemKind, NestedMetaItem, NestedMetaItemKind};
use ast::{Lit, Expr, Item, Local, Stmt, StmtKind};
use codemap::{respan, spanned, dummy_spanned};
use syntax_pos::{Span, BytePos, DUMMY_SP};
use errors::Handler;
use feature_gate::{Features, GatedCfg};
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::token::InternedString;
use parse::{ParseSess, token};
use ptr::P;
use util::ThinVec;

use std::cell::{RefCell, Cell};
use std::collections::HashSet;

thread_local! {
    static USED_ATTRS: RefCell<Vec<u64>> = RefCell::new(Vec::new())
}

enum AttrError {
    MultipleItem(InternedString),
    UnknownMetaItem(InternedString),
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
    let AttrId(id) = attr.node.id;
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
    let AttrId(id) = attr.node.id;
    USED_ATTRS.with(|slot| {
        let idx = (id / 64) as usize;
        let shift = id % 64;
        slot.borrow().get(idx).map(|bits| bits & (1 << shift) != 0)
            .unwrap_or(false)
    })
}

impl NestedMetaItem {
    /// Returns the MetaItem if self is a NestedMetaItemKind::MetaItem.
    pub fn meta_item(&self) -> Option<&P<MetaItem>> {
        match self.node {
            NestedMetaItemKind::MetaItem(ref item) => Some(&item),
            _ => None
        }
    }

    /// Returns the Lit if self is a NestedMetaItemKind::Literal.
    pub fn literal(&self) -> Option<&Lit> {
        match self.node {
            NestedMetaItemKind::Literal(ref lit) => Some(&lit),
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
    pub fn name(&self) -> Option<InternedString> {
        self.meta_item().and_then(|meta_item| Some(meta_item.name()))
    }

    /// Gets the string value if self is a MetaItem and the MetaItem is a
    /// MetaItemKind::NameValue variant containing a string, otherwise None.
    pub fn value_str(&self) -> Option<InternedString> {
        self.meta_item().and_then(|meta_item| meta_item.value_str())
    }

    /// Returns a MetaItem if self is a MetaItem with Kind Word.
    pub fn word(&self) -> Option<&P<MetaItem>> {
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
        let matches = name == &self.name()[..];
        if matches {
            mark_used(self);
        }
        matches
    }

    pub fn name(&self) -> InternedString { self.meta().name() }

    pub fn value_str(&self) -> Option<InternedString> {
        self.meta().value_str()
    }

    pub fn meta_item_list(&self) -> Option<&[NestedMetaItem]> {
        self.meta().meta_item_list()
    }

    pub fn is_word(&self) -> bool { self.meta().is_word() }

    pub fn span(&self) -> Span { self.meta().span }

    pub fn is_meta_item_list(&self) -> bool {
        self.meta_item_list().is_some()
    }

    /// Indicates if the attribute is a Value String.
    pub fn is_value_str(&self) -> bool {
        self.value_str().is_some()
    }
}

impl MetaItem {
    pub fn name(&self) -> InternedString {
        match self.node {
            MetaItemKind::Word(ref n) => (*n).clone(),
            MetaItemKind::NameValue(ref n, _) => (*n).clone(),
            MetaItemKind::List(ref n, _) => (*n).clone(),
        }
    }

    pub fn value_str(&self) -> Option<InternedString> {
        match self.node {
            MetaItemKind::NameValue(_, ref v) => {
                match v.node {
                    ast::LitKind::Str(ref s, _) => Some((*s).clone()),
                    _ => None,
                }
            },
            _ => None
        }
    }

    pub fn meta_item_list(&self) -> Option<&[NestedMetaItem]> {
        match self.node {
            MetaItemKind::List(_, ref l) => Some(&l[..]),
            _ => None
        }
    }

    pub fn is_word(&self) -> bool {
        match self.node {
            MetaItemKind::Word(_) => true,
            _ => false,
        }
    }

    pub fn span(&self) -> Span { self.span }

    pub fn check_name(&self, name: &str) -> bool {
        name == &self.name()[..]
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
    pub fn meta(&self) -> &MetaItem {
        &self.node.value
    }

    /// Convert self to a normal #[doc="foo"] comment, if it is a
    /// comment like `///` or `/** */`. (Returns self unchanged for
    /// non-sugared doc attributes.)
    pub fn with_desugared_doc<T, F>(&self, f: F) -> T where
        F: FnOnce(&Attribute) -> T,
    {
        if self.node.is_sugared_doc {
            let comment = self.value_str().unwrap();
            let meta = mk_name_value_item_str(
                InternedString::new("doc"),
                token::intern_and_get_ident(&strip_doc_comment_decoration(
                        &comment)));
            if self.node.style == ast::AttrStyle::Outer {
                f(&mk_attr_outer(self.node.id, meta))
            } else {
                f(&mk_attr_inner(self.node.id, meta))
            }
        } else {
            f(self)
        }
    }
}

/* Constructors */

pub fn mk_name_value_item_str(name: InternedString, value: InternedString)
                              -> P<MetaItem> {
    let value_lit = dummy_spanned(ast::LitKind::Str(value, ast::StrStyle::Cooked));
    mk_spanned_name_value_item(DUMMY_SP, name, value_lit)
}

pub fn mk_name_value_item(name: InternedString, value: ast::Lit)
                          -> P<MetaItem> {
    mk_spanned_name_value_item(DUMMY_SP, name, value)
}

pub fn mk_list_item(name: InternedString, items: Vec<NestedMetaItem>) -> P<MetaItem> {
    mk_spanned_list_item(DUMMY_SP, name, items)
}

pub fn mk_list_word_item(name: InternedString) -> ast::NestedMetaItem {
    dummy_spanned(NestedMetaItemKind::MetaItem(mk_spanned_word_item(DUMMY_SP, name)))
}

pub fn mk_word_item(name: InternedString) -> P<MetaItem> {
    mk_spanned_word_item(DUMMY_SP, name)
}

pub fn mk_spanned_name_value_item(sp: Span, name: InternedString, value: ast::Lit)
                          -> P<MetaItem> {
    P(respan(sp, MetaItemKind::NameValue(name, value)))
}

pub fn mk_spanned_list_item(sp: Span, name: InternedString, items: Vec<NestedMetaItem>)
                            -> P<MetaItem> {
    P(respan(sp, MetaItemKind::List(name, items)))
}

pub fn mk_spanned_word_item(sp: Span, name: InternedString) -> P<MetaItem> {
    P(respan(sp, MetaItemKind::Word(name)))
}



thread_local! { static NEXT_ATTR_ID: Cell<usize> = Cell::new(0) }

pub fn mk_attr_id() -> AttrId {
    let id = NEXT_ATTR_ID.with(|slot| {
        let r = slot.get();
        slot.set(r + 1);
        r
    });
    AttrId(id)
}

/// Returns an inner attribute with the given value.
pub fn mk_attr_inner(id: AttrId, item: P<MetaItem>) -> Attribute {
    mk_spanned_attr_inner(DUMMY_SP, id, item)
}

/// Returns an innter attribute with the given value and span.
pub fn mk_spanned_attr_inner(sp: Span, id: AttrId, item: P<MetaItem>) -> Attribute {
    respan(sp,
           Attribute_ {
            id: id,
            style: ast::AttrStyle::Inner,
            value: item,
            is_sugared_doc: false,
          })
}


/// Returns an outer attribute with the given value.
pub fn mk_attr_outer(id: AttrId, item: P<MetaItem>) -> Attribute {
    mk_spanned_attr_outer(DUMMY_SP, id, item)
}

/// Returns an outer attribute with the given value and span.
pub fn mk_spanned_attr_outer(sp: Span, id: AttrId, item: P<MetaItem>) -> Attribute {
    respan(sp,
           Attribute_ {
            id: id,
            style: ast::AttrStyle::Outer,
            value: item,
            is_sugared_doc: false,
          })
}

pub fn mk_doc_attr_outer(id: AttrId, item: P<MetaItem>, is_sugared_doc: bool) -> Attribute {
    dummy_spanned(Attribute_ {
        id: id,
        style: ast::AttrStyle::Outer,
        value: item,
        is_sugared_doc: is_sugared_doc,
    })
}

pub fn mk_sugared_doc_attr(id: AttrId, text: InternedString, lo: BytePos,
                           hi: BytePos)
                           -> Attribute {
    let style = doc_comment_style(&text);
    let lit = spanned(lo, hi, ast::LitKind::Str(text, ast::StrStyle::Cooked));
    let attr = Attribute_ {
        id: id,
        style: style,
        value: P(spanned(lo, hi, MetaItemKind::NameValue(InternedString::new("doc"), lit))),
        is_sugared_doc: true
    };
    spanned(lo, hi, attr)
}

/* Searching */
/// Check if `needle` occurs in `haystack` by a structural
/// comparison. This is slightly subtle, and relies on ignoring the
/// span included in the `==` comparison a plain MetaItem.
pub fn contains(haystack: &[P<MetaItem>], needle: &MetaItem) -> bool {
    debug!("attr::contains (name={})", needle.name());
    haystack.iter().any(|item| {
        debug!("  testing: {}", item.name());
        item.node == needle.node
    })
}

pub fn list_contains_name(items: &[NestedMetaItem], name: &str) -> bool {
    debug!("attr::list_contains_name (name={})", name);
    items.iter().any(|item| {
        debug!("  testing: {:?}", item.name());
        item.check_name(name)
    })
}

pub fn contains_name(attrs: &[Attribute], name: &str) -> bool {
    debug!("attr::contains_name (name={})", name);
    attrs.iter().any(|item| {
        debug!("  testing: {}", item.name());
        item.check_name(name)
    })
}

pub fn first_attr_value_str_by_name(attrs: &[Attribute], name: &str)
                                 -> Option<InternedString> {
    attrs.iter()
        .find(|at| at.check_name(name))
        .and_then(|at| at.value_str())
}

pub fn last_meta_item_value_str_by_name(items: &[P<MetaItem>], name: &str)
                                     -> Option<InternedString> {
    items.iter()
         .rev()
         .find(|mi| mi.check_name(name))
         .and_then(|i| i.value_str())
}

/* Higher-level applications */

pub fn find_crate_name(attrs: &[Attribute]) -> Option<InternedString> {
    first_attr_value_str_by_name(attrs, "crate_name")
}

/// Find the value of #[export_name=*] attribute and check its validity.
pub fn find_export_name_attr(diag: &Handler, attrs: &[Attribute]) -> Option<InternedString> {
    attrs.iter().fold(None, |ia,attr| {
        if attr.check_name("export_name") {
            if let s@Some(_) = attr.value_str() {
                s
            } else {
                struct_span_err!(diag, attr.span, E0558,
                                 "export_name attribute has invalid format")
                    .span_label(attr.span,
                                &format!("did you mean #[export_name=\"*\"]?"))
                    .emit();
                None
            }
        } else {
            ia
        }
    })
}

pub fn contains_extern_indicator(diag: &Handler, attrs: &[Attribute]) -> bool {
    contains_name(attrs, "no_mangle") ||
        find_export_name_attr(diag, attrs).is_some()
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
    attrs.iter().fold(InlineAttr::None, |ia,attr| {
        match attr.node.value.node {
            MetaItemKind::Word(ref n) if n == "inline" => {
                mark_used(attr);
                InlineAttr::Hint
            }
            MetaItemKind::List(ref n, ref items) if n == "inline" => {
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
    match cfg.node {
        ast::MetaItemKind::List(ref pred, ref mis) => {
            for mi in mis.iter() {
                if !mi.is_meta_item() {
                    handle_errors(&sess.span_diagnostic, mi.span, AttrError::UnsupportedLiteral);
                    return false;
                }
            }

            // The unwraps below may look dangerous, but we've already asserted
            // that they won't fail with the loop above.
            match &pred[..] {
                "any" => mis.iter().any(|mi| {
                    cfg_matches(mi.meta_item().unwrap(), sess, features)
                }),
                "all" => mis.iter().all(|mi| {
                    cfg_matches(mi.meta_item().unwrap(), sess, features)
                }),
                "not" => {
                    if mis.len() != 1 {
                        span_err!(sess.span_diagnostic, cfg.span, E0536, "expected 1 cfg-pattern");
                        return false;
                    }

                    !cfg_matches(mis[0].meta_item().unwrap(), sess, features)
                },
                p => {
                    span_err!(sess.span_diagnostic, cfg.span, E0537, "invalid predicate `{}`", p);
                    false
                }
            }
        },
        ast::MetaItemKind::Word(_) | ast::MetaItemKind::NameValue(..) => {
            if let (Some(feats), Some(gated_cfg)) = (features, GatedCfg::gate(cfg)) {
                gated_cfg.check_and_emit(sess, feats);
            }
            contains(&sess.config, cfg)
        }
    }
}

/// Represents the #[stable], #[unstable] and #[rustc_deprecated] attributes.
#[derive(RustcEncodable, RustcDecodable, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: InternedString,
    pub rustc_depr: Option<RustcDeprecation>,
}

/// The available stability levels.
#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub enum StabilityLevel {
    // Reason for the current stability level and the relevant rust-lang issue
    Unstable { reason: Option<InternedString>, issue: u32 },
    Stable { since: InternedString },
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub struct RustcDeprecation {
    pub since: InternedString,
    pub reason: InternedString,
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub struct Deprecation {
    pub since: Option<InternedString>,
    pub note: Option<InternedString>,
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

    'outer: for attr in attrs_iter {
        let tag = attr.name();
        let tag = &*tag;
        if tag != "rustc_deprecated" && tag != "unstable" && tag != "stable" {
            continue // not a stability level
        }

        mark_used(attr);

        if let Some(metas) = attr.meta_item_list() {
            let get = |meta: &MetaItem, item: &mut Option<InternedString>| {
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

            match tag {
                "rustc_deprecated" => {
                    if rustc_depr.is_some() {
                        span_err!(diagnostic, item_sp, E0540,
                                  "multiple rustc_deprecated attributes");
                        break
                    }

                    let mut since = None;
                    let mut reason = None;
                    for meta in metas {
                        if let Some(mi) = meta.meta_item() {
                            match &*mi.name() {
                                "since" => if !get(mi, &mut since) { continue 'outer },
                                "reason" => if !get(mi, &mut reason) { continue 'outer },
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

                    match (since, reason) {
                        (Some(since), Some(reason)) => {
                            rustc_depr = Some(RustcDeprecation {
                                since: since,
                                reason: reason,
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
                            match &*mi.name() {
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
                                    reason: reason,
                                    issue: {
                                        if let Ok(issue) = issue.parse() {
                                            issue
                                        } else {
                                            span_err!(diagnostic, attr.span(), E0545,
                                                      "incorrect 'issue'");
                                            continue
                                        }
                                    }
                                },
                                feature: feature,
                                rustc_depr: None,
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
                            match &*mi.name() {
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
                                    since: since,
                                },
                                feature: feature,
                                rustc_depr: None,
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
            if let Unstable {reason: ref mut reason @ None, ..} = stab.level {
                *reason = Some(rustc_depr.reason.clone())
            }
            stab.rustc_depr = Some(rustc_depr);
        } else {
            span_err!(diagnostic, item_sp, E0549,
                      "rustc_deprecated attribute must be paired with \
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
        if attr.name() != "deprecated" {
            continue
        }

        mark_used(attr);

        if depr.is_some() {
            span_err!(diagnostic, item_sp, E0550, "multiple deprecated attributes");
            break
        }

        depr = if let Some(metas) = attr.meta_item_list() {
            let get = |meta: &MetaItem, item: &mut Option<InternedString>| {
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
                    match &*mi.name() {
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

pub fn require_unique_names(diagnostic: &Handler, metas: &[P<MetaItem>]) {
    let mut set = HashSet::new();
    for meta in metas {
        let name = meta.name();

        if !set.insert(name.clone()) {
            panic!(diagnostic.span_fatal(meta.span,
                                         &format!("duplicate meta item `{}`", name)));
        }
    }
}


/// Parse #[repr(...)] forms.
///
/// Valid repr contents: any of the primitive integral type names (see
/// `int_type_of_word`, below) to specify enum discriminant type; `C`, to use
/// the same discriminant size that the corresponding C enum would or C
/// structure layout, and `packed` to remove padding.
pub fn find_repr_attrs(diagnostic: &Handler, attr: &Attribute) -> Vec<ReprAttr> {
    let mut acc = Vec::new();
    match attr.node.value.node {
        ast::MetaItemKind::List(ref s, ref items) if s == "repr" => {
            mark_used(attr);
            for item in items {
                if !item.is_meta_item() {
                    handle_errors(diagnostic, item.span, AttrError::UnsupportedLiteral);
                    continue
                }

                if let Some(mi) = item.word() {
                    let word = &*mi.name();
                    let hint = match word {
                        // Can't use "extern" because it's not a lexical identifier.
                        "C" => Some(ReprExtern),
                        "packed" => Some(ReprPacked),
                        "simd" => Some(ReprSimd),
                        _ => match int_type_of_word(word) {
                            Some(ity) => Some(ReprInt(ity)),
                            None => {
                                // Not a word we recognize
                                span_err!(diagnostic, item.span, E0552,
                                          "unrecognized representation hint");
                                None
                            }
                        }
                    };

                    if let Some(h) = hint {
                        acc.push(h);
                    }
                } else {
                    span_err!(diagnostic, item.span, E0553,
                              "unrecognized enum representation hint");
                }
            }
        }
        // Not a "repr" hint: ignore.
        _ => { }
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
        "isize" => Some(SignedInt(ast::IntTy::Is)),
        "usize" => Some(UnsignedInt(ast::UintTy::Us)),
        _ => None
    }
}

#[derive(PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone)]
pub enum ReprAttr {
    ReprAny,
    ReprInt(IntType),
    ReprExtern,
    ReprPacked,
    ReprSimd,
}

impl ReprAttr {
    pub fn is_ffi_safe(&self) -> bool {
        match *self {
            ReprAny => false,
            ReprInt(ity) => ity.is_ffi_safe(),
            ReprExtern => true,
            ReprPacked => false,
            ReprSimd => true,
        }
    }
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
    fn is_ffi_safe(self) -> bool {
        match self {
            SignedInt(ast::IntTy::I8) | UnsignedInt(ast::UintTy::U8) |
            SignedInt(ast::IntTy::I16) | UnsignedInt(ast::UintTy::U16) |
            SignedInt(ast::IntTy::I32) | UnsignedInt(ast::UintTy::U32) |
            SignedInt(ast::IntTy::I64) | UnsignedInt(ast::UintTy::U64) => true,
            SignedInt(ast::IntTy::Is) | UnsignedInt(ast::UintTy::Us) => false
        }
    }
}

pub trait HasAttrs: Sized {
    fn attrs(&self) -> &[ast::Attribute];
    fn map_attrs<F: FnOnce(Vec<ast::Attribute>) -> Vec<ast::Attribute>>(self, f: F) -> Self;
}

impl HasAttrs for Vec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        &self
    }
    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(self, f: F) -> Self {
        f(self)
    }
}

impl HasAttrs for ThinVec<Attribute> {
    fn attrs(&self) -> &[Attribute] {
        &self
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

macro_rules! derive_has_attrs_from_field {
    ($($ty:path),*) => { derive_has_attrs_from_field!($($ty: .attrs),*); };
    ($($ty:path : $(.$field:ident)*),*) => { $(
        impl HasAttrs for $ty {
            fn attrs(&self) -> &[Attribute] {
                self $(.$field)* .attrs()
            }

            fn map_attrs<F>(mut self, f: F) -> Self
                where F: FnOnce(Vec<Attribute>) -> Vec<Attribute>,
            {
                self $(.$field)* = self $(.$field)* .map_attrs(f);
                self
            }
        }
    )* }
}

derive_has_attrs_from_field! {
    Item, Expr, Local, ast::ForeignItem, ast::StructField, ast::ImplItem, ast::TraitItem, ast::Arm
}

derive_has_attrs_from_field! { Stmt: .node, ast::Variant: .node.attrs }
