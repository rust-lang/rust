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
use ast::{AttrId, Attribute, Name};
use ast::{MetaItem, MetaItemKind, NestedMetaItem, NestedMetaItemKind};
use ast::{Lit, Expr, Item, Local, Stmt, StmtKind};
use codemap::{Spanned, spanned, dummy_spanned, mk_sp};
use syntax_pos::{Span, BytePos, DUMMY_SP};
use errors::Handler;
use feature_gate::{Features, GatedCfg};
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::ParseSess;
use ptr::P;
use symbol::Symbol;
use util::ThinVec;

use std::cell::{RefCell, Cell};

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
    pub fn name(&self) -> Option<Name> {
        self.meta_item().and_then(|meta_item| Some(meta_item.name()))
    }

    /// Gets the string value if self is a MetaItem and the MetaItem is a
    /// MetaItemKind::NameValue variant containing a string, otherwise None.
    pub fn value_str(&self) -> Option<Symbol> {
        self.meta_item().and_then(|meta_item| meta_item.value_str())
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
        let matches = self.name() == name;
        if matches {
            mark_used(self);
        }
        matches
    }

    pub fn name(&self) -> Name { self.meta().name() }

    pub fn value_str(&self) -> Option<Symbol> {
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
    pub fn name(&self) -> Name {
        self.name
    }

    pub fn value_str(&self) -> Option<Symbol> {
        match self.node {
            MetaItemKind::NameValue(ref v) => {
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
    pub fn meta(&self) -> &MetaItem {
        &self.value
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
            if self.style == ast::AttrStyle::Outer {
                f(&mk_attr_outer(self.id, meta))
            } else {
                f(&mk_attr_inner(self.id, meta))
            }
        } else {
            f(self)
        }
    }
}

/* Constructors */

pub fn mk_name_value_item_str(name: Name, value: Symbol) -> MetaItem {
    let value_lit = dummy_spanned(ast::LitKind::Str(value, ast::StrStyle::Cooked));
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
pub fn mk_attr_inner(id: AttrId, item: MetaItem) -> Attribute {
    mk_spanned_attr_inner(DUMMY_SP, id, item)
}

/// Returns an innter attribute with the given value and span.
pub fn mk_spanned_attr_inner(sp: Span, id: AttrId, item: MetaItem) -> Attribute {
    Attribute {
        id: id,
        style: ast::AttrStyle::Inner,
        value: item,
        is_sugared_doc: false,
        span: sp,
    }
}


/// Returns an outer attribute with the given value.
pub fn mk_attr_outer(id: AttrId, item: MetaItem) -> Attribute {
    mk_spanned_attr_outer(DUMMY_SP, id, item)
}

/// Returns an outer attribute with the given value and span.
pub fn mk_spanned_attr_outer(sp: Span, id: AttrId, item: MetaItem) -> Attribute {
    Attribute {
        id: id,
        style: ast::AttrStyle::Outer,
        value: item,
        is_sugared_doc: false,
        span: sp,
    }
}

pub fn mk_sugared_doc_attr(id: AttrId, text: Symbol, lo: BytePos, hi: BytePos)
                           -> Attribute {
    let style = doc_comment_style(&text.as_str());
    let lit = spanned(lo, hi, ast::LitKind::Str(text, ast::StrStyle::Cooked));
    Attribute {
        id: id,
        style: style,
        value: MetaItem {
            span: mk_sp(lo, hi),
            name: Symbol::intern("doc"),
            node: MetaItemKind::NameValue(lit),
        },
        is_sugared_doc: true,
        span: mk_sp(lo, hi),
    }
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

pub fn first_attr_value_str_by_name(attrs: &[Attribute], name: &str) -> Option<Symbol> {
    attrs.iter()
        .find(|at| at.check_name(name))
        .and_then(|at| at.value_str())
}

/* Higher-level applications */

pub fn find_crate_name(attrs: &[Attribute]) -> Option<Symbol> {
    first_attr_value_str_by_name(attrs, "crate_name")
}

/// Find the value of #[export_name=*] attribute and check its validity.
pub fn find_export_name_attr(diag: &Handler, attrs: &[Attribute]) -> Option<Symbol> {
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
    attrs.iter().fold(InlineAttr::None, |ia, attr| {
        match attr.value.node {
            _ if attr.value.name != "inline" => ia,
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
        ast::MetaItemKind::Word | ast::MetaItemKind::NameValue(..) => {
            if let (Some(feats), Some(gated_cfg)) = (features, GatedCfg::gate(cfg)) {
                gated_cfg.check_and_emit(sess, feats);
            }
            sess.config.contains(&(cfg.name(), cfg.value_str()))
        }
    }
}

/// Represents the #[stable], #[unstable] and #[rustc_deprecated] attributes.
#[derive(RustcEncodable, RustcDecodable, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    pub rustc_depr: Option<RustcDeprecation>,
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

    'outer: for attr in attrs_iter {
        let tag = attr.name();
        if tag != "rustc_deprecated" && tag != "unstable" && tag != "stable" {
            continue // not a stability level
        }

        mark_used(attr);

        if let Some(metas) = attr.meta_item_list() {
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

            match &*tag.as_str() {
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
                            match &*mi.name().as_str() {
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
                                    reason: reason,
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
    match attr.value.node {
        ast::MetaItemKind::List(ref items) if attr.value.name == "repr" => {
            mark_used(attr);
            for item in items {
                if !item.is_meta_item() {
                    handle_errors(diagnostic, item.span, AttrError::UnsupportedLiteral);
                    continue
                }

                if let Some(mi) = item.word() {
                    let word = &*mi.name().as_str();
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
        "i128" => Some(SignedInt(ast::IntTy::I128)),
        "u128" => Some(UnsignedInt(ast::UintTy::U128)),
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
            SignedInt(ast::IntTy::I64) | UnsignedInt(ast::UintTy::U64) |
            SignedInt(ast::IntTy::I128) | UnsignedInt(ast::UintTy::U128) => true,
            SignedInt(ast::IntTy::Is) | UnsignedInt(ast::UintTy::Us) => false
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
        Spanned { node: self.node.map_attrs(f), span: self.span }
    }
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
