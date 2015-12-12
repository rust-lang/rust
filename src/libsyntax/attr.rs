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
use ast::{AttrId, Attribute, Attribute_, MetaItem, MetaWord, MetaNameValue, MetaList};
use ast::{Stmt, StmtDecl, StmtExpr, StmtMac, StmtSemi, DeclItem, DeclLocal};
use ast::{Expr, Item, Local, Decl};
use codemap::{Span, Spanned, spanned, dummy_spanned};
use codemap::BytePos;
use config::CfgDiag;
use diagnostic::SpanHandler;
use feature_gate::{GatedCfg, GatedCfgAttr};
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::token::InternedString;
use parse::token;
use ptr::P;

use std::cell::{RefCell, Cell};
use std::collections::HashSet;

thread_local! {
    static USED_ATTRS: RefCell<Vec<u64>> = RefCell::new(Vec::new())
}

pub fn mark_used(attr: &Attribute) {
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

pub trait AttrMetaMethods {
    fn check_name(&self, name: &str) -> bool {
        name == &self.name()[..]
    }

    /// Retrieve the name of the meta item, e.g. `foo` in `#[foo]`,
    /// `#[foo="bar"]` and `#[foo(bar)]`
    fn name(&self) -> InternedString;

    /// Gets the string value if self is a MetaNameValue variant
    /// containing a string, otherwise None.
    fn value_str(&self) -> Option<InternedString>;
    /// Gets a list of inner meta items from a list MetaItem type.
    fn meta_item_list(&self) -> Option<&[P<MetaItem>]>;

    fn span(&self) -> Span;
}

impl AttrMetaMethods for Attribute {
    fn check_name(&self, name: &str) -> bool {
        let matches = name == &self.name()[..];
        if matches {
            mark_used(self);
        }
        matches
    }
    fn name(&self) -> InternedString { self.meta().name() }
    fn value_str(&self) -> Option<InternedString> {
        self.meta().value_str()
    }
    fn meta_item_list(&self) -> Option<&[P<MetaItem>]> {
        self.node.value.meta_item_list()
    }
    fn span(&self) -> Span { self.meta().span }
}

impl AttrMetaMethods for MetaItem {
    fn name(&self) -> InternedString {
        match self.node {
            MetaWord(ref n) => (*n).clone(),
            MetaNameValue(ref n, _) => (*n).clone(),
            MetaList(ref n, _) => (*n).clone(),
        }
    }

    fn value_str(&self) -> Option<InternedString> {
        match self.node {
            MetaNameValue(_, ref v) => {
                match v.node {
                    ast::LitStr(ref s, _) => Some((*s).clone()),
                    _ => None,
                }
            },
            _ => None
        }
    }

    fn meta_item_list(&self) -> Option<&[P<MetaItem>]> {
        match self.node {
            MetaList(_, ref l) => Some(&l[..]),
            _ => None
        }
    }
    fn span(&self) -> Span { self.span }
}

// Annoying, but required to get test_cfg to work
impl AttrMetaMethods for P<MetaItem> {
    fn name(&self) -> InternedString { (**self).name() }
    fn value_str(&self) -> Option<InternedString> { (**self).value_str() }
    fn meta_item_list(&self) -> Option<&[P<MetaItem>]> {
        (**self).meta_item_list()
    }
    fn span(&self) -> Span { (**self).span() }
}


pub trait AttributeMethods {
    fn meta(&self) -> &MetaItem;
    fn with_desugared_doc<T, F>(&self, f: F) -> T where
        F: FnOnce(&Attribute) -> T;
}

impl AttributeMethods for Attribute {
    /// Extract the MetaItem from inside this Attribute.
    fn meta(&self) -> &MetaItem {
        &*self.node.value
    }

    /// Convert self to a normal #[doc="foo"] comment, if it is a
    /// comment like `///` or `/** */`. (Returns self unchanged for
    /// non-sugared doc attributes.)
    fn with_desugared_doc<T, F>(&self, f: F) -> T where
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
    let value_lit = dummy_spanned(ast::LitStr(value, ast::CookedStr));
    mk_name_value_item(name, value_lit)
}

pub fn mk_name_value_item(name: InternedString, value: ast::Lit)
                          -> P<MetaItem> {
    P(dummy_spanned(MetaNameValue(name, value)))
}

pub fn mk_list_item(name: InternedString, items: Vec<P<MetaItem>>) -> P<MetaItem> {
    P(dummy_spanned(MetaList(name, items)))
}

pub fn mk_word_item(name: InternedString) -> P<MetaItem> {
    P(dummy_spanned(MetaWord(name)))
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
    dummy_spanned(Attribute_ {
        id: id,
        style: ast::AttrStyle::Inner,
        value: item,
        is_sugared_doc: false,
    })
}

/// Returns an outer attribute with the given value.
pub fn mk_attr_outer(id: AttrId, item: P<MetaItem>) -> Attribute {
    dummy_spanned(Attribute_ {
        id: id,
        style: ast::AttrStyle::Outer,
        value: item,
        is_sugared_doc: false,
    })
}

pub fn mk_sugared_doc_attr(id: AttrId, text: InternedString, lo: BytePos,
                           hi: BytePos)
                           -> Attribute {
    let style = doc_comment_style(&text);
    let lit = spanned(lo, hi, ast::LitStr(text, ast::CookedStr));
    let attr = Attribute_ {
        id: id,
        style: style,
        value: P(spanned(lo, hi, MetaNameValue(InternedString::new("doc"),
                                               lit))),
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

pub fn contains_name<AM: AttrMetaMethods>(metas: &[AM], name: &str) -> bool {
    debug!("attr::contains_name (name={})", name);
    metas.iter().any(|item| {
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

pub fn sort_meta_items(items: Vec<P<MetaItem>>) -> Vec<P<MetaItem>> {
    // This is sort of stupid here, but we need to sort by
    // human-readable strings.
    let mut v = items.into_iter()
        .map(|mi| (mi.name(), mi))
        .collect::<Vec<(InternedString, P<MetaItem>)>>();

    v.sort_by(|&(ref a, _), &(ref b, _)| a.cmp(b));

    // There doesn't seem to be a more optimal way to do this
    v.into_iter().map(|(_, m)| m.map(|Spanned {node, span}| {
        Spanned {
            node: match node {
                MetaList(n, mis) => MetaList(n, sort_meta_items(mis)),
                _ => node
            },
            span: span
        }
    })).collect()
}

pub fn find_crate_name(attrs: &[Attribute]) -> Option<InternedString> {
    first_attr_value_str_by_name(attrs, "crate_name")
}

/// Find the value of #[export_name=*] attribute and check its validity.
pub fn find_export_name_attr(diag: &SpanHandler, attrs: &[Attribute]) -> Option<InternedString> {
    attrs.iter().fold(None, |ia,attr| {
        if attr.check_name("export_name") {
            if let s@Some(_) = attr.value_str() {
                s
            } else {
                diag.span_err(attr.span, "export_name attribute has invalid format");
                diag.handler.help("use #[export_name=\"*\"]");
                None
            }
        } else {
            ia
        }
    })
}

#[derive(Copy, Clone, PartialEq)]
pub enum InlineAttr {
    None,
    Hint,
    Always,
    Never,
}

/// Determine what `#[inline]` attribute is present in `attrs`, if any.
pub fn find_inline_attr(diagnostic: Option<&SpanHandler>, attrs: &[Attribute]) -> InlineAttr {
    attrs.iter().fold(InlineAttr::None, |ia,attr| {
        match attr.node.value.node {
            MetaWord(ref n) if *n == "inline" => {
                mark_used(attr);
                InlineAttr::Hint
            }
            MetaList(ref n, ref items) if *n == "inline" => {
                mark_used(attr);
                if items.len() != 1 {
                    diagnostic.map(|d|{ d.span_err(attr.span, "expected one argument"); });
                    InlineAttr::None
                } else if contains_name(&items[..], "always") {
                    InlineAttr::Always
                } else if contains_name(&items[..], "never") {
                    InlineAttr::Never
                } else {
                    diagnostic.map(|d|{ d.span_err((*items[0]).span, "invalid argument"); });
                    InlineAttr::None
                }
            }
            _ => ia
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
pub fn cfg_matches<T: CfgDiag>(cfgs: &[P<MetaItem>],
                           cfg: &ast::MetaItem,
                           diag: &mut T) -> bool {
    match cfg.node {
        ast::MetaList(ref pred, ref mis) if &pred[..] == "any" =>
            mis.iter().any(|mi| cfg_matches(cfgs, &**mi, diag)),
        ast::MetaList(ref pred, ref mis) if &pred[..] == "all" =>
            mis.iter().all(|mi| cfg_matches(cfgs, &**mi, diag)),
        ast::MetaList(ref pred, ref mis) if &pred[..] == "not" => {
            if mis.len() != 1 {
                diag.emit_error(|diagnostic| {
                    diagnostic.span_err(cfg.span, "expected 1 cfg-pattern");
                });
                return false;
            }
            !cfg_matches(cfgs, &*mis[0], diag)
        }
        ast::MetaList(ref pred, _) => {
            diag.emit_error(|diagnostic| {
                diagnostic.span_err(cfg.span,
                    &format!("invalid predicate `{}`", pred));
            });
            false
        },
        ast::MetaWord(_) | ast::MetaNameValue(..) => {
            diag.flag_gated(|feature_gated_cfgs| {
                feature_gated_cfgs.extend(
                    GatedCfg::gate(cfg).map(GatedCfgAttr::GatedCfg));
            });
            contains(cfgs, cfg)
        }
    }
}

/// Represents the #[stable], #[unstable] and #[rustc_deprecated] attributes.
#[derive(RustcEncodable, RustcDecodable, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: InternedString,
    pub depr: Option<Deprecation>,
}

/// The available stability levels.
#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub enum StabilityLevel {
    // Reason for the current stability level and the relevant rust-lang issue
    Unstable { reason: Option<InternedString>, issue: u32 },
    Stable { since: InternedString },
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd, Clone, Debug, Eq, Hash)]
pub struct Deprecation {
    pub since: InternedString,
    pub reason: InternedString,
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool { if let Unstable {..} = *self { true } else { false }}
    pub fn is_stable(&self) -> bool { if let Stable {..} = *self { true } else { false }}
}

fn find_stability_generic<'a, I>(diagnostic: &SpanHandler,
                                 attrs_iter: I,
                                 item_sp: Span)
                                 -> Option<Stability>
    where I: Iterator<Item = &'a Attribute>
{
    let mut stab: Option<Stability> = None;
    let mut depr: Option<Deprecation> = None;

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
                    diagnostic.span_err(meta.span, &format!("multiple '{}' items",
                                                             meta.name()));
                    return false
                }
                if let Some(v) = meta.value_str() {
                    *item = Some(v);
                    true
                } else {
                    diagnostic.span_err(meta.span, "incorrect meta item");
                    false
                }
            };

            match tag {
                "rustc_deprecated" => {
                    if depr.is_some() {
                        diagnostic.span_err(item_sp, "multiple rustc_deprecated attributes");
                        break
                    }

                    let mut since = None;
                    let mut reason = None;
                    for meta in metas {
                        match &*meta.name() {
                            "since" => if !get(meta, &mut since) { continue 'outer },
                            "reason" => if !get(meta, &mut reason) { continue 'outer },
                            _ => {
                                diagnostic.span_err(meta.span, &format!("unknown meta item '{}'",
                                                                        meta.name()));
                                continue 'outer
                            }
                        }
                    }

                    match (since, reason) {
                        (Some(since), Some(reason)) => {
                            depr = Some(Deprecation {
                                since: since,
                                reason: reason,
                            })
                        }
                        (None, _) => {
                            diagnostic.span_err(attr.span(), "missing 'since'");
                            continue
                        }
                        _ => {
                            diagnostic.span_err(attr.span(), "missing 'reason'");
                            continue
                        }
                    }
                }
                "unstable" => {
                    if stab.is_some() {
                        diagnostic.span_err(item_sp, "multiple stability levels");
                        break
                    }

                    let mut feature = None;
                    let mut reason = None;
                    let mut issue = None;
                    for meta in metas {
                        match &*meta.name() {
                            "feature" => if !get(meta, &mut feature) { continue 'outer },
                            "reason" => if !get(meta, &mut reason) { continue 'outer },
                            "issue" => if !get(meta, &mut issue) { continue 'outer },
                            _ => {
                                diagnostic.span_err(meta.span, &format!("unknown meta item '{}'",
                                                                        meta.name()));
                                continue 'outer
                            }
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
                                            diagnostic.span_err(attr.span(), "incorrect 'issue'");
                                            continue
                                        }
                                    }
                                },
                                feature: feature,
                                depr: None,
                            })
                        }
                        (None, _, _) => {
                            diagnostic.span_err(attr.span(), "missing 'feature'");
                            continue
                        }
                        _ => {
                            diagnostic.span_err(attr.span(), "missing 'issue'");
                            continue
                        }
                    }
                }
                "stable" => {
                    if stab.is_some() {
                        diagnostic.span_err(item_sp, "multiple stability levels");
                        break
                    }

                    let mut feature = None;
                    let mut since = None;
                    for meta in metas {
                        match &*meta.name() {
                            "feature" => if !get(meta, &mut feature) { continue 'outer },
                            "since" => if !get(meta, &mut since) { continue 'outer },
                            _ => {
                                diagnostic.span_err(meta.span, &format!("unknown meta item '{}'",
                                                                        meta.name()));
                                continue 'outer
                            }
                        }
                    }

                    match (feature, since) {
                        (Some(feature), Some(since)) => {
                            stab = Some(Stability {
                                level: Stable {
                                    since: since,
                                },
                                feature: feature,
                                depr: None,
                            })
                        }
                        (None, _) => {
                            diagnostic.span_err(attr.span(), "missing 'feature'");
                            continue
                        }
                        _ => {
                            diagnostic.span_err(attr.span(), "missing 'since'");
                            continue
                        }
                    }
                }
                _ => unreachable!()
            }
        } else {
            diagnostic.span_err(attr.span(), "incorrect stability attribute type");
            continue
        }
    }

    // Merge the deprecation info into the stability info
    if let Some(depr) = depr {
        if let Some(ref mut stab) = stab {
            if let Unstable {reason: ref mut reason @ None, ..} = stab.level {
                *reason = Some(depr.reason.clone())
            }
            stab.depr = Some(depr);
        } else {
            diagnostic.span_err(item_sp, "rustc_deprecated attribute must be paired with \
                                          either stable or unstable attribute");
        }
    }

    stab
}

/// Find the first stability attribute. `None` if none exists.
pub fn find_stability(diagnostic: &SpanHandler, attrs: &[Attribute],
                      item_sp: Span) -> Option<Stability> {
    find_stability_generic(diagnostic, attrs.iter(), item_sp)
}

pub fn require_unique_names(diagnostic: &SpanHandler, metas: &[P<MetaItem>]) {
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
pub fn find_repr_attrs(diagnostic: &SpanHandler, attr: &Attribute) -> Vec<ReprAttr> {
    let mut acc = Vec::new();
    match attr.node.value.node {
        ast::MetaList(ref s, ref items) if *s == "repr" => {
            mark_used(attr);
            for item in items {
                match item.node {
                    ast::MetaWord(ref word) => {
                        let hint = match &word[..] {
                            // Can't use "extern" because it's not a lexical identifier.
                            "C" => Some(ReprExtern),
                            "packed" => Some(ReprPacked),
                            "simd" => Some(ReprSimd),
                            _ => match int_type_of_word(&word) {
                                Some(ity) => Some(ReprInt(item.span, ity)),
                                None => {
                                    // Not a word we recognize
                                    diagnostic.span_err(item.span,
                                                        "unrecognized representation hint");
                                    None
                                }
                            }
                        };

                        match hint {
                            Some(h) => acc.push(h),
                            None => { }
                        }
                    }
                    // Not a word:
                    _ => diagnostic.span_err(item.span, "unrecognized enum representation hint")
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
        "i8" => Some(SignedInt(ast::TyI8)),
        "u8" => Some(UnsignedInt(ast::TyU8)),
        "i16" => Some(SignedInt(ast::TyI16)),
        "u16" => Some(UnsignedInt(ast::TyU16)),
        "i32" => Some(SignedInt(ast::TyI32)),
        "u32" => Some(UnsignedInt(ast::TyU32)),
        "i64" => Some(SignedInt(ast::TyI64)),
        "u64" => Some(UnsignedInt(ast::TyU64)),
        "isize" => Some(SignedInt(ast::TyIs)),
        "usize" => Some(UnsignedInt(ast::TyUs)),
        _ => None
    }
}

#[derive(PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone)]
pub enum ReprAttr {
    ReprAny,
    ReprInt(Span, IntType),
    ReprExtern,
    ReprPacked,
    ReprSimd,
}

impl ReprAttr {
    pub fn is_ffi_safe(&self) -> bool {
        match *self {
            ReprAny => false,
            ReprInt(_sp, ity) => ity.is_ffi_safe(),
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
            SignedInt(ast::TyI8) | UnsignedInt(ast::TyU8) |
            SignedInt(ast::TyI16) | UnsignedInt(ast::TyU16) |
            SignedInt(ast::TyI32) | UnsignedInt(ast::TyU32) |
            SignedInt(ast::TyI64) | UnsignedInt(ast::TyU64) => true,
            SignedInt(ast::TyIs) | UnsignedInt(ast::TyUs) => false
        }
    }
}

/// A list of attributes, behind a optional box as
/// a space optimization.
pub type ThinAttributes = Option<Box<Vec<Attribute>>>;

pub trait ThinAttributesExt {
    fn map_thin_attrs<F>(self, f: F) -> Self
        where F: FnOnce(Vec<Attribute>) -> Vec<Attribute>;
    fn prepend(mut self, attrs: Self) -> Self;
    fn append(mut self, attrs: Self) -> Self;
    fn update<F>(&mut self, f: F)
        where Self: Sized,
              F: FnOnce(Self) -> Self;
    fn as_attr_slice(&self) -> &[Attribute];
    fn into_attr_vec(self) -> Vec<Attribute>;
}

impl ThinAttributesExt for ThinAttributes {
    fn map_thin_attrs<F>(self, f: F) -> Self
        where F: FnOnce(Vec<Attribute>) -> Vec<Attribute>
    {
        f(self.map(|b| *b).unwrap_or(Vec::new())).into_thin_attrs()
    }

    fn prepend(self, attrs: ThinAttributes) -> Self {
        attrs.map_thin_attrs(|mut attrs| {
            attrs.extend(self.into_attr_vec());
            attrs
        })
    }

    fn append(self, attrs: ThinAttributes) -> Self {
        self.map_thin_attrs(|mut self_| {
            self_.extend(attrs.into_attr_vec());
            self_
        })
    }

    fn update<F>(&mut self, f: F)
        where Self: Sized,
              F: FnOnce(ThinAttributes) -> ThinAttributes
    {
        let self_ = f(self.take());
        *self = self_;
    }

    fn as_attr_slice(&self) -> &[Attribute] {
        match *self {
            Some(ref b) => b,
            None => &[],
        }
    }

    fn into_attr_vec(self) -> Vec<Attribute> {
        match self {
            Some(b) => *b,
            None => Vec::new(),
        }
    }
}

pub trait AttributesExt {
    fn into_thin_attrs(self) -> ThinAttributes;
}

impl AttributesExt for Vec<Attribute> {
    fn into_thin_attrs(self) -> ThinAttributes {
        if self.len() == 0 {
            None
        } else {
            Some(Box::new(self))
        }
    }
}

/// A cheap way to add Attributes to an AST node.
pub trait WithAttrs {
    // FIXME: Could be extended to anything IntoIter<Item=Attribute>
    fn with_attrs(self, attrs: ThinAttributes) -> Self;
}

impl WithAttrs for P<Expr> {
    fn with_attrs(self, attrs: ThinAttributes) -> Self {
        self.map(|mut e| {
            e.attrs.update(|a| a.append(attrs));
            e
        })
    }
}

impl WithAttrs for P<Item> {
    fn with_attrs(self, attrs: ThinAttributes) -> Self {
        self.map(|Item { ident, attrs: mut ats, id, node, vis, span }| {
            ats.extend(attrs.into_attr_vec());
            Item {
                ident: ident,
                attrs: ats,
                id: id,
                node: node,
                vis: vis,
                span: span,
            }
        })
    }
}

impl WithAttrs for P<Local> {
    fn with_attrs(self, attrs: ThinAttributes) -> Self {
        self.map(|Local { pat, ty, init, id, span, attrs: mut ats }| {
            ats.update(|a| a.append(attrs));
            Local {
                pat: pat,
                ty: ty,
                init: init,
                id: id,
                span: span,
                attrs: ats,
            }
        })
    }
}

impl WithAttrs for P<Decl> {
    fn with_attrs(self, attrs: ThinAttributes) -> Self {
        self.map(|Spanned { span, node }| {
            Spanned {
                span: span,
                node: match node {
                    DeclLocal(local) => DeclLocal(local.with_attrs(attrs)),
                    DeclItem(item) => DeclItem(item.with_attrs(attrs)),
                }
            }
        })
    }
}

impl WithAttrs for P<Stmt> {
    fn with_attrs(self, attrs: ThinAttributes) -> Self {
        self.map(|Spanned { span, node }| {
            Spanned {
                span: span,
                node: match node {
                    StmtDecl(decl, id) => StmtDecl(decl.with_attrs(attrs), id),
                    StmtExpr(expr, id) => StmtExpr(expr.with_attrs(attrs), id),
                    StmtSemi(expr, id) => StmtSemi(expr.with_attrs(attrs), id),
                    StmtMac(mac, style, mut ats) => {
                        ats.update(|a| a.append(attrs));
                        StmtMac(mac, style, ats)
                    }
                },
            }
        })
    }
}
