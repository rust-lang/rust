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

pub use self::InlineAttr::*;
pub use self::StabilityLevel::*;
pub use self::ReprAttr::*;
pub use self::IntType::*;

use ast;
use ast::{AttrId, Attribute, Attribute_, MetaItem, MetaWord, MetaNameValue, MetaList};
use codemap::{Span, Spanned, spanned, dummy_spanned};
use codemap::BytePos;
use diagnostic::SpanHandler;
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::token::{InternedString, intern_and_get_ident};
use parse::token;
use ptr::P;

use std::cell::{RefCell, Cell};
use std::collections::BitSet;
use std::collections::HashSet;
use std::fmt;

thread_local! { static USED_ATTRS: RefCell<BitSet> = RefCell::new(BitSet::new()) }

pub fn mark_used(attr: &Attribute) {
    let AttrId(id) = attr.node.id;
    USED_ATTRS.with(|slot| slot.borrow_mut().insert(id));
}

pub fn is_used(attr: &Attribute) -> bool {
    let AttrId(id) = attr.node.id;
    USED_ATTRS.with(|slot| slot.borrow().contains(&id))
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
    fn meta_item_list<'a>(&'a self) -> Option<&'a [P<MetaItem>]>;

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
    fn meta_item_list<'a>(&'a self) -> Option<&'a [P<MetaItem>]> {
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

    fn meta_item_list<'a>(&'a self) -> Option<&'a [P<MetaItem>]> {
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
    fn meta_item_list<'a>(&'a self) -> Option<&'a [P<MetaItem>]> {
        (**self).meta_item_list()
    }
    fn span(&self) -> Span { (**self).span() }
}


pub trait AttributeMethods {
    fn meta<'a>(&'a self) -> &'a MetaItem;
    fn with_desugared_doc<T, F>(&self, f: F) -> T where
        F: FnOnce(&Attribute) -> T;
}

impl AttributeMethods for Attribute {
    /// Extract the MetaItem from inside this Attribute.
    fn meta<'a>(&'a self) -> &'a MetaItem {
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
            if self.node.style == ast::AttrOuter {
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
        style: ast::AttrInner,
        value: item,
        is_sugared_doc: false,
    })
}

/// Returns an outer attribute with the given value.
pub fn mk_attr_outer(id: AttrId, item: P<MetaItem>) -> Attribute {
    dummy_spanned(Attribute_ {
        id: id,
        style: ast::AttrOuter,
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

#[derive(Copy, PartialEq)]
pub enum InlineAttr {
    InlineNone,
    InlineHint,
    InlineAlways,
    InlineNever,
}

/// Determine what `#[inline]` attribute is present in `attrs`, if any.
pub fn find_inline_attr(attrs: &[Attribute]) -> InlineAttr {
    // FIXME (#2809)---validate the usage of #[inline] and #[inline]
    attrs.iter().fold(InlineNone, |ia,attr| {
        match attr.node.value.node {
            MetaWord(ref n) if *n == "inline" => {
                mark_used(attr);
                InlineHint
            }
            MetaList(ref n, ref items) if *n == "inline" => {
                mark_used(attr);
                if contains_name(&items[..], "always") {
                    InlineAlways
                } else if contains_name(&items[..], "never") {
                    InlineNever
                } else {
                    InlineHint
                }
            }
            _ => ia
        }
    })
}

/// True if `#[inline]` or `#[inline(always)]` is present in `attrs`.
pub fn requests_inline(attrs: &[Attribute]) -> bool {
    match find_inline_attr(attrs) {
        InlineHint | InlineAlways => true,
        InlineNone | InlineNever => false,
    }
}

/// Tests if a cfg-pattern matches the cfg set
pub fn cfg_matches(diagnostic: &SpanHandler, cfgs: &[P<MetaItem>], cfg: &ast::MetaItem) -> bool {
    match cfg.node {
        ast::MetaList(ref pred, ref mis) if &pred[..] == "any" =>
            mis.iter().any(|mi| cfg_matches(diagnostic, cfgs, &**mi)),
        ast::MetaList(ref pred, ref mis) if &pred[..] == "all" =>
            mis.iter().all(|mi| cfg_matches(diagnostic, cfgs, &**mi)),
        ast::MetaList(ref pred, ref mis) if &pred[..] == "not" => {
            if mis.len() != 1 {
                diagnostic.span_err(cfg.span, "expected 1 cfg-pattern");
                return false;
            }
            !cfg_matches(diagnostic, cfgs, &*mis[0])
        }
        ast::MetaList(ref pred, _) => {
            diagnostic.span_err(cfg.span, &format!("invalid predicate `{}`", pred));
            false
        },
        ast::MetaWord(_) | ast::MetaNameValue(..) => contains(cfgs, cfg),
    }
}

/// Represents the #[deprecated] and friends attributes.
#[derive(RustcEncodable,RustcDecodable,Clone,Debug)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: InternedString,
    pub since: Option<InternedString>,
    pub deprecated_since: Option<InternedString>,
    // The reason for the current stability level. If deprecated, the
    // reason for deprecation.
    pub reason: Option<InternedString>,
}

/// The available stability levels.
#[derive(RustcEncodable,RustcDecodable,PartialEq,PartialOrd,Clone,Debug,Copy)]
pub enum StabilityLevel {
    Unstable,
    Stable,
}

impl fmt::Display for StabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

fn find_stability_generic<'a,
                              AM: AttrMetaMethods,
                              I: Iterator<Item=&'a AM>>
                             (diagnostic: &SpanHandler, attrs: I, item_sp: Span)
                             -> (Option<Stability>, Vec<&'a AM>) {

    let mut stab: Option<Stability> = None;
    let mut deprecated: Option<(InternedString, Option<InternedString>)> = None;
    let mut used_attrs: Vec<&'a AM> = vec![];

    'outer: for attr in attrs {
        let tag = attr.name();
        let tag = &tag[..];
        if tag != "deprecated" && tag != "unstable" && tag != "stable" {
            continue // not a stability level
        }

        used_attrs.push(attr);

        let (feature, since, reason) = match attr.meta_item_list() {
            Some(metas) => {
                let mut feature = None;
                let mut since = None;
                let mut reason = None;
                for meta in metas.iter() {
                    if meta.name() == "feature" {
                        match meta.value_str() {
                            Some(v) => feature = Some(v),
                            None => {
                                diagnostic.span_err(meta.span, "incorrect meta item");
                                continue 'outer;
                            }
                        }
                    }
                    if &meta.name()[..] == "since" {
                        match meta.value_str() {
                            Some(v) => since = Some(v),
                            None => {
                                diagnostic.span_err(meta.span, "incorrect meta item");
                                continue 'outer;
                            }
                        }
                    }
                    if &meta.name()[..] == "reason" {
                        match meta.value_str() {
                            Some(v) => reason = Some(v),
                            None => {
                                diagnostic.span_err(meta.span, "incorrect meta item");
                                continue 'outer;
                            }
                        }
                    }
                }
                (feature, since, reason)
            }
            None => {
                diagnostic.span_err(attr.span(), "incorrect stability attribute type");
                continue
            }
        };

        // Deprecated tags don't require feature names
        if feature == None && tag != "deprecated" {
            diagnostic.span_err(attr.span(), "missing 'feature'");
        }

        // Unstable tags don't require a version
        if since == None && tag != "unstable" {
            diagnostic.span_err(attr.span(), "missing 'since'");
        }

        if tag == "unstable" || tag == "stable" {
            if stab.is_some() {
                diagnostic.span_err(item_sp, "multiple stability levels");
            }

            let level = match tag {
                "unstable" => Unstable,
                "stable" => Stable,
                _ => unreachable!()
            };

            stab = Some(Stability {
                level: level,
                feature: feature.unwrap_or(intern_and_get_ident("bogus")),
                since: since,
                deprecated_since: None,
                reason: reason
            });
        } else { // "deprecated"
            if deprecated.is_some() {
                diagnostic.span_err(item_sp, "multiple deprecated attributes");
            }

            deprecated = Some((since.unwrap_or(intern_and_get_ident("bogus")), reason));
        }
    }

    // Merge the deprecation info into the stability info
    if deprecated.is_some() {
        match stab {
            Some(ref mut s) => {
                let (since, reason) = deprecated.unwrap();
                s.deprecated_since = Some(since);
                s.reason = reason;
            }
            None => {
                diagnostic.span_err(item_sp, "deprecated attribute must be paired with \
                                              either stable or unstable attribute");
            }
        }
    }

    (stab, used_attrs)
}

/// Find the first stability attribute. `None` if none exists.
pub fn find_stability(diagnostic: &SpanHandler, attrs: &[Attribute],
                      item_sp: Span) -> Option<Stability> {
    let (s, used) = find_stability_generic(diagnostic, attrs.iter(), item_sp);
    for used in used { mark_used(used) }
    return s;
}

pub fn require_unique_names(diagnostic: &SpanHandler, metas: &[P<MetaItem>]) {
    let mut set = HashSet::new();
    for meta in metas {
        let name = meta.name();

        if !set.insert(name.clone()) {
            diagnostic.span_fatal(meta.span,
                                  &format!("duplicate meta item `{}`", name));
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
        "int" => Some(SignedInt(ast::TyIs(true))),
        "uint" => Some(UnsignedInt(ast::TyUs(true))),
        "isize" => Some(SignedInt(ast::TyIs(false))),
        "usize" => Some(UnsignedInt(ast::TyUs(false))),
        _ => None
    }
}

#[derive(PartialEq, Debug, RustcEncodable, RustcDecodable, Copy)]
pub enum ReprAttr {
    ReprAny,
    ReprInt(Span, IntType),
    ReprExtern,
    ReprPacked,
}

impl ReprAttr {
    pub fn is_ffi_safe(&self) -> bool {
        match *self {
            ReprAny => false,
            ReprInt(_sp, ity) => ity.is_ffi_safe(),
            ReprExtern => true,
            ReprPacked => false
        }
    }
}

#[derive(Eq, Hash, PartialEq, Debug, RustcEncodable, RustcDecodable, Copy)]
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
            SignedInt(ast::TyIs(_)) | UnsignedInt(ast::TyUs(_)) => false
        }
    }
}
