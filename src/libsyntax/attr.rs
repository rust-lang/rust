// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Functions dealing with attributes and meta items

use ast;
use ast::{Attribute, Attribute_, MetaItem, MetaWord, MetaNameValue, MetaList};
use codemap::{Span, Spanned, spanned, dummy_spanned};
use codemap::BytePos;
use diagnostic::SpanHandler;
use parse::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::token::InternedString;
use parse::token;
use crateid::CrateId;

use collections::HashSet;
use std::vec_ng::Vec;

pub trait AttrMetaMethods {
    // This could be changed to `fn check_name(&self, name: InternedString) ->
    // bool` which would facilitate a side table recording which
    // attributes/meta items are used/unused.

    /// Retrieve the name of the meta item, e.g. foo in #[foo],
    /// #[foo="bar"] and #[foo(bar)]
    fn name(&self) -> InternedString;

    /**
     * Gets the string value if self is a MetaNameValue variant
     * containing a string, otherwise None.
     */
    fn value_str(&self) -> Option<InternedString>;
    /// Gets a list of inner meta items from a list MetaItem type.
    fn meta_item_list<'a>(&'a self) -> Option<&'a [@MetaItem]>;

    /**
     * If the meta item is a name-value type with a string value then returns
     * a tuple containing the name and string value, otherwise `None`
     */
    fn name_str_pair(&self) -> Option<(InternedString,InternedString)>;
}

impl AttrMetaMethods for Attribute {
    fn name(&self) -> InternedString { self.meta().name() }
    fn value_str(&self) -> Option<InternedString> {
        self.meta().value_str()
    }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [@MetaItem]> {
        self.node.value.meta_item_list()
    }
    fn name_str_pair(&self) -> Option<(InternedString,InternedString)> {
        self.meta().name_str_pair()
    }
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

    fn meta_item_list<'a>(&'a self) -> Option<&'a [@MetaItem]> {
        match self.node {
            MetaList(_, ref l) => Some(l.as_slice()),
            _ => None
        }
    }

    fn name_str_pair(&self) -> Option<(InternedString,InternedString)> {
        self.value_str().map(|s| (self.name(), s))
    }
}

// Annoying, but required to get test_cfg to work
impl AttrMetaMethods for @MetaItem {
    fn name(&self) -> InternedString { (**self).name() }
    fn value_str(&self) -> Option<InternedString> { (**self).value_str() }
    fn meta_item_list<'a>(&'a self) -> Option<&'a [@MetaItem]> {
        (**self).meta_item_list()
    }
    fn name_str_pair(&self) -> Option<(InternedString,InternedString)> {
        (**self).name_str_pair()
    }
}


pub trait AttributeMethods {
    fn meta(&self) -> @MetaItem;
    fn desugar_doc(&self) -> Attribute;
}

impl AttributeMethods for Attribute {
    /// Extract the MetaItem from inside this Attribute.
    fn meta(&self) -> @MetaItem {
        self.node.value
    }

    /// Convert self to a normal #[doc="foo"] comment, if it is a
    /// comment like `///` or `/** */`. (Returns self unchanged for
    /// non-sugared doc attributes.)
    fn desugar_doc(&self) -> Attribute {
        if self.node.is_sugared_doc {
            let comment = self.value_str().unwrap();
            let meta = mk_name_value_item_str(
                InternedString::new("doc"),
                token::intern_and_get_ident(strip_doc_comment_decoration(
                        comment.get())));
            mk_attr(meta)
        } else {
            *self
        }
    }
}

/* Constructors */

pub fn mk_name_value_item_str(name: InternedString, value: InternedString)
                              -> @MetaItem {
    let value_lit = dummy_spanned(ast::LitStr(value, ast::CookedStr));
    mk_name_value_item(name, value_lit)
}

pub fn mk_name_value_item(name: InternedString, value: ast::Lit)
                          -> @MetaItem {
    @dummy_spanned(MetaNameValue(name, value))
}

pub fn mk_list_item(name: InternedString, items: Vec<@MetaItem> ) -> @MetaItem {
    @dummy_spanned(MetaList(name, items))
}

pub fn mk_word_item(name: InternedString) -> @MetaItem {
    @dummy_spanned(MetaWord(name))
}

pub fn mk_attr(item: @MetaItem) -> Attribute {
    dummy_spanned(Attribute_ {
        style: ast::AttrInner,
        value: item,
        is_sugared_doc: false,
    })
}

pub fn mk_sugared_doc_attr(text: InternedString, lo: BytePos, hi: BytePos)
                           -> Attribute {
    let style = doc_comment_style(text.get());
    let lit = spanned(lo, hi, ast::LitStr(text, ast::CookedStr));
    let attr = Attribute_ {
        style: style,
        value: @spanned(lo, hi, MetaNameValue(InternedString::new("doc"),
                                              lit)),
        is_sugared_doc: true
    };
    spanned(lo, hi, attr)
}

/* Searching */
/// Check if `needle` occurs in `haystack` by a structural
/// comparison. This is slightly subtle, and relies on ignoring the
/// span included in the `==` comparison a plain MetaItem.
pub fn contains(haystack: &[@ast::MetaItem],
                needle: @ast::MetaItem) -> bool {
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
        item.name().equiv(&name)
    })
}

pub fn first_attr_value_str_by_name(attrs: &[Attribute], name: &str)
                                 -> Option<InternedString> {
    attrs.iter()
        .find(|at| at.name().equiv(&name))
        .and_then(|at| at.value_str())
}

pub fn last_meta_item_value_str_by_name(items: &[@MetaItem], name: &str)
                                     -> Option<InternedString> {
    items.rev_iter()
         .find(|mi| mi.name().equiv(&name))
         .and_then(|i| i.value_str())
}

/* Higher-level applications */

pub fn sort_meta_items(items: &[@MetaItem]) -> Vec<@MetaItem> {
    // This is sort of stupid here, but we need to sort by
    // human-readable strings.
    let mut v = items.iter()
        .map(|&mi| (mi.name(), mi))
        .collect::<Vec<(InternedString, @MetaItem)> >();

    v.sort_by(|&(ref a, _), &(ref b, _)| a.cmp(b));

    // There doesn't seem to be a more optimal way to do this
    v.move_iter().map(|(_, m)| {
        match m.node {
            MetaList(ref n, ref mis) => {
                @Spanned {
                    node: MetaList((*n).clone(),
                                   sort_meta_items(mis.as_slice())),
                    .. /*bad*/ (*m).clone()
                }
            }
            _ => m
        }
    }).collect()
}

/**
 * From a list of crate attributes get only the meta_items that affect crate
 * linkage
 */
pub fn find_linkage_metas(attrs: &[Attribute]) -> Vec<@MetaItem> {
    let mut result = Vec::new();
    for attr in attrs.iter().filter(|at| at.name().equiv(&("link"))) {
        match attr.meta().node {
            MetaList(_, ref items) => result.push_all(items.as_slice()),
            _ => ()
        }
    }
    result
}

pub fn find_crateid(attrs: &[Attribute]) -> Option<CrateId> {
    match first_attr_value_str_by_name(attrs, "crate_id") {
        None => None,
        Some(id) => from_str::<CrateId>(id.get()),
    }
}

#[deriving(Eq)]
pub enum InlineAttr {
    InlineNone,
    InlineHint,
    InlineAlways,
    InlineNever,
}

/// True if something like #[inline] is found in the list of attrs.
pub fn find_inline_attr(attrs: &[Attribute]) -> InlineAttr {
    // FIXME (#2809)---validate the usage of #[inline] and #[inline]
    attrs.iter().fold(InlineNone, |ia,attr| {
        match attr.node.value.node {
          MetaWord(ref n) if n.equiv(&("inline")) => InlineHint,
          MetaList(ref n, ref items) if n.equiv(&("inline")) => {
            if contains_name(items.as_slice(), "always") {
                InlineAlways
            } else if contains_name(items.as_slice(), "never") {
                InlineNever
            } else {
                InlineHint
            }
          }
          _ => ia
        }
    })
}

/// Tests if any `cfg(...)` meta items in `metas` match `cfg`. e.g.
///
/// test_cfg(`[foo="a", bar]`, `[cfg(foo), cfg(bar)]`) == true
/// test_cfg(`[foo="a", bar]`, `[cfg(not(bar))]`) == false
/// test_cfg(`[foo="a", bar]`, `[cfg(bar, foo="a")]`) == true
/// test_cfg(`[foo="a", bar]`, `[cfg(bar, foo="b")]`) == false
pub fn test_cfg<AM: AttrMetaMethods, It: Iterator<AM>>
    (cfg: &[@MetaItem], mut metas: It) -> bool {
    // having no #[cfg(...)] attributes counts as matching.
    let mut no_cfgs = true;

    // this would be much nicer as a chain of iterator adaptors, but
    // this doesn't work.
    let some_cfg_matches = metas.any(|mi| {
        debug!("testing name: {}", mi.name());
        if mi.name().equiv(&("cfg")) { // it is a #[cfg()] attribute
            debug!("is cfg");
            no_cfgs = false;
             // only #[cfg(...)] ones are understood.
            match mi.meta_item_list() {
                Some(cfg_meta) => {
                    debug!("is cfg(...)");
                    cfg_meta.iter().all(|cfg_mi| {
                        debug!("cfg({}[...])", cfg_mi.name());
                        match cfg_mi.node {
                            ast::MetaList(ref s, ref not_cfgs)
                            if s.equiv(&("not")) => {
                                debug!("not!");
                                // inside #[cfg(not(...))], so these need to all
                                // not match.
                                !not_cfgs.iter().all(|mi| {
                                    debug!("cfg(not({}[...]))", mi.name());
                                    contains(cfg, *mi)
                                })
                            }
                            _ => contains(cfg, *cfg_mi)
                        }
                    })
                }
                None => false
            }
        } else {
            false
        }
    });
    debug!("test_cfg (no_cfgs={}, some_cfg_matches={})", no_cfgs, some_cfg_matches);
    no_cfgs || some_cfg_matches
}

/// Represents the #[deprecated="foo"] (etc) attributes.
pub struct Stability {
    level: StabilityLevel,
    text: Option<InternedString>
}

/// The available stability levels.
#[deriving(Eq,Ord,Clone,Show)]
pub enum StabilityLevel {
    Deprecated,
    Experimental,
    Unstable,
    Stable,
    Frozen,
    Locked
}

/// Find the first stability attribute. `None` if none exists.
pub fn find_stability<AM: AttrMetaMethods, It: Iterator<AM>>(mut metas: It)
                      -> Option<Stability> {
    for m in metas {
        let level = match m.name().get() {
            "deprecated" => Deprecated,
            "experimental" => Experimental,
            "unstable" => Unstable,
            "stable" => Stable,
            "frozen" => Frozen,
            "locked" => Locked,
            _ => continue // not a stability level
        };

        return Some(Stability {
                level: level,
                text: m.value_str()
            });
    }
    None
}

pub fn require_unique_names(diagnostic: &SpanHandler, metas: &[@MetaItem]) {
    let mut set = HashSet::new();
    for meta in metas.iter() {
        let name = meta.name();

        if !set.insert(name.clone()) {
            diagnostic.span_fatal(meta.span,
                                  format!("duplicate meta item `{}`", name));
        }
    }
}


/**
 * Fold this over attributes to parse #[repr(...)] forms.
 *
 * Valid repr contents: any of the primitive integral type names (see
 * `int_type_of_word`, below) to specify the discriminant type; and `C`, to use
 * the same discriminant size that the corresponding C enum would.  These are
 * not allowed on univariant or zero-variant enums, which have no discriminant.
 *
 * If a discriminant type is so specified, then the discriminant will be
 * present (before fields, if any) with that type; reprensentation
 * optimizations which would remove it will not be done.
 */
pub fn find_repr_attr(diagnostic: &SpanHandler, attr: @ast::MetaItem, acc: ReprAttr)
    -> ReprAttr {
    let mut acc = acc;
    match attr.node {
        ast::MetaList(ref s, ref items) if s.equiv(&("repr")) => {
            for item in items.iter() {
                match item.node {
                    ast::MetaWord(ref word) => {
                        let hint = match word.get() {
                            // Can't use "extern" because it's not a lexical identifier.
                            "C" => ReprExtern,
                            _ => match int_type_of_word(word.get()) {
                                Some(ity) => ReprInt(item.span, ity),
                                None => {
                                    // Not a word we recognize
                                    diagnostic.span_err(item.span,
                                                        "unrecognized representation hint");
                                    ReprAny
                                }
                            }
                        };
                        if hint != ReprAny {
                            if acc == ReprAny {
                                acc = hint;
                            } else if acc != hint {
                                diagnostic.span_warn(item.span,
                                                     "conflicting representation hint ignored")
                            }
                        }
                    }
                    // Not a word:
                    _ => diagnostic.span_err(item.span, "unrecognized representation hint")
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
        "int" => Some(SignedInt(ast::TyI)),
        "uint" => Some(UnsignedInt(ast::TyU)),
        _ => None
    }
}

#[deriving(Eq, Show)]
pub enum ReprAttr {
    ReprAny,
    ReprInt(Span, IntType),
    ReprExtern
}

impl ReprAttr {
    pub fn is_ffi_safe(&self) -> bool {
        match *self {
            ReprAny => false,
            ReprInt(_sp, ity) => ity.is_ffi_safe(),
            ReprExtern => true
        }
    }
}

#[deriving(Eq, Show)]
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
            _ => false
        }
    }
}
