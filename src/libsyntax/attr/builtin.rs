//! Parsing and validation of builtin attributes

use crate::ast::{self, Attribute, MetaItem, NestedMetaItem};
use crate::feature_gate::{Features, GatedCfg};
use crate::print::pprust;
use crate::sess::ParseSess;

use errors::{Applicability, Handler};
use std::num::NonZeroU32;
use syntax_pos::hygiene::Transparency;
use syntax_pos::{symbol::Symbol, symbol::sym, Span};
use rustc_macros::HashStable_Generic;

use super::{mark_used, MetaItemKind};

use rustc_error_codes::*;

enum AttrError {
    MultipleItem(String),
    UnknownMetaItem(String, &'static [&'static str]),
    MissingSince,
    MissingFeature,
    MultipleStabilityLevels,
    UnsupportedLiteral(&'static str, /* is_bytestr */ bool),
}

/// A template that the attribute input must match.
/// Only top-level shape (`#[attr]` vs `#[attr(...)]` vs `#[attr = ...]`) is considered now.
#[derive(Clone, Copy)]
pub struct AttributeTemplate {
    pub word: bool,
    pub list: Option<&'static str>,
    pub name_value_str: Option<&'static str>,
}

impl AttributeTemplate {
    pub fn only_word() -> Self {
        Self { word: true, list: None, name_value_str: None }
    }

    /// Checks that the given meta-item is compatible with this template.
    pub fn compatible(&self, meta_item_kind: &ast::MetaItemKind) -> bool {
        match meta_item_kind {
            ast::MetaItemKind::Word => self.word,
            ast::MetaItemKind::List(..) => self.list.is_some(),
            ast::MetaItemKind::NameValue(lit) if lit.kind.is_str() => self.name_value_str.is_some(),
            ast::MetaItemKind::NameValue(..) => false,
        }
    }
}

fn handle_errors(sess: &ParseSess, span: Span, error: AttrError) {
    let diag = &sess.span_diagnostic;
    match error {
        AttrError::MultipleItem(item) => span_err!(diag, span, E0538,
                                                   "multiple '{}' items", item),
        AttrError::UnknownMetaItem(item, expected) => {
            let expected = expected
                .iter()
                .map(|name| format!("`{}`", name))
                .collect::<Vec<_>>();
            struct_span_err!(diag, span, E0541, "unknown meta item '{}'", item)
                .span_label(span, format!("expected one of {}", expected.join(", ")))
                .emit();
        }
        AttrError::MissingSince => span_err!(diag, span, E0542, "missing 'since'"),
        AttrError::MissingFeature => span_err!(diag, span, E0546, "missing 'feature'"),
        AttrError::MultipleStabilityLevels => span_err!(diag, span, E0544,
                                                        "multiple stability levels"),
        AttrError::UnsupportedLiteral(
            msg,
            is_bytestr,
        ) => {
            let mut err = struct_span_err!(diag, span, E0565, "{}", msg);
            if is_bytestr {
                if let Ok(lint_str) = sess.source_map().span_to_snippet(span) {
                    err.span_suggestion(
                        span,
                        "consider removing the prefix",
                        format!("{}", &lint_str[1..]),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            err.emit();
        }
    }
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub enum InlineAttr {
    None,
    Hint,
    Always,
    Never,
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub enum OptimizeAttr {
    None,
    Speed,
    Size,
}

#[derive(Copy, Clone, PartialEq)]
pub enum UnwindAttr {
    Allowed,
    Aborts,
}

/// Determine what `#[unwind]` attribute is present in `attrs`, if any.
pub fn find_unwind_attr(diagnostic: Option<&Handler>, attrs: &[Attribute]) -> Option<UnwindAttr> {
    attrs.iter().fold(None, |ia, attr| {
        if attr.check_name(sym::unwind) {
            if let Some(meta) = attr.meta() {
                if let MetaItemKind::List(items) = meta.kind {
                    if items.len() == 1 {
                        if items[0].check_name(sym::allowed) {
                            return Some(UnwindAttr::Allowed);
                        } else if items[0].check_name(sym::aborts) {
                            return Some(UnwindAttr::Aborts);
                        }
                    }

                    diagnostic.map(|d| {
                        struct_span_err!(d, attr.span, E0633, "malformed `unwind` attribute input")
                            .span_label(attr.span, "invalid argument")
                            .span_suggestions(
                                attr.span,
                                "the allowed arguments are `allowed` and `aborts`",
                                (vec!["allowed", "aborts"]).into_iter()
                                    .map(|s| format!("#[unwind({})]", s)),
                                Applicability::MachineApplicable,
                            ).emit();
                    });
                }
            }
        }

        ia
    })
}

/// Represents the #[stable], #[unstable], #[rustc_{deprecated,const_unstable}] attributes.
#[derive(RustcEncodable, RustcDecodable, Copy, Clone, Debug,
         PartialEq, Eq, Hash, HashStable_Generic)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    pub rustc_depr: Option<RustcDeprecation>,
    /// `None` means the function is stable but needs to be a stable const fn, too
    /// `Some` contains the feature gate required to be able to use the function
    /// as const fn
    pub const_stability: Option<Symbol>,
    /// whether the function has a `#[rustc_promotable]` attribute
    pub promotable: bool,
    /// whether the function has a `#[rustc_allow_const_fn_ptr]` attribute
    pub allow_const_fn_ptr: bool,
}

/// The available stability levels.
#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd,
         Copy, Clone, Debug, Eq, Hash, HashStable_Generic)]
pub enum StabilityLevel {
    // Reason for the current stability level and the relevant rust-lang issue
    Unstable { reason: Option<Symbol>, issue: Option<NonZeroU32>, is_soft: bool },
    Stable { since: Symbol },
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool {
        if let StabilityLevel::Unstable {..} = *self {
            true
        } else {
            false
        }
    }
    pub fn is_stable(&self) -> bool {
        if let StabilityLevel::Stable {..} = *self {
            true
        } else {
            false
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, PartialEq, PartialOrd,
         Copy, Clone, Debug, Eq, Hash, HashStable_Generic)]
pub struct RustcDeprecation {
    pub since: Symbol,
    pub reason: Symbol,
    /// A text snippet used to completely replace any use of the deprecated item in an expression.
    pub suggestion: Option<Symbol>,
}

/// Checks if `attrs` contains an attribute like `#![feature(feature_name)]`.
/// This will not perform any "sanity checks" on the form of the attributes.
pub fn contains_feature_attr(attrs: &[Attribute], feature_name: Symbol) -> bool {
    attrs.iter().any(|item| {
        item.check_name(sym::feature) &&
        item.meta_item_list().map(|list| {
            list.iter().any(|mi| mi.is_word() && mi.check_name(feature_name))
        }).unwrap_or(false)
    })
}

/// Collects stability info from all stability attributes in `attrs`.
/// Returns `None` if no stability attributes are found.
pub fn find_stability(sess: &ParseSess, attrs: &[Attribute],
                      item_sp: Span) -> Option<Stability> {
    find_stability_generic(sess, attrs.iter(), item_sp)
}

fn find_stability_generic<'a, I>(sess: &ParseSess,
                                 attrs_iter: I,
                                 item_sp: Span)
                                 -> Option<Stability>
    where I: Iterator<Item = &'a Attribute>
{
    use StabilityLevel::*;

    let mut stab: Option<Stability> = None;
    let mut rustc_depr: Option<RustcDeprecation> = None;
    let mut rustc_const_unstable: Option<Symbol> = None;
    let mut promotable = false;
    let mut allow_const_fn_ptr = false;
    let diagnostic = &sess.span_diagnostic;

    'outer: for attr in attrs_iter {
        if ![
            sym::rustc_deprecated,
            sym::rustc_const_unstable,
            sym::unstable,
            sym::stable,
            sym::rustc_promotable,
            sym::rustc_allow_const_fn_ptr,
        ].iter().any(|&s| attr.has_name(s)) {
            continue // not a stability level
        }

        mark_used(attr);

        let meta = attr.meta();

        if attr.has_name(sym::rustc_promotable) {
            promotable = true;
        }
        if attr.has_name(sym::rustc_allow_const_fn_ptr) {
            allow_const_fn_ptr = true;
        }
        // attributes with data
        else if let Some(MetaItem { kind: MetaItemKind::List(ref metas), .. }) = meta {
            let meta = meta.as_ref().unwrap();
            let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                if item.is_some() {
                    handle_errors(
                        sess,
                        meta.span,
                        AttrError::MultipleItem(pprust::path_to_string(&meta.path)),
                    );
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
                            match mi.name_or_empty() {
                                $(
                                    sym::$name => if !get(mi, &mut $name) { continue 'outer },
                                )+
                                _ => {
                                    let expected = &[ $( stringify!($name) ),+ ];
                                    handle_errors(
                                        sess,
                                        mi.span,
                                        AttrError::UnknownMetaItem(
                                            pprust::path_to_string(&mi.path),
                                            expected,
                                        ),
                                    );
                                    continue 'outer
                                }
                            }
                        } else {
                            handle_errors(
                                sess,
                                meta.span(),
                                AttrError::UnsupportedLiteral(
                                    "unsupported literal",
                                    false,
                                ),
                            );
                            continue 'outer
                        }
                    }
                }
            }

            match meta.name_or_empty() {
                sym::rustc_deprecated => {
                    if rustc_depr.is_some() {
                        span_err!(diagnostic, item_sp, E0540,
                                  "multiple rustc_deprecated attributes");
                        continue 'outer
                    }

                    get_meta!(since, reason, suggestion);

                    match (since, reason) {
                        (Some(since), Some(reason)) => {
                            rustc_depr = Some(RustcDeprecation {
                                since,
                                reason,
                                suggestion,
                            })
                        }
                        (None, _) => {
                            handle_errors(sess, attr.span, AttrError::MissingSince);
                            continue
                        }
                        _ => {
                            span_err!(diagnostic, attr.span, E0543, "missing 'reason'");
                            continue
                        }
                    }
                }
                sym::rustc_const_unstable => {
                    if rustc_const_unstable.is_some() {
                        span_err!(diagnostic, item_sp, E0553,
                                  "multiple rustc_const_unstable attributes");
                        continue 'outer
                    }

                    get_meta!(feature);
                    if let Some(feature) = feature {
                        rustc_const_unstable = Some(feature);
                    } else {
                        span_err!(diagnostic, attr.span, E0629, "missing 'feature'");
                        continue
                    }
                }
                sym::unstable => {
                    if stab.is_some() {
                        handle_errors(sess, attr.span, AttrError::MultipleStabilityLevels);
                        break
                    }

                    let mut feature = None;
                    let mut reason = None;
                    let mut issue = None;
                    let mut is_soft = false;
                    for meta in metas {
                        if let Some(mi) = meta.meta_item() {
                            match mi.name_or_empty() {
                                sym::feature => if !get(mi, &mut feature) { continue 'outer },
                                sym::reason => if !get(mi, &mut reason) { continue 'outer },
                                sym::issue => if !get(mi, &mut issue) { continue 'outer },
                                sym::soft => {
                                    if !mi.is_word() {
                                        let msg = "`soft` should not have any arguments";
                                        sess.span_diagnostic.span_err(mi.span, msg);
                                    }
                                    is_soft = true;
                                }
                                _ => {
                                    handle_errors(
                                        sess,
                                        meta.span(),
                                        AttrError::UnknownMetaItem(
                                            pprust::path_to_string(&mi.path),
                                            &["feature", "reason", "issue", "soft"]
                                        ),
                                    );
                                    continue 'outer
                                }
                            }
                        } else {
                            handle_errors(
                                sess,
                                meta.span(),
                                AttrError::UnsupportedLiteral(
                                    "unsupported literal",
                                    false,
                                ),
                            );
                            continue 'outer
                        }
                    }

                    match (feature, reason, issue) {
                        (Some(feature), reason, Some(issue)) => {
                            let issue = match &*issue.as_str() {
                                // FIXME(rossmacarthur): remove "0" because "none" should be used
                                // See #41260
                                "none" | "0" => None,
                                issue => {
                                    if let Ok(num) = issue.parse() {
                                        NonZeroU32::new(num)
                                    } else {
                                        span_err!(
                                            diagnostic,
                                            attr.span,
                                            E0545,
                                            "incorrect 'issue'"
                                        );
                                        continue
                                    }
                                }
                            };
                            stab = Some(Stability {
                                level: Unstable {
                                    reason,
                                    issue,
                                    is_soft,
                                },
                                feature,
                                rustc_depr: None,
                                const_stability: None,
                                promotable: false,
                                allow_const_fn_ptr: false,
                            })
                        }
                        (None, _, _) => {
                            handle_errors(sess, attr.span, AttrError::MissingFeature);
                            continue
                        }
                        _ => {
                            span_err!(diagnostic, attr.span, E0547, "missing 'issue'");
                            continue
                        }
                    }
                }
                sym::stable => {
                    if stab.is_some() {
                        handle_errors(sess, attr.span, AttrError::MultipleStabilityLevels);
                        break
                    }

                    let mut feature = None;
                    let mut since = None;
                    for meta in metas {
                        match meta {
                            NestedMetaItem::MetaItem(mi) => {
                                match mi.name_or_empty() {
                                    sym::feature => if !get(mi, &mut feature) { continue 'outer },
                                    sym::since => if !get(mi, &mut since) { continue 'outer },
                                    _ => {
                                        handle_errors(
                                            sess,
                                            meta.span(),
                                            AttrError::UnknownMetaItem(
                                                pprust::path_to_string(&mi.path),
                                                &["since", "note"],
                                            ),
                                        );
                                        continue 'outer
                                    }
                                }
                            },
                            NestedMetaItem::Literal(lit) => {
                                handle_errors(
                                    sess,
                                    lit.span,
                                    AttrError::UnsupportedLiteral(
                                        "unsupported literal",
                                        false,
                                    ),
                                );
                                continue 'outer
                            }
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
                                const_stability: None,
                                promotable: false,
                                allow_const_fn_ptr: false,
                            })
                        }
                        (None, _) => {
                            handle_errors(sess, attr.span, AttrError::MissingFeature);
                            continue
                        }
                        _ => {
                            handle_errors(sess, attr.span, AttrError::MissingSince);
                            continue
                        }
                    }
                }
                _ => unreachable!()
            }
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
    if let Some(feature) = rustc_const_unstable {
        if let Some(ref mut stab) = stab {
            stab.const_stability = Some(feature);
        } else {
            span_err!(diagnostic, item_sp, E0630,
                      "rustc_const_unstable attribute must be paired with \
                       either stable or unstable attribute");
        }
    }

    // Merge the const-unstable info into the stability info
    if promotable || allow_const_fn_ptr {
        if let Some(ref mut stab) = stab {
            stab.promotable = promotable;
            stab.allow_const_fn_ptr = allow_const_fn_ptr;
        } else {
            span_err!(diagnostic, item_sp, E0717,
                      "rustc_promotable and rustc_allow_const_fn_ptr attributes \
                      must be paired with either stable or unstable attribute");
        }
    }

    stab
}

pub fn find_crate_name(attrs: &[Attribute]) -> Option<Symbol> {
    super::first_attr_value_str_by_name(attrs, sym::crate_name)
}

/// Tests if a cfg-pattern matches the cfg set
pub fn cfg_matches(cfg: &ast::MetaItem, sess: &ParseSess, features: Option<&Features>) -> bool {
    eval_condition(cfg, sess, &mut |cfg| {
        if let (Some(feats), Some(gated_cfg)) = (features, GatedCfg::gate(cfg)) {
            gated_cfg.check_and_emit(sess, feats);
        }
        let error = |span, msg| { sess.span_diagnostic.span_err(span, msg); true };
        if cfg.path.segments.len() != 1 {
            return error(cfg.path.span, "`cfg` predicate key must be an identifier");
        }
        match &cfg.kind {
            MetaItemKind::List(..) => {
                error(cfg.span, "unexpected parentheses after `cfg` predicate key")
            }
            MetaItemKind::NameValue(lit) if !lit.kind.is_str() => {
                handle_errors(
                    sess,
                    lit.span,
                    AttrError::UnsupportedLiteral(
                        "literal in `cfg` predicate value must be a string",
                        lit.kind.is_bytestr()
                    ),
                );
                true
            }
            MetaItemKind::NameValue(..) | MetaItemKind::Word => {
                let ident = cfg.ident().expect("multi-segment cfg predicate");
                sess.config.contains(&(ident.name, cfg.value_str()))
            }
        }
    })
}

/// Evaluate a cfg-like condition (with `any` and `all`), using `eval` to
/// evaluate individual items.
pub fn eval_condition<F>(cfg: &ast::MetaItem, sess: &ParseSess, eval: &mut F)
                         -> bool
    where F: FnMut(&ast::MetaItem) -> bool
{
    match cfg.kind {
        ast::MetaItemKind::List(ref mis) => {
            for mi in mis.iter() {
                if !mi.is_meta_item() {
                    handle_errors(
                        sess,
                        mi.span(),
                        AttrError::UnsupportedLiteral(
                            "unsupported literal",
                            false
                        ),
                    );
                    return false;
                }
            }

            // The unwraps below may look dangerous, but we've already asserted
            // that they won't fail with the loop above.
            match cfg.name_or_empty() {
                sym::any => mis.iter().any(|mi| {
                    eval_condition(mi.meta_item().unwrap(), sess, eval)
                }),
                sym::all => mis.iter().all(|mi| {
                    eval_condition(mi.meta_item().unwrap(), sess, eval)
                }),
                sym::not => {
                    if mis.len() != 1 {
                        span_err!(sess.span_diagnostic, cfg.span, E0536, "expected 1 cfg-pattern");
                        return false;
                    }

                    !eval_condition(mis[0].meta_item().unwrap(), sess, eval)
                },
                _ => {
                    span_err!(
                        sess.span_diagnostic, cfg.span, E0537,
                        "invalid predicate `{}`",
                        pprust::path_to_string(&cfg.path)
                    );
                    false
                }
            }
        },
        ast::MetaItemKind::Word | ast::MetaItemKind::NameValue(..) => {
            eval(cfg)
        }
    }
}

#[derive(RustcEncodable, RustcDecodable, Clone, HashStable_Generic)]
pub struct Deprecation {
    pub since: Option<Symbol>,
    pub note: Option<Symbol>,
}

/// Finds the deprecation attribute. `None` if none exists.
pub fn find_deprecation(sess: &ParseSess, attrs: &[Attribute],
                        item_sp: Span) -> Option<Deprecation> {
    find_deprecation_generic(sess, attrs.iter(), item_sp)
}

fn find_deprecation_generic<'a, I>(sess: &ParseSess,
                                   attrs_iter: I,
                                   item_sp: Span)
                                   -> Option<Deprecation>
    where I: Iterator<Item = &'a Attribute>
{
    let mut depr: Option<Deprecation> = None;
    let diagnostic = &sess.span_diagnostic;

    'outer: for attr in attrs_iter {
        if !attr.check_name(sym::deprecated) {
            continue;
        }

        if depr.is_some() {
            span_err!(diagnostic, item_sp, E0550, "multiple deprecated attributes");
            break
        }

        let meta = match attr.meta() {
            Some(meta) => meta,
            None => continue,
        };
        depr = match &meta.kind {
            MetaItemKind::Word => Some(Deprecation { since: None, note: None }),
            MetaItemKind::NameValue(..) => {
                meta.value_str().map(|note| {
                    Deprecation { since: None, note: Some(note) }
                })
            }
            MetaItemKind::List(list) => {
                let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                    if item.is_some() {
                        handle_errors(
                            sess,
                            meta.span,
                            AttrError::MultipleItem(pprust::path_to_string(&meta.path)),
                        );
                        return false
                    }
                    if let Some(v) = meta.value_str() {
                        *item = Some(v);
                        true
                    } else {
                        if let Some(lit) = meta.name_value_literal() {
                            handle_errors(
                                sess,
                                lit.span,
                                AttrError::UnsupportedLiteral(
                                    "literal in `deprecated` \
                                    value must be a string",
                                    lit.kind.is_bytestr()
                                ),
                            );
                        } else {
                            span_err!(diagnostic, meta.span, E0551, "incorrect meta item");
                        }

                        false
                    }
                };

                let mut since = None;
                let mut note = None;
                for meta in list {
                    match meta {
                        NestedMetaItem::MetaItem(mi) => {
                            match mi.name_or_empty() {
                                sym::since => if !get(mi, &mut since) { continue 'outer },
                                sym::note => if !get(mi, &mut note) { continue 'outer },
                                _ => {
                                    handle_errors(
                                        sess,
                                        meta.span(),
                                        AttrError::UnknownMetaItem(
                                            pprust::path_to_string(&mi.path),
                                            &["since", "note"],
                                        ),
                                    );
                                    continue 'outer
                                }
                            }
                        }
                        NestedMetaItem::Literal(lit) => {
                            handle_errors(
                                sess,
                                lit.span,
                                AttrError::UnsupportedLiteral(
                                    "item in `deprecated` must be a key/value pair",
                                    false,
                                ),
                            );
                            continue 'outer
                        }
                    }
                }

                Some(Deprecation { since, note })
            }
        };
    }

    depr
}

#[derive(PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone)]
pub enum ReprAttr {
    ReprInt(IntType),
    ReprC,
    ReprPacked(u32),
    ReprSimd,
    ReprTransparent,
    ReprAlign(u32),
}

#[derive(Eq, PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone, HashStable_Generic)]
pub enum IntType {
    SignedInt(ast::IntTy),
    UnsignedInt(ast::UintTy)
}

impl IntType {
    #[inline]
    pub fn is_signed(self) -> bool {
        use IntType::*;

        match self {
            SignedInt(..) => true,
            UnsignedInt(..) => false
        }
    }
}

/// Parse #[repr(...)] forms.
///
/// Valid repr contents: any of the primitive integral type names (see
/// `int_type_of_word`, below) to specify enum discriminant type; `C`, to use
/// the same discriminant size that the corresponding C enum would or C
/// structure layout, `packed` to remove padding, and `transparent` to elegate representation
/// concerns to the only non-ZST field.
pub fn find_repr_attrs(sess: &ParseSess, attr: &Attribute) -> Vec<ReprAttr> {
    use ReprAttr::*;

    let mut acc = Vec::new();
    let diagnostic = &sess.span_diagnostic;
    if attr.has_name(sym::repr) {
        if let Some(items) = attr.meta_item_list() {
            mark_used(attr);
            for item in items {
                if !item.is_meta_item() {
                    handle_errors(
                        sess,
                        item.span(),
                        AttrError::UnsupportedLiteral(
                            "meta item in `repr` must be an identifier",
                            false,
                        ),
                    );
                    continue
                }

                let mut recognised = false;
                if item.is_word() {
                    let hint = match item.name_or_empty() {
                        sym::C => Some(ReprC),
                        sym::packed => Some(ReprPacked(1)),
                        sym::simd => Some(ReprSimd),
                        sym::transparent => Some(ReprTransparent),
                        name => int_type_of_word(name).map(ReprInt),
                    };

                    if let Some(h) = hint {
                        recognised = true;
                        acc.push(h);
                    }
                } else if let Some((name, value)) = item.name_value_literal() {
                    let parse_alignment = |node: &ast::LitKind| -> Result<u32, &'static str> {
                        if let ast::LitKind::Int(literal, ast::LitIntType::Unsuffixed) = node {
                            if literal.is_power_of_two() {
                                // rustc::ty::layout::Align restricts align to <= 2^29
                                if *literal <= 1 << 29 {
                                    Ok(*literal as u32)
                                } else {
                                    Err("larger than 2^29")
                                }
                            } else {
                                Err("not a power of two")
                            }
                        } else {
                            Err("not an unsuffixed integer")
                        }
                    };

                    let mut literal_error = None;
                    if name == sym::align {
                        recognised = true;
                        match parse_alignment(&value.kind) {
                            Ok(literal) => acc.push(ReprAlign(literal)),
                            Err(message) => literal_error = Some(message)
                        };
                    }
                    else if name == sym::packed {
                        recognised = true;
                        match parse_alignment(&value.kind) {
                            Ok(literal) => acc.push(ReprPacked(literal)),
                            Err(message) => literal_error = Some(message)
                        };
                    }
                    if let Some(literal_error) = literal_error {
                        span_err!(diagnostic, item.span(), E0589,
                                  "invalid `repr(align)` attribute: {}", literal_error);
                    }
                } else {
                    if let Some(meta_item) = item.meta_item() {
                        if meta_item.check_name(sym::align) {
                            if let MetaItemKind::NameValue(ref value) = meta_item.kind {
                                recognised = true;
                                let mut err = struct_span_err!(diagnostic, item.span(), E0693,
                                    "incorrect `repr(align)` attribute format");
                                match value.kind {
                                    ast::LitKind::Int(int, ast::LitIntType::Unsuffixed) => {
                                        err.span_suggestion(
                                            item.span(),
                                            "use parentheses instead",
                                            format!("align({})", int),
                                            Applicability::MachineApplicable
                                        );
                                    }
                                    ast::LitKind::Str(s, _) => {
                                        err.span_suggestion(
                                            item.span(),
                                            "use parentheses instead",
                                            format!("align({})", s),
                                            Applicability::MachineApplicable
                                        );
                                    }
                                    _ => {}
                                }
                                err.emit();
                            }
                        }
                    }
                }
                if !recognised {
                    // Not a word we recognize
                    span_err!(diagnostic, item.span(), E0552,
                              "unrecognized representation hint");
                }
            }
        }
    }
    acc
}

fn int_type_of_word(s: Symbol) -> Option<IntType> {
    use IntType::*;

    match s {
        sym::i8 => Some(SignedInt(ast::IntTy::I8)),
        sym::u8 => Some(UnsignedInt(ast::UintTy::U8)),
        sym::i16 => Some(SignedInt(ast::IntTy::I16)),
        sym::u16 => Some(UnsignedInt(ast::UintTy::U16)),
        sym::i32 => Some(SignedInt(ast::IntTy::I32)),
        sym::u32 => Some(UnsignedInt(ast::UintTy::U32)),
        sym::i64 => Some(SignedInt(ast::IntTy::I64)),
        sym::u64 => Some(UnsignedInt(ast::UintTy::U64)),
        sym::i128 => Some(SignedInt(ast::IntTy::I128)),
        sym::u128 => Some(UnsignedInt(ast::UintTy::U128)),
        sym::isize => Some(SignedInt(ast::IntTy::Isize)),
        sym::usize => Some(UnsignedInt(ast::UintTy::Usize)),
        _ => None
    }
}

pub enum TransparencyError {
    UnknownTransparency(Symbol, Span),
    MultipleTransparencyAttrs(Span, Span),
}

pub fn find_transparency(
    attrs: &[Attribute], is_legacy: bool
) -> (Transparency, Option<TransparencyError>) {
    let mut transparency = None;
    let mut error = None;
    for attr in attrs {
        if attr.check_name(sym::rustc_macro_transparency) {
            if let Some((_, old_span)) = transparency {
                error = Some(TransparencyError::MultipleTransparencyAttrs(old_span, attr.span));
                break;
            } else if let Some(value) = attr.value_str() {
                transparency = Some((match &*value.as_str() {
                    "transparent" => Transparency::Transparent,
                    "semitransparent" => Transparency::SemiTransparent,
                    "opaque" => Transparency::Opaque,
                    _ => {
                        error = Some(TransparencyError::UnknownTransparency(value, attr.span));
                        continue;
                    }
                }, attr.span));
            }
        }
    }
    let fallback = if is_legacy { Transparency::SemiTransparent } else { Transparency::Opaque };
    (transparency.map_or(fallback, |t| t.0), error)
}
