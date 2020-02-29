//! Parsing and validation of builtin attributes

use super::{find_by_name, mark_used};

use rustc_ast_pretty::pprust;
use rustc_errors::{struct_span_err, Applicability, Handler};
use rustc_feature::{find_gated_cfg, is_builtin_attr_name, Features, GatedCfg};
use rustc_macros::HashStable_Generic;
use rustc_session::parse::{feature_err, ParseSess};
use rustc_span::hygiene::Transparency;
use rustc_span::{symbol::sym, symbol::Symbol, Span};
use std::num::NonZeroU32;
use syntax::ast::{self, Attribute, MetaItem, MetaItemKind, NestedMetaItem};

pub fn is_builtin_attr(attr: &Attribute) -> bool {
    attr.is_doc_comment() || attr.ident().filter(|ident| is_builtin_attr_name(ident.name)).is_some()
}

enum AttrError {
    MultipleItem(String),
    UnknownMetaItem(String, &'static [&'static str]),
    MissingSince,
    MissingFeature,
    MultipleStabilityLevels,
    UnsupportedLiteral(&'static str, /* is_bytestr */ bool),
}

fn handle_errors(sess: &ParseSess, span: Span, error: AttrError) {
    let diag = &sess.span_diagnostic;
    match error {
        AttrError::MultipleItem(item) => {
            struct_span_err!(diag, span, E0538, "multiple '{}' items", item).emit();
        }
        AttrError::UnknownMetaItem(item, expected) => {
            let expected = expected.iter().map(|name| format!("`{}`", name)).collect::<Vec<_>>();
            struct_span_err!(diag, span, E0541, "unknown meta item '{}'", item)
                .span_label(span, format!("expected one of {}", expected.join(", ")))
                .emit();
        }
        AttrError::MissingSince => {
            struct_span_err!(diag, span, E0542, "missing 'since'").emit();
        }
        AttrError::MissingFeature => {
            struct_span_err!(diag, span, E0546, "missing 'feature'").emit();
        }
        AttrError::MultipleStabilityLevels => {
            struct_span_err!(diag, span, E0544, "multiple stability levels").emit();
        }
        AttrError::UnsupportedLiteral(msg, is_bytestr) => {
            let mut err = struct_span_err!(diag, span, E0565, "{}", msg);
            if is_bytestr {
                if let Ok(lint_str) = sess.source_map().span_to_snippet(span) {
                    err.span_suggestion(
                        span,
                        "consider removing the prefix",
                        lint_str[1..].to_string(),
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
                                (vec!["allowed", "aborts"])
                                    .into_iter()
                                    .map(|s| format!("#[unwind({})]", s)),
                                Applicability::MachineApplicable,
                            )
                            .emit();
                    });
                }
            }
        }

        ia
    })
}

/// Represents the #[stable], #[unstable], #[rustc_deprecated] attributes.
#[derive(
    RustcEncodable,
    RustcDecodable,
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    HashStable_Generic
)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    pub rustc_depr: Option<RustcDeprecation>,
}

/// Represents the #[rustc_const_unstable] and #[rustc_const_stable] attributes.
#[derive(
    RustcEncodable,
    RustcDecodable,
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    HashStable_Generic
)]
pub struct ConstStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    /// whether the function has a `#[rustc_promotable]` attribute
    pub promotable: bool,
    /// whether the function has a `#[rustc_allow_const_fn_ptr]` attribute
    pub allow_const_fn_ptr: bool,
}

/// The available stability levels.
#[derive(
    RustcEncodable,
    RustcDecodable,
    PartialEq,
    PartialOrd,
    Copy,
    Clone,
    Debug,
    Eq,
    Hash,
    HashStable_Generic
)]
pub enum StabilityLevel {
    // Reason for the current stability level and the relevant rust-lang issue
    Unstable { reason: Option<Symbol>, issue: Option<NonZeroU32>, is_soft: bool },
    Stable { since: Symbol },
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool {
        if let StabilityLevel::Unstable { .. } = *self { true } else { false }
    }
    pub fn is_stable(&self) -> bool {
        if let StabilityLevel::Stable { .. } = *self { true } else { false }
    }
}

#[derive(
    RustcEncodable,
    RustcDecodable,
    PartialEq,
    PartialOrd,
    Copy,
    Clone,
    Debug,
    Eq,
    Hash,
    HashStable_Generic
)]
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
        item.check_name(sym::feature)
            && item
                .meta_item_list()
                .map(|list| list.iter().any(|mi| mi.is_word() && mi.check_name(feature_name)))
                .unwrap_or(false)
    })
}

/// Collects stability info from all stability attributes in `attrs`.
/// Returns `None` if no stability attributes are found.
pub fn find_stability(
    sess: &ParseSess,
    attrs: &[Attribute],
    item_sp: Span,
) -> (Option<Stability>, Option<ConstStability>) {
    find_stability_generic(sess, attrs.iter(), item_sp)
}

fn find_stability_generic<'a, I>(
    sess: &ParseSess,
    attrs_iter: I,
    item_sp: Span,
) -> (Option<Stability>, Option<ConstStability>)
where
    I: Iterator<Item = &'a Attribute>,
{
    use StabilityLevel::*;

    let mut stab: Option<Stability> = None;
    let mut rustc_depr: Option<RustcDeprecation> = None;
    let mut const_stab: Option<ConstStability> = None;
    let mut promotable = false;
    let mut allow_const_fn_ptr = false;
    let diagnostic = &sess.span_diagnostic;

    'outer: for attr in attrs_iter {
        if ![
            sym::rustc_deprecated,
            sym::rustc_const_unstable,
            sym::rustc_const_stable,
            sym::unstable,
            sym::stable,
            sym::rustc_promotable,
            sym::rustc_allow_const_fn_ptr,
        ]
        .iter()
        .any(|&s| attr.has_name(s))
        {
            continue; // not a stability level
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
                    return false;
                }
                if let Some(v) = meta.value_str() {
                    *item = Some(v);
                    true
                } else {
                    struct_span_err!(diagnostic, meta.span, E0539, "incorrect meta item").emit();
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

            let meta_name = meta.name_or_empty();
            match meta_name {
                sym::rustc_deprecated => {
                    if rustc_depr.is_some() {
                        struct_span_err!(
                            diagnostic,
                            item_sp,
                            E0540,
                            "multiple rustc_deprecated attributes"
                        )
                        .emit();
                        continue 'outer;
                    }

                    get_meta!(since, reason, suggestion);

                    match (since, reason) {
                        (Some(since), Some(reason)) => {
                            rustc_depr = Some(RustcDeprecation { since, reason, suggestion })
                        }
                        (None, _) => {
                            handle_errors(sess, attr.span, AttrError::MissingSince);
                            continue;
                        }
                        _ => {
                            struct_span_err!(diagnostic, attr.span, E0543, "missing 'reason'")
                                .emit();
                            continue;
                        }
                    }
                }
                sym::rustc_const_unstable | sym::unstable => {
                    if meta_name == sym::unstable && stab.is_some() {
                        handle_errors(sess, attr.span, AttrError::MultipleStabilityLevels);
                        break;
                    } else if meta_name == sym::rustc_const_unstable && const_stab.is_some() {
                        handle_errors(sess, attr.span, AttrError::MultipleStabilityLevels);
                        break;
                    }

                    let mut feature = None;
                    let mut reason = None;
                    let mut issue = None;
                    let mut issue_num = None;
                    let mut is_soft = false;
                    for meta in metas {
                        if let Some(mi) = meta.meta_item() {
                            match mi.name_or_empty() {
                                sym::feature => {
                                    if !get(mi, &mut feature) {
                                        continue 'outer;
                                    }
                                }
                                sym::reason => {
                                    if !get(mi, &mut reason) {
                                        continue 'outer;
                                    }
                                }
                                sym::issue => {
                                    if !get(mi, &mut issue) {
                                        continue 'outer;
                                    }

                                    // These unwraps are safe because `get` ensures the meta item
                                    // is a name/value pair string literal.
                                    issue_num = match &*issue.unwrap().as_str() {
                                        "none" => None,
                                        issue => {
                                            let emit_diag = |msg: &str| {
                                                struct_span_err!(
                                                    diagnostic,
                                                    mi.span,
                                                    E0545,
                                                    "`issue` must be a non-zero numeric string \
                                                    or \"none\"",
                                                )
                                                .span_label(
                                                    mi.name_value_literal().unwrap().span,
                                                    msg,
                                                )
                                                .emit();
                                            };
                                            match issue.parse() {
                                                Ok(num) if num == 0 => {
                                                    emit_diag(
                                                        "`issue` must not be \"0\", \
                                                        use \"none\" instead",
                                                    );
                                                    continue 'outer;
                                                }
                                                Ok(num) => NonZeroU32::new(num),
                                                Err(err) => {
                                                    emit_diag(&err.to_string());
                                                    continue 'outer;
                                                }
                                            }
                                        }
                                    };
                                }
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
                                            &["feature", "reason", "issue", "soft"],
                                        ),
                                    );
                                    continue 'outer;
                                }
                            }
                        } else {
                            handle_errors(
                                sess,
                                meta.span(),
                                AttrError::UnsupportedLiteral("unsupported literal", false),
                            );
                            continue 'outer;
                        }
                    }

                    match (feature, reason, issue) {
                        (Some(feature), reason, Some(_)) => {
                            let level = Unstable { reason, issue: issue_num, is_soft };
                            if sym::unstable == meta_name {
                                stab = Some(Stability { level, feature, rustc_depr: None });
                            } else {
                                const_stab = Some(ConstStability {
                                    level,
                                    feature,
                                    promotable: false,
                                    allow_const_fn_ptr: false,
                                });
                            }
                        }
                        (None, _, _) => {
                            handle_errors(sess, attr.span, AttrError::MissingFeature);
                            continue;
                        }
                        _ => {
                            struct_span_err!(diagnostic, attr.span, E0547, "missing 'issue'")
                                .emit();
                            continue;
                        }
                    }
                }
                sym::rustc_const_stable | sym::stable => {
                    if meta_name == sym::stable && stab.is_some() {
                        handle_errors(sess, attr.span, AttrError::MultipleStabilityLevels);
                        break;
                    } else if meta_name == sym::rustc_const_stable && const_stab.is_some() {
                        handle_errors(sess, attr.span, AttrError::MultipleStabilityLevels);
                        break;
                    }

                    let mut feature = None;
                    let mut since = None;
                    for meta in metas {
                        match meta {
                            NestedMetaItem::MetaItem(mi) => match mi.name_or_empty() {
                                sym::feature => {
                                    if !get(mi, &mut feature) {
                                        continue 'outer;
                                    }
                                }
                                sym::since => {
                                    if !get(mi, &mut since) {
                                        continue 'outer;
                                    }
                                }
                                _ => {
                                    handle_errors(
                                        sess,
                                        meta.span(),
                                        AttrError::UnknownMetaItem(
                                            pprust::path_to_string(&mi.path),
                                            &["since", "note"],
                                        ),
                                    );
                                    continue 'outer;
                                }
                            },
                            NestedMetaItem::Literal(lit) => {
                                handle_errors(
                                    sess,
                                    lit.span,
                                    AttrError::UnsupportedLiteral("unsupported literal", false),
                                );
                                continue 'outer;
                            }
                        }
                    }

                    match (feature, since) {
                        (Some(feature), Some(since)) => {
                            let level = Stable { since };
                            if sym::stable == meta_name {
                                stab = Some(Stability { level, feature, rustc_depr: None });
                            } else {
                                const_stab = Some(ConstStability {
                                    level,
                                    feature,
                                    promotable: false,
                                    allow_const_fn_ptr: false,
                                });
                            }
                        }
                        (None, _) => {
                            handle_errors(sess, attr.span, AttrError::MissingFeature);
                            continue;
                        }
                        _ => {
                            handle_errors(sess, attr.span, AttrError::MissingSince);
                            continue;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    // Merge the deprecation info into the stability info
    if let Some(rustc_depr) = rustc_depr {
        if let Some(ref mut stab) = stab {
            stab.rustc_depr = Some(rustc_depr);
        } else {
            struct_span_err!(
                diagnostic,
                item_sp,
                E0549,
                "rustc_deprecated attribute must be paired with \
                       either stable or unstable attribute"
            )
            .emit();
        }
    }

    // Merge the const-unstable info into the stability info
    if promotable || allow_const_fn_ptr {
        if let Some(ref mut stab) = const_stab {
            stab.promotable = promotable;
            stab.allow_const_fn_ptr = allow_const_fn_ptr;
        } else {
            struct_span_err!(
                diagnostic,
                item_sp,
                E0717,
                "rustc_promotable and rustc_allow_const_fn_ptr attributes \
                      must be paired with either a rustc_const_unstable or a rustc_const_stable \
                      attribute"
            )
            .emit();
        }
    }

    (stab, const_stab)
}

pub fn find_crate_name(attrs: &[Attribute]) -> Option<Symbol> {
    super::first_attr_value_str_by_name(attrs, sym::crate_name)
}

/// Tests if a cfg-pattern matches the cfg set
pub fn cfg_matches(cfg: &ast::MetaItem, sess: &ParseSess, features: Option<&Features>) -> bool {
    eval_condition(cfg, sess, &mut |cfg| {
        let gate = find_gated_cfg(|sym| cfg.check_name(sym));
        if let (Some(feats), Some(gated_cfg)) = (features, gate) {
            gate_cfg(&gated_cfg, cfg.span, sess, feats);
        }
        let error = |span, msg| {
            sess.span_diagnostic.span_err(span, msg);
            true
        };
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
                        lit.kind.is_bytestr(),
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

fn gate_cfg(gated_cfg: &GatedCfg, cfg_span: Span, sess: &ParseSess, features: &Features) {
    let (cfg, feature, has_feature) = gated_cfg;
    if !has_feature(features) && !cfg_span.allows_unstable(*feature) {
        let explain = format!("`cfg({})` is experimental and subject to change", cfg);
        feature_err(sess, *feature, cfg_span, &explain).emit();
    }
}

/// Evaluate a cfg-like condition (with `any` and `all`), using `eval` to
/// evaluate individual items.
pub fn eval_condition(
    cfg: &ast::MetaItem,
    sess: &ParseSess,
    eval: &mut impl FnMut(&ast::MetaItem) -> bool,
) -> bool {
    match cfg.kind {
        ast::MetaItemKind::List(ref mis) => {
            for mi in mis.iter() {
                if !mi.is_meta_item() {
                    handle_errors(
                        sess,
                        mi.span(),
                        AttrError::UnsupportedLiteral("unsupported literal", false),
                    );
                    return false;
                }
            }

            // The unwraps below may look dangerous, but we've already asserted
            // that they won't fail with the loop above.
            match cfg.name_or_empty() {
                sym::any => {
                    mis.iter().any(|mi| eval_condition(mi.meta_item().unwrap(), sess, eval))
                }
                sym::all => {
                    mis.iter().all(|mi| eval_condition(mi.meta_item().unwrap(), sess, eval))
                }
                sym::not => {
                    if mis.len() != 1 {
                        struct_span_err!(
                            sess.span_diagnostic,
                            cfg.span,
                            E0536,
                            "expected 1 cfg-pattern"
                        )
                        .emit();
                        return false;
                    }

                    !eval_condition(mis[0].meta_item().unwrap(), sess, eval)
                }
                _ => {
                    struct_span_err!(
                        sess.span_diagnostic,
                        cfg.span,
                        E0537,
                        "invalid predicate `{}`",
                        pprust::path_to_string(&cfg.path)
                    )
                    .emit();
                    false
                }
            }
        }
        ast::MetaItemKind::Word | ast::MetaItemKind::NameValue(..) => eval(cfg),
    }
}

#[derive(RustcEncodable, RustcDecodable, Clone, HashStable_Generic)]
pub struct Deprecation {
    pub since: Option<Symbol>,
    pub note: Option<Symbol>,
}

/// Finds the deprecation attribute. `None` if none exists.
pub fn find_deprecation(
    sess: &ParseSess,
    attrs: &[Attribute],
    item_sp: Span,
) -> Option<Deprecation> {
    find_deprecation_generic(sess, attrs.iter(), item_sp)
}

fn find_deprecation_generic<'a, I>(
    sess: &ParseSess,
    attrs_iter: I,
    item_sp: Span,
) -> Option<Deprecation>
where
    I: Iterator<Item = &'a Attribute>,
{
    let mut depr: Option<Deprecation> = None;
    let diagnostic = &sess.span_diagnostic;

    'outer: for attr in attrs_iter {
        if !attr.check_name(sym::deprecated) {
            continue;
        }

        if depr.is_some() {
            struct_span_err!(diagnostic, item_sp, E0550, "multiple deprecated attributes").emit();
            break;
        }

        let meta = match attr.meta() {
            Some(meta) => meta,
            None => continue,
        };
        depr = match &meta.kind {
            MetaItemKind::Word => Some(Deprecation { since: None, note: None }),
            MetaItemKind::NameValue(..) => {
                meta.value_str().map(|note| Deprecation { since: None, note: Some(note) })
            }
            MetaItemKind::List(list) => {
                let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                    if item.is_some() {
                        handle_errors(
                            sess,
                            meta.span,
                            AttrError::MultipleItem(pprust::path_to_string(&meta.path)),
                        );
                        return false;
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
                                    lit.kind.is_bytestr(),
                                ),
                            );
                        } else {
                            struct_span_err!(diagnostic, meta.span, E0551, "incorrect meta item")
                                .emit();
                        }

                        false
                    }
                };

                let mut since = None;
                let mut note = None;
                for meta in list {
                    match meta {
                        NestedMetaItem::MetaItem(mi) => match mi.name_or_empty() {
                            sym::since => {
                                if !get(mi, &mut since) {
                                    continue 'outer;
                                }
                            }
                            sym::note => {
                                if !get(mi, &mut note) {
                                    continue 'outer;
                                }
                            }
                            _ => {
                                handle_errors(
                                    sess,
                                    meta.span(),
                                    AttrError::UnknownMetaItem(
                                        pprust::path_to_string(&mi.path),
                                        &["since", "note"],
                                    ),
                                );
                                continue 'outer;
                            }
                        },
                        NestedMetaItem::Literal(lit) => {
                            handle_errors(
                                sess,
                                lit.span,
                                AttrError::UnsupportedLiteral(
                                    "item in `deprecated` must be a key/value pair",
                                    false,
                                ),
                            );
                            continue 'outer;
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
    ReprNoNiche,
}

#[derive(Eq, PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, Clone, HashStable_Generic)]
pub enum IntType {
    SignedInt(ast::IntTy),
    UnsignedInt(ast::UintTy),
}

impl IntType {
    #[inline]
    pub fn is_signed(self) -> bool {
        use IntType::*;

        match self {
            SignedInt(..) => true,
            UnsignedInt(..) => false,
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
                    continue;
                }

                let mut recognised = false;
                if item.is_word() {
                    let hint = match item.name_or_empty() {
                        sym::C => Some(ReprC),
                        sym::packed => Some(ReprPacked(1)),
                        sym::simd => Some(ReprSimd),
                        sym::transparent => Some(ReprTransparent),
                        sym::no_niche => Some(ReprNoNiche),
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
                            Err(message) => literal_error = Some(message),
                        };
                    } else if name == sym::packed {
                        recognised = true;
                        match parse_alignment(&value.kind) {
                            Ok(literal) => acc.push(ReprPacked(literal)),
                            Err(message) => literal_error = Some(message),
                        };
                    }
                    if let Some(literal_error) = literal_error {
                        struct_span_err!(
                            diagnostic,
                            item.span(),
                            E0589,
                            "invalid `repr(align)` attribute: {}",
                            literal_error
                        )
                        .emit();
                    }
                } else {
                    if let Some(meta_item) = item.meta_item() {
                        if meta_item.check_name(sym::align) {
                            if let MetaItemKind::NameValue(ref value) = meta_item.kind {
                                recognised = true;
                                let mut err = struct_span_err!(
                                    diagnostic,
                                    item.span(),
                                    E0693,
                                    "incorrect `repr(align)` attribute format"
                                );
                                match value.kind {
                                    ast::LitKind::Int(int, ast::LitIntType::Unsuffixed) => {
                                        err.span_suggestion(
                                            item.span(),
                                            "use parentheses instead",
                                            format!("align({})", int),
                                            Applicability::MachineApplicable,
                                        );
                                    }
                                    ast::LitKind::Str(s, _) => {
                                        err.span_suggestion(
                                            item.span(),
                                            "use parentheses instead",
                                            format!("align({})", s),
                                            Applicability::MachineApplicable,
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
                    struct_span_err!(
                        diagnostic,
                        item.span(),
                        E0552,
                        "unrecognized representation hint"
                    )
                    .emit();
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
        _ => None,
    }
}

pub enum TransparencyError {
    UnknownTransparency(Symbol, Span),
    MultipleTransparencyAttrs(Span, Span),
}

pub fn find_transparency(
    attrs: &[Attribute],
    is_legacy: bool,
) -> (Transparency, Option<TransparencyError>) {
    let mut transparency = None;
    let mut error = None;
    for attr in attrs {
        if attr.check_name(sym::rustc_macro_transparency) {
            if let Some((_, old_span)) = transparency {
                error = Some(TransparencyError::MultipleTransparencyAttrs(old_span, attr.span));
                break;
            } else if let Some(value) = attr.value_str() {
                transparency = Some((
                    match &*value.as_str() {
                        "transparent" => Transparency::Transparent,
                        "semitransparent" => Transparency::SemiTransparent,
                        "opaque" => Transparency::Opaque,
                        _ => {
                            error = Some(TransparencyError::UnknownTransparency(value, attr.span));
                            continue;
                        }
                    },
                    attr.span,
                ));
            }
        }
    }
    let fallback = if is_legacy { Transparency::SemiTransparent } else { Transparency::Opaque };
    (transparency.map_or(fallback, |t| t.0), error)
}

pub fn allow_internal_unstable<'a>(
    attrs: &[Attribute],
    diag: &'a rustc_errors::Handler,
) -> Option<impl Iterator<Item = Symbol> + 'a> {
    let attr = find_by_name(attrs, sym::allow_internal_unstable)?;
    let list = attr.meta_item_list().or_else(|| {
        diag.span_err(attr.span, "allow_internal_unstable expects list of feature names");
        None
    })?;
    Some(list.into_iter().filter_map(move |it| {
        let name = it.ident().map(|ident| ident.name);
        if name.is_none() {
            diag.span_err(it.span(), "`allow_internal_unstable` expects feature names");
        }
        name
    }))
}
