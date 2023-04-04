//! Parsing and validation of builtin attributes

use rustc_ast::{self as ast, attr};
use rustc_ast::{Attribute, LitKind, MetaItem, MetaItemKind, MetaItemLit, NestedMetaItem, NodeId};
use rustc_ast_pretty::pprust;
use rustc_feature::{find_gated_cfg, is_builtin_attr_name, Features, GatedCfg};
use rustc_macros::HashStable_Generic;
use rustc_session::lint::builtin::UNEXPECTED_CFGS;
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_session::parse::{feature_err, ParseSess};
use rustc_session::Session;
use rustc_span::hygiene::Transparency;
use rustc_span::{symbol::sym, symbol::Symbol, Span};
use std::num::NonZeroU32;

use crate::session_diagnostics::{self, IncorrectReprFormatGenericCause};

/// The version placeholder that recently stabilized features contain inside the
/// `since` field of the `#[stable]` attribute.
///
/// For more, see [this pull request](https://github.com/rust-lang/rust/pull/100591).
pub const VERSION_PLACEHOLDER: &str = "CURRENT_RUSTC_VERSION";

pub fn rust_version_symbol() -> Symbol {
    let version = option_env!("CFG_VERSION").unwrap_or("<current>");
    let version = version.split(' ').next().unwrap();
    Symbol::intern(&version)
}

pub fn is_builtin_attr(attr: &Attribute) -> bool {
    attr.is_doc_comment() || attr.ident().filter(|ident| is_builtin_attr_name(ident.name)).is_some()
}

enum AttrError {
    MultipleItem(String),
    UnknownMetaItem(String, &'static [&'static str]),
    MissingSince,
    NonIdentFeature,
    MissingFeature,
    MultipleStabilityLevels,
    UnsupportedLiteral(UnsupportedLiteralReason, /* is_bytestr */ bool),
}

pub(crate) enum UnsupportedLiteralReason {
    Generic,
    CfgString,
    DeprecatedString,
    DeprecatedKvPair,
}

fn handle_errors(sess: &ParseSess, span: Span, error: AttrError) {
    match error {
        AttrError::MultipleItem(item) => {
            sess.emit_err(session_diagnostics::MultipleItem { span, item });
        }
        AttrError::UnknownMetaItem(item, expected) => {
            sess.emit_err(session_diagnostics::UnknownMetaItem { span, item, expected });
        }
        AttrError::MissingSince => {
            sess.emit_err(session_diagnostics::MissingSince { span });
        }
        AttrError::NonIdentFeature => {
            sess.emit_err(session_diagnostics::NonIdentFeature { span });
        }
        AttrError::MissingFeature => {
            sess.emit_err(session_diagnostics::MissingFeature { span });
        }
        AttrError::MultipleStabilityLevels => {
            sess.emit_err(session_diagnostics::MultipleStabilityLevels { span });
        }
        AttrError::UnsupportedLiteral(reason, is_bytestr) => {
            sess.emit_err(session_diagnostics::UnsupportedLiteral {
                span,
                reason,
                is_bytestr,
                start_point_span: sess.source_map().start_point(span),
            });
        }
    }
}

#[derive(Copy, Clone, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum InlineAttr {
    None,
    Hint,
    Always,
    Never,
}

#[derive(Clone, Encodable, Decodable, Debug, PartialEq, Eq, HashStable_Generic)]
pub enum InstructionSetAttr {
    ArmA32,
    ArmT32,
}

#[derive(Clone, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum OptimizeAttr {
    None,
    Speed,
    Size,
}

/// Represents the following attributes:
///
/// - `#[stable]`
/// - `#[unstable]`
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct Stability {
    pub level: StabilityLevel,
    pub feature: Symbol,
}

impl Stability {
    pub fn is_unstable(&self) -> bool {
        self.level.is_unstable()
    }

    pub fn is_stable(&self) -> bool {
        self.level.is_stable()
    }
}

/// Represents the `#[rustc_const_unstable]` and `#[rustc_const_stable]` attributes.
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct ConstStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    /// whether the function has a `#[rustc_promotable]` attribute
    pub promotable: bool,
}

impl ConstStability {
    pub fn is_const_unstable(&self) -> bool {
        self.level.is_unstable()
    }

    pub fn is_const_stable(&self) -> bool {
        self.level.is_stable()
    }
}

/// Represents the `#[rustc_default_body_unstable]` attribute.
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct DefaultBodyStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
}

/// The available stability levels.
#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, Hash)]
#[derive(HashStable_Generic)]
pub enum StabilityLevel {
    /// `#[unstable]`
    Unstable {
        /// Reason for the current stability level.
        reason: UnstableReason,
        /// Relevant `rust-lang/rust` issue.
        issue: Option<NonZeroU32>,
        is_soft: bool,
        /// If part of a feature is stabilized and a new feature is added for the remaining parts,
        /// then the `implied_by` attribute is used to indicate which now-stable feature previously
        /// contained a item.
        ///
        /// ```pseudo-Rust
        /// #[unstable(feature = "foo", issue = "...")]
        /// fn foo() {}
        /// #[unstable(feature = "foo", issue = "...")]
        /// fn foobar() {}
        /// ```
        ///
        /// ...becomes...
        ///
        /// ```pseudo-Rust
        /// #[stable(feature = "foo", since = "1.XX.X")]
        /// fn foo() {}
        /// #[unstable(feature = "foobar", issue = "...", implied_by = "foo")]
        /// fn foobar() {}
        /// ```
        implied_by: Option<Symbol>,
    },
    /// `#[stable]`
    Stable {
        /// Rust release which stabilized this feature.
        since: Symbol,
        /// Is this item allowed to be referred to on stable, despite being contained in unstable
        /// modules?
        allowed_through_unstable_modules: bool,
    },
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool {
        matches!(self, StabilityLevel::Unstable { .. })
    }
    pub fn is_stable(&self) -> bool {
        matches!(self, StabilityLevel::Stable { .. })
    }
}

#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, Hash)]
#[derive(HashStable_Generic)]
pub enum UnstableReason {
    None,
    Default,
    Some(Symbol),
}

impl UnstableReason {
    fn from_opt_reason(reason: Option<Symbol>) -> Self {
        // UnstableReason::Default constructed manually
        match reason {
            Some(r) => Self::Some(r),
            None => Self::None,
        }
    }

    pub fn to_opt_reason(&self) -> Option<Symbol> {
        match self {
            Self::None => None,
            Self::Default => Some(sym::unstable_location_reason_default),
            Self::Some(r) => Some(*r),
        }
    }
}

/// Collects stability info from `stable`/`unstable`/`rustc_allowed_through_unstable_modules`
/// attributes in `attrs`. Returns `None` if no stability attributes are found.
pub fn find_stability(
    sess: &Session,
    attrs: &[Attribute],
    item_sp: Span,
) -> Option<(Stability, Span)> {
    let mut stab: Option<(Stability, Span)> = None;
    let mut allowed_through_unstable_modules = false;

    for attr in attrs {
        match attr.name_or_empty() {
            sym::rustc_allowed_through_unstable_modules => allowed_through_unstable_modules = true,
            sym::unstable => {
                if stab.is_some() {
                    handle_errors(&sess.parse_sess, attr.span, AttrError::MultipleStabilityLevels);
                    break;
                }

                if let Some((feature, level)) = parse_unstability(sess, attr) {
                    stab = Some((Stability { level, feature }, attr.span));
                }
            }
            sym::stable => {
                if stab.is_some() {
                    handle_errors(&sess.parse_sess, attr.span, AttrError::MultipleStabilityLevels);
                    break;
                }
                if let Some((feature, level)) = parse_stability(sess, attr) {
                    stab = Some((Stability { level, feature }, attr.span));
                }
            }
            _ => {}
        }
    }

    if allowed_through_unstable_modules {
        match &mut stab {
            Some((
                Stability {
                    level: StabilityLevel::Stable { allowed_through_unstable_modules, .. },
                    ..
                },
                _,
            )) => *allowed_through_unstable_modules = true,
            _ => {
                sess.emit_err(session_diagnostics::RustcAllowedUnstablePairing { span: item_sp });
            }
        }
    }

    stab
}

/// Collects stability info from `rustc_const_stable`/`rustc_const_unstable`/`rustc_promotable`
/// attributes in `attrs`. Returns `None` if no stability attributes are found.
pub fn find_const_stability(
    sess: &Session,
    attrs: &[Attribute],
    item_sp: Span,
) -> Option<(ConstStability, Span)> {
    let mut const_stab: Option<(ConstStability, Span)> = None;
    let mut promotable = false;

    for attr in attrs {
        match attr.name_or_empty() {
            sym::rustc_promotable => promotable = true,
            sym::rustc_const_unstable => {
                if const_stab.is_some() {
                    handle_errors(&sess.parse_sess, attr.span, AttrError::MultipleStabilityLevels);
                    break;
                }

                if let Some((feature, level)) = parse_unstability(sess, attr) {
                    const_stab =
                        Some((ConstStability { level, feature, promotable: false }, attr.span));
                }
            }
            sym::rustc_const_stable => {
                if const_stab.is_some() {
                    handle_errors(&sess.parse_sess, attr.span, AttrError::MultipleStabilityLevels);
                    break;
                }
                if let Some((feature, level)) = parse_stability(sess, attr) {
                    const_stab =
                        Some((ConstStability { level, feature, promotable: false }, attr.span));
                }
            }
            _ => {}
        }
    }

    // Merge the const-unstable info into the stability info
    if promotable {
        match &mut const_stab {
            Some((stab, _)) => stab.promotable = promotable,
            _ => _ = sess.emit_err(session_diagnostics::RustcPromotablePairing { span: item_sp }),
        }
    }

    const_stab
}

/// Collects stability info from `rustc_default_body_unstable` attributes in `attrs`.
/// Returns `None` if no stability attributes are found.
pub fn find_body_stability(
    sess: &Session,
    attrs: &[Attribute],
) -> Option<(DefaultBodyStability, Span)> {
    let mut body_stab: Option<(DefaultBodyStability, Span)> = None;

    for attr in attrs {
        if attr.has_name(sym::rustc_default_body_unstable) {
            if body_stab.is_some() {
                handle_errors(&sess.parse_sess, attr.span, AttrError::MultipleStabilityLevels);
                break;
            }

            if let Some((feature, level)) = parse_unstability(sess, attr) {
                body_stab = Some((DefaultBodyStability { level, feature }, attr.span));
            }
        }
    }

    body_stab
}

/// Read the content of a `stable`/`rustc_const_stable` attribute, and return the feature name and
/// its stability information.
fn parse_stability(sess: &Session, attr: &Attribute) -> Option<(Symbol, StabilityLevel)> {
    let meta = attr.meta()?;
    let MetaItem { kind: MetaItemKind::List(ref metas), .. } = meta else { return None };
    let insert_or_error = |meta: &MetaItem, item: &mut Option<Symbol>| {
        if item.is_some() {
            handle_errors(
                &sess.parse_sess,
                meta.span,
                AttrError::MultipleItem(pprust::path_to_string(&meta.path)),
            );
            return false;
        }
        if let Some(v) = meta.value_str() {
            *item = Some(v);
            true
        } else {
            sess.emit_err(session_diagnostics::IncorrectMetaItem { span: meta.span });
            false
        }
    };

    let mut feature = None;
    let mut since = None;
    for meta in metas {
        let Some(mi) = meta.meta_item() else {
            handle_errors(
                &sess.parse_sess,
                meta.span(),
                AttrError::UnsupportedLiteral(UnsupportedLiteralReason::Generic, false),
            );
            return None;
        };

        match mi.name_or_empty() {
            sym::feature => {
                if !insert_or_error(mi, &mut feature) {
                    return None;
                }
            }
            sym::since => {
                if !insert_or_error(mi, &mut since) {
                    return None;
                }
            }
            _ => {
                handle_errors(
                    &sess.parse_sess,
                    meta.span(),
                    AttrError::UnknownMetaItem(
                        pprust::path_to_string(&mi.path),
                        &["feature", "since"],
                    ),
                );
                return None;
            }
        }
    }

    if let Some(s) = since && s.as_str() == VERSION_PLACEHOLDER {
        since = Some(rust_version_symbol());
    }

    match (feature, since) {
        (Some(feature), Some(since)) => {
            let level = StabilityLevel::Stable { since, allowed_through_unstable_modules: false };
            Some((feature, level))
        }
        (None, _) => {
            handle_errors(&sess.parse_sess, attr.span, AttrError::MissingFeature);
            None
        }
        _ => {
            handle_errors(&sess.parse_sess, attr.span, AttrError::MissingSince);
            None
        }
    }
}

/// Read the content of a `unstable`/`rustc_const_unstable`/`rustc_default_body_unstable`
/// attribute, and return the feature name and its stability information.
fn parse_unstability(sess: &Session, attr: &Attribute) -> Option<(Symbol, StabilityLevel)> {
    let meta = attr.meta()?;
    let MetaItem { kind: MetaItemKind::List(ref metas), .. } = meta else { return None };
    let insert_or_error = |meta: &MetaItem, item: &mut Option<Symbol>| {
        if item.is_some() {
            handle_errors(
                &sess.parse_sess,
                meta.span,
                AttrError::MultipleItem(pprust::path_to_string(&meta.path)),
            );
            return false;
        }
        if let Some(v) = meta.value_str() {
            *item = Some(v);
            true
        } else {
            sess.emit_err(session_diagnostics::IncorrectMetaItem { span: meta.span });
            false
        }
    };

    let mut feature = None;
    let mut reason = None;
    let mut issue = None;
    let mut issue_num = None;
    let mut is_soft = false;
    let mut implied_by = None;
    for meta in metas {
        let Some(mi) = meta.meta_item() else {
            handle_errors(
                &sess.parse_sess,
                meta.span(),
                AttrError::UnsupportedLiteral(UnsupportedLiteralReason::Generic, false),
            );
            return None;
        };

        match mi.name_or_empty() {
            sym::feature => {
                if !insert_or_error(mi, &mut feature) {
                    return None;
                }
            }
            sym::reason => {
                if !insert_or_error(mi, &mut reason) {
                    return None;
                }
            }
            sym::issue => {
                if !insert_or_error(mi, &mut issue) {
                    return None;
                }

                // These unwraps are safe because `insert_or_error` ensures the meta item
                // is a name/value pair string literal.
                issue_num = match issue.unwrap().as_str() {
                    "none" => None,
                    issue => match issue.parse::<NonZeroU32>() {
                        Ok(num) => Some(num),
                        Err(err) => {
                            sess.emit_err(
                                session_diagnostics::InvalidIssueString {
                                    span: mi.span,
                                    cause: session_diagnostics::InvalidIssueStringCause::from_int_error_kind(
                                        mi.name_value_literal_span().unwrap(),
                                        err.kind(),
                                    ),
                                },
                            );
                            return None;
                        }
                    },
                };
            }
            sym::soft => {
                if !mi.is_word() {
                    sess.emit_err(session_diagnostics::SoftNoArgs { span: mi.span });
                }
                is_soft = true;
            }
            sym::implied_by => {
                if !insert_or_error(mi, &mut implied_by) {
                    return None;
                }
            }
            _ => {
                handle_errors(
                    &sess.parse_sess,
                    meta.span(),
                    AttrError::UnknownMetaItem(
                        pprust::path_to_string(&mi.path),
                        &["feature", "reason", "issue", "soft", "implied_by"],
                    ),
                );
                return None;
            }
        }
    }

    match (feature, reason, issue) {
        (Some(feature), reason, Some(_)) => {
            if !rustc_lexer::is_ident(feature.as_str()) {
                handle_errors(&sess.parse_sess, attr.span, AttrError::NonIdentFeature);
                return None;
            }
            let level = StabilityLevel::Unstable {
                reason: UnstableReason::from_opt_reason(reason),
                issue: issue_num,
                is_soft,
                implied_by,
            };
            Some((feature, level))
        }
        (None, _, _) => {
            handle_errors(&sess.parse_sess, attr.span, AttrError::MissingFeature);
            return None;
        }
        _ => {
            sess.emit_err(session_diagnostics::MissingIssue { span: attr.span });
            return None;
        }
    }
}

pub fn find_crate_name(attrs: &[Attribute]) -> Option<Symbol> {
    attr::first_attr_value_str_by_name(attrs, sym::crate_name)
}

#[derive(Clone, Debug)]
pub struct Condition {
    pub name: Symbol,
    pub name_span: Span,
    pub value: Option<Symbol>,
    pub value_span: Option<Span>,
    pub span: Span,
}

/// Tests if a cfg-pattern matches the cfg set
pub fn cfg_matches(
    cfg: &ast::MetaItem,
    sess: &ParseSess,
    lint_node_id: NodeId,
    features: Option<&Features>,
) -> bool {
    eval_condition(cfg, sess, features, &mut |cfg| {
        try_gate_cfg(cfg.name, cfg.span, sess, features);
        if let Some(names_valid) = &sess.check_config.names_valid {
            if !names_valid.contains(&cfg.name) {
                sess.buffer_lint_with_diagnostic(
                    UNEXPECTED_CFGS,
                    cfg.span,
                    lint_node_id,
                    "unexpected `cfg` condition name",
                    BuiltinLintDiagnostics::UnexpectedCfg((cfg.name, cfg.name_span), None),
                );
            }
        }
        if let Some(value) = cfg.value {
            if let Some(values) = &sess.check_config.values_valid.get(&cfg.name) {
                if !values.contains(&value) {
                    sess.buffer_lint_with_diagnostic(
                        UNEXPECTED_CFGS,
                        cfg.span,
                        lint_node_id,
                        "unexpected `cfg` condition value",
                        BuiltinLintDiagnostics::UnexpectedCfg(
                            (cfg.name, cfg.name_span),
                            cfg.value_span.map(|vs| (value, vs)),
                        ),
                    );
                }
            }
        }
        sess.config.contains(&(cfg.name, cfg.value))
    })
}

fn try_gate_cfg(name: Symbol, span: Span, sess: &ParseSess, features: Option<&Features>) {
    let gate = find_gated_cfg(|sym| sym == name);
    if let (Some(feats), Some(gated_cfg)) = (features, gate) {
        gate_cfg(&gated_cfg, span, sess, feats);
    }
}

fn gate_cfg(gated_cfg: &GatedCfg, cfg_span: Span, sess: &ParseSess, features: &Features) {
    let (cfg, feature, has_feature) = gated_cfg;
    if !has_feature(features) && !cfg_span.allows_unstable(*feature) {
        let explain = format!("`cfg({cfg})` is experimental and subject to change");
        feature_err(sess, *feature, cfg_span, &explain).emit();
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    major: u16,
    minor: u16,
    patch: u16,
}

fn parse_version(s: &str, allow_appendix: bool) -> Option<Version> {
    let mut components = s.split('-');
    let d = components.next()?;
    if !allow_appendix && components.next().is_some() {
        return None;
    }
    let mut digits = d.splitn(3, '.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    let patch = digits.next().unwrap_or("0").parse().ok()?;
    Some(Version { major, minor, patch })
}

/// Evaluate a cfg-like condition (with `any` and `all`), using `eval` to
/// evaluate individual items.
pub fn eval_condition(
    cfg: &ast::MetaItem,
    sess: &ParseSess,
    features: Option<&Features>,
    eval: &mut impl FnMut(Condition) -> bool,
) -> bool {
    match &cfg.kind {
        ast::MetaItemKind::List(mis) if cfg.name_or_empty() == sym::version => {
            try_gate_cfg(sym::version, cfg.span, sess, features);
            let (min_version, span) = match &mis[..] {
                [NestedMetaItem::Lit(MetaItemLit { kind: LitKind::Str(sym, ..), span, .. })] => {
                    (sym, span)
                }
                [
                    NestedMetaItem::Lit(MetaItemLit { span, .. })
                    | NestedMetaItem::MetaItem(MetaItem { span, .. }),
                ] => {
                    sess.emit_err(session_diagnostics::ExpectedVersionLiteral { span: *span });
                    return false;
                }
                [..] => {
                    sess.emit_err(session_diagnostics::ExpectedSingleVersionLiteral {
                        span: cfg.span,
                    });
                    return false;
                }
            };
            let Some(min_version) = parse_version(min_version.as_str(), false) else {
                sess.emit_warning(session_diagnostics::UnknownVersionLiteral { span: *span });
                return false;
            };
            let rustc_version = parse_version(env!("CFG_RELEASE"), true).unwrap();

            // See https://github.com/rust-lang/rust/issues/64796#issuecomment-640851454 for details
            if sess.assume_incomplete_release {
                rustc_version > min_version
            } else {
                rustc_version >= min_version
            }
        }
        ast::MetaItemKind::List(mis) => {
            for mi in mis.iter() {
                if !mi.is_meta_item() {
                    handle_errors(
                        sess,
                        mi.span(),
                        AttrError::UnsupportedLiteral(UnsupportedLiteralReason::Generic, false),
                    );
                    return false;
                }
            }

            // The unwraps below may look dangerous, but we've already asserted
            // that they won't fail with the loop above.
            match cfg.name_or_empty() {
                sym::any => mis
                    .iter()
                    // We don't use any() here, because we want to evaluate all cfg condition
                    // as eval_condition can (and does) extra checks
                    .fold(false, |res, mi| {
                        res | eval_condition(mi.meta_item().unwrap(), sess, features, eval)
                    }),
                sym::all => mis
                    .iter()
                    // We don't use all() here, because we want to evaluate all cfg condition
                    // as eval_condition can (and does) extra checks
                    .fold(true, |res, mi| {
                        res & eval_condition(mi.meta_item().unwrap(), sess, features, eval)
                    }),
                sym::not => {
                    if mis.len() != 1 {
                        sess.emit_err(session_diagnostics::ExpectedOneCfgPattern {
                            span: cfg.span,
                        });
                        return false;
                    }

                    !eval_condition(mis[0].meta_item().unwrap(), sess, features, eval)
                }
                sym::target => {
                    if let Some(features) = features && !features.cfg_target_compact {
                        feature_err(
                            sess,
                            sym::cfg_target_compact,
                            cfg.span,
                            "compact `cfg(target(..))` is experimental and subject to change"
                        ).emit();
                    }

                    mis.iter().fold(true, |res, mi| {
                        let mut mi = mi.meta_item().unwrap().clone();
                        if let [seg, ..] = &mut mi.path.segments[..] {
                            seg.ident.name = Symbol::intern(&format!("target_{}", seg.ident.name));
                        }

                        res & eval_condition(&mi, sess, features, eval)
                    })
                }
                _ => {
                    sess.emit_err(session_diagnostics::InvalidPredicate {
                        span: cfg.span,
                        predicate: pprust::path_to_string(&cfg.path),
                    });
                    false
                }
            }
        }
        ast::MetaItemKind::Word | MetaItemKind::NameValue(..) if cfg.path.segments.len() != 1 => {
            sess.emit_err(session_diagnostics::CfgPredicateIdentifier { span: cfg.path.span });
            true
        }
        MetaItemKind::NameValue(lit) if !lit.kind.is_str() => {
            handle_errors(
                sess,
                lit.span,
                AttrError::UnsupportedLiteral(
                    UnsupportedLiteralReason::CfgString,
                    lit.kind.is_bytestr(),
                ),
            );
            true
        }
        ast::MetaItemKind::Word | ast::MetaItemKind::NameValue(..) => {
            let ident = cfg.ident().expect("multi-segment cfg predicate");
            eval(Condition {
                name: ident.name,
                name_span: ident.span,
                value: cfg.value_str(),
                value_span: cfg.name_value_literal_span(),
                span: cfg.span,
            })
        }
    }
}

#[derive(Copy, Debug, Encodable, Decodable, Clone, HashStable_Generic)]
pub struct Deprecation {
    pub since: Option<Symbol>,
    /// The note to issue a reason.
    pub note: Option<Symbol>,
    /// A text snippet used to completely replace any use of the deprecated item in an expression.
    ///
    /// This is currently unstable.
    pub suggestion: Option<Symbol>,

    /// Whether to treat the since attribute as being a Rust version identifier
    /// (rather than an opaque string).
    pub is_since_rustc_version: bool,
}

/// Finds the deprecation attribute. `None` if none exists.
pub fn find_deprecation(sess: &Session, attrs: &[Attribute]) -> Option<(Deprecation, Span)> {
    find_deprecation_generic(sess, attrs.iter())
}

fn find_deprecation_generic<'a, I>(sess: &Session, attrs_iter: I) -> Option<(Deprecation, Span)>
where
    I: Iterator<Item = &'a Attribute>,
{
    let mut depr: Option<(Deprecation, Span)> = None;
    let is_rustc = sess.features_untracked().staged_api;

    'outer: for attr in attrs_iter {
        if !attr.has_name(sym::deprecated) {
            continue;
        }

        let Some(meta) = attr.meta() else {
            continue;
        };
        let mut since = None;
        let mut note = None;
        let mut suggestion = None;
        match &meta.kind {
            MetaItemKind::Word => {}
            MetaItemKind::NameValue(..) => note = meta.value_str(),
            MetaItemKind::List(list) => {
                let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                    if item.is_some() {
                        handle_errors(
                            &sess.parse_sess,
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
                                &sess.parse_sess,
                                lit.span,
                                AttrError::UnsupportedLiteral(
                                    UnsupportedLiteralReason::DeprecatedString,
                                    lit.kind.is_bytestr(),
                                ),
                            );
                        } else {
                            sess.emit_err(session_diagnostics::IncorrectMetaItem2 {
                                span: meta.span,
                            });
                        }

                        false
                    }
                };

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
                            sym::suggestion => {
                                if !sess.features_untracked().deprecated_suggestion {
                                    sess.emit_err(session_diagnostics::DeprecatedItemSuggestion {
                                        span: mi.span,
                                        is_nightly: sess.is_nightly_build().then_some(()),
                                        details: (),
                                    });
                                }

                                if !get(mi, &mut suggestion) {
                                    continue 'outer;
                                }
                            }
                            _ => {
                                handle_errors(
                                    &sess.parse_sess,
                                    meta.span(),
                                    AttrError::UnknownMetaItem(
                                        pprust::path_to_string(&mi.path),
                                        if sess.features_untracked().deprecated_suggestion {
                                            &["since", "note", "suggestion"]
                                        } else {
                                            &["since", "note"]
                                        },
                                    ),
                                );
                                continue 'outer;
                            }
                        },
                        NestedMetaItem::Lit(lit) => {
                            handle_errors(
                                &sess.parse_sess,
                                lit.span,
                                AttrError::UnsupportedLiteral(
                                    UnsupportedLiteralReason::DeprecatedKvPair,
                                    false,
                                ),
                            );
                            continue 'outer;
                        }
                    }
                }
            }
        }

        if is_rustc {
            if since.is_none() {
                handle_errors(&sess.parse_sess, attr.span, AttrError::MissingSince);
                continue;
            }

            if note.is_none() {
                sess.emit_err(session_diagnostics::MissingNote { span: attr.span });
                continue;
            }
        }

        depr = Some((
            Deprecation { since, note, suggestion, is_since_rustc_version: is_rustc },
            attr.span,
        ));
    }

    depr
}

#[derive(PartialEq, Debug, Encodable, Decodable, Copy, Clone)]
pub enum ReprAttr {
    ReprInt(IntType),
    ReprC,
    ReprPacked(u32),
    ReprSimd,
    ReprTransparent,
    ReprAlign(u32),
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
#[derive(Encodable, Decodable, HashStable_Generic)]
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
/// structure layout, `packed` to remove padding, and `transparent` to delegate representation
/// concerns to the only non-ZST field.
pub fn find_repr_attrs(sess: &Session, attr: &Attribute) -> Vec<ReprAttr> {
    if attr.has_name(sym::repr) { parse_repr_attr(sess, attr) } else { Vec::new() }
}

pub fn parse_repr_attr(sess: &Session, attr: &Attribute) -> Vec<ReprAttr> {
    assert!(attr.has_name(sym::repr), "expected `#[repr(..)]`, found: {attr:?}");
    use ReprAttr::*;
    let mut acc = Vec::new();
    let diagnostic = &sess.parse_sess.span_diagnostic;

    if let Some(items) = attr.meta_item_list() {
        for item in items {
            let mut recognised = false;
            if item.is_word() {
                let hint = match item.name_or_empty() {
                    sym::C => Some(ReprC),
                    sym::packed => Some(ReprPacked(1)),
                    sym::simd => Some(ReprSimd),
                    sym::transparent => Some(ReprTransparent),
                    sym::align => {
                        sess.emit_err(session_diagnostics::InvalidReprAlignNeedArg {
                            span: item.span(),
                        });
                        recognised = true;
                        None
                    }
                    name => int_type_of_word(name).map(ReprInt),
                };

                if let Some(h) = hint {
                    recognised = true;
                    acc.push(h);
                }
            } else if let Some((name, value)) = item.name_value_literal() {
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
                } else if matches!(name, sym::C | sym::simd | sym::transparent)
                    || int_type_of_word(name).is_some()
                {
                    recognised = true;
                    sess.emit_err(session_diagnostics::InvalidReprHintNoParen {
                        span: item.span(),
                        name: name.to_ident_string(),
                    });
                }
                if let Some(literal_error) = literal_error {
                    sess.emit_err(session_diagnostics::InvalidReprGeneric {
                        span: item.span(),
                        repr_arg: name.to_ident_string(),
                        error_part: literal_error,
                    });
                }
            } else if let Some(meta_item) = item.meta_item() {
                match &meta_item.kind {
                    MetaItemKind::NameValue(value) => {
                        if meta_item.has_name(sym::align) || meta_item.has_name(sym::packed) {
                            let name = meta_item.name_or_empty().to_ident_string();
                            recognised = true;
                            sess.emit_err(session_diagnostics::IncorrectReprFormatGeneric {
                                span: item.span(),
                                repr_arg: &name,
                                cause: IncorrectReprFormatGenericCause::from_lit_kind(
                                    item.span(),
                                    &value.kind,
                                    &name,
                                ),
                            });
                        } else if matches!(
                            meta_item.name_or_empty(),
                            sym::C | sym::simd | sym::transparent
                        ) || int_type_of_word(meta_item.name_or_empty()).is_some()
                        {
                            recognised = true;
                            sess.emit_err(session_diagnostics::InvalidReprHintNoValue {
                                span: meta_item.span,
                                name: meta_item.name_or_empty().to_ident_string(),
                            });
                        }
                    }
                    MetaItemKind::List(_) => {
                        if meta_item.has_name(sym::align) {
                            recognised = true;
                            sess.emit_err(session_diagnostics::IncorrectReprFormatAlignOneArg {
                                span: meta_item.span,
                            });
                        } else if meta_item.has_name(sym::packed) {
                            recognised = true;
                            sess.emit_err(
                                session_diagnostics::IncorrectReprFormatPackedOneOrZeroArg {
                                    span: meta_item.span,
                                },
                            );
                        } else if matches!(
                            meta_item.name_or_empty(),
                            sym::C | sym::simd | sym::transparent
                        ) || int_type_of_word(meta_item.name_or_empty()).is_some()
                        {
                            recognised = true;
                            sess.emit_err(session_diagnostics::InvalidReprHintNoParen {
                                span: meta_item.span,
                                name: meta_item.name_or_empty().to_ident_string(),
                            });
                        }
                    }
                    _ => (),
                }
            }
            if !recognised {
                // Not a word we recognize. This will be caught and reported by
                // the `check_mod_attrs` pass, but this pass doesn't always run
                // (e.g. if we only pretty-print the source), so we have to gate
                // the `delay_span_bug` call as follows:
                if sess.opts.pretty.map_or(true, |pp| pp.needs_analysis()) {
                    diagnostic.delay_span_bug(item.span(), "unrecognized representation hint");
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
    macro_rules: bool,
) -> (Transparency, Option<TransparencyError>) {
    let mut transparency = None;
    let mut error = None;
    for attr in attrs {
        if attr.has_name(sym::rustc_macro_transparency) {
            if let Some((_, old_span)) = transparency {
                error = Some(TransparencyError::MultipleTransparencyAttrs(old_span, attr.span));
                break;
            } else if let Some(value) = attr.value_str() {
                transparency = Some((
                    match value {
                        sym::transparent => Transparency::Transparent,
                        sym::semitransparent => Transparency::SemiTransparent,
                        sym::opaque => Transparency::Opaque,
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
    let fallback = if macro_rules { Transparency::SemiTransparent } else { Transparency::Opaque };
    (transparency.map_or(fallback, |t| t.0), error)
}

pub fn allow_internal_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [Attribute],
) -> impl Iterator<Item = Symbol> + 'a {
    allow_unstable(sess, attrs, sym::allow_internal_unstable)
}

pub fn rustc_allow_const_fn_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [Attribute],
) -> impl Iterator<Item = Symbol> + 'a {
    allow_unstable(sess, attrs, sym::rustc_allow_const_fn_unstable)
}

fn allow_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [Attribute],
    symbol: Symbol,
) -> impl Iterator<Item = Symbol> + 'a {
    let attrs = attr::filter_by_name(attrs, symbol);
    let list = attrs
        .filter_map(move |attr| {
            attr.meta_item_list().or_else(|| {
                sess.emit_err(session_diagnostics::ExpectsFeatureList {
                    span: attr.span,
                    name: symbol.to_ident_string(),
                });
                None
            })
        })
        .flatten();

    list.into_iter().filter_map(move |it| {
        let name = it.ident().map(|ident| ident.name);
        if name.is_none() {
            sess.emit_err(session_diagnostics::ExpectsFeatures {
                span: it.span(),
                name: symbol.to_ident_string(),
            });
        }
        name
    })
}

pub fn parse_alignment(node: &ast::LitKind) -> Result<u32, &'static str> {
    if let ast::LitKind::Int(literal, ast::LitIntType::Unsuffixed) = node {
        if literal.is_power_of_two() {
            // rustc_middle::ty::layout::Align restricts align to <= 2^29
            if *literal <= 1 << 29 { Ok(*literal as u32) } else { Err("larger than 2^29") }
        } else {
            Err("not a power of two")
        }
    } else {
        Err("not an unsuffixed integer")
    }
}
