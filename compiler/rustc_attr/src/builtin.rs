//! Parsing and validation of builtin attributes

use std::num::NonZero;

use rustc_abi::Align;
use rustc_ast::attr::AttributeExt;
use rustc_ast::{self as ast, LitKind, MetaItem, MetaItemInner, MetaItemKind, MetaItemLit, NodeId};
use rustc_ast_pretty::pprust;
use rustc_errors::ErrorGuaranteed;
use rustc_feature::{Features, GatedCfg, find_gated_cfg, is_builtin_attr_name};
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_session::config::ExpectedValues;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::UNEXPECTED_CFGS;
use rustc_session::parse::feature_err;
use rustc_session::{RustcVersion, Session};
use rustc_span::Span;
use rustc_span::hygiene::Transparency;
use rustc_span::symbol::{Symbol, kw, sym};

use crate::session_diagnostics::{self, IncorrectReprFormatGenericCause};
use crate::{filter_by_name, first_attr_value_str_by_name, fluent_generated};

/// The version placeholder that recently stabilized features contain inside the
/// `since` field of the `#[stable]` attribute.
///
/// For more, see [this pull request](https://github.com/rust-lang/rust/pull/100591).
pub const VERSION_PLACEHOLDER: &str = "CURRENT_RUSTC_VERSION";

pub fn is_builtin_attr(attr: &impl AttributeExt) -> bool {
    attr.is_doc_comment() || attr.ident().is_some_and(|ident| is_builtin_attr_name(ident.name))
}

pub(crate) enum UnsupportedLiteralReason {
    Generic,
    CfgString,
    CfgBoolean,
    DeprecatedString,
    DeprecatedKvPair,
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

    pub fn stable_since(&self) -> Option<StableSince> {
        self.level.stable_since()
    }
}

/// Represents the `#[rustc_const_unstable]` and `#[rustc_const_stable]` attributes.
#[derive(Encodable, Decodable, Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable_Generic)]
pub struct ConstStability {
    pub level: StabilityLevel,
    pub feature: Symbol,
    /// This is true iff the `const_stable_indirect` attribute is present.
    pub const_stable_indirect: bool,
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
        issue: Option<NonZero<u32>>,
        is_soft: bool,
        /// If part of a feature is stabilized and a new feature is added for the remaining parts,
        /// then the `implied_by` attribute is used to indicate which now-stable feature previously
        /// contained an item.
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
        since: StableSince,
        /// Is this item allowed to be referred to on stable, despite being contained in unstable
        /// modules?
        allowed_through_unstable_modules: bool,
    },
}

/// Rust release in which a feature is stabilized.
#[derive(Encodable, Decodable, PartialEq, Copy, Clone, Debug, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable_Generic)]
pub enum StableSince {
    Version(RustcVersion),
    /// Stabilized in the upcoming version, whatever number that is.
    Current,
    /// Failed to parse a stabilization version.
    Err,
}

impl StabilityLevel {
    pub fn is_unstable(&self) -> bool {
        matches!(self, StabilityLevel::Unstable { .. })
    }
    pub fn is_stable(&self) -> bool {
        matches!(self, StabilityLevel::Stable { .. })
    }
    pub fn stable_since(&self) -> Option<StableSince> {
        match *self {
            StabilityLevel::Stable { since, .. } => Some(since),
            StabilityLevel::Unstable { .. } => None,
        }
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
    attrs: &[impl AttributeExt],
    item_sp: Span,
) -> Option<(Stability, Span)> {
    let mut stab: Option<(Stability, Span)> = None;
    let mut allowed_through_unstable_modules = false;

    for attr in attrs {
        match attr.name_or_empty() {
            sym::rustc_allowed_through_unstable_modules => allowed_through_unstable_modules = true,
            sym::unstable => {
                if stab.is_some() {
                    sess.dcx().emit_err(session_diagnostics::MultipleStabilityLevels {
                        span: attr.span(),
                    });
                    break;
                }

                if let Some((feature, level)) = parse_unstability(sess, attr) {
                    stab = Some((Stability { level, feature }, attr.span()));
                }
            }
            sym::stable => {
                if stab.is_some() {
                    sess.dcx().emit_err(session_diagnostics::MultipleStabilityLevels {
                        span: attr.span(),
                    });
                    break;
                }
                if let Some((feature, level)) = parse_stability(sess, attr) {
                    stab = Some((Stability { level, feature }, attr.span()));
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
                sess.dcx()
                    .emit_err(session_diagnostics::RustcAllowedUnstablePairing { span: item_sp });
            }
        }
    }

    stab
}

/// Collects stability info from `rustc_const_stable`/`rustc_const_unstable`/`rustc_promotable`
/// attributes in `attrs`. Returns `None` if no stability attributes are found.
pub fn find_const_stability(
    sess: &Session,
    attrs: &[impl AttributeExt],
    item_sp: Span,
) -> Option<(ConstStability, Span)> {
    let mut const_stab: Option<(ConstStability, Span)> = None;
    let mut promotable = false;
    let mut const_stable_indirect = false;

    for attr in attrs {
        match attr.name_or_empty() {
            sym::rustc_promotable => promotable = true,
            sym::rustc_const_stable_indirect => const_stable_indirect = true,
            sym::rustc_const_unstable => {
                if const_stab.is_some() {
                    sess.dcx().emit_err(session_diagnostics::MultipleStabilityLevels {
                        span: attr.span(),
                    });
                    break;
                }

                if let Some((feature, level)) = parse_unstability(sess, attr) {
                    const_stab = Some((
                        ConstStability {
                            level,
                            feature,
                            const_stable_indirect: false,
                            promotable: false,
                        },
                        attr.span(),
                    ));
                }
            }
            sym::rustc_const_stable => {
                if const_stab.is_some() {
                    sess.dcx().emit_err(session_diagnostics::MultipleStabilityLevels {
                        span: attr.span(),
                    });
                    break;
                }
                if let Some((feature, level)) = parse_stability(sess, attr) {
                    const_stab = Some((
                        ConstStability {
                            level,
                            feature,
                            const_stable_indirect: false,
                            promotable: false,
                        },
                        attr.span(),
                    ));
                }
            }
            _ => {}
        }
    }

    // Merge promotable and const_stable_indirect into stability info
    if promotable {
        match &mut const_stab {
            Some((stab, _)) => stab.promotable = promotable,
            _ => {
                _ = sess
                    .dcx()
                    .emit_err(session_diagnostics::RustcPromotablePairing { span: item_sp })
            }
        }
    }
    if const_stable_indirect {
        match &mut const_stab {
            Some((stab, _)) => {
                if stab.is_const_unstable() {
                    stab.const_stable_indirect = true;
                } else {
                    _ = sess.dcx().emit_err(session_diagnostics::RustcConstStableIndirectPairing {
                        span: item_sp,
                    })
                }
            }
            _ => {
                // This function has no const stability attribute, but has `const_stable_indirect`.
                // We ignore that; unmarked functions are subject to recursive const stability
                // checks by default so we do carry out the user's intent.
            }
        }
    }

    const_stab
}

/// Calculates the const stability for a const function in a `-Zforce-unstable-if-unmarked` crate
/// without the `staged_api` feature.
pub fn unmarked_crate_const_stab(
    _sess: &Session,
    attrs: &[impl AttributeExt],
    regular_stab: Stability,
) -> ConstStability {
    assert!(regular_stab.level.is_unstable());
    // The only attribute that matters here is `rustc_const_stable_indirect`.
    // We enforce recursive const stability rules for those functions.
    let const_stable_indirect =
        attrs.iter().any(|a| a.name_or_empty() == sym::rustc_const_stable_indirect);
    ConstStability {
        feature: regular_stab.feature,
        const_stable_indirect,
        promotable: false,
        level: regular_stab.level,
    }
}

/// Collects stability info from `rustc_default_body_unstable` attributes in `attrs`.
/// Returns `None` if no stability attributes are found.
pub fn find_body_stability(
    sess: &Session,
    attrs: &[impl AttributeExt],
) -> Option<(DefaultBodyStability, Span)> {
    let mut body_stab: Option<(DefaultBodyStability, Span)> = None;

    for attr in attrs {
        if attr.has_name(sym::rustc_default_body_unstable) {
            if body_stab.is_some() {
                sess.dcx()
                    .emit_err(session_diagnostics::MultipleStabilityLevels { span: attr.span() });
                break;
            }

            if let Some((feature, level)) = parse_unstability(sess, attr) {
                body_stab = Some((DefaultBodyStability { level, feature }, attr.span()));
            }
        }
    }

    body_stab
}

fn insert_or_error(sess: &Session, meta: &MetaItem, item: &mut Option<Symbol>) -> Option<()> {
    if item.is_some() {
        sess.dcx().emit_err(session_diagnostics::MultipleItem {
            span: meta.span,
            item: pprust::path_to_string(&meta.path),
        });
        None
    } else if let Some(v) = meta.value_str() {
        *item = Some(v);
        Some(())
    } else {
        sess.dcx().emit_err(session_diagnostics::IncorrectMetaItem { span: meta.span });
        None
    }
}

/// Read the content of a `stable`/`rustc_const_stable` attribute, and return the feature name and
/// its stability information.
fn parse_stability(sess: &Session, attr: &impl AttributeExt) -> Option<(Symbol, StabilityLevel)> {
    let metas = attr.meta_item_list()?;

    let mut feature = None;
    let mut since = None;
    for meta in metas {
        let Some(mi) = meta.meta_item() else {
            sess.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                span: meta.span(),
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: sess.source_map().start_point(meta.span()),
            });
            return None;
        };

        match mi.name_or_empty() {
            sym::feature => insert_or_error(sess, mi, &mut feature)?,
            sym::since => insert_or_error(sess, mi, &mut since)?,
            _ => {
                sess.dcx().emit_err(session_diagnostics::UnknownMetaItem {
                    span: meta.span(),
                    item: pprust::path_to_string(&mi.path),
                    expected: &["feature", "since"],
                });
                return None;
            }
        }
    }

    let feature = match feature {
        Some(feature) if rustc_lexer::is_ident(feature.as_str()) => Ok(feature),
        Some(_bad_feature) => {
            Err(sess.dcx().emit_err(session_diagnostics::NonIdentFeature { span: attr.span() }))
        }
        None => Err(sess.dcx().emit_err(session_diagnostics::MissingFeature { span: attr.span() })),
    };

    let since = if let Some(since) = since {
        if since.as_str() == VERSION_PLACEHOLDER {
            StableSince::Current
        } else if let Some(version) = parse_version(since) {
            StableSince::Version(version)
        } else {
            sess.dcx().emit_err(session_diagnostics::InvalidSince { span: attr.span() });
            StableSince::Err
        }
    } else {
        sess.dcx().emit_err(session_diagnostics::MissingSince { span: attr.span() });
        StableSince::Err
    };

    match feature {
        Ok(feature) => {
            let level = StabilityLevel::Stable { since, allowed_through_unstable_modules: false };
            Some((feature, level))
        }
        Err(ErrorGuaranteed { .. }) => None,
    }
}

/// Read the content of a `unstable`/`rustc_const_unstable`/`rustc_default_body_unstable`
/// attribute, and return the feature name and its stability information.
fn parse_unstability(sess: &Session, attr: &impl AttributeExt) -> Option<(Symbol, StabilityLevel)> {
    let metas = attr.meta_item_list()?;

    let mut feature = None;
    let mut reason = None;
    let mut issue = None;
    let mut issue_num = None;
    let mut is_soft = false;
    let mut implied_by = None;
    for meta in metas {
        let Some(mi) = meta.meta_item() else {
            sess.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                span: meta.span(),
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: sess.source_map().start_point(meta.span()),
            });
            return None;
        };

        match mi.name_or_empty() {
            sym::feature => insert_or_error(sess, mi, &mut feature)?,
            sym::reason => insert_or_error(sess, mi, &mut reason)?,
            sym::issue => {
                insert_or_error(sess, mi, &mut issue)?;

                // These unwraps are safe because `insert_or_error` ensures the meta item
                // is a name/value pair string literal.
                issue_num = match issue.unwrap().as_str() {
                    "none" => None,
                    issue => match issue.parse::<NonZero<u32>>() {
                        Ok(num) => Some(num),
                        Err(err) => {
                            sess.dcx().emit_err(
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
                    sess.dcx().emit_err(session_diagnostics::SoftNoArgs { span: mi.span });
                }
                is_soft = true;
            }
            sym::implied_by => insert_or_error(sess, mi, &mut implied_by)?,
            _ => {
                sess.dcx().emit_err(session_diagnostics::UnknownMetaItem {
                    span: meta.span(),
                    item: pprust::path_to_string(&mi.path),
                    expected: &["feature", "reason", "issue", "soft", "implied_by"],
                });
                return None;
            }
        }
    }

    let feature = match feature {
        Some(feature) if rustc_lexer::is_ident(feature.as_str()) => Ok(feature),
        Some(_bad_feature) => {
            Err(sess.dcx().emit_err(session_diagnostics::NonIdentFeature { span: attr.span() }))
        }
        None => Err(sess.dcx().emit_err(session_diagnostics::MissingFeature { span: attr.span() })),
    };

    let issue = issue.ok_or_else(|| {
        sess.dcx().emit_err(session_diagnostics::MissingIssue { span: attr.span() })
    });

    match (feature, issue) {
        (Ok(feature), Ok(_)) => {
            let level = StabilityLevel::Unstable {
                reason: UnstableReason::from_opt_reason(reason),
                issue: issue_num,
                is_soft,
                implied_by,
            };
            Some((feature, level))
        }
        (Err(ErrorGuaranteed { .. }), _) | (_, Err(ErrorGuaranteed { .. })) => None,
    }
}

pub fn find_crate_name(attrs: &[impl AttributeExt]) -> Option<Symbol> {
    first_attr_value_str_by_name(attrs, sym::crate_name)
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
    cfg: &ast::MetaItemInner,
    sess: &Session,
    lint_node_id: NodeId,
    features: Option<&Features>,
) -> bool {
    eval_condition(cfg, sess, features, &mut |cfg| {
        try_gate_cfg(cfg.name, cfg.span, sess, features);
        match sess.psess.check_config.expecteds.get(&cfg.name) {
            Some(ExpectedValues::Some(values)) if !values.contains(&cfg.value) => {
                sess.psess.buffer_lint(
                    UNEXPECTED_CFGS,
                    cfg.span,
                    lint_node_id,
                    BuiltinLintDiag::UnexpectedCfgValue(
                        (cfg.name, cfg.name_span),
                        cfg.value.map(|v| (v, cfg.value_span.unwrap())),
                    ),
                );
            }
            None if sess.psess.check_config.exhaustive_names => {
                sess.psess.buffer_lint(
                    UNEXPECTED_CFGS,
                    cfg.span,
                    lint_node_id,
                    BuiltinLintDiag::UnexpectedCfgName(
                        (cfg.name, cfg.name_span),
                        cfg.value.map(|v| (v, cfg.value_span.unwrap())),
                    ),
                );
            }
            _ => { /* not unexpected */ }
        }
        sess.psess.config.contains(&(cfg.name, cfg.value))
    })
}

fn try_gate_cfg(name: Symbol, span: Span, sess: &Session, features: Option<&Features>) {
    let gate = find_gated_cfg(|sym| sym == name);
    if let (Some(feats), Some(gated_cfg)) = (features, gate) {
        gate_cfg(gated_cfg, span, sess, feats);
    }
}

#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
fn gate_cfg(gated_cfg: &GatedCfg, cfg_span: Span, sess: &Session, features: &Features) {
    let (cfg, feature, has_feature) = gated_cfg;
    if !has_feature(features) && !cfg_span.allows_unstable(*feature) {
        let explain = format!("`cfg({cfg})` is experimental and subject to change");
        feature_err(sess, *feature, cfg_span, explain).emit();
    }
}

/// Parse a rustc version number written inside string literal in an attribute,
/// like appears in `since = "1.0.0"`. Suffixes like "-dev" and "-nightly" are
/// not accepted in this position, unlike when parsing CFG_RELEASE.
pub fn parse_version(s: Symbol) -> Option<RustcVersion> {
    let mut components = s.as_str().split('-');
    let d = components.next()?;
    if components.next().is_some() {
        return None;
    }
    let mut digits = d.splitn(3, '.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    let patch = digits.next().unwrap_or("0").parse().ok()?;
    Some(RustcVersion { major, minor, patch })
}

/// Evaluate a cfg-like condition (with `any` and `all`), using `eval` to
/// evaluate individual items.
pub fn eval_condition(
    cfg: &ast::MetaItemInner,
    sess: &Session,
    features: Option<&Features>,
    eval: &mut impl FnMut(Condition) -> bool,
) -> bool {
    let dcx = sess.dcx();

    let cfg = match cfg {
        ast::MetaItemInner::MetaItem(meta_item) => meta_item,
        ast::MetaItemInner::Lit(MetaItemLit { kind: LitKind::Bool(b), .. }) => {
            if let Some(features) = features {
                // we can't use `try_gate_cfg` as symbols don't differentiate between `r#true`
                // and `true`, and we want to keep the former working without feature gate
                gate_cfg(
                    &(
                        if *b { kw::True } else { kw::False },
                        sym::cfg_boolean_literals,
                        |features: &Features| features.cfg_boolean_literals(),
                    ),
                    cfg.span(),
                    sess,
                    features,
                );
            }
            return *b;
        }
        _ => {
            dcx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: cfg.span(),
                reason: UnsupportedLiteralReason::CfgBoolean,
                is_bytestr: false,
                start_point_span: sess.source_map().start_point(cfg.span()),
            });
            return false;
        }
    };

    match &cfg.kind {
        ast::MetaItemKind::List(mis) if cfg.name_or_empty() == sym::version => {
            try_gate_cfg(sym::version, cfg.span, sess, features);
            let (min_version, span) = match &mis[..] {
                [MetaItemInner::Lit(MetaItemLit { kind: LitKind::Str(sym, ..), span, .. })] => {
                    (sym, span)
                }
                [
                    MetaItemInner::Lit(MetaItemLit { span, .. })
                    | MetaItemInner::MetaItem(MetaItem { span, .. }),
                ] => {
                    dcx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: *span });
                    return false;
                }
                [..] => {
                    dcx.emit_err(session_diagnostics::ExpectedSingleVersionLiteral {
                        span: cfg.span,
                    });
                    return false;
                }
            };
            let Some(min_version) = parse_version(*min_version) else {
                dcx.emit_warn(session_diagnostics::UnknownVersionLiteral { span: *span });
                return false;
            };

            // See https://github.com/rust-lang/rust/issues/64796#issuecomment-640851454 for details
            if sess.psess.assume_incomplete_release {
                RustcVersion::CURRENT > min_version
            } else {
                RustcVersion::CURRENT >= min_version
            }
        }
        ast::MetaItemKind::List(mis) => {
            for mi in mis.iter() {
                if mi.meta_item_or_bool().is_none() {
                    dcx.emit_err(session_diagnostics::UnsupportedLiteral {
                        span: mi.span(),
                        reason: UnsupportedLiteralReason::Generic,
                        is_bytestr: false,
                        start_point_span: sess.source_map().start_point(mi.span()),
                    });
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
                    .fold(false, |res, mi| res | eval_condition(mi, sess, features, eval)),
                sym::all => mis
                    .iter()
                    // We don't use all() here, because we want to evaluate all cfg condition
                    // as eval_condition can (and does) extra checks
                    .fold(true, |res, mi| res & eval_condition(mi, sess, features, eval)),
                sym::not => {
                    let [mi] = mis.as_slice() else {
                        dcx.emit_err(session_diagnostics::ExpectedOneCfgPattern { span: cfg.span });
                        return false;
                    };

                    !eval_condition(mi, sess, features, eval)
                }
                sym::target => {
                    if let Some(features) = features
                        && !features.cfg_target_compact()
                    {
                        feature_err(
                            sess,
                            sym::cfg_target_compact,
                            cfg.span,
                            fluent_generated::attr_unstable_cfg_target_compact,
                        )
                        .emit();
                    }

                    mis.iter().fold(true, |res, mi| {
                        let Some(mut mi) = mi.meta_item().cloned() else {
                            dcx.emit_err(session_diagnostics::CfgPredicateIdentifier {
                                span: mi.span(),
                            });
                            return false;
                        };

                        if let [seg, ..] = &mut mi.path.segments[..] {
                            seg.ident.name = Symbol::intern(&format!("target_{}", seg.ident.name));
                        }

                        res & eval_condition(
                            &ast::MetaItemInner::MetaItem(mi),
                            sess,
                            features,
                            eval,
                        )
                    })
                }
                _ => {
                    dcx.emit_err(session_diagnostics::InvalidPredicate {
                        span: cfg.span,
                        predicate: pprust::path_to_string(&cfg.path),
                    });
                    false
                }
            }
        }
        ast::MetaItemKind::Word | MetaItemKind::NameValue(..) if cfg.path.segments.len() != 1 => {
            dcx.emit_err(session_diagnostics::CfgPredicateIdentifier { span: cfg.path.span });
            true
        }
        MetaItemKind::NameValue(lit) if !lit.kind.is_str() => {
            dcx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: lit.span,
                reason: UnsupportedLiteralReason::CfgString,
                is_bytestr: lit.kind.is_bytestr(),
                start_point_span: sess.source_map().start_point(lit.span),
            });
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
    pub since: DeprecatedSince,
    /// The note to issue a reason.
    pub note: Option<Symbol>,
    /// A text snippet used to completely replace any use of the deprecated item in an expression.
    ///
    /// This is currently unstable.
    pub suggestion: Option<Symbol>,
}

/// Release in which an API is deprecated.
#[derive(Copy, Debug, Encodable, Decodable, Clone, HashStable_Generic)]
pub enum DeprecatedSince {
    RustcVersion(RustcVersion),
    /// Deprecated in the future ("to be determined").
    Future,
    /// `feature(staged_api)` is off. Deprecation versions outside the standard
    /// library are allowed to be arbitrary strings, for better or worse.
    NonStandard(Symbol),
    /// Deprecation version is unspecified but optional.
    Unspecified,
    /// Failed to parse a deprecation version, or the deprecation version is
    /// unspecified and required. An error has already been emitted.
    Err,
}

impl Deprecation {
    /// Whether an item marked with #[deprecated(since = "X")] is currently
    /// deprecated (i.e., whether X is not greater than the current rustc
    /// version).
    pub fn is_in_effect(&self) -> bool {
        match self.since {
            DeprecatedSince::RustcVersion(since) => since <= RustcVersion::CURRENT,
            DeprecatedSince::Future => false,
            // The `since` field doesn't have semantic purpose without `#![staged_api]`.
            DeprecatedSince::NonStandard(_) => true,
            // Assume deprecation is in effect if "since" field is absent or invalid.
            DeprecatedSince::Unspecified | DeprecatedSince::Err => true,
        }
    }

    pub fn is_since_rustc_version(&self) -> bool {
        matches!(self.since, DeprecatedSince::RustcVersion(_))
    }
}

/// Finds the deprecation attribute. `None` if none exists.
pub fn find_deprecation(
    sess: &Session,
    features: &Features,
    attrs: &[impl AttributeExt],
) -> Option<(Deprecation, Span)> {
    let mut depr: Option<(Deprecation, Span)> = None;
    let is_rustc = features.staged_api();

    'outer: for attr in attrs {
        if !attr.has_name(sym::deprecated) {
            continue;
        }

        let mut since = None;
        let mut note = None;
        let mut suggestion = None;

        if attr.is_doc_comment() {
            continue;
        } else if attr.is_word() {
        } else if let Some(value) = attr.value_str() {
            note = Some(value)
        } else if let Some(list) = attr.meta_item_list() {
            let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                if item.is_some() {
                    sess.dcx().emit_err(session_diagnostics::MultipleItem {
                        span: meta.span,
                        item: pprust::path_to_string(&meta.path),
                    });
                    return false;
                }
                if let Some(v) = meta.value_str() {
                    *item = Some(v);
                    true
                } else {
                    if let Some(lit) = meta.name_value_literal() {
                        sess.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                            span: lit.span,
                            reason: UnsupportedLiteralReason::DeprecatedString,
                            is_bytestr: lit.kind.is_bytestr(),
                            start_point_span: sess.source_map().start_point(lit.span),
                        });
                    } else {
                        sess.dcx()
                            .emit_err(session_diagnostics::IncorrectMetaItem { span: meta.span });
                    }
                    false
                }
            };

            for meta in &list {
                match meta {
                    MetaItemInner::MetaItem(mi) => match mi.name_or_empty() {
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
                            if !features.deprecated_suggestion() {
                                sess.dcx().emit_err(
                                    session_diagnostics::DeprecatedItemSuggestion {
                                        span: mi.span,
                                        is_nightly: sess.is_nightly_build(),
                                        details: (),
                                    },
                                );
                            }

                            if !get(mi, &mut suggestion) {
                                continue 'outer;
                            }
                        }
                        _ => {
                            sess.dcx().emit_err(session_diagnostics::UnknownMetaItem {
                                span: meta.span(),
                                item: pprust::path_to_string(&mi.path),
                                expected: if features.deprecated_suggestion() {
                                    &["since", "note", "suggestion"]
                                } else {
                                    &["since", "note"]
                                },
                            });
                            continue 'outer;
                        }
                    },
                    MetaItemInner::Lit(lit) => {
                        sess.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                            span: lit.span,
                            reason: UnsupportedLiteralReason::DeprecatedKvPair,
                            is_bytestr: false,
                            start_point_span: sess.source_map().start_point(lit.span),
                        });
                        continue 'outer;
                    }
                }
            }
        } else {
            continue;
        }

        let since = if let Some(since) = since {
            if since.as_str() == "TBD" {
                DeprecatedSince::Future
            } else if !is_rustc {
                DeprecatedSince::NonStandard(since)
            } else if let Some(version) = parse_version(since) {
                DeprecatedSince::RustcVersion(version)
            } else {
                sess.dcx().emit_err(session_diagnostics::InvalidSince { span: attr.span() });
                DeprecatedSince::Err
            }
        } else if is_rustc {
            sess.dcx().emit_err(session_diagnostics::MissingSince { span: attr.span() });
            DeprecatedSince::Err
        } else {
            DeprecatedSince::Unspecified
        };

        if is_rustc && note.is_none() {
            sess.dcx().emit_err(session_diagnostics::MissingNote { span: attr.span() });
            continue;
        }

        depr = Some((Deprecation { since, note, suggestion }, attr.span()));
    }

    depr
}

#[derive(PartialEq, Debug, Encodable, Decodable, Copy, Clone)]
pub enum ReprAttr {
    ReprInt(IntType),
    ReprRust,
    ReprC,
    ReprPacked(Align),
    ReprSimd,
    ReprTransparent,
    ReprAlign(Align),
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
pub fn find_repr_attrs(sess: &Session, attr: &impl AttributeExt) -> Vec<ReprAttr> {
    if attr.has_name(sym::repr) { parse_repr_attr(sess, attr) } else { Vec::new() }
}

pub fn parse_repr_attr(sess: &Session, attr: &impl AttributeExt) -> Vec<ReprAttr> {
    assert!(attr.has_name(sym::repr), "expected `#[repr(..)]`, found: {attr:?}");
    use ReprAttr::*;
    let mut acc = Vec::new();
    let dcx = sess.dcx();

    if let Some(items) = attr.meta_item_list() {
        for item in items {
            let mut recognised = false;
            if item.is_word() {
                let hint = match item.name_or_empty() {
                    sym::Rust => Some(ReprRust),
                    sym::C => Some(ReprC),
                    sym::packed => Some(ReprPacked(Align::ONE)),
                    sym::simd => Some(ReprSimd),
                    sym::transparent => Some(ReprTransparent),
                    sym::align => {
                        sess.dcx().emit_err(session_diagnostics::InvalidReprAlignNeedArg {
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
            } else if let Some((name, value)) = item.singleton_lit_list() {
                let mut literal_error = None;
                let mut err_span = item.span();
                if name == sym::align {
                    recognised = true;
                    match parse_alignment(&value.kind) {
                        Ok(literal) => acc.push(ReprAlign(literal)),
                        Err(message) => {
                            err_span = value.span;
                            literal_error = Some(message)
                        }
                    };
                } else if name == sym::packed {
                    recognised = true;
                    match parse_alignment(&value.kind) {
                        Ok(literal) => acc.push(ReprPacked(literal)),
                        Err(message) => {
                            err_span = value.span;
                            literal_error = Some(message)
                        }
                    };
                } else if matches!(name, sym::Rust | sym::C | sym::simd | sym::transparent)
                    || int_type_of_word(name).is_some()
                {
                    recognised = true;
                    sess.dcx().emit_err(session_diagnostics::InvalidReprHintNoParen {
                        span: item.span(),
                        name: name.to_ident_string(),
                    });
                }
                if let Some(literal_error) = literal_error {
                    sess.dcx().emit_err(session_diagnostics::InvalidReprGeneric {
                        span: err_span,
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
                            sess.dcx().emit_err(session_diagnostics::IncorrectReprFormatGeneric {
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
                            sym::Rust | sym::C | sym::simd | sym::transparent
                        ) || int_type_of_word(meta_item.name_or_empty()).is_some()
                        {
                            recognised = true;
                            sess.dcx().emit_err(session_diagnostics::InvalidReprHintNoValue {
                                span: meta_item.span,
                                name: meta_item.name_or_empty().to_ident_string(),
                            });
                        }
                    }
                    MetaItemKind::List(nested_items) => {
                        if meta_item.has_name(sym::align) {
                            recognised = true;
                            if let [nested_item] = nested_items.as_slice() {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatExpectInteger {
                                        span: nested_item.span(),
                                    },
                                );
                            } else {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatAlignOneArg {
                                        span: meta_item.span,
                                    },
                                );
                            }
                        } else if meta_item.has_name(sym::packed) {
                            recognised = true;
                            if let [nested_item] = nested_items.as_slice() {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatPackedExpectInteger {
                                        span: nested_item.span(),
                                    },
                                );
                            } else {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatPackedOneOrZeroArg {
                                        span: meta_item.span,
                                    },
                                );
                            }
                        } else if matches!(
                            meta_item.name_or_empty(),
                            sym::Rust | sym::C | sym::simd | sym::transparent
                        ) || int_type_of_word(meta_item.name_or_empty()).is_some()
                        {
                            recognised = true;
                            sess.dcx().emit_err(session_diagnostics::InvalidReprHintNoParen {
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
                // the `span_delayed_bug` call as follows:
                if sess.opts.pretty.map_or(true, |pp| pp.needs_analysis()) {
                    dcx.span_delayed_bug(item.span(), "unrecognized representation hint");
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
    attrs: &[impl AttributeExt],
    macro_rules: bool,
) -> (Transparency, Option<TransparencyError>) {
    let mut transparency = None;
    let mut error = None;
    for attr in attrs {
        if attr.has_name(sym::rustc_macro_transparency) {
            if let Some((_, old_span)) = transparency {
                error = Some(TransparencyError::MultipleTransparencyAttrs(old_span, attr.span()));
                break;
            } else if let Some(value) = attr.value_str() {
                transparency = Some((
                    match value {
                        sym::transparent => Transparency::Transparent,
                        sym::semitransparent => Transparency::SemiTransparent,
                        sym::opaque => Transparency::Opaque,
                        _ => {
                            error =
                                Some(TransparencyError::UnknownTransparency(value, attr.span()));
                            continue;
                        }
                    },
                    attr.span(),
                ));
            }
        }
    }
    let fallback = if macro_rules { Transparency::SemiTransparent } else { Transparency::Opaque };
    (transparency.map_or(fallback, |t| t.0), error)
}

pub fn allow_internal_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [impl AttributeExt],
) -> impl Iterator<Item = Symbol> + 'a {
    allow_unstable(sess, attrs, sym::allow_internal_unstable)
}

pub fn rustc_allow_const_fn_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [impl AttributeExt],
) -> impl Iterator<Item = Symbol> + 'a {
    allow_unstable(sess, attrs, sym::rustc_allow_const_fn_unstable)
}

fn allow_unstable<'a>(
    sess: &'a Session,
    attrs: &'a [impl AttributeExt],
    symbol: Symbol,
) -> impl Iterator<Item = Symbol> + 'a {
    let attrs = filter_by_name(attrs, symbol);
    let list = attrs
        .filter_map(move |attr| {
            attr.meta_item_list().or_else(|| {
                sess.dcx().emit_err(session_diagnostics::ExpectsFeatureList {
                    span: attr.span(),
                    name: symbol.to_ident_string(),
                });
                None
            })
        })
        .flatten();

    list.into_iter().filter_map(move |it| {
        let name = it.ident().map(|ident| ident.name);
        if name.is_none() {
            sess.dcx().emit_err(session_diagnostics::ExpectsFeatures {
                span: it.span(),
                name: symbol.to_ident_string(),
            });
        }
        name
    })
}

pub fn parse_alignment(node: &ast::LitKind) -> Result<Align, &'static str> {
    if let ast::LitKind::Int(literal, ast::LitIntType::Unsuffixed) = node {
        // `Align::from_bytes` accepts 0 as an input, check is_power_of_two() first
        if literal.get().is_power_of_two() {
            // Only possible error is larger than 2^29
            literal
                .get()
                .try_into()
                .ok()
                .and_then(|v| Align::from_bytes(v).ok())
                .ok_or("larger than 2^29")
        } else {
            Err("not a power of two")
        }
    } else {
        Err("not an unsuffixed integer")
    }
}

/// Read the content of a `rustc_confusables` attribute, and return the list of candidate names.
pub fn parse_confusables(attr: &impl AttributeExt) -> Option<Vec<Symbol>> {
    let metas = attr.meta_item_list()?;

    let mut candidates = Vec::new();

    for meta in metas {
        let MetaItemInner::Lit(meta_lit) = meta else {
            return None;
        };
        candidates.push(meta_lit.symbol);
    }

    Some(candidates)
}
