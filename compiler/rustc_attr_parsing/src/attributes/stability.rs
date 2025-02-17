//! Parsing and validation of builtin attributes

use std::num::NonZero;

use rustc_ast::MetaItem;
use rustc_ast::attr::AttributeExt;
use rustc_ast_pretty::pprust;
use rustc_attr_data_structures::{
    ConstStability, DefaultBodyStability, Stability, StabilityLevel, StableSince, UnstableReason,
    VERSION_PLACEHOLDER,
};
use rustc_errors::ErrorGuaranteed;
use rustc_session::Session;
use rustc_span::{Span, Symbol, kw, sym};

use crate::attributes::util::UnsupportedLiteralReason;
use crate::{parse_version, session_diagnostics};

/// Collects stability info from `stable`/`unstable`/`rustc_allowed_through_unstable_modules`
/// attributes in `attrs`. Returns `None` if no stability attributes are found.
pub fn find_stability(
    sess: &Session,
    attrs: &[impl AttributeExt],
    item_sp: Span,
) -> Option<(Stability, Span)> {
    let mut stab: Option<(Stability, Span)> = None;
    let mut allowed_through_unstable_modules = None;

    for attr in attrs {
        match attr.name_or_empty() {
            sym::rustc_allowed_through_unstable_modules => {
                // The value is mandatory, but avoid ICEs in case such code reaches this function.
                allowed_through_unstable_modules = Some(attr.value_str().unwrap_or_else(|| {
                    sess.dcx().span_delayed_bug(
                        item_sp,
                        "`#[rustc_allowed_through_unstable_modules]` without deprecation message",
                    );
                    kw::Empty
                }))
            }
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

    if let Some(allowed_through_unstable_modules) = allowed_through_unstable_modules {
        match &mut stab {
            Some((
                Stability {
                    level: StabilityLevel::Stable { allowed_through_unstable_modules: in_stab, .. },
                    ..
                },
                _,
            )) => *in_stab = Some(allowed_through_unstable_modules),
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
            let level = StabilityLevel::Stable { since, allowed_through_unstable_modules: None };
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
