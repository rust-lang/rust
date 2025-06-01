use std::num::NonZero;

use rustc_attr_data_structures::{
    AttributeKind, DefaultBodyStability, PartialConstStability, Stability, StabilityLevel,
    StableSince, UnstableReason, VERSION_PLACEHOLDER,
};
use rustc_errors::ErrorGuaranteed;
use rustc_span::{Span, Symbol, sym};

use super::util::parse_version;
use super::{AcceptMapping, AttributeParser, SingleAttributeParser};
use crate::context::{AcceptContext, FinalizeContext};
use crate::parser::{ArgParser, MetaItemParser};
use crate::session_diagnostics::{self, UnsupportedLiteralReason};

macro_rules! reject_outside_std {
    ($cx: ident) => {
        // Emit errors for non-staged-api crates.
        if !$cx.features().staged_api() {
            $cx.emit_err(session_diagnostics::StabilityOutsideStd { span: $cx.attr_span });
            return;
        }
    };
}

#[derive(Default)]
pub(crate) struct StabilityParser {
    allowed_through_unstable_modules: Option<Symbol>,
    stability: Option<(Stability, Span)>,
}

impl StabilityParser {
    /// Checks, and emits an error when a stability (or unstability) was already set, which would be a duplicate.
    fn check_duplicate(&self, cx: &AcceptContext<'_>) -> bool {
        if let Some((_, _)) = self.stability {
            cx.emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            true
        } else {
            false
        }
    }
}

impl AttributeParser for StabilityParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (&[sym::stable], |this, cx, args| {
            reject_outside_std!(cx);
            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_stability(cx, args)
            {
                this.stability = Some((Stability { level, feature }, cx.attr_span));
            }
        }),
        (&[sym::unstable], |this, cx, args| {
            reject_outside_std!(cx);
            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_unstability(cx, args)
            {
                this.stability = Some((Stability { level, feature }, cx.attr_span));
            }
        }),
        (&[sym::rustc_allowed_through_unstable_modules], |this, cx, args| {
            reject_outside_std!(cx);
            this.allowed_through_unstable_modules = args.name_value().and_then(|i| i.value_as_str())
        }),
    ];

    fn finalize(mut self, cx: &FinalizeContext<'_>) -> Option<AttributeKind> {
        if let Some(atum) = self.allowed_through_unstable_modules {
            if let Some((
                Stability {
                    level: StabilityLevel::Stable { ref mut allowed_through_unstable_modules, .. },
                    ..
                },
                _,
            )) = self.stability
            {
                *allowed_through_unstable_modules = Some(atum);
            } else {
                cx.dcx().emit_err(session_diagnostics::RustcAllowedUnstablePairing {
                    span: cx.target_span,
                });
            }
        }

        let (stability, span) = self.stability?;

        Some(AttributeKind::Stability { stability, span })
    }
}

// FIXME(jdonszelmann) change to Single
#[derive(Default)]
pub(crate) struct BodyStabilityParser {
    stability: Option<(DefaultBodyStability, Span)>,
}

impl AttributeParser for BodyStabilityParser {
    const ATTRIBUTES: AcceptMapping<Self> =
        &[(&[sym::rustc_default_body_unstable], |this, cx, args| {
            reject_outside_std!(cx);
            if this.stability.is_some() {
                cx.dcx()
                    .emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            } else if let Some((feature, level)) = parse_unstability(cx, args) {
                this.stability = Some((DefaultBodyStability { level, feature }, cx.attr_span));
            }
        })];

    fn finalize(self, _cx: &FinalizeContext<'_>) -> Option<AttributeKind> {
        let (stability, span) = self.stability?;

        Some(AttributeKind::BodyStability { stability, span })
    }
}

pub(crate) struct ConstStabilityIndirectParser;
// FIXME(jdonszelmann): single word attribute group when we have these
impl SingleAttributeParser for ConstStabilityIndirectParser {
    const PATH: &'static [Symbol] = &[sym::rustc_const_stable_indirect];

    // ignore
    fn on_duplicate(_cx: &AcceptContext<'_>, _first_span: Span) {}

    fn convert(_cx: &AcceptContext<'_>, _args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::ConstStabilityIndirect)
    }
}

#[derive(Default)]
pub(crate) struct ConstStabilityParser {
    promotable: bool,
    stability: Option<(PartialConstStability, Span)>,
}

impl ConstStabilityParser {
    /// Checks, and emits an error when a stability (or unstability) was already set, which would be a duplicate.
    fn check_duplicate(&self, cx: &AcceptContext<'_>) -> bool {
        if let Some((_, _)) = self.stability {
            cx.emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            true
        } else {
            false
        }
    }
}

impl AttributeParser for ConstStabilityParser {
    const ATTRIBUTES: AcceptMapping<Self> = &[
        (&[sym::rustc_const_stable], |this, cx, args| {
            reject_outside_std!(cx);

            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_stability(cx, args)
            {
                this.stability = Some((
                    PartialConstStability { level, feature, promotable: false },
                    cx.attr_span,
                ));
            }
        }),
        (&[sym::rustc_const_unstable], |this, cx, args| {
            reject_outside_std!(cx);
            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_unstability(cx, args)
            {
                this.stability = Some((
                    PartialConstStability { level, feature, promotable: false },
                    cx.attr_span,
                ));
            }
        }),
        (&[sym::rustc_promotable], |this, cx, _| {
            reject_outside_std!(cx);
            this.promotable = true;
        }),
    ];

    fn finalize(mut self, cx: &FinalizeContext<'_>) -> Option<AttributeKind> {
        if self.promotable {
            if let Some((ref mut stab, _)) = self.stability {
                stab.promotable = true;
            } else {
                cx.dcx()
                    .emit_err(session_diagnostics::RustcPromotablePairing { span: cx.target_span });
            }
        }

        let (stability, span) = self.stability?;

        Some(AttributeKind::ConstStability { stability, span })
    }
}

/// Tries to insert the value of a `key = value` meta item into an option.
///
/// Emits an error when either the option was already Some, or the arguments weren't of form
/// `name = value`
fn insert_value_into_option_or_error(
    cx: &AcceptContext<'_>,
    param: &MetaItemParser<'_>,
    item: &mut Option<Symbol>,
) -> Option<()> {
    if item.is_some() {
        cx.emit_err(session_diagnostics::MultipleItem {
            span: param.span(),
            item: param.path_without_args().to_string(),
        });
        None
    } else if let Some(v) = param.args().name_value()
        && let Some(s) = v.value_as_str()
    {
        *item = Some(s);
        Some(())
    } else {
        cx.emit_err(session_diagnostics::IncorrectMetaItem {
            span: param.span(),
            suggestion: None,
        });
        None
    }
}

/// Read the content of a `stable`/`rustc_const_stable` attribute, and return the feature name and
/// its stability information.
pub(crate) fn parse_stability(
    cx: &AcceptContext<'_>,
    args: &ArgParser<'_>,
) -> Option<(Symbol, StabilityLevel)> {
    let mut feature = None;
    let mut since = None;

    for param in args.list()?.mixed() {
        let param_span = param.span();
        let Some(param) = param.meta_item() else {
            cx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: param_span,
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: cx.sess().source_map().start_point(param_span),
            });
            return None;
        };

        match param.path_without_args().word_sym() {
            Some(sym::feature) => insert_value_into_option_or_error(cx, &param, &mut feature)?,
            Some(sym::since) => insert_value_into_option_or_error(cx, &param, &mut since)?,
            _ => {
                cx.emit_err(session_diagnostics::UnknownMetaItem {
                    span: param_span,
                    item: param.path_without_args().to_string(),
                    expected: &["feature", "since"],
                });
                return None;
            }
        }
    }

    let feature = match feature {
        Some(feature) if rustc_lexer::is_ident(feature.as_str()) => Ok(feature),
        Some(_bad_feature) => {
            Err(cx.emit_err(session_diagnostics::NonIdentFeature { span: cx.attr_span }))
        }
        None => Err(cx.emit_err(session_diagnostics::MissingFeature { span: cx.attr_span })),
    };

    let since = if let Some(since) = since {
        if since.as_str() == VERSION_PLACEHOLDER {
            StableSince::Current
        } else if let Some(version) = parse_version(since) {
            StableSince::Version(version)
        } else {
            cx.emit_err(session_diagnostics::InvalidSince { span: cx.attr_span });
            StableSince::Err
        }
    } else {
        cx.emit_err(session_diagnostics::MissingSince { span: cx.attr_span });
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

// Read the content of a `unstable`/`rustc_const_unstable`/`rustc_default_body_unstable`
/// attribute, and return the feature name and its stability information.
pub(crate) fn parse_unstability(
    cx: &AcceptContext<'_>,
    args: &ArgParser<'_>,
) -> Option<(Symbol, StabilityLevel)> {
    let mut feature = None;
    let mut reason = None;
    let mut issue = None;
    let mut issue_num = None;
    let mut is_soft = false;
    let mut implied_by = None;
    for param in args.list()?.mixed() {
        let Some(param) = param.meta_item() else {
            cx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: param.span(),
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: cx.sess().source_map().start_point(param.span()),
            });
            return None;
        };

        match param.path_without_args().word_sym() {
            Some(sym::feature) => insert_value_into_option_or_error(cx, &param, &mut feature)?,
            Some(sym::reason) => insert_value_into_option_or_error(cx, &param, &mut reason)?,
            Some(sym::issue) => {
                insert_value_into_option_or_error(cx, &param, &mut issue)?;

                // These unwraps are safe because `insert_value_into_option_or_error` ensures the meta item
                // is a name/value pair string literal.
                issue_num = match issue.unwrap().as_str() {
                    "none" => None,
                    issue_str => match issue_str.parse::<NonZero<u32>>() {
                        Ok(num) => Some(num),
                        Err(err) => {
                            cx.emit_err(
                                session_diagnostics::InvalidIssueString {
                                    span: param.span(),
                                    cause: session_diagnostics::InvalidIssueStringCause::from_int_error_kind(
                                        param.args().name_value().unwrap().value_span,
                                        err.kind(),
                                    ),
                                },
                            );
                            return None;
                        }
                    },
                };
            }
            Some(sym::soft) => {
                if !param.args().no_args() {
                    cx.emit_err(session_diagnostics::SoftNoArgs { span: param.span() });
                }
                is_soft = true;
            }
            Some(sym::implied_by) => {
                insert_value_into_option_or_error(cx, &param, &mut implied_by)?
            }
            _ => {
                cx.emit_err(session_diagnostics::UnknownMetaItem {
                    span: param.span(),
                    item: param.path_without_args().to_string(),
                    expected: &["feature", "reason", "issue", "soft", "implied_by"],
                });
                return None;
            }
        }
    }

    let feature = match feature {
        Some(feature) if rustc_lexer::is_ident(feature.as_str()) => Ok(feature),
        Some(_bad_feature) => {
            Err(cx.emit_err(session_diagnostics::NonIdentFeature { span: cx.attr_span }))
        }
        None => Err(cx.emit_err(session_diagnostics::MissingFeature { span: cx.attr_span })),
    };

    let issue =
        issue.ok_or_else(|| cx.emit_err(session_diagnostics::MissingIssue { span: cx.attr_span }));

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
