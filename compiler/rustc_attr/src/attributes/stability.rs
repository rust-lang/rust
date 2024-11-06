use std::num::NonZero;

use rustc_hir::{
    AttributeKind, DefaultBodyStability, PartialConstStability, Stability, StabilityLevel,
    StableSince, UnstableReason, VERSION_PLACEHOLDER,
};
use rustc_span::{ErrorGuaranteed, Span, Symbol, sym};

use super::util::parse_version;
use super::{AttributeFilter, AttributeGroup, AttributeMapping, SingleAttributeGroup};
use crate::attribute_filter;
use crate::context::{AttributeAcceptContext, AttributeGroupContext};
use crate::parser::{ArgParser, MetaItemParser, NameValueParser};
use crate::session_diagnostics::{self, UnsupportedLiteralReason};

#[derive(Default)]
pub(crate) struct StabilityGroup {
    allowed_through_unstable_modules: bool,
    stability: Option<(Stability, Span)>,
}

impl StabilityGroup {
    /// Checks, and emits an error when a stability (or unstability) was already set, which would be a duplicate.
    fn check_duplicate(&self, cx: &AttributeAcceptContext<'_>) -> bool {
        if let Some((_, span)) = self.stability {
            cx.dcx().emit_err(session_diagnostics::MultipleStabilityLevels { span });
            true
        } else {
            false
        }
    }
}

impl AttributeGroup for StabilityGroup {
    const ATTRIBUTES: AttributeMapping<Self> = &[
        (&[sym::stable], |this, cx, args| {
            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_stability(cx, args)
            {
                this.stability = Some((Stability { level, feature }, cx.attr_span));
            }
        }),
        (&[sym::unstable], |this, cx, args| {
            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_unstability(cx, args)
            {
                this.stability = Some((Stability { level, feature }, cx.attr_span));
            }
        }),
        (&[sym::rustc_allowed_through_unstable_modules], |this, _, _| {
            this.allowed_through_unstable_modules = true;
        }),
    ];

    fn finalize(self, cx: &AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)> {
        let (mut stability, span) = self.stability?;

        if self.allowed_through_unstable_modules {
            if let StabilityLevel::Stable { ref mut allowed_through_unstable_modules, .. } =
                stability.level
            {
                *allowed_through_unstable_modules = true;
            } else {
                cx.dcx().emit_err(session_diagnostics::RustcAllowedUnstablePairing {
                    span: cx.target_span,
                });
            }
        }

        // Emit errors for non-staged-api crates.
        if !cx.features().staged_api() {
            cx.dcx().emit_err(session_diagnostics::StabilityOutsideStd { span });
        }

        Some((AttributeKind::Stability { stability, span }, attribute_filter!(allow all)))
    }
}

// FIXME(jdonszelmann) change to Single
#[derive(Default)]
pub(crate) struct BodyStabilityGroup {
    stability: Option<(DefaultBodyStability, Span)>,
}

impl AttributeGroup for BodyStabilityGroup {
    const ATTRIBUTES: AttributeMapping<Self> =
        &[(&[sym::rustc_default_body_unstable], |this, cx, args| {
            if this.stability.is_some() {
                cx.dcx()
                    .emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            } else if let Some((feature, level)) = parse_stability(cx, args) {
                this.stability = Some((DefaultBodyStability { level, feature }, cx.attr_span));
            }
        })];

    fn finalize(self, cx: &AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)> {
        let (stability, span) = self.stability?;

        // Emit errors for non-staged-api crates.
        if !cx.features().staged_api() {
            cx.dcx().emit_err(session_diagnostics::StabilityOutsideStd { span });
        }

        Some((AttributeKind::BodyStability { stability, span }, attribute_filter!(allow all)))
    }
}

pub(crate) struct ConstStabilityIndirectGroup;
// TODO: single word attribute group
impl SingleAttributeGroup for ConstStabilityIndirectGroup {
    const PATH: &'static [rustc_span::Symbol] = &[sym::rustc_const_stable_indirect];

    // ignore
    fn on_duplicate(_cx: &AttributeAcceptContext<'_>, _first_span: Span) {}

    fn convert(
        _cx: &AttributeAcceptContext<'_>,
        _args: &super::GenericArgParser<'_, rustc_ast::Expr>,
    ) -> Option<(AttributeKind, AttributeFilter)> {
        Some((AttributeKind::ConstStabilityIndirect, attribute_filter!(allow all)))
    }
}

#[derive(Default)]
pub(crate) struct ConstStabilityGroup {
    promotable: bool,
    stability: Option<(PartialConstStability, Span)>,
}

impl ConstStabilityGroup {
    /// Checks, and emits an error when a stability (or unstability) was already set, which would be a duplicate.
    fn check_duplicate(&self, cx: &AttributeAcceptContext<'_>) -> bool {
        if let Some((_, span)) = self.stability {
            cx.dcx().emit_err(session_diagnostics::MultipleStabilityLevels { span });
            true
        } else {
            false
        }
    }
}

impl AttributeGroup for ConstStabilityGroup {
    const ATTRIBUTES: AttributeMapping<Self> = &[
        (&[sym::rustc_const_stable], |this, cx, args| {
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
            if !this.check_duplicate(cx)
                && let Some((feature, level)) = parse_unstability(cx, args)
            {
                this.stability = Some((
                    PartialConstStability { level, feature, promotable: false },
                    cx.attr_span,
                ));
            }
        }),
        (&[sym::rustc_promotable], |this, _, _| {
            this.promotable = true;
        }),
    ];

    fn finalize(
        mut self,
        cx: &AttributeGroupContext<'_>,
    ) -> Option<(AttributeKind, AttributeFilter)> {
        if self.promotable {
            if let Some((ref mut stab, _)) = self.stability {
                stab.promotable = true;
            } else {
                cx.dcx()
                    .emit_err(session_diagnostics::RustcPromotablePairing { span: cx.target_span });
            }
        }

        let (stability, span) = self.stability?;

        // Emit errors for non-staged-api crates.
        if !cx.features().staged_api() {
            cx.dcx().emit_err(session_diagnostics::StabilityOutsideStd { span });
        }

        Some((AttributeKind::ConstStability { stability, span }, attribute_filter!(allow all)))
    }
}

/// Tries to insert the value of a `key = value` meta item into an option.
///
/// Emits an error when either the option was already Some, or the arguments weren't of form
/// `name = value`
fn insert_value_into_option_or_error<'a>(
    cx: &AttributeAcceptContext<'_>,
    param: &impl MetaItemParser<'a>,
    item: &mut Option<Symbol>,
) -> Option<()> {
    if item.is_some() {
        cx.dcx().emit_err(session_diagnostics::MultipleItem {
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
        cx.dcx().emit_err(session_diagnostics::IncorrectMetaItem {
            span: param.span(),
            suggestion: None,
        });
        None
    }
}

/// Read the content of a `stable`/`rustc_const_stable` attribute, and return the feature name and
/// its stability information.
pub(crate) fn parse_stability<'a>(
    cx: &AttributeAcceptContext<'_>,
    args: &'a impl ArgParser<'a>,
) -> Option<(Symbol, StabilityLevel)> {
    let mut feature = None;
    let mut since = None;

    for param in args.list()?.mixed() {
        let param_span = param.span();
        let Some(param) = param.meta_item() else {
            cx.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                span: param_span,
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: cx.sess().source_map().start_point(param_span),
            });
            return None;
        };

        match param.word_or_empty_without_args().name {
            sym::feature => insert_value_into_option_or_error(cx, &param, &mut feature)?,
            sym::since => insert_value_into_option_or_error(cx, &param, &mut since)?,
            _ => {
                cx.dcx().emit_err(session_diagnostics::UnknownMetaItem {
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
            Err(cx.dcx().emit_err(session_diagnostics::NonIdentFeature { span: cx.attr_span }))
        }
        None => Err(cx.dcx().emit_err(session_diagnostics::MissingFeature { span: cx.attr_span })),
    };

    let since = if let Some(since) = since {
        if since.as_str() == VERSION_PLACEHOLDER {
            StableSince::Current
        } else if let Some(version) = parse_version(since) {
            StableSince::Version(version, since)
        } else {
            cx.dcx().emit_err(session_diagnostics::InvalidSince { span: cx.attr_span });
            StableSince::Err
        }
    } else {
        cx.dcx().emit_err(session_diagnostics::MissingSince { span: cx.attr_span });
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

// Read the content of a `unstable`/`rustc_const_unstable`/`rustc_default_body_unstable`
/// attribute, and return the feature name and its stability information.
pub(crate) fn parse_unstability<'a>(
    cx: &AttributeAcceptContext<'_>,
    args: &'a impl ArgParser<'a>,
) -> Option<(Symbol, StabilityLevel)> {
    let mut feature = None;
    let mut reason = None;
    let mut issue = None;
    let mut issue_num = None;
    let mut is_soft = false;
    let mut implied_by = None;
    for param in args.list()?.mixed() {
        let param_span = param.span();
        let Some(param) = param.meta_item() else {
            cx.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                span: param_span,
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: cx.sess().source_map().start_point(param_span),
            });
            return None;
        };

        let (word, args) = param.word_or_empty();
        match word.name {
            sym::feature => insert_value_into_option_or_error(cx, &param, &mut feature)?,
            sym::reason => insert_value_into_option_or_error(cx, &param, &mut reason)?,
            sym::issue => {
                insert_value_into_option_or_error(cx, &param, &mut issue)?;

                // These unwraps are safe because `insert_value_into_option_or_error` ensures the meta item
                // is a name/value pair string literal.
                issue_num = match issue.unwrap().as_str() {
                    "none" => None,
                    issue => match issue.parse::<NonZero<u32>>() {
                        Ok(num) => Some(num),
                        Err(err) => {
                            cx.dcx().emit_err(
                                session_diagnostics::InvalidIssueString {
                                    span: param_span,
                                    cause: session_diagnostics::InvalidIssueStringCause::from_int_error_kind(
                                        param_span,
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
                if !args.no_args() {
                    cx.dcx().emit_err(session_diagnostics::SoftNoArgs { span: param_span });
                }
                is_soft = true;
            }
            sym::implied_by => insert_value_into_option_or_error(cx, &param, &mut implied_by)?,
            _ => {
                cx.dcx().emit_err(session_diagnostics::UnknownMetaItem {
                    span: param_span,
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
            Err(cx.dcx().emit_err(session_diagnostics::NonIdentFeature { span: cx.attr_span }))
        }
        None => Err(cx.dcx().emit_err(session_diagnostics::MissingFeature { span: cx.attr_span })),
    };

    let issue = issue
        .ok_or_else(|| cx.dcx().emit_err(session_diagnostics::MissingIssue { span: cx.attr_span }));

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
