use std::num::NonZero;

use rustc_errors::ErrorGuaranteed;
use rustc_hir::{
    DefaultBodyStability, MethodKind, PartialConstStability, Stability, StabilityLevel,
    StableSince, Target, UnstableReason, VERSION_PLACEHOLDER,
};

use super::prelude::*;
use super::util::parse_version;
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

const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
    Allow(Target::Fn),
    Allow(Target::Struct),
    Allow(Target::Enum),
    Allow(Target::Union),
    Allow(Target::Method(MethodKind::Inherent)),
    Allow(Target::Method(MethodKind::Trait { body: false })),
    Allow(Target::Method(MethodKind::Trait { body: true })),
    Allow(Target::Method(MethodKind::TraitImpl)),
    Allow(Target::Impl { of_trait: false }),
    Allow(Target::Impl { of_trait: true }),
    Allow(Target::MacroDef),
    Allow(Target::Crate),
    Allow(Target::Mod),
    Allow(Target::Use), // FIXME I don't think this does anything?
    Allow(Target::Const),
    Allow(Target::AssocConst),
    Allow(Target::AssocTy),
    Allow(Target::Trait),
    Allow(Target::TraitAlias),
    Allow(Target::TyAlias),
    Allow(Target::Variant),
    Allow(Target::Field),
    Allow(Target::Param),
    Allow(Target::Static),
    Allow(Target::ForeignFn),
    Allow(Target::ForeignStatic),
    Allow(Target::ExternCrate),
]);

#[derive(Default)]
pub(crate) struct StabilityParser {
    allowed_through_unstable_modules: Option<Symbol>,
    stability: Option<(Stability, Span)>,
}

impl StabilityParser {
    /// Checks, and emits an error when a stability (or unstability) was already set, which would be a duplicate.
    fn check_duplicate<S: Stage>(&self, cx: &AcceptContext<'_, '_, S>) -> bool {
        if let Some((_, _)) = self.stability {
            cx.emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            true
        } else {
            false
        }
    }
}

impl<S: Stage> AttributeParser<S> for StabilityParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (
            &[sym::stable],
            template!(List: &[r#"feature = "name", since = "version""#]),
            |this, cx, args| {
                reject_outside_std!(cx);
                if !this.check_duplicate(cx)
                    && let Some((feature, level)) = parse_stability(cx, args)
                {
                    this.stability = Some((Stability { level, feature }, cx.attr_span));
                }
            },
        ),
        (
            &[sym::unstable],
            template!(List: &[r#"feature = "name", reason = "...", issue = "N""#]),
            |this, cx, args| {
                reject_outside_std!(cx);
                if !this.check_duplicate(cx)
                    && let Some((feature, level)) = parse_unstability(cx, args)
                {
                    this.stability = Some((Stability { level, feature }, cx.attr_span));
                }
            },
        ),
        (
            &[sym::rustc_allowed_through_unstable_modules],
            template!(NameValueStr: "deprecation message"),
            |this, cx, args| {
                reject_outside_std!(cx);
                let Some(nv) = args.name_value() else {
                    cx.expected_name_value(cx.attr_span, None);
                    return;
                };
                let Some(value_str) = nv.value_as_str() else {
                    cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                    return;
                };
                this.allowed_through_unstable_modules = Some(value_str);
            },
        ),
    ];
    const ALLOWED_TARGETS: AllowedTargets = ALLOWED_TARGETS;

    fn finalize(mut self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
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

        if let Some((Stability { level: StabilityLevel::Stable { .. }, .. }, _)) = self.stability {
            for other_attr in cx.all_attrs {
                if other_attr.word_is(sym::unstable_feature_bound) {
                    cx.emit_err(session_diagnostics::UnstableFeatureBoundIncompatibleStability {
                        span: cx.target_span,
                    });
                }
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

impl<S: Stage> AttributeParser<S> for BodyStabilityParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::rustc_default_body_unstable],
        template!(List: &[r#"feature = "name", reason = "...", issue = "N""#]),
        |this, cx, args| {
            reject_outside_std!(cx);
            if this.stability.is_some() {
                cx.dcx()
                    .emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            } else if let Some((feature, level)) = parse_unstability(cx, args) {
                this.stability = Some((DefaultBodyStability { level, feature }, cx.attr_span));
            }
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets = ALLOWED_TARGETS;

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        let (stability, span) = self.stability?;

        Some(AttributeKind::BodyStability { stability, span })
    }
}

pub(crate) struct ConstStabilityIndirectParser;
impl<S: Stage> NoArgsAttributeParser<S> for ConstStabilityIndirectParser {
    const PATH: &[Symbol] = &[sym::rustc_const_stable_indirect];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::ConstStabilityIndirect;
}

#[derive(Default)]
pub(crate) struct ConstStabilityParser {
    promotable: bool,
    stability: Option<(PartialConstStability, Span)>,
}

impl ConstStabilityParser {
    /// Checks, and emits an error when a stability (or unstability) was already set, which would be a duplicate.
    fn check_duplicate<S: Stage>(&self, cx: &AcceptContext<'_, '_, S>) -> bool {
        if let Some((_, _)) = self.stability {
            cx.emit_err(session_diagnostics::MultipleStabilityLevels { span: cx.attr_span });
            true
        } else {
            false
        }
    }
}

impl<S: Stage> AttributeParser<S> for ConstStabilityParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (
            &[sym::rustc_const_stable],
            template!(List: &[r#"feature = "name""#]),
            |this, cx, args| {
                reject_outside_std!(cx);

                if !this.check_duplicate(cx)
                    && let Some((feature, level)) = parse_stability(cx, args)
                {
                    this.stability = Some((
                        PartialConstStability { level, feature, promotable: false },
                        cx.attr_span,
                    ));
                }
            },
        ),
        (
            &[sym::rustc_const_unstable],
            template!(List: &[r#"feature = "name""#]),
            |this, cx, args| {
                reject_outside_std!(cx);
                if !this.check_duplicate(cx)
                    && let Some((feature, level)) = parse_unstability(cx, args)
                {
                    this.stability = Some((
                        PartialConstStability { level, feature, promotable: false },
                        cx.attr_span,
                    ));
                }
            },
        ),
        (&[sym::rustc_promotable], template!(Word), |this, cx, _| {
            reject_outside_std!(cx);
            this.promotable = true;
        }),
    ];
    const ALLOWED_TARGETS: AllowedTargets = ALLOWED_TARGETS;

    fn finalize(mut self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
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
fn insert_value_into_option_or_error<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    param: &MetaItemParser<'_>,
    item: &mut Option<Symbol>,
    name: Ident,
) -> Option<()> {
    if item.is_some() {
        cx.duplicate_key(name.span, name.name);
        None
    } else if let Some(v) = param.args().name_value()
        && let Some(s) = v.value_as_str()
    {
        *item = Some(s);
        Some(())
    } else {
        cx.expected_name_value(param.span(), Some(name.name));
        None
    }
}

/// Read the content of a `stable`/`rustc_const_stable` attribute, and return the feature name and
/// its stability information.
pub(crate) fn parse_stability<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    args: &ArgParser<'_>,
) -> Option<(Symbol, StabilityLevel)> {
    let mut feature = None;
    let mut since = None;

    let ArgParser::List(list) = args else {
        cx.expected_list(cx.attr_span);
        return None;
    };

    for param in list.mixed() {
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

        let word = param.path().word();
        match word.map(|i| i.name) {
            Some(sym::feature) => {
                insert_value_into_option_or_error(cx, &param, &mut feature, word.unwrap())?
            }
            Some(sym::since) => {
                insert_value_into_option_or_error(cx, &param, &mut since, word.unwrap())?
            }
            _ => {
                cx.emit_err(session_diagnostics::UnknownMetaItem {
                    span: param_span,
                    item: param.path().to_string(),
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
            let err = cx.emit_err(session_diagnostics::InvalidSince { span: cx.attr_span });
            StableSince::Err(err)
        }
    } else {
        let err = cx.emit_err(session_diagnostics::MissingSince { span: cx.attr_span });
        StableSince::Err(err)
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
pub(crate) fn parse_unstability<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    args: &ArgParser<'_>,
) -> Option<(Symbol, StabilityLevel)> {
    let mut feature = None;
    let mut reason = None;
    let mut issue = None;
    let mut issue_num = None;
    let mut is_soft = false;
    let mut implied_by = None;
    let mut old_name = None;

    let ArgParser::List(list) = args else {
        cx.expected_list(cx.attr_span);
        return None;
    };

    for param in list.mixed() {
        let Some(param) = param.meta_item() else {
            cx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: param.span(),
                reason: UnsupportedLiteralReason::Generic,
                is_bytestr: false,
                start_point_span: cx.sess().source_map().start_point(param.span()),
            });
            return None;
        };

        let word = param.path().word();
        match word.map(|i| i.name) {
            Some(sym::feature) => {
                insert_value_into_option_or_error(cx, &param, &mut feature, word.unwrap())?
            }
            Some(sym::reason) => {
                insert_value_into_option_or_error(cx, &param, &mut reason, word.unwrap())?
            }
            Some(sym::issue) => {
                insert_value_into_option_or_error(cx, &param, &mut issue, word.unwrap())?;

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
                if let Err(span) = args.no_args() {
                    cx.emit_err(session_diagnostics::SoftNoArgs { span });
                }
                is_soft = true;
            }
            Some(sym::implied_by) => {
                insert_value_into_option_or_error(cx, &param, &mut implied_by, word.unwrap())?
            }
            Some(sym::old_name) => {
                insert_value_into_option_or_error(cx, &param, &mut old_name, word.unwrap())?
            }
            _ => {
                cx.emit_err(session_diagnostics::UnknownMetaItem {
                    span: param.span(),
                    item: param.path().to_string(),
                    expected: &["feature", "reason", "issue", "soft", "implied_by", "old_name"],
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
                old_name,
            };
            Some((feature, level))
        }
        (Err(ErrorGuaranteed { .. }), _) | (_, Err(ErrorGuaranteed { .. })) => None,
    }
}
