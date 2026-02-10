use rustc_hir::attrs::RustcAbiAttrKind;
use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;

use super::prelude::*;

pub(crate) struct IgnoreParser;

impl<S: Stage> SingleAttributeParser<S> for IgnoreParser {
    const PATH: &[Symbol] = &[sym::ignore];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Fn), Error(Target::WherePredicate)]);
    const TEMPLATE: AttributeTemplate = template!(
        Word, NameValueStr: "reason",
        "https://doc.rust-lang.org/reference/attributes/testing.html#the-ignore-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        Some(AttributeKind::Ignore {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => {
                    let Some(str_value) = name_value.value_as_str() else {
                        cx.warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                        return None;
                    };
                    Some(str_value)
                }
                ArgParser::List(_) => {
                    cx.warn_ill_formed_attribute_input(ILL_FORMED_ATTRIBUTE_INPUT);
                    return None;
                }
            },
        })
    }
}

pub(crate) struct ShouldPanicParser;

impl<S: Stage> SingleAttributeParser<S> for ShouldPanicParser {
    const PATH: &[Symbol] = &[sym::should_panic];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Fn), Error(Target::WherePredicate)]);
    const TEMPLATE: AttributeTemplate = template!(
        Word, List: &[r#"expected = "reason""#], NameValueStr: "reason",
        "https://doc.rust-lang.org/reference/attributes/testing.html#the-should_panic-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        Some(AttributeKind::ShouldPanic {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => {
                    let Some(str_value) = name_value.value_as_str() else {
                        cx.expected_string_literal(
                            name_value.value_span,
                            Some(name_value.value_as_lit()),
                        );
                        return None;
                    };
                    Some(str_value)
                }
                ArgParser::List(list) => {
                    let Some(single) = list.single() else {
                        cx.expected_single_argument(list.span);
                        return None;
                    };
                    let Some(single) = single.meta_item() else {
                        cx.expected_name_value(single.span(), Some(sym::expected));
                        return None;
                    };
                    if !single.path().word_is(sym::expected) {
                        cx.expected_specific_argument_strings(list.span, &[sym::expected]);
                        return None;
                    }
                    let Some(nv) = single.args().name_value() else {
                        cx.expected_name_value(single.span(), Some(sym::expected));
                        return None;
                    };
                    let Some(expected) = nv.value_as_str() else {
                        cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                        return None;
                    };
                    Some(expected)
                }
            },
        })
    }
}

pub(crate) struct RustcVarianceParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcVarianceParser {
    const PATH: &[Symbol] = &[sym::rustc_variance];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcVariance;
}

pub(crate) struct RustcVarianceOfOpaquesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcVarianceOfOpaquesParser {
    const PATH: &[Symbol] = &[sym::rustc_variance_of_opaques];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcVarianceOfOpaques;
}

pub(crate) struct ReexportTestHarnessMainParser;

impl<S: Stage> SingleAttributeParser<S> for ReexportTestHarnessMainParser {
    const PATH: &[Symbol] = &[sym::reexport_test_harness_main];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(
                args.span().unwrap_or(cx.inner_span),
                Some(sym::reexport_test_harness_main),
            );
            return None;
        };

        let Some(name) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::ReexportTestHarnessMain(name))
    }
}

pub(crate) struct RustcAbiParser;

impl<S: Stage> SingleAttributeParser<S> for RustcAbiParser {
    const PATH: &[Symbol] = &[sym::rustc_abi];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(OneOf: &[sym::debug, sym::assert_eq]);
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::TyAlias),
        Allow(Target::Fn),
        Allow(Target::ForeignFn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(args) = args.list() else {
            cx.expected_specific_argument_and_list(cx.attr_span, &[sym::assert_eq, sym::debug]);
            return None;
        };

        let Some(arg) = args.single() else {
            cx.expected_single_argument(cx.attr_span);
            return None;
        };

        let fail_incorrect_argument =
            |span| cx.expected_specific_argument(span, &[sym::assert_eq, sym::debug]);

        let Some(arg) = arg.meta_item() else {
            fail_incorrect_argument(args.span);
            return None;
        };

        let kind: RustcAbiAttrKind = match arg.path().word_sym() {
            Some(sym::assert_eq) => RustcAbiAttrKind::AssertEq,
            Some(sym::debug) => RustcAbiAttrKind::Debug,
            None | Some(_) => {
                fail_incorrect_argument(arg.span());
                return None;
            }
        };

        Some(AttributeKind::RustcAbi { attr_span: cx.attr_span, kind })
    }
}

pub(crate) struct RustcDelayedBugFromInsideQueryParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDelayedBugFromInsideQueryParser {
    const PATH: &[Symbol] = &[sym::rustc_delayed_bug_from_inside_query];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDelayedBugFromInsideQuery;
}

pub(crate) struct RustcEvaluateWhereClausesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcEvaluateWhereClausesParser {
    const PATH: &[Symbol] = &[sym::rustc_evaluate_where_clauses];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcEvaluateWhereClauses;
}

pub(crate) struct RustcOutlivesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcOutlivesParser {
    const PATH: &[Symbol] = &[sym::rustc_outlives];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcOutlives;
}

pub(crate) struct TestRunnerParser;

impl<S: Stage> SingleAttributeParser<S> for TestRunnerParser {
    const PATH: &[Symbol] = &[sym::test_runner];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["path"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };

        let Some(meta) = single.meta_item() else {
            cx.unexpected_literal(single.span());
            return None;
        };

        Some(AttributeKind::TestRunner(meta.path().0.clone()))
    }
}
