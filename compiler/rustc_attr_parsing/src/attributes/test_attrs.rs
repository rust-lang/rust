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

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::Ignore {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => {
                    let Some(str_value) = name_value.value_as_str() else {
                        let suggestions = <Self as SingleAttributeParser<S>>::TEMPLATE
                            .suggestions(cx.attr_style, "ignore");
                        let span = cx.attr_span;
                        cx.emit_lint(
                            AttributeLintKind::IllFormedAttributeInput { suggestions },
                            span,
                        );
                        return None;
                    };
                    Some(str_value)
                }
                ArgParser::List(_) => {
                    let suggestions = <Self as SingleAttributeParser<S>>::TEMPLATE
                        .suggestions(cx.attr_style, "ignore");
                    let span = cx.attr_span;
                    cx.emit_lint(AttributeLintKind::IllFormedAttributeInput { suggestions }, span);
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

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
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
