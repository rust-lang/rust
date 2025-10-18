use super::prelude::*;

pub(crate) struct CrateNameParser;

impl<S: Stage> SingleAttributeParser<S> for CrateNameParser {
    const PATH: &[Symbol] = &[sym::crate_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ArgParser::NameValue(n) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        let Some(name) = n.value_as_str() else {
            cx.expected_string_literal(n.value_span, Some(n.value_as_lit()));
            return None;
        };

        Some(AttributeKind::CrateName {
            name,
            name_span: n.value_span,
            attr_span: cx.attr_span,
            style: cx.attr_style,
        })
    }
}

pub(crate) struct RecursionLimitParser;

impl<S: Stage> SingleAttributeParser<S> for RecursionLimitParser {
    const PATH: &[Symbol] = &[sym::recursion_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N", "https://doc.rust-lang.org/reference/attributes/limits.html#the-recursion_limit-attribute");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::RecursionLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct MoveSizeLimitParser;

impl<S: Stage> SingleAttributeParser<S> for MoveSizeLimitParser {
    const PATH: &[Symbol] = &[sym::move_size_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::MoveSizeLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct TypeLengthLimitParser;

impl<S: Stage> SingleAttributeParser<S> for TypeLengthLimitParser {
    const PATH: &[Symbol] = &[sym::type_length_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::TypeLengthLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct PatternComplexityLimitParser;

impl<S: Stage> SingleAttributeParser<S> for PatternComplexityLimitParser {
    const PATH: &[Symbol] = &[sym::pattern_complexity_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::PatternComplexityLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct NoCoreParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoCoreParser {
    const PATH: &[Symbol] = &[sym::no_core];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoCore;
}

pub(crate) struct NoStdParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoStdParser {
    const PATH: &[Symbol] = &[sym::no_std];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoStd;
}

pub(crate) struct RustcCoherenceIsCoreParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcCoherenceIsCoreParser {
    const PATH: &[Symbol] = &[sym::rustc_coherence_is_core];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcCoherenceIsCore;
}

pub(crate) struct UnstableRemovedParser;

impl<S: Stage> SingleAttributeParser<S> for UnstableRemovedParser {
    const PATH: &[Symbol] = &[sym::unstable_removed];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(
        List: &[
            r#"feature = "name""#,
            r#"since = "version""#,
            r#"issue = "number""#,
            r#"reason = "text" (optional)"#,
        ],
        "https://doc.rust-lang.org/nightly/unstable-book/"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let list = args.list()?;
        let mut feature = None;
        let mut since = None;
        let mut issue = None;
        let mut reason = None;

        for item in list.mixed() {
            let Some(mi) = item.meta_item() else {
                cx.expected_list(item.span());
                continue;
            };

            let Some(nv) = mi.args().name_value() else {
                cx.expected_list(item.span());
                continue;
            };

            let key = mi.path().word_sym();
            let Some(value) = nv.value_as_str() else {
                cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                continue;
            };

            match key {
                Some(sym::feature) => feature = Some(value),
                Some(sym::since) => since = Some(value),
                Some(sym::issue) => issue = Some(value),
                Some(sym::reason) => reason = Some(value),
                _ => {
                    cx.expected_list(item.span());
                }
            }
        }

        Some(AttributeKind::UnstableRemoved {
            feature: feature?,
            since: since?,
            issue: issue?,
            reason,
            attr_span: cx.attr_span,
        })
    }
}
