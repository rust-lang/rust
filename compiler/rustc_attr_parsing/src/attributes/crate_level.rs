use super::prelude::*;
use crate::session_diagnostics::{ExpectedSingleWord, LimitInvalid};

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

pub(crate) struct FeatureParser;

impl<S: Stage> CombineAttributeParser<S> for FeatureParser {
    const PATH: &[Symbol] = &[sym::feature];
    type Item = Ident;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::Feature;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::CrateLevel;
    const TEMPLATE: AttributeTemplate = template!(List: &["feature1, feature2, ..."]);

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span);
            return Vec::new();
        };

        if list.is_empty() {
            cx.warn_empty_attribute(cx.attr_span);
        }

        let mut res = Vec::new();

        for elem in list.mixed() {
            let Some(elem) = elem.meta_item() else {
                cx.expected_identifier(elem.span());
                continue;
            };
            if let Err(arg_span) = elem.args().no_args() {
                cx.expected_no_args(arg_span);
                continue;
            }

            let path = elem.path();
            let Some(ident) = path.word() else {
                let first_segment = elem.path().segments().next().expect("at least one segment");
                cx.emit_err(ExpectedSingleWord {
                    description: "rust features",
                    span: path.span(),
                    first_segment_span: first_segment.span,
                    first_segment: first_segment.name,
                });
                continue;
            };

            res.push(ident);
        }

        res
    }
}

pub(crate) struct RegisterToolParser;

impl<S: Stage> CombineAttributeParser<S> for RegisterToolParser {
    const PATH: &[Symbol] = &[sym::register_tool];
    type Item = Ident;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::RegisterTool;

    // FIXME: recursion limit is allowed on all targets and ignored,
    //        even though it should only be valid on crates of course
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(List: &["tool1, tool2, ..."]);

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span);
            return Vec::new();
        };

        if list.is_empty() {
            cx.warn_empty_attribute(cx.attr_span);
        }

        let mut res = Vec::new();

        for elem in list.mixed() {
            let Some(elem) = elem.meta_item() else {
                cx.expected_identifier(elem.span());
                continue;
            };
            if let Err(arg_span) = elem.args().no_args() {
                cx.expected_no_args(arg_span);
                continue;
            }

            let path = elem.path();
            let Some(ident) = path.word() else {
                let first_segment = elem.path().segments().next().expect("at least one segment");
                cx.emit_err(ExpectedSingleWord {
                    description: "tools",
                    span: path.span(),
                    first_segment_span: first_segment.span,
                    first_segment: first_segment.name,
                });
                continue;
            };

            res.push(ident);
        }

        res
    }
}
