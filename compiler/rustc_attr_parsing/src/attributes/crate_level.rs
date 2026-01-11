use rustc_hir::attrs::WindowsSubsystemKind;

use super::prelude::*;

pub(crate) struct CrateNameParser;

impl<S: Stage> SingleAttributeParser<S> for CrateNameParser {
    const PATH: &[Symbol] = &[sym::crate_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(n) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        let Some(name) = n.value_as_str() else {
            cx.expected_string_literal(n.value_span, Some(n.value_as_lit()));
            return None;
        };

        Some(AttributeKind::CrateName { name, name_span: n.value_span, attr_span: cx.attr_span })
    }
}

pub(crate) struct RecursionLimitParser;

impl<S: Stage> SingleAttributeParser<S> for RecursionLimitParser {
    const PATH: &[Symbol] = &[sym::recursion_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N", "https://doc.rust-lang.org/reference/attributes/limits.html#the-recursion_limit-attribute");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
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
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
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
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
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
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
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
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoCore;
}

pub(crate) struct NoStdParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoStdParser {
    const PATH: &[Symbol] = &[sym::no_std];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoStd;
}

pub(crate) struct RustcCoherenceIsCoreParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcCoherenceIsCoreParser {
    const PATH: &[Symbol] = &[sym::rustc_coherence_is_core];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcCoherenceIsCore;
}

pub(crate) struct WindowsSubsystemParser;

impl<S: Stage> SingleAttributeParser<S> for WindowsSubsystemParser {
    const PATH: &[Symbol] = &[sym::windows_subsystem];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: ["windows", "console"], "https://doc.rust-lang.org/reference/runtime.html#the-windows_subsystem-attribute");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(
                args.span().unwrap_or(cx.inner_span),
                Some(sym::windows_subsystem),
            );
            return None;
        };

        let kind = match nv.value_as_str() {
            Some(sym::console) => WindowsSubsystemKind::Console,
            Some(sym::windows) => WindowsSubsystemKind::Windows,
            Some(_) | None => {
                cx.expected_specific_argument_strings(nv.value_span, &[sym::console, sym::windows]);
                return None;
            }
        };

        Some(AttributeKind::WindowsSubsystem(kind, cx.attr_span))
    }
}
