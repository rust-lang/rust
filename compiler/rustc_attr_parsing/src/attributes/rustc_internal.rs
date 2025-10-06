use super::prelude::*;
use super::util::parse_single_integer;

pub(crate) struct RustcLayoutScalarValidRangeStart;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeStart {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_start];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["start"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeStart(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcLayoutScalarValidRangeEnd;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeEnd {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_end];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["end"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeEnd(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcObjectLifetimeDefaultParser;

impl<S: Stage> SingleAttributeParser<S> for RustcObjectLifetimeDefaultParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_object_lifetime_default];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }

        Some(AttributeKind::RustcObjectLifetimeDefault)
    }
}

pub(crate) struct RustcSimdMonomorphizeLaneLimitParser;

impl<S: Stage> SingleAttributeParser<S> for RustcSimdMonomorphizeLaneLimitParser {
    const PATH: &[Symbol] = &[sym::rustc_simd_monomorphize_lane_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        Some(AttributeKind::RustcSimdMonomorphizeLaneLimit(cx.parse_limit_int(nv)?))
    }
}
