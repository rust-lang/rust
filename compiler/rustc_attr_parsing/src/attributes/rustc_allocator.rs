use super::prelude::*;

pub(crate) struct RustcAllocatorParser;

impl NoArgsAttributeParser for RustcAllocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocator;
}

pub(crate) struct RustcAllocatorZeroedParser;

impl NoArgsAttributeParser for RustcAllocatorZeroedParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator_zeroed];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocatorZeroed;
}

pub(crate) struct RustcAllocatorZeroedVariantParser;

impl SingleAttributeParser for RustcAllocatorZeroedVariantParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator_zeroed_variant];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "function");
    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(name) = args.name_value().and_then(NameValueParser::value_as_str) else {
            let attr_span = cx.attr_span;
            cx.adcx().expected_name_value(attr_span, None);
            return None;
        };

        Some(AttributeKind::RustcAllocatorZeroedVariant { name })
    }
}

pub(crate) struct RustcDeallocatorParser;

impl NoArgsAttributeParser for RustcDeallocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_deallocator];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDeallocator;
}

pub(crate) struct RustcReallocatorParser;

impl NoArgsAttributeParser for RustcReallocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_reallocator];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcReallocator;
}
