use super::prelude::*;

pub(crate) struct RustcAllocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcAllocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocator;
}

pub(crate) struct RustcAllocatorZeroedParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcAllocatorZeroedParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator_zeroed];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocatorZeroed;
}

pub(crate) struct RustcAllocatorZeroedVariantParser;

impl<S: Stage> SingleAttributeParser<S> for RustcAllocatorZeroedVariantParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator_zeroed_variant];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "function");
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(name) = args.name_value().and_then(NameValueParser::value_as_str) else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::RustcAllocatorZeroedVariant { name })
    }
}

pub(crate) struct RustcDeallocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDeallocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_deallocator];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDeallocator;
}

pub(crate) struct RustcReallocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcReallocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_reallocator];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcReallocator;
}
