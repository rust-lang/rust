use super::prelude::*;

pub(crate) struct RustcAllocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcAllocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocator;
}

pub(crate) struct RustcAllocatorZeroedParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcAllocatorZeroedParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator_zeroed];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocatorZeroed;
}

pub(crate) struct RustcAllocatorZeroedVariantParser;

impl<S: Stage> SingleAttributeParser<S> for RustcAllocatorZeroedVariantParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator_zeroed_variant];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "function");
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let nv = cx.expect_name_value(args, cx.attr_span, None)?;
        let Some(name) = nv.value_as_str() else {
            cx.adcx().expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::RustcAllocatorZeroedVariant { name })
    }
}

pub(crate) struct RustcDeallocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDeallocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_deallocator];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDeallocator;
}

pub(crate) struct RustcReallocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcReallocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_reallocator];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcReallocator;
}
