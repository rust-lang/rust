use super::prelude::*;

pub(crate) struct RustcAllocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcAllocatorParser {
    const PATH: &[Symbol] = &[sym::rustc_allocator];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcAllocator;
}