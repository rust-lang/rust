use super::prelude::*;

pub(crate) struct TestTraceParser;

impl<S: Stage> NoArgsAttributeParser<S> for TestTraceParser {
    const PATH: &[Symbol] = &[sym::test_trace];

    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);

    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::TestTrace;
}
