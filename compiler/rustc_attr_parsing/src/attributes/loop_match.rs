use super::prelude::*;

pub(crate) struct LoopMatchParser;
impl<S: Stage> NoArgsAttributeParser<S> for LoopMatchParser {
    const PATH: &[Symbol] = &[sym::loop_match];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Expression)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::LoopMatch;
}

pub(crate) struct ConstContinueParser;
impl<S: Stage> NoArgsAttributeParser<S> for ConstContinueParser {
    const PATH: &[Symbol] = &[sym::const_continue];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Expression)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ConstContinue;
}
