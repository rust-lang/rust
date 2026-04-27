use super::prelude::*;

pub(crate) struct LoopMatchParser;
impl NoArgsAttributeParser for LoopMatchParser {
    const PATH: &[Symbol] = &[sym::loop_match];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Expression)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::LoopMatch;
}

pub(crate) struct ConstContinueParser;
impl NoArgsAttributeParser for ConstContinueParser {
    const PATH: &[Symbol] = &[sym::const_continue];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Expression)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ConstContinue;
}
