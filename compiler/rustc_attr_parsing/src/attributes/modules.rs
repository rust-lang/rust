use super::prelude::*;

pub(crate) struct TransparentParser;
impl<S: Stage> NoArgsAttributeParser<S> for TransparentParser {
    const PATH: &[Symbol] = &[sym::transparent];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Mod)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Transparent;
}
