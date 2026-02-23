use super::prelude::*;

pub(crate) struct NoLinkParser;
impl<S: Stage> NoArgsAttributeParser<S> for NoLinkParser {
    const PATH: &[Symbol] = &[sym::no_link];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::ExternCrate),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NoLink;
}
