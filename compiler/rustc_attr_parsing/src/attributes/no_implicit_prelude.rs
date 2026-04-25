use super::prelude::*;

pub(crate) struct NoImplicitPreludeParser;

impl NoArgsAttributeParser for NoImplicitPreludeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_implicit_prelude];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Mod), Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoImplicitPrelude;
}
