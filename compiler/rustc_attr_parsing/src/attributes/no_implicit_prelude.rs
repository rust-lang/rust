use rustc_feature::AttributeStability;

use super::prelude::*;

pub(crate) struct NoImplicitPreludeParser;

impl NoArgsAttributeParser for NoImplicitPreludeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_implicit_prelude];

    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowList(&[Allow(Target::Mod), Allow(Target::Crate)]);
    const STABILITY: AttributeStability = AttributeStability::Stable;
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NoImplicitPrelude;
}
