//! Attribute parsing for the `#[splat]` function argument overloading attribute.
//! This attribute modifies typecheck to support overload resolution, then modifies codegen for performance.

use rustc_feature::AttributeStability;

use super::prelude::*;

pub(crate) struct SplatParser;

impl NoArgsAttributeParser for SplatParser {
    const PATH: &[Symbol] = &[sym::splat];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[Allow(Target::Param)]);
    const STABILITY: AttributeStability =
        unstable!(splat, "the `#[splat]` attribute is experimental");
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Splat;
}
