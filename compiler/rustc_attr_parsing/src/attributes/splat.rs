//! Attribute parsing for the `#[splat]` function argument overloading attribute.
//! This attribute modifies typecheck to support overload resolution, then modifies codegen for performance.

use super::prelude::*;

pub(crate) struct SplatParser;

impl NoArgsAttributeParser for SplatParser {
    const PATH: &[Symbol] = &[sym::splat];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        // FIXME(splat): do we want to allow MacroCall if the macro creates an argument
        Allow(Target::Param),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Splat;
}
