//! Attribute parsing for the `#[splat]` function argument overloading attribute.
//! This attribute modifies typecheck to support overload resolution, then modifies codegen for performance.

use super::prelude::*;

pub(crate) struct SplatParser;

impl<S: Stage> NoArgsAttributeParser<S> for SplatParser {
    const PATH: &[Symbol] = &[sym::splat];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Param),
        // FIXME(splat): only allow MacroCall if the macro creates an argument
        Allow(Target::MacroCall),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Splat;
}
