//! Attributes that can be found in function body.

use super::prelude::*;

pub(crate) struct CoroutineParser;

impl NoArgsAttributeParser for CoroutineParser {
    const PATH: &[Symbol] = &[sym::coroutine];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Closure)]);
    const CREATE: fn(rustc_span::Span) -> AttributeKind = |_| AttributeKind::Coroutine;
}

pub(crate) struct FusedParser;

impl NoArgsAttributeParser for FusedParser {
    const PATH: &[Symbol] = &[sym::fused];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Closure),
        Allow(Target::Expression),
    ]);
    const CREATE: fn(rustc_span::Span) -> AttributeKind = AttributeKind::Fused;
}
