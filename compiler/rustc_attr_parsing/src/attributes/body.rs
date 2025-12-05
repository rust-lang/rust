//! Attributes that can be found in function body.

use super::prelude::*;

pub(crate) struct CoroutineParser;

impl<S: Stage> NoArgsAttributeParser<S> for CoroutineParser {
    const PATH: &[Symbol] = &[sym::coroutine];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Closure)]);
    const CREATE: fn(rustc_span::Span) -> AttributeKind = |span| AttributeKind::Coroutine(span);
}
