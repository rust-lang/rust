use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::{Stage};
use crate::target_checking::{ALL_TARGETS, AllowedTargets};


pub(crate) struct RadProtectedParser;
impl<S: Stage> NoArgsAttributeParser<S> for RadProtectedParser {
    const PATH: &[Symbol] = &[sym::rad_protected_mir];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    const CREATE: fn(rustc_span::Span) -> AttributeKind = |span| AttributeKind::RadProtected(span);
}

/// Internal marker for `#[rad_protected(shadow_only)]`.
///
/// Unlike [`RadProtectedParser`], this one drives only the speculative-shadow MIR pass
/// (`rad_protected_function_shadow`): no process triplication and no checkpoint calls are
/// injected, so the dumped MIR shows the shadow transformation on its own.
pub(crate) struct RadProtectedShadowParser;
impl<S: Stage> NoArgsAttributeParser<S> for RadProtectedShadowParser {
    const PATH: &[Symbol] = &[sym::rad_protected_shadow];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    const CREATE: fn(rustc_span::Span) -> AttributeKind =
        |span| AttributeKind::RadProtectedShadow(span);
}
