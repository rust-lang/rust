use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::{Stage};
use crate::target_checking::{ALL_TARGETS, AllowedTargets};


pub(crate) struct RadProtectedParser;
impl<S: Stage> NoArgsAttributeParser<S> for RadProtectedParser {
    const PATH: &[Symbol] = &[sym::rad_protected];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    const CREATE: fn(rustc_span::Span) -> AttributeKind = |span| AttributeKind::RadProtected(span);
}
