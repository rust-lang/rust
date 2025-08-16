use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::Allow;

pub(crate) struct NoImplicitPreludeParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoImplicitPreludeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_implicit_prelude];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Mod), Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoImplicitPrelude;
}
