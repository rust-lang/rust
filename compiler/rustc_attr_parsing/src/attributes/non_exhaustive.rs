use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::{Allow, Warn};

pub(crate) struct NonExhaustiveParser;

impl<S: Stage> NoArgsAttributeParser<S> for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Variant),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
        Warn(Target::MacroCall),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NonExhaustive;
}
