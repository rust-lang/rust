use super::prelude::*;
pub(crate) struct NonExhaustiveParser;

impl<S: Stage> NoArgsAttributeParser<S> for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const GATED: AttributeGate = Ungated;
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
