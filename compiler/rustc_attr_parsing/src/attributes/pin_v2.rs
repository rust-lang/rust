use super::prelude::*;

pub(crate) struct PinV2Parser;

impl<S: Stage> NoArgsAttributeParser<S> for PinV2Parser {
    const PATH: &[Symbol] = &[sym::pin_v2];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const GATED: rustc_feature::AttributeGate = gated!(pin_ergonomics, experimental!(pin_v2));
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::PinV2;
}
