use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::diagnostics::NonExhaustiveWithDefaultFieldValues;

pub(crate) struct NonExhaustiveParser;

impl NoArgsAttributeParser for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct { has_default_field_values: false }),
        Allow(Target::Struct { has_default_field_values: true }),
        Allow(Target::Variant),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
        Warn(Target::MacroCall),
    ]);
    const STABILITY: AttributeStability = AttributeStability::Stable;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NonExhaustive;

    fn finalize_check(cx: &FinalizeCheckContext<'_, '_>, attr_span: Span) {
        if let Target::Struct { has_default_field_values: true } = cx.target {
            cx.emit_err(NonExhaustiveWithDefaultFieldValues {
                attr_span,
                defn_span: cx.target_span,
            });
        }
    }
}
