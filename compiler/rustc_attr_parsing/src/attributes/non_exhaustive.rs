use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::diagnostics::NonExhaustiveWithDefaultFieldValues;

pub(crate) struct NonExhaustiveParser;

impl NoArgsAttributeParser for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Variant),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
        Warn(Target::MacroCall),
    ]);
    const STABILITY: AttributeStability = AttributeStability::Stable;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NonExhaustive;

    fn finalize_check(cx: &FinalizeCheckContext<'_, '_>, attr_span: Span) {
        if cx.target != Target::Struct {
            return;
        }
        let Some(item) = cx.target_item else {
            return;
        };
        let rustc_ast::ast::ItemKind::Struct(
            _,
            _,
            rustc_ast::ast::VariantData::Struct { fields, .. },
        ) = &item.kind
        else {
            return;
        };
        if !fields.is_empty() && fields.iter().any(|f| f.default_value().is_some()) {
            cx.emit_err(NonExhaustiveWithDefaultFieldValues {
                attr_span,
                defn_span: cx.target_span,
            });
        }
    }
}
