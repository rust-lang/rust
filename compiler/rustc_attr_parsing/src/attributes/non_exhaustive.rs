use rustc_ast::VariantData;
use rustc_attr_ir::target::AstTarget;
use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::diagnostics::NonExhaustiveWithDefaultFieldValues;

pub(crate) struct NonExhaustiveParser;

impl NoArgsAttributeParser for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];

    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Variant),
        Warn(Target::MacroCall), // #158816: 600+ regressions from crossterm 0.18..=0.21
    ]);
    const STABILITY: AttributeStability = AttributeStability::Stable;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NonExhaustive;

    fn finalize_check(cx: &mut FinalizeCheckContext<'_, '_>, attr_span: Span) {
        if let AstTarget::Struct(_, _, data) | AstTarget::Variant(rustc_ast::Variant { data, .. }) =
            cx.ast_target
            && let VariantData::Struct { fields, .. } = data
            && fields.iter().any(|f| f.default_value().is_some())
        {
            cx.emit_err(NonExhaustiveWithDefaultFieldValues {
                attr_span,
                defn_span: cx.target_span,
            });
        }
    }
}
