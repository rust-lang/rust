use rustc_feature::AttributeStability;
use rustc_hir::target::GenericParamKind;

use super::prelude::*;

pub(crate) struct MayDangleParser;
impl NoArgsAttributeParser for MayDangleParser {
    const PATH: &[Symbol] = &[sym::may_dangle];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::GenericParam { kind: GenericParamKind::Type, has_default: false }),
        Allow(Target::GenericParam { kind: GenericParamKind::Type, has_default: true }),
        Allow(Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: false }),
        Allow(Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: true }),
    ]);
    const STABILITY: AttributeStability = unstable!(dropck_eyepatch);
    const CREATE: fn(span: Span) -> AttributeKind = AttributeKind::MayDangle;
}
