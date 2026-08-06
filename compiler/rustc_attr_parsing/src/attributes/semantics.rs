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

pub(crate) struct ComptimeParser;
impl NoArgsAttributeParser for ComptimeParser {
    const PATH: &[Symbol] = &[sym::rustc_comptime];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Fn),
        Allow(Target::Impl { of_trait: false }),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcComptime;
}
