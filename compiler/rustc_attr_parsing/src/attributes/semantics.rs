use rustc_attr_ir::target::GenericParamKind;
use rustc_feature::AttributeStability;

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
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Fn),
        Allow(Target::Impl { of_trait: false }),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcComptime;
}

pub(crate) struct AlwaysGcaParser;
impl NoArgsAttributeParser for AlwaysGcaParser {
    const PATH: &[Symbol] = &[sym::rustc_always_gca];
    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowList(&[Allow(Target::AssocConst(AssocCtxt::Trait))]);
    const STABILITY: AttributeStability = unstable!(min_generic_const_args);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::AlwaysGca;
}
