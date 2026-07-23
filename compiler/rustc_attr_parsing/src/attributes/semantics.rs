use rustc_ast::Safety;
use rustc_feature::AttributeStability;
use rustc_hir::target::GenericParamKind;
use rustc_span::edition::Edition;

use super::prelude::*;
use crate::AttributeSafety;

pub(crate) struct MayDangleParser;
impl SingleAttributeParser for MayDangleParser {
    const PATH: &[Symbol] = &[sym::may_dangle];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::GenericParam { kind: GenericParamKind::Type, has_default: false }),
        Allow(Target::GenericParam { kind: GenericParamKind::Type, has_default: true }),
        Allow(Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: false }),
        Allow(Target::GenericParam { kind: GenericParamKind::Lifetime, has_default: true }),
    ]);
    const STABILITY: AttributeStability = unstable!(dropck_eyepatch);
    const TEMPLATE: AttributeTemplate = template!(Word);
    const SAFETY: AttributeSafety = AttributeSafety::Unsafe {
        note: "the `may_dangle` attribute enforces invariants that the compiler can't check. \
            Review its documentation and make sure this implementation \
            upholds those invariants before adding the `unsafe` keyword",
        unsafe_since: Some(Edition::EditionFuture), // FIXME(may_dangle migration) set to None
    };

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        cx.expect_no_args(args)?;

        Some(AttributeKind::MayDangle {
            unsafe_used: matches!(cx.attr_safety, Safety::Unsafe(_)),
            span: cx.attr_span,
            inner_span: cx.inner_span,
        })
    }
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
