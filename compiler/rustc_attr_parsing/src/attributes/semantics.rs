use rustc_ast::ItemKind;
use rustc_attr_ir::lang_items::LangItem;
use rustc_attr_ir::target::{AstTarget, GenericParamKind};
use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::diagnostics::InvalidMayDangle;

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

    fn finalize_check(cx: &mut FinalizeCheckContext<'_, '_>, attr_span: Span) {
        if matches!(
            cx.target,
            Target::GenericParam { kind: GenericParamKind::Type | GenericParamKind::Lifetime, .. }
        ) && !matches!(
            cx.ast_target,
            AstTarget::GenericParam { owner: Some(item), .. }
                if matches!(
                    &item.kind,
                    ItemKind::Impl(impl_)
                        if impl_.of_trait.as_deref().is_some_and(|header| {
                            cx.is_lang_item(header.trait_ref.ref_id, LangItem::Drop)
                        })
                )
        ) {
            cx.emit_err(InvalidMayDangle { attr_span });
        }
    }
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
