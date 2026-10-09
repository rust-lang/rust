use rustc_feature::AttributeStability;
use rustc_lint_defs::builtin::NO_LINK_ATTR_USED;

use super::prelude::*;
use crate::diagnostics::NoLinkAttrUsed;

pub(crate) struct NoLinkParser;
impl NoArgsAttributeParser for NoLinkParser {
    const PATH: &[Symbol] = &[sym::no_link];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::ExternCrate),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
    ]);
    const STABILITY: AttributeStability = AttributeStability::Stable;
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NoLink;

    fn finalize_check(cx: &mut FinalizeCheckContext<'_, '_>, attr_span: Span) {
        cx.emit_lint(NO_LINK_ATTR_USED, NoLinkAttrUsed, attr_span);
    }
}
