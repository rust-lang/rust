use super::prelude::*;

pub(crate) struct MayDangleParser;
impl NoArgsAttributeParser for MayDangleParser {
    const PATH: &[Symbol] = &[sym::may_dangle];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(CHECKED_LATER); //FIXME Still checked fully in `check_attr.rs`
    const CREATE: fn(span: Span) -> AttributeKind = AttributeKind::MayDangle;
}
