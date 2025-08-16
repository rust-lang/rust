use super::prelude::*;

pub(crate) struct MayDangleParser;
impl<S: Stage> NoArgsAttributeParser<S> for MayDangleParser {
    const PATH: &[Symbol] = &[sym::may_dangle];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); //FIXME Still checked fully in `check_attr.rs`
    const CREATE: fn(span: Span) -> AttributeKind = AttributeKind::MayDangle;
}
