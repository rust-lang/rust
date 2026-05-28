use rustc_feature::AttributeStability;
use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::NoArgsAttributeParser;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::Allow;
use crate::unstable;

pub(crate) struct PinV2Parser;

impl NoArgsAttributeParser for PinV2Parser {
    const PATH: &[Symbol] = &[sym::pin_v2];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Union),
    ]);
    const STABILITY: AttributeStability = unstable!(pin_ergonomics);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::PinV2;
}
