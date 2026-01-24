use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

pub(crate) struct DoNotRecommendParser;

impl<S: Stage> NoArgsAttributeParser<S> for DoNotRecommendParser {
    const PATH: &[Symbol] = &[sym::diagnostic, sym::do_not_recommend];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); // Checked in check_attr.

    const CREATE: fn(Span) -> AttributeKind =
        |attr_span| AttributeKind::DoNotRecommend { attr_span };
}
