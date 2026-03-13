use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

pub(crate) struct DoNotRecommendParser;
impl<S: Stage> SingleAttributeParser<S> for DoNotRecommendParser {
    const PATH: &[Symbol] = &[sym::diagnostic, sym::do_not_recommend];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); // Checked in check_attr.
    const TEMPLATE: AttributeTemplate = template!(Word /*doesn't matter */);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let attr_span = cx.attr_span;
        if !matches!(args, ArgParser::NoArgs) {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::DoNotRecommendDoesNotExpectArgs,
                attr_span,
            );
        }
        Some(AttributeKind::DoNotRecommend { attr_span })
    }
}
