use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::{Symbol, sym};

use super::super::prelude::*;

pub(crate) struct DoNotRecommendParser;
impl<S: Stage> SingleAttributeParser<S> for DoNotRecommendParser {
    const PATH: &[Symbol] = &[sym::diagnostic, sym::do_not_recommend];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const GATED: rustc_feature::AttributeGate = Ungated;
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
