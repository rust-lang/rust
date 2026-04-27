use rustc_errors::Diagnostic;
use rustc_feature::{AttributeTemplate, template};
use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_session::lint::builtin::{
    MALFORMED_DIAGNOSTIC_ATTRIBUTES, MISPLACED_DIAGNOSTIC_ATTRIBUTES,
};
use rustc_span::{Symbol, sym};

use crate::attributes::{OnDuplicate, SingleAttributeParser};
use crate::context::AcceptContext;
use crate::errors::IncorrectDoNotRecommendLocation;
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

pub(crate) struct DoNotRecommendParser;
impl SingleAttributeParser for DoNotRecommendParser {
    const PATH: &[Symbol] = &[sym::diagnostic, sym::do_not_recommend];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    // "Allowed" on any target, noop on all but trait impls
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(Word /*doesn't matter */);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let attr_span = cx.attr_span;
        if !matches!(args, ArgParser::NoArgs) {
            cx.emit_dyn_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                |dcx, level| crate::errors::DoNotRecommendDoesNotExpectArgs.into_diag(dcx, level),
                attr_span,
            );
        }

        if !matches!(cx.target, Target::Impl { of_trait: true }) {
            let target_span = cx.target_span;
            cx.emit_dyn_lint(
                MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                move |dcx, level| {
                    IncorrectDoNotRecommendLocation { target_span }.into_diag(dcx, level)
                },
                attr_span,
            );
            return None;
        }

        Some(AttributeKind::DoNotRecommend { attr_span })
    }
}
