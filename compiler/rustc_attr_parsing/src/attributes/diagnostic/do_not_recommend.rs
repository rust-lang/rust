use rustc_feature::AttributeStability;
use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_session::lint::builtin::MALFORMED_DIAGNOSTIC_ATTRIBUTES;
use rustc_span::{Symbol, sym};

use crate::attributes::prelude::Allow;
use crate::attributes::{OnDuplicate, SingleAttributeParser};
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::target_checking::AllowedTargets;
use crate::{AttributeTemplate, template};

pub(crate) struct DoNotRecommendParser;
impl SingleAttributeParser for DoNotRecommendParser {
    const PATH: &[Symbol] = &[sym::diagnostic, sym::do_not_recommend];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    // "Allowed" on any target, noop on all but trait impls
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Impl { of_trait: true })]);
    const TEMPLATE: AttributeTemplate = template!(Word /*doesn't matter */);
    const STABILITY: AttributeStability = AttributeStability::Stable;

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let attr_span = cx.attr_span;
        if !matches!(args, ArgParser::NoArgs) {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                crate::diagnostics::DoNotRecommendDoesNotExpectArgs,
                attr_span,
            );
        }

        Some(AttributeKind::DoNotRecommend)
    }
}
