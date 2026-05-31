use rustc_feature::{AttributeStability, AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{OnDuplicate, SingleAttributeParser};
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};
use crate::unstable;

pub(crate) struct RustcDummyParser;
impl SingleAttributeParser for RustcDummyParser {
    const PATH: &[Symbol] = &[sym::rustc_dummy];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(Word); // Anything, really
    const STABILITY: AttributeStability =
        unstable!(rustc_attrs, "the `#[rustc_dummy]` attribute is used for rustc unit tests");

    fn convert(_: &mut AcceptContext<'_, '_>, _: &ArgParser) -> Option<AttributeKind> {
        Some(AttributeKind::RustcDummy)
    }
}
