use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use super::prelude::*;

pub(crate) struct RustcDummyParser;
impl<S: Stage> SingleAttributeParser<S> for RustcDummyParser {
    const PATH: &[Symbol] = &[sym::rustc_dummy];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const GATED: AttributeGate = gated_rustc_attr!(TEST, rustc_dummy);
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(Word); // Anything, really

    fn convert(_: &mut AcceptContext<'_, '_, S>, _: &ArgParser) -> Option<AttributeKind> {
        Some(AttributeKind::RustcDummy)
    }
}
