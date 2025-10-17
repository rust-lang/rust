use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

pub(crate) struct DummyParser;
impl<S: Stage> SingleAttributeParser<S> for DummyParser {
    const PATH: &[Symbol] = &[sym::rustc_dummy];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(Word); // Anything, really

    fn convert(_: &mut AcceptContext<'_, '_, S>, _: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::Dummy)
    }
}
