use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct DummyParser;
impl<S: Stage> SingleAttributeParser<S> for DummyParser {
    const PATH: &[Symbol] = &[sym::rustc_dummy];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Ignore;
    const TEMPLATE: AttributeTemplate = template!(Word); // Anything, really

    fn convert(_: &mut AcceptContext<'_, '_, S>, _: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::Dummy)
    }
}
