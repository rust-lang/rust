use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct NonExhaustiveParser;

impl<S: Stage> SingleAttributeParser<S> for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;

    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }
        Some(AttributeKind::NonExhaustive(cx.attr_span))
    }
}
