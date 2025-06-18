use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct AsPtrParser;

impl<S: Stage> SingleAttributeParser<S> for AsPtrParser {
    const PATH: &[Symbol] = &[sym::rustc_as_ptr];

    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;

    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, _args: &ArgParser<'_>) -> Option<AttributeKind> {
        // FIXME: check that there's no args (this is currently checked elsewhere)
        Some(AttributeKind::AsPtr(cx.attr_span))
    }
}
