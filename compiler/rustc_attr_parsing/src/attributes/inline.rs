use rustc_span::sym;

use crate::attributes::SingleAttributeParser;

pub(crate) struct InlineParser;

impl SingleAttributeParser for InlineParser {
    const PATH: &'static [rustc_span::Symbol] = &[sym::inline];

    fn on_duplicate(cx: &super::AcceptContext<'_>, first_span: rustc_span::Span) {
        todo!()
    }

    fn convert(cx: &super::AcceptContext<'_>, args: &crate::parser::ArgParser<'_>) -> Option<rustc_attr_data_structures::AttributeKind> {
        todo!()
    }
}
