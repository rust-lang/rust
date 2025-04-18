use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use super::{AcceptContext, SingleAttributeParser};
use crate::parser::ArgParser;

pub(crate) struct EiiMangleExternParser;

impl SingleAttributeParser for EiiMangleExternParser {
    const PATH: &'static [Symbol] = &[sym::eii_mangle_extern];

    fn on_duplicate(_cx: &AcceptContext<'_>, _first_span: Span) {}
    fn convert(_cx: &AcceptContext<'_>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        assert!(args.no_args());
        Some(AttributeKind::EiiMangleExtern)
    }
}
