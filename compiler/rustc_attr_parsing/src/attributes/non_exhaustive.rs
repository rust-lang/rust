use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;

pub(crate) struct NonExhaustiveParser;

impl<S: Stage> NoArgsAttributeParser<S> for NonExhaustiveParser {
    const PATH: &[Symbol] = &[sym::non_exhaustive];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NonExhaustive;
}
