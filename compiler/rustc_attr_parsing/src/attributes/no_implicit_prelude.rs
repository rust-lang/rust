use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;

pub(crate) struct NoImplicitPreludeParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoImplicitPreludeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_implicit_prelude];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoImplicitPrelude;
}
