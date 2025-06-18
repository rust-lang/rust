use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;

pub(crate) struct AsPtrParser;
impl<S: Stage> NoArgsAttributeParser<S> for AsPtrParser {
    const PATH: &[Symbol] = &[sym::rustc_as_ptr];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    fn create(span: Span) -> AttributeKind {
        AttributeKind::AsPtr(span)
    }
}

pub(crate) struct PubTransparentParser;
impl<S: Stage> NoArgsAttributeParser<S> for PubTransparentParser {
    const PATH: &[Symbol] = &[sym::rustc_pub_transparent];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    fn create(span: Span) -> AttributeKind {
        AttributeKind::PubTransparent(span)
    }
}
