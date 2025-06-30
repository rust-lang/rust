use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct LoopMatchParser;
impl<S: Stage> SingleAttributeParser<S> for LoopMatchParser {
    const PATH: &[Symbol] = &[sym::loop_match];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, _args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::LoopMatch(cx.attr_span))
    }
}

pub(crate) struct ConstContinueParser;
impl<S: Stage> SingleAttributeParser<S> for ConstContinueParser {
    const PATH: &[Symbol] = &[sym::const_continue];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, _args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::ConstContinue(cx.attr_span))
    }
}
