use rustc_ast::LitKind;
use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct RustcLayoutScalarValidRangeStart;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeStart {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_start];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(List: "start");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        parse_rustc_layout_scalar_valid_range(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeStart(n, cx.attr_span))
    }
}

pub(crate) struct RustcLayoutScalarValidRangeEnd;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeEnd {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_end];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(List: "end");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        parse_rustc_layout_scalar_valid_range(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeEnd(n, cx.attr_span))
    }
}

fn parse_rustc_layout_scalar_valid_range<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser<'_>,
) -> Option<Box<u128>> {
    let Some(list) = args.list() else {
        cx.expected_list(cx.attr_span);
        return None;
    };
    let Some(single) = list.single() else {
        cx.expected_single_argument(list.span);
        return None;
    };
    let Some(lit) = single.lit() else {
        cx.expected_integer_literal(single.span());
        return None;
    };
    let LitKind::Int(num, _ty) = lit.kind else {
        cx.expected_integer_literal(single.span());
        return None;
    };
    Some(Box::new(num.0))
}

pub(crate) struct RustcObjectLifetimeDefaultParser;

impl<S: Stage> SingleAttributeParser<S> for RustcObjectLifetimeDefaultParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_object_lifetime_default];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }

        Some(AttributeKind::RustcObjectLifetimeDefault)
    }
}
