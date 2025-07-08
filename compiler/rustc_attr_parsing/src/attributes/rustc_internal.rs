use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage, parse_single_integer};
use crate::parser::ArgParser;

pub(crate) struct RustcLayoutScalarValidRangeStart;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeStart {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_start];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(List: "start");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeStart(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcLayoutScalarValidRangeEnd;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeEnd {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_end];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(List: "end");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeEnd(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcObjectLifetimeDefaultParser;

impl<S: Stage> SingleAttributeParser<S> for RustcObjectLifetimeDefaultParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_object_lifetime_default];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
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

pub(crate) struct RustcScalableVectorParser;

impl<S: Stage> SingleAttributeParser<S> for RustcScalableVectorParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_scalable_vector];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(Word, List: "count");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if args.no_args().is_ok() {
            return Some(AttributeKind::RustcScalableVector {
                element_count: None,
                span: cx.attr_span,
            });
        }

        parse_single_integer(cx, args).map(|n| AttributeKind::RustcScalableVector {
            element_count: Some(n),
            span: cx.attr_span,
        })
    }
}
