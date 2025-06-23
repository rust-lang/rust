use rustc_attr_data_structures::{AttributeKind, OptimizeAttr};
use rustc_feature::{AttributeTemplate, template};
use rustc_span::sym;

use super::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct OptimizeParser;

impl<S: Stage> SingleAttributeParser<S> for OptimizeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::optimize];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(List: "size|speed|none");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };

        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };

        let res = match single.meta_item().and_then(|i| i.path().word().map(|i| i.name)) {
            Some(sym::size) => OptimizeAttr::Size,
            Some(sym::speed) => OptimizeAttr::Speed,
            Some(sym::none) => OptimizeAttr::DoNotOptimize,
            _ => {
                cx.expected_specific_argument(single.span(), vec!["size", "speed", "none"]);
                OptimizeAttr::Default
            }
        };

        Some(AttributeKind::Optimize(res, cx.attr_span))
    }
}

pub(crate) struct ColdParser;

impl<S: Stage> SingleAttributeParser<S> for ColdParser {
    const PATH: &[rustc_span::Symbol] = &[sym::cold];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if !args.no_args() {
            cx.expected_no_args(args.span().unwrap_or(cx.attr_span));
            return None;
        };

        Some(AttributeKind::Cold(cx.attr_span))
    }
}

pub(crate) struct NoMangleParser;

impl<S: Stage> SingleAttributeParser<S> for NoMangleParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_mangle];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if !args.no_args() {
            cx.expected_no_args(args.span().unwrap_or(cx.attr_span));
            return None;
        };

        Some(AttributeKind::NoMangle(cx.attr_span))
    }
}
