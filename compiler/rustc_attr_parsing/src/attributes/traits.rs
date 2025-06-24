use core::mem;

use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct SkipDuringMethodDispatchParser;

impl<S: Stage> SingleAttributeParser<S> for SkipDuringMethodDispatchParser {
    const PATH: &[Symbol] = &[sym::rustc_skip_during_method_dispatch];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    const TEMPLATE: AttributeTemplate = template!(List: "array, boxed_slice");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let mut array = false;
        let mut boxed_slice = false;
        let Some(args) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };
        if args.is_empty() {
            cx.expected_at_least_one_argument(args.span);
            return None;
        }
        for arg in args.mixed() {
            let Some(arg) = arg.meta_item() else {
                cx.unexpected_literal(arg.span());
                continue;
            };
            if let Err(span) = arg.args().no_args() {
                cx.expected_no_args(span);
            }
            let path = arg.path();
            let (key, skip): (Symbol, &mut bool) = match path.word_sym() {
                Some(key @ sym::array) => (key, &mut array),
                Some(key @ sym::boxed_slice) => (key, &mut boxed_slice),
                _ => {
                    cx.expected_specific_argument(path.span(), vec!["array", "boxed_slice"]);
                    continue;
                }
            };
            if mem::replace(skip, true) {
                cx.duplicate_key(arg.span(), key);
            }
        }
        Some(AttributeKind::SkipDuringMethodDispatch { array, boxed_slice, span: cx.attr_span })
    }
}
