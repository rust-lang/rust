use rustc_attr_data_structures::AttributeKind;
use rustc_attr_data_structures::lints::AttributeLintKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct IgnoreParser;

impl<S: Stage> SingleAttributeParser<S> for IgnoreParser {
    const PATH: &[Symbol] = &[sym::ignore];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word, NameValueStr: "reason");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::Ignore {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => {
                    let Some(str_value) = name_value.value_as_str() else {
                        let suggestions = <Self as SingleAttributeParser<S>>::TEMPLATE
                            .suggestions(false, "ignore");
                        let span = cx.attr_span;
                        cx.emit_lint(
                            AttributeLintKind::IllFormedAttributeInput { suggestions },
                            span,
                        );
                        return None;
                    };
                    Some(str_value)
                }
                ArgParser::List(_) => {
                    let suggestions =
                        <Self as SingleAttributeParser<S>>::TEMPLATE.suggestions(false, "ignore");
                    let span = cx.attr_span;
                    cx.emit_lint(AttributeLintKind::IllFormedAttributeInput { suggestions }, span);
                    return None;
                }
            },
        })
    }
}
