use rustc_errors::DiagArgValue;
use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics;

pub(crate) struct MustUseParser;

impl<S: Stage> SingleAttributeParser<S> for MustUseParser {
    const PATH: &[Symbol] = &[sym::must_use];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(
        Word, NameValueStr: "reason",
        "https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::MustUse {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => {
                    let Some(value_str) = name_value.value_as_str() else {
                        cx.expected_string_literal(
                            name_value.value_span,
                            Some(&name_value.value_as_lit()),
                        );
                        return None;
                    };
                    Some(value_str)
                }
                ArgParser::List(_) => {
                    let suggestions =
                        <Self as SingleAttributeParser<S>>::TEMPLATE.suggestions(false, "must_use");
                    cx.emit_err(session_diagnostics::IllFormedAttributeInputLint {
                        num_suggestions: suggestions.len(),
                        suggestions: DiagArgValue::StrListSepByAnd(
                            suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                        ),
                        span: cx.attr_span,
                    });
                    return None;
                }
            },
        })
    }
}
