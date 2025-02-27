use rustc_attr_data_structures::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use super::SingleAttributeParser;
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::session_diagnostics::{ExpectedNameValue, IncorrectMetaItem, UnusedMultiple};

pub(crate) struct CratenameParser;

impl SingleAttributeParser for CratenameParser {
    const PATH: &'static [Symbol] = &[sym::crate_name];

    fn on_duplicate(cx: &AcceptContext<'_>, first_span: Span) {
        // FIXME(jdonszelmann): better duplicate reporting (WIP)
        cx.emit_err(UnusedMultiple {
            this: cx.attr_span,
            other: first_span,
            name: sym::crate_name,
        });
    }

    fn convert(cx: &AcceptContext<'_>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let ArgParser::NameValue(n) = args {
            if let Some(name) = n.value_as_str() {
                Some(AttributeKind::CrateName {
                    name,
                    name_span: n.value_span,
                    style: cx.attr_style,
                })
            } else {
                cx.emit_err(IncorrectMetaItem { span: cx.attr_span, suggestion: None });

                None
            }
        } else {
            cx.emit_err(ExpectedNameValue { span: cx.attr_span, name: sym::crate_name });

            None
        }
    }
}
