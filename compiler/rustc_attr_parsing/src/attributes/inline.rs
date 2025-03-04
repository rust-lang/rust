// FIXME(jdonszelmann): merge these two parsers and error when both attributes are present here.
//                      note: need to model better how duplicate attr errors work when not using
//                      SingleAttributeParser which is what we have two of here.

use rustc_attr_data_structures::{AttributeKind, InlineAttr};
use rustc_errors::DiagArgValue;
use rustc_feature::{AttributeTemplate, template};
use rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT;
use rustc_span::sym;

use super::{AcceptContext, AttributeOrder, OnDuplicate};
use crate::attributes::SingleAttributeParser;
use crate::context::Stage;
use crate::parser::ArgParser;
use crate::session_diagnostics::IllFormedAttributeInput;

pub(crate) struct InlineParser;

impl<S: Stage> SingleAttributeParser<S> for InlineParser {
    const PATH: &'static [rustc_span::Symbol] = &[sym::inline];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(Word, List: "always|never");

    fn convert(cx: &AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        match args {
            ArgParser::NoArgs => Some(AttributeKind::Inline(InlineAttr::Hint, cx.attr_span)),
            ArgParser::List(list) => {
                let Some(l) = list.single() else {
                    cx.expected_single_argument(list.span);
                    return None;
                };

                match l.meta_item().and_then(|i| i.word_without_args().map(|i| i.name)) {
                    Some(sym::always) => {
                        Some(AttributeKind::Inline(InlineAttr::Always, cx.attr_span))
                    }
                    Some(sym::never) => {
                        Some(AttributeKind::Inline(InlineAttr::Never, cx.attr_span))
                    }
                    _ => {
                        cx.expected_specific_argument(l.span(), vec!["always", "never"]);
                        return None;
                    }
                }
            }
            ArgParser::NameValue(_) => {
                let suggestions =
                    <Self as SingleAttributeParser<S>>::TEMPLATE.suggestions(false, "inline");
                cx.emit_lint(
                    ILL_FORMED_ATTRIBUTE_INPUT,
                    cx.attr_span,
                    IllFormedAttributeInput {
                        num_suggestions: suggestions.len(),
                        suggestions: DiagArgValue::StrListSepByAnd(
                            suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                        ),
                    },
                );
                return None;
            }
        }
    }
}

pub(crate) struct RustcForceInlineParser;

impl<S: Stage> SingleAttributeParser<S> for RustcForceInlineParser {
    const PATH: &'static [rustc_span::Symbol] = &[sym::rustc_force_inline];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(Word, List: "reason", NameValueStr: "reason");

    fn convert(cx: &AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let reason = match args {
            ArgParser::NoArgs => None,
            ArgParser::List(list) => {
                let Some(l) = list.single() else {
                    cx.expected_single_argument(list.span);
                    return None;
                };

                let Some(reason) = l.lit().and_then(|i| i.kind.str()) else {
                    cx.expected_string_literal(l.span());
                    return None;
                };

                Some(reason)
            }
            ArgParser::NameValue(v) => {
                let Some(reason) = v.value_as_str() else {
                    cx.expected_string_literal(v.value_span);
                    return None;
                };

                Some(reason)
            }
        };

        Some(AttributeKind::Inline(
            InlineAttr::Force { attr_span: cx.attr_span, reason },
            cx.attr_span,
        ))
    }
}
