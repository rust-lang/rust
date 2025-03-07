// FIXME(jdonszelmann): merge these two parsers and error when both attributes are present here.
//                      note: need to model better how duplicate attr errors work when not using
//                      SingleAttributeParser which is what we have two of here.

use rustc_attr_data_structures::{AttributeKind, InlineAttr};
use rustc_errors::{E0534, E0535, struct_span_code_err};
use rustc_span::sym;

use super::{AcceptContext, AttributeOrder, OnDuplicate};
use crate::attributes::SingleAttributeParser;
use crate::context::Stage;
use crate::parser::ArgParser;
use crate::session_diagnostics::IncorrectMetaItem;

pub(crate) struct InlineParser;

impl<S: Stage> SingleAttributeParser<S> for InlineParser {
    const PATH: &'static [rustc_span::Symbol] = &[sym::inline];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;

    fn convert(cx: &AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        match args {
            ArgParser::NoArgs => Some(AttributeKind::Inline(InlineAttr::Hint, cx.attr_span)),
            ArgParser::List(list) => {
                let Some(l) = list.single() else {
                    struct_span_code_err!(cx.dcx(), cx.attr_span, E0534, "expected one argument")
                        .emit();
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
                        struct_span_code_err!(cx.dcx(), l.span(), E0535, "invalid argument")
                            .with_help("valid inline arguments are `always` and `never`")
                            .emit();
                        return None;
                    }
                }
            }
            ArgParser::NameValue(_) => {
                // silently ignored, we warn somewhere else.
                // FIXME(jdonszelmann): that warning *should* go here.
                None
            }
        }
    }
}

pub(crate) struct RustcForceInlineParser;

impl<S: Stage> SingleAttributeParser<S> for RustcForceInlineParser {
    const PATH: &'static [rustc_span::Symbol] = &[sym::rustc_force_inline];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;

    fn convert(cx: &AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let reason = match args {
            ArgParser::NoArgs => None,
            ArgParser::List(list) => {
                cx.emit_err(IncorrectMetaItem {
                    span: list.span,
                    suggestion: None,
                });
                return None
            }
            ArgParser::NameValue(v) => {
                let Some(str) = v.value_as_str() else {
                    cx.emit_err(IncorrectMetaItem {
                        span: v.value_span,
                        suggestion: None,
                    });
                    return None;
                };

                Some(str)
            }
        };

        Some(AttributeKind::Inline(InlineAttr::Force { attr_span: cx.attr_span, reason }, cx.attr_span))
    }
}
