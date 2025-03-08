use rustc_attr_data_structures::{AttributeKind, DeprecatedSince, Deprecation};
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Span, Symbol, sym};

use super::util::parse_version;
use super::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics;

pub(crate) struct DeprecationParser;

fn get<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    name: Symbol,
    param_span: Span,
    arg: &ArgParser<'_>,
    item: &Option<Symbol>,
) -> Option<Symbol> {
    if item.is_some() {
        cx.duplicate_key(param_span, name);
        return None;
    }
    if let Some(v) = arg.name_value() {
        if let Some(value_str) = v.value_as_str() {
            Some(value_str)
        } else {
            cx.expected_string_literal(v.value_span, Some(&v.value_as_lit()));
            None
        }
    } else {
        cx.expected_name_value(param_span, Some(name));
        None
    }
}

impl<S: Stage> SingleAttributeParser<S> for DeprecationParser {
    const PATH: &[Symbol] = &[sym::deprecated];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(
        Word,
        List: r#"/*opt*/ since = "version", /*opt*/ note = "reason""#,
        NameValueStr: "reason"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let features = cx.features();

        let mut since = None;
        let mut note = None;
        let mut suggestion = None;

        let is_rustc = features.staged_api();

        match args {
            ArgParser::NoArgs => {
                // ok
            }
            ArgParser::List(list) => {
                for param in list.mixed() {
                    let Some(param) = param.meta_item() else {
                        cx.unexpected_literal(param.span());
                        return None;
                    };

                    let ident_name = param.path().word_sym();

                    match ident_name {
                        Some(name @ sym::since) => {
                            since = Some(get(cx, name, param.span(), param.args(), &since)?);
                        }
                        Some(name @ sym::note) => {
                            note = Some(get(cx, name, param.span(), param.args(), &note)?);
                        }
                        Some(name @ sym::suggestion) => {
                            if !features.deprecated_suggestion() {
                                cx.emit_err(session_diagnostics::DeprecatedItemSuggestion {
                                    span: param.span(),
                                    is_nightly: cx.sess().is_nightly_build(),
                                    details: (),
                                });
                            }

                            suggestion =
                                Some(get(cx, name, param.span(), param.args(), &suggestion)?);
                        }
                        _ => {
                            cx.unknown_key(
                                param.span(),
                                param.path().to_string(),
                                if features.deprecated_suggestion() {
                                    &["since", "note", "suggestion"]
                                } else {
                                    &["since", "note"]
                                },
                            );
                            return None;
                        }
                    }
                }
            }
            ArgParser::NameValue(v) => {
                let Some(value) = v.value_as_str() else {
                    cx.expected_string_literal(v.value_span, Some(v.value_as_lit()));
                    return None;
                };
                note = Some(value);
            }
        }

        let since = if let Some(since) = since {
            if since.as_str() == "TBD" {
                DeprecatedSince::Future
            } else if !is_rustc {
                DeprecatedSince::NonStandard(since)
            } else if let Some(version) = parse_version(since) {
                DeprecatedSince::RustcVersion(version)
            } else {
                cx.emit_err(session_diagnostics::InvalidSince { span: cx.attr_span });
                DeprecatedSince::Err
            }
        } else if is_rustc {
            cx.emit_err(session_diagnostics::MissingSince { span: cx.attr_span });
            DeprecatedSince::Err
        } else {
            DeprecatedSince::Unspecified
        };

        if is_rustc && note.is_none() {
            cx.emit_err(session_diagnostics::MissingNote { span: cx.attr_span });
            return None;
        }

        Some(AttributeKind::Deprecation {
            deprecation: Deprecation { since, note, suggestion },
            span: cx.attr_span,
        })
    }
}
