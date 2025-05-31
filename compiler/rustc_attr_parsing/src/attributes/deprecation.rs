use rustc_attr_data_structures::{AttributeKind, DeprecatedSince, Deprecation};
use rustc_span::{Span, Symbol, sym};

use super::SingleAttributeParser;
use super::util::parse_version;
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::session_diagnostics;
use crate::session_diagnostics::UnsupportedLiteralReason;

pub(crate) struct DeprecationParser;

fn get(
    cx: &AcceptContext<'_>,
    name: Symbol,
    param_span: Span,
    arg: &ArgParser<'_>,
    item: &Option<Symbol>,
) -> Option<Symbol> {
    if item.is_some() {
        cx.emit_err(session_diagnostics::MultipleItem { span: param_span, item: name.to_string() });
        return None;
    }
    if let Some(v) = arg.name_value() {
        if let Some(value_str) = v.value_as_str() {
            Some(value_str)
        } else {
            let lit = v.value_as_lit();
            cx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: v.value_span,
                reason: UnsupportedLiteralReason::DeprecatedString,
                is_bytestr: lit.kind.is_bytestr(),
                start_point_span: cx.sess().source_map().start_point(lit.span),
            });
            None
        }
    } else {
        // FIXME(jdonszelmann): suggestion?
        cx.emit_err(session_diagnostics::IncorrectMetaItem { span: param_span, suggestion: None });
        None
    }
}

impl SingleAttributeParser for DeprecationParser {
    const PATH: &'static [Symbol] = &[sym::deprecated];

    fn on_duplicate(cx: &AcceptContext<'_>, first_span: Span) {
        // FIXME(jdonszelmann): merge with errors from check_attrs.rs
        cx.emit_err(session_diagnostics::UnusedMultiple {
            this: cx.attr_span,
            other: first_span,
            name: sym::deprecated,
        });
    }

    fn convert(cx: &AcceptContext<'_>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let features = cx.features();

        let mut since = None;
        let mut note = None;
        let mut suggestion = None;

        let is_rustc = features.staged_api();

        if let Some(value) = args.name_value()
            && let Some(value_str) = value.value_as_str()
        {
            note = Some(value_str)
        } else if let Some(list) = args.list() {
            for param in list.mixed() {
                let param_span = param.span();
                let Some(param) = param.meta_item() else {
                    cx.emit_err(session_diagnostics::UnsupportedLiteral {
                        span: param_span,
                        reason: UnsupportedLiteralReason::DeprecatedKvPair,
                        is_bytestr: false,
                        start_point_span: cx.sess().source_map().start_point(param_span),
                    });
                    return None;
                };

                let ident_name = param.path().word_sym();

                match ident_name {
                    Some(name @ sym::since) => {
                        since = Some(get(cx, name, param_span, param.args(), &since)?);
                    }
                    Some(name @ sym::note) => {
                        note = Some(get(cx, name, param_span, param.args(), &note)?);
                    }
                    Some(name @ sym::suggestion) => {
                        if !features.deprecated_suggestion() {
                            cx.emit_err(session_diagnostics::DeprecatedItemSuggestion {
                                span: param_span,
                                is_nightly: cx.sess().is_nightly_build(),
                                details: (),
                            });
                        }

                        suggestion = Some(get(cx, name, param_span, param.args(), &suggestion)?);
                    }
                    _ => {
                        cx.emit_err(session_diagnostics::UnknownMetaItem {
                            span: param_span,
                            item: param.path().to_string(),
                            expected: if features.deprecated_suggestion() {
                                &["since", "note", "suggestion"]
                            } else {
                                &["since", "note"]
                            },
                        });
                        return None;
                    }
                }
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
