use rustc_hir::{AttributeKind, DeprecatedSince, Deprecation};
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol, sym};

use super::util::parse_version;
use super::{AttributeFilter, SingleAttributeGroup};
use crate::context::AttributeAcceptContext;
use crate::parser::{ArgParser, GenericArgParser, MetaItemParser, NameValueParser};
use crate::session_diagnostics::UnsupportedLiteralReason;
use crate::{attribute_filter, session_diagnostics};

pub(crate) struct DeprecationGroup;

fn get<'a>(
    cx: &AttributeAcceptContext<'_>,
    ident: Ident,
    param_span: Span,
    arg: impl ArgParser<'a>,
    item: &mut Option<Symbol>,
) -> bool {
    if item.is_some() {
        cx.dcx().emit_err(session_diagnostics::MultipleItem {
            span: param_span,
            item: ident.to_string(),
        });
        return false;
    }
    if let Some(v) = arg.name_value() {
        if let Some(value_str) = v.value_as_str() {
            *item = Some(value_str);
            true
        } else {
            let lit = v.value_as_lit();
            cx.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                span: v.value_span(),
                reason: UnsupportedLiteralReason::DeprecatedString,
                is_bytestr: lit.kind.is_bytestr(),
                start_point_span: cx.sess().source_map().start_point(lit.span),
            });
            false
        }
    } else {
        // FIXME(jdonszelmann): suggestion?
        cx.dcx().emit_err(session_diagnostics::IncorrectMetaItem {
            span: param_span,
            suggestion: None,
        });
        false
    }
}

impl SingleAttributeGroup for DeprecationGroup {
    const PATH: &'static [rustc_span::Symbol] = &[sym::deprecated];

    fn on_duplicate(cx: &crate::context::AttributeAcceptContext<'_>, first_span: rustc_span::Span) {
        // FIXME(jdonszelmann): merge with errors from check_attrs.rs
        cx.dcx().emit_err(session_diagnostics::UnusedMultiple {
            this: cx.attr_span,
            other: first_span,
            name: sym::deprecated,
        });
    }

    fn convert(
        cx: &AttributeAcceptContext<'_>,
        args: &GenericArgParser<'_, rustc_ast::Expr>,
    ) -> Option<(AttributeKind, AttributeFilter)> {
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
                    cx.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                        span: param_span,
                        reason: UnsupportedLiteralReason::DeprecatedKvPair,
                        is_bytestr: false,
                        start_point_span: cx.sess().source_map().start_point(param_span),
                    });
                    return None;
                };

                let (ident, arg) = param.word_or_empty();

                match ident.name {
                    sym::since => {
                        if !get(cx, ident, param_span, arg, &mut since) {
                            return None;
                        }
                    }
                    sym::note => {
                        if !get(cx, ident, param_span, arg, &mut note) {
                            return None;
                        }
                    }
                    sym::suggestion => {
                        if !features.deprecated_suggestion() {
                            cx.dcx().emit_err(session_diagnostics::DeprecatedItemSuggestion {
                                span: param_span,
                                is_nightly: cx.sess().is_nightly_build(),
                                details: (),
                            });
                        }

                        if !get(cx, ident, param_span, arg, &mut suggestion) {
                            return None;
                        }
                    }
                    _ => {
                        cx.dcx().emit_err(session_diagnostics::UnknownMetaItem {
                            span: param_span,
                            item: ident.to_string(),
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
                cx.dcx().emit_err(session_diagnostics::InvalidSince { span: cx.attr_span });
                DeprecatedSince::Err
            }
        } else if is_rustc {
            cx.dcx().emit_err(session_diagnostics::MissingSince { span: cx.attr_span });
            DeprecatedSince::Err
        } else {
            DeprecatedSince::Unspecified
        };

        if is_rustc && note.is_none() {
            cx.dcx().emit_err(session_diagnostics::MissingNote { span: cx.attr_span });
            return None;
        }

        Some((
            AttributeKind::Deprecation {
                deprecation: Deprecation { since, note, suggestion },
                span: cx.attr_span,
            },
            attribute_filter!(allow all),
        ))
    }
}
