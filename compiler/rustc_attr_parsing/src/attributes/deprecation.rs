//! Parsing and validation of builtin attributes

use rustc_ast::attr::AttributeExt;
use rustc_ast::{MetaItem, MetaItemInner};
use rustc_ast_pretty::pprust;
use rustc_attr_data_structures::{DeprecatedSince, Deprecation};
use rustc_feature::Features;
use rustc_session::Session;
use rustc_span::{Span, Symbol, sym};

use super::util::UnsupportedLiteralReason;
use crate::{parse_version, session_diagnostics};

/// Finds the deprecation attribute. `None` if none exists.
pub fn find_deprecation(
    sess: &Session,
    features: &Features,
    attrs: &[impl AttributeExt],
) -> Option<(Deprecation, Span)> {
    let mut depr: Option<(Deprecation, Span)> = None;
    let is_rustc = features.staged_api();

    'outer: for attr in attrs {
        if !attr.has_name(sym::deprecated) {
            continue;
        }

        let mut since = None;
        let mut note = None;
        let mut suggestion = None;

        if attr.is_doc_comment() {
            continue;
        } else if attr.is_word() {
        } else if let Some(value) = attr.value_str() {
            note = Some(value)
        } else if let Some(list) = attr.meta_item_list() {
            let get = |meta: &MetaItem, item: &mut Option<Symbol>| {
                if item.is_some() {
                    sess.dcx().emit_err(session_diagnostics::MultipleItem {
                        span: meta.span,
                        item: pprust::path_to_string(&meta.path),
                    });
                    return false;
                }
                if let Some(v) = meta.value_str() {
                    *item = Some(v);
                    true
                } else {
                    if let Some(lit) = meta.name_value_literal() {
                        sess.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                            span: lit.span,
                            reason: UnsupportedLiteralReason::DeprecatedString,
                            is_bytestr: lit.kind.is_bytestr(),
                            start_point_span: sess.source_map().start_point(lit.span),
                        });
                    } else {
                        sess.dcx()
                            .emit_err(session_diagnostics::IncorrectMetaItem { span: meta.span });
                    }
                    false
                }
            };

            for meta in &list {
                match meta {
                    MetaItemInner::MetaItem(mi) => match mi.name_or_empty() {
                        sym::since => {
                            if !get(mi, &mut since) {
                                continue 'outer;
                            }
                        }
                        sym::note => {
                            if !get(mi, &mut note) {
                                continue 'outer;
                            }
                        }
                        sym::suggestion => {
                            if !features.deprecated_suggestion() {
                                sess.dcx().emit_err(
                                    session_diagnostics::DeprecatedItemSuggestion {
                                        span: mi.span,
                                        is_nightly: sess.is_nightly_build(),
                                        details: (),
                                    },
                                );
                            }

                            if !get(mi, &mut suggestion) {
                                continue 'outer;
                            }
                        }
                        _ => {
                            sess.dcx().emit_err(session_diagnostics::UnknownMetaItem {
                                span: meta.span(),
                                item: pprust::path_to_string(&mi.path),
                                expected: if features.deprecated_suggestion() {
                                    &["since", "note", "suggestion"]
                                } else {
                                    &["since", "note"]
                                },
                            });
                            continue 'outer;
                        }
                    },
                    MetaItemInner::Lit(lit) => {
                        sess.dcx().emit_err(session_diagnostics::UnsupportedLiteral {
                            span: lit.span,
                            reason: UnsupportedLiteralReason::DeprecatedKvPair,
                            is_bytestr: false,
                            start_point_span: sess.source_map().start_point(lit.span),
                        });
                        continue 'outer;
                    }
                }
            }
        } else {
            continue;
        }

        let since = if let Some(since) = since {
            if since.as_str() == "TBD" {
                DeprecatedSince::Future
            } else if !is_rustc {
                DeprecatedSince::NonStandard(since)
            } else if let Some(version) = parse_version(since) {
                DeprecatedSince::RustcVersion(version)
            } else {
                sess.dcx().emit_err(session_diagnostics::InvalidSince { span: attr.span() });
                DeprecatedSince::Err
            }
        } else if is_rustc {
            sess.dcx().emit_err(session_diagnostics::MissingSince { span: attr.span() });
            DeprecatedSince::Err
        } else {
            DeprecatedSince::Unspecified
        };

        if is_rustc && note.is_none() {
            sess.dcx().emit_err(session_diagnostics::MissingNote { span: attr.span() });
            continue;
        }

        depr = Some((Deprecation { since, note, suggestion }, attr.span()));
    }

    depr
}
