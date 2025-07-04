use rustc_ast::{LitKind, MetaItem, MetaItemInner, MetaItemLit, NodeId};
use rustc_attr_data_structures::{AttributeKind, CfgEntry, RustcVersion};
use rustc_feature::{AttributeTemplate, Features, template};
use rustc_session::Session;
use rustc_session::config::ExpectedValues;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::UNEXPECTED_CFGS;
use rustc_session::parse::feature_err;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::attributes::{CombineAttributeParser, ConvertFn};
use crate::context::{AcceptContext, Stage};
use crate::parser::{ArgParser, MetaItemOrLitParser, NameValueParser};
use crate::{
    CfgMatchesLintEmitter, fluent_generated, parse_version, session_diagnostics, try_gate_cfg,
};

pub(crate) struct CfgParser;
impl<S: Stage> CombineAttributeParser<S> for CfgParser {
    type Item = CfgEntry;
    const PATH: &[Symbol] = &[sym::cfg];
    const CONVERT: ConvertFn<Self::Item> =
        |items, span| AttributeKind::Cfg(CfgEntry::All(items, span), span);
    const TEMPLATE: AttributeTemplate = template!(List: "predicate");

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span);
            return None;
        };
        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };
        parse_cfg_entry(cx, single)
    }
}

//TODO this produces <cfg> as identifier in malformed-attrs.rs
pub(crate) struct CfgTraceParser;
impl<S: Stage> CombineAttributeParser<S> for CfgTraceParser {
    type Item = CfgEntry;
    const PATH: &[Symbol] = &[sym::cfg_trace];
    const CONVERT: ConvertFn<Self::Item> =
        |items, span| AttributeKind::Cfg(CfgEntry::All(items, span), span);
    const TEMPLATE: AttributeTemplate = template!(List: "predicate");

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span);
            return None;
        };
        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };
        parse_cfg_entry(cx, single)
    }
}

fn parse_cfg_entry<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    item: &MetaItemOrLitParser<'_>,
) -> Option<CfgEntry> {
    Some(match item {
        MetaItemOrLitParser::MetaItemParser(meta) => {
            match meta.args() {
                ArgParser::List(list) => {
                    match meta.path().word_sym() {
                        Some(sym::not) => {
                            let Some(single) = list.single() else {
                                cx.expected_single_argument(list.span);
                                return None;
                            };
                            CfgEntry::Not(Box::new(parse_cfg_entry(cx, single)?), list.span)
                        }
                        Some(sym::any) => CfgEntry::Any(
                            list.mixed()
                                .flat_map(|sub_item| parse_cfg_entry(cx, sub_item))
                                .collect(),
                            list.span,
                        ),
                        Some(sym::all) => CfgEntry::All(
                            list.mixed()
                                .flat_map(|sub_item| parse_cfg_entry(cx, sub_item))
                                .collect(),
                            list.span,
                        ),
                        Some(sym::target) => {
                            if !cx.features().cfg_target_compact() {
                                feature_err(
                                    cx.sess(),
                                    sym::cfg_target_compact,
                                    list.span,
                                    fluent_generated::attr_parsing_unstable_cfg_target_compact,
                                )
                                .emit();
                            }

                            let mut result = ThinVec::new();
                            for sub_item in list.mixed() {
                                // First, validate that this is a NameValue item
                                let Some(sub_item) = sub_item.meta_item() else {
                                    cx.expected_name_value(sub_item.span(), None);
                                    continue;
                                };
                                let Some(nv) = sub_item.args().name_value() else {
                                    cx.expected_name_value(sub_item.span(), None);
                                    continue;
                                };

                                // Then, parse it as a name-value item
                                let Some(name) = sub_item.path().word_sym() else { todo!() };
                                let name = Symbol::intern(&format!("target_{name}"));
                                if let Some(cfg) = parse_name_value(
                                    name,
                                    sub_item.path().span(),
                                    Some(nv),
                                    sub_item.span(),
                                    cx,
                                ) {
                                    result.push(cfg);
                                }
                            }
                            CfgEntry::All(result, list.span)
                        }
                        Some(sym::version) => {
                            try_gate_cfg(sym::version, list.span, cx.sess(), Some(cx.features()));

                            todo!()

                            // let (min_version, span) = match &mis[..] {
                            //     [MetaItemInner::Lit(MetaItemLit { kind: LitKind::Str(sym, ..), span, .. })] => {
                            //         (sym, span)
                            //     }
                            //     [
                            //     MetaItemInner::Lit(MetaItemLit { span, .. })
                            //     | MetaItemInner::MetaItem(MetaItem { span, .. }),
                            //     ] => {
                            //         dcx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: *span });
                            //         return false;
                            //     }
                            //     [..] => {
                            //         dcx.emit_err(session_diagnostics::ExpectedSingleVersionLiteral {
                            //             span: cfg.span,
                            //         });
                            //         return false;
                            //     }
                            // };
                            // let Some(min_version) = parse_version(*min_version) else {
                            //     dcx.emit_warn(session_diagnostics::UnknownVersionLiteral { span: *span });
                            //     return false;
                            // };
                            //
                            // // See https://github.com/rust-lang/rust/issues/64796#issuecomment-640851454 for details
                            // if sess.psess.assume_incomplete_release {
                            //     RustcVersion::current_overridable() > min_version
                            // } else {
                            //     RustcVersion::current_overridable() >= min_version
                            // }
                        }
                        _ => {
                            cx.emit_err(session_diagnostics::InvalidPredicate {
                                span: meta.span(),
                                predicate: meta.path().to_string(),
                            });
                            return None;
                        }
                    }
                }
                a @ (ArgParser::NoArgs | ArgParser::NameValue(_)) => {
                    let Some(name) = meta.path().word_sym() else {
                        cx.emit_err(session_diagnostics::CfgPredicateIdentifier {
                            span: meta.path().span(),
                        });
                        return None;
                    };
                    parse_name_value(name, meta.path().span(), a.name_value(), meta.span(), cx)?
                }
            }
        }
        MetaItemOrLitParser::Lit(lit) => match lit.kind {
            LitKind::Bool(b) => CfgEntry::Bool(b, lit.span),
            _ => {
                cx.expected_boolean_literal(lit.span);
                return None;
            }
        },
        MetaItemOrLitParser::Err(_, _) => return None,
    })
}

fn parse_name_value<S: Stage>(
    name: Symbol,
    name_span: Span,
    value: Option<&NameValueParser>,
    span: Span,
    cx: &mut AcceptContext<'_, '_, S>,
) -> Option<CfgEntry> {
    try_gate_cfg(name, span, cx.sess(), Some(cx.features()));

    let value = match value {
        None => None,
        Some(value) => {
            let Some(value_str) = value.value_as_str() else {
                cx.expected_string_literal(value.value_span, Some(value.value_as_lit()));
                return None;
            };
            Some((value_str, value.value_span))
        }
    };

    Some(CfgEntry::NameValue { name, name_span, value, span })
}

pub fn eval_config_entry(
    sess: &Session,
    cfg_entry: &CfgEntry,
    id: NodeId,
    features: Option<&Features>,
) -> EvalConfigResult {
    match cfg_entry {
        CfgEntry::All(subs, span) => {
            if subs.iter().all(|sub| eval_config_entry(sess, sub, id, features).as_bool()) {
                EvalConfigResult::True
            } else {
                //TODO if only one is false, more exact reason
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
            }
        }
        CfgEntry::Any(subs, span) => {
            if subs.iter().any(|sub| eval_config_entry(sess, sub, id, features).as_bool()) {
                EvalConfigResult::True
            } else {
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
            }
        }
        CfgEntry::Not(sub, span) => {
            if eval_config_entry(sess, sub, id, features).as_bool() {
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
            } else {
                EvalConfigResult::True
            }
        }
        CfgEntry::Bool(b, span) => {
            if *b {
                EvalConfigResult::True
            } else {
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
            }
        }
        CfgEntry::NameValue { name, name_span, value, span } => {
            try_gate_cfg(*name, *span, sess, features);

            match sess.psess.check_config.expecteds.get(name) {
                Some(ExpectedValues::Some(values)) if !values.contains(&value.map(|(v, _)| v)) => {
                    id.emit_span_lint(
                        sess,
                        UNEXPECTED_CFGS,
                        *span,
                        BuiltinLintDiag::UnexpectedCfgValue((*name, *name_span), *value),
                    );
                }
                None if sess.psess.check_config.exhaustive_names => {
                    id.emit_span_lint(
                        sess,
                        UNEXPECTED_CFGS,
                        *span,
                        BuiltinLintDiag::UnexpectedCfgName((*name, *name_span), *value),
                    );
                }
                _ => { /* not unexpected */ }
            }

            if sess.psess.config.contains(&(*name, value.map(|(v, _)| v))) {
                EvalConfigResult::True
            } else {
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
            }
        }
    }
}

pub enum EvalConfigResult {
    True,
    False { reason: CfgEntry, reason_span: Span },
}

impl EvalConfigResult {
    pub fn as_bool(&self) -> bool {
        match self {
            EvalConfigResult::True => true,
            EvalConfigResult::False { .. } => false,
        }
    }
}
