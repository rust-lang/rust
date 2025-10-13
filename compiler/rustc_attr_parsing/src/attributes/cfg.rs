use rustc_ast::{LitKind, NodeId};
use rustc_feature::{AttributeTemplate, Features, template};
use rustc_hir::RustcVersion;
use rustc_hir::attrs::CfgEntry;
use rustc_session::Session;
use rustc_session::config::ExpectedValues;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::UNEXPECTED_CFGS;
use rustc_session::parse::feature_err;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::context::{AcceptContext, ShouldEmit, Stage};
use crate::parser::{ArgParser, MetaItemListParser, MetaItemOrLitParser, NameValueParser};
use crate::{
    CfgMatchesLintEmitter, fluent_generated, parse_version, session_diagnostics, try_gate_cfg,
};

pub const CFG_TEMPLATE: AttributeTemplate = template!(
    List: &["predicate"],
    "https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg-attribute"
);

pub fn parse_cfg_attr<'c, S: Stage>(
    cx: &'c mut AcceptContext<'_, '_, S>,
    args: &'c ArgParser<'_>,
) -> Option<CfgEntry> {
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

pub(crate) fn parse_cfg_entry<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    item: &MetaItemOrLitParser<'_>,
) -> Option<CfgEntry> {
    Some(match item {
        MetaItemOrLitParser::MetaItemParser(meta) => match meta.args() {
            ArgParser::List(list) => match meta.path().word_sym() {
                Some(sym::not) => {
                    let Some(single) = list.single() else {
                        cx.expected_single_argument(list.span);
                        return None;
                    };
                    CfgEntry::Not(Box::new(parse_cfg_entry(cx, single)?), list.span)
                }
                Some(sym::any) => CfgEntry::Any(
                    list.mixed().flat_map(|sub_item| parse_cfg_entry(cx, sub_item)).collect(),
                    list.span,
                ),
                Some(sym::all) => CfgEntry::All(
                    list.mixed().flat_map(|sub_item| parse_cfg_entry(cx, sub_item)).collect(),
                    list.span,
                ),
                Some(sym::target) => parse_cfg_entry_target(cx, list, meta.span())?,
                Some(sym::version) => parse_cfg_entry_version(cx, list, meta.span())?,
                _ => {
                    cx.emit_err(session_diagnostics::InvalidPredicate {
                        span: meta.span(),
                        predicate: meta.path().to_string(),
                    });
                    return None;
                }
            },
            a @ (ArgParser::NoArgs | ArgParser::NameValue(_)) => {
                let Some(name) = meta.path().word_sym() else {
                    cx.emit_err(session_diagnostics::CfgPredicateIdentifier {
                        span: meta.path().span(),
                    });
                    return None;
                };
                parse_name_value(name, meta.path().span(), a.name_value(), meta.span(), cx)?
            }
        },
        MetaItemOrLitParser::Lit(lit) => match lit.kind {
            LitKind::Bool(b) => CfgEntry::Bool(b, lit.span),
            _ => {
                cx.emit_err(session_diagnostics::CfgPredicateIdentifier { span: lit.span });
                return None;
            }
        },
        MetaItemOrLitParser::Err(_, _) => return None,
    })
}

fn parse_cfg_entry_version<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    list: &MetaItemListParser<'_>,
    meta_span: Span,
) -> Option<CfgEntry> {
    try_gate_cfg(sym::version, meta_span, cx.sess(), cx.features_option());
    let Some(version) = list.single() else {
        cx.emit_err(session_diagnostics::ExpectedSingleVersionLiteral { span: list.span });
        return None;
    };
    let Some(version_lit) = version.lit() else {
        cx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: version.span() });
        return None;
    };
    let Some(version_str) = version_lit.value_str() else {
        cx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: version_lit.span });
        return None;
    };

    let min_version = parse_version(version_str).or_else(|| {
        cx.sess()
            .dcx()
            .emit_warn(session_diagnostics::UnknownVersionLiteral { span: version_lit.span });
        None
    });

    Some(CfgEntry::Version(min_version, list.span))
}

fn parse_cfg_entry_target<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    list: &MetaItemListParser<'_>,
    meta_span: Span,
) -> Option<CfgEntry> {
    if let Some(features) = cx.features_option()
        && !features.cfg_target_compact()
    {
        feature_err(
            cx.sess(),
            sym::cfg_target_compact,
            meta_span,
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
        let Some(name) = sub_item.path().word_sym() else {
            cx.emit_err(session_diagnostics::CfgPredicateIdentifier {
                span: sub_item.path().span(),
            });
            return None;
        };
        let name = Symbol::intern(&format!("target_{name}"));
        if let Some(cfg) =
            parse_name_value(name, sub_item.path().span(), Some(nv), sub_item.span(), cx)
        {
            result.push(cfg);
        }
    }
    Some(CfgEntry::All(result, list.span))
}

fn parse_name_value<S: Stage>(
    name: Symbol,
    name_span: Span,
    value: Option<&NameValueParser>,
    span: Span,
    cx: &mut AcceptContext<'_, '_, S>,
) -> Option<CfgEntry> {
    try_gate_cfg(name, span, cx.sess(), cx.features_option());

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
    emit_lints: ShouldEmit,
) -> EvalConfigResult {
    match cfg_entry {
        CfgEntry::All(subs, ..) => {
            let mut all = None;
            for sub in subs {
                let res = eval_config_entry(sess, sub, id, features, emit_lints);
                // We cannot short-circuit because `eval_config_entry` emits some lints
                if !res.as_bool() {
                    all.get_or_insert(res);
                }
            }
            all.unwrap_or_else(|| EvalConfigResult::True)
        }
        CfgEntry::Any(subs, span) => {
            let mut any = None;
            for sub in subs {
                let res = eval_config_entry(sess, sub, id, features, emit_lints);
                // We cannot short-circuit because `eval_config_entry` emits some lints
                if res.as_bool() {
                    any.get_or_insert(res);
                }
            }
            any.unwrap_or_else(|| EvalConfigResult::False {
                reason: cfg_entry.clone(),
                reason_span: *span,
            })
        }
        CfgEntry::Not(sub, span) => {
            if eval_config_entry(sess, sub, id, features, emit_lints).as_bool() {
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
            if let ShouldEmit::ErrorsAndLints = emit_lints {
                match sess.psess.check_config.expecteds.get(name) {
                    Some(ExpectedValues::Some(values))
                        if !values.contains(&value.map(|(v, _)| v)) =>
                    {
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
            }

            if sess.psess.config.contains(&(*name, value.map(|(v, _)| v))) {
                EvalConfigResult::True
            } else {
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
            }
        }
        CfgEntry::Version(min_version, version_span) => {
            let Some(min_version) = min_version else {
                return EvalConfigResult::False {
                    reason: cfg_entry.clone(),
                    reason_span: *version_span,
                };
            };
            // See https://github.com/rust-lang/rust/issues/64796#issuecomment-640851454 for details
            let min_version_ok = if sess.psess.assume_incomplete_release {
                RustcVersion::current_overridable() > *min_version
            } else {
                RustcVersion::current_overridable() >= *min_version
            };
            if min_version_ok {
                EvalConfigResult::True
            } else {
                EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *version_span }
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
