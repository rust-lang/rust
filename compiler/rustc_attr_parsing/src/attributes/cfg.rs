use std::convert::identity;

use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::DelimSpan;
use rustc_ast::{AttrItem, Attribute, CRATE_NODE_ID, LitKind, ast, token};
use rustc_errors::{Applicability, PResult};
use rustc_feature::{
    AttrSuggestionStyle, AttributeTemplate, Features, GatedCfg, find_gated_cfg, template,
};
use rustc_hir::attrs::CfgEntry;
use rustc_hir::lints::AttributeLintKind;
use rustc_hir::{AttrPath, RustcVersion};
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_parse::{exp, parse_in};
use rustc_session::Session;
use rustc_session::config::ExpectedValues;
use rustc_session::lint::builtin::UNEXPECTED_CFGS;
use rustc_session::parse::{ParseSess, feature_err};
use rustc_span::{ErrorGuaranteed, Span, Symbol, sym};
use thin_vec::ThinVec;

use crate::context::{AcceptContext, ShouldEmit, Stage};
use crate::parser::{ArgParser, MetaItemListParser, MetaItemOrLitParser, NameValueParser};
use crate::session_diagnostics::{
    AttributeParseError, AttributeParseErrorReason, CfgAttrBadDelim, MetaBadDelimSugg,
    ParsedDescription,
};
use crate::{AttributeParser, fluent_generated, parse_version, session_diagnostics};

pub const CFG_TEMPLATE: AttributeTemplate = template!(
    List: &["predicate"],
    "https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg-attribute"
);

const CFG_ATTR_TEMPLATE: AttributeTemplate = template!(
    List: &["predicate, attr1, attr2, ..."],
    "https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute"
);

pub fn parse_cfg<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser,
) -> Option<CfgEntry> {
    let ArgParser::List(list) = args else {
        cx.expected_list(cx.attr_span, args);
        return None;
    };
    let Some(single) = list.single() else {
        cx.expected_single_argument(list.span);
        return None;
    };
    parse_cfg_entry(cx, single).ok()
}

pub fn parse_cfg_entry<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    item: &MetaItemOrLitParser,
) -> Result<CfgEntry, ErrorGuaranteed> {
    Ok(match item {
        MetaItemOrLitParser::MetaItemParser(meta) => match meta.args() {
            ArgParser::List(list) => match meta.path().word_sym() {
                Some(sym::not) => {
                    let Some(single) = list.single() else {
                        return Err(cx.expected_single_argument(list.span));
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
                    return Err(cx.emit_err(session_diagnostics::InvalidPredicate {
                        span: meta.span(),
                        predicate: meta.path().to_string(),
                    }));
                }
            },
            a @ (ArgParser::NoArgs | ArgParser::NameValue(_)) => {
                let Some(name) = meta.path().word_sym().filter(|s| !s.is_path_segment_keyword())
                else {
                    return Err(cx.expected_identifier(meta.path().span()));
                };
                parse_name_value(name, meta.path().span(), a.name_value(), meta.span(), cx)?
            }
        },
        MetaItemOrLitParser::Lit(lit) => match lit.kind {
            LitKind::Bool(b) => CfgEntry::Bool(b, lit.span),
            _ => return Err(cx.expected_identifier(lit.span)),
        },
        MetaItemOrLitParser::Err(_, err) => return Err(*err),
    })
}

fn parse_cfg_entry_version<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    list: &MetaItemListParser,
    meta_span: Span,
) -> Result<CfgEntry, ErrorGuaranteed> {
    try_gate_cfg(sym::version, meta_span, cx.sess(), cx.features_option());
    let Some(version) = list.single() else {
        return Err(
            cx.emit_err(session_diagnostics::ExpectedSingleVersionLiteral { span: list.span })
        );
    };
    let Some(version_lit) = version.lit() else {
        return Err(
            cx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: version.span() })
        );
    };
    let Some(version_str) = version_lit.value_str() else {
        return Err(
            cx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: version_lit.span })
        );
    };

    let min_version = parse_version(version_str).or_else(|| {
        cx.sess()
            .dcx()
            .emit_warn(session_diagnostics::UnknownVersionLiteral { span: version_lit.span });
        None
    });

    Ok(CfgEntry::Version(min_version, list.span))
}

fn parse_cfg_entry_target<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    list: &MetaItemListParser,
    meta_span: Span,
) -> Result<CfgEntry, ErrorGuaranteed> {
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
        let Some(name) = sub_item.path().word_sym().filter(|s| !s.is_path_segment_keyword()) else {
            return Err(cx.expected_identifier(sub_item.path().span()));
        };
        let name = Symbol::intern(&format!("target_{name}"));
        if let Ok(cfg) =
            parse_name_value(name, sub_item.path().span(), Some(nv), sub_item.span(), cx)
        {
            result.push(cfg);
        }
    }
    Ok(CfgEntry::All(result, list.span))
}

pub(crate) fn parse_name_value<S: Stage>(
    name: Symbol,
    name_span: Span,
    value: Option<&NameValueParser>,
    span: Span,
    cx: &mut AcceptContext<'_, '_, S>,
) -> Result<CfgEntry, ErrorGuaranteed> {
    try_gate_cfg(name, span, cx.sess(), cx.features_option());

    let value = match value {
        None => None,
        Some(value) => {
            let Some(value_str) = value.value_as_str() else {
                return Err(
                    cx.expected_string_literal(value.value_span, Some(value.value_as_lit()))
                );
            };
            Some((value_str, value.value_span))
        }
    };

    match cx.sess.psess.check_config.expecteds.get(&name) {
        Some(ExpectedValues::Some(values)) if !values.contains(&value.map(|(v, _)| v)) => cx
            .emit_lint(
                UNEXPECTED_CFGS,
                AttributeLintKind::UnexpectedCfgValue((name, name_span), value),
                span,
            ),
        None if cx.sess.psess.check_config.exhaustive_names => cx.emit_lint(
            UNEXPECTED_CFGS,
            AttributeLintKind::UnexpectedCfgName((name, name_span), value),
            span,
        ),
        _ => { /* not unexpected */ }
    }

    Ok(CfgEntry::NameValue { name, value: value.map(|(v, _)| v), span })
}

pub fn eval_config_entry(sess: &Session, cfg_entry: &CfgEntry) -> EvalConfigResult {
    match cfg_entry {
        CfgEntry::All(subs, ..) => {
            for sub in subs {
                let res = eval_config_entry(sess, sub);
                if !res.as_bool() {
                    return res;
                }
            }
            EvalConfigResult::True
        }
        CfgEntry::Any(subs, span) => {
            for sub in subs {
                let res = eval_config_entry(sess, sub);
                if res.as_bool() {
                    return res;
                }
            }
            EvalConfigResult::False { reason: cfg_entry.clone(), reason_span: *span }
        }
        CfgEntry::Not(sub, span) => {
            if eval_config_entry(sess, sub).as_bool() {
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
        CfgEntry::NameValue { name, value, span } => {
            if sess.psess.config.contains(&(*name, *value)) {
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

pub fn parse_cfg_attr(
    cfg_attr: &Attribute,
    sess: &Session,
    features: Option<&Features>,
) -> Option<(CfgEntry, Vec<(AttrItem, Span)>)> {
    match cfg_attr.get_normal_item().args.unparsed_ref().unwrap() {
        ast::AttrArgs::Delimited(ast::DelimArgs { dspan, delim, tokens }) if !tokens.is_empty() => {
            check_cfg_attr_bad_delim(&sess.psess, *dspan, *delim);
            match parse_in(&sess.psess, tokens.clone(), "`cfg_attr` input", |p| {
                parse_cfg_attr_internal(p, sess, features, cfg_attr)
            }) {
                Ok(r) => return Some(r),
                Err(e) => {
                    let suggestions = CFG_ATTR_TEMPLATE
                        .suggestions(AttrSuggestionStyle::Attribute(cfg_attr.style), sym::cfg_attr);
                    e.with_span_suggestions(
                        cfg_attr.span,
                        "must be of the form",
                        suggestions,
                        Applicability::HasPlaceholders,
                    )
                    .with_note(format!(
                        "for more information, visit <{}>",
                        CFG_ATTR_TEMPLATE.docs.expect("cfg_attr has docs")
                    ))
                    .emit();
                }
            }
        }
        _ => {
            let (span, reason) = if let ast::AttrArgs::Delimited(ast::DelimArgs { dspan, .. }) =
                cfg_attr.get_normal_item().args.unparsed_ref()?
            {
                (dspan.entire(), AttributeParseErrorReason::ExpectedAtLeastOneArgument)
            } else {
                (cfg_attr.span, AttributeParseErrorReason::ExpectedList)
            };

            sess.dcx().emit_err(AttributeParseError {
                span,
                attr_span: cfg_attr.span,
                template: CFG_ATTR_TEMPLATE,
                path: AttrPath::from_ast(&cfg_attr.get_normal_item().path, identity),
                description: ParsedDescription::Attribute,
                reason,
                suggestions: CFG_ATTR_TEMPLATE
                    .suggestions(AttrSuggestionStyle::Attribute(cfg_attr.style), sym::cfg_attr),
            });
        }
    }
    None
}

fn check_cfg_attr_bad_delim(psess: &ParseSess, span: DelimSpan, delim: Delimiter) {
    if let Delimiter::Parenthesis = delim {
        return;
    }
    psess.dcx().emit_err(CfgAttrBadDelim {
        span: span.entire(),
        sugg: MetaBadDelimSugg { open: span.open, close: span.close },
    });
}

/// Parses `cfg_attr(pred, attr_item_list)` where `attr_item_list` is comma-delimited.
fn parse_cfg_attr_internal<'a>(
    parser: &mut Parser<'a>,
    sess: &'a Session,
    features: Option<&Features>,
    attribute: &Attribute,
) -> PResult<'a, (CfgEntry, Vec<(ast::AttrItem, Span)>)> {
    // Parse cfg predicate
    let pred_start = parser.token.span;
    let meta = MetaItemOrLitParser::parse_single(parser, ShouldEmit::ErrorsAndLints)?;
    let pred_span = pred_start.with_hi(parser.token.span.hi());

    let cfg_predicate = AttributeParser::parse_single_args(
        sess,
        attribute.span,
        attribute.get_normal_item().span(),
        attribute.style,
        AttrPath { segments: attribute.path().into_boxed_slice(), span: attribute.span },
        Some(attribute.get_normal_item().unsafety),
        ParsedDescription::Attribute,
        pred_span,
        CRATE_NODE_ID,
        features,
        ShouldEmit::ErrorsAndLints,
        &meta,
        parse_cfg_entry,
        &CFG_ATTR_TEMPLATE,
    )
    .map_err(|_err: ErrorGuaranteed| {
        // We have an `ErrorGuaranteed` so this delayed bug cannot fail, but we need a `Diag` for the `PResult` so we make one anyways
        let mut diag = sess.dcx().struct_err(
            "cfg_entry parsing failing with `ShouldEmit::ErrorsAndLints` should emit a error.",
        );
        diag.downgrade_to_delayed_bug();
        diag
    })?;

    parser.expect(exp!(Comma))?;

    // Presumably, the majority of the time there will only be one attr.
    let mut expanded_attrs = Vec::with_capacity(1);
    while parser.token != token::Eof {
        let lo = parser.token.span;
        let item = parser.parse_attr_item(ForceCollect::Yes)?;
        expanded_attrs.push((item, lo.to(parser.prev_token.span)));
        if !parser.eat(exp!(Comma)) {
            break;
        }
    }

    Ok((cfg_predicate, expanded_attrs))
}

fn try_gate_cfg(name: Symbol, span: Span, sess: &Session, features: Option<&Features>) {
    let gate = find_gated_cfg(|sym| sym == name);
    if let (Some(feats), Some(gated_cfg)) = (features, gate) {
        gate_cfg(gated_cfg, span, sess, feats);
    }
}

#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
fn gate_cfg(gated_cfg: &GatedCfg, cfg_span: Span, sess: &Session, features: &Features) {
    let (cfg, feature, has_feature) = gated_cfg;
    if !has_feature(features) && !cfg_span.allows_unstable(*feature) {
        let explain = format!("`cfg({cfg})` is experimental and subject to change");
        feature_err(sess, *feature, cfg_span, explain).emit();
    }
}
