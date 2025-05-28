use rustc_ast::{LitKind, MetaItem, MetaItemInner, MetaItemKind, MetaItemLit, NodeId};
use rustc_ast_pretty::pprust;
use rustc_attr_data_structures::RustcVersion;
use rustc_feature::{Features, GatedCfg, find_gated_cfg};
use rustc_session::Session;
use rustc_session::config::ExpectedValues;
use rustc_session::lint::builtin::UNEXPECTED_CFGS;
use rustc_session::lint::{BuiltinLintDiag, Lint};
use rustc_session::parse::feature_err;
use rustc_span::{Span, Symbol, sym};

use crate::session_diagnostics::{self, UnsupportedLiteralReason};
use crate::{fluent_generated, parse_version};

/// Emitter of a builtin lint from `cfg_matches`.
///
/// Used to support emiting a lint (currently on check-cfg), either:
///  - as an early buffered lint (in `rustc`)
///  - or has a "normal" lint from HIR (in `rustdoc`)
pub trait CfgMatchesLintEmitter {
    fn emit_span_lint(&self, sess: &Session, lint: &'static Lint, sp: Span, diag: BuiltinLintDiag);
}

impl CfgMatchesLintEmitter for NodeId {
    fn emit_span_lint(&self, sess: &Session, lint: &'static Lint, sp: Span, diag: BuiltinLintDiag) {
        sess.psess.buffer_lint(lint, sp, *self, diag);
    }
}

#[derive(Clone, Debug)]
pub struct Condition {
    pub name: Symbol,
    pub name_span: Span,
    pub value: Option<Symbol>,
    pub value_span: Option<Span>,
    pub span: Span,
}

/// Tests if a cfg-pattern matches the cfg set
pub fn cfg_matches(
    cfg: &MetaItemInner,
    sess: &Session,
    lint_emitter: impl CfgMatchesLintEmitter,
    features: Option<&Features>,
) -> bool {
    eval_condition(cfg, sess, features, &mut |cfg| {
        try_gate_cfg(cfg.name, cfg.span, sess, features);
        match sess.psess.check_config.expecteds.get(&cfg.name) {
            Some(ExpectedValues::Some(values)) if !values.contains(&cfg.value) => {
                lint_emitter.emit_span_lint(
                    sess,
                    UNEXPECTED_CFGS,
                    cfg.span,
                    BuiltinLintDiag::UnexpectedCfgValue(
                        (cfg.name, cfg.name_span),
                        cfg.value.map(|v| (v, cfg.value_span.unwrap())),
                    ),
                );
            }
            None if sess.psess.check_config.exhaustive_names => {
                lint_emitter.emit_span_lint(
                    sess,
                    UNEXPECTED_CFGS,
                    cfg.span,
                    BuiltinLintDiag::UnexpectedCfgName(
                        (cfg.name, cfg.name_span),
                        cfg.value.map(|v| (v, cfg.value_span.unwrap())),
                    ),
                );
            }
            _ => { /* not unexpected */ }
        }
        sess.psess.config.contains(&(cfg.name, cfg.value))
    })
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

/// Evaluate a cfg-like condition (with `any` and `all`), using `eval` to
/// evaluate individual items.
pub fn eval_condition(
    cfg: &MetaItemInner,
    sess: &Session,
    features: Option<&Features>,
    eval: &mut impl FnMut(Condition) -> bool,
) -> bool {
    let dcx = sess.dcx();

    let cfg = match cfg {
        MetaItemInner::MetaItem(meta_item) => meta_item,
        MetaItemInner::Lit(MetaItemLit { kind: LitKind::Bool(b), .. }) => {
            return *b;
        }
        _ => {
            dcx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: cfg.span(),
                reason: UnsupportedLiteralReason::CfgBoolean,
                is_bytestr: false,
                start_point_span: sess.source_map().start_point(cfg.span()),
            });
            return false;
        }
    };

    match &cfg.kind {
        MetaItemKind::List(mis) if cfg.has_name(sym::version) => {
            try_gate_cfg(sym::version, cfg.span, sess, features);
            let (min_version, span) = match &mis[..] {
                [MetaItemInner::Lit(MetaItemLit { kind: LitKind::Str(sym, ..), span, .. })] => {
                    (sym, span)
                }
                [
                    MetaItemInner::Lit(MetaItemLit { span, .. })
                    | MetaItemInner::MetaItem(MetaItem { span, .. }),
                ] => {
                    dcx.emit_err(session_diagnostics::ExpectedVersionLiteral { span: *span });
                    return false;
                }
                [..] => {
                    dcx.emit_err(session_diagnostics::ExpectedSingleVersionLiteral {
                        span: cfg.span,
                    });
                    return false;
                }
            };
            let Some(min_version) = parse_version(*min_version) else {
                dcx.emit_warn(session_diagnostics::UnknownVersionLiteral { span: *span });
                return false;
            };

            // See https://github.com/rust-lang/rust/issues/64796#issuecomment-640851454 for details
            if sess.psess.assume_incomplete_release {
                RustcVersion::current_overridable() > min_version
            } else {
                RustcVersion::current_overridable() >= min_version
            }
        }
        MetaItemKind::List(mis) => {
            for mi in mis.iter() {
                if mi.meta_item_or_bool().is_none() {
                    dcx.emit_err(session_diagnostics::UnsupportedLiteral {
                        span: mi.span(),
                        reason: UnsupportedLiteralReason::Generic,
                        is_bytestr: false,
                        start_point_span: sess.source_map().start_point(mi.span()),
                    });
                    return false;
                }
            }

            // The unwraps below may look dangerous, but we've already asserted
            // that they won't fail with the loop above.
            match cfg.name() {
                Some(sym::any) => mis
                    .iter()
                    // We don't use any() here, because we want to evaluate all cfg condition
                    // as eval_condition can (and does) extra checks
                    .fold(false, |res, mi| res | eval_condition(mi, sess, features, eval)),
                Some(sym::all) => mis
                    .iter()
                    // We don't use all() here, because we want to evaluate all cfg condition
                    // as eval_condition can (and does) extra checks
                    .fold(true, |res, mi| res & eval_condition(mi, sess, features, eval)),
                Some(sym::not) => {
                    let [mi] = mis.as_slice() else {
                        dcx.emit_err(session_diagnostics::ExpectedOneCfgPattern { span: cfg.span });
                        return false;
                    };

                    !eval_condition(mi, sess, features, eval)
                }
                Some(sym::target) => {
                    if let Some(features) = features
                        && !features.cfg_target_compact()
                    {
                        feature_err(
                            sess,
                            sym::cfg_target_compact,
                            cfg.span,
                            fluent_generated::attr_parsing_unstable_cfg_target_compact,
                        )
                        .emit();
                    }

                    mis.iter().fold(true, |res, mi| {
                        let Some(mut mi) = mi.meta_item().cloned() else {
                            dcx.emit_err(session_diagnostics::CfgPredicateIdentifier {
                                span: mi.span(),
                            });
                            return false;
                        };

                        if let [seg, ..] = &mut mi.path.segments[..] {
                            seg.ident.name = Symbol::intern(&format!("target_{}", seg.ident.name));
                        }

                        res & eval_condition(&MetaItemInner::MetaItem(mi), sess, features, eval)
                    })
                }
                _ => {
                    dcx.emit_err(session_diagnostics::InvalidPredicate {
                        span: cfg.span,
                        predicate: pprust::path_to_string(&cfg.path),
                    });
                    false
                }
            }
        }
        MetaItemKind::Word | MetaItemKind::NameValue(..) if cfg.path.segments.len() != 1 => {
            dcx.emit_err(session_diagnostics::CfgPredicateIdentifier { span: cfg.path.span });
            true
        }
        MetaItemKind::NameValue(lit) if !lit.kind.is_str() => {
            dcx.emit_err(session_diagnostics::UnsupportedLiteral {
                span: lit.span,
                reason: UnsupportedLiteralReason::CfgString,
                is_bytestr: lit.kind.is_bytestr(),
                start_point_span: sess.source_map().start_point(lit.span),
            });
            true
        }
        MetaItemKind::Word | MetaItemKind::NameValue(..) => {
            let ident = cfg.ident().expect("multi-segment cfg predicate");
            eval(Condition {
                name: ident.name,
                name_span: ident.span,
                value: cfg.value_str(),
                value_span: cfg.name_value_literal_span(),
                span: cfg.span,
            })
        }
    }
}
