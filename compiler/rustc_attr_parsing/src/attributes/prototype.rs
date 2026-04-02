//! Attributes that are only used on function prototypes.

use rustc_feature::{AttributeTemplate, template};
use rustc_hir::Target;
use rustc_hir::attrs::{AttributeKind, MirDialect, MirPhase};
use rustc_span::{Span, Symbol, sym};

use super::OnDuplicate;
use crate::attributes::SingleAttributeParser;
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::Allow;

pub(crate) struct CustomMirParser;

impl<S: Stage> SingleAttributeParser<S> for CustomMirParser {
    const PATH: &[rustc_span::Symbol] = &[sym::custom_mir];

    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);

    const TEMPLATE: AttributeTemplate = template!(List: &[r#"dialect = "...", phase = "...""#]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            let attr_span = cx.attr_span;
            cx.adcx().expected_list(attr_span, args);
            return None;
        };

        let mut dialect = None;
        let mut phase = None;
        let mut failed = false;

        for item in list.mixed() {
            let Some(meta_item) = item.meta_item() else {
                cx.adcx().expected_name_value(item.span(), None);
                failed = true;
                break;
            };

            if let Some(arg) = meta_item.word_is(sym::dialect) {
                extract_value(cx, sym::dialect, arg, meta_item.span(), &mut dialect, &mut failed);
            } else if let Some(arg) = meta_item.word_is(sym::phase) {
                extract_value(cx, sym::phase, arg, meta_item.span(), &mut phase, &mut failed);
            } else if let Some(..) = meta_item.path().word() {
                cx.adcx().expected_specific_argument(meta_item.span(), &[sym::dialect, sym::phase]);
                failed = true;
            } else {
                cx.adcx().expected_name_value(meta_item.span(), None);
                failed = true;
            };
        }

        let dialect = parse_dialect(cx, dialect, &mut failed);
        let phase = parse_phase(cx, phase, &mut failed);
        check_custom_mir(cx, dialect, phase, &mut failed);

        if failed {
            return None;
        }

        Some(AttributeKind::CustomMir(dialect, phase, cx.attr_span))
    }
}

fn extract_value<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    key: Symbol,
    arg: &ArgParser,
    span: Span,
    out_val: &mut Option<(Symbol, Span)>,
    failed: &mut bool,
) {
    if out_val.is_some() {
        cx.adcx().duplicate_key(span, key);
        *failed = true;
        return;
    }

    let Some(val) = arg.name_value() else {
        cx.adcx().expected_single_argument(arg.span().unwrap_or(span));
        *failed = true;
        return;
    };

    let Some(value_sym) = val.value_as_str() else {
        cx.adcx().expected_string_literal(val.value_span, Some(val.value_as_lit()));
        *failed = true;
        return;
    };

    *out_val = Some((value_sym, val.value_span));
}

fn parse_dialect<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    dialect: Option<(Symbol, Span)>,
    failed: &mut bool,
) -> Option<(MirDialect, Span)> {
    let (dialect, span) = dialect?;

    let dialect = match dialect {
        sym::analysis => MirDialect::Analysis,
        sym::built => MirDialect::Built,
        sym::runtime => MirDialect::Runtime,

        _ => {
            cx.adcx().expected_specific_argument(span, &[sym::analysis, sym::built, sym::runtime]);
            *failed = true;
            return None;
        }
    };

    Some((dialect, span))
}

fn parse_phase<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    phase: Option<(Symbol, Span)>,
    failed: &mut bool,
) -> Option<(MirPhase, Span)> {
    let (phase, span) = phase?;

    let phase = match phase {
        sym::initial => MirPhase::Initial,
        sym::post_cleanup => MirPhase::PostCleanup,
        sym::optimized => MirPhase::Optimized,

        _ => {
            cx.adcx().expected_specific_argument(
                span,
                &[sym::initial, sym::post_cleanup, sym::optimized],
            );
            *failed = true;
            return None;
        }
    };

    Some((phase, span))
}

fn check_custom_mir<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    dialect: Option<(MirDialect, Span)>,
    phase: Option<(MirPhase, Span)>,
    failed: &mut bool,
) {
    let attr_span = cx.attr_span;
    let Some((dialect, dialect_span)) = dialect else {
        if let Some((_, phase_span)) = phase {
            *failed = true;
            cx.emit_err(session_diagnostics::CustomMirPhaseRequiresDialect {
                attr_span,
                phase_span,
            });
        }
        return;
    };

    match dialect {
        MirDialect::Analysis => {
            if let Some((MirPhase::Optimized, phase_span)) = phase {
                *failed = true;
                cx.emit_err(session_diagnostics::CustomMirIncompatibleDialectAndPhase {
                    dialect,
                    phase: MirPhase::Optimized,
                    attr_span,
                    dialect_span,
                    phase_span,
                });
            }
        }

        MirDialect::Built => {
            if let Some((phase, phase_span)) = phase {
                *failed = true;
                cx.emit_err(session_diagnostics::CustomMirIncompatibleDialectAndPhase {
                    dialect,
                    phase,
                    attr_span,
                    dialect_span,
                    phase_span,
                });
            }
        }
        MirDialect::Runtime => {}
    }
}
