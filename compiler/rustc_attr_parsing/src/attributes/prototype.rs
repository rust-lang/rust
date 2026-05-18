//! Attributes that are only used on function prototypes.

use rustc_feature::{AttributeTemplate, template};
use rustc_hir::Target;
use rustc_hir::attrs::{AttributeKind, MirDialect, MirPhase};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::SingleAttributeParser;
use crate::context::AcceptContext;
use crate::parser::{ArgParser, NameValueParser};
use crate::session_diagnostics;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::Allow;

pub(crate) struct CustomMirParser;

impl SingleAttributeParser for CustomMirParser {
    const PATH: &[rustc_span::Symbol] = &[sym::custom_mir];

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);

    const TEMPLATE: AttributeTemplate = template!(List: &[r#"dialect = "...", phase = "...""#]);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let list = cx.expect_list(args, cx.attr_span)?;

        let mut dialect = None;
        let mut phase = None;
        let mut failed = false;

        for item in list.mixed() {
            let Some((path, arg)) = cx.expect_name_value(item, item.span(), None) else {
                failed = true;
                break;
            };

            match path.name {
                sym::dialect => {
                    extract_value(cx, sym::dialect, arg, item.span(), &mut dialect, &mut failed)
                }
                sym::phase => {
                    extract_value(cx, sym::phase, arg, item.span(), &mut phase, &mut failed)
                }
                _ => {
                    cx.adcx().expected_specific_argument(item.span(), &[sym::dialect, sym::phase]);
                    failed = true;
                }
            }
        }

        let dialect = parse_dialect(cx, dialect, &mut failed);
        let phase = parse_phase(cx, phase, &mut failed);
        check_custom_mir(cx, dialect, phase, &mut failed);

        if failed {
            return None;
        }

        Some(AttributeKind::CustomMir(dialect, phase))
    }
}

fn extract_value(
    cx: &mut AcceptContext<'_, '_>,
    key: Symbol,
    val: &NameValueParser,
    span: Span,
    out_val: &mut Option<(Symbol, Span)>,
    failed: &mut bool,
) {
    if out_val.is_some() {
        cx.adcx().duplicate_key(span, key);
        *failed = true;
        return;
    }

    let Some(value_sym) = cx.expect_string_literal(val) else {
        *failed = true;
        return;
    };

    *out_val = Some((value_sym, val.value_span));
}

fn parse_dialect(
    cx: &mut AcceptContext<'_, '_>,
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

fn parse_phase(
    cx: &mut AcceptContext<'_, '_>,
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

fn check_custom_mir(
    cx: &mut AcceptContext<'_, '_>,
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
