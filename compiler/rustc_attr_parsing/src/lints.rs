use rustc_attr_data_structures::lints::{AttributeLint, AttributeLintKind};
use rustc_errors::{DiagArgValue, LintEmitter};
use rustc_hir::HirId;

use crate::session_diagnostics;
use crate::session_diagnostics::{UnsafeAttrOutsideUnsafeLint, UnsafeAttrOutsideUnsafeSuggestion};

pub fn emit_attribute_lint<L: LintEmitter>(lint: &AttributeLint<HirId>, lint_emitter: L) {
    let AttributeLint { id, span, kind } = lint;

    match kind {
        &AttributeLintKind::UnusedDuplicate { this, other, warning } => lint_emitter
            .emit_node_span_lint(
                rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
                *id,
                *span,
                session_diagnostics::UnusedDuplicate { this, other, warning },
            ),
        AttributeLintKind::IllFormedAttributeInput { suggestions } => {
            lint_emitter.emit_node_span_lint(
                rustc_session::lint::builtin::ILL_FORMED_ATTRIBUTE_INPUT,
                *id,
                *span,
                session_diagnostics::IllFormedAttributeInput {
                    num_suggestions: suggestions.len(),
                    suggestions: DiagArgValue::StrListSepByAnd(
                        suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                    ),
                },
            );
        }
        AttributeLintKind::EmptyAttribute { first_span } => lint_emitter.emit_node_span_lint(
            rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
            *id,
            *first_span,
            session_diagnostics::EmptyAttributeList { attr_span: *first_span },
        ),
        &AttributeLintKind::UnsafeAttrOutsideUnsafe { attribute_name_span, sugg_spans } => {
            lint_emitter.emit_node_span_lint(
                rustc_session::lint::builtin::UNSAFE_ATTR_OUTSIDE_UNSAFE,
                *id,
                *span,
                UnsafeAttrOutsideUnsafeLint {
                    span: attribute_name_span,
                    suggestion: UnsafeAttrOutsideUnsafeSuggestion {
                        left: sugg_spans.0.shrink_to_lo(),
                        right: sugg_spans.1.shrink_to_hi(),
                    },
                },
            )
        }
    }
}
