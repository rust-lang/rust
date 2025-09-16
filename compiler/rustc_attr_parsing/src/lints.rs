use std::borrow::Cow;

use rustc_errors::{DiagArgValue, LintEmitter};
use rustc_hir::Target;
use rustc_hir::lints::{AttributeLint, AttributeLintKind};
use rustc_span::sym;

use crate::session_diagnostics;

pub fn emit_attribute_lint<L: LintEmitter>(lint: &AttributeLint<L::Id>, lint_emitter: L) {
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
        AttributeLintKind::InvalidMacroExportArguments { suggestions } => lint_emitter
            .emit_node_span_lint(
                rustc_session::lint::builtin::INVALID_MACRO_EXPORT_ARGUMENTS,
                *id,
                *span,
                session_diagnostics::IllFormedAttributeInput {
                    num_suggestions: suggestions.len(),
                    suggestions: DiagArgValue::StrListSepByAnd(
                        suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                    ),
                },
            ),
        AttributeLintKind::EmptyAttribute { first_span, attr_path, valid_without_list } => {
            lint_emitter.emit_node_span_lint(
                rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
                *id,
                *first_span,
                session_diagnostics::EmptyAttributeList {
                    attr_span: *first_span,
                    attr_path: attr_path.clone(),
                    valid_without_list: *valid_without_list,
                },
            )
        }
        AttributeLintKind::InvalidTarget { name, target, applied, only } => lint_emitter
            .emit_node_span_lint(
                // This check is here because `deprecated` had its own lint group and removing this would be a breaking change
                if name.segments[0].name == sym::deprecated
                    && ![
                        Target::Closure,
                        Target::Expression,
                        Target::Statement,
                        Target::Arm,
                        Target::MacroCall,
                    ]
                    .contains(target)
                {
                    rustc_session::lint::builtin::USELESS_DEPRECATED
                } else {
                    rustc_session::lint::builtin::UNUSED_ATTRIBUTES
                },
                *id,
                *span,
                session_diagnostics::InvalidTargetLint {
                    name: name.clone(),
                    target: target.plural_name(),
                    applied: DiagArgValue::StrListSepByAnd(
                        applied.into_iter().map(|i| Cow::Owned(i.to_string())).collect(),
                    ),
                    only,
                    attr_span: *span,
                },
            ),

        &AttributeLintKind::InvalidStyle { ref name, is_used_as_inner, target, target_span } => {
            lint_emitter.emit_node_span_lint(
                rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
                *id,
                *span,
                session_diagnostics::InvalidAttrStyle {
                    name: name.clone(),
                    is_used_as_inner,
                    target_span: (!is_used_as_inner).then_some(target_span),
                    target,
                },
            )
        }
    }
}
