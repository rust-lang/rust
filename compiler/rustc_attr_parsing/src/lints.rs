use rustc_errors::{AttributeLintDecorator, DeferedAttributeLintDecorator, DiagArgValue};
use rustc_hir::lints::{AttributeLint, AttributeLintKind};
use rustc_hir::{HirId, Target};
use rustc_session::lint::Lint;
use rustc_span::sym;

use crate::session_diagnostics;

pub(crate) fn lint_name(kind: &AttributeLintKind) -> &'static Lint {
    use rustc_session::lint::builtin::*;
    match kind {
        AttributeLintKind::UnusedDuplicate { .. } => UNUSED_ATTRIBUTES,
        AttributeLintKind::IllFormedAttributeInput { .. } => ILL_FORMED_ATTRIBUTE_INPUT,
        AttributeLintKind::EmptyAttribute { .. } => UNUSED_ATTRIBUTES,
        AttributeLintKind::InvalidTarget { name, target, .. } => {
            // This check is here because `deprecated` had its own lint group and removing this would be a breaking change
            if *name == sym::deprecated
                && ![Target::Closure, Target::Expression, Target::Statement, Target::Arm]
                    .contains(target)
            {
                rustc_session::lint::builtin::USELESS_DEPRECATED
            } else {
                rustc_session::lint::builtin::UNUSED_ATTRIBUTES
            }
        }
    }
}

pub fn decorate_attribute_lint_kind<L: AttributeLintDecorator>(
    kind: &AttributeLintKind,
    lint_emitter: L,
) {
    match kind {
        &AttributeLintKind::UnusedDuplicate { this, other, warning } => {
            lint_emitter.decorate(session_diagnostics::UnusedDuplicate { this, other, warning })
        }
        AttributeLintKind::IllFormedAttributeInput { suggestions } => {
            lint_emitter.decorate(session_diagnostics::IllFormedAttributeInput {
                num_suggestions: suggestions.len(),
                suggestions: DiagArgValue::StrListSepByAnd(
                    suggestions.into_iter().map(|s| format!("`{s}`").into()).collect(),
                ),
            });
        }

        AttributeLintKind::EmptyAttribute { first_span } => lint_emitter
            .decorate(session_diagnostics::EmptyAttributeList { attr_span: *first_span }),
        &AttributeLintKind::InvalidTarget { name, target, ref applied, only } => lint_emitter
            .decorate(session_diagnostics::InvalidTargetLint {
                name,
                target: target.plural_name(),
                applied: applied.clone(),
                only,
            }),
    }
}

pub fn emit_attribute_lint<L: DeferedAttributeLintDecorator<ID = HirId>>(
    lint: &AttributeLint<HirId>,
    lint_emitter: L,
) {
    let AttributeLint { id, span, kind } = lint;

    let lint_name = lint_name(&lint.kind);

    let emit = lint_emitter.prepare(lint_name, *id, *span);
    decorate_attribute_lint_kind(kind, emit);
}
