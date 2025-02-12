use rustc_attr_data_structures::lints::{AttributeLint, AttributeLintKind};
use rustc_errors::LintEmitter;
use rustc_hir::HirId;

use crate::session_diagnostics;

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
    }
}
