use crate::builtin;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::lint::struct_lint_level;
use rustc_middle::{lint::LintExpectation, ty::TyCtxt};
use rustc_session::lint::LintExpectationId;
use rustc_span::symbol::sym;
use rustc_span::MultiSpan;

pub fn check_expectations(tcx: TyCtxt<'_>) {
    if !tcx.sess.features_untracked().enabled(sym::lint_reasons) {
        return;
    }

    let fulfilled_expectations = tcx.sess.diagnostic().steal_fulfilled_expectation_ids();
    let lint_expectations: &FxHashMap<LintExpectationId, LintExpectation> =
        &tcx.lint_levels(()).lint_expectations;

    for (id, expectation) in lint_expectations {
        if fulfilled_expectations.contains(id) {
            continue;
        }

        emit_unfulfilled_expectation_lint(tcx, expectation);
    }
}

fn emit_unfulfilled_expectation_lint(tcx: TyCtxt<'_>, expectation: &LintExpectation) {
    // FIXME  The current implementation doesn't cover cases where the
    // `unfulfilled_lint_expectations` is actually expected by another lint
    // expectation. This can be added here as we have the lint level of this
    // expectation, and we can also mark the lint expectation it would fulfill
    // as such. This is currently not implemented to get some early feedback
    // before diving deeper into this.
    struct_lint_level(
        tcx.sess,
        builtin::UNFULFILLED_LINT_EXPECTATIONS,
        expectation.emission_level,
        expectation.emission_level_source,
        Some(MultiSpan::from_span(expectation.emission_span)),
        |diag| {
            let mut diag = diag.build("this lint expectation is unfulfilled");
            if let Some(rationale) = expectation.reason {
                diag.note(&rationale.as_str());
            }
            diag.emit();
        },
    );
}
