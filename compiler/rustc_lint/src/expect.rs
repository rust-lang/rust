use crate::builtin;
use rustc_hir::HirId;
use rustc_middle::{lint::LintExpectation, ty::TyCtxt};
use rustc_session::lint::LintExpectationId;
use rustc_span::symbol::sym;

pub fn check_expectations(tcx: TyCtxt<'_>) {
    if !tcx.sess.features_untracked().enabled(sym::lint_reasons) {
        return;
    }

    let fulfilled_expectations = tcx.sess.diagnostic().steal_fulfilled_expectation_ids();
    let lint_expectations = &tcx.lint_levels(()).lint_expectations;

    for (id, expectation) in lint_expectations {
        if !fulfilled_expectations.contains(id) {
            // This check will always be true, since `lint_expectations` only
            // holds stable ids
            if let LintExpectationId::Stable { hir_id, .. } = id {
                emit_unfulfilled_expectation_lint(tcx, *hir_id, expectation);
            } else {
                unreachable!("at this stage all `LintExpectationId`s are stable");
            }
        }
    }
}

fn emit_unfulfilled_expectation_lint(
    tcx: TyCtxt<'_>,
    hir_id: HirId,
    expectation: &LintExpectation,
) {
    tcx.struct_span_lint_hir(
        builtin::UNFULFILLED_LINT_EXPECTATIONS,
        hir_id,
        expectation.emission_span,
        |diag| {
            let mut diag = diag.build("this lint expectation is unfulfilled");
            if let Some(rationale) = expectation.reason {
                diag.note(&rationale.as_str());
            }

            if expectation.is_unfulfilled_lint_expectations {
                diag.note("the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message");
            }

            diag.emit();
        },
    );
}
