use crate::builtin;
use rustc_errors::fluent;
use rustc_hir::HirId;
use rustc_middle::ty::query::Providers;
use rustc_middle::{lint::LintExpectation, ty::TyCtxt};
use rustc_session::lint::LintExpectationId;
use rustc_span::symbol::sym;
use rustc_span::Symbol;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_expectations, ..*providers };
}

fn check_expectations(tcx: TyCtxt<'_>, tool_filter: Option<Symbol>) {
    if !tcx.sess.features_untracked().enabled(sym::lint_reasons) {
        return;
    }

    let lint_expectations = tcx.lint_expectations(());
    let fulfilled_expectations = tcx.sess.diagnostic().steal_fulfilled_expectation_ids();

    tracing::debug!(?lint_expectations, ?fulfilled_expectations);

    for (id, expectation) in lint_expectations {
        // This check will always be true, since `lint_expectations` only
        // holds stable ids
        if let LintExpectationId::Stable { hir_id, .. } = id {
            if !fulfilled_expectations.contains(&id)
                && tool_filter.map_or(true, |filter| expectation.lint_tool == Some(filter))
            {
                emit_unfulfilled_expectation_lint(tcx, *hir_id, expectation);
            }
        } else {
            unreachable!("at this stage all `LintExpectationId`s are stable");
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
            let mut diag = diag.build(fluent::lint::expectation);
            if let Some(rationale) = expectation.reason {
                diag.note(rationale.as_str());
            }

            if expectation.is_unfulfilled_lint_expectations {
                diag.note(fluent::lint::note);
            }

            diag.emit();
        },
    );
}
