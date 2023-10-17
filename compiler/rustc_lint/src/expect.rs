use crate::lints::{Expectation, ExpectationNote};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::UNFULFILLED_LINT_EXPECTATIONS;
use rustc_session::lint::LintExpectationId;
use rustc_span::symbol::sym;
use rustc_span::Symbol;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_expectations, ..*providers };
}

fn check_expectations(tcx: TyCtxt<'_>, tool_filter: Option<Symbol>) {
    if !tcx.features().active(sym::lint_reasons) {
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
                let rationale = expectation.reason.map(|rationale| ExpectationNote { rationale });
                let note = expectation.is_unfulfilled_lint_expectations.then_some(());
                tcx.emit_spanned_lint(
                    UNFULFILLED_LINT_EXPECTATIONS,
                    *hir_id,
                    expectation.emission_span,
                    Expectation { rationale, note },
                );
            }
        } else {
            unreachable!("at this stage all `LintExpectationId`s are stable");
        }
    }
}
