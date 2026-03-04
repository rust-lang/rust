use rustc_data_structures::fx::FxHashSet;
use rustc_middle::lint::LintExpectation;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::LintExpectationId;
use rustc_session::lint::builtin::UNFULFILLED_LINT_EXPECTATIONS;
use rustc_span::Symbol;

use crate::lints::{Expectation, ExpectationNote};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { lint_expectations, check_expectations, ..*providers };
}

fn lint_expectations(tcx: TyCtxt<'_>, (): ()) -> Vec<(LintExpectationId, LintExpectation)> {
    let krate = tcx.hir_crate_items(());

    let mut expectations = Vec::new();

    for owner in krate.owners() {
        // Deduplicate expectations
        let mut inner_expectations = Vec::new();
        let lints = tcx.shallow_lint_levels_on(owner);
        for expectation in &lints.expectations {
            let canonicalized = canonicalize_id(&expectation.0);
            if !inner_expectations.iter().any(|(id, _)| canonicalize_id(id) == canonicalized) {
                inner_expectations.push(expectation.clone());
            }
        }
        expectations.extend(inner_expectations);
    }

    expectations
}

fn canonicalize_id(expect_id: &LintExpectationId) -> (rustc_span::AttrId, u16) {
    match *expect_id {
        LintExpectationId::Unstable { attr_id, lint_index, .. } => (attr_id, lint_index),
        LintExpectationId::Stable { attr_id, lint_index, .. } => (attr_id, lint_index),
    }
}

fn check_expectations(tcx: TyCtxt<'_>, tool_filter: Option<Symbol>) {
    let lint_expectations = tcx.lint_expectations(());
    let fulfilled_expectations = tcx.dcx().steal_fulfilled_expectation_ids();

    // Turn a `LintExpectationId` into a `(AttrId, lint_index)` pair.
    let fulfilled_expectations: FxHashSet<_> =
        fulfilled_expectations.iter().map(canonicalize_id).collect();

    for (expect_id, expectation) in lint_expectations {
        // This check will always be true, since `lint_expectations` only holds stable ids
        let LintExpectationId::Stable { hir_id, .. } = expect_id else {
            unreachable!("at this stage all `LintExpectationId`s are stable");
        };

        let expect_id = canonicalize_id(expect_id);

        if !fulfilled_expectations.contains(&expect_id)
            && tool_filter.is_none_or(|filter| expectation.lint_tool == Some(filter))
        {
            let rationale = expectation.reason.map(|rationale| ExpectationNote { rationale });
            let note = expectation.is_unfulfilled_lint_expectations;
            tcx.emit_node_span_lint(
                UNFULFILLED_LINT_EXPECTATIONS,
                *hir_id,
                expectation.emission_span,
                Expectation { rationale, note },
            );
        }
    }
}
