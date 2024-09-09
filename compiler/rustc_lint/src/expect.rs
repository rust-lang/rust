use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{HirId, CRATE_OWNER_ID};
use rustc_middle::lint::LintExpectation;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::UNFULFILLED_LINT_EXPECTATIONS;
use rustc_session::lint::{Level, LintExpectationId};
use rustc_span::Symbol;

use crate::lints::{Expectation, ExpectationNote};

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { lint_expectations, check_expectations, ..*providers };
}

fn lint_expectations(tcx: TyCtxt<'_>, (): ()) -> Vec<(LintExpectationId, LintExpectation)> {
    let krate = tcx.hir_crate_items(());

    let mut expectations = Vec::new();
    let mut unstable_to_stable_ids = FxIndexMap::default();

    let mut record_stable = |attr_id, hir_id, attr_index| {
        let expect_id = LintExpectationId::Stable { hir_id, attr_index, lint_index: None };
        unstable_to_stable_ids.entry(attr_id).or_insert(expect_id);
    };
    let mut push_expectations = |owner| {
        let lints = tcx.shallow_lint_levels_on(owner);
        if lints.expectations.is_empty() {
            return;
        }

        expectations.extend_from_slice(&lints.expectations);

        let attrs = tcx.hir_attrs(owner);
        for &(local_id, attrs) in attrs.map.iter() {
            // Some attributes appear multiple times in HIR, to ensure they are correctly taken
            // into account where they matter. This means we cannot just associate the AttrId to
            // the first HirId where we see it, but need to check it actually appears in a lint
            // level.
            // FIXME(cjgillot): Can this cause an attribute to appear in multiple expectation ids?
            if !lints.specs.contains_key(&local_id) {
                continue;
            }
            for (attr_index, attr) in attrs.iter().enumerate() {
                let Some(Level::Expect(_)) = Level::from_attr(attr) else { continue };
                record_stable(attr.id, HirId { owner, local_id }, attr_index.try_into().unwrap());
            }
        }
    };

    push_expectations(CRATE_OWNER_ID);
    for owner in krate.owners() {
        push_expectations(owner);
    }

    tcx.dcx().update_unstable_expectation_id(unstable_to_stable_ids);
    expectations
}

fn check_expectations(tcx: TyCtxt<'_>, tool_filter: Option<Symbol>) {
    let lint_expectations = tcx.lint_expectations(());
    let fulfilled_expectations = tcx.dcx().steal_fulfilled_expectation_ids();

    for (id, expectation) in lint_expectations {
        // This check will always be true, since `lint_expectations` only
        // holds stable ids
        if let LintExpectationId::Stable { hir_id, .. } = id {
            if !fulfilled_expectations.contains(id)
                && tool_filter.map_or(true, |filter| expectation.lint_tool == Some(filter))
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
        } else {
            unreachable!("at this stage all `LintExpectationId`s are stable");
        }
    }
}
