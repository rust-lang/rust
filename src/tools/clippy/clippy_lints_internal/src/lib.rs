#![feature(let_chains, rustc_private)]
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::must_use_candidate,
    rustc::diagnostic_outside_of_impl,
    rustc::untranslatable_diagnostic
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications,
    rustc::internal
)]
// Disable this rustc lint for now, as it was also done in rustc
#![allow(rustc::potential_query_instability)]
// None of these lints need a version.
#![allow(clippy::missing_clippy_version_attribute)]

extern crate rustc_ast;
extern crate rustc_attr_parsing;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_lint_defs;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod almost_standard_lint_formulation;
mod collapsible_calls;
mod interning_literals;
mod invalid_paths;
mod lint_without_lint_pass;
mod msrv_attr_impl;
mod outer_expn_data_pass;
mod produce_ice;
mod unnecessary_def_path;
mod unsorted_clippy_utils_paths;

use rustc_lint::{Lint, LintStore};

static LINTS: &[&Lint] = &[
    almost_standard_lint_formulation::ALMOST_STANDARD_LINT_FORMULATION,
    collapsible_calls::COLLAPSIBLE_SPAN_LINT_CALLS,
    interning_literals::INTERNING_LITERALS,
    invalid_paths::INVALID_PATHS,
    lint_without_lint_pass::DEFAULT_LINT,
    lint_without_lint_pass::INVALID_CLIPPY_VERSION_ATTRIBUTE,
    lint_without_lint_pass::LINT_WITHOUT_LINT_PASS,
    lint_without_lint_pass::MISSING_CLIPPY_VERSION_ATTRIBUTE,
    msrv_attr_impl::MISSING_MSRV_ATTR_IMPL,
    outer_expn_data_pass::OUTER_EXPN_EXPN_DATA,
    produce_ice::PRODUCE_ICE,
    unnecessary_def_path::UNNECESSARY_DEF_PATH,
    unsorted_clippy_utils_paths::UNSORTED_CLIPPY_UTILS_PATHS,
];

pub fn register_lints(store: &mut LintStore) {
    store.register_lints(LINTS);

    store.register_early_pass(|| Box::new(unsorted_clippy_utils_paths::UnsortedClippyUtilsPaths));
    store.register_early_pass(|| Box::new(produce_ice::ProduceIce));
    store.register_late_pass(|_| Box::new(collapsible_calls::CollapsibleCalls));
    store.register_late_pass(|_| Box::new(invalid_paths::InvalidPaths));
    store.register_late_pass(|_| Box::<interning_literals::InterningDefinedSymbol>::default());
    store.register_late_pass(|_| Box::<lint_without_lint_pass::LintWithoutLintPass>::default());
    store.register_late_pass(|_| Box::<unnecessary_def_path::UnnecessaryDefPath>::default());
    store.register_late_pass(|_| Box::new(outer_expn_data_pass::OuterExpnDataPass));
    store.register_late_pass(|_| Box::new(msrv_attr_impl::MsrvAttrImpl));
    store.register_late_pass(|_| Box::new(almost_standard_lint_formulation::AlmostStandardFormulation::new()));
}
