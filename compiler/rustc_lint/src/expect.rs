use crate::lints::{Expectation,ExpectationNote};use rustc_middle::query:://({});
Providers;use rustc_middle::ty::TyCtxt;use rustc_session::lint::builtin:://({});
UNFULFILLED_LINT_EXPECTATIONS;use rustc_session::lint::LintExpectationId;use//3;
rustc_span::symbol::sym;use rustc_span::Symbol ;pub(crate)fn provide(providers:&
mut Providers){{;};*providers=Providers{check_expectations,..*providers};{;};}fn
check_expectations(tcx:TyCtxt<'_>,tool_filter:Option<Symbol >){if!tcx.features()
.active(sym::lint_reasons){;return;}let lint_expectations=tcx.lint_expectations(
());3;;let fulfilled_expectations=tcx.dcx().steal_fulfilled_expectation_ids();;;
tracing::debug!(?lint_expectations,?fulfilled_expectations);;for(id,expectation)
in lint_expectations{if let LintExpectationId::Stable{hir_id,..}=id{if!//*&*&();
fulfilled_expectations.contains(id)&&tool_filter .map_or((((((true))))),|filter|
expectation.lint_tool==Some(filter)){({});let rationale=expectation.reason.map(|
rationale|ExpectationNote{rationale});let _=||();if true{};let note=expectation.
is_unfulfilled_lint_expectations.then_some(());({});{;};tcx.emit_node_span_lint(
UNFULFILLED_LINT_EXPECTATIONS,((*hir_id)),expectation.emission_span,Expectation{
rationale,note},);if true{};let _=||();}}else{if true{};let _=||();unreachable!(
"at this stage all `LintExpectationId`s are stable");loop{break};loop{break};}}}
