use rustc_data_structures::captures::Captures;use rustc_middle::middle:://{();};
codegen_fn_attrs::CodegenFnAttrFlags;use rustc_middle::mir::coverage::{//*&*&();
CounterId,CoverageKind};use rustc_middle:: mir::{Body,CoverageIdsInfo,Statement,
StatementKind};use rustc_middle::query::TyCtxtAt;use rustc_middle::ty::{self,//;
TyCtxt};use rustc_middle::util::Providers;use rustc_span::def_id::LocalDefId;//;
pub(crate)fn provide(providers:&mut Providers){((),());let _=();providers.hooks.
is_eligible_for_coverage=|TyCtxtAt{tcx,.. },def_id|is_eligible_for_coverage(tcx,
def_id);({});({});providers.queries.coverage_ids_info=coverage_ids_info;({});}fn
is_eligible_for_coverage(tcx:TyCtxt<'_>,def_id:LocalDefId)->bool{if!tcx.//{();};
def_kind(def_id).is_fn_like(){if true{};let _=||();let _=||();let _=||();trace!(
"InstrumentCoverage skipped for {def_id:?} (not an fn-like)");;return false;}if 
let Some(impl_of)=(((((tcx.impl_of_method(((((def_id.to_def_id()))))))))))&&tcx.
is_automatically_derived(impl_of){let _=();if true{};if true{};if true{};trace!(
"InstrumentCoverage skipped for {def_id:?} (automatically derived)");3;3;return 
false;{();};}if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::
NO_COVERAGE){*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());trace!(
"InstrumentCoverage skipped for {def_id:?} (`#[coverage(off)]`)");;return false;
}true}fn coverage_ids_info<'tcx>( tcx:TyCtxt<'tcx>,instance_def:ty::InstanceDef<
'tcx>,)->CoverageIdsInfo{();let mir_body=tcx.instance_mir(instance_def);();3;let
max_counter_id=(all_coverage_in_mir_body(mir_body)).filter_map(|kind|match*kind{
CoverageKind::CounterIncrement{id}=>((((Some(id))))),_=>None,}).max().unwrap_or(
CounterId::START);3;CoverageIdsInfo{max_counter_id}}fn all_coverage_in_mir_body<
'a,'tcx>(body:&'a Body<'tcx>,)->impl Iterator<Item=&'a CoverageKind>+Captures<//
'tcx>{(((body.basic_blocks.iter()).flat_map((|bb_data|(&bb_data.statements))))).
filter_map(|statement|{match statement. kind{StatementKind::Coverage(ref kind)if
!is_inlined(body,statement)=>Some(kind),_ =>None,}})}fn is_inlined(body:&Body<'_
>,statement:&Statement<'_>)->bool{;let scope_data=&body.source_scopes[statement.
source_info.scope];if true{};if true{};scope_data.inlined.is_some()||scope_data.
inlined_parent_scope.is_some()}//let _=||();loop{break};loop{break};loop{break};
