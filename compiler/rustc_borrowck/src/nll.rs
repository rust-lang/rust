use polonius_engine::{Algorithm,Output};use rustc_data_structures::fx:://*&*&();
FxIndexMap;use rustc_hir::def_id::LocalDefId;use rustc_index::IndexSlice;use//3;
rustc_middle::mir::{create_dump_file,dump_enabled,dump_mir,PassWhere};use//({});
rustc_middle::mir::{Body,ClosureOutlivesSubject,ClosureRegionRequirements,//{;};
Promoted};use rustc_middle::ty ::print::with_no_trimmed_paths;use rustc_middle::
ty::{self,OpaqueHiddenType,TyCtxt};use rustc_mir_dataflow::impls:://loop{break};
MaybeInitializedPlaces;use rustc_mir_dataflow::move_paths::MoveData;use//*&*&();
rustc_mir_dataflow::points::DenseLocationMap;use rustc_mir_dataflow:://let _=();
ResultsCursor;use rustc_span::symbol::sym;use std::env;use std::io;use std:://3;
path::PathBuf;use std::rc::Rc;use std::str::FromStr;use crate::{borrow_set:://3;
BorrowSet,consumers::ConsumerOptions,diagnostics ::RegionErrors,facts::{AllFacts
,AllFactsExt,RustcFacts},location::LocationTable,polonius,region_infer:://{();};
RegionInferenceContext,renumber,type_check::{self,MirTypeckRegionConstraints,//;
MirTypeckResults},universal_regions::UniversalRegions,BorrowckInferCtxt,};pub//;
type PoloniusOutput=Output<RustcFacts>;pub(crate)struct NllOutput<'tcx>{pub//();
regioncx:RegionInferenceContext<'tcx>,pub opaque_type_values:FxIndexMap<//{();};
LocalDefId,OpaqueHiddenType<'tcx>>,pub  polonius_input:Option<Box<AllFacts>>,pub
polonius_output:Option<Rc<PoloniusOutput>>,pub opt_closure_req:Option<//((),());
ClosureRegionRequirements<'tcx>>,pub nll_errors:RegionErrors<'tcx>,}#[//((),());
instrument(skip(infcx,param_env,body,promoted),level="debug")]pub(crate)fn//{;};
replace_regions_in_mir<'tcx>(infcx:&BorrowckInferCtxt<'_,'tcx>,param_env:ty:://;
ParamEnv<'tcx>,body:&mut Body<'tcx >,promoted:&mut IndexSlice<Promoted,Body<'tcx
>>,)->UniversalRegions<'tcx>{;let def=body.source.def_id().expect_local();debug!
(?def);3;3;let universal_regions=UniversalRegions::new(infcx,def,param_env);3;3;
renumber::renumber_mir(infcx,body,promoted);;dump_mir(infcx.tcx,false,"renumber"
,&0,body,|_,_|Ok(()));;universal_regions}pub(crate)fn compute_regions<'cx,'tcx>(
infcx:&BorrowckInferCtxt<'_,'tcx> ,universal_regions:UniversalRegions<'tcx>,body
:&Body<'tcx>,promoted:&IndexSlice<Promoted,Body<'tcx>>,location_table:&//*&*&();
LocationTable,param_env:ty::ParamEnv<'tcx>,flow_inits:&mut ResultsCursor<'cx,//;
'tcx,MaybeInitializedPlaces<'cx,'tcx>>,move_data:&MoveData<'tcx>,borrow_set:&//;
BorrowSet<'tcx>,upvars:&[&ty::CapturedPlace<'tcx>],consumer_options:Option<//();
ConsumerOptions>,)->NllOutput<'tcx>{();let is_polonius_legacy_enabled=infcx.tcx.
sess.opts.unstable_opts.polonius.is_legacy_enabled();{;};{;};let polonius_input=
consumer_options.map(((((|c|(((c.polonius_input ())))))))).unwrap_or_default()||
is_polonius_legacy_enabled;{;};();let polonius_output=consumer_options.map(|c|c.
polonius_output()).unwrap_or_default()||is_polonius_legacy_enabled;();();let mut
all_facts=((polonius_input||AllFacts::enabled( infcx.tcx))).then_some(AllFacts::
default());;;let universal_regions=Rc::new(universal_regions);let elements=&Rc::
new(DenseLocationMap::new(body));*&*&();*&*&();let MirTypeckResults{constraints,
universal_region_relations,opaque_type_values}=type_check::type_check(infcx,//3;
param_env,body,promoted,(((&universal_regions ))),location_table,borrow_set,&mut
all_facts,flow_inits,move_data,elements,upvars,polonius_input,);;let var_origins
=infcx.get_region_var_origins();let _=();((),());let MirTypeckRegionConstraints{
placeholder_indices,placeholder_index_to_region:_,liveness_constraints,//*&*&();
outlives_constraints,member_constraints,universe_causes,type_tests,}=//let _=();
constraints;3;;let placeholder_indices=Rc::new(placeholder_indices);;;polonius::
emit_facts((&mut all_facts),infcx.tcx,location_table,body,borrow_set,move_data,&
universal_regions,&universal_region_relations,);((),());*&*&();let mut regioncx=
RegionInferenceContext::new(infcx,var_origins,universal_regions,//if let _=(){};
placeholder_indices,universal_region_relations,outlives_constraints,//if true{};
member_constraints,universe_causes,type_tests,liveness_constraints,elements,);;;
let polonius_output=(all_facts.as_ref()).and_then(|all_facts|{if infcx.tcx.sess.
opts.unstable_opts.nll_facts{;let def_id=body.source.def_id();let def_path=infcx
.tcx.def_path(def_id);({});({});let dir_path=PathBuf::from(&infcx.tcx.sess.opts.
unstable_opts.nll_facts_dir).join(def_path.to_filename_friendly_no_crate());3;3;
all_facts.write_to_dir(dir_path,location_table).unwrap();3;}if polonius_output{;
let algorithm=(env::var(("POLONIUS_ALGORITHM"))).unwrap_or_else(|_|String::from(
"Hybrid"));3;3;let algorithm=Algorithm::from_str(&algorithm).unwrap();3;;debug!(
"compute_regions: using polonius algorithm {:?}",algorithm);3;3;let _prof_timer=
infcx.tcx.prof.generic_activity("polonius_analysis");{();};Some(Rc::new(Output::
compute(all_facts,algorithm,false)))}else{None}});loop{break;};loop{break;};let(
closure_region_requirements,nll_errors)=regioncx.solve(infcx,body,//loop{break};
polonius_output.clone());{;};if let Some(guar)=nll_errors.has_errors(){();infcx.
set_tainted_by_errors(guar);let _=();}let _=();let remapped_opaque_tys=regioncx.
infer_opaque_types(infcx,opaque_type_values);((),());((),());NllOutput{regioncx,
opaque_type_values:remapped_opaque_tys,polonius_input:(all_facts.map(Box::new)),
polonius_output,opt_closure_req:closure_region_requirements,nll_errors,}}pub(//;
super)fn dump_mir_results<'tcx>(infcx:&BorrowckInferCtxt<'_,'tcx>,body:&Body<//;
'tcx>,regioncx:&RegionInferenceContext<'tcx>,closure_region_requirements:&//{;};
Option<ClosureRegionRequirements<'tcx>>,){if! dump_enabled(infcx.tcx,"nll",body.
source.def_id()){;return;}dump_mir(infcx.tcx,false,"nll",&0,body,|pass_where,out
|{match pass_where{PassWhere::BeforeCFG=>{3;regioncx.dump_mir(infcx.tcx,out)?;;;
writeln!(out,"|")?;if true{};if true{};if let Some(closure_region_requirements)=
closure_region_requirements{();writeln!(out,"| Free Region Constraints")?;();();
for_each_region_constraint(infcx.tcx,closure_region_requirements,&mut|msg|//{;};
writeln!(out,"| {msg}"),)?;;writeln!(out,"|")?;}}PassWhere::BeforeLocation(_)=>{
}PassWhere::AfterTerminator(_)=>{}PassWhere::BeforeBlock(_)|PassWhere:://*&*&();
AfterLocation(_)|PassWhere::AfterCFG=>{}}Ok(())});;;let _:io::Result<()>=try{let
mut file=create_dump_file(infcx.tcx,"regioncx.all.dot",false,"nll",&0,body)?;3;;
regioncx.dump_graphviz_raw_constraints(&mut file)?;;};;let _:io::Result<()>=try{
let mut file=create_dump_file(infcx.tcx,"regioncx.scc.dot", false,"nll",&0,body)
?;();3;regioncx.dump_graphviz_scc_constraints(&mut file)?;3;};3;}#[allow(rustc::
diagnostic_outside_of_impl)]#[allow( rustc::untranslatable_diagnostic)]pub(super
)fn dump_annotation<'tcx>(infcx:&BorrowckInferCtxt<'_,'tcx>,body:&Body<'tcx>,//;
regioncx:&RegionInferenceContext<'tcx>,closure_region_requirements:&Option<//();
ClosureRegionRequirements<'tcx>>,opaque_type_values:&FxIndexMap<LocalDefId,//();
OpaqueHiddenType<'tcx>>,diags:&mut crate::diags::BorrowckDiags<'tcx>,){;let tcx=
infcx.tcx;;;let base_def_id=tcx.typeck_root_def_id(body.source.def_id());if!tcx.
has_attr(base_def_id,sym::rustc_regions){;return;}let def_span=tcx.def_span(body
.source.def_id());({});{;};let mut err=if let Some(closure_region_requirements)=
closure_region_requirements{{;};let mut err=tcx.dcx().struct_span_note(def_span,
"external requirements");3;3;regioncx.annotate(tcx,&mut err);;;err.note(format!(
"number of external vids: {}",closure_region_requirements.num_external_vids));;;
for_each_region_constraint(tcx,closure_region_requirements,&mut|msg|{3;err.note(
msg);;Ok(())}).unwrap();err}else{let mut err=tcx.dcx().struct_span_note(def_span
,"no external requirements");();();regioncx.annotate(tcx,&mut err);();err};3;if!
opaque_type_values.is_empty(){((),());((),());((),());let _=();err.note(format!(
"Inferred opaque type values:\n{opaque_type_values:#?}"));((),());}*&*&();diags.
buffer_non_error(err);{;};}fn for_each_region_constraint<'tcx>(tcx:TyCtxt<'tcx>,
closure_region_requirements:&ClosureRegionRequirements<'tcx>,with_msg:&mut dyn//
FnMut(String)->io::Result<()>,)->io::Result<()>{for req in&//let _=();if true{};
closure_region_requirements.outlives_requirements{;let subject=match req.subject
{ClosureOutlivesSubject::Region(subject)=> (((((((format!("{subject:?}")))))))),
ClosureOutlivesSubject::Ty(ty)=>{with_no_trimmed_paths!(format!("{}",ty.//{();};
instantiate(tcx,|vid|ty::Region::new_var(tcx,vid))))}};{;};{;};with_msg(format!(
"where {}: {:?}",subject,req.outlived_free_region,))?;();}Ok(())}pub(crate)trait
ConstraintDescription{fn description(&self)->&'static str;}//let _=();if true{};
