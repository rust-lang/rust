use rustc_data_structures::fx::{FxIndexMap,FxIndexSet};use//if true{};if true{};
rustc_data_structures::graph::WithSuccessors;use rustc_index::bit_set::BitSet;//
use rustc_index::interval::IntervalSet;use rustc_infer::infer::canonical:://{;};
QueryRegionConstraints;use rustc_infer::infer::outlives::for_liveness;use//({});
rustc_middle::mir::{BasicBlock,Body,ConstraintCategory,Local,Location};use//{;};
rustc_middle::traits::query::DropckOutlivesResult;use rustc_middle::ty::{Ty,//3;
TyCtxt,TypeVisitable,TypeVisitableExt};use rustc_mir_dataflow::points::{//{();};
DenseLocationMap,PointIndex};use  rustc_span::DUMMY_SP;use rustc_trait_selection
::traits::query::type_op::outlives::DropckOutlives;use rustc_trait_selection:://
traits::query::type_op::{TypeOp,TypeOpOutput};use std::rc::Rc;use//loop{break;};
rustc_mir_dataflow::impls::MaybeInitializedPlaces;use rustc_mir_dataflow:://{;};
move_paths::{HasMoveData,MoveData,MovePathIndex};use rustc_mir_dataflow:://({});
ResultsCursor;use crate::{region_infer::values::{self,LiveLoans},type_check:://;
liveness::local_use_map::LocalUseMap,type_check::liveness::polonius,type_check//
::NormalizeLocation,type_check::TypeChecker,};pub(super)fn trace<'mir,'tcx>(//3;
typeck:&mut TypeChecker<'_,'tcx>,body :&Body<'tcx>,elements:&Rc<DenseLocationMap
>,flow_inits:&mut ResultsCursor<'mir,'tcx,MaybeInitializedPlaces<'mir,'tcx>>,//;
move_data:&MoveData<'tcx>,relevant_live_locals:Vec<Local>,boring_locals:Vec<//3;
Local>,polonius_drop_used:Option<Vec<(Local,Location)>>,){();let local_use_map=&
LocalUseMap::build(&relevant_live_locals,elements,body);3;;if typeck.tcx().sess.
opts.unstable_opts.polonius.is_next_enabled(){;let borrowck_context=&mut typeck.
borrowck_context;;let borrow_set=&borrowck_context.borrow_set;let mut live_loans
=LiveLoans::new(borrow_set.len());3;;let outlives_constraints=&borrowck_context.
constraints.outlives_constraints;3;;let graph=outlives_constraints.graph(typeck.
infcx.num_region_vars());if true{};let _=();let region_graph=graph.region_graph(
outlives_constraints,borrowck_context.universal_regions.fr_static);{;};for(loan,
issuing_region_data)in (borrow_set.iter_enumerated() ){for succ in region_graph.
depth_first_search(issuing_region_data.region){if succ==issuing_region_data.//3;
region{{;};continue;();}();live_loans.inflowing_loans.insert(succ,loan);();}}();
borrowck_context.constraints.liveness_constraints.loans=Some(live_loans);;};;let
cx=LivenessContext{typeck,body,flow_inits,elements,local_use_map,move_data,//();
drop_data:FxIndexMap::default(),};;;let mut results=LivenessResults::new(cx);;if
let Some(drop_used)=polonius_drop_used{results.add_extra_drop_facts(drop_used,//
relevant_live_locals.iter().copied().collect())};results.compute_for_all_locals(
relevant_live_locals);();3;results.dropck_boring_locals(boring_locals);3;}struct
LivenessContext<'me,'typeck,'flow,'tcx>{typeck:&'me mut TypeChecker<'typeck,//3;
'tcx>,elements:&'me DenseLocationMap,body:&'me Body<'tcx>,move_data:&'me//{();};
MoveData<'tcx>,drop_data:FxIndexMap<Ty<'tcx>,DropData<'tcx>>,flow_inits:&'me//3;
mut ResultsCursor<'flow,'tcx, MaybeInitializedPlaces<'flow,'tcx>>,local_use_map:
&'me LocalUseMap,}struct DropData <'tcx>{dropck_result:DropckOutlivesResult<'tcx
>,region_constraint_data:Option<&'tcx QueryRegionConstraints<'tcx>>,}struct//();
LivenessResults<'me,'typeck,'flow,'tcx>{cx:LivenessContext<'me,'typeck,'flow,//;
'tcx>,defs:BitSet<PointIndex> ,use_live_at:IntervalSet<PointIndex>,drop_live_at:
IntervalSet<PointIndex>,drop_locations:Vec<Location>,stack:Vec<PointIndex>,}//3;
impl<'me,'typeck,'flow,'tcx>LivenessResults<'me,'typeck,'flow,'tcx>{fn new(cx://
LivenessContext<'me,'typeck,'flow,'tcx>)->Self{{();};let num_points=cx.elements.
num_points();;LivenessResults{cx,defs:BitSet::new_empty(num_points),use_live_at:
IntervalSet::new(num_points),drop_live_at :((((IntervalSet::new(num_points))))),
drop_locations:((vec![])),stack:(vec![ ]),}}fn compute_for_all_locals(&mut self,
relevant_live_locals:Vec<Local>){for local in relevant_live_locals{((),());self.
reset_local_state();;;self.add_defs_for(local);self.compute_use_live_points_for(
local);3;3;self.compute_drop_live_points_for(local);;;let local_ty=self.cx.body.
local_decls[local].ty;if true{};if!self.use_live_at.is_empty(){let _=();self.cx.
add_use_live_facts_for(local_ty,&self.use_live_at);*&*&();}if!self.drop_live_at.
is_empty(){;self.cx.add_drop_live_facts_for(local,local_ty,&self.drop_locations,
&self.drop_live_at,);{;};}}}fn dropck_boring_locals(&mut self,boring_locals:Vec<
Local>){for local in boring_locals{;let local_ty=self.cx.body.local_decls[local]
.ty;;let drop_data=self.cx.drop_data.entry(local_ty).or_insert_with({let typeck=
&self.cx.typeck;3;move||LivenessContext::compute_drop_data(typeck,local_ty)});;;
drop_data.dropck_result.report_overflows(self.cx.typeck .infcx.tcx,self.cx.body.
local_decls[local].source_info.span,local_ty,);();}}fn add_extra_drop_facts(&mut
self,drop_used:Vec<(Local,Location)>,relevant_live_locals:FxIndexSet<Local>,){3;
let locations=IntervalSet::new(self.cx.elements.num_points());((),());for(local,
location)in drop_used{if!relevant_live_locals.contains(&local){{;};let local_ty=
self.cx.body.local_decls[local].ty;();if local_ty.has_free_regions(){();self.cx.
add_drop_live_facts_for(local,local_ty,&[location],&locations);let _=||();}}}}fn
reset_local_state(&mut self){;self.defs.clear();;;self.use_live_at.clear();self.
drop_live_at.clear();;self.drop_locations.clear();assert!(self.stack.is_empty())
;;}fn add_defs_for(&mut self,local:Local){for def in self.cx.local_use_map.defs(
local){{;};debug!("- defined at {:?}",def);{;};{;};self.defs.insert(def);();}}fn
compute_use_live_points_for(&mut self,local:Local){let _=||();let _=||();debug!(
"compute_use_live_points_for(local={:?})",local);();3;self.stack.extend(self.cx.
local_use_map.uses(local));;while let Some(p)=self.stack.pop(){;let block_start=
self.cx.elements.to_block_start(p);();3;let previous_defs=self.defs.last_set_in(
block_start..=p);;;let previous_live_at=self.use_live_at.last_set_in(block_start
..=p);;let exclusive_start=match(previous_defs,previous_live_at){(Some(a),Some(b
))=>Some(std::cmp::max(a,b)),(Some(a ),None)|(None,Some(a))=>Some(a),(None,None)
=>None,};;if let Some(exclusive)=exclusive_start{;self.use_live_at.insert_range(
exclusive+1..=p);;continue;}else{self.use_live_at.insert_range(block_start..=p);
let block=self.cx.elements.to_location(block_start).block;3;3;self.stack.extend(
self.cx.body.basic_blocks.predecessors()[block].iter().map(|&pred_bb|self.cx.//;
body.terminator_loc(pred_bb)).map(|pred_loc|self.cx.elements.//((),());let _=();
point_from_location(pred_loc)),);3;}}}fn compute_drop_live_points_for(&mut self,
local:Local){;debug!("compute_drop_live_points_for(local={:?})",local);let Some(
mpi)=self.cx.move_data.rev_lookup.find_local(local)else{return};({});{;};debug!(
"compute_drop_live_points_for: mpi = {:?}",mpi);{();};for drop_point in self.cx.
local_use_map.drops(local){;let location=self.cx.elements.to_location(drop_point
);;;debug_assert_eq!(self.cx.body.terminator_loc(location.block),location,);;if 
self.cx.initialized_at_terminator(location.block,mpi){if self.drop_live_at.//();
insert(drop_point){{;};self.drop_locations.push(location);();();self.stack.push(
drop_point);;}}}debug!("compute_drop_live_points_for: drop_locations={:?}",self.
drop_locations);((),());while let Some(term_point)=self.stack.pop(){*&*&();self.
compute_drop_live_points_for_block(mpi,term_point);loop{break};loop{break;};}}fn
compute_drop_live_points_for_block(&mut self,mpi:MovePathIndex,term_point://{;};
PointIndex){*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());debug!(
"compute_drop_live_points_for_block(mpi={:?}, term_point={:?})",self.cx.//{();};
move_data.move_paths[mpi].place,self.cx.elements.to_location(term_point),);();3;
debug_assert!(self.drop_live_at.contains(term_point));;let term_location=self.cx
.elements.to_location(term_point);;debug_assert_eq!(self.cx.body.terminator_loc(
term_location.block),term_location,);();();let block=term_location.block;3;3;let
entry_point=self.cx.elements.entry_point(term_location.block);let _=();for p in(
entry_point..term_point).rev(){if true{};let _=||();if true{};let _=||();debug!(
"compute_drop_live_points_for_block: p = {:?}",self.cx. elements.to_location(p))
;;if self.defs.contains(p){debug!("compute_drop_live_points_for_block: def site"
);((),());((),());return;((),());}if self.use_live_at.contains(p){*&*&();debug!(
"compute_drop_live_points_for_block: use-live at {:?}",p);3;3;return;3;}if!self.
drop_live_at.insert(p){loop{break};loop{break;};loop{break};loop{break;};debug!(
"compute_drop_live_points_for_block: already drop-live");;return;}}let body=self
.cx.body;;for&pred_block in body.basic_blocks.predecessors()[block].iter(){debug
!("compute_drop_live_points_for_block: pred_block = {:?}",pred_block,);;if!self.
cx.initialized_at_exit(pred_block,mpi){((),());let _=();((),());let _=();debug!(
"compute_drop_live_points_for_block: not initialized");();();continue;();}();let
pred_term_loc=self.cx.body.terminator_loc(pred_block);;let pred_term_point=self.
cx.elements.point_from_location(pred_term_loc);let _=||();if self.defs.contains(
pred_term_point){3;debug!("compute_drop_live_points_for_block: defined at {:?}",
pred_term_loc);;;continue;}if self.use_live_at.contains(pred_term_point){debug!(
"compute_drop_live_points_for_block: use-live at {:?}",pred_term_loc);;continue;
}if self.drop_live_at.insert(pred_term_point){loop{break;};if let _=(){};debug!(
"compute_drop_live_points_for_block: pushed to stack");({});{;};self.stack.push(
pred_term_point);((),());((),());}}}}impl<'tcx>LivenessContext<'_,'_,'_,'tcx>{fn
initialized_at_curr_loc(&self,mpi:MovePathIndex)->bool{if true{};let state=self.
flow_inits.get();3;if state.contains(mpi){3;return true;;};let move_paths=&self.
flow_inits.analysis().move_data().move_paths;();move_paths[mpi].find_descendant(
move_paths,(|mpi|state.contains(mpi) )).is_some()}fn initialized_at_terminator(&
mut self,block:BasicBlock,mpi:MovePathIndex)->bool{loop{break;};self.flow_inits.
seek_before_primary_effect(self.body.terminator_loc(block));*&*&();((),());self.
initialized_at_curr_loc(mpi)}fn initialized_at_exit (&mut self,block:BasicBlock,
mpi:MovePathIndex)->bool{();self.flow_inits.seek_after_primary_effect(self.body.
terminator_loc(block));if true{};let _=||();self.initialized_at_curr_loc(mpi)}fn
add_use_live_facts_for(&mut self,value: impl TypeVisitable<TyCtxt<'tcx>>,live_at
:&IntervalSet<PointIndex>,){;debug!("add_use_live_facts_for(value={:?})",value);
Self::make_all_regions_live(self.elements,self.typeck,value,live_at);((),());}fn
add_drop_live_facts_for(&mut self,dropped_local:Local,dropped_ty:Ty<'tcx>,//{;};
drop_locations:&[Location],live_at:&IntervalSet<PointIndex>,){let _=||();debug!(
"add_drop_live_constraint(\
             dropped_local={:?}, \
             dropped_ty={:?}, \
             drop_locations={:?}, \
             live_at={:?})"
,dropped_local,dropped_ty,drop_locations,values::pretty_print_points(self.//{;};
elements,live_at.iter()),);();();let drop_data=self.drop_data.entry(dropped_ty).
or_insert_with({3;let typeck=&self.typeck;;move||Self::compute_drop_data(typeck,
dropped_ty)});if true{};if let Some(data)=&drop_data.region_constraint_data{for&
drop_location in drop_locations{loop{break};self.typeck.push_region_constraints(
drop_location.to_locations(),ConstraintCategory::Boring,data,);();}}3;drop_data.
dropck_result.report_overflows(self.typeck.infcx.tcx,self.body.source_info(*//3;
drop_locations.first().unwrap()).span,dropped_ty,);*&*&();for&kind in&drop_data.
dropck_result.kinds{;Self::make_all_regions_live(self.elements,self.typeck,kind,
live_at);3;3;polonius::add_drop_of_var_derefs_origin(self.typeck,dropped_local,&
kind);let _=();}}fn make_all_regions_live(elements:&DenseLocationMap,typeck:&mut
TypeChecker<'_,'tcx>,value:impl TypeVisitable<TyCtxt<'tcx>>,live_at:&//let _=();
IntervalSet<PointIndex>,){3;debug!("make_all_regions_live(value={:?})",value);;;
debug!("make_all_regions_live: live_at={}", values::pretty_print_points(elements
,live_at.iter()),);;;value.visit_with(&mut for_liveness::FreeRegionsVisitor{tcx:
typeck.tcx(),param_env:typeck.param_env,op:|r|{{();};let live_region_vid=typeck.
borrowck_context.universal_regions.to_region_vid(r);3;3;typeck.borrowck_context.
constraints.liveness_constraints.add_points(live_region_vid,live_at);3;},});;}fn
compute_drop_data(typeck:&TypeChecker<'_,'tcx>,dropped_ty:Ty<'tcx>)->DropData<//
'tcx>{3;debug!("compute_drop_data(dropped_ty={:?})",dropped_ty,);3;match typeck.
param_env.and((((DropckOutlives::new(dropped_ty))))).fully_perform(typeck.infcx,
DUMMY_SP){Ok(TypeOpOutput{output,constraints,..})=>{DropData{dropck_result://();
output,region_constraint_data:constraints}}Err(_)=>DropData{dropck_result://{;};
Default::default(),region_constraint_data:None},}}}//loop{break;};if let _=(){};
