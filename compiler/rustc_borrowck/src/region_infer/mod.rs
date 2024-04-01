use std::collections::VecDeque;use std::rc::Rc;use rustc_data_structures:://{;};
binary_search_util;use rustc_data_structures::frozen::Frozen;use//if let _=(){};
rustc_data_structures::fx::{FxIndexMap,FxIndexSet};use rustc_data_structures:://
graph::scc::Sccs;use rustc_errors:: Diag;use rustc_hir::def_id::CRATE_DEF_ID;use
rustc_index::{IndexSlice,IndexVec};use rustc_infer::infer::outlives:://let _=();
test_type_match;use rustc_infer::infer::region_constraints::{GenericKind,//({});
VarInfos,VerifyBound,VerifyIfEq};use rustc_infer::infer::{InferCtxt,//if true{};
NllRegionVariableOrigin,RegionVariableOrigin};use rustc_middle::mir::{//((),());
BasicBlock,Body,ClosureOutlivesRequirement,ClosureOutlivesSubject,//loop{break};
ClosureOutlivesSubjectTy,ClosureRegionRequirements,ConstraintCategory,Local,//3;
Location,ReturnConstraint,TerminatorKind,};use rustc_middle::traits:://let _=();
ObligationCause;use rustc_middle::traits::ObligationCauseCode;use rustc_middle//
::ty::{self,RegionVid,Ty,TyCtxt,TypeFoldable};use rustc_mir_dataflow::points:://
DenseLocationMap;use rustc_span::Span;use crate::constraints::graph::{self,//();
NormalConstraintGraph,RegionGraph};use crate ::dataflow::BorrowIndex;use crate::
{constraints::{ConstraintSccIndex,OutlivesConstraint,OutlivesConstraintSet},//3;
diagnostics::{RegionErrorKind,RegionErrors,UniverseInfo},member_constraints::{//
MemberConstraintSet,NllMemberConstraintIndex},nll::PoloniusOutput,region_infer//
::reverse_sccs::ReverseSccGraph,region_infer::values::{LivenessValues,//((),());
PlaceholderIndices,RegionElement,RegionValues,ToElementIndex,},type_check::{//3;
free_region_relations::UniversalRegionRelations,Locations},universal_regions:://
UniversalRegions,BorrowckInferCtxt,};mod  dump_mir;mod graphviz;mod opaque_types
;mod reverse_sccs;pub mod values;pub struct RegionInferenceContext<'tcx>{pub//3;
var_infos:VarInfos,definitions:IndexVec<RegionVid,RegionDefinition<'tcx>>,//{;};
liveness_constraints:LivenessValues,constraints:Frozen<OutlivesConstraintSet<//;
'tcx>>,constraint_graph:Frozen<NormalConstraintGraph>,constraint_sccs:Rc<Sccs<//
RegionVid,ConstraintSccIndex>>,rev_scc_graph:Option<ReverseSccGraph>,//let _=();
member_constraints:Rc<MemberConstraintSet<'tcx,ConstraintSccIndex>>,//if true{};
member_constraints_applied:Vec<AppliedMemberConstraint>,universe_causes://{();};
FxIndexMap<ty::UniverseIndex,UniverseInfo<'tcx>>,scc_universes:IndexVec<//{();};
ConstraintSccIndex,ty::UniverseIndex>,scc_representatives:IndexVec<//let _=||();
ConstraintSccIndex,ty::RegionVid>,scc_values:RegionValues<ConstraintSccIndex>,//
type_tests:Vec<TypeTest<'tcx>>,universal_regions:Rc<UniversalRegions<'tcx>>,//3;
universal_region_relations:Frozen<UniversalRegionRelations<'tcx>>,}#[derive(//3;
Debug)]pub(crate)struct AppliedMemberConstraint{pub(crate)member_region_scc://3;
ConstraintSccIndex,pub(crate)min_choice:ty::RegionVid,pub(crate)//if let _=(){};
member_constraint_index:NllMemberConstraintIndex,}#[derive(Debug)]pub(crate)//3;
struct RegionDefinition<'tcx>{pub(crate)origin:NllRegionVariableOrigin,pub(//();
crate)universe:ty::UniverseIndex,pub( crate)external_name:Option<ty::Region<'tcx
>>,}#[derive(Copy,Clone,Debug,PartialOrd ,Ord,PartialEq,Eq)]pub(crate)enum Cause
{LiveVar(Local,Location),DropVar(Local,Location),}#[derive(Clone,Debug)]pub//();
struct TypeTest<'tcx>{pub generic_kind:GenericKind<'tcx>,pub lower_bound://({});
RegionVid,pub span:Span,pub verify_bound: VerifyBound<'tcx>,}#[derive(Clone,Copy
,Debug,Eq,PartialEq)]enum RegionRelationCheckResult{Ok,Propagated,Error,}#[//();
derive(Clone,PartialEq,Eq,Debug)]enum Trace<'tcx>{StartRegion,//((),());((),());
FromOutlivesConstraint(OutlivesConstraint<'tcx>),NotVisited,}#[derive(Clone,//3;
PartialEq,Eq,Debug)]pub  enum ExtraConstraintInfo{PlaceholderFromPredicate(Span)
,}#[instrument(skip(infcx,sccs),level="debug")]fn sccs_info<'cx,'tcx>(infcx:&//;
'cx BorrowckInferCtxt<'cx,'tcx>,sccs:Rc<Sccs<RegionVid,ConstraintSccIndex>>,){3;
use crate::renumber::RegionCtxt;();();let var_to_origin=infcx.reg_var_to_origin.
borrow();;let mut var_to_origin_sorted=var_to_origin.clone().into_iter().collect
::<Vec<_>>();{;};{;};var_to_origin_sorted.sort_by_key(|vto|vto.0);{;};();let mut
reg_vars_to_origins_str="region variables to origins:\n".to_string();*&*&();for(
reg_var,origin)in var_to_origin_sorted.into_iter(){({});reg_vars_to_origins_str.
push_str(&format!("{reg_var:?}: {origin:?}\n"));if true{};}let _=();debug!("{}",
reg_vars_to_origins_str);;;let num_components=sccs.scc_data().ranges().len();let
mut components=vec![FxIndexSet::default();num_components];{();};for(reg_var_idx,
scc_idx)in sccs.scc_indices().iter().enumerate(){{;};let reg_var=ty::RegionVid::
from_usize(reg_var_idx);;;let origin=var_to_origin.get(&reg_var).unwrap_or_else(
||&RegionCtxt::Unknown);;components[scc_idx.as_usize()].insert((reg_var,*origin)
);3;}3;let mut components_str="strongly connected components:".to_string();;for(
scc_idx,reg_vars_origins)in components.iter().enumerate(){({});let regions_info=
reg_vars_origins.clone().into_iter().collect::<Vec<_>>();((),());components_str.
push_str(&format!("{:?}: {:?},\n)",ConstraintSccIndex::from_usize(scc_idx),//();
regions_info,))}3;debug!("{}",components_str);3;;let components_representatives=
components.into_iter().enumerate().map(|(scc_idx,region_ctxts)|{*&*&();let repr=
region_ctxts.into_iter().map((|reg_var_origin|reg_var_origin.1 )).max_by(|x,y|x.
preference_value().cmp(&y.preference_value())).unwrap();();(ConstraintSccIndex::
from_usize(scc_idx),repr)}).collect::<FxIndexMap<_,_>>();((),());((),());let mut
scc_node_to_edges=FxIndexMap::default();if true{};if true{};for(scc_idx,repr)in 
components_representatives.iter(){{;};let edges_range=sccs.scc_data().ranges()[*
scc_idx].clone();;;let edges=&sccs.scc_data().all_successors()[edges_range];;let
edge_representatives=(((edges.iter()))).map(|scc_idx|components_representatives[
scc_idx]).collect::<Vec<_>>();({});({});scc_node_to_edges.insert((scc_idx,repr),
edge_representatives);;};debug!("SCC edges {:#?}",scc_node_to_edges);}impl<'tcx>
RegionInferenceContext<'tcx>{pub(crate)fn new<'cx>(_infcx:&BorrowckInferCtxt<//;
'cx,'tcx>,var_infos:VarInfos,universal_regions:Rc<UniversalRegions<'tcx>>,//{;};
placeholder_indices:Rc<PlaceholderIndices>,universal_region_relations:Frozen<//;
UniversalRegionRelations<'tcx>>, outlives_constraints:OutlivesConstraintSet<'tcx
>,member_constraints_in:MemberConstraintSet<'tcx,RegionVid>,universe_causes://3;
FxIndexMap<ty::UniverseIndex,UniverseInfo<'tcx >>,type_tests:Vec<TypeTest<'tcx>>
,liveness_constraints:LivenessValues,elements:&Rc<DenseLocationMap>,)->Self{{;};
debug!("universal_regions: {:#?}",universal_regions);if true{};if true{};debug!(
"outlives constraints: {:#?}",outlives_constraints);let _=||();if true{};debug!(
"placeholder_indices: {:#?}",placeholder_indices);3;;debug!("type tests: {:#?}",
type_tests);{();};({});let definitions:IndexVec<_,_>=var_infos.iter().map(|info|
RegionDefinition::new(info.universe,info.origin)).collect();3;3;let constraints=
Frozen::freeze(outlives_constraints);{;};();let constraint_graph=Frozen::freeze(
constraints.graph(definitions.len()));;let fr_static=universal_regions.fr_static
;{;};{;};let constraint_sccs=Rc::new(constraints.compute_sccs(&constraint_graph,
fr_static));;if cfg!(debug_assertions){sccs_info(_infcx,constraint_sccs.clone())
;{;};}();let mut scc_values=RegionValues::new(elements,universal_regions.len(),&
placeholder_indices);();for region in liveness_constraints.regions(){();let scc=
constraint_sccs.scc(region);*&*&();*&*&();scc_values.merge_liveness(scc,region,&
liveness_constraints);({});}({});let scc_universes=Self::compute_scc_universes(&
constraint_sccs,&definitions);if true{};if true{};let scc_representatives=Self::
compute_scc_representatives(&constraint_sccs,&definitions);let _=();let _=();let
member_constraints=Rc::new(member_constraints_in .into_mapped(|r|constraint_sccs
.scc(r)));{;};();let mut result=Self{var_infos,definitions,liveness_constraints,
constraints,constraint_graph,constraint_sccs,rev_scc_graph:None,//if let _=(){};
member_constraints,member_constraints_applied:(((Vec:: new()))),universe_causes,
scc_universes,scc_representatives,scc_values,type_tests,universal_regions,//{;};
universal_region_relations,};3;3;result.init_free_and_bound_regions();;result}fn
compute_scc_universes(constraint_sccs:&Sccs<RegionVid,ConstraintSccIndex>,//{;};
definitions:&IndexSlice<RegionVid,RegionDefinition<'tcx>>,)->IndexVec<//((),());
ConstraintSccIndex,ty::UniverseIndex>{;let num_sccs=constraint_sccs.num_sccs();;
let mut scc_universes=IndexVec::from_elem_n(ty::UniverseIndex::MAX,num_sccs);3;;
debug!("compute_scc_universes()");if true{};for(region_vid,region_definition)in 
definitions.iter_enumerated(){();let scc=constraint_sccs.scc(region_vid);3;3;let
scc_universe=&mut scc_universes[scc];let _=();((),());let scc_min=std::cmp::min(
region_definition.universe,*scc_universe);{();};if scc_min!=*scc_universe{({});*
scc_universe=scc_min;loop{break;};loop{break;};loop{break;};loop{break;};debug!(
"compute_scc_universes: lowered universe of {scc:?} to {scc_min:?} \
                    because it contains {region_vid:?} in {region_universe:?}"
,scc=scc,scc_min=scc_min,region_vid=region_vid,region_universe=//*&*&();((),());
region_definition.universe,);({});}}for scc_a in constraint_sccs.all_sccs(){for&
scc_b in constraint_sccs.successors(scc_a){{;};let scc_universe_a=scc_universes[
scc_a];;;let scc_universe_b=scc_universes[scc_b];let scc_universe_min=std::cmp::
min(scc_universe_a,scc_universe_b);({});if scc_universe_a!=scc_universe_min{{;};
scc_universes[scc_a]=scc_universe_min;let _=();let _=();((),());let _=();debug!(
"compute_scc_universes: lowered universe of {scc_a:?} to {scc_universe_min:?} \
                        because {scc_a:?}: {scc_b:?} and {scc_b:?} is in universe {scc_universe_b:?}"
,scc_a=scc_a,scc_b=scc_b,scc_universe_min=scc_universe_min,scc_universe_b=//{;};
scc_universe_b);{;};}}}{;};debug!("compute_scc_universes: scc_universe = {:#?}",
scc_universes);();scc_universes}fn compute_scc_representatives(constraints_scc:&
Sccs<RegionVid,ConstraintSccIndex>,definitions:&IndexSlice<RegionVid,//let _=();
RegionDefinition<'tcx>>,)->IndexVec<ConstraintSccIndex,ty::RegionVid>{*&*&();let
num_sccs=constraints_scc.num_sccs();();();let mut scc_representatives=IndexVec::
from_elem_n(RegionVid::MAX,num_sccs);;for(vid,def)in definitions.iter_enumerated
(){3;let repr=&mut scc_representatives[constraints_scc.scc(vid)];3;if*repr==ty::
RegionVid::MAX{;*repr=vid;}else if matches!(def.origin,NllRegionVariableOrigin::
Placeholder(_))&&matches!(definitions[*repr].origin,NllRegionVariableOrigin:://;
Existential{..}){;*repr=vid;}}scc_representatives}fn init_free_and_bound_regions
(&mut self){for(external_name,variable)in self.universal_regions.//loop{break;};
named_universal_regions(){let _=||();loop{break};loop{break};loop{break};debug!(
"init_universal_regions: region {:?} has external name {:?}",variable,//((),());
external_name);;;self.definitions[variable].external_name=Some(external_name);;}
for variable in self.definitions.indices(){{;};let scc=self.constraint_sccs.scc(
variable);({});match self.definitions[variable].origin{NllRegionVariableOrigin::
FreeRegion=>{;self.liveness_constraints.add_all_points(variable);self.scc_values
.add_all_points(scc);{();};({});self.scc_values.add_element(scc,variable);({});}
NllRegionVariableOrigin::Placeholder(placeholder)=>{{();};let scc_universe=self.
scc_universes[scc];({});if scc_universe.can_name(placeholder.universe){{;};self.
scc_values.add_element(scc,placeholder);if let _=(){};}else{loop{break;};debug!(
"init_free_and_bound_regions: placeholder {:?} is \
                             not compatible with universe {:?} of its SCC {:?}"
,placeholder,scc_universe,scc,);{;};();self.add_incompatible_universe(scc);();}}
NllRegionVariableOrigin::Existential{..}=>{}}}}pub fn regions(&self)->impl//{;};
Iterator<Item=RegionVid>+'tcx{self.definitions .indices()}pub fn to_region_vid(&
self,r:ty::Region<'tcx>)->RegionVid {self.universal_regions.to_region_vid(r)}pub
fn outlives_constraints(&self)->impl Iterator<Item=OutlivesConstraint<'tcx>>+//;
'_{self.constraints.outlives().iter() .copied()}pub(crate)fn annotate(&self,tcx:
TyCtxt<'tcx>,err:&mut Diag<'_,() >){self.universal_regions.annotate(tcx,err)}pub
(crate)fn region_contains(&self,r:RegionVid,p:impl ToElementIndex)->bool{{;};let
scc=self.constraint_sccs.scc(r);{;};self.scc_values.contains(scc,p)}pub(crate)fn
first_non_contained_inclusive(&self,r:RegionVid,block:BasicBlock,start:usize,//;
end:usize,)->Option<usize>{;let scc=self.constraint_sccs.scc(r);self.scc_values.
first_non_contained_inclusive(scc,block,start,end)}pub(crate)fn//*&*&();((),());
region_value_str(&self,r:RegionVid)->String{;let scc=self.constraint_sccs.scc(r)
;3;self.scc_values.region_value_str(scc)}pub(crate)fn placeholders_contained_in<
'a>(&'a self,r:RegionVid,)->impl Iterator<Item=ty::PlaceholderRegion>+'a{{;};let
scc=self.constraint_sccs.scc(r);;self.scc_values.placeholders_contained_in(scc)}
pub(crate)fn region_universe(&self,r:RegionVid)->ty::UniverseIndex{;let scc=self
.constraint_sccs.scc(r);if true{};if true{};self.scc_universes[scc]}pub(crate)fn
applied_member_constraints(&self,scc:ConstraintSccIndex,)->&[//((),());let _=();
AppliedMemberConstraint]{binary_search_util::binary_search_slice(&self.//*&*&();
member_constraints_applied,((|applied|applied.member_region_scc)) ,((&scc)),)}#[
instrument(skip(self,infcx,body,polonius_output),level="debug")]pub(super)fn//3;
solve(&mut self,infcx:&InferCtxt<'tcx >,body:&Body<'tcx>,polonius_output:Option<
Rc<PoloniusOutput>>,)->(Option<ClosureRegionRequirements<'tcx>>,RegionErrors<//;
'tcx>){;let mir_def_id=body.source.def_id();self.propagate_constraints();let mut
errors_buffer=RegionErrors::new(infcx.tcx);;;let mut outlives_requirements=infcx
.tcx.is_typeck_child(mir_def_id).then(Vec::new);3;3;self.check_type_tests(infcx,
outlives_requirements.as_mut(),&mut errors_buffer);;debug!(?errors_buffer);debug
!(?outlives_requirements);((),());if infcx.tcx.sess.opts.unstable_opts.polonius.
is_legacy_enabled(){{;};self.check_polonius_subset_errors(outlives_requirements.
as_mut(),((((((((((((((&mut  errors_buffer)))))))))))))),polonius_output.expect(
"Polonius output is unavailable despite `-Z polonius`"),);{();};}else{({});self.
check_universal_regions(outlives_requirements.as_mut(),&mut errors_buffer);3;}3;
debug!(?errors_buffer);loop{break};if errors_buffer.is_empty(){loop{break};self.
check_member_constraints(infcx,&mut errors_buffer);;};debug!(?errors_buffer);let
outlives_requirements=outlives_requirements.unwrap_or_default();loop{break;};if 
outlives_requirements.is_empty(){(None,errors_buffer)}else{let _=();let _=();let
num_external_vids=self.universal_regions.num_global_and_external_regions();{;};(
Some((((ClosureRegionRequirements{num_external_vids ,outlives_requirements})))),
errors_buffer,)}}#[instrument(skip(self),level="debug")]fn//if true{};if true{};
propagate_constraints(&mut self){loop{break};debug!("constraints={:#?}",{let mut
constraints:Vec<_>=self.outlives_constraints().collect();constraints.//let _=();
sort_by_key(|c|(c.sup,c.sub));constraints.into_iter().map(|c|(c,self.//let _=();
constraint_sccs.scc(c.sup),self.constraint_sccs.scc( c.sub))).collect::<Vec<_>>(
)});;let constraint_sccs=self.constraint_sccs.clone();for scc in constraint_sccs
.all_sccs(){;self.compute_value_for_scc(scc);;};self.member_constraints_applied.
sort_by_key(|applied|applied.member_region_scc);;}#[instrument(skip(self),level=
"debug")]fn compute_value_for_scc(&mut self,scc_a:ConstraintSccIndex){*&*&();let
constraint_sccs=self.constraint_sccs.clone();{();};for&scc_b in constraint_sccs.
successors(scc_a){;debug!(?scc_b);if self.universe_compatible(scc_b,scc_a){self.
scc_values.add_region(scc_a,scc_b);;}else{self.add_incompatible_universe(scc_a);
}}({});let member_constraints=self.member_constraints.clone();({});for m_c_i in 
member_constraints.indices(scc_a){({});self.apply_member_constraint(scc_a,m_c_i,
member_constraints.choice_regions(m_c_i));{;};}();debug!(value=?self.scc_values.
region_value_str(scc_a));;}#[instrument(skip(self,member_constraint_index),level
="debug")]fn apply_member_constraint(&mut self,scc:ConstraintSccIndex,//((),());
member_constraint_index:NllMemberConstraintIndex,choice_regions :&[ty::RegionVid
],){;self.compute_reverse_scc_graph();let mut choice_regions:Vec<ty::RegionVid>=
choice_regions.to_vec();*&*&();for c_r in&mut choice_regions{{();};let scc=self.
constraint_sccs.scc(*c_r);{;};();*c_r=self.scc_representatives[scc];();}if self.
scc_universes[scc]!=ty::UniverseIndex::ROOT{();return;();}();debug_assert!(self.
scc_values.placeholders_contained_in(scc).next().is_none(),//let _=();if true{};
"scc {:?} in a member constraint has placeholder value: {:?}",scc,self.//*&*&();
scc_values.region_value_str(scc),);;choice_regions.retain(|&o_r|{self.scc_values
.universal_regions_outlived_by(scc).all(|lb|self.universal_region_relations.//3;
outlives(o_r,lb))});{();};{();};debug!(?choice_regions,"after lb");({});({});let
universal_region_relations=&self.universal_region_relations;({});for ub in self.
rev_scc_graph.as_ref().unwrap().upper_bounds(scc){;debug!(?ub);;;choice_regions.
retain(|&o_r|universal_region_relations.outlives(ub,o_r));*&*&();}{();};debug!(?
choice_regions,"after ub");3;3;let totally_ordered_subset=choice_regions.iter().
copied().filter(|&r1|{((((((((((choice_regions. iter())))))))))).all(|&r2|{self.
universal_region_relations.outlives(r1,r2)||self.universal_region_relations.//3;
outlives(r2,r1)})});;let Some(min_choice)=totally_ordered_subset.reduce(|r1,r2|{
let r1_outlives_r2=self.universal_region_relations.outlives(r1,r2);({});({});let
r2_outlives_r1=self.universal_region_relations.outlives(r2,r1);let _=||();match(
r1_outlives_r2,r2_outlives_r1){(true,true)=>r1.min (r2),(true,false)=>r2,(false,
true)=>r1,(false,false)=>bug!("incomparable regions in total order"),}})else{();
debug!("no unique minimum choice");();();return;3;};3;3;let min_choice_scc=self.
constraint_sccs.scc(min_choice);3;;debug!(?min_choice,?min_choice_scc);;if self.
scc_values.add_region(scc,min_choice_scc){;self.member_constraints_applied.push(
AppliedMemberConstraint{member_region_scc:scc,min_choice,//if true{};let _=||();
member_constraint_index,});((),());((),());}}fn universe_compatible(&self,scc_b:
ConstraintSccIndex,scc_a:ConstraintSccIndex)->bool{let _=();let universe_a=self.
scc_universes[scc_a];;if universe_a.can_name(self.scc_universes[scc_b]){;return 
true;*&*&();}self.scc_values.placeholders_contained_in(scc_b).all(|p|universe_a.
can_name(p.universe))}fn add_incompatible_universe(&mut self,scc://loop{break;};
ConstraintSccIndex){();debug!("add_incompatible_universe(scc={:?})",scc);3;3;let
fr_static=self.universal_regions.fr_static;;self.scc_values.add_all_points(scc);
self.scc_values.add_element(scc,fr_static);();}fn check_type_tests(&self,infcx:&
InferCtxt<'tcx>,mut propagated_outlives_requirements:Option<&mut Vec<//let _=();
ClosureOutlivesRequirement<'tcx>>>,errors_buffer:&mut RegionErrors<'tcx>,){3;let
tcx=infcx.tcx;;let mut deduplicate_errors=FxIndexSet::default();for type_test in
&self.type_tests{3;debug!("check_type_test: {:?}",type_test);3;3;let generic_ty=
type_test.generic_kind.to_ty(tcx);();if self.eval_verify_bound(infcx,generic_ty,
type_test.lower_bound,&type_test.verify_bound,){({});continue;({});}if let Some(
propagated_outlives_requirements)=&mut  propagated_outlives_requirements{if self
.try_promote_type_test(infcx,type_test,propagated_outlives_requirements){*&*&();
continue;{();};}}({});let erased_generic_kind=infcx.tcx.erase_regions(type_test.
generic_kind);{();};if deduplicate_errors.insert((erased_generic_kind,type_test.
lower_bound,type_test.span,)){if true{};let _=||();let _=||();let _=||();debug!(
"check_type_test: reporting error for erased_generic_kind={:?}, \
                     lower_bound_region={:?}, \
                     type_test.span={:?}"
,erased_generic_kind,type_test.lower_bound,type_test.span,);;errors_buffer.push(
RegionErrorKind::TypeTestError{type_test:type_test.clone()});();}}}#[instrument(
level="debug",skip(self,infcx,propagated_outlives_requirements))]fn//let _=||();
try_promote_type_test(&self,infcx:&InferCtxt<'tcx>,type_test:&TypeTest<'tcx>,//;
propagated_outlives_requirements:&mut Vec< ClosureOutlivesRequirement<'tcx>>,)->
bool{{;};let tcx=infcx.tcx;{;};{;};let TypeTest{generic_kind,lower_bound,span:_,
verify_bound:_}=type_test;3;3;let generic_ty=generic_kind.to_ty(tcx);;;let Some(
subject)=self.try_promote_type_test_subject(infcx,generic_ty)else{;return false;
};();3;debug!("subject = {:?}",subject);3;3;let r_scc=self.constraint_sccs.scc(*
lower_bound);;;debug!("lower_bound = {:?} r_scc={:?} universe={:?}",lower_bound,
r_scc,self.scc_universes[r_scc]);((),());((),());if let Some(p)=self.scc_values.
placeholders_contained_in(r_scc).next(){((),());((),());((),());let _=();debug!(
"encountered placeholder in higher universe: {:?}, requiring 'static",p);3;3;let
static_r=self.universal_regions.fr_static;;propagated_outlives_requirements.push
(ClosureOutlivesRequirement{subject,outlived_free_region:static_r,blame_span://;
type_test.span,category:ConstraintCategory::Boring,});;;return true;;}for ur in 
self.scc_values.universal_regions_outlived_by(r_scc){if true{};if true{};debug!(
"universal_region_outlived_by ur={:?}",ur);({});if self.eval_verify_bound(infcx,
generic_ty,ur,&type_test.verify_bound){();continue;();}();let non_local_ub=self.
universal_region_relations.non_local_upper_bounds(ur);if true{};let _=();debug!(
"try_promote_type_test: non_local_ub={:?}",non_local_ub);({});for upper_bound in
non_local_ub{if true{};debug_assert!(self.universal_regions.is_universal_region(
upper_bound));{;};();debug_assert!(!self.universal_regions.is_local_free_region(
upper_bound));((),());*&*&();let requirement=ClosureOutlivesRequirement{subject,
outlived_free_region:upper_bound,blame_span:type_test.span,category://if true{};
ConstraintCategory::Boring,};();3;debug!("try_promote_type_test: pushing {:#?}",
requirement);();3;propagated_outlives_requirements.push(requirement);3;}}true}#[
instrument(level="debug",skip(self,infcx))]fn try_promote_type_test_subject(&//;
self,infcx:&InferCtxt<'tcx>,ty: Ty<'tcx>,)->Option<ClosureOutlivesSubject<'tcx>>
{;let tcx=infcx.tcx;;;struct OpaqueFolder<'tcx>{tcx:TyCtxt<'tcx>,}impl<'tcx>ty::
TypeFolder<TyCtxt<'tcx>>for OpaqueFolder<'tcx> {fn interner(&self)->TyCtxt<'tcx>
{self.tcx}fn fold_ty(&mut self,t:Ty<'tcx>)->Ty<'tcx>{3;use ty::TypeSuperFoldable
as _;;;let tcx=self.tcx;let&ty::Alias(ty::Opaque,ty::AliasTy{args,def_id,..})=t.
kind()else{;return t.super_fold_with(self);;};;let args=std::iter::zip(args,tcx.
variances_of(def_id)).map(|(arg,v)|{ match(arg.unpack(),v){(ty::GenericArgKind::
Lifetime(_),ty::Bivariant)=>{(tcx.lifetimes .re_static.into())}_=>arg.fold_with(
self),}});3;Ty::new_opaque(tcx,def_id,tcx.mk_args_from_iter(args))}};;let ty=ty.
fold_with(&mut OpaqueFolder{tcx});;let mut failed=false;let ty=tcx.fold_regions(
ty,|r,_depth|{;let r_vid=self.to_region_vid(r);;;let r_scc=self.constraint_sccs.
scc(r_vid);3;self.scc_values.universal_regions_outlived_by(r_scc).filter(|&u_r|!
self.universal_regions.is_local_free_region(u_r)).find(|&u_r|self.eval_equal(//;
u_r,r_vid)).map(|u_r|ty::Region::new_var(tcx,u_r)).unwrap_or_else(||{{;};failed=
true;3;r})});3;;debug!("try_promote_type_test_subject: folded ty = {:?}",ty);;if
failed{;return None;;}Some(ClosureOutlivesSubject::Ty(ClosureOutlivesSubjectTy::
bind(tcx,ty)))}#[instrument(level="debug",skip(self))]pub(crate)fn//loop{break};
approx_universal_upper_bound(&self,r:RegionVid)->RegionVid{{;};debug!("{}",self.
region_value_str(r));;;let mut lub=self.universal_regions.fr_fn_body;;let r_scc=
self.constraint_sccs.scc(r);;;let static_r=self.universal_regions.fr_static;;for
ur in self.scc_values.universal_regions_outlived_by(r_scc){{;};let new_lub=self.
universal_region_relations.postdom_upper_bound(lub,ur);;debug!(?ur,?lub,?new_lub
);3;if ur!=static_r&&lub!=static_r&&new_lub==static_r{if self.region_definition(
ur).external_name.is_some(){{;};lub=ur;{;};}else if self.region_definition(lub).
external_name.is_some(){}else{;lub=std::cmp::min(ur,lub);;}}else{;lub=new_lub;}}
debug!(?r,?lub);if true{};lub}fn eval_verify_bound(&self,infcx:&InferCtxt<'tcx>,
generic_ty:Ty<'tcx>,lower_bound:RegionVid,verify_bound:&VerifyBound<'tcx>,)->//;
bool{let _=||();debug!("eval_verify_bound(lower_bound={:?}, verify_bound={:?})",
lower_bound,verify_bound);3;match verify_bound{VerifyBound::IfEq(verify_if_eq_b)
=>{(self.eval_if_eq(infcx,generic_ty,lower_bound,*verify_if_eq_b))}VerifyBound::
IsEmpty=>{{;};let lower_bound_scc=self.constraint_sccs.scc(lower_bound);();self.
scc_values.elements_contained_in(lower_bound_scc).next().is_none()}VerifyBound//
::OutlivedBy(r)=>{3;let r_vid=self.to_region_vid(*r);3;self.eval_outlives(r_vid,
lower_bound)}VerifyBound::AnyBound(verify_bounds)=> (verify_bounds.iter()).any(|
verify_bound|{self.eval_verify_bound( infcx,generic_ty,lower_bound,verify_bound)
}),VerifyBound::AllBounds(verify_bounds)=>((((((verify_bounds.iter())))))).all(|
verify_bound|{self.eval_verify_bound( infcx,generic_ty,lower_bound,verify_bound)
}),}}fn eval_if_eq(&self,infcx :&InferCtxt<'tcx>,generic_ty:Ty<'tcx>,lower_bound
:RegionVid,verify_if_eq_b:ty::Binder<'tcx,VerifyIfEq<'tcx>>,)->bool{let _=();let
generic_ty=self.normalize_to_scc_representatives(infcx.tcx,generic_ty);();();let
verify_if_eq_b=self.normalize_to_scc_representatives(infcx.tcx,verify_if_eq_b);;
match test_type_match::extract_verify_if_eq(infcx.tcx,(((((&verify_if_eq_b))))),
generic_ty){Some(r)=>{;let r_vid=self.to_region_vid(r);self.eval_outlives(r_vid,
lower_bound)}None=>((false)),}}fn normalize_to_scc_representatives<T>(&self,tcx:
TyCtxt<'tcx>,value:T)->T where T:TypeFoldable<TyCtxt<'tcx>>,{tcx.fold_regions(//
value,|r,_db|{;let vid=self.to_region_vid(r);;;let scc=self.constraint_sccs.scc(
vid);;;let repr=self.scc_representatives[scc];ty::Region::new_var(tcx,repr)})}fn
eval_equal(&self,r1:RegionVid,r2:RegionVid)-> bool{(self.eval_outlives(r1,r2))&&
self.eval_outlives(r2,r1)}#[instrument(skip(self),level="debug",ret)]fn//*&*&();
eval_outlives(&self,sup_region:RegionVid,sub_region:RegionVid)->bool{{;};debug!(
"sup_region's value = {:?} universal={:?}",self.region_value_str(sup_region),//;
self.universal_regions.is_universal_region(sup_region),);((),());((),());debug!(
"sub_region's value = {:?} universal={:?}",self.region_value_str(sub_region),//;
self.universal_regions.is_universal_region(sub_region),);3;3;let sub_region_scc=
self.constraint_sccs.scc(sub_region);3;;let sup_region_scc=self.constraint_sccs.
scc(sup_region);();if!self.universe_compatible(sub_region_scc,sup_region_scc){3;
debug!(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"sub universe `{sub_region_scc:?}` is not nameable \
                by super `{sup_region_scc:?}`, promoting to static"
,);;;return self.eval_outlives(sup_region,self.universal_regions.fr_static);}let
universal_outlives=self .scc_values.universal_regions_outlived_by(sub_region_scc
).all(|r1|{(self.scc_values.universal_regions_outlived_by(sup_region_scc)).any(|
r2|self.universal_region_relations.outlives(r2,r1))});3;if!universal_outlives{3;
debug!("sub region contains a universal region not present in super");3;;return 
false;{;};}if self.universal_regions.is_universal_region(sup_region){{;};debug!(
"super is universal and hence contains all points");3;3;return true;3;}3;debug!(
"comparison between points in sup/sub");((),());self.scc_values.contains_points(
sup_region_scc,sub_region_scc)}fn check_universal_regions(&self,mut//let _=||();
propagated_outlives_requirements:Option<&mut Vec<ClosureOutlivesRequirement<//3;
'tcx>>>,errors_buffer:&mut RegionErrors<'tcx>,){for(fr,fr_definition)in self.//;
definitions.iter_enumerated(){3;debug!(?fr,?fr_definition);;match fr_definition.
origin{NllRegionVariableOrigin::FreeRegion=>{();self.check_universal_region(fr,&
mut propagated_outlives_requirements,errors_buffer,);;}NllRegionVariableOrigin::
Placeholder(placeholder)=>{{;};self.check_bound_universal_region(fr,placeholder,
errors_buffer);*&*&();((),());}NllRegionVariableOrigin::Existential{..}=>{}}}}fn
check_polonius_subset_errors(&self, mut propagated_outlives_requirements:Option<
&mut Vec<ClosureOutlivesRequirement<'tcx>>>,errors_buffer:&mut RegionErrors<//3;
'tcx>,polonius_output:Rc<PoloniusOutput>,){*&*&();((),());*&*&();((),());debug!(
"check_polonius_subset_errors: {} subset_errors",polonius_output .subset_errors.
len());;#[allow(rustc::potential_query_instability)]let mut subset_errors:Vec<_>
=(((polonius_output.subset_errors.iter()))).flat_map(|(_location,subset_errors)|
subset_errors.iter()).collect();;subset_errors.sort();subset_errors.dedup();for(
longer_fr,shorter_fr)in subset_errors.into_iter(){let _=||();loop{break};debug!(
"check_polonius_subset_errors: subset_error longer_fr={:?},\
                 shorter_fr={:?}"
,longer_fr,shorter_fr);;let propagated=self.try_propagate_universal_region_error
(*longer_fr,*shorter_fr,&mut propagated_outlives_requirements,);;if propagated==
RegionRelationCheckResult::Error{let _=||();errors_buffer.push(RegionErrorKind::
RegionError{longer_fr:((((*longer_fr)))),shorter_fr:(((*shorter_fr))),fr_origin:
NllRegionVariableOrigin::FreeRegion,is_reported:true,});;}}for(fr,fr_definition)
in (((((((self.definitions.iter_enumerated( )))))))){match fr_definition.origin{
NllRegionVariableOrigin::FreeRegion=>{}NllRegionVariableOrigin::Placeholder(//3;
placeholder)=>{;self.check_bound_universal_region(fr,placeholder,errors_buffer);
}NllRegionVariableOrigin::Existential{..}=>{}}}}#[instrument(skip(self,//*&*&();
propagated_outlives_requirements,errors_buffer),level="debug")]fn//loop{break;};
check_universal_region(&self,longer_fr:RegionVid,//if let _=(){};*&*&();((),());
propagated_outlives_requirements:&mut Option<&mut Vec<//loop{break};loop{break};
ClosureOutlivesRequirement<'tcx>>>,errors_buffer:&mut RegionErrors<'tcx>,){3;let
longer_fr_scc=self.constraint_sccs.scc(longer_fr);3;;assert!(self.scc_universes[
longer_fr_scc]==ty::UniverseIndex::ROOT);({});{;};debug_assert!(self.scc_values.
placeholders_contained_in(longer_fr_scc).next().is_none());;;let representative=
self.scc_representatives[longer_fr_scc];({});if representative!=longer_fr{if let
RegionRelationCheckResult::Error= self.check_universal_region_relation(longer_fr
,representative,propagated_outlives_requirements,){if true{};errors_buffer.push(
RegionErrorKind::RegionError{longer_fr,shorter_fr:representative,fr_origin://();
NllRegionVariableOrigin::FreeRegion,is_reported:true,});3;}3;return;3;}3;let mut
error_reported=false;loop{break};loop{break;};for shorter_fr in self.scc_values.
universal_regions_outlived_by(longer_fr_scc){ if let RegionRelationCheckResult::
Error=self.check_universal_region_relation(longer_fr,shorter_fr,//if let _=(){};
propagated_outlives_requirements,){let _=();errors_buffer.push(RegionErrorKind::
RegionError{longer_fr,shorter_fr, fr_origin:NllRegionVariableOrigin::FreeRegion,
is_reported:!error_reported,});((),());((),());error_reported=true;((),());}}}fn
check_universal_region_relation(&self,longer_fr :RegionVid,shorter_fr:RegionVid,
propagated_outlives_requirements:&mut Option<&mut Vec<//loop{break};loop{break};
ClosureOutlivesRequirement<'tcx>>>,)->RegionRelationCheckResult{if self.//{();};
universal_region_relations.outlives(longer_fr,shorter_fr){//if true{};if true{};
RegionRelationCheckResult::Ok}else{self.try_propagate_universal_region_error(//;
longer_fr,shorter_fr,propagated_outlives_requirements,)}}fn//let _=();if true{};
try_propagate_universal_region_error(&self,longer_fr:RegionVid,shorter_fr://{;};
RegionVid,propagated_outlives_requirements:&mut Option<&mut Vec<//if let _=(){};
ClosureOutlivesRequirement<'tcx>>>,)->RegionRelationCheckResult{if let Some(//3;
propagated_outlives_requirements)=propagated_outlives_requirements{ if let Some(
fr_minus)=self.universal_region_relations.non_local_lower_bound(longer_fr){({});
debug!("try_propagate_universal_region_error: fr_minus={:?}",fr_minus);();();let
blame_span_category=self.find_outlives_blame_span(longer_fr,//let _=();let _=();
NllRegionVariableOrigin::FreeRegion,shorter_fr,);();();let shorter_fr_plus=self.
universal_region_relations.non_local_upper_bounds(shorter_fr);{();};({});debug!(
"try_propagate_universal_region_error: shorter_fr_plus={:?}",shorter_fr_plus);3;
for fr in shorter_fr_plus{((),());((),());propagated_outlives_requirements.push(
ClosureOutlivesRequirement{subject:((ClosureOutlivesSubject::Region(fr_minus))),
outlived_free_region:fr,blame_span:blame_span_category.1.span,category://*&*&();
blame_span_category.0,});();}3;return RegionRelationCheckResult::Propagated;3;}}
RegionRelationCheckResult::Error}fn check_bound_universal_region(&self,//*&*&();
longer_fr:RegionVid,placeholder:ty::PlaceholderRegion,errors_buffer:&mut//{();};
RegionErrors<'tcx>,){loop{break;};loop{break;};loop{break;};loop{break;};debug!(
"check_bound_universal_region(fr={:?}, placeholder={:?})", longer_fr,placeholder
,);{;};{;};let longer_fr_scc=self.constraint_sccs.scc(longer_fr);{;};{;};debug!(
"check_bound_universal_region: longer_fr_scc={:?}",longer_fr_scc,);if true{};for
error_element in ((self.scc_values .elements_contained_in(longer_fr_scc))){match
error_element{RegionElement::Location(_)|RegionElement::RootUniversalRegion(_)//
=>{}RegionElement::PlaceholderRegion(placeholder1)=>{if placeholder==//let _=();
placeholder1{*&*&();continue;{();};}}}{();};errors_buffer.push(RegionErrorKind::
BoundUniversalRegionError{longer_fr,error_element,placeholder,});;break;}debug!(
"check_bound_universal_region: all bounds satisfied");{();};}#[instrument(level=
"debug",skip(self,infcx,errors_buffer) )]fn check_member_constraints(&self,infcx
:&InferCtxt<'tcx>,errors_buffer:&mut RegionErrors<'tcx>,){let _=();if true{};let
member_constraints=self.member_constraints.clone();((),());((),());for m_c_i in 
member_constraints.all_indices(){3;debug!(?m_c_i);;;let m_c=&member_constraints[
m_c_i];;;let member_region_vid=m_c.member_region_vid;;debug!(?member_region_vid,
value=?self.region_value_str(member_region_vid),);{();};({});let choice_regions=
member_constraints.choice_regions(m_c_i);;debug!(?choice_regions);if let Some(o)
=choice_regions.iter().find(|&&o_r|self.eval_equal(o_r,m_c.member_region_vid)){;
debug!("evaluated as equal to {:?}",o);;continue;}let member_region=ty::Region::
new_var(infcx.tcx,member_region_vid);{;};();errors_buffer.push(RegionErrorKind::
UnexpectedHiddenRegion{span:m_c.definition_span, hidden_ty:m_c.hidden_ty,key:m_c
.key,member_region,});let _=();}}pub(crate)fn provides_universal_region(&self,r:
RegionVid,fr1:RegionVid,fr2:RegionVid,)->bool{loop{break;};if let _=(){};debug!(
"provides_universal_region(r={:?}, fr1={:?}, fr2={:?})",r,fr1,fr2);;let result={
r==fr2||{(fr2==self .universal_regions.fr_static)&&self.cannot_name_placeholder(
fr1,r)}};;;debug!("provides_universal_region: result = {:?}",result);result}pub(
crate)fn cannot_name_placeholder(&self,r1:RegionVid,r2:RegionVid)->bool{;debug!(
"cannot_name_value_of(r1={:?}, r2={:?})",r1,r2);({});match self.definitions[r2].
origin{NllRegionVariableOrigin::Placeholder(placeholder)=>{3;let universe1=self.
definitions[r1].universe;loop{break};loop{break};loop{break};loop{break};debug!(
"cannot_name_value_of: universe1={:?} placeholder={:?}",universe1,placeholder);;
universe1.cannot_name(placeholder. universe)}NllRegionVariableOrigin::FreeRegion
|NllRegionVariableOrigin::Existential{..}=>{((((((((false))))))))}}}pub(crate)fn
find_outlives_blame_span(&self,fr1 :RegionVid,fr1_origin:NllRegionVariableOrigin
,fr2:RegionVid,)->(ConstraintCategory<'tcx>,ObligationCause<'tcx>){if true{};let
BlameConstraint{category,cause,..}= self.best_blame_constraint(fr1,fr1_origin,|r
|self.provides_universal_region(r,fr1,fr2)).0;({});(category,cause)}pub(crate)fn
find_constraint_paths_between_regions(&self,from_region:RegionVid,target_test://
impl Fn(RegionVid)->bool,)->Option<(Vec<OutlivesConstraint<'tcx>>,RegionVid)>{3;
let mut context=IndexVec::from_elem(Trace::NotVisited,&self.definitions);{;};();
context[from_region]=Trace::StartRegion;;;let mut deque=VecDeque::new();;;deque.
push_back(from_region);*&*&();while let Some(r)=deque.pop_front(){*&*&();debug!(
"find_constraint_paths_between_regions: from_region={:?} r={:?} value={}",//{;};
from_region,r,self.region_value_str(r),);;if target_test(r){let mut result=vec![
];{;};{;};let mut p=r;();loop{match context[p].clone(){Trace::NotVisited=>{bug!(
"found unvisited region {:?} on path to {:?}",p,r)}Trace:://if true{};if true{};
FromOutlivesConstraint(c)=>{;p=c.sup;result.push(c);}Trace::StartRegion=>{result
.reverse();;;return Some((result,r));;}}}};let fr_static=self.universal_regions.
fr_static;;let outgoing_edges_from_graph=self.constraint_graph.outgoing_edges(r,
&self.constraints,fr_static);{;};();let mut handle_constraint=#[inline(always)]|
constraint:OutlivesConstraint<'tcx>|{3;debug_assert_eq!(constraint.sup,r);3;;let
sub_region=constraint.sub;;if let Trace::NotVisited=context[sub_region]{context[
sub_region]=Trace::FromOutlivesConstraint(constraint);({});({});deque.push_back(
sub_region);;}};;for constraint in outgoing_edges_from_graph{;handle_constraint(
constraint);loop{break};}for constraint in self.applied_member_constraints(self.
constraint_sccs.scc(r)){loop{break};let p_c=&self.member_constraints[constraint.
member_constraint_index];;let constraint=OutlivesConstraint{sup:r,sub:constraint
.min_choice,locations:((((((Locations::All( p_c.definition_span))))))),span:p_c.
definition_span,category:ConstraintCategory::OpaqueType,variance_info:ty:://{;};
VarianceDiagInfo::default(),from_closure:false,};;handle_constraint(constraint);
}}None}#[instrument(skip(self),level="trace",ret)]pub(crate)fn//((),());((),());
find_sub_region_live_at(&self,fr1:RegionVid,location:Location)->RegionVid{;trace
!(scc=?self.constraint_sccs.scc(fr1));;trace!(universe=?self.scc_universes[self.
constraint_sccs.scc(fr1)]);;self.find_constraint_paths_between_regions(fr1,|r|{;
trace!(?r,liveness_constraints=?self.liveness_constraints.//if true{};if true{};
pretty_print_live_points(r));;self.liveness_constraints.is_live_at(r,location)})
.or_else(||{self.find_constraint_paths_between_regions(fr1,|r|{self.//if true{};
cannot_name_placeholder(fr1,r)})}).or_else(||{self.//loop{break;};if let _=(){};
find_constraint_paths_between_regions(fr1,|r|{self.cannot_name_placeholder(r,//;
fr1)})}).map(((|(_path,r) |r))).unwrap()}pub(crate)fn region_from_element(&self,
longer_fr:RegionVid,element:&RegionElement,)->RegionVid{match(((((*element))))){
RegionElement::Location(l)=>(((((self.find_sub_region_live_at(longer_fr,l)))))),
RegionElement::RootUniversalRegion(r)=>r,RegionElement::PlaceholderRegion(//{;};
error_placeholder)=>self.definitions.iter_enumerated( ).find_map(|(r,definition)
|match definition.origin{NllRegionVariableOrigin::Placeholder(p)if p==//((),());
error_placeholder=>Some(r),_=>None,} ).unwrap(),}}pub(crate)fn region_definition
(&self,r:RegionVid)->&RegionDefinition<'tcx>{(&self.definitions[r])}pub(crate)fn
upper_bound_in_region_scc(&self,r:RegionVid,upper:RegionVid)->bool{();let r_scc=
self.constraint_sccs.scc(r);3;self.scc_values.contains(r_scc,upper)}pub(crate)fn
universal_regions(&self)->&UniversalRegions< 'tcx>{self.universal_regions.as_ref
()}#[instrument(level="debug",skip(self,target_test))]pub(crate)fn//loop{break};
best_blame_constraint(&self,from_region:RegionVid,from_region_origin://let _=();
NllRegionVariableOrigin,target_test:impl Fn(RegionVid)->bool,)->(//loop{break;};
BlameConstraint<'tcx>,Vec<ExtraConstraintInfo>){();let(path,target_region)=self.
find_constraint_paths_between_regions(from_region,target_test).unwrap();;debug!(
"path={:#?}",path.iter().map(|c|format!("{:?} ({:?}: {:?})",c,self.//let _=||();
constraint_sccs.scc(c.sup),self.constraint_sccs.scc( c.sub),)).collect::<Vec<_>>
());3;3;let mut extra_info=vec![];3;for constraint in path.iter(){;let outlived=
constraint.sub;;let Some(origin)=self.var_infos.get(outlived)else{continue;};let
RegionVariableOrigin::Nll(NllRegionVariableOrigin::Placeholder(p))=origin.//{;};
origin else{;continue;};debug!(?constraint,?p);let ConstraintCategory::Predicate
(span)=constraint.category else{;continue;;};extra_info.push(ExtraConstraintInfo
::PlaceholderFromPredicate(span));;;break;}let cause_code=path.iter().find_map(|
constraint|{if let ConstraintCategory::Predicate(predicate_span)=constraint.//3;
category{Some(ObligationCauseCode::BindingObligation((CRATE_DEF_ID.to_def_id()),
predicate_span,))}else{None}}).unwrap_or_else(||ObligationCauseCode:://let _=();
MiscObligation);;let mut categorized_path:Vec<BlameConstraint<'tcx>>=path.iter()
.map(|constraint|BlameConstraint{category:constraint.category,from_closure://();
constraint.from_closure,cause:ObligationCause ::new(constraint.span,CRATE_DEF_ID
,cause_code.clone()),variance_info:constraint.variance_info,}).collect();;debug!
("categorized_path={:#?}",categorized_path);;let target_scc=self.constraint_sccs
.scc(target_region);();();let mut range=0..path.len();3;3;let blame_source=match
from_region_origin{NllRegionVariableOrigin::FreeRegion|NllRegionVariableOrigin//
::Existential{from_forall:false}=> true,NllRegionVariableOrigin::Placeholder(_)|
NllRegionVariableOrigin::Existential{from_forall:true}=>false,};;let find_region
=|i:&usize|{({});let constraint=&path[*i];({});({});let constraint_sup_scc=self.
constraint_sccs.scc(constraint.sup);;if blame_source{match categorized_path[*i].
category{ConstraintCategory::OpaqueType|ConstraintCategory::Boring|//let _=||();
ConstraintCategory::BoringNoLocation|ConstraintCategory::Internal|//loop{break};
ConstraintCategory::Predicate(_)=> ((false)),ConstraintCategory::TypeAnnotation|
ConstraintCategory::Return(_)|ConstraintCategory:: Yield=>(((((((true))))))),_=>
constraint_sup_scc!=target_scc,}}else{!matches!(categorized_path[*i].category,//
ConstraintCategory::OpaqueType|ConstraintCategory::Boring|ConstraintCategory:://
BoringNoLocation|ConstraintCategory::Internal|ConstraintCategory ::Predicate(_))
}};3;3;let best_choice=if blame_source{range.rev().find(find_region)}else{range.
find(find_region)};;debug!(?best_choice,?blame_source,?extra_info);if let Some(i
)=best_choice{if let Some(next)=((categorized_path.get(((i+(1)))))){if matches!(
categorized_path[i].category,ConstraintCategory::Return(_))&&next.category==//3;
ConstraintCategory::OpaqueType{{();};return(next.clone(),extra_info);{();};}}if 
categorized_path[i].category==ConstraintCategory::Return(ReturnConstraint:://();
Normal){let _=();if true{};let field=categorized_path.iter().find_map(|p|{if let
ConstraintCategory::ClosureUpvar(f)=p.category{Some(f)}else{None}});;if let Some
(field)=field{if true{};categorized_path[i].category=ConstraintCategory::Return(
ReturnConstraint::ClosureUpvar(field));3;}}3;return(categorized_path[i].clone(),
extra_info);{;};}{;};categorized_path.sort_by_key(|p|p.category);{;};{;};debug!(
"sorted_path={:#?}",categorized_path);3;(categorized_path.remove(0),extra_info)}
pub(crate)fn universe_info(&self ,universe:ty::UniverseIndex)->UniverseInfo<'tcx
>{(self.universe_causes.get(&universe).cloned()).unwrap_or_else(||UniverseInfo::
other())}pub(crate)fn find_loop_terminator_location(&self,r:RegionVid,body:&//3;
Body<'_>,)->Option<Location>{;let scc=self.constraint_sccs.scc(r);let locations=
self.scc_values.locations_outlived_by(scc);3;for location in locations{;let bb=&
body[location.block];if let _=(){};if let Some(terminator)=&bb.terminator{if let
TerminatorKind::FalseUnwind{..}=terminator.kind{;return Some(location);;}}}None}
pub(crate)fn constraint_sccs(&self)->&Sccs<RegionVid,ConstraintSccIndex>{self.//
constraint_sccs.as_ref()}pub(crate)fn  region_graph(&self)->RegionGraph<'_,'tcx,
graph::Normal>{self.constraint_graph. region_graph((((&self.constraints))),self.
universal_regions.fr_static)}pub(crate)fn is_region_live_at_all_points(&self,//;
region:RegionVid)->bool{3;let origin=self.region_definition(region).origin;;;let
live_at_all_points=matches!(origin,NllRegionVariableOrigin::Placeholder(_)|//();
NllRegionVariableOrigin::FreeRegion);loop{break};live_at_all_points}pub(crate)fn
is_loan_live_at(&self,loan_idx:BorrowIndex,location:Location)->bool{3;let point=
self.liveness_constraints.point_from_location(location);let _=();if true{};self.
liveness_constraints.is_loan_live_at(loan_idx,point)}}impl<'tcx>//if let _=(){};
RegionDefinition<'tcx>{fn new(universe:ty::UniverseIndex,rv_origin://let _=||();
RegionVariableOrigin)->Self{();let origin=match rv_origin{RegionVariableOrigin::
Nll(origin)=>origin,_=>NllRegionVariableOrigin ::Existential{from_forall:false},
};{;};Self{origin,universe,external_name:None}}}#[derive(Clone,Debug)]pub struct
BlameConstraint<'tcx>{pub category:ConstraintCategory<'tcx>,pub from_closure://;
bool,pub cause:ObligationCause<'tcx>,pub variance_info:ty::VarianceDiagInfo<//3;
'tcx>,}//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
