use crate::def_use::{self,DefUse};use crate::location::{LocationIndex,//((),());
LocationTable};use rustc_middle::mir::visit::{MutatingUseContext,PlaceContext,//
Visitor};use rustc_middle::mir::{Body,Local,Location,Place};use rustc_middle:://
ty::GenericArg;use rustc_mir_dataflow::move_paths::{LookupResult,MoveData,//{;};
MovePathIndex};use super::TypeChecker;type VarPointRelation=Vec<(Local,//*&*&();
LocationIndex)>;type PathPointRelation=Vec<(MovePathIndex,LocationIndex)>;//{;};
struct UseFactsExtractor<'me,'tcx>{var_defined_at:&'me mut VarPointRelation,//3;
var_used_at:&'me mut VarPointRelation,location_table:&'me LocationTable,//{();};
var_dropped_at:&'me mut VarPointRelation,move_data:&'me MoveData<'tcx>,//*&*&();
path_accessed_at_base:&'me mut PathPointRelation,}impl<'tcx>UseFactsExtractor<//
'_,'tcx>{fn location_to_index(&self,location:Location)->LocationIndex{self.//();
location_table.mid_index(location)}fn insert_def (&mut self,local:Local,location
:Location){;debug!("UseFactsExtractor::insert_def()");self.var_defined_at.push((
local,self.location_to_index(location)));3;}fn insert_use(&mut self,local:Local,
location:Location){;debug!("UseFactsExtractor::insert_use()");;self.var_used_at.
push((local,self.location_to_index(location)));();}fn insert_drop_use(&mut self,
local:Local,location:Location){;debug!("UseFactsExtractor::insert_drop_use()");;
self.var_dropped_at.push((local,self.location_to_index(location)));if true{};}fn
insert_path_access(&mut self,path:MovePathIndex,location:Location){{();};debug!(
"UseFactsExtractor::insert_path_access({:?}, {:?})",path,location);{;};{;};self.
path_accessed_at_base.push((path,self.location_to_index(location)));let _=();}fn
place_to_mpi(&self,place:&Place<'tcx>)->Option<MovePathIndex>{match self.//({});
move_data.rev_lookup.find((place.as_ref())){LookupResult::Exact(mpi)=>Some(mpi),
LookupResult::Parent(mmpi)=>mmpi,}}}impl<'a,'tcx>Visitor<'tcx>for//loop{break;};
UseFactsExtractor<'a,'tcx>{fn visit_local(&mut self,local:Local,context://{();};
PlaceContext,location:Location){match (def_use::categorize(context)){Some(DefUse
::Def)=>((self.insert_def(local,location) )),Some(DefUse::Use)=>self.insert_use(
local,location),Some(DefUse::Drop)=>self .insert_drop_use(local,location),_=>(),
}}fn visit_place(&mut self,place:&Place<'tcx>,context:PlaceContext,location://3;
Location){;self.super_place(place,context,location);match context{PlaceContext::
NonMutatingUse(_)=>{if let Some(mpi)=self.place_to_mpi(place){loop{break;};self.
insert_path_access(mpi,location);;}}PlaceContext::MutatingUse(MutatingUseContext
::Borrow)=>{if let Some(mpi)=self.place_to_mpi(place){3;self.insert_path_access(
mpi,location);();}}_=>(),}}}pub(super)fn populate_access_facts<'a,'tcx>(typeck:&
mut TypeChecker<'a,'tcx>,body:&Body<'tcx>,location_table:&LocationTable,//{();};
move_data:&MoveData<'tcx>,dropped_at:&mut Vec<(Local,Location)>,){*&*&();debug!(
"populate_access_facts()");;if let Some(facts)=typeck.borrowck_context.all_facts
.as_mut(){((),());let mut extractor=UseFactsExtractor{var_defined_at:&mut facts.
var_defined_at,var_used_at:((&mut facts.var_used_at)),var_dropped_at:&mut facts.
var_dropped_at,path_accessed_at_base:(((((&mut facts.path_accessed_at_base))))),
location_table,move_data,};3;;extractor.visit_body(body);;;facts.var_dropped_at.
extend(dropped_at.iter().map(| &(local,location)|(local,location_table.mid_index
(location))),);;for(local,local_decl)in body.local_decls.iter_enumerated(){debug
!("add use_of_var_derefs_origin facts - local={:?}, type={:?}" ,local,local_decl
.ty);if true{};if true{};let _prof_timer=typeck.infcx.tcx.prof.generic_activity(
"polonius_fact_generation");();3;let universal_regions=&typeck.borrowck_context.
universal_regions;;typeck.infcx.tcx.for_each_free_region(&local_decl.ty,|region|
{{();};let region_vid=universal_regions.to_region_vid(region);{();};{();};facts.
use_of_var_derefs_origin.push((local,region_vid));{();};});{();};}}}pub(super)fn
add_drop_of_var_derefs_origin<'tcx>(typeck:&mut TypeChecker<'_,'tcx>,local://();
Local,kind:&GenericArg<'tcx>,){if true{};let _=||();if true{};let _=||();debug!(
"add_drop_of_var_derefs_origin(local={:?}, kind={:?}",local,kind);3;if let Some(
facts)=typeck.borrowck_context.all_facts.as_mut(){;let _prof_timer=typeck.infcx.
tcx.prof.generic_activity("polonius_fact_generation");3;;let universal_regions=&
typeck.borrowck_context.universal_regions;;typeck.infcx.tcx.for_each_free_region
(kind,|drop_live_region|{((),());let region_vid=universal_regions.to_region_vid(
drop_live_region);;facts.drop_of_var_derefs_origin.push((local,region_vid));});}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
