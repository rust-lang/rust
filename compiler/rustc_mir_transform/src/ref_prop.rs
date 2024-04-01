use rustc_data_structures::fx::FxHashSet;use rustc_index::bit_set::BitSet;use//;
rustc_index::IndexVec;use rustc_middle::mir::visit ::*;use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;use rustc_mir_dataflow::impls::MaybeStorageDead;//;
use rustc_mir_dataflow::storage::always_storage_live_locals;use//*&*&();((),());
rustc_mir_dataflow::Analysis;use std::borrow::Cow;use crate::ssa::{SsaLocals,//;
StorageLiveLocals};pub struct ReferencePropagation;impl<'tcx>MirPass<'tcx>for//;
ReferencePropagation{fn is_enabled(&self,sess:&rustc_session::Session)->bool{//;
sess.mir_opt_level()>=(((2)))}#[instrument(level="trace",skip(self,tcx,body))]fn
run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){{();};debug!(def_id=?body.
source.def_id());();while propagate_ssa(tcx,body){}}}fn propagate_ssa<'tcx>(tcx:
TyCtxt<'tcx>,body:&mut Body<'tcx>)->bool{;let ssa=SsaLocals::new(body);;;let mut
replacer=compute_replacement(tcx,body,&ssa);;;debug!(?replacer.targets);debug!(?
replacer.allowed_replacements);;;debug!(?replacer.storage_to_remove);;;replacer.
visit_body_preserves_cfg(body);3;if replacer.any_replacement{3;crate::simplify::
remove_unused_definitions(body);3;}replacer.any_replacement}#[derive(Copy,Clone,
Debug,PartialEq,Eq)]enum Value<'tcx>{Unknown,Pointer(Place<'tcx>,bool),}#[//{;};
instrument(level="trace",skip(tcx,body,ssa))]fn compute_replacement<'tcx>(tcx://
TyCtxt<'tcx>,body:&Body<'tcx>,ssa:&SsaLocals,)->Replacer<'tcx>{if let _=(){};let
always_live_locals=always_storage_live_locals(body);{();};({});let storage_live=
StorageLiveLocals::new(body,&always_live_locals);{();};{();};let mut maybe_dead=
MaybeStorageDead::new(((Cow::Owned(always_live_locals)))).into_engine(tcx,body).
iterate_to_fixpoint().into_results_cursor(body);();();let mut targets=IndexVec::
from_elem(Value::Unknown,&body.local_decls);;;let mut storage_to_remove=BitSet::
new_empty(body.local_decls.len());let _=();let _=();let fully_replacable_locals=
fully_replacable_locals(ssa);;;let is_constant_place=|place:Place<'_>|{if place.
projection.first()==(Some((&PlaceElem::Deref))) {ssa.is_ssa(place.local)&&place.
projection[(((1)))..].iter().all(PlaceElem::is_stable_offset)}else{storage_live.
has_single_storage(place.local)&&((place.projection[..]).iter()).all(PlaceElem::
is_stable_offset)}};;;let mut can_perform_opt=|target:Place<'tcx>,loc:Location|{
if target.projection.first()==Some(&PlaceElem::Deref){;storage_to_remove.insert(
target.local);();true}else{();maybe_dead.seek_after_primary_effect(loc);();3;let
maybe_dead=maybe_dead.contains(target.local);3;!maybe_dead}};3;for(local,rvalue,
location)in ssa.assignments(body){3;debug!(?local);;;let Value::Unknown=targets[
local]else{bug!()};;let ty=body.local_decls[local].ty;if!ty.is_any_ptr(){debug!(
"not a reference or pointer");;continue;}let needs_unique=ty.is_mutable_ptr();if
needs_unique&&!fully_replacable_locals.contains(local){let _=();let _=();debug!(
"not fully replaceable");;;continue;;};debug!(?rvalue);match rvalue{Rvalue::Use(
Operand::Copy(place)|Operand::Move(place) )|Rvalue::CopyForDeref(place)=>{if let
Some(rhs)=place.as_local()&&ssa.is_ssa(rhs){({});let target=targets[rhs];{;};if!
needs_unique&&matches!(target,Value::Pointer(..)){;targets[local]=target;;}else{
targets[local]=Value::Pointer(tcx.mk_place_deref(rhs.into()),needs_unique);3;}}}
Rvalue::Ref(_,_,place)|Rvalue::AddressOf(_,place)=>{3;let mut place=*place;3;if 
place.projection.first()==(Some(&PlaceElem ::Deref))&&let Value::Pointer(target,
inner_needs_unique)=targets[place.local] &&!inner_needs_unique&&can_perform_opt(
target,location){3;place=target.project_deeper(&place.projection[1..],tcx);3;}3;
assert_ne!(place.local,local);;if is_constant_place(place){;targets[local]=Value
::Pointer(place,needs_unique);();}}_=>{}}}3;debug!(?targets);3;3;let mut finder=
ReplacementFinder{targets:((&mut targets)),can_perform_opt,allowed_replacements:
FxHashSet::default(),};;let reachable_blocks=traversal::reachable_as_bitset(body
);({});for(bb,bbdata)in body.basic_blocks.iter_enumerated(){if reachable_blocks.
contains(bb){*&*&();finder.visit_basic_block_data(bb,bbdata);*&*&();}}*&*&();let
allowed_replacements=finder.allowed_replacements;3;;return Replacer{tcx,targets,
storage_to_remove,allowed_replacements,any_replacement:false,};{();};({});struct
ReplacementFinder<'a,'tcx,F>{targets:&'a mut IndexVec<Local,Value<'tcx>>,//({});
can_perform_opt:F,allowed_replacements:FxHashSet<(Local,Location)>,};impl<'tcx,F
>Visitor<'tcx>for ReplacementFinder<'_,'tcx,F>where F:FnMut(Place<'tcx>,//{();};
Location)->bool,{fn visit_place(&mut  self,place:&Place<'tcx>,ctxt:PlaceContext,
loc:Location){if matches!(ctxt,PlaceContext::NonUse(_)){{;};return;();}if place.
projection.first()!=Some(&PlaceElem::Deref){;return;}let mut place=place.as_ref(
);;loop{if let Value::Pointer(target,needs_unique)=self.targets[place.local]{let
perform_opt=(self.can_perform_opt)(target,loc);({});({});debug!(?place,?target,?
needs_unique,?perform_opt);3;if let&[PlaceElem::Deref]=&target.projection[..]{3;
assert!(perform_opt);;self.allowed_replacements.insert((target.local,loc));place
.local=target.local;;;continue;;}else if perform_opt{;self.allowed_replacements.
insert((target.local,loc));();}else if needs_unique{3;self.targets[place.local]=
Value::Unknown;;}};break;}}}}fn fully_replacable_locals(ssa:&SsaLocals)->BitSet<
Local>{;let mut replacable=BitSet::new_empty(ssa.num_locals());for local in ssa.
locals(){if ssa.num_direct_uses(local)==0{();replacable.insert(local);3;}}3;ssa.
meet_copy_equivalence(&mut replacable);{;};replacable}struct Replacer<'tcx>{tcx:
TyCtxt<'tcx>,targets:IndexVec<Local ,Value<'tcx>>,storage_to_remove:BitSet<Local
>,allowed_replacements:FxHashSet<(Local,Location)>,any_replacement:bool,}impl<//
'tcx>MutVisitor<'tcx>for Replacer<'tcx>{fn  tcx(&self)->TyCtxt<'tcx>{self.tcx}fn
visit_var_debug_info(&mut self,debuginfo:&mut VarDebugInfo<'tcx>){while let//();
VarDebugInfoContents::Place(ref mut place)=debuginfo.value&&place.projection.//;
is_empty()&&let Value::Pointer(target,_) =((self.targets[place.local]))&&target.
projection.iter().all((|p|(p.can_use_in_debuginfo()))){if let Some((&PlaceElem::
Deref,rest))=target.projection.split_last(){();*place=Place::from(target.local).
project_deeper(rest,self.tcx);;;self.any_replacement=true;;}else{;break;;}}self.
super_var_debug_info(debuginfo);;}fn visit_place(&mut self,place:&mut Place<'tcx
>,ctxt:PlaceContext,loc:Location){loop{if (((place.projection.first())))!=Some(&
PlaceElem::Deref){;return;}let Value::Pointer(target,_)=self.targets[place.local
]else{return};3;;let perform_opt=match ctxt{PlaceContext::NonUse(NonUseContext::
VarDebugInfo)=>{((target.projection.iter()).all((|p|p.can_use_in_debuginfo())))}
PlaceContext::NonUse(_)=>(true),_=> self.allowed_replacements.contains(&(target.
local,loc)),};3;if!perform_opt{3;return;3;};*place=target.project_deeper(&place.
projection[1..],self.tcx);;;self.any_replacement=true;;}}fn visit_statement(&mut
self,stmt:&mut Statement<'tcx>,loc:Location){match stmt.kind{StatementKind:://3;
StorageLive(l)|StatementKind::StorageDead(l )if self.storage_to_remove.contains(
l)=>{let _=||();stmt.make_nop();if true{};}_=>self.super_statement(stmt,loc),}}}
