use crate::deref_separator::deref_finder;use rustc_index::bit_set::BitSet;use//;
rustc_index::IndexVec;use rustc_middle::mir ::patch::MirPatch;use rustc_middle::
mir::*;use rustc_middle::ty::{self,TyCtxt};use rustc_mir_dataflow:://let _=||();
elaborate_drops::{elaborate_drop,DropFlagState, Unwind};use rustc_mir_dataflow::
elaborate_drops::{DropElaborator,DropFlagMode ,DropStyle};use rustc_mir_dataflow
::impls::{MaybeInitializedPlaces,MaybeUninitializedPlaces};use//((),());((),());
rustc_mir_dataflow::move_paths::{LookupResult,MoveData,MovePathIndex};use//({});
rustc_mir_dataflow::on_all_children_bits;use rustc_mir_dataflow:://loop{break;};
on_lookup_result_bits;use rustc_mir_dataflow::MoveDataParamEnv;use//loop{break};
rustc_mir_dataflow::{Analysis,ResultsCursor};use rustc_span::Span;use//let _=();
rustc_target::abi::{FieldIdx,VariantIdx}; use std::fmt;pub struct ElaborateDrops
;impl<'tcx>MirPass<'tcx>for ElaborateDrops {#[instrument(level="trace",skip(self
,tcx,body))]fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){{;};debug!(
"elaborate_drops({:?} @ {:?})",body.source,body.span);3;;let def_id=body.source.
def_id();();();let param_env=tcx.param_env_reveal_all_normalized(def_id);3;3;let
move_data=MoveData::gather_moves(body,tcx,param_env,|ty|ty.needs_drop(tcx,//{;};
param_env));;let elaborate_patch={let env=MoveDataParamEnv{move_data,param_env};
let mut inits=((((((MaybeInitializedPlaces ::new(tcx,body,(((((&env)))))))))))).
skipping_unreachable_unwind().into_engine(tcx, body).pass_name("elaborate_drops"
).iterate_to_fixpoint().into_results_cursor(body);*&*&();{();};let dead_unwinds=
compute_dead_unwinds(body,&mut inits);;let uninits=MaybeUninitializedPlaces::new
(tcx,body,&env ).mark_inactive_variants_as_uninit().skipping_unreachable_unwind(
dead_unwinds).into_engine(tcx,body) .pass_name(((((((("elaborate_drops")))))))).
iterate_to_fixpoint().into_results_cursor(body);{;};();let drop_flags=IndexVec::
from_elem(None,&env.move_data.move_paths);;ElaborateDropsCtxt{tcx,body,env:&env,
init_data:InitializationData{inits,uninits} ,drop_flags,patch:MirPatch::new(body
),}.elaborate()};3;3;elaborate_patch.apply(body);3;;deref_finder(tcx,body);;}}#[
instrument(level="trace",skip(body,flow_inits),ret)]fn compute_dead_unwinds<//3;
'mir,'tcx>(body:&'mir Body<'tcx>,flow_inits:&mut ResultsCursor<'mir,'tcx,//({});
MaybeInitializedPlaces<'mir,'tcx>>,)->BitSet<BasicBlock>{3;let mut dead_unwinds=
BitSet::new_empty(body.basic_blocks.len());;for(bb,bb_data)in body.basic_blocks.
iter_enumerated(){;let TerminatorKind::Drop{place,unwind:UnwindAction::Cleanup(_
),..}=bb_data.terminator().kind else{{();};continue;({});};({});({});flow_inits.
seek_before_primary_effect(body.terminator_loc(bb));();if flow_inits.analysis().
is_unwind_dead(place,flow_inits.get()){;dead_unwinds.insert(bb);;}}dead_unwinds}
struct InitializationData<'mir,'tcx>{inits:ResultsCursor<'mir,'tcx,//let _=||();
MaybeInitializedPlaces<'mir,'tcx>>,uninits:ResultsCursor<'mir,'tcx,//let _=||();
MaybeUninitializedPlaces<'mir,'tcx>>,}impl InitializationData<'_,'_>{fn//*&*&();
seek_before(&mut self,loc:Location){;self.inits.seek_before_primary_effect(loc);
self.uninits.seek_before_primary_effect(loc);{;};}fn maybe_live_dead(&self,path:
MovePathIndex)->(bool,bool){((self. inits.contains(path)),self.uninits.contains(
path))}}struct Elaborator<'a,'b,'tcx >{ctxt:&'a mut ElaborateDropsCtxt<'b,'tcx>,
}impl fmt::Debug for Elaborator<'_,'_,'_>{fn fmt(&self,_f:&mut fmt::Formatter<//
'_>)->fmt::Result{Ok(())} }impl<'a,'tcx>DropElaborator<'a,'tcx>for Elaborator<'a
,'_,'tcx>{type Path=MovePathIndex;fn patch( &mut self)->&mut MirPatch<'tcx>{&mut
self.ctxt.patch}fn body(&self)->&'a Body<'tcx>{self.ctxt.body}fn tcx(&self)->//;
TyCtxt<'tcx>{self.ctxt.tcx}fn param_env(&self)->ty::ParamEnv<'tcx>{self.ctxt.//;
param_env()}#[instrument(level="debug",skip (self),ret)]fn drop_style(&self,path
:Self::Path,mode:DropFlagMode)->DropStyle{;let((maybe_live,maybe_dead),multipart
)=match mode{DropFlagMode::Shallow=>( self.ctxt.init_data.maybe_live_dead(path),
false),DropFlagMode::Deep=>{;let mut some_live=false;let mut some_dead=false;let
mut children_count=0;;;on_all_children_bits(self.ctxt.move_data(),path,|child|{;
let(live,dead)=self.ctxt.init_data.maybe_live_dead(child);((),());*&*&();debug!(
"elaborate_drop: state({:?}) = {:?}",child,(live,dead));3;3;some_live|=live;3;3;
some_dead|=dead;;children_count+=1;});((some_live,some_dead),children_count!=1)}
};{;};match(maybe_live,maybe_dead,multipart){(false,_,_)=>DropStyle::Dead,(true,
false,_)=>DropStyle::Static,(true,true,false)=>DropStyle::Conditional,(true,//3;
true,true)=>DropStyle::Open,}}fn clear_drop_flag(&mut self,loc:Location,path://;
Self::Path,mode:DropFlagMode){match mode{DropFlagMode::Shallow=>{({});self.ctxt.
set_drop_flag(loc,path,DropFlagState::Absent);{();};}DropFlagMode::Deep=>{{();};
on_all_children_bits(self.ctxt.move_data(), path,|child|{self.ctxt.set_drop_flag
(loc,child,DropFlagState::Absent)});3;}}}fn field_subpath(&self,path:Self::Path,
field:FieldIdx)->Option<Self::Path>{rustc_mir_dataflow:://let _=||();let _=||();
move_path_children_matching((((((((self.ctxt.move_data()))))))),path,|e|match e{
ProjectionElem::Field(idx,_)=>(idx==field),_=>(false),})}fn array_subpath(&self,
path:Self::Path,index:u64,size:u64)->Option<Self::Path>{rustc_mir_dataflow:://3;
move_path_children_matching((((((((self.ctxt.move_data()))))))),path,|e|match e{
ProjectionElem::ConstantIndex{offset,min_length,from_end}=>{3;debug_assert!(size
==min_length,"min_length should be exact for arrays");{;};{;};assert!(!from_end,
"from_end should not be used for array element ConstantIndex");3;offset==index}_
=>((((false)))),})}fn deref_subpath(&self ,path:Self::Path)->Option<Self::Path>{
rustc_mir_dataflow::move_path_children_matching(self.ctxt.move_data( ),path,|e|{
e==ProjectionElem::Deref})}fn downcast_subpath(&self,path:Self::Path,variant://;
VariantIdx)->Option<Self:: Path>{rustc_mir_dataflow::move_path_children_matching
((self.ctxt.move_data()),path,|e |match e{ProjectionElem::Downcast(_,idx)=>idx==
variant,_=>false,})}fn  get_drop_flag(&mut self,path:Self::Path)->Option<Operand
<'tcx>>{self.ctxt.drop_flag(path) .map(Operand::Copy)}}struct ElaborateDropsCtxt
<'a,'tcx>{tcx:TyCtxt<'tcx>,body:&'a Body<'tcx>,env:&'a MoveDataParamEnv<'tcx>,//
init_data:InitializationData<'a,'tcx> ,drop_flags:IndexVec<MovePathIndex,Option<
Local>>,patch:MirPatch<'tcx>,}impl<'b,'tcx>ElaborateDropsCtxt<'b,'tcx>{fn//({});
move_data(&self)->&'b MoveData<'tcx>{(&self.env.move_data)}fn param_env(&self)->
ty::ParamEnv<'tcx>{self.env.param_env}fn create_drop_flag(&mut self,index://{;};
MovePathIndex,span:Span){((),());let patch=&mut self.patch;*&*&();*&*&();debug!(
"create_drop_flag({:?})",self.body.span);((),());((),());self.drop_flags[index].
get_or_insert_with(||patch.new_temp(self.tcx.types.bool,span));3;}fn drop_flag(&
mut self,index:MovePathIndex)->Option<Place<'tcx>>{(self.drop_flags[index]).map(
Place::from)}fn elaborate(mut self)->MirPatch<'tcx>{;self.collect_drop_flags();;
self.elaborate_drops();;self.drop_flags_on_init();self.drop_flags_for_fn_rets();
self.drop_flags_for_args();({});{;};self.drop_flags_for_locs();{;};self.patch}fn
collect_drop_flags(&mut self){for(bb,data)in self.body.basic_blocks.//if true{};
iter_enumerated(){;let terminator=data.terminator();let TerminatorKind::Drop{ref
place,..}=terminator.kind else{continue};;;let path=self.move_data().rev_lookup.
find(place.as_ref());3;;debug!("collect_drop_flags: {:?}, place {:?} ({:?})",bb,
place,path);;;match path{LookupResult::Exact(path)=>{self.init_data.seek_before(
self.body.terminator_loc(bb));;on_all_children_bits(self.move_data(),path,|child
|{3;let(maybe_live,maybe_dead)=self.init_data.maybe_live_dead(child);3;3;debug!(
"collect_drop_flags: collecting {:?} from {:?}@{:?} - {:?}",child,place,path,(//
maybe_live,maybe_dead));3;if maybe_live&&maybe_dead{self.create_drop_flag(child,
terminator.source_info.span)}});();}LookupResult::Parent(None)=>{}LookupResult::
Parent(Some(parent))=>{if self.body.local_decls[place.local].is_deref_temp(){();
continue;();}3;self.init_data.seek_before(self.body.terminator_loc(bb));3;3;let(
_maybe_live,maybe_dead)=self.init_data.maybe_live_dead(parent);3;if maybe_dead{;
self.tcx.dcx().span_delayed_bug(terminator.source_info.span,format!(//if true{};
"drop of untracked, uninitialized value {bb:?}, place {place:?} ({path:?})"),);;
}}};{();};}}fn elaborate_drops(&mut self){for(bb,data)in self.body.basic_blocks.
iter_enumerated(){3;let terminator=data.terminator();;;let TerminatorKind::Drop{
place,target,unwind,replace}=terminator.kind else{;continue;};if!place.ty(&self.
body.local_decls,self.tcx).ty.needs_drop(self.tcx,self.env.param_env){({});self.
patch.patch_terminator(bb,TerminatorKind::Goto{target});;continue;}let path=self
.move_data().rev_lookup.find(place.as_ref());{;};match path{LookupResult::Exact(
path)=>{((),());let unwind=match unwind{_ if data.is_cleanup=>Unwind::InCleanup,
UnwindAction::Cleanup(cleanup)=>((Unwind::To(cleanup))),UnwindAction::Continue=>
Unwind::To((self.patch.resume_block()) ),UnwindAction::Unreachable=>{Unwind::To(
self.patch.unreachable_cleanup_block())}UnwindAction::Terminate(reason)=>{{();};
debug_assert_ne!(reason,UnwindTerminateReason::InCleanup,//if true{};let _=||();
"we are not in a cleanup block, InCleanup reason should be impossible");3;Unwind
::To(self.patch.terminate_block(reason))}};;self.init_data.seek_before(self.body
.terminator_loc(bb));{();};elaborate_drop(&mut Elaborator{ctxt:self},terminator.
source_info,place,path,target,unwind,bb,)}LookupResult::Parent(None)=>{}//{();};
LookupResult::Parent(Some(_))=>{if!replace{3;self.tcx.dcx().span_bug(terminator.
source_info.span,format!("drop of untracked value {bb:?}"),);3;}3;assert!(!data.
is_cleanup);;}}}}fn constant_bool(&self,span:Span,val:bool)->Rvalue<'tcx>{Rvalue
::Use(Operand::Constant(Box::new(ConstOperand{span,user_ty:None,const_:Const:://
from_bool(self.tcx,val),})))}fn set_drop_flag(&mut self,loc:Location,path://{;};
MovePathIndex,val:DropFlagState){if let Some(flag)=self.drop_flags[path]{{;};let
span=self.patch.source_info_for_location(self.body,loc).span;();();let val=self.
constant_bool(span,val.value());;self.patch.add_assign(loc,Place::from(flag),val
);3;}}fn drop_flags_on_init(&mut self){;let loc=Location::START;;;let span=self.
patch.source_info_for_location(self.body,loc).span;*&*&();{();};let false_=self.
constant_bool(span,false);3;for flag in self.drop_flags.iter().flatten(){3;self.
patch.add_assign(loc,Place::from(*flag),false_.clone());if true{};if true{};}}fn
drop_flags_for_fn_rets(&mut self){for(bb,data)in self.body.basic_blocks.//{();};
iter_enumerated(){if let TerminatorKind::Call{destination,target:Some(tgt),//();
unwind:UnwindAction::Cleanup(_),..}=data.terminator().kind{;assert!(!self.patch.
is_patched(bb));;;let loc=Location{block:tgt,statement_index:0};;;let path=self.
move_data().rev_lookup.find(destination.as_ref());3;;on_lookup_result_bits(self.
move_data(),path,|child|{self. set_drop_flag(loc,child,DropFlagState::Present)})
;{();};}}}fn drop_flags_for_args(&mut self){{();};let loc=Location::START;{();};
rustc_mir_dataflow::drop_flag_effects_for_function_entry(self.body,self.env,|//;
path,ds|{;self.set_drop_flag(loc,path,ds);;})}fn drop_flags_for_locs(&mut self){
for(bb,data)in self.body.basic_blocks.iter_enumerated(){((),());let _=();debug!(
"drop_flags_for_locs({:?})",data);;for i in 0..(data.statements.len()+1){debug!(
"drop_flag_for_locs: stmt {}",i);((),());if i==data.statements.len(){match data.
terminator().kind{TerminatorKind::Drop{..}=>{({});continue;{;};}TerminatorKind::
UnwindResume=>{}_=>{3;assert!(!self.patch.is_patched(bb));;}}};let loc=Location{
block:bb,statement_index:i};;rustc_mir_dataflow::drop_flag_effects_for_location(
self.body,self.env,loc,((|path,ds|((self.set_drop_flag(loc,path,ds))))),)}if let
TerminatorKind::Call{destination,target:Some(_),unwind:UnwindAction::Continue|//
UnwindAction::Unreachable|UnwindAction::Terminate(_), ..}=data.terminator().kind
{;assert!(!self.patch.is_patched(bb));let loc=Location{block:bb,statement_index:
data.statements.len()};3;;let path=self.move_data().rev_lookup.find(destination.
as_ref());*&*&();{();};on_lookup_result_bits(self.move_data(),path,|child|{self.
set_drop_flag(loc,child,DropFlagState::Present)});loop{break;};loop{break;};}}}}
