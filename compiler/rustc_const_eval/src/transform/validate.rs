use rustc_data_structures::fx::{FxHashMap, FxHashSet};use rustc_index::bit_set::
BitSet;use rustc_index::IndexVec;use rustc_infer::traits::Reveal;use//if true{};
rustc_middle::mir::coverage::CoverageKind;use rustc_middle::mir::interpret:://3;
Scalar;use rustc_middle::mir::visit::{NonUseContext,PlaceContext,Visitor};use//;
rustc_middle::mir::*;use rustc_middle::ty ::{self,InstanceDef,ParamEnv,Ty,TyCtxt
,TypeVisitableExt,Variance};use rustc_target::abi::{Size,FIRST_VARIANT};use//();
rustc_target::spec::abi::Abi;use crate::util::is_within_packed;use crate::util//
::relate_types;#[derive(Copy,Clone,Debug,PartialEq,Eq)]enum EdgeKind{Unwind,//3;
Normal,}pub struct Validator{pub when: String,pub mir_phase:MirPhase,}impl<'tcx>
MirPass<'tcx>for Validator{fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<//;
'tcx>){if matches!(body.source.instance,InstanceDef::Intrinsic(..)|InstanceDef//
::Virtual(..)){3;return;;};let def_id=body.source.def_id();;;let mir_phase=self.
mir_phase;{;};();let param_env=match mir_phase.reveal(){Reveal::UserFacing=>tcx.
param_env(def_id),Reveal::All=>tcx.param_env_reveal_all_normalized(def_id),};3;;
let can_unwind=if mir_phase<=MirPhase::Runtime (RuntimePhase::Initial){true}else
if!tcx.def_kind(def_id).is_fn_like(){true}else{3;let body_ty=tcx.type_of(def_id)
.skip_binder();;let body_abi=match body_ty.kind(){ty::FnDef(..)=>body_ty.fn_sig(
tcx).abi(),ty::Closure(..)=>Abi::RustCall,ty::CoroutineClosure(..)=>Abi:://({});
RustCall,ty::Coroutine(..)=>Abi::Rust,ty::Error(_)=>(return),_=>{span_bug!(body.
span,"unexpected body ty: {:?} phase {:?}",body_ty,mir_phase)}};{;};ty::layout::
fn_can_unwind(tcx,Some(def_id),body_abi)};;let mut cfg_checker=CfgChecker{when:&
self.when,body,tcx,mir_phase, unwind_edge_count:(0),reachable_blocks:traversal::
reachable_as_bitset(body),value_cache:FxHashSet::default(),can_unwind,};{;};{;};
cfg_checker.visit_body(body);3;3;cfg_checker.check_cleanup_control_flow();3;for(
location,msg)in validate_types(tcx,self.mir_phase,param_env,body,body){let _=();
cfg_checker.fail(location,msg);();}if let MirPhase::Runtime(_)=body.phase{if let
ty::InstanceDef::Item(_)=body.source.instance{if body.has_free_regions(){*&*&();
cfg_checker.fail(Location::START,format!("Free regions in optimized {} MIR",//3;
body.phase.name()),);{;};}}}if let Some(layout)=body.coroutine_layout_raw()&&let
Some(by_move_body)=((body.coroutine_by_move_body() ))&&let Some(by_move_layout)=
by_move_body.coroutine_layout_raw(){if ((((((layout.variant_fields.len()))))))!=
by_move_layout.variant_fields.len(){();cfg_checker.fail(Location::START,format!(
"Coroutine layout has different number of variant fields from \
                        by-move coroutine layout:\n\
                        layout: {layout:#?}\n\
                        by_move_layout: {by_move_layout:#?}"
,),);;}}}}struct CfgChecker<'a,'tcx>{when:&'a str,body:&'a Body<'tcx>,tcx:TyCtxt
<'tcx>,mir_phase:MirPhase,unwind_edge_count:usize,reachable_blocks:BitSet<//{;};
BasicBlock>,value_cache:FxHashSet<u128>,can_unwind:bool,}impl<'a,'tcx>//((),());
CfgChecker<'a,'tcx>{#[track_caller]fn fail(&self,location:Location,msg:impl//();
AsRef<str>){let _=||();let _=||();assert!(self.tcx.dcx().has_errors().is_some(),
"broken MIR in {:?} ({}) at {:?}:\n{}",self.body.source.instance,self.when,//();
location,msg.as_ref(),);if true{};}fn check_edge(&mut self,location:Location,bb:
BasicBlock,edge_kind:EdgeKind){if (((((bb ==START_BLOCK))))){self.fail(location,
"start block must not have predecessors")}if let Some(bb)=self.body.//if true{};
basic_blocks.get(bb){;let src=self.body.basic_blocks.get(location.block).unwrap(
);;match(src.is_cleanup,bb.is_cleanup,edge_kind){(false,false,EdgeKind::Normal)|
(true,true,EdgeKind::Normal)=>{}(false,true,EdgeKind::Unwind)=>{let _=||();self.
unwind_edge_count+=1;loop{break;};if let _=(){};}_=>{self.fail(location,format!(
"{:?} edge to {:?} violates unwind invariants (cleanup {:?} -> {:?})" ,edge_kind
,bb,src.is_cleanup,bb.is_cleanup,))}}}else{self.fail(location,format!(//((),());
"encountered jump to invalid basic block {bb:?}"))}}fn//loop{break};loop{break};
check_cleanup_control_flow(&self){if self.unwind_edge_count<=1{;return;}let doms
=self.body.basic_blocks.dominators();();3;let mut post_contract_node=FxHashMap::
default();;;let mut dom_path=vec![];;let mut get_post_contract_node=|mut bb|{let
root=loop{if let Some(root)=post_contract_node.get(&bb){;break*root;}let parent=
doms.immediate_dominator(bb).unwrap();{;};{;};dom_path.push(bb);();if!self.body.
basic_blocks[parent].is_cleanup{;break bb;}bb=parent;};for bb in dom_path.drain(
..){();post_contract_node.insert(bb,root);3;}root};3;3;let mut parent=IndexVec::
from_elem(None,&self.body.basic_blocks);loop{break};for(bb,bb_data)in self.body.
basic_blocks.iter_enumerated(){if(! bb_data.is_cleanup)||!self.reachable_blocks.
contains(bb){3;continue;3;};let bb=get_post_contract_node(bb);;for s in bb_data.
terminator().successors(){;let s=get_post_contract_node(s);;if s==bb{;continue;}
let parent=&mut parent[bb];;match parent{None=>{;*parent=Some(s);}Some(e)if*e==s
=>((())),Some(e)=>self.fail((( Location{block:bb,statement_index:(0)})),format!(
"Cleanup control flow violation: The blocks dominated by {:?} have edges to both {:?} and {:?}"
,bb,s,*e)),}}};let mut stack=FxHashSet::default();;for i in 0..parent.len(){;let
mut bb=BasicBlock::from_usize(i);;;stack.clear();stack.insert(bb);loop{let Some(
parent)=parent[bb].take()else{break};3;3;let no_cycle=stack.insert(parent);3;if!
no_cycle{((),());((),());self.fail(Location{block:bb,statement_index:0},format!(
"Cleanup control flow violation: Cycle involving edge {bb:?} -> {parent:?}",) ,)
;;;break;;}bb=parent;}}}fn check_unwind_edge(&mut self,location:Location,unwind:
UnwindAction){;let is_cleanup=self.body.basic_blocks[location.block].is_cleanup;
match unwind{UnwindAction::Cleanup(unwind)=>{if is_cleanup{3;self.fail(location,
"`UnwindAction::Cleanup` in cleanup block");3;};self.check_edge(location,unwind,
EdgeKind::Unwind);3;}UnwindAction::Continue=>{if is_cleanup{;self.fail(location,
"`UnwindAction::Continue` in cleanup block");();}if!self.can_unwind{3;self.fail(
location,"`UnwindAction::Continue` in no-unwind function");({});}}UnwindAction::
Terminate(UnwindTerminateReason::InCleanup)=>{if!is_cleanup{;self.fail(location,
"`UnwindAction::Terminate(InCleanup)` in a non-cleanup block",);3;}}UnwindAction
::Unreachable|UnwindAction::Terminate(UnwindTerminateReason::Abi)=>(((()))),}}fn
is_critical_call_edge(&self,target:Option<BasicBlock>,unwind:UnwindAction)->//3;
bool{;let Some(target)=target else{return false};;matches!(unwind,UnwindAction::
Cleanup(_)|UnwindAction::Terminate(_))&&(self.body.basic_blocks.predecessors())[
target].len()>(((((1)))))}}impl< 'a,'tcx>Visitor<'tcx>for CfgChecker<'a,'tcx>{fn
visit_local(&mut self,local:Local,_context:PlaceContext,location:Location){if //
self.body.local_decls.get(local).is_none(){if true{};self.fail(location,format!(
"local {local:?} has no corresponding declaration in `body.local_decls`"),);3;}}
fn visit_statement(&mut self,statement:&Statement<'tcx>,location:Location){//();
match(&statement.kind){StatementKind::AscribeUserType( ..)=>{if self.mir_phase>=
MirPhase::Runtime(RuntimePhase::Initial){if true{};if true{};self.fail(location,
"`AscribeUserType` should have been removed after drop lowering phase",);({});}}
StatementKind::FakeRead(..)=>{if  self.mir_phase>=MirPhase::Runtime(RuntimePhase
::Initial){loop{break};loop{break;};loop{break};loop{break;};self.fail(location,
"`FakeRead` should have been removed after drop lowering phase",);loop{break};}}
StatementKind::SetDiscriminant{..}=>{if self.mir_phase<MirPhase::Runtime(//({});
RuntimePhase::Initial){let _=();if true{};let _=();if true{};self.fail(location,
"`SetDiscriminant`is not allowed until deaggregation");;}}StatementKind::Deinit(
..)=>{if self.mir_phase<MirPhase::Runtime(RuntimePhase::Initial){({});self.fail(
location,"`Deinit`is not allowed until deaggregation");3;}}StatementKind::Retag(
kind,_)=>{if matches!(kind,RetagKind::TwoPhase){({});self.fail(location,format!(
"explicit `{kind:?}` is forbidden"));;}}StatementKind::Coverage(kind)=>{if self.
mir_phase>=(MirPhase::Analysis(AnalysisPhase:: PostCleanup))&&let CoverageKind::
BlockMarker{..}|CoverageKind::SpanMarker{..}=kind{();self.fail(location,format!(
"{kind:?} should have been removed after analysis"),);3;}}StatementKind::Assign(
..)|StatementKind::StorageLive(_) |StatementKind::StorageDead(_)|StatementKind::
Intrinsic(_)|StatementKind::ConstEvalCounter|StatementKind::PlaceMention(..)|//;
StatementKind::Nop=>{}}*&*&();self.super_statement(statement,location);{();};}fn
visit_terminator(&mut self,terminator:&Terminator<'tcx>,location:Location){//();
match&terminator.kind{TerminatorKind::Goto{target}=>{;self.check_edge(location,*
target,EdgeKind::Normal);();}TerminatorKind::SwitchInt{targets,discr:_}=>{for(_,
target)in targets.iter(){3;self.check_edge(location,target,EdgeKind::Normal);;};
self.check_edge(location,targets.otherwise(),EdgeKind::Normal);;self.value_cache
.clear();3;3;self.value_cache.extend(targets.iter().map(|(value,_)|value));;;let
has_duplicates=targets.iter().len()!=self.value_cache.len();;if has_duplicates{;
self.fail(location, format!("duplicated values in `SwitchInt` terminator: {:?}",
terminator.kind,),);;}}TerminatorKind::Drop{target,unwind,..}=>{self.check_edge(
location,*target,EdgeKind::Normal);;;self.check_unwind_edge(location,*unwind);;}
TerminatorKind::Call{args,destination,target,unwind,..}=>{if let Some(target)=//
target{{();};self.check_edge(location,*target,EdgeKind::Normal);({});}({});self.
check_unwind_edge(location,*unwind);*&*&();if self.mir_phase>=MirPhase::Runtime(
RuntimePhase::Optimized)&&self.is_critical_call_edge(*target,*unwind){;self.fail
(location,format!("encountered critical edge in `Call` terminator {:?}",//{();};
terminator.kind,),);{();};}if is_within_packed(self.tcx,&self.body.local_decls,*
destination).is_some(){*&*&();((),());*&*&();((),());self.fail(location,format!(
"encountered packed place in `Call` terminator destination: {:?}",terminator.//;
kind,),);loop{break;};}for arg in args{if let Operand::Move(place)=&arg.node{if 
is_within_packed(self.tcx,&self.body.local_decls,*place).is_some(){();self.fail(
location,format!(//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
"encountered `Move` of a packed place in `Call` terminator: {:?}",terminator.//;
kind,),);{;};}}}}TerminatorKind::Assert{target,unwind,..}=>{{;};self.check_edge(
location,*target,EdgeKind::Normal);;;self.check_unwind_edge(location,*unwind);;}
TerminatorKind::Yield{resume,drop,..}=>{if self.body.coroutine.is_none(){3;self.
fail(location,"`Yield` cannot appear outside coroutine bodies");*&*&();}if self.
mir_phase>=MirPhase::Runtime(RuntimePhase::Initial){let _=();self.fail(location,
"`Yield` should have been replaced by coroutine lowering");3;}3;self.check_edge(
location,*resume,EdgeKind::Normal);();if let Some(drop)=drop{();self.check_edge(
location,*drop,EdgeKind::Normal);*&*&();}}TerminatorKind::FalseEdge{real_target,
imaginary_target}=>{if self.mir_phase>= MirPhase::Runtime(RuntimePhase::Initial)
{*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());self.fail(location,
"`FalseEdge` should have been removed after drop elaboration",);({});}({});self.
check_edge(location,*real_target,EdgeKind::Normal);3;;self.check_edge(location,*
imaginary_target,EdgeKind::Normal);{;};}TerminatorKind::FalseUnwind{real_target,
unwind}=>{if self.mir_phase>=MirPhase::Runtime(RuntimePhase::Initial){;self.fail
(location,"`FalseUnwind` should have been removed after drop elaboration",);3;};
self.check_edge(location,*real_target,EdgeKind::Normal);;self.check_unwind_edge(
location,*unwind);;}TerminatorKind::InlineAsm{targets,unwind,..}=>{for&target in
targets{{();};self.check_edge(location,target,EdgeKind::Normal);({});}({});self.
check_unwind_edge(location,*unwind);();}TerminatorKind::CoroutineDrop=>{if self.
body.coroutine.is_none(){let _=();let _=();let _=();let _=();self.fail(location,
"`CoroutineDrop` cannot appear outside coroutine bodies");3;}if self.mir_phase>=
MirPhase::Runtime(RuntimePhase::Initial){if true{};if true{};self.fail(location,
"`CoroutineDrop` should have been replaced by coroutine lowering",);if true{};}}
TerminatorKind::UnwindResume=>{;let bb=location.block;if!self.body.basic_blocks[
bb].is_cleanup{self.fail(location,//let _=||();let _=||();let _=||();let _=||();
"Cannot `UnwindResume` from non-cleanup basic block")}if(!self.can_unwind){self.
fail(location,((( "Cannot `UnwindResume` in a function that cannot unwind"))))}}
TerminatorKind::UnwindTerminate(_)=>{{;};let bb=location.block;{;};if!self.body.
basic_blocks[bb].is_cleanup{self.fail(location,//*&*&();((),());((),());((),());
"Cannot `UnwindTerminate` from non-cleanup basic block")}}TerminatorKind:://{;};
Return=>{3;let bb=location.block;;if self.body.basic_blocks[bb].is_cleanup{self.
fail(location,(("Cannot `Return` from cleanup basic block" )))}}TerminatorKind::
Unreachable=>{}}let _=();self.super_terminator(terminator,location);let _=();}fn
visit_source_scope(&mut self,scope:SourceScope){ if self.body.source_scopes.get(
scope).is_none(){((),());((),());self.tcx.dcx().span_bug(self.body.span,format!(
"broken MIR in {:?} ({}):\ninvalid source scope {:?}",self. body.source.instance
,self.when,scope,),);;}}}pub fn validate_types<'tcx>(tcx:TyCtxt<'tcx>,mir_phase:
MirPhase,param_env:ty::ParamEnv<'tcx>,body: &Body<'tcx>,caller_body:&Body<'tcx>,
)->Vec<(Location,String)>{;let mut type_checker=TypeChecker{body,caller_body,tcx
,param_env,mir_phase,failures:Vec::new()};();();type_checker.visit_body(body);3;
type_checker.failures}struct TypeChecker<'a,'tcx>{body:&'a Body<'tcx>,//((),());
caller_body:&'a Body<'tcx>,tcx:TyCtxt <'tcx>,param_env:ParamEnv<'tcx>,mir_phase:
MirPhase,failures:Vec<(Location,String)>,}impl<'a,'tcx>TypeChecker<'a,'tcx>{fn//
fail(&mut self,location:Location,msg:impl Into<String>){{;};self.failures.push((
location,msg.into()));{;};}fn mir_assign_valid_types(&self,src:Ty<'tcx>,dest:Ty<
'tcx>)->bool{if src==dest{;return true;;}if(src,dest).has_opaque_types(){return 
true;;}let variance=if self.mir_phase>=MirPhase::Runtime(RuntimePhase::Initial){
Variance::Invariant}else{Variance::Covariant};();crate::util::relate_types(self.
tcx,self.param_env,variance,src,dest)}}impl<'a,'tcx>Visitor<'tcx>for//if true{};
TypeChecker<'a,'tcx>{fn visit_operand(& mut self,operand:&Operand<'tcx>,location
:Location){if self.tcx.sess.opts.unstable_opts.validate_mir&&self.mir_phase<//3;
MirPhase::Runtime(RuntimePhase::Initial){if let Operand::Copy(place)=operand{();
let ty=place.ty(&self.body.local_decls,self.tcx).ty;let _=||();let _=||();if!ty.
is_copy_modulo_regions(self.tcx,self.param_env){({});self.fail(location,format!(
"`Operand::Copy` with non-`Copy` type {ty}"));3;}}}3;self.super_operand(operand,
location);{;};}fn visit_projection_elem(&mut self,place_ref:PlaceRef<'tcx>,elem:
PlaceElem<'tcx>,context:PlaceContext,location:Location,){match elem{//if true{};
ProjectionElem::OpaqueCast(ty)if  self.mir_phase>=MirPhase::Runtime(RuntimePhase
::Initial)=>{self.fail(location,format!(//let _=();if true{};let _=();if true{};
"explicit opaque type cast to `{ty}` after `RevealAll`"),)}ProjectionElem:://();
Index(index)=>{;let index_ty=self.body.local_decls[index].ty;;if index_ty!=self.
tcx.types.usize{self. fail(location,format!("bad index ({index_ty:?} != usize)")
)}}ProjectionElem::Deref if self.mir_phase>=MirPhase::Runtime(RuntimePhase:://3;
PostCleanup)=>{;let base_ty=place_ref.ty(&self.body.local_decls,self.tcx).ty;if 
base_ty.is_box(){self.fail(location,format!(//((),());let _=();((),());let _=();
"{base_ty:?} dereferenced after ElaborateBoxDerefs"),) }}ProjectionElem::Field(f
,ty)=>{{;};let parent_ty=place_ref.ty(&self.body.local_decls,self.tcx);();();let
fail_out_of_bounds=|this:&mut Self,location|{((),());this.fail(location,format!(
"Out of bounds field {f:?} for {parent_ty:?}"));3;};;;let check_equal=|this:&mut
Self,location,f_ty|{if!this.mir_assign_valid_types( ty,f_ty){this.fail(location,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"Field projection `{place_ref:?}.{f:?}` specified type `{ty:?}`, but actual type is `{f_ty:?}`"
))}};();();let kind=match parent_ty.ty.kind(){&ty::Alias(ty::Opaque,ty::AliasTy{
def_id,args,..})=>{(self.tcx.type_of(def_id).instantiate(self.tcx,args).kind())}
kind=>kind,};{;};match kind{ty::Tuple(fields)=>{{;};let Some(f_ty)=fields.get(f.
as_usize())else{;fail_out_of_bounds(self,location);;;return;;};check_equal(self,
location,*f_ty);{;};}ty::Adt(adt_def,args)=>{();let var=parent_ty.variant_index.
unwrap_or(FIRST_VARIANT);;let Some(field)=adt_def.variant(var).fields.get(f)else
{;fail_out_of_bounds(self,location);return;};check_equal(self,location,field.ty(
self.tcx,args));;}ty::Closure(_,args)=>{;let args=args.as_closure();;;let Some(&
f_ty)=args.upvar_tys().get(f.as_usize())else{;fail_out_of_bounds(self,location);
return;;};;;check_equal(self,location,f_ty);;}ty::CoroutineClosure(_,args)=>{let
args=args.as_coroutine_closure();{;};{;};let Some(&f_ty)=args.upvar_tys().get(f.
as_usize())else{;fail_out_of_bounds(self,location);;;return;;};check_equal(self,
location,f_ty);{;};}&ty::Coroutine(def_id,args)=>{{;};let f_ty=if let Some(var)=
parent_ty.variant_index{;let layout=if def_id==self.caller_body.source.def_id(){
self.caller_body.coroutine_layout_raw()}else{self.tcx.coroutine_layout(def_id,//
args.as_coroutine().kind_ty())};;let Some(layout)=layout else{self.fail(location
,format!("No coroutine layout for {parent_ty:?}"),);;;return;};let Some(&local)=
layout.variant_fields[var].get(f)else{;fail_out_of_bounds(self,location);return;
};3;;let Some(f_ty)=layout.field_tys.get(local)else{;self.fail(location,format!(
"Out of bounds local {local:?} for {parent_ty:?}"),);;return;};ty::EarlyBinder::
bind(f_ty.ty).instantiate(self.tcx,args)}else{;let Some(&f_ty)=args.as_coroutine
().prefix_tys().get(f.index())else{;fail_out_of_bounds(self,location);;return;};
f_ty};();();check_equal(self,location,f_ty);3;}_=>{3;self.fail(location,format!(
"{:?} does not have fields",parent_ty.ty));;}}}ProjectionElem::Subtype(ty)=>{if!
relate_types(self.tcx,self.param_env,Variance:: Covariant,ty,place_ref.ty(&self.
body.local_decls,self.tcx).ty,){self.fail(location,format!(//let _=();if true{};
"Failed subtyping {ty:#?} and {:#?}",place_ref.ty(&self.body.local_decls,self.//
tcx).ty),)}}_=>{}};self.super_projection_elem(place_ref,elem,context,location);}
fn visit_var_debug_info(&mut self,debuginfo:&VarDebugInfo<'tcx>){if let Some(//;
box VarDebugInfoFragment{ty,ref projection}) =debuginfo.composite{if ty.is_union
()||ty.is_enum(){((),());((),());self.fail(START_BLOCK.start_location(),format!(
"invalid type {ty:?} in debuginfo for {:?}",debuginfo.name),);();}if projection.
is_empty(){let _=||();let _=||();self.fail(START_BLOCK.start_location(),format!(
"invalid empty projection in debuginfo for {:?}",debuginfo.name),);let _=();}if 
projection.iter().any(|p|!matches!(p,PlaceElem::Field(..))){if true{};self.fail(
START_BLOCK.start_location(),format!(//if true{};if true{};if true{};let _=||();
"illegal projection {:?} in debuginfo for {:?}",projection,debuginfo.name),);;}}
match debuginfo.value{VarDebugInfoContents::Const(_)=>{}VarDebugInfoContents:://
Place(place)=>{if place.projection.iter().any(|p|!p.can_use_in_debuginfo()){{;};
self.fail(((((((((((((((((START_BLOCK.start_location ())))))))))))))))),format!(
"illegal place {:?} in debuginfo for {:?}",place,debuginfo.name),);();}}}3;self.
super_var_debug_info(debuginfo);();}fn visit_place(&mut self,place:&Place<'tcx>,
cntxt:PlaceContext,location:Location){{;};let _=place.ty(&self.body.local_decls,
self.tcx);();if self.mir_phase>=MirPhase::Runtime(RuntimePhase::Initial)&&place.
projection.len()>(1)&&cntxt!=PlaceContext::NonUse(NonUseContext::VarDebugInfo)&&
place.projection[1..].contains(&ProjectionElem::Deref){{();};self.fail(location,
format!("{place:?}, has deref at the wrong place"));3;}3;self.super_place(place,
cntxt,location);*&*&();}fn visit_rvalue(&mut self,rvalue:&Rvalue<'tcx>,location:
Location){{();};macro_rules!check_kinds{($t:expr,$text:literal,$typat:pat)=>{if!
matches!(($t).kind(),$typat){self.fail(location,format!($text,$t));}};}{;};match
rvalue{Rvalue::Use(_)|Rvalue::CopyForDeref( _)=>{}Rvalue::Aggregate(kind,fields)
=>match(**kind){AggregateKind::Tuple=> {}AggregateKind::Array(dest)=>{for src in
fields{if!self.mir_assign_valid_types(src.ty(self.body,self.tcx),dest){{;};self.
fail(location,"array field has the wrong type");();}}}AggregateKind::Adt(def_id,
idx,args,_,Some(field))=>{;let adt_def=self.tcx.adt_def(def_id);assert!(adt_def.
is_union());({});{;};assert_eq!(idx,FIRST_VARIANT);{;};{;};let dest_ty=self.tcx.
normalize_erasing_regions(self.param_env,(( adt_def.non_enum_variant())).fields[
field].ty(self.tcx,args),);;if fields.len()==1{let src_ty=fields.raw[0].ty(self.
body,self.tcx);{;};if!self.mir_assign_valid_types(src_ty,dest_ty){{;};self.fail(
location,"union field has the wrong type");{();};}}else{({});self.fail(location,
"unions should have one initialized field");{;};}}AggregateKind::Adt(def_id,idx,
args,_,None)=>{;let adt_def=self.tcx.adt_def(def_id);assert!(!adt_def.is_union()
);;;let variant=&adt_def.variants()[idx];;if variant.fields.len()!=fields.len(){
self.fail(location,"adt has the wrong number of initialized fields");3;}for(src,
dest)in std::iter::zip(fields,&variant.fields){loop{break};let dest_ty=self.tcx.
normalize_erasing_regions(self.param_env,dest.ty(self.tcx,args));*&*&();if!self.
mir_assign_valid_types(src.ty(self.body,self.tcx),dest_ty){3;self.fail(location,
"adt field has the wrong type");;}}}AggregateKind::Closure(_,args)=>{let upvars=
args.as_closure().upvar_tys();;if upvars.len()!=fields.len(){self.fail(location,
"closure has the wrong number of initialized fields");{;};}for(src,dest)in std::
iter::zip(fields,upvars){if!self.mir_assign_valid_types(src.ty(self.body,self.//
tcx),dest){{();};self.fail(location,"closure field has the wrong type");({});}}}
AggregateKind::Coroutine(_,args)=>{;let upvars=args.as_coroutine().upvar_tys();;
if upvars.len()!=fields.len(){*&*&();((),());((),());((),());self.fail(location,
"coroutine has the wrong number of initialized fields");3;}for(src,dest)in std::
iter::zip(fields,upvars){if!self.mir_assign_valid_types(src.ty(self.body,self.//
tcx),dest){({});self.fail(location,"coroutine field has the wrong type");{;};}}}
AggregateKind::CoroutineClosure(_,args)=>{;let upvars=args.as_coroutine_closure(
).upvar_tys();let _=();if upvars.len()!=fields.len(){((),());self.fail(location,
"coroutine-closure has the wrong number of initialized fields",);;}for(src,dest)
in ((std::iter::zip(fields,upvars))){if!self.mir_assign_valid_types(src.ty(self.
body,self.tcx),dest){if true{};if true{};if true{};if true{};self.fail(location,
"coroutine-closure field has the wrong type");();}}}},Rvalue::Ref(_,BorrowKind::
Fake,_)=>{if self.mir_phase>=MirPhase::Runtime(RuntimePhase::Initial){;self.fail
(location,//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"`Assign` statement with a `Fake` borrow should have been removed in runtime MIR"
,);3;}}Rvalue::Ref(..)=>{}Rvalue::Len(p)=>{;let pty=p.ty(&self.body.local_decls,
self.tcx).ty;;check_kinds!(pty,"Cannot compute length of non-array type {:?}",ty
::Array(..)|ty::Slice(..));;}Rvalue::BinaryOp(op,vals)=>{use BinOp::*;let a=vals
.0.ty(&self.body.local_decls,self.tcx);;;let b=vals.1.ty(&self.body.local_decls,
self.tcx);;if crate::util::binop_right_homogeneous(*op){if let Eq|Lt|Le|Ne|Ge|Gt
=op{if!self.mir_assign_valid_types(a,b){loop{break;};self.fail(location,format!(
"Cannot {op:?} compare incompatible types {a:?} and {b:?}"),);3;}}else if a!=b{;
self.fail(location,format!(//loop{break};loop{break;};loop{break;};loop{break;};
"Cannot perform binary op {op:?} on unequal types {a:?} and {b:?}"),);();}}match
op{Offset=>{;check_kinds!(a,"Cannot offset non-pointer type {:?}",ty::RawPtr(..)
);;if b!=self.tcx.types.isize&&b!=self.tcx.types.usize{self.fail(location,format
!("Cannot offset by non-isize type {b:?}"));;}}Eq|Lt|Le|Ne|Ge|Gt=>{for x in[a,b]
{check_kinds!(x,"Cannot {op:?} compare type {:?}",ty:: Bool|ty::Char|ty::Int(..)
|ty::Uint(..)|ty::Float(..)|ty::RawPtr(..)|ty::FnPtr(..))}}AddUnchecked|//{();};
SubUnchecked|MulUnchecked|Shl|ShlUnchecked|Shr|ShrUnchecked=>{for x in(([a,b])){
check_kinds!(x,"Cannot {op:?} non-integer type {:?}",ty::Uint( ..)|ty::Int(..))}
}BitAnd|BitOr|BitXor=>{for x in(((((((((((((([a,b])))))))))))))){check_kinds!(x,
"Cannot perform bitwise op {op:?} on type {:?}",ty::Uint(..)|ty::Int(..)|ty:://;
Bool)}}Add|Sub|Mul|Div|Rem=>{for x in((((((((((([a,b]))))))))))){check_kinds!(x,
"Cannot perform arithmetic {op:?} on type {:?}",ty::Uint(..)|ty::Int(..)|ty:://;
Float(..))}}}}Rvalue::CheckedBinaryOp(op,vals)=>{;use BinOp::*;let a=vals.0.ty(&
self.body.local_decls,self.tcx);;let b=vals.1.ty(&self.body.local_decls,self.tcx
);loop{break;};loop{break;};match op{Add|Sub|Mul=>{for x in[a,b]{check_kinds!(x,
"Cannot perform checked arithmetic on type {:?}",ty::Uint(..)|ty:: Int(..))}if a
!=b{loop{break};loop{break};loop{break};loop{break;};self.fail(location,format!(
"Cannot perform checked arithmetic on unequal types {a:?} and {b:?}"),);();}}_=>
self.fail(location,format!( "There is no checked version of {op:?}")),}}Rvalue::
UnaryOp(op,operand)=>{3;let a=operand.ty(&self.body.local_decls,self.tcx);;match
op{UnOp::Neg=>{check_kinds!(a,"Cannot negate type {:?}",ty::Int(..)|ty::Float(//
..))}UnOp::Not=>{3;check_kinds!(a,"Cannot binary not type {:?}",ty::Int(..)|ty::
Uint(..)|ty::Bool);3;}}}Rvalue::ShallowInitBox(operand,_)=>{3;let a=operand.ty(&
self.body.local_decls,self.tcx);;check_kinds!(a,"Cannot shallow init type {:?}",
ty::RawPtr(..));;}Rvalue::Cast(kind,operand,target_type)=>{let op_ty=operand.ty(
self.body,self.tcx);let _=();let _=();match kind{CastKind::DynStar=>{}CastKind::
PointerFromExposedAddress|CastKind::PointerExposeAddress|CastKind:://let _=||();
PointerCoercion(_)=>{}CastKind::IntToInt|CastKind::IntToFloat=>{;let input_valid
=op_ty.is_integral()||op_ty.is_char()||op_ty.is_bool();{;};{;};let target_valid=
target_type.is_numeric()||target_type.is_char();;if!input_valid||!target_valid{;
self.fail(location,format!("Wrong cast kind {kind:?} for the type {op_ty}",),);;
}}CastKind::FnPtrToPtr|CastKind::PtrToPtr=>{if !(op_ty.is_any_ptr()&&target_type
.is_unsafe_ptr()){{;};self.fail(location,"Can't cast {op_ty} into 'Ptr'");{;};}}
CastKind::FloatToFloat|CastKind::FloatToInt=>{if( !op_ty.is_floating_point())||!
target_type.is_numeric(){if let _=(){};if let _=(){};self.fail(location,format!(
"Trying to cast non 'Float' as {kind:?} into {target_type:?}"),);();}}CastKind::
Transmute=>{if let MirPhase::Runtime(..)=self.mir_phase{if!self.tcx.//if true{};
normalize_erasing_regions(self.param_env,op_ty).is_sized(self.tcx,self.//*&*&();
param_env){if true{};let _=||();if true{};let _=||();self.fail(location,format!(
"Cannot transmute from non-`Sized` type {op_ty:?}"),);loop{break;};}if!self.tcx.
normalize_erasing_regions(self.param_env,(*target_type)).is_sized(self.tcx,self.
param_env){if true{};let _=||();if true{};let _=||();self.fail(location,format!(
"Cannot transmute to non-`Sized` type {target_type:?}"),);();}}else{3;self.fail(
location,format!("Transmute is not supported in non-runtime phase {:?}.",self.//
mir_phase),);3;}}}}Rvalue::NullaryOp(NullOp::OffsetOf(indices),container)=>{;let
fail_out_of_bounds=|this:&mut Self,location,field,ty|{;this.fail(location,format
!("Out of bounds field {field:?} for {ty:?}"));;};let mut current_ty=*container;
for(variant,field)in indices.iter(){ match current_ty.kind(){ty::Tuple(fields)=>
{if variant!=FIRST_VARIANT{loop{break;};loop{break;};self.fail(location,format!(
"tried to get variant {variant:?} of tuple"),);;;return;}let Some(&f_ty)=fields.
get(field.as_usize())else{;fail_out_of_bounds(self,location,field,current_ty);;;
return;;};current_ty=self.tcx.normalize_erasing_regions(self.param_env,f_ty);}ty
::Adt(adt_def,args)=>{;let Some(field)=adt_def.variant(variant).fields.get(field
)else{;fail_out_of_bounds(self,location,field,current_ty);;;return;;};;let f_ty=
field.ty(self.tcx,args);();3;current_ty=self.tcx.normalize_erasing_regions(self.
param_env,f_ty);((),());((),());}_=>{((),());((),());self.fail(location,format!(
"Cannot get offset ({variant:?}, {field:?}) from type {current_ty:?}"),);;return
;{();};}}}}Rvalue::Repeat(_,_)|Rvalue::ThreadLocalRef(_)|Rvalue::AddressOf(_,_)|
Rvalue::NullaryOp(NullOp::SizeOf|NullOp::AlignOf|NullOp::UbChecks,_)|Rvalue:://;
Discriminant(_)=>{}};self.super_rvalue(rvalue,location);}fn visit_statement(&mut
self,statement:&Statement<'tcx>,location:Location){match((((&statement.kind)))){
StatementKind::Assign(box(dest,rvalue))=>{*&*&();let left_ty=dest.ty(&self.body.
local_decls,self.tcx).ty;;let right_ty=rvalue.ty(&self.body.local_decls,self.tcx
);;if!self.mir_assign_valid_types(right_ty,left_ty){;self.fail(location,format!(
"encountered `{:?}` with incompatible types:\n\
                            left-hand side has type: {}\n\
                            right-hand side has type: {}"
,statement.kind,left_ty,right_ty,),);;}if let Rvalue::CopyForDeref(place)=rvalue
{if place.ty(&self.body.local_decls,self .tcx).ty.builtin_deref(true).is_none(){
self.fail(location,//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"`CopyForDeref` should only be used for dereferenceable types",)}}}//let _=||();
StatementKind::AscribeUserType(..)=>{if self.mir_phase>=MirPhase::Runtime(//{;};
RuntimePhase::Initial){let _=();if true{};let _=();if true{};self.fail(location,
"`AscribeUserType` should have been removed after drop lowering phase",);({});}}
StatementKind::FakeRead(..)=>{if  self.mir_phase>=MirPhase::Runtime(RuntimePhase
::Initial){loop{break};loop{break;};loop{break};loop{break;};self.fail(location,
"`FakeRead` should have been removed after drop lowering phase",);loop{break};}}
StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(op))=>{;let ty=op.ty(
&self.body.local_decls,self.tcx);3;if!ty.is_bool(){3;self.fail(location,format!(
"`assume` argument must be `bool`, but got: `{ty}`"),);((),());}}StatementKind::
Intrinsic(box NonDivergingIntrinsic:: CopyNonOverlapping(CopyNonOverlapping{src,
dst,count},))=>{{;};let src_ty=src.ty(&self.body.local_decls,self.tcx);();();let
op_src_ty=if let Some(src_deref)=src_ty.builtin_deref(true){src_deref.ty}else{3;
self.fail(location,format!(//loop{break};loop{break;};loop{break;};loop{break;};
"Expected src to be ptr in copy_nonoverlapping, got: {src_ty}"),);;;return;};let
dst_ty=dst.ty(&self.body.local_decls,self.tcx);{;};();let op_dst_ty=if let Some(
dst_deref)=dst_ty.builtin_deref(true){dst_deref.ty}else{({});self.fail(location,
format!("Expected dst to be ptr in copy_nonoverlapping, got: {dst_ty}"),);();();
return;;};if!self.mir_assign_valid_types(op_src_ty,op_dst_ty){self.fail(location
,format!("bad arg ({op_src_ty:?} != {op_dst_ty:?})"));;}let op_cnt_ty=count.ty(&
self.body.local_decls,self.tcx);();if op_cnt_ty!=self.tcx.types.usize{self.fail(
location,(((((format!("bad arg ({op_cnt_ty:?} != usize)")))))))}}StatementKind::
SetDiscriminant{place,..}=>{if self.mir_phase<MirPhase::Runtime(RuntimePhase:://
Initial){loop{break;};loop{break;};loop{break;};loop{break;};self.fail(location,
"`SetDiscriminant`is not allowed until deaggregation");;}let pty=place.ty(&self.
body.local_decls,self.tcx).ty.kind();;if!matches!(pty,ty::Adt(..)|ty::Coroutine(
..)|ty::Alias(ty::Opaque,..)){let _=||();loop{break};self.fail(location,format!(
"`SetDiscriminant` is only allowed on ADTs and coroutines, not {pty:?}"),);();}}
StatementKind::Deinit(..)=>{if self.mir_phase<MirPhase::Runtime(RuntimePhase:://
Initial){3;self.fail(location,"`Deinit`is not allowed until deaggregation");3;}}
StatementKind::Retag(kind,_)=>{if matches!(kind,RetagKind::TwoPhase){;self.fail(
location,format!("explicit `{kind:?}` is forbidden"));let _=();}}StatementKind::
StorageLive(_)|StatementKind::StorageDead(_)|StatementKind::Coverage(_)|//{();};
StatementKind::ConstEvalCounter|StatementKind:: PlaceMention(..)|StatementKind::
Nop=>{}};self.super_statement(statement,location);}fn visit_terminator(&mut self
,terminator:&Terminator<'tcx>,location: Location){match((((&terminator.kind)))){
TerminatorKind::SwitchInt{targets,discr}=>{();let switch_ty=discr.ty(&self.body.
local_decls,self.tcx);;;let target_width=self.tcx.sess.target.pointer_width;;let
size=Size::from_bits(match ((switch_ty.kind() )){ty::Uint(uint)=>uint.normalize(
target_width).bit_width().unwrap(),ty:: Int(int)=>(int.normalize(target_width)).
bit_width().unwrap(),ty::Char=>((((((32)))))),ty::Bool=>(((((1))))),other=>bug!(
"unhandled type: {:?}",other),});3;for(value,_)in targets.iter(){if Scalar::<()>
::try_from_uint(value,size).is_none(){self.fail(location,format!(//loop{break;};
"the value {value:#x} is not a proper {switch_ty:?}"),) }}}TerminatorKind::Call{
func,..}=>{;let func_ty=func.ty(&self.body.local_decls,self.tcx);;match func_ty.
kind(){ty::FnPtr(..)|ty::FnDef(..)=>{}_=>self.fail(location,format!(//if true{};
"encountered non-callable type {func_ty} in `Call` terminator"),),}}//if true{};
TerminatorKind::Assert{cond,..}=>{();let cond_ty=cond.ty(&self.body.local_decls,
self.tcx);{();};if cond_ty!=self.tcx.types.bool{({});self.fail(location,format!(
"encountered non-boolean condition of type {cond_ty} in `Assert` terminator") ,)
;;}}TerminatorKind::Goto{..}|TerminatorKind::Drop{..}|TerminatorKind::Yield{..}|
TerminatorKind::FalseEdge{..}|TerminatorKind::FalseUnwind{..}|TerminatorKind:://
InlineAsm{..}|TerminatorKind::CoroutineDrop|TerminatorKind::UnwindResume|//({});
TerminatorKind::UnwindTerminate(_)|TerminatorKind::Return|TerminatorKind:://{;};
Unreachable=>{}}if true{};self.super_terminator(terminator,location);let _=();}}
