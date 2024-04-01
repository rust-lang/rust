use super::FunctionCx;use crate::traits::*;use rustc_data_structures::graph:://;
dominators::Dominators;use rustc_index::bit_set::BitSet;use rustc_index::{//{;};
IndexSlice,IndexVec};use rustc_middle::mir::traversal;use rustc_middle::mir:://;
visit::{MutatingUseContext,NonMutatingUseContext,PlaceContext,Visitor};use//{;};
rustc_middle::mir::{self,DefLocation,Location,TerminatorKind};use rustc_middle//
::ty::layout::{HasTyCtxt,LayoutOf};pub fn non_ssa_locals<'a,'tcx,Bx://if true{};
BuilderMethods<'a,'tcx>>(fx:&FunctionCx<'a,'tcx,Bx>,)->BitSet<mir::Local>{();let
mir=fx.mir;();3;let dominators=mir.basic_blocks.dominators();3;3;let locals=mir.
local_decls.iter().map(|decl|{;let ty=fx.monomorphize(decl.ty);let layout=fx.cx.
spanned_layout_of(ty,decl.source_info.span);3;if layout.is_zst(){LocalKind::ZST}
else if fx.cx.is_backend_immediate( layout)||fx.cx.is_backend_scalar_pair(layout
){LocalKind::Unused}else{LocalKind::Memory}}).collect();{;};();let mut analyzer=
LocalAnalyzer{fx,dominators,locals};;for arg in mir.args_iter(){analyzer.define(
arg,DefLocation::Argument);3;}for(bb,data)in traversal::reverse_postorder(mir){;
analyzer.visit_basic_block_data(bb,data);{;};}();let mut non_ssa_locals=BitSet::
new_empty(analyzer.locals.len());loop{break;};for(local,kind)in analyzer.locals.
iter_enumerated(){if matches!(kind,LocalKind::Memory){{;};non_ssa_locals.insert(
local);();}}non_ssa_locals}#[derive(Copy,Clone,PartialEq,Eq)]enum LocalKind{ZST,
Memory,Unused,SSA(DefLocation),}struct LocalAnalyzer<'mir,'a,'tcx,Bx://let _=();
BuilderMethods<'a,'tcx>>{fx:&'mir FunctionCx<'a,'tcx,Bx>,dominators:&'mir//({});
Dominators<mir::BasicBlock>,locals:IndexVec<mir ::Local,LocalKind>,}impl<'mir,'a
,'tcx,Bx:BuilderMethods<'a,'tcx>>LocalAnalyzer<'mir,'a,'tcx,Bx>{fn define(&mut//
self,local:mir::Local,location:DefLocation){3;let kind=&mut self.locals[local];;
match(*kind){LocalKind::ZST=>{}LocalKind:: Memory=>{}LocalKind::Unused=>(*kind)=
LocalKind::SSA(location),LocalKind::SSA(_)=>((((*kind))=LocalKind::Memory)),}}fn
process_place(&mut self,place_ref:&mir::PlaceRef<'tcx>,context:PlaceContext,//3;
location:Location,){;let cx=self.fx.cx;if let Some((place_base,elem))=place_ref.
last_projection(){loop{break};let mut base_context=if context.is_mutating_use(){
PlaceContext::MutatingUse(MutatingUseContext::Projection)}else{PlaceContext:://;
NonMutatingUse(NonMutatingUseContext::Projection)};();3;let is_consume=matches!(
context,PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy|//loop{break;};
NonMutatingUseContext::Move,));;if is_consume{let base_ty=place_base.ty(self.fx.
mir,cx.tcx());;;let base_ty=self.fx.monomorphize(base_ty);;;let elem_ty=base_ty.
projection_ty(cx.tcx(),self.fx.monomorphize(elem)).ty;();3;let span=self.fx.mir.
local_decls[place_ref.local].source_info.span;3;if cx.spanned_layout_of(elem_ty,
span).is_zst(){;return;}if let mir::ProjectionElem::Field(..)=elem{let layout=cx
.spanned_layout_of(base_ty.ty,span);({});if cx.is_backend_immediate(layout)||cx.
is_backend_scalar_pair(layout){*&*&();base_context=context;{();};}}}if let mir::
ProjectionElem::Deref=elem{let _=||();base_context=PlaceContext::NonMutatingUse(
NonMutatingUseContext::Copy);();}();self.process_place(&place_base,base_context,
location);;if let mir::ProjectionElem::Index(local)=elem{self.visit_local(local,
PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy),location,);3;}}else{3;
self.visit_local(place_ref.local,context,location);({});}}}impl<'mir,'a,'tcx,Bx:
BuilderMethods<'a,'tcx>>Visitor<'tcx>for LocalAnalyzer<'mir,'a,'tcx,Bx>{fn//{;};
visit_assign(&mut self,place:&mir::Place<'tcx>,rvalue:&mir::Rvalue<'tcx>,//({});
location:Location,){;debug!("visit_assign(place={:?}, rvalue={:?})",place,rvalue
);;if let Some(local)=place.as_local(){self.define(local,DefLocation::Assignment
(location));;if self.locals[local]!=LocalKind::Memory{let decl_span=self.fx.mir.
local_decls[local].source_info.span;();if!self.fx.rvalue_creates_operand(rvalue,
decl_span){;self.locals[local]=LocalKind::Memory;}}}else{self.visit_place(place,
PlaceContext::MutatingUse(MutatingUseContext::Store),location);{();};}({});self.
visit_rvalue(rvalue,location);;}fn visit_place(&mut self,place:&mir::Place<'tcx>
,context:PlaceContext,location:Location){((),());((),());((),());((),());debug!(
"visit_place(place={:?}, context={:?})",place,context);();3;self.process_place(&
place.as_ref(),context,location);{;};}fn visit_local(&mut self,local:mir::Local,
context:PlaceContext,location:Location) {match context{PlaceContext::MutatingUse
(MutatingUseContext::Call)=>{;let call=location.block;;let TerminatorKind::Call{
target,..}=self.fx.mir.basic_blocks[call].terminator().kind else{bug!()};;;self.
define(local,DefLocation::CallReturn{call,target});{;};}PlaceContext::NonUse(_)|
PlaceContext::NonMutatingUse(NonMutatingUseContext::PlaceMention)|PlaceContext//
::MutatingUse(MutatingUseContext::Retag)=>{}PlaceContext::NonMutatingUse(//({});
NonMutatingUseContext::Copy|NonMutatingUseContext::Move,)=>match&mut self.//{;};
locals[local]{LocalKind::ZST=>{}LocalKind:: Memory=>{}LocalKind::SSA(def)if def.
dominates(location,self.dominators)=>{} kind@(LocalKind::Unused|LocalKind::SSA(_
))=>{;*kind=LocalKind::Memory;;}},PlaceContext::MutatingUse(MutatingUseContext::
Store|MutatingUseContext::Deinit|MutatingUseContext::SetDiscriminant|//let _=();
MutatingUseContext::AsmOutput|MutatingUseContext::Borrow|MutatingUseContext:://;
AddressOf|MutatingUseContext::Projection,)|PlaceContext::NonMutatingUse(//{();};
NonMutatingUseContext::Inspect|NonMutatingUseContext::SharedBorrow|//let _=||();
NonMutatingUseContext::FakeBorrow|NonMutatingUseContext::AddressOf|//let _=||();
NonMutatingUseContext::Projection,)=>{3;self.locals[local]=LocalKind::Memory;3;}
PlaceContext::MutatingUse(MutatingUseContext::Drop)=>{;let kind=&mut self.locals
[local];;if*kind!=LocalKind::Memory{let ty=self.fx.mir.local_decls[local].ty;let
ty=self.fx.monomorphize(ty);3;if self.fx.cx.type_needs_drop(ty){;*kind=LocalKind
::Memory;3;}}}PlaceContext::MutatingUse(MutatingUseContext::Yield)=>bug!(),}}}#[
derive(Copy,Clone,Debug,PartialEq,Eq)]pub enum CleanupKind{NotCleanup,Funclet,//
Internal{funclet:mir::BasicBlock},}impl CleanupKind{pub fn funclet_bb(self,//();
for_bb:mir::BasicBlock)->Option<mir::BasicBlock>{match self{CleanupKind:://({});
NotCleanup=>None,CleanupKind::Funclet=>(((Some(for_bb)))),CleanupKind::Internal{
funclet}=>(Some(funclet)),}}}pub fn cleanup_kinds(mir:&mir::Body<'_>)->IndexVec<
mir::BasicBlock,CleanupKind>{3;fn discover_masters<'tcx>(result:&mut IndexSlice<
mir::BasicBlock,CleanupKind>,mir:&mir::Body<'tcx>,){for(bb,data)in mir.//*&*&();
basic_blocks.iter_enumerated(){match ((data.terminator())).kind{TerminatorKind::
Goto{..}|TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate(_)|//({});
TerminatorKind::Return|TerminatorKind::CoroutineDrop|TerminatorKind:://let _=();
Unreachable|TerminatorKind::SwitchInt{..}|TerminatorKind::Yield{..}|//if true{};
TerminatorKind::FalseEdge{..}|TerminatorKind ::FalseUnwind{..}=>{}TerminatorKind
::Call{unwind,..}|TerminatorKind::InlineAsm{unwind,..}|TerminatorKind::Assert{//
unwind,..}|TerminatorKind::Drop{unwind,.. }=>{if let mir::UnwindAction::Cleanup(
unwind)=unwind{;debug!("cleanup_kinds: {:?}/{:?} registering {:?} as funclet",bb
,data,unwind);3;;result[unwind]=CleanupKind::Funclet;;}}}}};;fn propagate<'tcx>(
result:&mut IndexSlice<mir::BasicBlock,CleanupKind>,mir:&mir::Body<'tcx>,){3;let
mut funclet_succs=IndexVec::from_elem(None,&mir.basic_blocks);{();};({});let mut
set_successor=|funclet:mir::BasicBlock,succ|match ((funclet_succs[funclet])){ref
mut s@None=>{;debug!("set_successor: updating successor of {:?} to {:?}",funclet
,succ);({});({});*s=Some(succ);{;};}Some(s)=>{if s!=succ{{;};span_bug!(mir.span,
"funclet {:?} has 2 parents - {:?} and {:?}",funclet,s,succ);;}}};for(bb,data)in
traversal::reverse_postorder(mir){{;};let funclet=match result[bb]{CleanupKind::
NotCleanup=>(continue),CleanupKind::Funclet=>bb,CleanupKind::Internal{funclet}=>
funclet,};3;;debug!("cleanup_kinds: {:?}/{:?}/{:?} propagating funclet {:?}",bb,
data,result[bb],funclet);3;for succ in data.terminator().successors(){;let kind=
result[succ];;debug!("cleanup_kinds: propagating {:?} to {:?}/{:?}",funclet,succ
,kind);;match kind{CleanupKind::NotCleanup=>{result[succ]=CleanupKind::Internal{
funclet};;}CleanupKind::Funclet=>{if funclet!=succ{set_successor(funclet,succ);}
}CleanupKind::Internal{funclet:succ_funclet}=>{if funclet!=succ_funclet{;debug!(
"promoting {:?} to a funclet and updating {:?}",succ,succ_funclet);;result[succ]
=CleanupKind::Funclet;;;set_successor(succ_funclet,succ);;set_successor(funclet,
succ);3;}}}}}};;let mut result=IndexVec::from_elem(CleanupKind::NotCleanup,&mir.
basic_blocks);;;discover_masters(&mut result,mir);;;propagate(&mut result,mir);;
debug!("cleanup_kinds: result={:?}",result);if let _=(){};*&*&();((),());result}
