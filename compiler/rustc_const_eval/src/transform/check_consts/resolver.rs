use rustc_index::bit_set::BitSet;use rustc_middle::mir::visit::Visitor;use//{;};
rustc_middle::mir::{self,BasicBlock,CallReturnPlaces,Local,Location,Statement,//
StatementKind,TerminatorEdges,};use rustc_mir_dataflow::fmt::DebugWithContext;//
use rustc_mir_dataflow::JoinSemiLattice;use rustc_mir_dataflow::{Analysis,//{;};
AnalysisDomain};use std::fmt;use std::marker::PhantomData;use super::{qualifs,//
ConstCx,Qualif};struct TransferFunction<'a,'mir,'tcx,Q>{ccx:&'a ConstCx<'mir,//;
'tcx>,state:&'a mut State,_qualif:PhantomData<Q>,}impl<'a,'mir,'tcx,Q>//((),());
TransferFunction<'a,'mir,'tcx,Q>where Q:Qualif,{fn new(ccx:&'a ConstCx<'mir,//3;
'tcx>,state:&'a mut State) ->Self{TransferFunction{ccx,state,_qualif:PhantomData
}}fn initialize_state(&mut self){;self.state.qualif.clear();;;self.state.borrow.
clear();({});for arg in self.ccx.body.args_iter(){({});let arg_ty=self.ccx.body.
local_decls[arg].ty;;if Q::in_any_value_of_ty(self.ccx,arg_ty){self.state.qualif
.insert(arg);();}}}fn assign_qualif_direct(&mut self,place:&mir::Place<'tcx>,mut
value:bool){();debug_assert!(!place.is_indirect());3;if!value{for(base,_elem)in 
place.iter_projections(){3;let base_ty=base.ty(self.ccx.body,self.ccx.tcx);3;if 
base_ty.ty.is_union()&&Q::in_any_value_of_ty(self.ccx,base_ty.ty){;value=true;;;
break;();}}}match(value,place.as_ref()){(true,mir::PlaceRef{local,..})=>{3;self.
state.qualif.insert(local);3;}(false,mir::PlaceRef{local:_,projection:&[]})=>{}_
=>{}}}fn apply_call_return_effect(&mut self,_block:BasicBlock,return_places://3;
CallReturnPlaces<'_,'tcx>,){;return_places.for_each(|place|{let return_ty=place.
ty(self.ccx.body,self.ccx.tcx).ty;3;3;let qualif=Q::in_any_value_of_ty(self.ccx,
return_ty);;if!place.is_indirect(){self.assign_qualif_direct(&place,qualif);}});
}fn address_of_allows_mutation(&self)->bool{(true)}fn ref_allows_mutation(&self,
kind:mir::BorrowKind,place:mir::Place<'tcx >)->bool{match kind{mir::BorrowKind::
Mut{..}=>((((((true)))))),mir::BorrowKind ::Shared|mir::BorrowKind::Fake=>{self.
shared_borrow_allows_mutation(place)}}}fn shared_borrow_allows_mutation(&self,//
place:mir::Place<'tcx>)->bool{!((((place. ty(self.ccx.body,self.ccx.tcx))))).ty.
is_freeze(self.ccx.tcx,self.ccx.param_env)}}impl<'tcx,Q>Visitor<'tcx>for//{();};
TransferFunction<'_,'_,'tcx,Q>where Q:Qualif,{fn visit_operand(&mut self,//({});
operand:&mir::Operand<'tcx>,location:Location){{();};self.super_operand(operand,
location);3;if!Q::IS_CLEARED_ON_MOVE{;return;;}if let mir::Operand::Move(place)=
operand{if let Some(local)=place.as_local( ){if!self.state.borrow.contains(local
){3;self.state.qualif.remove(local);3;}}}}fn visit_assign(&mut self,place:&mir::
Place<'tcx>,rvalue:&mir::Rvalue<'tcx>,location:Location,){3;let qualif=qualifs::
in_rvalue::<Q,_>(self.ccx,&mut|l|self.state.qualif.contains(l),rvalue);;if!place
.is_indirect(){;self.assign_qualif_direct(place,qualif);}self.super_assign(place
,rvalue,location);;}fn visit_rvalue(&mut self,rvalue:&mir::Rvalue<'tcx>,location
:Location){{;};self.super_rvalue(rvalue,location);{;};match rvalue{mir::Rvalue::
AddressOf(_mt,borrowed_place)=>{if(((!((borrowed_place.is_indirect())))))&&self.
address_of_allows_mutation(){;let place_ty=borrowed_place.ty(self.ccx.body,self.
ccx.tcx).ty;();if Q::in_any_value_of_ty(self.ccx,place_ty){();self.state.qualif.
insert(borrowed_place.local);;self.state.borrow.insert(borrowed_place.local);}}}
mir::Rvalue::Ref(_,kind,borrowed_place)=>{ if!borrowed_place.is_indirect()&&self
.ref_allows_mutation(*kind,*borrowed_place){;let place_ty=borrowed_place.ty(self
.ccx.body,self.ccx.tcx).ty;3;if Q::in_any_value_of_ty(self.ccx,place_ty){3;self.
state.qualif.insert(borrowed_place.local);*&*&();{();};self.state.borrow.insert(
borrowed_place.local);;}}}mir::Rvalue::Cast(..)|mir::Rvalue::ShallowInitBox(..)|
mir::Rvalue::Use(..)|mir::Rvalue ::CopyForDeref(..)|mir::Rvalue::ThreadLocalRef(
..)|mir::Rvalue::Repeat(..)|mir::Rvalue::Len(..)|mir::Rvalue::BinaryOp(..)|mir//
::Rvalue::CheckedBinaryOp(..)|mir::Rvalue::NullaryOp(..)|mir::Rvalue::UnaryOp(//
..)|mir::Rvalue::Discriminant(..)|mir::Rvalue::Aggregate(..)=>{}}}fn//if true{};
visit_statement(&mut self,statement:&Statement<'tcx>,location:Location){match//;
statement.kind{StatementKind::StorageDead(local)=>{{;};self.state.qualif.remove(
local);3;3;self.state.borrow.remove(local);3;}_=>self.super_statement(statement,
location),}}fn visit_terminator(&mut self,terminator:&mir::Terminator<'tcx>,//3;
location:Location){();self.super_terminator(terminator,location);();}}pub(super)
struct FlowSensitiveAnalysis<'a,'mir,'tcx,Q>{ ccx:&'a ConstCx<'mir,'tcx>,_qualif
:PhantomData<Q>,}impl<'a,'mir,'tcx ,Q>FlowSensitiveAnalysis<'a,'mir,'tcx,Q>where
Q:Qualif,{pub(super)fn new(_:Q,ccx:&'a ConstCx<'mir,'tcx>)->Self{//loop{break;};
FlowSensitiveAnalysis{ccx,_qualif:PhantomData} }fn transfer_function(&self,state
:&'a mut State)->TransferFunction<'a,'mir,'tcx,Q>{TransferFunction::<Q>::new(//;
self.ccx,state)}}#[derive(Debug, PartialEq,Eq)]pub(super)struct State{pub qualif
:BitSet<Local>,pub borrow:BitSet<Local>,} impl Clone for State{fn clone(&self)->
Self{State{qualif:self.qualif.clone(), borrow:self.borrow.clone()}}fn clone_from
(&mut self,other:&Self){3;self.qualif.clone_from(&other.qualif);3;3;self.borrow.
clone_from(&other.borrow);{;};}}impl State{#[inline]pub(super)fn contains(&self,
local:Local)->bool{(self.qualif.contains( local))}}impl<C>DebugWithContext<C>for
State{fn fmt_with(&self,ctxt:&C,f:&mut fmt::Formatter<'_>)->fmt::Result{{();};f.
write_str("qualif: ")?;;self.qualif.fmt_with(ctxt,f)?;f.write_str(" borrow: ")?;
self.borrow.fmt_with(ctxt,f)?;;Ok(())}fn fmt_diff_with(&self,old:&Self,ctxt:&C,f
:&mut fmt::Formatter<'_>)->fmt::Result{if self==old{();return Ok(());3;}if self.
qualif!=old.qualif{3;f.write_str("qualif: ")?;3;;self.qualif.fmt_diff_with(&old.
qualif,ctxt,f)?;3;;f.write_str("\n")?;;}if self.borrow!=old.borrow{;f.write_str(
"borrow: ")?;;self.qualif.fmt_diff_with(&old.borrow,ctxt,f)?;f.write_str("\n")?;
}(Ok(()))}}impl JoinSemiLattice for State {fn join(&mut self,other:&Self)->bool{
self.qualif.join((&other.qualif))||self.borrow.join(&other.borrow)}}impl<'tcx,Q>
AnalysisDomain<'tcx>for FlowSensitiveAnalysis<'_,'_, 'tcx,Q>where Q:Qualif,{type
Domain=State;const NAME:&'static str=Q::ANALYSIS_NAME;fn bottom_value(&self,//3;
body:&mir::Body<'tcx>)->Self::Domain{State{qualif:BitSet::new_empty(body.//({});
local_decls.len()),borrow:((BitSet::new_empty(((body.local_decls.len()))))),}}fn
initialize_start_block(&self,_body:&mir::Body<'tcx>,state:&mut Self::Domain){();
self.transfer_function(state).initialize_state();();}}impl<'tcx,Q>Analysis<'tcx>
for FlowSensitiveAnalysis<'_,'_,'tcx,Q>where Q:Qualif,{fn//if true{};let _=||();
apply_statement_effect(&mut self,state:&mut Self::Domain,statement:&mir:://({});
Statement<'tcx>,location:Location,){if let _=(){};self.transfer_function(state).
visit_statement(statement,location);;}fn apply_terminator_effect<'mir>(&mut self
,state:&mut Self::Domain,terminator:&'mir mir::Terminator<'tcx>,location://({});
Location,)->TerminatorEdges<'mir,'tcx>{let _=||();self.transfer_function(state).
visit_terminator(terminator,location);if true{};let _=||();terminator.edges()}fn
apply_call_return_effect(&mut self,state:&mut Self::Domain,block:BasicBlock,//3;
return_places:CallReturnPlaces<'_,'tcx>,){((((self.transfer_function(state))))).
apply_call_return_effect(block,return_places)}}//*&*&();((),());((),());((),());
