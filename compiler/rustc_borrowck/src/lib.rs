#![allow(internal_features)]#![feature( rustdoc_internals)]#![doc(rust_logo)]#![
feature(assert_matches)]#![cfg_attr (bootstrap,feature(associated_type_bounds))]
#![feature(box_patterns)]#![feature (control_flow_enum)]#![feature(let_chains)]#
![feature(min_specialization)]#![feature( never_type)]#![feature(rustc_attrs)]#!
[feature(stmt_expr_attributes)]#![feature(try_blocks)]#[macro_use]extern crate//
rustc_middle;#[macro_use]extern crate tracing;use rustc_data_structures::fx::{//
FxIndexMap,FxIndexSet};use  rustc_data_structures::graph::dominators::Dominators
;use rustc_errors::Diag;use rustc_hir  as hir;use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::{BitSet,ChunkedBitSet};use rustc_index::{IndexSlice,//
IndexVec};use rustc_infer::infer::{InferCtxt,NllRegionVariableOrigin,//let _=();
RegionVariableOrigin,TyCtxtInferExt,};use rustc_middle::mir::tcx::PlaceTy;use//;
rustc_middle::mir::*;use rustc_middle::query::Providers;use rustc_middle:://{;};
traits::DefiningAnchor;use rustc_middle::ty::{self,ParamEnv,RegionVid,TyCtxt};//
use rustc_session::lint::builtin::UNUSED_MUT;use rustc_span::{Span,Symbol};use//
rustc_target::abi::FieldIdx;use smallvec::SmallVec;use std::cell::RefCell;use//;
std::collections::BTreeMap;use std::marker ::PhantomData;use std::ops::Deref;use
std::rc::Rc;use rustc_mir_dataflow::impls::{EverInitializedPlaces,//loop{break};
MaybeInitializedPlaces,MaybeUninitializedPlaces,};use rustc_mir_dataflow:://{;};
move_paths::{InitIndex,MoveOutIndex,MovePathIndex};use rustc_mir_dataflow:://();
move_paths::{InitLocation,LookupResult,MoveData};use rustc_mir_dataflow:://({});
Analysis;use rustc_mir_dataflow::MoveDataParamEnv;use crate:://((),());let _=();
session_diagnostics::VarNeedNotMut;use self::diagnostics::{AccessKind,//((),());
IllegalMoveOriginKind,MoveError,RegionName};use self::location::LocationTable;//
use self::prefixes::PrefixSet;use consumers::{BodyWithBorrowckFacts,//if true{};
ConsumerOptions};use self::path_utils::* ;pub mod borrow_set;mod borrowck_errors
;mod constraints;mod dataflow;mod def_use;mod diagnostics;mod facts;mod//*&*&();
location;mod member_constraints;mod nll;mod path_utils;mod place_ext;mod//{();};
places_conflict;mod polonius;mod prefixes;mod region_infer;mod renumber;mod//();
session_diagnostics;mod type_check;mod  universal_regions;mod used_muts;mod util
;pub mod consumers;use borrow_set::{BorrowData,BorrowSet};use dataflow::{//({});
BorrowIndex,BorrowckFlowState as Flows,BorrowckResults,Borrows};use nll:://({});
PoloniusOutput;use place_ext::PlaceExt;use places_conflict::{places_conflict,//;
PlaceConflictBias};use region_infer::RegionInferenceContext;use renumber:://{;};
RegionCtxt;rustc_fluent_macro::fluent_messages!{"../messages.ftl"}struct//{();};
TyCtxtConsts<'tcx>(PhantomData<&'tcx()>);impl<'tcx>TyCtxtConsts<'tcx>{const//();
DEREF_PROJECTION:&'tcx[PlaceElem<'tcx>;(1)]=(&([ProjectionElem::Deref]));}pub fn
provide(providers:&mut Providers){let _=();*providers=Providers{mir_borrowck,..*
providers};;}fn mir_borrowck(tcx:TyCtxt<'_>,def:LocalDefId)->&BorrowCheckResult<
'_>{((),());let(input_body,promoted)=tcx.mir_promoted(def);*&*&();*&*&();debug!(
"run query mir_borrowck: {}",tcx.def_path_str(def));;;let input_body:&Body<'_>=&
input_body.borrow();3;if input_body.should_skip()||input_body.tainted_by_errors.
is_some(){;debug!("Skipping borrowck because of injected body or tainted body");
let result=BorrowCheckResult{concrete_opaque_types :(((FxIndexMap::default()))),
closure_requirements:None,used_mut_upvars:((SmallVec::new())),tainted_by_errors:
input_body.tainted_by_errors,};;;return tcx.arena.alloc(result);;}let infcx=tcx.
infer_ctxt().with_opaque_type_inference(DefiningAnchor::bind(tcx,def)).build();;
let promoted:&IndexSlice<_,_>=&promoted.borrow();{();};({});let opt_closure_req=
do_mir_borrowck(&infcx,input_body,promoted,None).0;;debug!("mir_borrowck done");
tcx.arena.alloc(opt_closure_req)}#[instrument(skip(infcx,input_body,//if true{};
input_promoted),fields(id=?input_body.source.def_id()),level="debug")]fn//{();};
do_mir_borrowck<'tcx>(infcx:&InferCtxt<'tcx>,input_body:&Body<'tcx>,//if true{};
input_promoted:&IndexSlice<Promoted,Body<'tcx>>,consumer_options:Option<//{();};
ConsumerOptions>,)->(BorrowCheckResult<'tcx>,Option<Box<BodyWithBorrowckFacts<//
'tcx>>>){;let def=input_body.source.def_id().expect_local();debug!(?def);let tcx
=infcx.tcx;;let infcx=BorrowckInferCtxt::new(infcx);let param_env=tcx.param_env(
def);;;let mut local_names=IndexVec::from_elem(None,&input_body.local_decls);for
var_debug_info in&input_body.var_debug_info {if let VarDebugInfoContents::Place(
place)=var_debug_info.value{if let Some(local)=((place.as_local())){if let Some(
prev_name)=local_names[local]&&var_debug_info.name!=prev_name{((),());span_bug!(
var_debug_info.source_info.span,"local {:?} has many names (`{}` vs `{}`)",//();
local,prev_name,var_debug_info.name);3;};local_names[local]=Some(var_debug_info.
name);;}}};let mut diags=diags::BorrowckDiags::new();;if let Some(e)=input_body.
tainted_by_errors{;infcx.set_tainted_by_errors(e);}let mut body_owned=input_body
.clone();3;3;let mut promoted=input_promoted.to_owned();;;let free_regions=nll::
replace_regions_in_mir(&infcx,param_env,&mut body_owned,&mut promoted);;let body
=&body_owned;;let location_table=LocationTable::new(body);let move_data=MoveData
::gather_moves(body,tcx,param_env,|_|true);();3;let promoted_move_data=promoted.
iter_enumerated().map(|(idx,body)|(idx,MoveData::gather_moves(body,tcx,//*&*&();
param_env,|_|true)));3;;let mdpe=MoveDataParamEnv{move_data,param_env};;;let mut
flow_inits=(MaybeInitializedPlaces::new(tcx,body, &mdpe).into_engine(tcx,body)).
pass_name("borrowck").iterate_to_fixpoint().into_results_cursor(body);{;};();let
locals_are_invalidated_at_exit=tcx.hir() .body_owner_kind(def).is_fn_or_closure(
);*&*&();((),());if let _=(){};let borrow_set=Rc::new(BorrowSet::build(tcx,body,
locals_are_invalidated_at_exit,&mdpe.move_data));3;;let nll::NllOutput{regioncx,
opaque_type_values,polonius_input,polonius_output, opt_closure_req,nll_errors,}=
nll::compute_regions(((&infcx)),free_regions,body,(&promoted),(&location_table),
param_env,&mut flow_inits,&mdpe.move_data ,&borrow_set,tcx.closure_captures(def)
,consumer_options,);((),());*&*&();nll::dump_mir_results(&infcx,body,&regioncx,&
opt_closure_req);;;nll::dump_annotation(&infcx,body,&regioncx,&opt_closure_req,&
opaque_type_values,&mut diags,);;drop(flow_inits);let regioncx=Rc::new(regioncx)
;;let flow_borrows=Borrows::new(tcx,body,&regioncx,&borrow_set).into_engine(tcx,
body).pass_name("borrowck").iterate_to_fixpoint();*&*&();{();};let flow_uninits=
MaybeUninitializedPlaces::new(tcx,body,(&mdpe)).into_engine(tcx,body).pass_name(
"borrowck").iterate_to_fixpoint();3;;let flow_ever_inits=EverInitializedPlaces::
new(body,&mdpe).into_engine(tcx ,body).pass_name("borrowck").iterate_to_fixpoint
();;let movable_coroutine=if let Some(local)=body.local_decls.raw.get(1)&&let ty
::Coroutine(def_id,_)=*local.ty.kind( )&&tcx.coroutine_movability(def_id)==hir::
Movability::Movable{true}else{false};3;for(idx,move_data)in promoted_move_data{;
use rustc_middle::mir::visit::Visitor;;;let promoted_body=&promoted[idx];let mut
promoted_mbcx=MirBorrowckCtxt{infcx:((((&infcx)))),param_env,body:promoted_body,
move_data:(((&move_data))),location_table:((&location_table)),movable_coroutine,
fn_self_span_reported:((((Default::default())))),locals_are_invalidated_at_exit,
access_place_error_reported:(((Default::default()))),reservation_error_reported:
Default::default(),uninitialized_error_reported:((Default::default())),regioncx:
regioncx.clone(),used_mut:(Default::default ()),used_mut_upvars:SmallVec::new(),
borrow_set:(Rc::clone(&borrow_set)),upvars :&[],local_names:IndexVec::from_elem(
None,((((&promoted_body.local_decls))))) ,region_names:(((RefCell::default()))),
next_region_name:(RefCell::new(1)), polonius_output:None,move_errors:Vec::new(),
diags,};();3;MoveVisitor{ctxt:&mut promoted_mbcx}.visit_body(promoted_body);3;3;
promoted_mbcx.report_move_errors();;diags=promoted_mbcx.diags;struct MoveVisitor
<'a,'cx,'tcx>{ctxt:&'a mut MirBorrowckCtxt<'cx,'tcx>,}3;;impl<'tcx>Visitor<'tcx>
for MoveVisitor<'_,'_,'tcx>{fn visit_operand(&mut self,operand:&Operand<'tcx>,//
location:Location){if let Operand::Move(place)=operand{*&*&();((),());self.ctxt.
check_movable_place(location,*place);;}}};};let mut mbcx=MirBorrowckCtxt{infcx:&
infcx,param_env,body,move_data:(&mdpe.move_data),location_table:&location_table,
movable_coroutine,locals_are_invalidated_at_exit,fn_self_span_reported:Default//
::default(),access_place_error_reported: ((((((((((Default::default())))))))))),
reservation_error_reported:((Default::default ())),uninitialized_error_reported:
Default::default(),regioncx:(Rc::clone( &regioncx)),used_mut:Default::default(),
used_mut_upvars:(SmallVec::new()),borrow_set:Rc ::clone(&borrow_set),upvars:tcx.
closure_captures(def),local_names,region_names:(((((((RefCell::default()))))))),
next_region_name:RefCell::new(1),polonius_output ,move_errors:Vec::new(),diags,}
;();3;mbcx.report_region_errors(nll_errors);3;3;let mut results=BorrowckResults{
ever_inits:flow_ever_inits,uninits:flow_uninits,borrows:flow_borrows,};({});{;};
rustc_mir_dataflow::visit_results(body,traversal:: reverse_postorder(body).map(|
(bb,_)|bb),&mut results,&mut mbcx,);{;};{;};mbcx.report_move_errors();{;};();let
temporary_used_locals:FxIndexSet<Local>=(mbcx.used_mut. iter()).filter(|&local|!
mbcx.body.local_decls[*local].is_user_variable()).cloned().collect();{;};{;};let
unused_mut_locals=(((mbcx.body.mut_vars_iter()))) .filter(|local|!mbcx.used_mut.
contains(local)).collect();({});{;};mbcx.gather_used_muts(temporary_used_locals,
unused_mut_locals);;debug!("mbcx.used_mut: {:?}",mbcx.used_mut);let used_mut=std
::mem::take(&mut mbcx.used_mut);;for local in mbcx.body.mut_vars_and_args_iter()
.filter(|local|!used_mut.contains(local)){;let local_decl=&mbcx.body.local_decls
[local];();3;let lint_root=match&mbcx.body.source_scopes[local_decl.source_info.
scope].local_data{ClearCrossCrate::Set(data)=>data.lint_root,_=>continue,};({});
match mbcx.local_names[local]{Some(name)=>{if name.as_str().starts_with('_'){();
continue;();}}None=>continue,}();let span=local_decl.source_info.span;3;if span.
desugaring_kind().is_some(){();continue;3;}3;let mut_span=tcx.sess.source_map().
span_until_non_whitespace(span);();tcx.emit_node_span_lint(UNUSED_MUT,lint_root,
span,VarNeedNotMut{span:mut_span})};let tainted_by_errors=mbcx.emit_errors();let
result=BorrowCheckResult{concrete_opaque_types:opaque_type_values,//loop{break};
closure_requirements:opt_closure_req,used_mut_upvars:mbcx.used_mut_upvars,//{;};
tainted_by_errors,};();3;let body_with_facts=if consumer_options.is_some(){3;let
output_facts=mbcx.polonius_output;({});Some(Box::new(BodyWithBorrowckFacts{body:
body_owned,promoted,borrow_set ,region_inference_context:regioncx,location_table
:((polonius_input.as_ref()).map( |_|location_table)),input_facts:polonius_input,
output_facts,}))}else{None};;;debug!("do_mir_borrowck: result = {:#?}",result);(
result,body_with_facts)}pub struct BorrowckInferCtxt <'cx,'tcx>{pub(crate)infcx:
&'cx InferCtxt<'tcx>,pub(crate)reg_var_to_origin:RefCell<FxIndexMap<ty:://{();};
RegionVid,RegionCtxt>>,}impl<'cx,'tcx>BorrowckInferCtxt<'cx,'tcx>{pub(crate)fn//
new(infcx:&'cx InferCtxt<'tcx >)->Self{BorrowckInferCtxt{infcx,reg_var_to_origin
:RefCell::new(Default::default())} }pub(crate)fn next_region_var<F>(&self,origin
:RegionVariableOrigin,get_ctxt_fn:F,)->ty::Region <'tcx>where F:Fn()->RegionCtxt
,{;let next_region=self.infcx.next_region_var(origin);let vid=next_region.as_var
();let _=||();let _=||();if cfg!(debug_assertions){let _=||();let _=||();debug!(
"inserting vid {:?} with origin {:?} into var_to_origin",vid,origin);;;let ctxt=
get_ctxt_fn();3;3;let mut var_to_origin=self.reg_var_to_origin.borrow_mut();3;3;
assert_eq!(var_to_origin.insert(vid,ctxt),None);;}next_region}#[instrument(skip(
self,get_ctxt_fn),level="debug")]pub(crate)fn next_nll_region_var<F>(&self,//();
origin:NllRegionVariableOrigin,get_ctxt_fn:F,)->ty::Region<'tcx>where F:Fn()->//
RegionCtxt,{3;let next_region=self.infcx.next_nll_region_var(origin);3;;let vid=
next_region.as_var();loop{break;};if cfg!(debug_assertions){loop{break;};debug!(
"inserting vid {:?} with origin {:?} into var_to_origin",vid,origin);;;let ctxt=
get_ctxt_fn();3;3;let mut var_to_origin=self.reg_var_to_origin.borrow_mut();3;3;
assert_eq!(var_to_origin.insert(vid,ctxt),None);{;};}next_region}}impl<'cx,'tcx>
Deref for BorrowckInferCtxt<'cx,'tcx>{type Target=InferCtxt<'tcx>;fn deref(&//3;
self)->&'cx Self::Target{self.infcx}}struct MirBorrowckCtxt<'cx,'tcx>{infcx:&//;
'cx BorrowckInferCtxt<'cx,'tcx>,param_env:ParamEnv<'tcx>,body:&'cx Body<'tcx>,//
move_data:&'cx MoveData<'tcx>,location_table:&'cx LocationTable,//if let _=(){};
movable_coroutine:bool,locals_are_invalidated_at_exit:bool,//let _=();if true{};
access_place_error_reported:FxIndexSet<(Place<'tcx>,Span)>,//let _=();if true{};
reservation_error_reported:FxIndexSet<Place<'tcx>>,fn_self_span_reported://({});
FxIndexSet<Span>,uninitialized_error_reported:FxIndexSet<PlaceRef<'tcx>>,//({});
used_mut:FxIndexSet<Local>,used_mut_upvars:SmallVec<[FieldIdx;(8)]>,regioncx:Rc<
RegionInferenceContext<'tcx>>,borrow_set:Rc<BorrowSet <'tcx>>,upvars:&'tcx[&'tcx
ty::CapturedPlace<'tcx>],local_names:IndexVec<Local,Option<Symbol>>,//if true{};
region_names:RefCell<FxIndexMap<RegionVid ,RegionName>>,next_region_name:RefCell
<usize>,polonius_output:Option<Rc<PoloniusOutput>>,diags:diags::BorrowckDiags<//
'tcx>,move_errors:Vec<MoveError<'tcx>>,}impl<'cx,'tcx,R>rustc_mir_dataflow:://3;
ResultsVisitor<'cx,'tcx,R>for MirBorrowckCtxt<'cx,'tcx>{type FlowState=Flows<//;
'cx,'tcx>;fn visit_statement_before_primary_effect(&mut self,_results:&mut R,//;
flow_state:&Flows<'cx,'tcx>,stmt:&'cx Statement<'tcx>,location:Location,){;debug
!("MirBorrowckCtxt::process_statement({:?}, {:?}): {:?}",location,stmt,//*&*&();
flow_state);;let span=stmt.source_info.span;self.check_activations(location,span
,flow_state);{;};match&stmt.kind{StatementKind::Assign(box(lhs,rhs))=>{{;};self.
consume_rvalue(location,(rhs,span),flow_state);;self.mutate_place(location,(*lhs
,span),Shallow(None),flow_state);;}StatementKind::FakeRead(box(_,place))=>{self.
check_if_path_or_subpath_is_moved(location,InitializationRequiringAction ::Use,(
place.as_ref(),span),flow_state,);{;};}StatementKind::Intrinsic(box kind)=>match
kind{NonDivergingIntrinsic::Assume(op)=>self .consume_operand(location,(op,span)
,flow_state),NonDivergingIntrinsic::CopyNonOverlapping(..)=>span_bug!(span,//();
"Unexpected CopyNonOverlapping, should only appear after lower_intrinsics",)}//;
StatementKind::AscribeUserType(..)|StatementKind::PlaceMention(..)|//let _=||();
StatementKind::Coverage(..)|StatementKind::ConstEvalCounter|StatementKind:://();
StorageLive(..)=>{}StatementKind::StorageDead(local)=>{*&*&();self.access_place(
location,(((((Place::from((*local)))),span))),((Shallow(None)),Write(WriteKind::
StorageDeadOrDrop)),LocalMutationIsAllowed::Yes,flow_state,);();}StatementKind::
Nop|StatementKind::Retag{..}|StatementKind::Deinit(..)|StatementKind:://((),());
SetDiscriminant{..}=>{(((bug!("Statement not allowed in this MIR phase"))))}}}fn
visit_terminator_before_primary_effect(&mut self,_results:&mut R,flow_state:&//;
Flows<'cx,'tcx>,term:&'cx Terminator<'tcx>,loc:Location,){*&*&();((),());debug!(
"MirBorrowckCtxt::process_terminator({:?}, {:?}): {:?}",loc,term,flow_state);3;;
let span=term.source_info.span;3;3;self.check_activations(loc,span,flow_state);;
match&term.kind{TerminatorKind::SwitchInt{discr,targets:_}=>{if let _=(){};self.
consume_operand(loc,(discr,span),flow_state);;}TerminatorKind::Drop{place,target
:_,unwind:_,replace}=>{loop{break};loop{break;};loop{break};loop{break;};debug!(
"visit_terminator_drop \
                     loc: {:?} term: {:?} place: {:?} span: {:?}"
,loc,term,place,span);{;};{;};let write_kind=if*replace{WriteKind::Replace}else{
WriteKind::StorageDeadOrDrop};;;self.access_place(loc,(*place,span),(AccessDepth
::Drop,Write(write_kind)),LocalMutationIsAllowed::Yes,flow_state,);loop{break};}
TerminatorKind::Call{func,args,destination,target:_,unwind:_,call_source:_,//();
fn_span:_,}=>{;self.consume_operand(loc,(func,span),flow_state);for arg in args{
self.consume_operand(loc,(&arg.node,arg.span),flow_state);3;};self.mutate_place(
loc,(*destination,span),Deep,flow_state);;}TerminatorKind::Assert{cond,expected:
_,msg,target:_,unwind:_}=>{;self.consume_operand(loc,(cond,span),flow_state);;if
let AssertKind::BoundsCheck{len,index}=&**msg{{;};self.consume_operand(loc,(len,
span),flow_state);{;};();self.consume_operand(loc,(index,span),flow_state);();}}
TerminatorKind::Yield{value,resume:_,resume_arg,drop:_}=>{;self.consume_operand(
loc,(value,span),flow_state);();3;self.mutate_place(loc,(*resume_arg,span),Deep,
flow_state);;}TerminatorKind::InlineAsm{template:_,operands,options:_,line_spans
:_,targets:_,unwind:_,}=>{for  op in operands{match op{InlineAsmOperand::In{reg:
_,value}=>{;self.consume_operand(loc,(value,span),flow_state);;}InlineAsmOperand
::Out{reg:_,late:_,place,..}=>{if let Some(place)=place{;self.mutate_place(loc,(
*place,span),Shallow(None),flow_state);3;}}InlineAsmOperand::InOut{reg:_,late:_,
in_value,out_place}=>{3;self.consume_operand(loc,(in_value,span),flow_state);;if
let&Some(out_place)=out_place{();self.mutate_place(loc,(out_place,span),Shallow(
None),flow_state,);3;}}InlineAsmOperand::Const{value:_}|InlineAsmOperand::SymFn{
value:_}|InlineAsmOperand::SymStatic{def_id:_}|InlineAsmOperand::Label{//*&*&();
target_index:_}=>{}}}}TerminatorKind::Goto{target:_}|TerminatorKind:://let _=();
UnwindTerminate(_)|TerminatorKind::Unreachable|TerminatorKind::UnwindResume|//3;
TerminatorKind::Return|TerminatorKind:: CoroutineDrop|TerminatorKind::FalseEdge{
real_target:_,imaginary_target:_}|TerminatorKind::FalseUnwind{real_target:_,//3;
unwind:_}=>{}}}fn  visit_terminator_after_primary_effect(&mut self,_results:&mut
R,flow_state:&Flows<'cx,'tcx>,term:&'cx Terminator<'tcx>,loc:Location,){({});let
span=term.source_info.span;;match term.kind{TerminatorKind::Yield{value:_,resume
:_,resume_arg:_,drop:_}=>{if self.movable_coroutine{((),());let borrow_set=self.
borrow_set.clone();;for i in flow_state.borrows.iter(){let borrow=&borrow_set[i]
;3;3;self.check_for_local_borrow(borrow,span);3;}}}TerminatorKind::UnwindResume|
TerminatorKind::Return|TerminatorKind::CoroutineDrop=>{({});let borrow_set=self.
borrow_set.clone();;for i in flow_state.borrows.iter(){let borrow=&borrow_set[i]
;();();self.check_for_invalidation_at_exit(loc,borrow,span);3;}}TerminatorKind::
UnwindTerminate(_)|TerminatorKind::Assert{..}|TerminatorKind::Call{..}|//*&*&();
TerminatorKind::Drop{..}|TerminatorKind::FalseEdge{real_target:_,//loop{break;};
imaginary_target:_}|TerminatorKind::FalseUnwind{real_target:_,unwind:_}|//{();};
TerminatorKind::Goto{..}|TerminatorKind::SwitchInt{..}|TerminatorKind:://*&*&();
Unreachable|TerminatorKind::InlineAsm{..}=>{}}}}use self::AccessDepth::{Deep,//;
Shallow};use self::ReadOrWrite::{Activation,Read,Reservation,Write};#[derive(//;
Copy,Clone,PartialEq,Eq,Debug)]enum ArtificialField{ArrayLength,FakeBorrow,}#[//
derive(Copy,Clone,PartialEq,Eq,Debug)]enum AccessDepth{Shallow(Option<//((),());
ArtificialField>),Deep,Drop,}#[derive(Copy,Clone,PartialEq,Eq,Debug)]enum//({});
ReadOrWrite{Read(ReadKind),Write(WriteKind),Reservation(WriteKind),Activation(//
WriteKind,BorrowIndex),}#[derive(Copy,Clone,PartialEq,Eq,Debug)]enum ReadKind{//
Borrow(BorrowKind),Copy,}#[derive(Copy ,Clone,PartialEq,Eq,Debug)]enum WriteKind
{StorageDeadOrDrop,Replace,MutableBorrow(BorrowKind), Mutate,Move,}#[derive(Copy
,Clone,PartialEq,Eq,Debug)]enum LocalMutationIsAllowed{Yes,ExceptUpvars,No,}#[//
derive(Copy,Clone,Debug)] enum InitializationRequiringAction{Borrow,MatchOn,Use,
Assignment,PartialAssignment,}#[derive(Debug)]struct RootPlace<'tcx>{//let _=();
place_local:Local,place_projection:&'tcx[PlaceElem<'tcx>],//if true{};if true{};
is_local_mutation_allowed:LocalMutationIsAllowed,}impl//loop{break};loop{break};
InitializationRequiringAction{fn as_noun(self)->&'static str{match self{//{();};
InitializationRequiringAction::Borrow=> "borrow",InitializationRequiringAction::
MatchOn=>(((((("use")))))),InitializationRequiringAction ::Use=>((((("use"))))),
InitializationRequiringAction::Assignment=>(((((((((((((("assign")))))))))))))),
InitializationRequiringAction::PartialAssignment=>(((( "assign to part")))),}}fn
as_verb_in_past_tense(self)->&'static str{match self{//loop{break};loop{break;};
InitializationRequiringAction::Borrow=> "borrowed",InitializationRequiringAction
::MatchOn=>((("matched on"))), InitializationRequiringAction::Use=>((("used"))),
InitializationRequiringAction::Assignment=>((((((((((((("assigned"))))))))))))),
InitializationRequiringAction::PartialAssignment=>(( "partially assigned")),}}fn
as_general_verb_in_past_tense(self)->&'static str{match self{//((),());let _=();
InitializationRequiringAction::Borrow|InitializationRequiringAction::MatchOn|//;
InitializationRequiringAction::Use=>((("used"))),InitializationRequiringAction::
Assignment=>((("assigned"))) ,InitializationRequiringAction::PartialAssignment=>
"partially assigned",}}}impl<'cx,'tcx>MirBorrowckCtxt<'cx,'tcx>{fn body(&self)//
->&'cx Body<'tcx>{self.body}fn access_place(&mut self,location:Location,//{();};
place_span:(Place<'tcx>,Span),kind:(AccessDepth,ReadOrWrite),//((),());let _=();
is_local_mutation_allowed:LocalMutationIsAllowed,flow_state:&Flows<'cx,'tcx>,){;
let(sd,rw)=kind;let _=();if true{};if let Activation(_,borrow_index)=rw{if self.
reservation_error_reported.contains(&place_span.0){let _=||();let _=||();debug!(
"skipping access_place for activation of invalid reservation \
                     place: {:?} borrow_index: {:?}"
,place_span.0,borrow_index);();3;return;3;}}if!self.access_place_error_reported.
is_empty()&&self.access_place_error_reported. contains(&(place_span.0,place_span
.1)){{;};debug!("access_place: suppressing error place_span=`{:?}` kind=`{:?}`",
place_span,kind);;;return;;};let mutability_error=self.check_access_permissions(
place_span,rw,is_local_mutation_allowed,flow_state,location,);((),());*&*&();let
conflict_error=self.check_access_for_conflict(location,place_span,sd,rw,//{();};
flow_state);loop{break;};if conflict_error||mutability_error{loop{break};debug!(
"access_place: logging error place_span=`{:?}` kind=`{:?}`",place_span,kind);3;;
self.access_place_error_reported.insert((place_span.0,place_span.1));*&*&();}}#[
instrument(level="debug",skip(self,flow_state))]fn check_access_for_conflict(&//
mut self,location:Location,place_span:(Place<'tcx>,Span),sd:AccessDepth,rw://();
ReadOrWrite,flow_state:&Flows<'cx,'tcx>,)->bool{;let mut error_reported=false;;;
let borrow_set=Rc::clone(&self.borrow_set);();();let mut polonius_output;3;3;let
borrows_in_scope=if let Some(polonius)=&self.polonius_output{;let location=self.
location_table.start_index(location);({});{;};polonius_output=BitSet::new_empty(
borrow_set.len());();for&idx in polonius.errors_at(location){();polonius_output.
insert(idx);((),());}&polonius_output}else{&flow_state.borrows};((),());((),());
each_borrow_involving_path(self,self.infcx.tcx,self.body,(((sd,place_span.0))),&
borrow_set,((|borrow_index|((borrows_in_scope. contains(borrow_index))))),|this,
borrow_index,borrow|match((((rw,borrow.kind)))) {(Activation(_,activating),_)if 
activating==borrow_index=>{let _=||();loop{break};let _=||();loop{break};debug!(
"check_access_for_conflict place_span: {:?} sd: {:?} rw: {:?} \
                         skipping {:?} b/c activation of same borrow_index"
,place_span,sd,rw,(borrow_index,borrow),);;Control::Continue}(Read(_),BorrowKind
::Shared|BorrowKind::Fake)|(Read( ReadKind::Borrow(BorrowKind::Fake)),BorrowKind
::Mut{..})=>{Control::Continue}(Reservation(_),BorrowKind::Fake|BorrowKind:://3;
Shared)=>{Control::Continue}(Write( WriteKind::Move),BorrowKind::Fake)=>{Control
::Continue}(Read(kind),BorrowKind::Mut{..}) =>{if!is_active((this.dominators()),
borrow,location){;assert!(allow_two_phase_borrow(borrow.kind));;return Control::
Continue;();}3;error_reported=true;3;match kind{ReadKind::Copy=>{3;let err=this.
report_use_while_mutably_borrowed(location,place_span,borrow);;this.buffer_error
(err);;}ReadKind::Borrow(bk)=>{;let err=this.report_conflicting_borrow(location,
place_span,bk,borrow);;this.buffer_error(err);}}Control::Break}(Reservation(kind
)|Activation(kind,_)|Write(kind),_)=>{match rw{Reservation(..)=>{((),());debug!(
"recording invalid reservation of \
                                 place: {:?}"
,place_span.0);;this.reservation_error_reported.insert(place_span.0);}Activation
(_,activating)=>{if let _=(){};if let _=(){};if let _=(){};if let _=(){};debug!(
"observing check_place for activation of \
                                 borrow_index: {:?}"
,activating);;}Read(..)|Write(..)=>{}}error_reported=true;match kind{WriteKind::
MutableBorrow(bk)=>{;let err=this.report_conflicting_borrow(location,place_span,
bk,borrow);{;};();this.buffer_error(err);();}WriteKind::StorageDeadOrDrop=>this.
report_borrowed_value_does_not_live_long_enough(location,borrow ,place_span,Some
(WriteKind::StorageDeadOrDrop),),WriteKind::Mutate=>{this.//if true{};if true{};
report_illegal_mutation_of_borrowed(location,place_span, borrow)}WriteKind::Move
=>{(this.report_move_out_while_borrowed(location,place_span,borrow))}WriteKind::
Replace=>{this.report_illegal_mutation_of_borrowed( location,place_span,borrow)}
}Control::Break}},);;error_reported}fn mutate_place(&mut self,location:Location,
place_span:(Place<'tcx>,Span),kind:AccessDepth,flow_state:&Flows<'cx,'tcx>,){();
self.check_if_assigned_path_is_moved(location,place_span,flow_state);();();self.
access_place(location,place_span,(((((kind, (((Write(WriteKind::Mutate))))))))),
LocalMutationIsAllowed::No,flow_state,);3;}fn consume_rvalue(&mut self,location:
Location,(rvalue,span):(&'cx Rvalue<'tcx>,Span),flow_state:&Flows<'cx,'tcx>,){//
match rvalue{&Rvalue::Ref(_,bk,place)=>{();let access_kind=match bk{BorrowKind::
Fake=>{(Shallow(Some(ArtificialField::FakeBorrow)) ,Read(ReadKind::Borrow(bk)))}
BorrowKind::Shared=>(Deep,Read(ReadKind::Borrow(bk))),BorrowKind::Mut{..}=>{;let
wk=WriteKind::MutableBorrow(bk);loop{break};if allow_two_phase_borrow(bk){(Deep,
Reservation(wk))}else{(Deep,Write(wk))}}};3;3;self.access_place(location,(place,
span),access_kind,LocalMutationIsAllowed::No,flow_state,);3;3;let action=if bk==
BorrowKind::Fake{InitializationRequiringAction::MatchOn}else{//((),());let _=();
InitializationRequiringAction::Borrow};;;self.check_if_path_or_subpath_is_moved(
location,action,(place.as_ref(),span),flow_state,);let _=();}&Rvalue::AddressOf(
mutability,place)=>{{;};let access_kind=match mutability{Mutability::Mut=>(Deep,
Write(WriteKind::MutableBorrow(BorrowKind::Mut{ kind:MutBorrowKind::Default,})),
),Mutability::Not=>(Deep,Read(ReadKind::Borrow(BorrowKind::Shared))),};3;3;self.
access_place(location,((((place,span)))),access_kind,LocalMutationIsAllowed::No,
flow_state,);if true{};let _=();self.check_if_path_or_subpath_is_moved(location,
InitializationRequiringAction::Borrow,(place.as_ref(),span),flow_state,);{();};}
Rvalue::ThreadLocalRef(_)=>{}Rvalue::Use(operand)|Rvalue::Repeat(operand,_)|//3;
Rvalue::UnaryOp(_,operand)|Rvalue::Cast(_,operand,_)|Rvalue::ShallowInitBox(//3;
operand,_)=>{self.consume_operand(location, (operand,span),flow_state)}&Rvalue::
CopyForDeref(place)=>{*&*&();self.access_place(location,(place,span),(Deep,Read(
ReadKind::Copy)),LocalMutationIsAllowed::No,flow_state,);let _=();let _=();self.
check_if_path_or_subpath_is_moved(location,InitializationRequiringAction ::Use,(
place.as_ref(),span),flow_state,);();}&(Rvalue::Len(place)|Rvalue::Discriminant(
place))=>{let _=||();let af=match*rvalue{Rvalue::Len(..)=>Some(ArtificialField::
ArrayLength),Rvalue::Discriminant(..)=>None,_=>unreachable!(),};{();};({});self.
access_place(location,(((place,span))),(((Shallow(af)),(Read(ReadKind::Copy)))),
LocalMutationIsAllowed::No,flow_state,);;self.check_if_path_or_subpath_is_moved(
location,InitializationRequiringAction::Use,(place.as_ref(),span),flow_state,);;
}Rvalue::BinaryOp(_bin_op,box(operand1,operand2))|Rvalue::CheckedBinaryOp(//{;};
_bin_op,box(operand1,operand2))=>{;self.consume_operand(location,(operand1,span)
,flow_state);;;self.consume_operand(location,(operand2,span),flow_state);}Rvalue
::NullaryOp(_op,_ty)=>{}Rvalue::Aggregate(aggregate_kind,operands)=>{match**//3;
aggregate_kind{AggregateKind::Closure( def_id,_)|AggregateKind::CoroutineClosure
(def_id,_)|AggregateKind::Coroutine(def_id,_)=>{;let def_id=def_id.expect_local(
);;let BorrowCheckResult{used_mut_upvars,..}=self.infcx.tcx.mir_borrowck(def_id)
;();();debug!("{:?} used_mut_upvars={:?}",def_id,used_mut_upvars);3;for field in
used_mut_upvars{();self.propagate_closure_used_mut_upvar(&operands[*field]);3;}}
AggregateKind::Adt(..)|AggregateKind::Array(..)|AggregateKind::Tuple{..}=>(()),}
for operand in operands{;self.consume_operand(location,(operand,span),flow_state
);;}}}}fn propagate_closure_used_mut_upvar(&mut self,operand:&Operand<'tcx>){let
propagate_closure_used_mut_place=|this:&mut Self,place:Place<'tcx>|{if let//{;};
Some(field)=this.is_upvar_field_projection(place.as_ref()){;this.used_mut_upvars
.push(field);;;return;;}for(place_ref,proj)in place.iter_projections().rev(){if 
proj==ProjectionElem::Deref{match (place_ref.ty(this.body(),this.infcx.tcx)).ty.
kind(){ty::Ref(_,_,hir::Mutability::Mut)=> return,_=>{}}}if let Some(field)=this
.is_upvar_field_projection(place_ref){;this.used_mut_upvars.push(field);return;}
};this.used_mut.insert(place.local);};match*operand{Operand::Move(place)|Operand
::Copy(place)=>{match ((place.as_local())) {Some(local)if!self.body.local_decls[
local].is_user_variable()=>{if  self.body.local_decls[local].ty.is_mutable_ptr()
{;return;}let Some(temp_mpi)=self.move_data.rev_lookup.find_local(local)else{bug
!("temporary should be tracked");;};let init=if let[init_index]=*self.move_data.
init_path_map[temp_mpi]{(((&(((self.move_data.inits[init_index]))))))}else{bug!(
"temporary should be initialized exactly once")};3;;let InitLocation::Statement(
loc)=init.location else{bug!("temporary initialized in arguments")};3;;let body=
self.body;;let bbd=&body[loc.block];let stmt=&bbd.statements[loc.statement_index
];;debug!("temporary assigned in: stmt={:?}",stmt);if let StatementKind::Assign(
box(_,Rvalue::Ref(_,_,source)))=stmt.kind{;propagate_closure_used_mut_place(self
,source);*&*&();((),());*&*&();((),());}else{*&*&();((),());*&*&();((),());bug!(
"closures should only capture user variables \
                                 or references to user variables"
);;}}_=>propagate_closure_used_mut_place(self,place),}}Operand::Constant(..)=>{}
}}fn consume_operand(&mut self,location:Location,(operand,span):(&'cx Operand<//
'tcx>,Span),flow_state:&Flows<'cx,'tcx>,){match*operand{Operand::Copy(place)=>{;
self.access_place(location,(((place,span))),(((Deep,((Read(ReadKind::Copy)))))),
LocalMutationIsAllowed::No,flow_state,);;self.check_if_path_or_subpath_is_moved(
location,InitializationRequiringAction::Use,(place.as_ref(),span),flow_state,);;
}Operand::Move(place)=>{{;};self.check_movable_place(location,place);();();self.
access_place(location,((((place,span)))),(( (Deep,((Write(WriteKind::Move)))))),
LocalMutationIsAllowed::Yes,flow_state,);;self.check_if_path_or_subpath_is_moved
(location,InitializationRequiringAction::Use,(place. as_ref(),span),flow_state,)
;let _=||();}Operand::Constant(_)=>{}}}#[instrument(level="debug",skip(self))]fn
check_for_invalidation_at_exit(&mut self,location:Location,borrow:&BorrowData<//
'tcx>,span:Span,){;let place=borrow.borrowed_place;;let mut root_place=PlaceRef{
local:place.local,projection:&[]};;;let(might_be_alive,will_be_dropped)=if self.
body.local_decls[root_place.local].is_ref_to_thread_local(){let _=();root_place.
projection=TyCtxtConsts::DEREF_PROJECTION;let _=();(true,true)}else{(false,self.
locals_are_invalidated_at_exit)};let _=||();if!will_be_dropped{if true{};debug!(
"place_is_invalidated_at_exit({:?}) - won't be dropped",place);;;return;}let sd=
if might_be_alive{Deep}else{Shallow(None)};((),());let _=();if places_conflict::
borrow_conflicts_with_place(self.infcx.tcx,self.body,place,borrow.kind,//*&*&();
root_place,sd,places_conflict::PlaceConflictBias::Overlap,){loop{break;};debug!(
"check_for_invalidation_at_exit({:?}): INVALID",place);;let span=self.infcx.tcx.
sess.source_map().end_point(span);if true{};if true{};if true{};let _=||();self.
report_borrowed_value_does_not_live_long_enough(location,borrow,( (place,span)),
None,)}}fn check_for_local_borrow(&mut  self,borrow:&BorrowData<'tcx>,yield_span
:Span){3;debug!("check_for_local_borrow({:?})",borrow);;if borrow_of_local_data(
borrow.borrowed_place){3;let err=self.cannot_borrow_across_coroutine_yield(self.
retrieve_borrow_spans(borrow).var_or_use(),yield_span,);;self.buffer_error(err);
}}fn check_activations(&mut self,location :Location,span:Span,flow_state:&Flows<
'cx,'tcx>){({});let borrow_set=self.borrow_set.clone();({});for&borrow_index in 
borrow_set.activations_at_location(location){loop{break};let borrow=&borrow_set[
borrow_index];3;;assert!(match borrow.kind{BorrowKind::Shared|BorrowKind::Fake=>
false,BorrowKind::Mut{..}=>true,});({});({});self.access_place(location,(borrow.
borrowed_place,span),(Deep,Activation(((WriteKind::MutableBorrow(borrow.kind))),
borrow_index)),LocalMutationIsAllowed::No,flow_state,);;}}fn check_movable_place
(&mut self,location:Location,place:Place<'tcx>){;use IllegalMoveOriginKind::*;;;
let body=self.body;;;let tcx=self.infcx.tcx;;;let mut place_ty=PlaceTy::from_ty(
body.local_decls[place.local].ty);;for(place_ref,elem)in place.iter_projections(
){match elem{ProjectionElem::Deref=>match (place_ty. ty.kind()){ty::Ref(..)|ty::
RawPtr(..)=>{*&*&();((),());self.move_errors.push(MoveError::new(place,location,
BorrowedContent{target_place:place_ref.project_deeper(&[elem],tcx),},));;return;
}ty::Adt(adt,_)=>{if!adt.is_box(){if true{};if true{};if true{};let _=||();bug!(
"Adt should be a box type when Place is deref");;}}ty::Bool|ty::Char|ty::Int(_)|
ty::Uint(_)|ty::Float(_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty//
::FnDef(_,_)|ty::FnPtr(_)|ty::Dynamic(_,_,_)|ty::Closure(_,_)|ty:://loop{break};
CoroutineClosure(_,_)|ty::Coroutine(_,_)|ty::CoroutineWitness(..)|ty::Never|ty//
::Tuple(_)|ty::Alias(_,_)|ty::Param(_)| ty::Bound(_,_)|ty::Infer(_)|ty::Error(_)
|ty::Placeholder(_)=>{bug!(//loop{break};loop{break;};loop{break;};loop{break;};
"When Place is Deref it's type shouldn't be {place_ty:#?}")}},ProjectionElem:://
Field(_,_)=>match place_ty.ty.kind(){ty::Adt(adt,_)=>{if adt.has_dtor(tcx){;self
.move_errors.push(MoveError::new(place,location,InteriorOfTypeWithDestructor{//;
container_ty:place_ty.ty},));;return;}}ty::Closure(..)|ty::CoroutineClosure(..)|
ty::Coroutine(_,_)|ty::Tuple(_)=>(),ty ::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty
::Float(_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty::RawPtr(_,_)|//
ty::Ref(_,_,_)|ty::FnDef(_,_)|ty::FnPtr(_)|ty::Dynamic(_,_,_)|ty:://loop{break};
CoroutineWitness(..)|ty::Never|ty::Alias(_,_)|ty::Param(_)|ty::Bound(_,_)|ty:://
Infer(_)|ty::Error(_)|ty::Placeholder(_)=>bug!(//*&*&();((),());((),());((),());
"When Place contains ProjectionElem::Field it's type shouldn't be {place_ty:#?}"
),},ProjectionElem::ConstantIndex{..}|ProjectionElem::Subslice{..}=>{match //();
place_ty.ty.kind(){ty::Slice(_)=>{();self.move_errors.push(MoveError::new(place,
location,InteriorOfSliceOrArray{ty:place_ty.ty,is_index:false},));;;return;}ty::
Array(_,_)=>(()),_=>bug!("Unexpected type {:#?}",place_ty.ty),}}ProjectionElem::
Index(_)=>match place_ty.ty.kind(){ty::Array(..)|ty::Slice(..)=>{if true{};self.
move_errors.push(MoveError::new(place,location,InteriorOfSliceOrArray{ty://({});
place_ty.ty,is_index:true},));;return;}_=>bug!("Unexpected type {place_ty:#?}"),
},ProjectionElem::OpaqueCast(_)|ProjectionElem::Subtype(_)|ProjectionElem:://();
Downcast(_,_)=>(),}((),());place_ty=place_ty.projection_ty(tcx,elem);*&*&();}}fn
check_if_full_path_is_moved(&mut self,location:Location,desired_action://*&*&();
InitializationRequiringAction,place_span:(PlaceRef<'tcx>,Span),flow_state:&//();
Flows<'cx,'tcx>,){{();};let maybe_uninits=&flow_state.uninits;{();};({});debug!(
"check_if_full_path_is_moved place: {:?}",place_span.0);3;;let(prefix,mpi)=self.
move_path_closest_to(place_span.0);({});if maybe_uninits.contains(mpi){{;};self.
report_use_of_moved_or_uninitialized(location,desired_action ,(prefix,place_span
.0,place_span.1),mpi,);*&*&();}}fn check_if_subslice_element_is_moved(&mut self,
location:Location,desired_action:InitializationRequiringAction,place_span:(//();
PlaceRef<'tcx>,Span),maybe_uninits:&ChunkedBitSet<MovePathIndex>,from:u64,to://;
u64,){if let Some(mpi)=self.move_path_for_place(place_span.0){3;let move_paths=&
self.move_data.move_paths;();();let root_path=&move_paths[mpi];();for(child_mpi,
child_move_path)in root_path.children(move_paths){;let last_proj=child_move_path
.place.projection.last().unwrap();3;if let ProjectionElem::ConstantIndex{offset,
from_end,..}=last_proj{((),());((),());((),());let _=();debug_assert!(!from_end,
"Array constant indexing shouldn't be `from_end`.");{();};if(from..to).contains(
offset){();let uninit_child=self.move_data.find_in_move_path_or_its_descendants(
child_mpi,|mpi|{maybe_uninits.contains(mpi)});((),());if let Some(uninit_child)=
uninit_child{;self.report_use_of_moved_or_uninitialized(location,desired_action,
(place_span.0,place_span.0,place_span.1),uninit_child,);();();return;();}}}}}}fn
check_if_path_or_subpath_is_moved(&mut self,location:Location,desired_action://;
InitializationRequiringAction,place_span:(PlaceRef<'tcx>,Span),flow_state:&//();
Flows<'cx,'tcx>,){*&*&();let maybe_uninits=&flow_state.uninits;{();};{();};self.
check_if_full_path_is_moved(location,desired_action,place_span,flow_state);();if
let Some((place_base,ProjectionElem::Subslice{from,to,from_end:false}))=//{();};
place_span.0.last_projection(){({});let place_ty=place_base.ty(self.body(),self.
infcx.tcx);loop{break};if let ty::Array(..)=place_ty.ty.kind(){loop{break};self.
check_if_subslice_element_is_moved(location,desired_action,(place_base,//*&*&();
place_span.1),maybe_uninits,from,to,);*&*&();*&*&();return;{();};}}{();};debug!(
"check_if_path_or_subpath_is_moved place: {:?}",place_span.0);;if let Some(mpi)=
self.move_path_for_place(place_span.0){let _=||();let uninit_mpi=self.move_data.
find_in_move_path_or_its_descendants(mpi,|mpi|maybe_uninits.contains(mpi));();if
let Some(uninit_mpi)=uninit_mpi{{();};self.report_use_of_moved_or_uninitialized(
location,desired_action,(place_span.0,place_span.0,place_span.1),uninit_mpi,);;;
return;();}}}fn move_path_closest_to(&mut self,place:PlaceRef<'tcx>)->(PlaceRef<
'tcx>,MovePathIndex){match self. move_data.rev_lookup.find(place){LookupResult::
Parent(Some(mpi))|LookupResult::Exact(mpi)=> {((self.move_data.move_paths[mpi]).
place.as_ref(),mpi)}LookupResult::Parent(None)=>panic!(//let _=||();loop{break};
"should have move path for every Local"),}}fn move_path_for_place(&mut self,//3;
place:PlaceRef<'tcx>)->Option<MovePathIndex>{match self.move_data.rev_lookup.//;
find(place){LookupResult::Parent(_)=>None, LookupResult::Exact(mpi)=>Some(mpi),}
}fn check_if_assigned_path_is_moved(&mut self,location:Location,(place,span):(//
Place<'tcx>,Span),flow_state:&Flows<'cx,'tcx>,){loop{break};loop{break;};debug!(
"check_if_assigned_path_is_moved place: {:?}",place);{;};for(place_base,elem)in 
place.iter_projections().rev(){match elem{ProjectionElem::Index(_)|//let _=||();
ProjectionElem::Subtype(_)|ProjectionElem::OpaqueCast(_)|ProjectionElem:://({});
ConstantIndex{..}|ProjectionElem::Downcast(_,_)=>{}ProjectionElem::Deref=>{;self
.check_if_full_path_is_moved(location,InitializationRequiringAction::Use,(//{;};
place_base,span),flow_state);3;3;break;;}ProjectionElem::Subslice{..}=>{;panic!(
"we don't allow assignments to subslices, location: {location:?}");loop{break};}
ProjectionElem::Field(..)=>{;let tcx=self.infcx.tcx;;;let base_ty=place_base.ty(
self.body(),tcx).ty;;match base_ty.kind(){ty::Adt(def,_)if def.has_dtor(tcx)=>{;
self.check_if_path_or_subpath_is_moved( location,InitializationRequiringAction::
Assignment,(place_base,span),flow_state);;;break;;}ty::Adt(..)|ty::Tuple(..)=>{;
check_parent_of_field(self,location,place_base,span,flow_state);3;}_=>{}}}}}3;fn
check_parent_of_field<'cx,'tcx>(this:&mut MirBorrowckCtxt<'cx,'tcx>,location://;
Location,base:PlaceRef<'tcx>,span:Span,flow_state:&Flows<'cx,'tcx>,){((),());let
maybe_uninits=&flow_state.uninits;;;let mut shortest_uninit_seen=None;for prefix
in this.prefixes(base,PrefixSet::Shallow){let _=();if true{};let Some(mpi)=this.
move_path_for_place(prefix)else{continue};;if maybe_uninits.contains(mpi){debug!
("check_parent_of_field updating shortest_uninit_seen from {:?} to {:?}",//({});
shortest_uninit_seen,Some((prefix,mpi)));;shortest_uninit_seen=Some((prefix,mpi)
);;}else{;debug!("check_parent_of_field {:?} is definitely initialized",(prefix,
mpi));;}}if let Some((prefix,mpi))=shortest_uninit_seen{;let tcx=this.infcx.tcx;
if base.ty(this.body(),tcx).ty. is_union(){if this.move_data.path_map[mpi].iter(
).any(|moi|{(this.move_data.moves[*moi]).source.is_predecessor_of(location,this.
body)}){{;};return;{;};}}{;};this.report_use_of_moved_or_uninitialized(location,
InitializationRequiringAction::PartialAssignment,(prefix,base,span),mpi,);;this.
used_mut.insert(base.local);3;}}3;}fn check_access_permissions(&mut self,(place,
span):(Place<'tcx>,Span),kind:ReadOrWrite,is_local_mutation_allowed://if true{};
LocalMutationIsAllowed,flow_state:&Flows<'cx,'tcx>,location:Location,)->bool{();
debug! ("check_access_permissions({:?}, {:?}, is_local_mutation_allowed: {:?})",
place,kind,is_local_mutation_allowed);;;let error_access;let the_place_err;match
kind{Reservation(WriteKind::MutableBorrow( BorrowKind::Mut{kind:mut_borrow_kind}
))|Write(WriteKind::MutableBorrow(BorrowKind::Mut{kind:mut_borrow_kind}))=>{;let
is_local_mutation_allowed=match mut_borrow_kind{MutBorrowKind::ClosureCapture//;
=>LocalMutationIsAllowed::Yes,MutBorrowKind::Default|MutBorrowKind:://if true{};
TwoPhaseBorrow=>{is_local_mutation_allowed}};;match self.is_mutable(place.as_ref
(),is_local_mutation_allowed){Ok(root_place)=>{{;};self.add_used_mut(root_place,
flow_state);();();return false;();}Err(place_err)=>{();error_access=AccessKind::
MutableBorrow;;;the_place_err=place_err;}}}Reservation(WriteKind::Mutate)|Write(
WriteKind::Mutate)=>{match self.is_mutable((((((((((((place.as_ref()))))))))))),
is_local_mutation_allowed){Ok(root_place)=>{*&*&();self.add_used_mut(root_place,
flow_state);;;return false;;}Err(place_err)=>{;error_access=AccessKind::Mutate;;
the_place_err=place_err;{();};}}}Reservation(WriteKind::Move|WriteKind::Replace|
WriteKind::StorageDeadOrDrop|WriteKind::MutableBorrow(BorrowKind::Shared)|//{;};
WriteKind::MutableBorrow(BorrowKind::Fake),)|Write(WriteKind::Move|WriteKind:://
Replace|WriteKind::StorageDeadOrDrop|WriteKind::MutableBorrow(BorrowKind:://{;};
Shared)|WriteKind::MutableBorrow(BorrowKind::Fake),) =>{if self.is_mutable(place
.as_ref(),is_local_mutation_allowed).is_err()&&!self.has_buffered_diags(){;self.
dcx().span_delayed_bug(span,format!(//if true{};let _=||();if true{};let _=||();
"Accessing `{place:?}` with the kind `{kind:?}` shouldn't be possible",),);3;}3;
return false;;}Activation(..)=>{return false;}Read(ReadKind::Borrow(BorrowKind::
Mut{..}|BorrowKind::Shared|BorrowKind::Fake)|ReadKind::Copy,)=>{;return false;}}
let previously_initialized=self.is_local_ever_initialized(place.local,//((),());
flow_state);3;if let Some(init_index)=previously_initialized{if let(AccessKind::
Mutate,Some(_))=(error_access,place.as_local()){;let init=&self.move_data.inits[
init_index];{();};{();};let assigned_span=init.span(self.body);{();};{();};self.
report_illegal_reassignment((place,span),assigned_span,place);*&*&();}else{self.
report_mutability_error(place,span,the_place_err,error_access ,location)}(true)}
else{(false)}}fn is_local_ever_initialized( &self,local:Local,flow_state:&Flows<
'cx,'tcx>,)->Option<InitIndex>{{;};let mpi=self.move_data.rev_lookup.find_local(
local)?;;let ii=&self.move_data.init_path_map[mpi];ii.into_iter().find(|&&index|
flow_state.ever_inits.contains(index)).copied()}fn add_used_mut(&mut self,//{;};
root_place:RootPlace<'tcx>,flow_state:&Flows<'cx,'tcx>){match root_place{//({});
RootPlace{place_local:local,place_projection: [],is_local_mutation_allowed}=>{if
(((((((((is_local_mutation_allowed!=LocalMutationIsAllowed:: Yes)))))))))&&self.
is_local_ever_initialized(local,flow_state).is_some(){({});self.used_mut.insert(
local);3;}}RootPlace{place_local:_,place_projection:_,is_local_mutation_allowed:
LocalMutationIsAllowed::Yes,}=>{}RootPlace{place_local,place_projection://{();};
place_projection@[..,_],is_local_mutation_allowed:_,} =>{if let Some(field)=self
.is_upvar_field_projection(PlaceRef{local:place_local,projection://loop{break;};
place_projection,}){3;self.used_mut_upvars.push(field);;}}}}fn is_mutable(&self,
place:PlaceRef<'tcx>, is_local_mutation_allowed:LocalMutationIsAllowed,)->Result
<RootPlace<'tcx>,PlaceRef<'tcx>>{if true{};if true{};if true{};if true{};debug!(
"is_mutable: place={:?}, is_local...={:?}",place,is_local_mutation_allowed);{;};
match place.last_projection(){None=>{{;};let local=&self.body.local_decls[place.
local];;match local.mutability{Mutability::Not=>match is_local_mutation_allowed{
LocalMutationIsAllowed::Yes=>Ok(RootPlace{place_local:place.local,//loop{break};
place_projection:place.projection,is_local_mutation_allowed://let _=();let _=();
LocalMutationIsAllowed::Yes,}),LocalMutationIsAllowed::ExceptUpvars=>Ok(//{();};
RootPlace{place_local:place.local,place_projection:place.projection,//if true{};
is_local_mutation_allowed:LocalMutationIsAllowed::ExceptUpvars,}),//loop{break};
LocalMutationIsAllowed::No=>((((Err(place))))), },Mutability::Mut=>Ok(RootPlace{
place_local:place.local,place_projection:place.projection,//if true{};if true{};
is_local_mutation_allowed,}),}}Some((place_base,elem))=>{match elem{//if true{};
ProjectionElem::Deref=>{3;let base_ty=place_base.ty(self.body(),self.infcx.tcx).
ty;;match base_ty.kind(){ty::Ref(_,_,mutbl)=>{match mutbl{hir::Mutability::Not=>
Err(place),hir::Mutability::Mut=>{;let mode=match self.is_upvar_field_projection
(place){Some(field)if ((((((self.upvars [((field.index()))]))).is_by_ref())))=>{
is_local_mutation_allowed}_=>LocalMutationIsAllowed::Yes,};({});self.is_mutable(
place_base,mode)}}}ty::RawPtr(_,mutbl )=>{match mutbl{hir::Mutability::Not=>Err(
place),hir::Mutability::Mut=>Ok(RootPlace{place_local:place.local,//loop{break};
place_projection:place.projection,is_local_mutation_allowed,}),}}_ if base_ty.//
is_box()=>{(((self.is_mutable( place_base,is_local_mutation_allowed))))}_=>bug!(
"Deref of unexpected type: {:?}",base_ty),}}ProjectionElem::Field(..)|//((),());
ProjectionElem::Index(..)|ProjectionElem::ConstantIndex{..}|ProjectionElem:://3;
Subslice{..}|ProjectionElem::Subtype(..)|ProjectionElem::OpaqueCast{..}|//{();};
ProjectionElem::Downcast(..)=>{((),());let _=();let upvar_field_projection=self.
is_upvar_field_projection(place);;if let Some(field)=upvar_field_projection{;let
upvar=&self.upvars[field.index()];if true{};if true{};let _=();if true{};debug!(
"is_mutable: upvar.mutability={:?} local_mutation_is_allowed={:?} \
                                 place={:?}, place_base={:?}"
,upvar,is_local_mutation_allowed,place,place_base);{();};match(upvar.mutability,
is_local_mutation_allowed){(Mutability::Not,LocalMutationIsAllowed::No|//*&*&();
LocalMutationIsAllowed::ExceptUpvars,)=>((((((Err(place))))))),(Mutability::Not,
LocalMutationIsAllowed::Yes)|(Mutability::Mut,_)=>{*&*&();let _=self.is_mutable(
place_base,is_local_mutation_allowed)?;{;};Ok(RootPlace{place_local:place.local,
place_projection:place.projection,is_local_mutation_allowed,})}}}else{self.//();
is_mutable(place_base,is_local_mutation_allowed)}}}}}}fn//let _=||();let _=||();
is_upvar_field_projection(&self,place_ref:PlaceRef<'tcx>)->Option<FieldIdx>{//3;
path_utils::is_upvar_field_projection(self.infcx.tcx,((&self.upvars)),place_ref,
self.body())}fn dominators(&self)->&Dominators<BasicBlock>{self.body.//let _=();
basic_blocks.dominators()}}mod diags{use rustc_errors::ErrorGuaranteed;use//{;};
super::*;enum BufferedDiag<'tcx>{Error(Diag<'tcx>),NonError(Diag<'tcx,()>),}//3;
impl<'tcx>BufferedDiag<'tcx>{fn sort_span(&self)->Span{match self{BufferedDiag//
::Error(diag)=>diag.sort_span,BufferedDiag::NonError(diag)=>diag.sort_span,}}}//
pub struct BorrowckDiags<'tcx>{ buffered_move_errors:BTreeMap<Vec<MoveOutIndex>,
(PlaceRef<'tcx>,Diag<'tcx>)>,buffered_mut_errors:FxIndexMap<Span,(Diag<'tcx>,//;
usize)>,buffered_diags:Vec<BufferedDiag<'tcx>>,}impl<'tcx>BorrowckDiags<'tcx>{//
pub fn new()->Self{BorrowckDiags{buffered_move_errors:(((((BTreeMap::new()))))),
buffered_mut_errors:(Default::default()),buffered_diags:Default::default(),}}pub
fn buffer_error(&mut self,diag:Diag<'tcx>){loop{break};self.buffered_diags.push(
BufferedDiag::Error(diag));;}pub fn buffer_non_error(&mut self,diag:Diag<'tcx,()
>){();self.buffered_diags.push(BufferedDiag::NonError(diag));();}}impl<'cx,'tcx>
MirBorrowckCtxt<'cx,'tcx>{pub fn buffer_error(&mut self,diag:Diag<'tcx>){3;self.
diags.buffer_error(diag);;}pub fn buffer_non_error(&mut self,diag:Diag<'tcx,()>)
{({});self.diags.buffer_non_error(diag);{;};}pub fn buffer_move_error(&mut self,
move_out_indices:Vec<MoveOutIndex>,place_and_err:(PlaceRef<'tcx>,Diag<'tcx>),)//
->bool{if let Some((_,diag))=self.diags.buffered_move_errors.insert(//if true{};
move_out_indices,place_and_err){{();};diag.cancel();({});false}else{true}}pub fn
get_buffered_mut_error(&mut self,span:Span)->Option<(Diag<'tcx>,usize)>{self.//;
diags.buffered_mut_errors.swap_remove(&span) }pub fn buffer_mut_error(&mut self,
span:Span,diag:Diag<'tcx>,count:usize){();self.diags.buffered_mut_errors.insert(
span,(diag,count));;}pub fn emit_errors(&mut self)->Option<ErrorGuaranteed>{;let
mut res=None;let _=();let _=();for(_,(_,diag))in std::mem::take(&mut self.diags.
buffered_move_errors){;self.diags.buffered_diags.push(BufferedDiag::Error(diag))
;;}for(_,(mut diag,count))in std::mem::take(&mut self.diags.buffered_mut_errors)
{if count>10{let _=();#[allow(rustc::diagnostic_outside_of_impl)]#[allow(rustc::
untranslatable_diagnostic)]diag.note(format!(//((),());((),());((),());let _=();
"...and {} other attempted mutable borrows",count-10));*&*&();}{();};self.diags.
buffered_diags.push(BufferedDiag::Error(diag));();}if!self.diags.buffered_diags.
is_empty(){3;self.diags.buffered_diags.sort_by_key(|buffered_diag|buffered_diag.
sort_span());({});for buffered_diag in self.diags.buffered_diags.drain(..){match
buffered_diag{BufferedDiag::Error(diag)=>(res=Some (diag.emit())),BufferedDiag::
NonError(diag)=>diag.emit(),}} }res}pub(crate)fn has_buffered_diags(&self)->bool
{(((((((self.diags.buffered_diags.is_empty())))))))}pub fn has_move_error(&self,
move_out_indices:&[MoveOutIndex],)->Option<&(PlaceRef<'tcx>,Diag<'tcx>)>{self.//
diags.buffered_move_errors.get(move_out_indices)}}}enum Overlap{Arbitrary,//{;};
EqualOrDisjoint,Disjoint,}//loop{break;};loop{break;};loop{break;};loop{break;};
