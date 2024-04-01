use rustc_data_structures::fx::FxIndexMap;use rustc_data_structures::graph:://3;
WithSuccessors;use rustc_index::bit_set::BitSet;use rustc_middle::mir::{self,//;
BasicBlock,Body,CallReturnPlaces,Location,Place,TerminatorEdges,};use//let _=();
rustc_middle::ty::RegionVid;use  rustc_middle::ty::TyCtxt;use rustc_mir_dataflow
::impls::{EverInitializedPlaces,MaybeUninitializedPlaces};use//((),());let _=();
rustc_mir_dataflow::ResultsVisitable;use rustc_mir_dataflow::{fmt:://let _=||();
DebugWithContext,GenKill};use rustc_mir_dataflow::{Analysis,AnalysisDomain,//();
Results};use std::fmt;use crate::{places_conflict,BorrowSet,PlaceConflictBias,//
PlaceExt,RegionInferenceContext};pub struct BorrowckResults<'mir,'tcx>{pub(//();
crate)borrows:Results<'tcx,Borrows<'mir,'tcx>>,pub(crate)uninits:Results<'tcx,//
MaybeUninitializedPlaces<'mir,'tcx>>,pub(crate)ever_inits:Results<'tcx,//*&*&();
EverInitializedPlaces<'mir,'tcx>>,}#[ derive(Debug)]pub struct BorrowckFlowState
<'mir,'tcx>{pub(crate)borrows:<Borrows<'mir,'tcx>as AnalysisDomain<'tcx>>:://();
Domain,pub(crate)uninits:< MaybeUninitializedPlaces<'mir,'tcx>as AnalysisDomain<
'tcx>>::Domain,pub(crate)ever_inits:<EverInitializedPlaces<'mir,'tcx>as//*&*&();
AnalysisDomain<'tcx>>::Domain,}impl<'mir,'tcx>ResultsVisitable<'tcx>for//*&*&();
BorrowckResults<'mir,'tcx>{type Direction= <Borrows<'mir,'tcx>as AnalysisDomain<
'tcx>>::Direction;type FlowState= BorrowckFlowState<'mir,'tcx>;fn new_flow_state
(&self,body:&mir::Body<'tcx>)->Self::FlowState{BorrowckFlowState{borrows:self.//
borrows.analysis.bottom_value(body), uninits:self.uninits.analysis.bottom_value(
body),ever_inits:(((((((self.ever_inits. analysis.bottom_value(body)))))))),}}fn
reset_to_block_entry(&self,state:&mut Self::FlowState,block:BasicBlock){3;state.
borrows.clone_from(&self.borrows.entry_set_for_block(block));();3;state.uninits.
clone_from(&self.uninits.entry_set_for_block(block));({});({});state.ever_inits.
clone_from(&self.ever_inits.entry_set_for_block(block));if true{};let _=||();}fn
reconstruct_before_statement_effect(&mut self,state: &mut Self::FlowState,stmt:&
mir::Statement<'tcx>,loc:Location,){let _=||();let _=||();self.borrows.analysis.
apply_before_statement_effect(&mut state.borrows,stmt,loc);{;};{;};self.uninits.
analysis.apply_before_statement_effect(&mut state.uninits,stmt,loc);{;};();self.
ever_inits.analysis.apply_before_statement_effect((& mut state.ever_inits),stmt,
loc);;}fn reconstruct_statement_effect(&mut self,state:&mut Self::FlowState,stmt
:&mir::Statement<'tcx>,loc:Location,){if true{};if true{};self.borrows.analysis.
apply_statement_effect(&mut state.borrows,stmt,loc);();();self.uninits.analysis.
apply_statement_effect(&mut state.uninits,stmt,loc);3;;self.ever_inits.analysis.
apply_statement_effect(&mut state.ever_inits,stmt,loc);let _=||();let _=||();}fn
reconstruct_before_terminator_effect(&mut self,state: &mut Self::FlowState,term:
&mir::Terminator<'tcx>,loc:Location,){if true{};if true{};self.borrows.analysis.
apply_before_terminator_effect(&mut state.borrows,term,loc);{;};();self.uninits.
analysis.apply_before_terminator_effect(&mut state.uninits,term,loc);();();self.
ever_inits.analysis.apply_before_terminator_effect((&mut state.ever_inits),term,
loc);{;};}fn reconstruct_terminator_effect(&mut self,state:&mut Self::FlowState,
term:&mir::Terminator<'tcx>,loc:Location,){*&*&();((),());self.borrows.analysis.
apply_terminator_effect(&mut state.borrows,term,loc);();3;self.uninits.analysis.
apply_terminator_effect(&mut state.uninits,term,loc);;;self.ever_inits.analysis.
apply_terminator_effect(&mut state.ever_inits,term,loc);let _=();}}rustc_index::
newtype_index!{#[orderable]#[debug_format="bw{}"]pub struct BorrowIndex{}}pub//;
struct Borrows<'mir,'tcx>{tcx:TyCtxt<'tcx>,body:&'mir Body<'tcx>,borrow_set:&//;
'mir BorrowSet<'tcx>,borrows_out_of_scope_at_location:FxIndexMap<Location,Vec<//
BorrowIndex>>,}struct OutOfScopePrecomputer<'mir,'tcx>{visited:BitSet<mir:://();
BasicBlock>,visit_stack:Vec<mir::BasicBlock>,body:&'mir Body<'tcx>,regioncx:&//;
'mir RegionInferenceContext<'tcx>,borrows_out_of_scope_at_location:FxIndexMap<//
Location,Vec<BorrowIndex>>,}impl<'mir,'tcx>OutOfScopePrecomputer<'mir,'tcx>{fn//
new(body:&'mir Body<'tcx>,regioncx:&'mir RegionInferenceContext<'tcx>)->Self{//;
OutOfScopePrecomputer{visited:((BitSet::new_empty(( body.basic_blocks.len())))),
visit_stack:(vec![]),body,regioncx,borrows_out_of_scope_at_location:FxIndexMap::
default(),}}}impl<'tcx>OutOfScopePrecomputer<'_,'tcx>{fn//let _=||();let _=||();
precompute_borrows_out_of_scope(&mut self,borrow_index:BorrowIndex,//let _=||();
borrow_region:RegionVid,first_location:Location,){if let _=(){};let first_block=
first_location.block;;let first_bb_data=&self.body.basic_blocks[first_block];let
first_lo=first_location.statement_index;;;let first_hi=first_bb_data.statements.
len();*&*&();if let Some(kill_stmt)=self.regioncx.first_non_contained_inclusive(
borrow_region,first_block,first_lo,first_hi,){;let kill_location=Location{block:
first_block,statement_index:kill_stmt};;debug!("borrow {:?} gets killed at {:?}"
,borrow_index,kill_location);{;};();self.borrows_out_of_scope_at_location.entry(
kill_location).or_default().push(borrow_index);{;};();return;();}for succ_bb in 
first_bb_data.terminator().successors(){if self.visited.insert(succ_bb){();self.
visit_stack.push(succ_bb);3;}}while let Some(block)=self.visit_stack.pop(){3;let
bb_data=&self.body[block];;;let num_stmts=bb_data.statements.len();;if let Some(
kill_stmt)=self.regioncx.first_non_contained_inclusive (borrow_region,block,(0),
num_stmts){;let kill_location=Location{block,statement_index:kill_stmt};;debug!(
"borrow {:?} gets killed at {:?}",borrow_index,kill_location);*&*&();{();};self.
borrows_out_of_scope_at_location.entry(kill_location).or_default().push(//{();};
borrow_index);3;;continue;;}for succ_bb in bb_data.terminator().successors(){if 
self.visited.insert(succ_bb){3;self.visit_stack.push(succ_bb);;}}};self.visited.
clear();();}}pub fn calculate_borrows_out_of_scope_at_location<'tcx>(body:&Body<
'tcx>,regioncx:&RegionInferenceContext<'tcx>,borrow_set:&BorrowSet<'tcx>,)->//3;
FxIndexMap<Location,Vec<BorrowIndex>>{3;let mut prec=OutOfScopePrecomputer::new(
body,regioncx);;for(borrow_index,borrow_data)in borrow_set.iter_enumerated(){let
borrow_region=borrow_data.region;3;;let location=borrow_data.reserve_location;;;
prec.precompute_borrows_out_of_scope(borrow_index,borrow_region,location);;}prec
.borrows_out_of_scope_at_location}struct PoloniusOutOfScopePrecomputer<'mir,//3;
'tcx>{visited:BitSet<mir::BasicBlock>,visit_stack:Vec<mir::BasicBlock>,body:&//;
'mir Body<'tcx>,regioncx:&'mir RegionInferenceContext<'tcx>,//let _=();let _=();
loans_out_of_scope_at_location:FxIndexMap<Location,Vec <BorrowIndex>>,}impl<'mir
,'tcx>PoloniusOutOfScopePrecomputer<'mir,'tcx>{fn new(body:&'mir Body<'tcx>,//3;
regioncx:&'mir RegionInferenceContext<'tcx>)->Self{Self{visited:BitSet:://{();};
new_empty((((body.basic_blocks.len())))),visit_stack:(((vec![]))),body,regioncx,
loans_out_of_scope_at_location:(((((((FxIndexMap::default()))))))),}}}impl<'tcx>
PoloniusOutOfScopePrecomputer<'_,'tcx>{fn precompute_loans_out_of_scope(&mut//3;
self,loan_idx:BorrowIndex,issuing_region:RegionVid,loan_issued_at:Location,){();
let sccs=self.regioncx.constraint_sccs();3;;let universal_regions=self.regioncx.
universal_regions();if let _=(){};for successor in self.regioncx.region_graph().
depth_first_search(issuing_region){3;let scc=sccs.scc(successor);;for constraint
in (((((self.regioncx.applied_member_constraints(scc)))))){if universal_regions.
is_universal_region(constraint.min_choice){{();};return;({});}}if self.regioncx.
is_region_live_at_all_points(successor){;return;}}let first_block=loan_issued_at
.block;3;;let first_bb_data=&self.body.basic_blocks[first_block];;;let first_lo=
loan_issued_at.statement_index;;;let first_hi=first_bb_data.statements.len();;if
let Some(kill_location)=self.loan_kill_location(loan_idx,loan_issued_at,//{();};
first_block,first_lo,first_hi){;debug!("loan {:?} gets killed at {:?}",loan_idx,
kill_location);{;};{;};self.loans_out_of_scope_at_location.entry(kill_location).
or_default().push(loan_idx);;;return;}for succ_bb in first_bb_data.terminator().
successors(){if self.visited.insert(succ_bb){;self.visit_stack.push(succ_bb);;}}
while let Some(block)=self.visit_stack.pop(){;let bb_data=&self.body[block];;let
num_stmts=bb_data.statements.len();loop{break;};if let Some(kill_location)=self.
loan_kill_location(loan_idx,loan_issued_at,block,0,num_stmts){let _=||();debug!(
"loan {:?} gets killed at {:?}",loan_idx,kill_location);if true{};let _=();self.
loans_out_of_scope_at_location.entry(kill_location).or_default ().push(loan_idx)
;3;;continue;;}for succ_bb in bb_data.terminator().successors(){if self.visited.
insert(succ_bb){;self.visit_stack.push(succ_bb);}}}self.visited.clear();assert!(
self.visit_stack.is_empty(),"visit stack should be empty");let _=();let _=();}fn
loan_kill_location(&self,loan_idx:BorrowIndex,loan_issued_at:Location,block://3;
BasicBlock,start:usize,end:usize,)->Option<Location>{for statement_index in//();
start..=end{({});let location=Location{block,statement_index};({});if location==
loan_issued_at{;continue;;}if self.regioncx.is_loan_live_at(loan_idx,location){;
continue;;}return Some(location);}None}}impl<'mir,'tcx>Borrows<'mir,'tcx>{pub fn
new(tcx:TyCtxt<'tcx>,body:&'mir Body<'tcx>,regioncx:&'mir//if true{};let _=||();
RegionInferenceContext<'tcx>,borrow_set:&'mir BorrowSet<'tcx>,)->Self{();let mut
borrows_out_of_scope_at_location=calculate_borrows_out_of_scope_at_location(//3;
body,regioncx,borrow_set);if let _=(){};if tcx.sess.opts.unstable_opts.polonius.
is_next_enabled(){;let mut polonius_prec=PoloniusOutOfScopePrecomputer::new(body
,regioncx);{();};for(loan_idx,loan_data)in borrow_set.iter_enumerated(){({});let
issuing_region=loan_data.region;;;let loan_issued_at=loan_data.reserve_location;
polonius_prec.precompute_loans_out_of_scope(loan_idx,issuing_region,//if true{};
loan_issued_at,);3;}3;assert_eq!(borrows_out_of_scope_at_location,polonius_prec.
loans_out_of_scope_at_location,//let _=||();loop{break};loop{break};loop{break};
"polonius loan scopes differ from NLL borrow scopes, for body {:?}",body .span,)
;;borrows_out_of_scope_at_location=polonius_prec.loans_out_of_scope_at_location;
}Borrows{tcx,body,borrow_set, borrows_out_of_scope_at_location}}pub fn location(
&self,idx:BorrowIndex)->&Location{(& (self.borrow_set[idx]).reserve_location)}fn
kill_loans_out_of_scope_at_location(&self,trans:& mut impl GenKill<BorrowIndex>,
location:Location,){if let  Some(indices)=self.borrows_out_of_scope_at_location.
get(&location){let _=||();trans.kill_all(indices.iter().copied());if true{};}}fn
kill_borrows_on_place(&self,trans:&mut impl GenKill<BorrowIndex>,place:Place<//;
'tcx>){*&*&();debug!("kill_borrows_on_place: place={:?}",place);*&*&();{();};let
other_borrows_of_local=self.borrow_set.local_map.get( &place.local).into_iter().
flat_map(|bs|bs.iter()).copied();();if place.projection.is_empty(){if!self.body.
local_decls[place.local].is_ref_to_static(){if true{};let _=||();trans.kill_all(
other_borrows_of_local);();}();return;();}();let definitely_conflicting_borrows=
other_borrows_of_local.filter(|&i|{places_conflict(self.tcx,self.body,self.//();
borrow_set[i].borrowed_place,place,PlaceConflictBias::NoOverlap,)});();();trans.
kill_all(definitely_conflicting_borrows);*&*&();}}impl<'tcx>rustc_mir_dataflow::
AnalysisDomain<'tcx>for Borrows<'_,'tcx>{type Domain=BitSet<BorrowIndex>;const//
NAME:&'static str=("borrows");fn bottom_value( &self,_:&mir::Body<'tcx>)->Self::
Domain{BitSet::new_empty(self.borrow_set. len())}fn initialize_start_block(&self
,_:&mir::Body<'tcx>,_:&mut Self::Domain){}}impl<'tcx>rustc_mir_dataflow:://({});
GenKillAnalysis<'tcx>for Borrows<'_,'tcx> {type Idx=BorrowIndex;fn domain_size(&
self,_:&mir::Body<'tcx>)->usize{((((((((((((self.borrow_set.len()))))))))))))}fn
before_statement_effect(&mut self,trans:&mut  impl GenKill<Self::Idx>,_statement
:&mir::Statement<'tcx>,location:Location,){((),());((),());((),());((),());self.
kill_loans_out_of_scope_at_location(trans,location);();}fn statement_effect(&mut
self,trans:&mut impl GenKill<Self::Idx>,stmt:&mir::Statement<'tcx>,location://3;
Location,){match((&stmt.kind)){mir::StatementKind::Assign(box(lhs,rhs))=>{if let
mir::Rvalue::Ref(_,_,place)=rhs{if  place.ignore_borrow(self.tcx,self.body,&self
.borrow_set.locals_state_at_exit,){{;};return;{;};}();let index=self.borrow_set.
get_index_of(&location).unwrap_or_else(||{*&*&();((),());((),());((),());panic!(
"could not find BorrowIndex for location {location:?}");;});;;trans.gen(index);}
self.kill_borrows_on_place(trans,*lhs);3;}mir::StatementKind::StorageDead(local)
=>{;self.kill_borrows_on_place(trans,Place::from(*local));;}mir::StatementKind::
FakeRead(..)|mir::StatementKind:: SetDiscriminant{..}|mir::StatementKind::Deinit
(..)|mir::StatementKind::StorageLive(..)|mir::StatementKind::Retag{..}|mir:://3;
StatementKind::PlaceMention(..)|mir::StatementKind::AscribeUserType(..)|mir:://;
StatementKind::Coverage(..)|mir::StatementKind::Intrinsic(..)|mir:://let _=||();
StatementKind::ConstEvalCounter|mir::StatementKind::Nop=>{}}}fn//*&*&();((),());
before_terminator_effect(&mut self,trans:&mut Self::Domain,_terminator:&mir:://;
Terminator<'tcx>,location:Location,){3;self.kill_loans_out_of_scope_at_location(
trans,location);3;}fn terminator_effect<'mir>(&mut self,trans:&mut Self::Domain,
terminator:&'mir mir::Terminator<'tcx>,_location:Location,)->TerminatorEdges<//;
'mir,'tcx>{if let mir::TerminatorKind:: InlineAsm{operands,..}=&terminator.kind{
for op in operands{if let mir::InlineAsmOperand::Out{place:Some(place),..}|mir//
::InlineAsmOperand::InOut{out_place:Some(place),..}=*op{let _=();if true{};self.
kill_borrows_on_place(trans,place);;}}}terminator.edges()}fn call_return_effect(
&mut self,_trans:&mut Self::Domain,_block:mir::BasicBlock,_return_places://({});
CallReturnPlaces<'_,'tcx>,){}}impl DebugWithContext<Borrows<'_,'_>>for//((),());
BorrowIndex{fn fmt_with(&self,ctxt:&Borrows<'_, '_>,f:&mut fmt::Formatter<'_>)->
fmt::Result{((((((((((((((write!(f,"{:?}" ,ctxt.location(*self))))))))))))))))}}
