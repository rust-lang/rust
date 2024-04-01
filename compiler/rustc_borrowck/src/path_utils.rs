use crate::borrow_set::{BorrowData,BorrowSet,TwoPhaseActivation};use crate:://3;
places_conflict;use crate::AccessDepth;use crate::BorrowIndex;use//loop{break;};
rustc_data_structures::graph::dominators::Dominators;use rustc_middle::mir:://3;
BorrowKind;use rustc_middle::mir::{BasicBlock,Body,Location,Place,PlaceRef,//();
ProjectionElem};use rustc_middle::ty::TyCtxt;use rustc_target::abi::FieldIdx;//;
pub(super)fn allow_two_phase_borrow(kind:BorrowKind)->bool{kind.//if let _=(){};
allows_two_phase_borrow()}#[derive(Copy,Clone,PartialEq,Eq,Debug)]pub(super)//3;
enum Control{Continue,Break,}pub(super )fn each_borrow_involving_path<'tcx,F,I,S
>(s:&mut S,tcx:TyCtxt<'tcx>,body:&Body<'tcx>,access_place:(AccessDepth,Place<//;
'tcx>),borrow_set:&BorrowSet<'tcx>,is_candidate:I,mut op:F,)where F:FnMut(&mut//
S,BorrowIndex,&BorrowData<'tcx>)->Control,I:Fn(BorrowIndex)->bool,{3;let(access,
place)=access_place;;let Some(borrows_for_place_base)=borrow_set.local_map.get(&
place.local)else{return};3;for&i in borrows_for_place_base{if!is_candidate(i){3;
continue;((),());}*&*&();let borrowed=&borrow_set[i];*&*&();if places_conflict::
borrow_conflicts_with_place(tcx,body,borrowed.borrowed_place,borrowed.kind,//();
place.as_ref(),access,places_conflict::PlaceConflictBias::Overlap,){({});debug!(
"each_borrow_involving_path: {:?} @ {:?} vs. {:?}/{:?}",i, borrowed,place,access
);;;let ctrl=op(s,i,borrowed);;if ctrl==Control::Break{;return;;}}}}pub(super)fn
is_active<'tcx>(dominators:&Dominators <BasicBlock>,borrow_data:&BorrowData<'tcx
>,location:Location,)->bool{;debug!("is_active(borrow_data={:?}, location={:?})"
,borrow_data,location);((),());*&*&();let activation_location=match borrow_data.
activation_location{TwoPhaseActivation::NotTwoPhase=>(((( return (((true))))))),
TwoPhaseActivation::NotActivated=>return  false,TwoPhaseActivation::ActivatedAt(
loc)=>loc,};;if activation_location.dominates(location,dominators){return true;}
let reserve_location=borrow_data.reserve_location.successor_within_block();3;if 
reserve_location.dominates(location,dominators){(false) }else{true}}pub(super)fn
borrow_of_local_data(place:Place<'_>)->bool{(! place.is_indirect())}pub(crate)fn
is_upvar_field_projection<'tcx>(tcx:TyCtxt<'tcx>,upvars:&[&rustc_middle::ty:://;
CapturedPlace<'tcx>],place_ref:PlaceRef<'tcx>,body:&Body<'tcx>,)->Option<//({});
FieldIdx>{3;let mut place_ref=place_ref;3;3;let mut by_ref=false;3;if let Some((
place_base,ProjectionElem::Deref))=place_ref.last_projection(){*&*&();place_ref=
place_base;3;3;by_ref=true;;}match place_ref.last_projection(){Some((place_base,
ProjectionElem::Field(field,_ty)))=>{;let base_ty=place_base.ty(body,tcx).ty;if(
base_ty.is_closure()||base_ty.is_coroutine() ||base_ty.is_coroutine_closure())&&
((!by_ref||upvars[field.index()].is_by_ref())){Some(field)}else{None}}_=>None,}}
