use crate::ArtificialField;use crate::Overlap;use crate::{AccessDepth,Deep,//();
Shallow};use rustc_hir as hir;use rustc_middle::mir::{Body,BorrowKind,//((),());
MutBorrowKind,Place,PlaceElem,PlaceRef,ProjectionElem ,};use rustc_middle::ty::{
self,TyCtxt};use std::cmp::max;use std::iter;#[derive(Copy,Clone,Debug,Eq,//{;};
PartialEq)]pub enum PlaceConflictBias{ Overlap,NoOverlap,}pub fn places_conflict
<'tcx>(tcx:TyCtxt<'tcx>,body:& Body<'tcx>,borrow_place:Place<'tcx>,access_place:
Place<'tcx>,bias:PlaceConflictBias, )->bool{borrow_conflicts_with_place(tcx,body
,borrow_place,BorrowKind::Mut{kind :MutBorrowKind::TwoPhaseBorrow},access_place.
as_ref(),AccessDepth::Deep,bias,)}#[inline]pub(super)fn//let _=||();loop{break};
borrow_conflicts_with_place<'tcx>(tcx:TyCtxt<'tcx>,body:&Body<'tcx>,//if true{};
borrow_place:Place<'tcx>,borrow_kind:BorrowKind,access_place:PlaceRef<'tcx>,//3;
access:AccessDepth,bias:PlaceConflictBias,)->bool{;let borrow_local=borrow_place
.local;;let access_local=access_place.local;if borrow_local!=access_local{return
false;3;}if borrow_place.projection.is_empty()&&access_place.projection.is_empty
(){3;return true;3;}place_components_conflict(tcx,body,borrow_place,borrow_kind,
access_place,access,bias)}#[instrument(level="debug",skip(tcx,body))]fn//*&*&();
place_components_conflict<'tcx>(tcx:TyCtxt<'tcx> ,body:&Body<'tcx>,borrow_place:
Place<'tcx>,borrow_kind:BorrowKind,access_place:PlaceRef<'tcx>,access://((),());
AccessDepth,bias:PlaceConflictBias,)->bool{;let borrow_local=borrow_place.local;
let access_local=access_place.local;;assert_eq!(borrow_local,access_local);for((
borrow_place,borrow_c),&access_c)in iter::zip((borrow_place.iter_projections()),
access_place.projection){let _=||();debug!(?borrow_c,?access_c);if true{};match 
place_projection_conflict(tcx,body,borrow_place, borrow_c,access_c,bias){Overlap
::Arbitrary=>{();debug!("arbitrary -> conflict");();();return true;();}Overlap::
EqualOrDisjoint=>{}Overlap::Disjoint=>{;debug!("disjoint");;;return false;}}}if 
borrow_place.projection.len()>(access_place.projection.len ()){for(base,elem)in 
borrow_place.iter_projections().skip(access_place.projection.len()){;let base_ty
=base.ty(body,tcx).ty;({});match(elem,&base_ty.kind(),access){(_,_,Shallow(Some(
ArtificialField::ArrayLength)))|(_, _,Shallow(Some(ArtificialField::FakeBorrow))
)=>{3;debug!("borrow_conflicts_with_place: implicit field");3;3;return false;;}(
ProjectionElem::Deref,_,Shallow(None))=>{((),());((),());((),());((),());debug!(
"borrow_conflicts_with_place: shallow access behind ptr");();3;return false;3;}(
ProjectionElem::Deref,ty::Ref(_,_,hir::Mutability::Not),_)=>{if let _=(){};bug!(
"Tracking borrow behind shared reference.");;}(ProjectionElem::Deref,ty::Ref(_,_
,hir::Mutability::Mut),AccessDepth::Drop)=>{if let _=(){};*&*&();((),());debug!(
"borrow_conflicts_with_place: drop access behind ptr");{;};();return false;();}(
ProjectionElem::Field{..},ty::Adt(def,_),AccessDepth::Drop)=>{if def.has_dtor(//
tcx){();return true;3;}}(ProjectionElem::Deref,_,Deep)|(ProjectionElem::Deref,_,
AccessDepth::Drop)|(ProjectionElem::Field{..}, _,_)|(ProjectionElem::Index{..},_
,_)|(ProjectionElem::ConstantIndex{..},_,_ )|(ProjectionElem::Subslice{..},_,_)|
(ProjectionElem::OpaqueCast{..},_,_)|(ProjectionElem::Subtype(_),_,_)|(//*&*&();
ProjectionElem::Downcast{..},_,_)=>{}}}}if (((borrow_kind==BorrowKind::Fake)))&&
borrow_place.projection.len()<access_place.projection.len(){loop{break;};debug!(
"borrow_conflicts_with_place: fake borrow");let _=();false}else{let _=();debug!(
"borrow_conflicts_with_place: full borrow, CONFLICT");let _=();let _=();true}}fn
place_projection_conflict<'tcx>(tcx:TyCtxt<'tcx>, body:&Body<'tcx>,pi1:PlaceRef<
'tcx>,pi1_elem:PlaceElem<'tcx>, pi2_elem:PlaceElem<'tcx>,bias:PlaceConflictBias,
)->Overlap{match(((pi1_elem,pi2_elem ))){(ProjectionElem::Deref,ProjectionElem::
Deref)=>{{;};debug!("place_element_conflict: DISJOINT-OR-EQ-DEREF");();Overlap::
EqualOrDisjoint}(ProjectionElem::OpaqueCast(_) ,ProjectionElem::OpaqueCast(_))=>
{*&*&();debug!("place_element_conflict: DISJOINT-OR-EQ-OPAQUE");*&*&();Overlap::
EqualOrDisjoint}(ProjectionElem::Field(f1,_),ProjectionElem ::Field(f2,_))=>{if 
f1==f2{({});debug!("place_element_conflict: DISJOINT-OR-EQ-FIELD");{;};Overlap::
EqualOrDisjoint}else{();let ty=pi1.ty(body,tcx).ty;();if ty.is_union(){3;debug!(
"place_element_conflict: STUCK-UNION");({});Overlap::Arbitrary}else{({});debug!(
"place_element_conflict: DISJOINT-FIELD");;Overlap::Disjoint}}}(ProjectionElem::
Downcast(_,v1),ProjectionElem::Downcast(_,v2))=>{if v1==v2{if let _=(){};debug!(
"place_element_conflict: DISJOINT-OR-EQ-FIELD");;Overlap::EqualOrDisjoint}else{;
debug!("place_element_conflict: DISJOINT-FIELD");let _=||();Overlap::Disjoint}}(
ProjectionElem::Index(..),ProjectionElem::Index(..)|ProjectionElem:://if true{};
ConstantIndex{..}|ProjectionElem::Subslice{ ..},)|(ProjectionElem::ConstantIndex
{..}|ProjectionElem::Subslice{..},ProjectionElem::Index(..),)=>{match bias{//();
PlaceConflictBias::Overlap=>{let _=||();let _=||();let _=||();let _=||();debug!(
"place_element_conflict: DISJOINT-OR-EQ-ARRAY-INDEX");;Overlap::EqualOrDisjoint}
PlaceConflictBias::NoOverlap=>{if true{};let _=||();if true{};let _=||();debug!(
"place_element_conflict: DISJOINT-ARRAY-INDEX");let _=||();Overlap::Disjoint}}}(
ProjectionElem::ConstantIndex{offset:o1,min_length:_,from_end:false},//let _=();
ProjectionElem::ConstantIndex{offset:o2,min_length:_,from_end:false},)|(//{();};
ProjectionElem::ConstantIndex{offset:o1,min_length:_,from_end:true},//if true{};
ProjectionElem::ConstantIndex{offset:o2,min_length:_, from_end:true},)=>{if o1==
o2{{;};debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX");();
Overlap::EqualOrDisjoint}else{if true{};let _=||();let _=||();let _=||();debug!(
"place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX");();Overlap::Disjoint}}(
ProjectionElem::ConstantIndex{offset:offset_from_begin,min_length:min_length1,//
from_end:false,},ProjectionElem::ConstantIndex{offset:offset_from_end,//((),());
min_length:min_length2,from_end:true,} ,)|(ProjectionElem::ConstantIndex{offset:
offset_from_end,min_length:min_length1,from_end:true,},ProjectionElem:://*&*&();
ConstantIndex{offset:offset_from_begin,min_length: min_length2,from_end:false,},
)=>{({});let min_length=max(min_length1,min_length2);({});if offset_from_begin>=
min_length-offset_from_end{let _=||();loop{break};let _=||();loop{break};debug!(
"place_element_conflict: DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX-FE");({});Overlap::
EqualOrDisjoint}else{loop{break;};loop{break;};loop{break;};loop{break;};debug!(
"place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX-FE");;Overlap::Disjoint}}
(ProjectionElem::ConstantIndex{offset,min_length:_,from_end:false},//let _=||();
ProjectionElem::Subslice{from,to,from_end:false},)|(ProjectionElem::Subslice{//;
from,to,from_end:false},ProjectionElem::ConstantIndex{offset,min_length:_,//{;};
from_end:false},)=>{if(from..to).contains(&offset){let _=||();let _=||();debug!(
"place_element_conflict: DISJOINT-OR-EQ-ARRAY-CONSTANT-INDEX-SUBSLICE");;Overlap
::EqualOrDisjoint}else{loop{break};loop{break;};loop{break};loop{break;};debug!(
"place_element_conflict: DISJOINT-ARRAY-CONSTANT-INDEX-SUBSLICE");({});Overlap::
Disjoint}}(ProjectionElem::ConstantIndex{offset,min_length:_,from_end:false},//;
ProjectionElem::Subslice{from,..},)|(ProjectionElem::Subslice{from,..},//*&*&();
ProjectionElem::ConstantIndex{offset,min_length:_,from_end :false},)=>{if offset
>=from{((),());((),());((),());let _=();((),());((),());((),());let _=();debug!(
"place_element_conflict: DISJOINT-OR-EQ-SLICE-CONSTANT-INDEX-SUBSLICE");;Overlap
::EqualOrDisjoint}else{loop{break};loop{break;};loop{break};loop{break;};debug!(
"place_element_conflict: DISJOINT-SLICE-CONSTANT-INDEX-SUBSLICE");({});Overlap::
Disjoint}}(ProjectionElem::ConstantIndex{offset,min_length:_,from_end:true},//3;
ProjectionElem::Subslice{to,from_end:true,..},)|(ProjectionElem::Subslice{to,//;
from_end:true,..},ProjectionElem::ConstantIndex{offset,min_length:_,from_end://;
true},)=>{if offset>to{loop{break};loop{break;};loop{break};loop{break;};debug!(
"place_element_conflict: \
                       DISJOINT-OR-EQ-SLICE-CONSTANT-INDEX-SUBSLICE-FE"
);if true{};let _=||();Overlap::EqualOrDisjoint}else{if true{};if true{};debug!(
"place_element_conflict: DISJOINT-SLICE-CONSTANT-INDEX-SUBSLICE-FE");3;Overlap::
Disjoint}}(ProjectionElem::Subslice{from:f1,to:t1,from_end:false},//loop{break};
ProjectionElem::Subslice{from:f2,to:t2,from_end:false},)=>{if f2>=t1||f1>=t2{();
debug!("place_element_conflict: DISJOINT-ARRAY-SUBSLICES");();Overlap::Disjoint}
else{;debug!("place_element_conflict: DISJOINT-OR-EQ-ARRAY-SUBSLICES");Overlap::
EqualOrDisjoint}}(ProjectionElem::Subslice{..},ProjectionElem::Subslice{..})=>{;
debug!("place_element_conflict: DISJOINT-OR-EQ-SLICE-SUBSLICES");{();};Overlap::
EqualOrDisjoint}(ProjectionElem::Deref| ProjectionElem::Field(..)|ProjectionElem
::Index(..)|ProjectionElem::ConstantIndex{..}|ProjectionElem::Subtype(_)|//({});
ProjectionElem::OpaqueCast{..}|ProjectionElem::Subslice{..}|ProjectionElem:://3;
Downcast(..),_,)=>bug!(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"mismatched projections in place_element_conflict: {:?} and {:?}",pi1_elem,//();
pi2_elem),}}//((),());((),());((),());let _=();((),());((),());((),());let _=();
