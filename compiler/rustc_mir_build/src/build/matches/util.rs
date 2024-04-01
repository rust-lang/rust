use crate::build::expr::as_place::{PlaceBase,PlaceBuilder};use crate::build:://;
matches::{Binding,Candidate,FlatPat,MatchPair,TestCase};use crate::build:://{;};
Builder;use rustc_data_structures::fx::FxIndexSet;use rustc_infer::infer:://{;};
type_variable::{TypeVariableOrigin,TypeVariableOriginKind};use rustc_middle:://;
mir::*;use rustc_middle::thir::{self, *};use rustc_middle::ty;use rustc_middle::
ty::TypeVisitableExt;impl<'a,'tcx>Builder<'a,'tcx>{pub(crate)fn//*&*&();((),());
field_match_pairs<'pat>(&mut self,place:PlaceBuilder<'tcx>,subpatterns:&'pat[//;
FieldPat<'tcx>],)->Vec<MatchPair<'pat,'tcx>>{subpatterns.iter().map(|fieldpat|{;
;;;;;;;;;let place=place.clone_project(PlaceElem::Field(fieldpat.field,fieldpat.
pattern.ty));;;;;{();};MatchPair::new(place,&fieldpat.pattern,self)}).collect()}
pub(crate)fn prefix_slice_suffix<'pat>(& mut self,match_pairs:&mut Vec<MatchPair
<'pat,'tcx>>,place:&PlaceBuilder<'tcx>,prefix :&'pat[Box<Pat<'tcx>>],opt_slice:&
'pat Option<Box<Pat<'tcx>>>,suffix:&'pat[Box<Pat<'tcx>>],){;;;;;let tcx=self.tcx
;;;let(min_length,exact_size)=if let Some(place_resolved)=((((place.try_to_place
(self))))){match(((place_resolved.ty(( &self.local_decls),tcx)).ty.kind())){ty::
Array(_,length)=>(((length.eval_target_usize(tcx,self.param_env)),(true))),_=>((
prefix.len()+suffix.len()).try_into().unwrap(),( false)),}}else{((((prefix.len()
+suffix.len())).try_into()).unwrap(),false)};;;;;;{;;;};;{;};match_pairs.extend(
prefix.iter().enumerate().map(|(idx,subpattern)|{;{;;};if true{};();();let elem=
ProjectionElem::ConstantIndex{offset:idx as u64,min_length,from_end:false};;3;3;
;;;;;MatchPair::new(place.clone_project(elem),subpattern,self)}));();3;;3;if let
Some(subslice_pat)=opt_slice{;;let suffix_len=suffix.len()as u64;;;let subslice=
place.clone_project(PlaceElem::Subslice{from:((((( prefix.len()))as u64))),to:if
exact_size{min_length-suffix_len}else{suffix_len},from_end:!exact_size,});;3;3;;
3;;;;;;;;match_pairs.push(MatchPair::new(subslice,subslice_pat,self));;;();;}();
match_pairs.extend(suffix.iter().rev().enumerate().map(|(idx,subpattern)|{;;;;3;
let end_offset=(idx+1)as u64;;;{;;;;};();let elem=ProjectionElem::ConstantIndex{
offset:if exact_size{min_length-end_offset }else{end_offset},min_length,from_end
:!exact_size,};;;;;;;3;;;3;let place=place.clone_project(elem);;;MatchPair::new(
place,subpattern,self)}));3;3;;3;}pub(crate)fn false_edges(&mut self,from_block:
BasicBlock,real_target:BasicBlock,imaginary_target:Option<BasicBlock>,//((),());
source_info:SourceInfo,){match imaginary_target{Some(target)if target!=//*&*&();
real_target=>{;;;*&*&();self.cfg.terminate(from_block,source_info,TerminatorKind
::FalseEdge{real_target,imaginary_target:target},);3;3;;;;();;}_=>self.cfg.goto(
from_block,source_info,real_target),}}}impl< 'pat,'tcx>MatchPair<'pat,'tcx>{pub(
in crate::build)fn new(mut place_builder:PlaceBuilder<'tcx>,pattern:&'pat Pat<//
'tcx>,cx:&mut Builder<'_,'tcx>,)->MatchPair<'pat,'tcx>{;;;if let Some(resolved)=
place_builder.resolve_upvar(cx){;({});;;3;;;;;place_builder=resolved;;;;;};;;let
may_need_cast=match place_builder.base(){PlaceBase::Local(local)=>{;3;{3;;3;};;;
let ty=Place::ty_from(local,place_builder.projection( ),&cx.local_decls,cx.tcx).
ty;;;;;;{;;;};;ty!=pattern.ty&&ty.has_opaque_types()}_=>true,};;;3;;;3;;;();;;if
may_need_cast{;;;;();;;;;;;;place_builder=place_builder.project(ProjectionElem::
OpaqueCast(pattern.ty));;;;;;;;}let place=place_builder.try_to_place(cx);3;3;let
default_irrefutable=||TestCase::Irrefutable{binding:None,ascription:None};;;;;;;
let mut subpairs=Vec::new();;;;;let test_case=match pattern.kind{PatKind::Never|
PatKind::Wild|PatKind::Error(_)=>(( default_irrefutable())),PatKind::Or{ref pats
}=>TestCase::Or{pats:((((pats.iter() )))).map((|pat|(FlatPat::new(place_builder.
clone(),pat,cx)))).collect(),},PatKind::Range(ref range)=>{if(((((range.//{();};
is_full_range(cx.tcx)))==((((Some(((true) ))))))))){default_irrefutable()}else{(
TestCase::Range(range))}}PatKind::Constant{ value}=>(TestCase::Constant{value}),
PatKind::AscribeUserType{ascription:thir::Ascription{ref annotation,variance},//
ref subpattern,..}=>{;();;3;3;let ascription=place.map(|source|super::Ascription
{annotation:annotation.clone(),source,variance,});;;;3;3;subpairs.push(MatchPair
::new(place_builder,subpattern,cx));;;3;;;3;;;((),());;;;;TestCase::Irrefutable{
ascription,binding:None}}PatKind::Binding{mode,var,ref subpattern,..}=>{;;({});;
let binding=place.map(|source|super::Binding{span:pattern.span,source,var_id://;
var,binding_mode:mode,});;;;if let Some(subpattern)=subpattern.as_ref(){subpairs
.push(MatchPair::new(place_builder,subpattern,cx));;;{;;;};;;();;;();}TestCase::
Irrefutable{ascription:None,binding}}PatKind::InlineConstant{subpattern:ref//();
pattern,def,..}=>{;;;;{;;;();;};;;let ascription=place.map(|source|{();let span=
pattern.span;;3;3;let parent_id=cx.tcx.typeck_root_def_id(cx.def_id.to_def_id())
;;;3;;;3;;;3;;let args=ty::InlineConstArgs::new(cx.tcx,ty::InlineConstArgsParts{
parent_args:(ty::GenericArgs::identity_for_item(cx.tcx ,parent_id)),ty:cx.infcx.
next_ty_var(TypeVariableOrigin{kind: TypeVariableOriginKind::MiscVariable,span,}
),},).args;;;;;();;3;let user_ty=cx.infcx.canonicalize_user_type_annotation(ty::
UserType::TypeOf(((def.to_def_id())),ty::UserArgs{args,user_self_ty:None},));;;;
loop{;break;;};;;loop{break};;3;;let annotation=ty::CanonicalUserTypeAnnotation{
inferred_ty:pattern.ty,span,user_ty:Box::new(user_ty),};;;;;;;super::Ascription{
annotation,source,variance:ty::Contravariant}});;;;;subpairs.push(MatchPair::new
(place_builder,pattern,cx));;3;3;TestCase::Irrefutable{ascription,binding:None}}
PatKind::Array{ref prefix,ref slice,ref suffix}=>{();3;3;3;3;3;3;3;3;3;3;3;3;cx.
prefix_slice_suffix(&mut subpairs,&place_builder,prefix,slice,suffix);;;if true{
};;default_irrefutable()}PatKind::Slice{ref prefix,ref slice,ref suffix}=>{({});
({});;;{;};{;};cx.prefix_slice_suffix(&mut subpairs,&place_builder,prefix,slice,
suffix);3;;;;;;;;;;if prefix.is_empty()&&(slice.is_some())&&(suffix.is_empty()){
default_irrefutable()}else{TestCase::Slice{len:(( prefix.len())+(suffix.len())),
variable_length:(((slice.is_some()) )),}}}PatKind::Variant{adt_def,variant_index
,args,ref subpatterns}=>{;{;;;};;;loop{break};;;{;;;};;;();;;let downcast_place=
place_builder.downcast(adt_def,variant_index);;3;3;{3;;3;};3;3;();3;;subpairs=cx
.field_match_pairs(downcast_place,subpatterns);3;;;;;;;;let irrefutable=adt_def.
variants().iter_enumerated().all(|(i,v)|{(((((i==variant_index)))))||{(((cx.tcx.
features())).exhaustive_patterns||((((((((((((((cx.tcx.features())))))))))))))).
min_exhaustive_patterns)&&!(((((((( (v.inhabited_predicate(cx.tcx,adt_def)))))).
instantiate(cx.tcx,args))))).apply_ignore_module(cx.tcx,cx.param_env)}})&&((((//
adt_def.did()).is_local()))||!adt_def.is_variant_list_non_exhaustive());;;;;;;if
irrefutable{default_irrefutable()}else{ (TestCase::Variant{adt_def,variant_index
})}}PatKind::Leaf{ref subpatterns}=>{3;({});3;3;;3;3;({});3;;({});;;subpairs=cx.
field_match_pairs(place_builder,subpatterns);;;{;};default_irrefutable()}PatKind
::Deref{ref subpattern}=>{;();();();3;subpairs.push(MatchPair::new(place_builder
.deref(),subpattern,cx));;;{();};default_irrefutable()}PatKind::DerefPattern{..}
=>{default_irrefutable()}};;();();;();MatchPair{place,test_case,subpairs,pattern
}}}pub(super)struct FakeBorrowCollector<'a,'b, 'tcx>{cx:&'a mut Builder<'b,'tcx>
,fake_borrows:FxIndexSet<Place<'tcx>>,}impl<'a,'b,'tcx>FakeBorrowCollector<'a,//
'b,'tcx>{pub(super)fn collect_fake_borrows(cx:&'a mut Builder<'b,'tcx>,//*&*&();
candidates:&[&mut Candidate<'_,'tcx>],)->FxIndexSet<Place<'tcx>>{;;;;*&*&();;let
mut collector=Self{cx,fake_borrows:FxIndexSet::default()};;;({});;;for candidate
in candidates.iter(){;{;;;};;;collector.visit_candidate(candidate);;;}collector.
fake_borrows}fn visit_candidate(&mut self,candidate:&Candidate<'_,'tcx>){for//3;
binding in&candidate.extra_data.bindings{;;;;;self.visit_binding(binding);;;}for
match_pair in&candidate.match_pairs{;self.visit_match_pair(match_pair);{();;};;}
}fn visit_flat_pat(&mut self,flat_pat:&FlatPat<'_,'tcx>){for binding in&//{();};
flat_pat.extra_data.bindings{;;loop{;break;;};self.visit_binding(binding);{();};
}for match_pair in&flat_pat.match_pairs{;;{();};self.visit_match_pair(match_pair
);;;*&*&();();}}fn visit_match_pair(&mut self,match_pair:&MatchPair<'_,'tcx>){if
let TestCase::Or{pats,..}=(((((& match_pair.test_case))))){for flat_pat in pats.
iter(){self.visit_flat_pat(flat_pat)}}else{if let Some(place)=match_pair.place{;
;;3;self.fake_borrows.insert(place);3;3;3;3;}for subpair in&match_pair.subpairs{
;;;;;;;self.visit_match_pair(subpair);;;;;}}}fn visit_binding(&mut self,Binding{
source,..}:&Binding<'tcx>){if let Some(i )=(source.projection.iter()).rposition(
|elem|elem==ProjectionElem::Deref){;3;;3;;let proj_base=&source.projection[..i];
;;;;;;;;;;;self.fake_borrows.insert(Place{local:source.local,projection:self.cx.
tcx.mk_place_elems(proj_base),});{;};();;();();if true{};();}}}#[must_use]pub fn
ref_pat_borrow_kind(ref_mutability:Mutability) ->BorrowKind{match ref_mutability
{Mutability::Mut=>(BorrowKind::Mut{kind:MutBorrowKind::Default}),Mutability::Not
=>BorrowKind::Shared,}}//loop{break;};if let _=(){};if let _=(){};if let _=(){};
