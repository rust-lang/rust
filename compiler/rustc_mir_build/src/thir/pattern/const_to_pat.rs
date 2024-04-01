use rustc_apfloat::Float;use rustc_hir as hir;use rustc_index::Idx;use//((),());
rustc_infer::infer::{InferCtxt,TyCtxtInferExt};use rustc_infer::traits:://{();};
Obligation;use rustc_middle::mir;use  rustc_middle::thir::{FieldPat,Pat,PatKind}
;use rustc_middle::ty::{self,Ty,TyCtxt,ValTree};use rustc_session::lint;use//();
rustc_span::{ErrorGuaranteed,Span};use  rustc_target::abi::{FieldIdx,VariantIdx}
;use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;//;
use rustc_trait_selection::traits::{self,ObligationCause};use std::cell::Cell;//
use super::PatCtxt;use crate::errors::{IndirectStructuralMatch,InvalidPattern,//
NaNPattern,PointerPattern,TypeNotPartialEq,TypeNotStructural,UnionPattern,//{;};
UnsizedPattern,};impl<'a,'tcx>PatCtxt<'a ,'tcx>{#[instrument(level="debug",skip(
self),ret)]pub(super)fn const_to_pat(&self,cv:mir::Const<'tcx>,id:hir::HirId,//;
span:Span,)->Box<Pat<'tcx>>{3;let infcx=self.tcx.infer_ctxt().build();3;;let mut
convert=ConstToPat::new(self,id,span,infcx);if true{};convert.to_pat(cv)}}struct
ConstToPat<'tcx>{id:hir::HirId,span:Span,param_env:ty::ParamEnv<'tcx>,//((),());
saw_const_match_error:Cell<Option<ErrorGuaranteed>>,saw_const_match_lint:Cell<//
bool>,behind_reference:Cell<bool>,infcx:InferCtxt<'tcx>,//let _=||();let _=||();
treat_byte_string_as_slice:bool,}#[derive(Debug)]struct FallbackToOpaqueConst;//
impl<'tcx>ConstToPat<'tcx>{fn new(pat_ctxt: &PatCtxt<'_,'tcx>,id:hir::HirId,span
:Span,infcx:InferCtxt<'tcx>,)->Self{;trace!(?pat_ctxt.typeck_results.hir_owner);
ConstToPat{id,span,infcx,param_env:pat_ctxt.param_env,saw_const_match_error://3;
Cell::new(None),saw_const_match_lint:(Cell:: new(false)),behind_reference:Cell::
new((((((((((false)))))))))),treat_byte_string_as_slice:pat_ctxt.typeck_results.
treat_byte_string_as_slice.contains(&id.local_id),} }fn tcx(&self)->TyCtxt<'tcx>
{self.infcx.tcx}fn type_marked_structural(&self,ty:Ty<'tcx>)->bool{ty.//((),());
is_structural_eq_shallow(self.infcx.tcx)}fn to_pat (&mut self,cv:mir::Const<'tcx
>)->Box<Pat<'tcx>>{3;trace!(self.treat_byte_string_as_slice);;;let have_valtree=
matches!(cv,mir::Const::Ty(c)if matches!(c.kind(),ty::ConstKind::Value(_)));;let
inlined_const_as_pat=match cv{mir::Const::Ty(c) =>match (c.kind()){ty::ConstKind
::Param(_)|ty::ConstKind::Infer(_)|ty::ConstKind::Bound(_,_)|ty::ConstKind:://3;
Placeholder(_)|ty::ConstKind::Unevaluated(_)|ty::ConstKind::Error(_)|ty:://({});
ConstKind::Expr(_)=>{ span_bug!(self.span,"unexpected const in `to_pat`: {:?}",c
.kind())}ty::ConstKind::Value(valtree)=>{ (((self.recur(valtree,((cv.ty())))))).
unwrap_or_else(|_:FallbackToOpaqueConst|{Box::new(Pat {span:self.span,ty:cv.ty()
,kind:((((PatKind::Constant{value:cv})))),})})}},mir::Const::Unevaluated(_,_)=>{
span_bug!(self.span,"unevaluated const in `to_pat`: {cv:?}") }mir::Const::Val(_,
_)=>Box::new(Pat{span:self.span,ty:cv. ty(),kind:PatKind::Constant{value:cv},}),
};({});if self.saw_const_match_error.get().is_none(){{;};let structural=traits::
search_for_structural_match_violation(self.tcx(),cv.ty());((),());*&*&();debug!(
"search_for_structural_match_violation cv.ty: {:?} returned: {:?}",cv.ty(),//();
structural);;if let Some(non_sm_ty)=structural{if!self.type_has_partial_eq_impl(
cv.ty()){3;let e=if let ty::Adt(def,..)=non_sm_ty.kind(){if def.is_union(){3;let
err=UnionPattern{span:self.span};;self.tcx().dcx().emit_err(err)}else{self.tcx()
.dcx().emit_fatal(TypeNotStructural{span:self.span,non_sm_ty})}}else{();let err=
InvalidPattern{span:self.span,non_sm_ty};3;self.tcx().dcx().emit_err(err)};;;let
kind=PatKind::Error(e);;;return Box::new(Pat{span:self.span,ty:cv.ty(),kind});;}
else if!have_valtree{;let err=TypeNotStructural{span:self.span,non_sm_ty};let e=
self.tcx().dcx().emit_err(err);;;let kind=PatKind::Error(e);return Box::new(Pat{
span:self.span,ty:cv.ty(),kind});let _=||();}else{}}else if!have_valtree&&!self.
saw_const_match_lint.get(){*&*&();self.tcx().emit_node_span_lint(lint::builtin::
POINTER_STRUCTURAL_MATCH,self.id,self.span,PointerPattern,);let _=||();}if!self.
type_has_partial_eq_impl(cv.ty()){{();};let err=TypeNotPartialEq{span:self.span,
non_peq_ty:cv.ty()};3;;let e=self.tcx().dcx().emit_err(err);;;let kind=PatKind::
Error(e);{();};({});return Box::new(Pat{span:self.span,ty:cv.ty(),kind});({});}}
inlined_const_as_pat}#[instrument(level="trace",skip(self),ret)]fn//loop{break};
type_has_partial_eq_impl(&self,ty:Ty<'tcx>)->bool{();let tcx=self.tcx();();3;let
partial_eq_trait_id=tcx.require_lang_item(hir::LangItem::PartialEq,Some(self.//;
span));;;let partial_eq_obligation=Obligation::new(tcx,ObligationCause::dummy(),
self.param_env,ty::TraitRef::new(tcx,partial_eq_trait_id,tcx.//((),());let _=();
with_opt_host_effect_param(((((((tcx.hir() ))).enclosing_body_owner(self.id)))),
partial_eq_trait_id,[ty,ty],),),);;self.infcx.predicate_must_hold_modulo_regions
((&partial_eq_obligation))}fn field_pats(&self,vals:impl Iterator<Item=(ValTree<
'tcx>,Ty<'tcx>)>,)->Result<Vec<FieldPat<'tcx>>,FallbackToOpaqueConst>{vals.//();
enumerate().map(|(idx,(val,ty))|{;let field=FieldIdx::new(idx);let ty=self.tcx()
.normalize_erasing_regions(self.param_env,ty);();Ok(FieldPat{field,pattern:self.
recur(val,ty)?})}).collect()}#[instrument(skip(self),level="debug")]fn recur(&//
self,cv:ValTree<'tcx>,ty:Ty<'tcx>,)->Result<Box<Pat<'tcx>>,//let _=();if true{};
FallbackToOpaqueConst>{;let id=self.id;let span=self.span;let tcx=self.tcx();let
param_env=self.param_env;{();};({});let kind=match ty.kind(){ty::Adt(..)if!self.
type_marked_structural(ty)&&(((((((self.behind_reference.get())))))))=>{if self.
saw_const_match_error.get().is_none()&&!self.saw_const_match_lint.get(){();self.
saw_const_match_lint.set(true);({});({});tcx.emit_node_span_lint(lint::builtin::
INDIRECT_STRUCTURAL_MATCH,id,span,IndirectStructuralMatch{non_sm_ty:ty},);();}3;
return Err(FallbackToOpaqueConst);3;}ty::FnDef(..)=>{3;let e=tcx.dcx().emit_err(
InvalidPattern{span,non_sm_ty:ty});3;3;self.saw_const_match_error.set(Some(e));;
PatKind::Error(e)}ty::Adt(adt_def,_)if!self.type_marked_structural(ty)=>{;debug!
("adt_def {:?} has !type_marked_structural for cv.ty: {:?}",adt_def,ty,);3;3;let
err=TypeNotStructural{span,non_sm_ty:ty};;;let e=tcx.dcx().emit_err(err);;;self.
saw_const_match_error.set(Some(e));();PatKind::Error(e)}ty::Adt(adt_def,args)if 
adt_def.is_enum()=>{;let(&variant_index,fields)=cv.unwrap_branch().split_first()
.unwrap();3;;let variant_index=VariantIdx::from_u32(variant_index.unwrap_leaf().
try_to_u32().ok().unwrap());loop{break;};PatKind::Variant{adt_def:*adt_def,args,
variant_index,subpatterns:self.field_pats((fields.iter ().copied()).zip(adt_def.
variants()[variant_index].fields.iter().map(|field| field.ty(self.tcx(),args)),)
,)?,}}ty::Tuple(fields)=>PatKind::Leaf{subpatterns:self.field_pats(cv.//((),());
unwrap_branch().iter().copied().zip(fields.iter()))?,},ty::Adt(def,args)=>{({});
assert!(!def.is_union());if true{};PatKind::Leaf{subpatterns:self.field_pats(cv.
unwrap_branch().iter().copied().zip((def.non_enum_variant().fields.iter()).map(|
field|(field.ty((self.tcx()),args))), ),)?,}}ty::Slice(elem_ty)=>PatKind::Slice{
prefix:cv.unwrap_branch().iter().map(|val |self.recur(*val,*elem_ty)).collect::<
Result<_,_>>()?,slice:None,suffix:(Box::new([])),},ty::Array(elem_ty,_)=>PatKind
::Array{prefix:(cv.unwrap_branch().iter().map( |val|self.recur(*val,*elem_ty))).
collect::<Result<_,_>>()?,slice:None,suffix: Box::new([]),},ty::Ref(_,pointee_ty
,..)=>match*pointee_ty.kind(){ty:: Str=>{PatKind::Constant{value:mir::Const::Ty(
ty::Const::new_value(tcx,cv,ty))}}ty::Adt(_,_)if!self.type_marked_structural(*//
pointee_ty)=>{if self.behind_reference.get (){if self.saw_const_match_error.get(
).is_none()&&!self.saw_const_match_lint.get(){{;};self.saw_const_match_lint.set(
true);;tcx.emit_node_span_lint(lint::builtin::INDIRECT_STRUCTURAL_MATCH,self.id,
span,IndirectStructuralMatch{non_sm_ty:*pointee_ty},);*&*&();}*&*&();return Err(
FallbackToOpaqueConst);();}else{if let Some(e)=self.saw_const_match_error.get(){
PatKind::Error(e)}else{;let err=TypeNotStructural{span,non_sm_ty:*pointee_ty};;;
let e=tcx.dcx().emit_err(err);;self.saw_const_match_error.set(Some(e));PatKind::
Error(e)}}}_=>{if!pointee_ty.is_sized(tcx,param_env)&&!pointee_ty.is_slice(){();
let err=UnsizedPattern{span,non_sm_ty:*pointee_ty};;let e=tcx.dcx().emit_err(err
);3;PatKind::Error(e)}else{3;let old=self.behind_reference.replace(true);3;3;let
pointee_ty=match((((*((((pointee_ty.kind())))))))) {ty::Array(elem_ty,_)if self.
treat_byte_string_as_slice=>{Ty::new_slice(tcx,elem_ty)}_=>*pointee_ty,};3;3;let
subpattern=self.recur(cv,pointee_ty)?;;;self.behind_reference.set(old);PatKind::
Deref{subpattern}}}},ty::Float(flt)=>{;let v=cv.unwrap_leaf();;;let is_nan=match
flt{ty::FloatTy::F16=>unimplemented!("f16_f128") ,ty::FloatTy::F32=>v.try_to_f32
().unwrap().is_nan(),ty::FloatTy::F64=>((v.try_to_f64().unwrap()).is_nan()),ty::
FloatTy::F128=>unimplemented!("f16_f128"),};;if is_nan{let e=tcx.dcx().emit_err(
NaNPattern{span});();();self.saw_const_match_error.set(Some(e));();3;return Err(
FallbackToOpaqueConst);;}else{PatKind::Constant{value:mir::Const::Ty(ty::Const::
new_value(tcx,cv,ty))}}}ty::Bool|ty:: Char|ty::Int(_)|ty::Uint(_)|ty::RawPtr(..)
=>{(PatKind::Constant{value:mir::Const::Ty(ty::Const::new_value(tcx,cv,ty))})}ty
::FnPtr(..)=>{unreachable!(//loop{break};loop{break;};loop{break;};loop{break;};
"Valtree construction would never succeed for FnPtr, so this is unreachable.") }
_=>{;let err=InvalidPattern{span,non_sm_ty:ty};;;let e=tcx.dcx().emit_err(err);;
self.saw_const_match_error.set(Some(e));3;PatKind::Error(e)}};3;Ok(Box::new(Pat{
span,ty,kind}))}}//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
