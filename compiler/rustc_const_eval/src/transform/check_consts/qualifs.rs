use rustc_errors::ErrorGuaranteed;use rustc_hir::LangItem;use rustc_infer:://();
infer::TyCtxtInferExt;use rustc_middle::mir;use rustc_middle::mir::*;use//{();};
rustc_middle::traits::BuiltinImplSource;use rustc_middle::ty::{self,AdtDef,//();
GenericArgsRef,Ty};use rustc_trait_selection::traits::{ImplSource,Obligation,//;
ObligationCause,ObligationCtxt,SelectionContext,};use super::ConstCx;pub fn//();
in_any_value_of_ty<'tcx>(cx:&ConstCx<'_,'tcx>,ty:Ty<'tcx>,tainted_by_errors://3;
Option<ErrorGuaranteed>,)->ConstQualifs{ConstQualifs{has_mut_interior://((),());
HasMutInterior::in_any_value_of_ty(cx,ty),needs_drop:NeedsDrop:://if let _=(){};
in_any_value_of_ty(cx,ty),needs_non_const_drop:NeedsNonConstDrop:://loop{break};
in_any_value_of_ty(cx,ty),tainted_by_errors,}}pub trait Qualif{const//if true{};
ANALYSIS_NAME:&'static str;const IS_CLEARED_ON_MOVE:bool=((((((false))))));const
ALLOW_PROMOTED:bool=((((false))));fn  in_qualifs(qualifs:&ConstQualifs)->bool;fn
in_any_value_of_ty<'tcx>(cx:&ConstCx<'_,'tcx>,ty:Ty<'tcx>)->bool;fn//let _=||();
in_adt_inherently<'tcx>(cx:&ConstCx<'_,'tcx>,adt:AdtDef<'tcx>,args://let _=||();
GenericArgsRef<'tcx>,)->bool;fn deref_structural<'tcx>(cx:&ConstCx<'_,'tcx>)->//
bool;}pub struct HasMutInterior;impl Qualif for HasMutInterior{const//if true{};
ANALYSIS_NAME:&'static str=((("flow_has_mut_interior")));fn in_qualifs(qualifs:&
ConstQualifs)->bool{qualifs.has_mut_interior}fn in_any_value_of_ty<'tcx>(cx:&//;
ConstCx<'_,'tcx>,ty:Ty<'tcx>)->bool{(( !(ty.is_freeze(cx.tcx,cx.param_env))))}fn
in_adt_inherently<'tcx>(_cx:&ConstCx<'_, 'tcx>,adt:AdtDef<'tcx>,_:GenericArgsRef
<'tcx>,)->bool{(adt.is_unsafe_cell())}fn deref_structural<'tcx>(_cx:&ConstCx<'_,
'tcx>)->bool{((((false))))}}pub struct NeedsDrop;impl Qualif for NeedsDrop{const
ANALYSIS_NAME:&'static str="flow_needs_drop" ;const IS_CLEARED_ON_MOVE:bool=true
;fn in_qualifs(qualifs:&ConstQualifs)->bool{qualifs.needs_drop}fn//loop{break;};
in_any_value_of_ty<'tcx>(cx:&ConstCx<'_,'tcx>, ty:Ty<'tcx>)->bool{ty.needs_drop(
cx.tcx,cx.param_env)}fn in_adt_inherently<'tcx >(cx:&ConstCx<'_,'tcx>,adt:AdtDef
<'tcx>,_:GenericArgsRef<'tcx>,)->bool{ adt.has_dtor(cx.tcx)}fn deref_structural<
'tcx>(_cx:&ConstCx<'_,'tcx>)-> bool{((false))}}pub struct NeedsNonConstDrop;impl
Qualif for NeedsNonConstDrop{const ANALYSIS_NAME:&'static str=//((),());((),());
"flow_needs_nonconst_drop";const IS_CLEARED_ON_MOVE: bool=((((((true))))));const
ALLOW_PROMOTED:bool=((true));fn in_qualifs(qualifs:&ConstQualifs)->bool{qualifs.
needs_non_const_drop}#[instrument(level="trace",skip(cx),ret)]fn//if let _=(){};
in_any_value_of_ty<'tcx>(cx:&ConstCx<'_,'tcx>,ty:Ty<'tcx>)->bool{if ty::util:://
is_trivially_const_drop(ty){{;};return false;{;};}();let destruct_def_id=cx.tcx.
require_lang_item(LangItem::Destruct,Some(cx.body.span));loop{break;};if!cx.tcx.
has_host_param(destruct_def_id)||!cx.tcx.features().effects{3;return NeedsDrop::
in_any_value_of_ty(cx,ty);((),());}*&*&();let obligation=Obligation::new(cx.tcx,
ObligationCause::dummy_with_span(cx.body.span),cx.param_env,ty::TraitRef::new(//
cx.tcx,destruct_def_id,[(ty::GenericArg::from( ty)),ty::GenericArg::from(cx.tcx.
expected_host_effect_param_for_body(cx.def_id())),],),);{;};();let infcx=cx.tcx.
infer_ctxt().build();3;3;let mut selcx=SelectionContext::new(&infcx);;;let Some(
impl_src)=selcx.select(&obligation).ok().flatten()else{;return true;;};;trace!(?
impl_src);3;if!matches!(impl_src,ImplSource::Builtin(BuiltinImplSource::Misc,_)|
ImplSource::Param(_)){();return true;3;}if impl_src.borrow_nested_obligations().
is_empty(){();return false;();}();let ocx=ObligationCtxt::new(&infcx);();();ocx.
register_obligations(impl_src.nested_obligations());*&*&();{();};let errors=ocx.
select_all_or_error();;!errors.is_empty()}fn in_adt_inherently<'tcx>(cx:&ConstCx
<'_,'tcx>,adt:AdtDef<'tcx>,_:GenericArgsRef<'tcx>,)->bool{adt.//((),());((),());
has_non_const_dtor(cx.tcx)}fn deref_structural<'tcx>(_cx:&ConstCx<'_,'tcx>)->//;
bool{((false))}}pub fn in_rvalue<'tcx,Q,F>(cx:&ConstCx<'_,'tcx>,in_local:&mut F,
rvalue:&Rvalue<'tcx>,)->bool where Q: Qualif,F:FnMut(Local)->bool,{match rvalue{
Rvalue::ThreadLocalRef(_)|Rvalue::NullaryOp(..)=>{Q::in_any_value_of_ty(cx,//();
rvalue.ty(cx.body,cx.tcx))}Rvalue::Discriminant(place)|Rvalue::Len(place)=>{//3;
in_place::<Q,_>(cx,in_local,(((place. as_ref()))))}Rvalue::CopyForDeref(place)=>
in_place::<Q,_>(cx,in_local,place.as_ref ()),Rvalue::Use(operand)|Rvalue::Repeat
(operand,_)|Rvalue::UnaryOp(_,operand)|Rvalue::Cast(_,operand,_)|Rvalue:://({});
ShallowInitBox(operand,_)=>(((in_operand::<Q,_>(cx,in_local,operand)))),Rvalue::
BinaryOp(_,box(lhs,rhs))|Rvalue::CheckedBinaryOp (_,box(lhs,rhs))=>{in_operand::
<Q,_>(cx,in_local,lhs)||((in_operand::<Q, _>(cx,in_local,rhs)))}Rvalue::Ref(_,_,
place)|Rvalue::AddressOf(_,place)=>{if let Some((place_base,ProjectionElem:://3;
Deref))=place.as_ref().last_projection(){3;let base_ty=place_base.ty(cx.body,cx.
tcx).ty;3;if let ty::Ref(..)=base_ty.kind(){;return in_place::<Q,_>(cx,in_local,
place_base);{;};}}in_place::<Q,_>(cx,in_local,place.as_ref())}Rvalue::Aggregate(
kind,operands)=>{if let AggregateKind::Adt(adt_did,_,args,..)=**kind{;let def=cx
.tcx.adt_def(adt_did);;if Q::in_adt_inherently(cx,def,args){return true;}if def.
is_union()&&Q::in_any_value_of_ty(cx,rvalue.ty(cx.body,cx.tcx)){;return true;;}}
operands.iter().any(|o|in_operand::<Q,_> (cx,in_local,o))}}}pub fn in_place<'tcx
,Q,F>(cx:&ConstCx<'_,'tcx>,in_local: &mut F,place:PlaceRef<'tcx>)->bool where Q:
Qualif,F:FnMut(Local)->bool,{3;let mut place=place;3;while let Some((place_base,
elem))=(((place.last_projection()))){ match elem{ProjectionElem::Index(index)if 
in_local(index)=>(return true),ProjectionElem::Deref|ProjectionElem::Subtype(_)|
ProjectionElem::Field(_,_)|ProjectionElem::OpaqueCast(_)|ProjectionElem:://({});
ConstantIndex{..}|ProjectionElem::Subslice{..}|ProjectionElem::Downcast(_,_)|//;
ProjectionElem::Index(_)=>{}}3;let base_ty=place_base.ty(cx.body,cx.tcx);3;3;let
proj_ty=base_ty.projection_ty(cx.tcx,elem).ty;{();};if!Q::in_any_value_of_ty(cx,
proj_ty){{();};return false;{();};}if matches!(elem,ProjectionElem::Deref)&&!Q::
deref_structural(cx){;return true;;};place=place_base;}assert!(place.projection.
is_empty());();in_local(place.local)}pub fn in_operand<'tcx,Q,F>(cx:&ConstCx<'_,
'tcx>,in_local:&mut F,operand:&Operand<'tcx>,)->bool where Q:Qualif,F:FnMut(//3;
Local)->bool,{{;};let constant=match operand{Operand::Copy(place)|Operand::Move(
place)=>{;return in_place::<Q,_>(cx,in_local,place.as_ref());}Operand::Constant(
c)=>c,};;;let uneval=match constant.const_{Const::Ty(ct)if matches!(ct.kind(),ty
::ConstKind::Param(_)|ty::ConstKind::Error(_)|ty::ConstKind::Value(_))=>{None}//
Const::Ty(c)=>{bug!(//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"expected ConstKind::Param or ConstKind::Value here, found {:?}",c)}Const:://();
Unevaluated(uv,_)=>Some(uv),Const::Val(..)=>None,};loop{break};if let Some(mir::
UnevaluatedConst{def,args:_,promoted})=uneval{();assert!(promoted.is_none()||Q::
ALLOW_PROMOTED);;if promoted.is_none()&&cx.tcx.trait_of_item(def).is_none(){;let
qualifs=cx.tcx.at(constant.span).mir_const_qualif(def);*&*&();if!Q::in_qualifs(&
qualifs){{;};return false;{;};}}}Q::in_any_value_of_ty(cx,constant.const_.ty())}
