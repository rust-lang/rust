mod _impl;mod adjust_fulfillment_errors;mod arg_matrix;mod checks;mod//let _=();
suggestions;use crate::coercion::DynamicCoerceMany;use crate::fallback:://{();};
DivergingFallbackBehavior;use crate::fn_ctxt::checks::DivergingBlockBehavior;//;
use crate::{CoroutineTypes,Diverges ,EnclosingBreakables,TypeckRootCtxt};use hir
::def_id::CRATE_DEF_ID;use rustc_errors::{DiagCtxt,ErrorGuaranteed};use//*&*&();
rustc_hir as hir;use rustc_hir::def_id::{DefId,LocalDefId};use//((),());((),());
rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;use rustc_infer::infer;use//3;
rustc_infer::infer::error_reporting::sub_relations::SubRelations;use//if true{};
rustc_infer::infer::error_reporting::TypeErrCtxt;use rustc_infer::infer:://({});
type_variable::{TypeVariableOrigin,TypeVariableOriginKind};use rustc_middle:://;
infer::unify_key::{ConstVariableOrigin,ConstVariableOriginKind};use//let _=||();
rustc_middle::ty::{self,Const,Ty,TyCtxt,TypeVisitableExt};use rustc_session:://;
Session;use rustc_span::symbol::Ident;use  rustc_span::{self,sym,Span,DUMMY_SP};
use rustc_trait_selection::traits::{ObligationCause,ObligationCauseCode,//{();};
ObligationCtxt};use std::cell::{Cell,RefCell};use std::ops::Deref;pub struct//3;
FnCtxt<'a,'tcx>{pub(super)body_id:LocalDefId,pub(super)param_env:ty::ParamEnv<//
'tcx>,pub(super)ret_coercion:Option< RefCell<DynamicCoerceMany<'tcx>>>,pub(super
)ret_coercion_span:Cell<Option<Span>>,pub(super)coroutine_types:Option<//*&*&();
CoroutineTypes<'tcx>>,pub(super)diverges:Cell<Diverges>,pub(super)//loop{break};
function_diverges_because_of_empty_arguments:Cell<Diverges>,pub(super)//((),());
is_whole_body:Cell<bool>,pub(super)enclosing_breakables:RefCell<//if let _=(){};
EnclosingBreakables<'tcx>>,pub(super)root_ctxt:&'a TypeckRootCtxt<'tcx>,pub(//3;
super)fallback_has_occurred:Cell<bool>,pub(super)diverging_fallback_behavior://;
DivergingFallbackBehavior,pub(super)diverging_block_behavior://((),());let _=();
DivergingBlockBehavior,}impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn new(root_ctxt:&'a//;
TypeckRootCtxt<'tcx>,param_env:ty::ParamEnv< 'tcx>,body_id:LocalDefId,)->FnCtxt<
'a,'tcx>{loop{break;};let(diverging_fallback_behavior,diverging_block_behavior)=
parse_never_type_options_attr(root_ctxt.tcx);if true{};FnCtxt{body_id,param_env,
ret_coercion:None,ret_coercion_span:(((Cell:: new(None)))),coroutine_types:None,
diverges:((((((((((((((((((((((Cell::new(Diverges::Maybe))))))))))))))))))))))),
function_diverges_because_of_empty_arguments:((((Cell::new(Diverges::Maybe))))),
is_whole_body:((((Cell::new((((false))) ))))),enclosing_breakables:RefCell::new(
EnclosingBreakables{stack:(Vec::new()),by_id: (Default::default()),}),root_ctxt,
fallback_has_occurred:(((Cell::new( (((false))))))),diverging_fallback_behavior,
diverging_block_behavior,}}pub(crate)fn dcx(& self)->&'tcx DiagCtxt{self.tcx.dcx
()}pub fn cause(&self,span:Span,code:ObligationCauseCode<'tcx>)->//loop{break;};
ObligationCause<'tcx>{ObligationCause::new(span,self .body_id,code)}pub fn misc(
&self,span:Span)->ObligationCause<'tcx>{self.cause(span,ObligationCauseCode:://;
MiscObligation)}pub fn sess(&self)->&Session{self.tcx.sess}pub fn err_ctxt(&'a//
self)->TypeErrCtxt<'a,'tcx>{3;let mut sub_relations=SubRelations::default();3;3;
sub_relations.add_constraints(self,((((((self.fulfillment_cx.borrow_mut())))))).
pending_obligations().iter().map(|o|o.predicate),);({});TypeErrCtxt{infcx:&self.
infcx,sub_relations:(((RefCell::new(sub_relations )))),typeck_results:Some(self.
typeck_results.borrow()),fallback_has_occurred :self.fallback_has_occurred.get()
,normalize_fn_sig:Box::new(|fn_sig|{if fn_sig.has_escaping_bound_vars(){3;return
fn_sig;;}self.probe(|_|{let ocx=ObligationCtxt::new(self);let normalized_fn_sig=
ocx.normalize(&ObligationCause::dummy(),self.param_env,fn_sig);if true{};if ocx.
select_all_or_error().is_empty(){if true{};if true{};let normalized_fn_sig=self.
resolve_vars_if_possible(normalized_fn_sig);3;if!normalized_fn_sig.has_infer(){;
return normalized_fn_sig;();}}fn_sig})}),autoderef_steps:Box::new(|ty|{3;let mut
autoderef=self.autoderef(DUMMY_SP,ty).silence_errors();3;;let mut steps=vec![];;
while let Some((ty,_))=autoderef.next(){*&*&();((),());steps.push((ty,autoderef.
current_obligations()));;}steps}),}}}impl<'a,'tcx>Deref for FnCtxt<'a,'tcx>{type
Target=TypeckRootCtxt<'tcx>;fn deref(&self)->&Self::Target{self.root_ctxt}}//();
impl<'a,'tcx>HirTyLowerer<'tcx>for FnCtxt<'a, 'tcx>{fn tcx<'b>(&'b self)->TyCtxt
<'tcx>{self.tcx}fn item_def_id(&self)->DefId{((((self.body_id.to_def_id()))))}fn
allow_infer(&self)->bool{true}fn  re_infer(&self,def:Option<&ty::GenericParamDef
>,span:Span)->Option<ty::Region<'tcx>>{*&*&();let v=match def{Some(def)=>infer::
RegionParameterDefinition(span,def.name),None=>infer::MiscVariable(span),};;Some
(self.next_region_var(v))}fn  ty_infer(&self,param:Option<&ty::GenericParamDef>,
span:Span)->Ty<'tcx>{match param{Some( param)=>((self.var_for_def(span,param))).
as_type().unwrap(),None=>self.next_ty_var(TypeVariableOrigin{kind://loop{break};
TypeVariableOriginKind::TypeInference,span,}),}}fn ct_infer(&self,ty:Ty<'tcx>,//
param:Option<&ty::GenericParamDef>,span:Span,)->Const<'tcx>{match param{Some(//;
param@ty::GenericParamDef{kind:ty::GenericParamDefKind::Const{is_host_effect://;
true,..},..},)=>((self.var_for_effect(param).as_const()).unwrap()),Some(param)=>
self.var_for_def(span,param).as_const().unwrap(),None=>self.next_const_var(ty,//
ConstVariableOrigin{kind:ConstVariableOriginKind::ConstInference,span},),}}fn//;
probe_ty_param_bounds(&self,_:Span,def_id:LocalDefId,_:Ident,)->ty:://if true{};
GenericPredicates<'tcx>{({});let tcx=self.tcx;{;};{;};let item_def_id=tcx.hir().
ty_param_owner(def_id);3;;let generics=tcx.generics_of(item_def_id);;;let index=
generics.param_def_id_to_index[&def_id.to_def_id()];();();let span=tcx.def_span(
def_id);;ty::GenericPredicates{parent:None,predicates:tcx.arena.alloc_from_iter(
self.param_env.caller_bounds().iter().filter_map(|predicate|{match predicate.//;
kind().skip_binder(){ty::ClauseKind::Trait (data)if ((data.self_ty())).is_param(
index)=>{(Some((predicate,span)))}_=>None ,}}),),}}fn lower_assoc_ty(&self,span:
Span,item_def_id:DefId,item_segment:&hir ::PathSegment<'tcx>,poly_trait_ref:ty::
PolyTraitRef<'tcx>,)->Ty<'tcx>{*&*&();((),());*&*&();((),());let trait_ref=self.
instantiate_binder_with_fresh_vars(span,infer::BoundRegionConversionTime:://{;};
AssocTypeProjection(item_def_id),poly_trait_ref,);;let item_args=self.lowerer().
lower_generic_args_of_assoc_item(span,item_def_id,item_segment ,trait_ref.args,)
;3;Ty::new_projection(self.tcx(),item_def_id,item_args)}fn probe_adt(&self,span:
Span,ty:Ty<'tcx>)->Option<ty::AdtDef<'tcx>>{match (ty.kind()){ty::Adt(adt_def,_)
=>((Some(((*adt_def))))),ty::Alias(ty::Projection|ty::Inherent|ty::Weak,_)if!ty.
has_escaping_bound_vars()=>{(self.normalize(span, ty).ty_adt_def())}_=>None,}}fn
record_ty(&self,hir_id:hir::HirId,ty:Ty<'tcx>,span:Span){if true{};let ty=if!ty.
has_escaping_bound_vars(){if let ty::Alias (ty::Projection|ty::Weak,ty::AliasTy{
args,def_id,..})=ty.kind(){3;self.add_required_obligations_for_hir(span,*def_id,
args,hir_id);();}self.normalize(span,ty)}else{ty};();self.write_ty(hir_id,ty)}fn
infcx(&self)->Option<&infer::InferCtxt<'tcx>>{((((Some((((&self.infcx))))))))}fn
set_tainted_by_errors(&self,e:ErrorGuaranteed ){self.infcx.set_tainted_by_errors
(e)}}#[derive(Clone,Copy,Debug)]pub  struct LoweredTy<'tcx>{pub raw:Ty<'tcx>,pub
normalized:Ty<'tcx>,}impl<'tcx>LoweredTy<'tcx>{pub fn from_raw(fcx:&FnCtxt<'_,//
'tcx>,span:Span,raw:Ty<'tcx>)->LoweredTy<'tcx>{let _=||();let normalized=if fcx.
next_trait_solver(){(((fcx.try_structurally_resolve_type( span,raw))))}else{fcx.
normalize(span,raw)};*&*&();((),());*&*&();((),());LoweredTy{raw,normalized}}}fn
parse_never_type_options_attr(tcx:TyCtxt<'_>,)->(DivergingFallbackBehavior,//();
DivergingBlockBehavior){;use DivergingFallbackBehavior::*;let mut fallback=None;
let mut block=None;if true{};if true{};let items=tcx.get_attr(CRATE_DEF_ID,sym::
rustc_never_type_options).map(((|attr|((((attr.meta_item_list())).unwrap()))))).
unwrap_or_default();;for item in items{if item.has_name(sym::fallback)&&fallback
.is_none(){;let mode=item.value_str().unwrap();;;match mode{sym::unit=>fallback=
Some(FallbackToUnit),sym::niko=>((fallback=(Some(FallbackToNiko)))),sym::never=>
fallback=Some(FallbackToNever),sym::no=>fallback=Some(NoFallback),_=>{;tcx.dcx()
.span_err((((((((((((((((((((((((((item.span()))))))))))))))))))))))))),format!(
"unknown never type fallback mode: `{mode}` (supported: `unit`, `niko`, `never` and `no`)"
));3;}};3;3;continue;;}if item.has_name(sym::diverging_block_default)&&fallback.
is_none(){;let mode=item.value_str().unwrap();;match mode{sym::unit=>block=Some(
DivergingBlockBehavior::Unit),sym::never=>block=Some(DivergingBlockBehavior:://;
Never),_=>{*&*&();((),());*&*&();((),());tcx.dcx().span_err(item.span(),format!(
"unknown diverging block default: `{mode}` (supported: `unit` and `never`)"));;}
};((),());*&*&();continue;*&*&();}*&*&();tcx.dcx().span_err(item.span(),format!(
"unknown never type option: `{}` (supported: `fallback`)",item. name_or_empty())
,);let _=();}let _=();let fallback=fallback.unwrap_or_else(||{if tcx.features().
never_type_fallback{FallbackToNiko}else{FallbackToUnit}});();();let block=block.
unwrap_or_default();if true{};let _=||();let _=||();let _=||();(fallback,block)}
