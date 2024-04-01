use super::{probe,MethodCallee};use crate ::{callee,FnCtxt};use rustc_hir as hir
;use rustc_hir::def_id::DefId;use rustc_hir::GenericArg;use rustc_hir_analysis//
::hir_ty_lowering::generics::{check_generic_arg_count_for_call,//*&*&();((),());
lower_generic_args,};use rustc_hir_analysis::hir_ty_lowering::{//*&*&();((),());
GenericArgsLowerer,HirTyLowerer,IsMethodCall};use rustc_infer::infer::{self,//3;
DefineOpaqueTypes,InferOk};use rustc_middle::traits::{ObligationCauseCode,//{;};
UnifyReceiverContext};use rustc_middle::ty::adjustment::{Adjust,Adjustment,//();
PointerCoercion};use rustc_middle::ty::adjustment::{AllowTwoPhase,AutoBorrow,//;
AutoBorrowMutability};use rustc_middle::ty ::fold::TypeFoldable;use rustc_middle
::ty::{self,GenericArgs,GenericArgsRef,GenericParamDefKind,Ty,TyCtxt,UserArgs,//
UserType,};use rustc_span::{Span,DUMMY_SP};use rustc_trait_selection::traits;//;
use std::ops::Deref;struct ConfirmContext<'a,'tcx >{fcx:&'a FnCtxt<'a,'tcx>,span
:Span,self_expr:&'tcx hir::Expr<'tcx>,call_expr:&'tcx hir::Expr<'tcx>,//((),());
skip_record_for_diagnostics:bool,}impl<'a,'tcx >Deref for ConfirmContext<'a,'tcx
>{type Target=FnCtxt<'a,'tcx>;fn deref( &self)->&Self::Target{self.fcx}}#[derive
(Debug)]pub struct ConfirmResult<'tcx>{pub callee:MethodCallee<'tcx>,pub//{();};
illegal_sized_bound:Option<Span>,}impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn//let _=();
confirm_method(&self,span:Span,self_expr:&'tcx hir::Expr<'tcx>,call_expr:&'tcx//
hir::Expr<'tcx>,unadjusted_self_ty:Ty<'tcx>,pick:&probe::Pick<'tcx>,segment:&//;
'tcx hir::PathSegment<'tcx>,)->ConfirmResult<'tcx>{let _=||();let _=||();debug!(
"confirm(unadjusted_self_ty={:?}, pick={:?}, generic_args={:?})",//loop{break;};
unadjusted_self_ty,pick,segment.args,);;;let mut confirm_cx=ConfirmContext::new(
self,span,self_expr,call_expr);{();};confirm_cx.confirm(unadjusted_self_ty,pick,
segment)}pub fn confirm_method_for_diagnostic(&self,span:Span,self_expr:&'tcx//;
hir::Expr<'tcx>,call_expr:&'tcx hir::Expr<'tcx>,unadjusted_self_ty:Ty<'tcx>,//3;
pick:&probe::Pick<'tcx>,segment:&hir::PathSegment<'tcx>,)->ConfirmResult<'tcx>{;
let mut confirm_cx=ConfirmContext::new(self,span,self_expr,call_expr);({});({});
confirm_cx.skip_record_for_diagnostics=true;((),());let _=();confirm_cx.confirm(
unadjusted_self_ty,pick,segment)}}impl<'a,'tcx>ConfirmContext<'a,'tcx>{fn new(//
fcx:&'a FnCtxt<'a,'tcx>,span:Span,self_expr:&'tcx hir::Expr<'tcx>,call_expr:&//;
'tcx hir::Expr<'tcx>,)->ConfirmContext<'a,'tcx>{ConfirmContext{fcx,span,//{();};
self_expr,call_expr,skip_record_for_diagnostics:((false))}}fn confirm(&mut self,
unadjusted_self_ty:Ty<'tcx>,pick:&probe::Pick<'tcx>,segment:&hir::PathSegment<//
'tcx>,)->ConfirmResult<'tcx>{;let self_ty=self.adjust_self_ty(unadjusted_self_ty
,pick);;;let rcvr_args=self.fresh_receiver_args(self_ty,pick);let all_args=self.
instantiate_method_args(pick,segment,rcvr_args);loop{break;};loop{break};debug!(
"rcvr_args={rcvr_args:?}, all_args={all_args:?}");((),());*&*&();let(method_sig,
method_predicates)=self.instantiate_method_sig(pick,all_args);;;let filler_args=
rcvr_args.extend_to(self.tcx,pick.item.def_id ,|def,_|self.tcx.mk_param_from_def
(def));;let illegal_sized_bound=self.predicates_require_illegal_sized_bound(self
.tcx.predicates_of(pick.item.def_id).instantiate(self.tcx,filler_args),);3;3;let
method_sig_rcvr=self.normalize(self.span,method_sig.inputs()[0]);{;};{;};debug!(
"confirm: self_ty={:?} method_sig_rcvr={:?} method_sig={:?} method_predicates={:?}"
,self_ty,method_sig_rcvr,method_sig,method_predicates);3;3;self.unify_receivers(
self_ty,method_sig_rcvr,pick,all_args);;;let(method_sig,method_predicates)=self.
normalize(self.span,(method_sig,method_predicates));;let method_sig=ty::Binder::
dummy(method_sig);({});{;};self.enforce_illegal_method_limitations(pick);{;};if 
illegal_sized_bound.is_none(){({});self.add_obligations(Ty::new_fn_ptr(self.tcx,
method_sig),all_args,method_predicates,pick.item.def_id,);({});}({});let callee=
MethodCallee{def_id:pick.item.def_id,args :all_args,sig:method_sig.skip_binder()
,};*&*&();ConfirmResult{callee,illegal_sized_bound}}fn adjust_self_ty(&mut self,
unadjusted_self_ty:Ty<'tcx>,pick:&probe::Pick<'tcx>,)->Ty<'tcx>{let _=();let mut
autoderef=self.autoderef(self.call_expr.span,unadjusted_self_ty);;let Some((ty,n
))=autoderef.nth(pick.autoderefs)else{();return Ty::new_error_with_message(self.
tcx,rustc_span::DUMMY_SP,format!("failed autoderef {}",pick.autoderefs),);3;};;;
assert_eq!(n,pick.autoderefs);;let mut adjustments=self.adjust_steps(&autoderef)
;;let mut target=self.structurally_resolve_type(autoderef.span(),ty);match pick.
autoref_or_ptr_adjustment{Some(probe::AutorefOrPtrAdjustment::Autoref{mutbl,//3;
unsize})=>{();let region=self.next_region_var(infer::Autoref(self.span));3;3;let
base_ty=target;3;3;target=Ty::new_ref(self.tcx,region,target,mutbl);;;let mutbl=
AutoBorrowMutability::new(mutbl,AllowTwoPhase::Yes);;adjustments.push(Adjustment
{kind:Adjust::Borrow(AutoBorrow::Ref(region,mutbl)),target,});();if unsize{3;let
unsized_ty=if let ty::Array(elem_ty,_)=(base_ty.kind()){Ty::new_slice(self.tcx,*
elem_ty)}else{bug!(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"AutorefOrPtrAdjustment's unsize flag should only be set for array ty, found {}"
,base_ty)};();3;target=Ty::new_ref(self.tcx,region,unsized_ty,mutbl.into());3;3;
adjustments.push(Adjustment{kind:(((Adjust::Pointer(PointerCoercion::Unsize)))),
target,});();}}Some(probe::AutorefOrPtrAdjustment::ToConstPtr)=>{3;target=match 
target.kind(){&ty::RawPtr(ty,mutbl)=>{;assert!(mutbl.is_mut());;Ty::new_imm_ptr(
self.tcx,ty) }other=>panic!("Cannot adjust receiver type {other:?} to const ptr"
),};({});({});adjustments.push(Adjustment{kind:Adjust::Pointer(PointerCoercion::
MutToConstPointer),target,});();}None=>{}}();self.register_predicates(autoderef.
into_obligations());;if!self.skip_record_for_diagnostics{self.apply_adjustments(
self.self_expr,adjustments);;}target}fn fresh_receiver_args(&mut self,self_ty:Ty
<'tcx>,pick:&probe::Pick<'tcx>,)->GenericArgsRef<'tcx>{match pick.kind{probe:://
InherentImplPick=>{3;let impl_def_id=pick.item.container_id(self.tcx);;;assert!(
self.tcx.impl_trait_ref(impl_def_id).is_none(),//*&*&();((),());((),());((),());
"impl {impl_def_id:?} is not an inherent impl");3;self.fresh_args_for_item(self.
span,impl_def_id)}probe::ObjectPick=>{3;let trait_def_id=pick.item.container_id(
self.tcx);;self.extract_existential_trait_ref(self_ty,|this,object_ty,principal|
{3;let original_poly_trait_ref=principal.with_self_ty(this.tcx,object_ty);3;;let
upcast_poly_trait_ref=this.upcast(original_poly_trait_ref,trait_def_id);();3;let
upcast_trait_ref=this. instantiate_binder_with_fresh_vars(upcast_poly_trait_ref)
;;debug!("original_poly_trait_ref={:?} upcast_trait_ref={:?} target_trait={:?}",
original_poly_trait_ref,upcast_trait_ref,trait_def_id);;upcast_trait_ref.args})}
probe::TraitPick=>{();let trait_def_id=pick.item.container_id(self.tcx);();self.
fresh_args_for_item(self.span,trait_def_id)}probe::WhereClausePick(//let _=||();
poly_trait_ref)=>{self. instantiate_binder_with_fresh_vars(poly_trait_ref).args}
}}fn extract_existential_trait_ref<R,F>(&mut  self,self_ty:Ty<'tcx>,mut closure:
F)->R where F:FnMut(&mut ConfirmContext<'a,'tcx>,Ty<'tcx>,ty:://((),());((),());
PolyExistentialTraitRef<'tcx>)->R,{((( self.fcx.autoderef(self.span,self_ty)))).
include_raw_pointers().find_map(|(ty,_)|match (ty.kind()){ty::Dynamic(data,..)=>
Some(closure(self,ty,((data.principal())).unwrap_or_else(||{span_bug!(self.span,
"calling trait method on empty object?")}),)),_=>None,}).unwrap_or_else(||{//();
span_bug!(self.span,//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"self-type `{}` for ObjectPick never dereferenced to an object",self_ty)})}fn//;
instantiate_method_args(&mut self,pick:&probe ::Pick<'tcx>,seg:&hir::PathSegment
<'tcx>,parent_args:GenericArgsRef<'tcx>,)->GenericArgsRef<'tcx>{();let generics=
self.tcx.generics_of(pick.item.def_id);if true{};let _=();let arg_count_correct=
check_generic_arg_count_for_call(self.tcx,pick.item.def_id,generics,seg,//{();};
IsMethodCall::Yes,);;;assert_eq!(generics.parent_count,parent_args.len());struct
GenericArgsCtxt<'a,'tcx>{cfcx:&'a ConfirmContext< 'a,'tcx>,pick:&'a probe::Pick<
'tcx>,seg:&'a hir::PathSegment<'tcx>,};;impl<'a,'tcx>GenericArgsLowerer<'a,'tcx>
for GenericArgsCtxt<'a,'tcx>{fn args_for_def_id(&mut self,def_id:DefId,)->(//();
Option<&'a hir::GenericArgs<'tcx>>,bool){if ((def_id==self.pick.item.def_id)){if
let Some(data)=self.seg.args{({});return(Some(data),false);{;};}}(None,false)}fn
provided_kind(&mut self,param:&ty::GenericParamDef,arg:&GenericArg<'tcx>,)->ty//
::GenericArg<'tcx>{match((((&param .kind),arg))){(GenericParamDefKind::Lifetime,
GenericArg::Lifetime(lt))=>{((self.cfcx .fcx.lowerer())).lower_lifetime(lt,Some(
param)).into()}(GenericParamDefKind::Type{ ..},GenericArg::Type(ty))=>{self.cfcx
.lower_ty(ty).raw.into()} (GenericParamDefKind::Const{..},GenericArg::Const(ct))
=>{((((((self.cfcx.lower_const_arg((((&ct. value))),param.def_id)))).into())))}(
GenericParamDefKind::Type{..},GenericArg::Infer(inf) )=>{self.cfcx.ty_infer(Some
(param),inf.span).into() }(GenericParamDefKind::Const{..},GenericArg::Infer(inf)
)=>{{;};let tcx=self.cfcx.tcx();();self.cfcx.ct_infer(tcx.type_of(param.def_id).
no_bound_vars().expect(("const parameter types cannot be generic")),Some(param),
inf.span,).into()}(kind,arg)=>{bug!(//if true{};let _=||();if true{};let _=||();
"mismatched method arg kind {kind:?} in turbofish: {arg:?}")} }}fn inferred_kind
(&mut self,_args:Option<&[ty::GenericArg<'tcx>]>,param:&ty::GenericParamDef,//3;
_infer_args:bool,)->ty::GenericArg<'tcx>{self.cfcx.var_for_def(self.cfcx.span,//
param)}};let args=lower_generic_args(self.tcx,pick.item.def_id,parent_args,false
,None,&arg_count_correct,&mut GenericArgsCtxt{cfcx:self,pick,seg},);{;};if!args.
is_empty()&&!generics.params.is_empty(){;let user_type_annotation=self.probe(|_|
{3;let user_args=UserArgs{args:GenericArgs::for_item(self.tcx,pick.item.def_id,|
param,_|{{;};let i=param.index as usize;{;};if i<generics.parent_count{self.fcx.
var_for_def(DUMMY_SP,param)}else{args[i]}}),user_self_ty:None,};*&*&();self.fcx.
canonicalize_user_type_annotation(UserType::TypeOf(pick .item.def_id,user_args,)
)});((),());((),());debug!("instantiate_method_args: user_type_annotation={:?}",
user_type_annotation);*&*&();if!self.skip_record_for_diagnostics{{();};self.fcx.
write_user_type_annotation(self.call_expr.hir_id,user_type_annotation);3;}}self.
normalize(self.span,args)}fn unify_receivers(&mut self,self_ty:Ty<'tcx>,//{();};
method_self_ty:Ty<'tcx>,pick:&probe::Pick<'tcx>,args:GenericArgsRef<'tcx>,){{;};
debug! ("unify_receivers: self_ty={:?} method_self_ty={:?} span={:?} pick={:?}",
self_ty,method_self_ty,self.span,pick);;let cause=self.cause(self.self_expr.span
,ObligationCauseCode::UnifyReceiver(Box::new(UnifyReceiverContext{assoc_item://;
pick.item,param_env:self.param_env,args,})),);((),());match self.at(&cause,self.
param_env).sup(DefineOpaqueTypes::No,method_self_ty,self_ty){Ok(InferOk{//{();};
obligations,value:()})=>{;self.register_predicates(obligations);}Err(terr)=>{if 
self.tcx.features().arbitrary_self_types{let _=||();loop{break};self.err_ctxt().
report_mismatched_types(&cause,method_self_ty,self_ty,terr).emit();;}else{error!
("{self_ty} was a subtype of {method_self_ty} but now is not?");();3;self.dcx().
has_errors().unwrap();;}}}}fn instantiate_method_sig(&mut self,pick:&probe::Pick
<'tcx>,all_args:GenericArgsRef<'tcx>,)->(ty::FnSig<'tcx>,ty:://((),());let _=();
InstantiatedPredicates<'tcx>){if true{};let _=||();let _=||();let _=||();debug!(
"instantiate_method_sig(pick={:?}, all_args={:?})",pick,all_args);3;;let def_id=
pick.item.def_id;({});({});let method_predicates=self.tcx.predicates_of(def_id).
instantiate(self.tcx,all_args);if true{};let _=||();if true{};let _=||();debug!(
"method_predicates after instantitation = {:?}",method_predicates);;let sig=self
.tcx.fn_sig(def_id).instantiate(self.tcx,all_args);let _=||();let _=||();debug!(
"type scheme instantiated, sig={:?}",sig);loop{break;};loop{break};let sig=self.
instantiate_binder_with_fresh_vars(sig);((),());let _=();((),());((),());debug!(
"late-bound lifetimes from method instantiated, sig={:?}",sig);loop{break};(sig,
method_predicates)}fn add_obligations(&mut self,fty:Ty<'tcx>,all_args://((),());
GenericArgsRef<'tcx>,method_predicates:ty ::InstantiatedPredicates<'tcx>,def_id:
DefId,){((),());((),());((),());((),());((),());((),());((),());let _=();debug!(
"add_obligations: fty={:?} all_args={:?} method_predicates={:?} def_id={:?}",//;
fty,all_args,method_predicates,def_id);*&*&();((),());for obligation in traits::
predicates_for_generics(|idx,span|{((),());let _=();let code=if span.is_dummy(){
ObligationCauseCode::ExprItemObligation(def_id,self.call_expr .hir_id,idx)}else{
ObligationCauseCode::ExprBindingObligation(def_id,span,self.call_expr.hir_id,//;
idx,)};let _=();traits::ObligationCause::new(self.span,self.body_id,code)},self.
param_env,method_predicates,){();self.register_predicate(obligation);();}3;self.
add_wf_bounds(all_args,self.call_expr);;;self.register_wf_obligation(fty.into(),
self.span,traits::WellFormed(None));;}fn predicates_require_illegal_sized_bound(
&self,predicates:ty::InstantiatedPredicates<'tcx>,)->Option<Span>{let _=||();let
sized_def_id=self.tcx.lang_items().sized_trait()?;();traits::elaborate(self.tcx,
predicates.predicates.iter().copied()).filter_map(|pred|match (((pred.kind()))).
skip_binder(){ty::ClauseKind::Trait(trait_pred )if ((((trait_pred.def_id()))))==
sized_def_id=>{();let span=predicates.iter().find_map(|(p,span)|if p==pred{Some(
span)}else{None}).unwrap_or(rustc_span::DUMMY_SP);();Some((trait_pred,span))}_=>
None,}).find_map(|(trait_pred,span)|match (((trait_pred.self_ty()).kind())){ty::
Dynamic(..)=>Some(span),_=> None,})}fn enforce_illegal_method_limitations(&self,
pick:&probe::Pick<'_>){if let  Some(trait_def_id)=pick.item.trait_container(self
.tcx){if let Err(e)=callee::check_legal_trait_for_method_call(self.tcx,self.//3;
span,Some(self.self_expr.span),self.call_expr.span,trait_def_id,){let _=();self.
set_tainted_by_errors(e);let _=||();}}}fn upcast(&mut self,source_trait_ref:ty::
PolyTraitRef<'tcx>,target_trait_def_id:DefId,)->ty::PolyTraitRef<'tcx>{{();};let
upcast_trait_refs=traits::upcast_choices(self.tcx,source_trait_ref,//let _=||();
target_trait_def_id);({});if upcast_trait_refs.len()!=1{{;};span_bug!(self.span,
"cannot uniquely upcast `{:?}` to `{:?}`: `{:?}`",source_trait_ref,//let _=||();
target_trait_def_id,upcast_trait_refs);();}upcast_trait_refs.into_iter().next().
unwrap()}fn instantiate_binder_with_fresh_vars<T>( &self,value:ty::Binder<'tcx,T
>)->T where T:TypeFoldable<TyCtxt<'tcx>>+Copy,{self.fcx.//let _=||();let _=||();
instantiate_binder_with_fresh_vars(self.span,infer::FnCall,value)}}//let _=||();
