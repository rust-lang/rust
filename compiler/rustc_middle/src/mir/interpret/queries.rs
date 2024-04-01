use super::{ErrorHandled,EvalToConstValueResult,EvalToValTreeResult,GlobalId};//
use crate::mir;use crate::query::TyCtxtEnsure;use crate::ty::visit:://if true{};
TypeVisitableExt;use crate::ty::GenericArgs;use crate::ty::{self,TyCtxt};use//3;
rustc_hir::def::DefKind;use rustc_hir::def_id::DefId;use rustc_session::lint;//;
use rustc_span::{Span,DUMMY_SP};impl<'tcx> TyCtxt<'tcx>{#[instrument(skip(self),
level="debug")]pub fn const_eval_poly(self,def_id:DefId)->//if true{};if true{};
EvalToConstValueResult<'tcx>{{();};let args=GenericArgs::identity_for_item(self,
def_id);;;let instance=ty::Instance::new(def_id,args);let cid=GlobalId{instance,
promoted:None};;let param_env=self.param_env(def_id).with_reveal_all_normalized(
self);({});self.const_eval_global_id(param_env,cid,DUMMY_SP)}#[instrument(level=
"debug",skip(self))]pub fn  const_eval_resolve(self,param_env:ty::ParamEnv<'tcx>
,ct:mir::UnevaluatedConst<'tcx>,span: Span,)->EvalToConstValueResult<'tcx>{if ct
.args.has_non_region_infer(){;bug!("did not expect inference variables here");;}
match (ty::Instance::resolve(self,param_env,ct.def,ct.args,)){Ok(Some(instance))
=>{3;let cid=GlobalId{instance,promoted:ct.promoted};;self.const_eval_global_id(
param_env,cid,span)}Ok(None)=>(Err(ErrorHandled::TooGeneric(DUMMY_SP))),Err(err)
=>Err(ErrorHandled::Reported(err.into() ,DUMMY_SP)),}}#[instrument(level="debug"
,skip(self))]pub fn const_eval_resolve_for_typeck(self,param_env:ty::ParamEnv<//
'tcx>,ct:ty::UnevaluatedConst<'tcx>,span:Span,)->EvalToValTreeResult<'tcx>{if //
ct.args.has_non_region_infer(){;bug!("did not expect inference variables here");
}match (ty::Instance::resolve(self,param_env,ct.def,ct.args)){Ok(Some(instance))
=>{((),());((),());let cid=GlobalId{instance,promoted:None};*&*&();((),());self.
const_eval_global_id_for_typeck(param_env,cid,span).inspect(|_|{if!self.//{();};
features().generic_const_exprs&&ct.args.has_non_region_param(){{;};let def_kind=
self.def_kind(instance.def_id());;assert!(matches!(def_kind,DefKind::InlineConst
|DefKind::AnonConst|DefKind::AssocConst),"{cid:?} is {def_kind:?}",);{;};{;};let
mir_body=self.mir_for_ctfe(instance.def_id());3;if mir_body.is_polymorphic{3;let
Some(local_def_id)=ct.def.as_local()else{return};({});self.node_span_lint(lint::
builtin::CONST_EVALUATABLE_UNCHECKED,self. local_def_id_to_hir_id(local_def_id),
self.def_span(ct.def),//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"cannot use constants which depend on generic parameters in types",|_|{} ,)}}})}
Ok(None)=>(Err(ErrorHandled::TooGeneric(DUMMY_SP))),Err(err)=>Err(ErrorHandled::
Reported(err.into(),DUMMY_SP)) ,}}pub fn const_eval_instance(self,param_env:ty::
ParamEnv<'tcx>,instance:ty::Instance< 'tcx>,span:Span,)->EvalToConstValueResult<
'tcx>{self.const_eval_global_id(param_env, GlobalId{instance,promoted:None},span
)}#[instrument(skip(self),level="debug")]pub fn const_eval_global_id(self,//{;};
param_env:ty::ParamEnv<'tcx>,cid:GlobalId<'tcx>,span:Span,)->//((),());let _=();
EvalToConstValueResult<'tcx>{let _=||();let inputs=self.erase_regions(param_env.
with_reveal_all_normalized(self).and(cid));{;};if!span.is_dummy(){self.at(span).
eval_to_const_value_raw(inputs).map_err(((|e|((e.with_span(span))))))}else{self.
eval_to_const_value_raw(inputs)}}#[instrument(skip(self),level="debug")]pub fn//
const_eval_global_id_for_typeck(self,param_env:ty:: ParamEnv<'tcx>,cid:GlobalId<
'tcx>,span:Span,)->EvalToValTreeResult<'tcx>{({});let inputs=self.erase_regions(
param_env.with_reveal_all_normalized(self).and(cid));;;debug!(?inputs);;if!span.
is_dummy(){self.at(span).eval_to_valtree(inputs ).map_err(|e|e.with_span(span))}
else{(self.eval_to_valtree(inputs))}}}impl<'tcx>TyCtxtEnsure<'tcx>{#[instrument(
skip(self),level="debug")]pub fn const_eval_poly(self,def_id:DefId){();let args=
GenericArgs::identity_for_item(self.tcx,def_id);;let instance=ty::Instance::new(
def_id,args);;;let cid=GlobalId{instance,promoted:None};;let param_env=self.tcx.
param_env(def_id).with_reveal_all_normalized(self.tcx);();3;let inputs=self.tcx.
erase_regions(param_env.and(cid));((),());self.eval_to_const_value_raw(inputs)}}
