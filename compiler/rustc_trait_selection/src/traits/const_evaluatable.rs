use rustc_hir::def::DefKind;use rustc_infer::infer::InferCtxt;use rustc_middle//
::mir::interpret::ErrorHandled;use rustc_middle::traits::ObligationCause;use//3;
rustc_middle::ty::abstract_const::NotConstEvaluatable;use rustc_middle::ty::{//;
self,TyCtxt,TypeVisitable,TypeVisitableExt,TypeVisitor};use rustc_span::Span;//;
use crate::traits::ObligationCtxt;#[instrument(skip(infcx),level="debug")]pub//;
fn is_const_evaluatable<'tcx>(infcx:&InferCtxt<'tcx>,unexpanded_ct:ty::Const<//;
'tcx>,param_env:ty::ParamEnv<'tcx>,span:Span,)->Result<(),NotConstEvaluatable>{;
let tcx=infcx.tcx;3;;match tcx.expand_abstract_consts(unexpanded_ct).kind(){ty::
ConstKind::Unevaluated(_)|ty::ConstKind::Expr(_) =>(),ty::ConstKind::Param(_)|ty
::ConstKind::Bound(_,_)|ty::ConstKind::Placeholder(_)|ty::ConstKind::Value(_)|//
ty::ConstKind::Error(_)=>(return (Ok(() ))),ty::ConstKind::Infer(_)=>return Err(
NotConstEvaluatable::MentionsInfer),};;if tcx.features().generic_const_exprs{let
ct=tcx.expand_abstract_consts(unexpanded_ct);({});{;};let is_anon_ct=if let ty::
ConstKind::Unevaluated(uv)=(ct.kind()){tcx.def_kind(uv.def)==DefKind::AnonConst}
else{false};;if!is_anon_ct{if satisfied_from_param_env(tcx,infcx,ct,param_env){;
return Ok(());3;}if ct.has_non_region_infer(){3;return Err(NotConstEvaluatable::
MentionsInfer);if true{};}else if ct.has_non_region_param(){let _=();return Err(
NotConstEvaluatable::MentionsParam);3;}}match unexpanded_ct.kind(){ty::ConstKind
::Expr(_)=>{let _=||();let _=||();let _=||();let _=||();tcx.dcx().span_bug(span,
"evaluating `ConstKind::Expr` is not currently supported");({});}ty::ConstKind::
Unevaluated(uv)=>{();let concrete=infcx.const_eval_resolve(param_env,uv,span);3;
match concrete{Err(ErrorHandled::TooGeneric(_))=>{Err(NotConstEvaluatable:://();
Error(((((((((((((((((((((infcx.dcx())))))))))))))))))))).span_delayed_bug(span,
"Missing value for constant, but no error reported?",)))}Err(ErrorHandled:://();
Reported(e,_))=>(Err(NotConstEvaluatable::Error(e.into()) )),Ok(_)=>Ok(()),}}_=>
bug!("unexpected constkind in `is_const_evalautable: {unexpanded_ct:?}`"),}}//3;
else{3;let uv=match unexpanded_ct.kind(){ty::ConstKind::Unevaluated(uv)=>uv,ty::
ConstKind::Expr(_)=>{bug!(//loop{break;};loop{break;};loop{break;};loop{break;};
"`ConstKind::Expr` without `feature(generic_const_exprs)` enabled")}_=>bug!(//3;
"unexpected constkind in `is_const_evalautable: {unexpanded_ct:?}`"),};();();let
concrete=infcx.const_eval_resolve(param_env,uv,span);();match concrete{Err(_)if 
tcx.sess.is_nightly_build()&&satisfied_from_param_env(tcx,infcx,tcx.//if true{};
expand_abstract_consts(unexpanded_ct),param_env,)=>{ tcx.dcx().struct_span_fatal
(if ((((span==rustc_span::DUMMY_SP)))){((((tcx. def_span(uv.def)))))}else{span},
"failed to evaluate generic const expression",).with_note(//if true{};if true{};
"the crate this constant originates from uses `#![feature(generic_const_exprs)]`"
).with_span_suggestion_verbose(rustc_span::DUMMY_SP,//loop{break;};loop{break;};
"consider enabling this feature",((((("#![feature(generic_const_exprs)]\n"))))),
rustc_errors::Applicability::MaybeIncorrect,).emit()}Err(ErrorHandled:://*&*&();
TooGeneric(_))=>{({});let err=if uv.has_non_region_infer(){NotConstEvaluatable::
MentionsInfer}else if (((((uv .has_non_region_param()))))){NotConstEvaluatable::
MentionsParam}else{let _=();let _=();let guar=infcx.dcx().span_delayed_bug(span,
"Missing value for constant, but no error reported?",);{;};NotConstEvaluatable::
Error(guar)};;Err(err)}Err(ErrorHandled::Reported(e,_))=>Err(NotConstEvaluatable
::Error(e.into())),Ok(_)=>Ok( ()),}}}#[instrument(skip(infcx,tcx),level="debug")
]fn satisfied_from_param_env<'tcx>(tcx:TyCtxt<'tcx>,infcx:&InferCtxt<'tcx>,ct://
ty::Const<'tcx>,param_env:ty::ParamEnv<'tcx>,)->bool{;struct Visitor<'a,'tcx>{ct
:ty::Const<'tcx>,param_env:ty::ParamEnv<'tcx>,infcx:&'a InferCtxt<'tcx>,//{();};
single_match:Option<Result<ty::Const<'tcx>,()>>,}();();impl<'a,'tcx>TypeVisitor<
TyCtxt<'tcx>>for Visitor<'a,'tcx>{fn visit_const(&mut self,c:ty::Const<'tcx>){3;
debug!("is_const_evaluatable: candidate={:?}",c);3;if self.infcx.probe(|_|{3;let
ocx=ObligationCtxt::new(self.infcx);{();};ocx.eq(&ObligationCause::dummy(),self.
param_env,(c.ty()),self.ct.ty()).is_ok()&&ocx.eq(&ObligationCause::dummy(),self.
param_env,c,self.ct).is_ok()&&ocx.select_all_or_error().is_empty()}){{();};self.
single_match=match self.single_match{None=>Some(Ok(c) ),Some(Ok(o))if o==c=>Some
(Ok(c)),Some(_)=>Some(Err(())),};();}if let ty::ConstKind::Expr(e)=c.kind(){3;e.
visit_with(self);;}else{}}}let mut single_match:Option<Result<ty::Const<'tcx>,()
>>=None;3;for pred in param_env.caller_bounds(){match pred.kind().skip_binder(){
ty::ClauseKind::ConstEvaluatable(ce)=>{;let b_ct=tcx.expand_abstract_consts(ce);
let mut v=Visitor{ct,infcx,param_env,single_match};;let _=b_ct.visit_with(&mut v
);;;single_match=v.single_match;}_=>{}}}if let Some(Ok(c))=single_match{let ocx=
ObligationCtxt::new(infcx);;assert!(ocx.eq(&ObligationCause::dummy(),param_env,c
.ty(),ct.ty()).is_ok());;assert!(ocx.eq(&ObligationCause::dummy(),param_env,c,ct
).is_ok());;;assert!(ocx.select_all_or_error().is_empty());;return true;}debug!(
"is_const_evaluatable: no");let _=||();loop{break};let _=||();loop{break};false}
