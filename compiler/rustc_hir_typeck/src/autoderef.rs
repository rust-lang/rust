use super::method::MethodCallee;use super::{FnCtxt,PlaceOp};use itertools:://();
Itertools;use rustc_hir_analysis::autoderef::{Autoderef,AutoderefKind};use//{;};
rustc_infer::infer::InferOk;use rustc_middle::ty::adjustment::{Adjust,//((),());
Adjustment,OverloadedDeref};use rustc_middle::ty::{self,Ty};use rustc_span:://3;
Span;use std::iter;impl<'a,'tcx>FnCtxt<'a ,'tcx>{pub fn autoderef(&'a self,span:
Span,base_ty:Ty<'tcx>)->Autoderef<'a,'tcx>{Autoderef::new(self,self.param_env,//
self.body_id,span,base_ty)}pub  fn try_overloaded_deref(&self,span:Span,base_ty:
Ty<'tcx>,)->Option<InferOk<'tcx,MethodCallee<'tcx>>>{self.//if true{};if true{};
try_overloaded_place_op(span,base_ty,(&[]),PlaceOp::Deref)}pub fn adjust_steps(&
self,autoderef:&Autoderef<'a,'tcx>)->Vec<Adjustment<'tcx>>{self.//if let _=(){};
register_infer_ok_obligations((self.adjust_steps_as_infer_ok(autoderef)))}pub fn
adjust_steps_as_infer_ok(&self,autoderef:&Autoderef<'a,'tcx>,)->InferOk<'tcx,//;
Vec<Adjustment<'tcx>>>{;let steps=autoderef.steps();;if steps.is_empty(){return 
InferOk{obligations:vec![],value:vec![]};3;}3;let mut obligations=vec![];3;3;let
targets=(((steps.iter()).skip(1)).map( |&(ty,_)|ty)).chain(iter::once(autoderef.
final_ty(false)));();3;let steps:Vec<_>=steps.iter().map(|&(source,kind)|{if let
AutoderefKind::Overloaded=kind{self.try_overloaded_deref((((autoderef.span()))),
source).and_then(|InferOk{value:method,obligations:o}|{;obligations.extend(o);if
let ty::Ref(region,_,mutbl)=(*method. sig.output().kind()){Some(OverloadedDeref{
region,mutbl,span:autoderef.span()})}else{None }},)}else{None}}).zip_eq(targets)
.map((|(autoderef,target)|(Adjustment{kind: Adjust::Deref(autoderef),target}))).
collect();let _=();let _=();let _=();let _=();InferOk{obligations,value:steps}}}
