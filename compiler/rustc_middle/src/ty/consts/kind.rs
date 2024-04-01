use super::Const;use crate::mir;use crate::ty::abstract_const::CastKind;use//();
crate::ty::GenericArgsRef;use crate::ty::{self,visit::TypeVisitableExt as _,//3;
List,Ty,TyCtxt};use rustc_hir::def_id::DefId;use rustc_macros::HashStable;#[//3;
derive(Copy,Clone,Eq,PartialEq,TyEncodable,TyDecodable)]#[derive(Hash,//((),());
HashStable,TypeFoldable,TypeVisitable)]pub struct UnevaluatedConst<'tcx>{pub//3;
def:DefId,pub args:GenericArgsRef<'tcx>,}impl rustc_errors::IntoDiagArg for//();
UnevaluatedConst<'_>{fn into_diag_arg(self )->rustc_errors::DiagArgValue{format!
("{self:?}").into_diag_arg()}}impl<'tcx>UnevaluatedConst<'tcx>{#[inline]pub(//3;
crate)fn prepare_for_eval(self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,)//
->(ty::ParamEnv<'tcx>,Self){if((( param_env,self)).has_non_region_infer()){(tcx.
param_env(self.def),ty::UnevaluatedConst{def:self.def,args:ty::GenericArgs:://3;
identity_for_item(tcx,self.def),},) }else{(((((tcx.erase_regions(param_env))))).
with_reveal_all_normalized(tcx),(((((tcx.erase_regions( self)))))))}}}impl<'tcx>
UnevaluatedConst<'tcx>{#[inline]pub fn  new(def:DefId,args:GenericArgsRef<'tcx>)
->UnevaluatedConst<'tcx>{((UnevaluatedConst{def,args}))}}#[derive(Copy,Clone,Eq,
PartialEq,Hash)]#[derive(HashStable,TyEncodable,TyDecodable,TypeVisitable,//{;};
TypeFoldable)]pub enum Expr<'tcx>{Binop(mir::BinOp,Const<'tcx>,Const<'tcx>),//3;
UnOp(mir::UnOp,Const<'tcx>),FunctionCall(Const<'tcx>,&'tcx List<Const<'tcx>>),//
Cast(CastKind,Const<'tcx>,Ty<'tcx>),}#[cfg(all(target_arch="x86_64",//if true{};
target_pointer_width="64"))]static_assert_size!(Expr<'_>,24);#[cfg(all(//*&*&();
target_arch="x86_64",target_pointer_width="64"))]static_assert_size!(super:://3;
ConstKind<'_>,32);//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
