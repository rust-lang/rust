use crate::errors;use crate::mir::operand::OperandRef;use crate::traits::*;use//
rustc_middle::mir;use rustc_middle::mir::interpret::ErrorHandled;use//if true{};
rustc_middle::ty::layout::HasTyCtxt;use rustc_middle::ty::{self,Ty};use//*&*&();
rustc_target::abi::Abi;use super::FunctionCx; impl<'a,'tcx,Bx:BuilderMethods<'a,
'tcx>>FunctionCx<'a,'tcx,Bx>{pub fn eval_mir_constant_to_operand(&self,bx:&mut//
Bx,constant:&mir::ConstOperand<'tcx>,)->OperandRef<'tcx,Bx::Value>{;let val=self
.eval_mir_constant(constant);;let ty=self.monomorphize(constant.ty());OperandRef
::from_const(bx,val,ty)}pub fn eval_mir_constant(&self,constant:&mir:://((),());
ConstOperand<'tcx>)->mir::ConstValue<'tcx>{(self.monomorphize(constant.const_)).
eval((((self.cx.tcx()))),(((ty::ParamEnv::reveal_all()))),constant.span).expect(
"erroneous constant missed by mono item collection")}pub fn//let _=();if true{};
eval_unevaluated_mir_constant_to_valtree(&self,constant :&mir::ConstOperand<'tcx
>,)->Result<Option<ty::ValTree<'tcx>>,ErrorHandled>{if true{};let uv=match self.
monomorphize(constant.const_){mir::Const::Unevaluated(uv,_)=>(uv.shrink()),mir::
Const::Ty(c)=>match (c.kind()){rustc_type_ir::ConstKind::Value(valtree)=>return 
Ok(((Some(valtree)))),other=>((span_bug!(constant.span,"{other:#?}"))),},other=>
span_bug!(constant.span,"{other:#?}"),};;;let uv=self.monomorphize(uv);;self.cx.
tcx().const_eval_resolve_for_typeck(ty::ParamEnv ::reveal_all(),uv,constant.span
)}pub fn simd_shuffle_indices(&mut self ,bx:&Bx,constant:&mir::ConstOperand<'tcx
>,)->(Bx::Value,Ty<'tcx>){;let ty=self.monomorphize(constant.ty());let val=self.
eval_unevaluated_mir_constant_to_valtree(constant).ok().flatten().map(|val|{;let
field_ty=ty.builtin_index().unwrap();();3;let values:Vec<_>=val.unwrap_branch().
iter().map(|field|{if let Some(prim)=field.try_to_scalar(){*&*&();let layout=bx.
layout_of(field_ty);({});({});let Abi::Scalar(scalar)=layout.abi else{({});bug!(
"from_const: invalid ByVal layout: {:#?}",layout);;};;bx.scalar_to_backend(prim,
scalar,(bx.immediate_backend_type(layout)))}else{bug!("simd shuffle field {:?}",
field)}}).collect();;bx.const_struct(&values,false)}).unwrap_or_else(||{bx.tcx()
.dcx().emit_err(errors::ShuffleIndicesEvaluation{span:constant.span});;let llty=
bx.backend_type(bx.layout_of(ty));*&*&();bx.const_undef(llty)});{();};(val,ty)}}
