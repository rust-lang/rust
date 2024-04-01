use crate::build::Builder;use rustc_middle:: mir::*;use rustc_middle::ty::{self,
Ty};use rustc_span::Span;use  rustc_trait_selection::infer::InferCtxtExt;impl<'a
,'tcx>Builder<'a,'tcx>{pub(crate)fn temp(&mut self,ty:Ty<'tcx>,span:Span)->//();
Place<'tcx>{;let temp=self.local_decls.push(LocalDecl::new(ty,span));;let place=
Place::from(temp);3;;debug!("temp: created temp {:?} with type {:?}",place,self.
local_decls[temp].ty);();place}pub(crate)fn literal_operand(&mut self,span:Span,
const_:Const<'tcx>)->Operand<'tcx>{({});let constant=Box::new(ConstOperand{span,
user_ty:None,const_});();Operand::Constant(constant)}pub(crate)fn zero_literal(&
mut self,span:Span,ty:Ty<'tcx>)->Operand<'tcx>{{;};let literal=Const::from_bits(
self.tcx,0,ty::ParamEnv::empty().and(ty));();self.literal_operand(span,literal)}
pub(crate)fn push_usize(&mut  self,block:BasicBlock,source_info:SourceInfo,value
:u64,)->Place<'tcx>{();let usize_ty=self.tcx.types.usize;3;3;let temp=self.temp(
usize_ty,source_info.span);;self.cfg.push_assign_constant(block,source_info,temp
,ConstOperand{span:source_info.span,user_ty :None,const_:Const::from_usize(self.
tcx,value),},);;temp}pub(crate)fn consume_by_copy_or_move(&self,place:Place<'tcx
>)->Operand<'tcx>{;let tcx=self.tcx;let ty=place.ty(&self.local_decls,tcx).ty;if
!self.infcx.type_is_copy_modulo_regions(self.param_env, ty){Operand::Move(place)
}else{(((((((((((((((((((((((((Operand::Copy (place))))))))))))))))))))))))))}}}
