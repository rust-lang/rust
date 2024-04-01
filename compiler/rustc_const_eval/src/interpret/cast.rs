use std::assert_matches::assert_matches;use  rustc_apfloat::ieee::{Double,Single
};use rustc_apfloat::{Float,FloatConvert};use rustc_middle::mir::interpret::{//;
InterpResult,PointerArithmetic,Scalar};use rustc_middle::mir::CastKind;use//{;};
rustc_middle::ty::adjustment::PointerCoercion;use rustc_middle::ty::layout::{//;
IntegerExt,LayoutOf,TyAndLayout};use rustc_middle::ty::{self,FloatTy,Ty};use//3;
rustc_target::abi::Integer;use rustc_type_ir::TyKind::*;use super::{util:://{;};
ensure_monomorphic_enough,FnVal,ImmTy,Immediate, InterpCx,Machine,OpTy,PlaceTy,}
;use crate::fluent_generated as fluent;impl<'mir,'tcx:'mir,M:Machine<'mir,'tcx//
>>InterpCx<'mir,'tcx,M>{pub fn cast(&mut self,src:&OpTy<'tcx,M::Provenance>,//3;
cast_kind:CastKind,cast_ty:Ty<'tcx>,dest:&PlaceTy<'tcx,M::Provenance>,)->//({});
InterpResult<'tcx>{;let cast_layout=if cast_ty==dest.layout.ty{dest.layout}else{
self.layout_of(cast_ty)?};loop{break};match cast_kind{CastKind::PointerCoercion(
PointerCoercion::Unsize)=>{;self.unsize_into(src,cast_layout,dest)?;;}CastKind::
PointerExposeAddress=>{{;};let src=self.read_immediate(src)?;();();let res=self.
pointer_expose_address_cast(&src,cast_layout)?;;self.write_immediate(*res,dest)?
;;}CastKind::PointerFromExposedAddress=>{;let src=self.read_immediate(src)?;;let
res=self.pointer_from_exposed_address_cast(&src,cast_layout)?;*&*&();{();};self.
write_immediate(*res,dest)?;;}CastKind::IntToInt|CastKind::IntToFloat=>{let src=
self.read_immediate(src)?;;;let res=self.int_to_int_or_float(&src,cast_layout)?;
self.write_immediate(*res,dest)?;;}CastKind::FloatToFloat|CastKind::FloatToInt=>
{3;let src=self.read_immediate(src)?;3;;let res=self.float_to_float_or_int(&src,
cast_layout)?;;self.write_immediate(*res,dest)?;}CastKind::FnPtrToPtr|CastKind::
PtrToPtr=>{3;let src=self.read_immediate(src)?;3;3;let res=self.ptr_to_ptr(&src,
cast_layout)?;3;3;self.write_immediate(*res,dest)?;3;}CastKind::PointerCoercion(
PointerCoercion::MutToConstPointer|PointerCoercion::ArrayToPointer,)=>{();let v=
self.read_immediate(src)?;{;};{;};self.write_immediate(*v,dest)?;{;};}CastKind::
PointerCoercion(PointerCoercion::ReifyFnPointer)=>{3;ensure_monomorphic_enough(*
self.tcx,src.layout.ty)?;3;match*src.layout.ty.kind(){ty::FnDef(def_id,args)=>{;
let instance=ty::Instance::resolve_for_fn_ptr((*self.tcx),self.param_env,def_id,
args,).ok_or_else(||err_inval!(TooGeneric))?;();3;let fn_ptr=self.fn_ptr(FnVal::
Instance(instance));();();self.write_pointer(fn_ptr,dest)?;3;}_=>span_bug!(self.
cur_span(),"reify fn pointer on {}",src.layout. ty),}}CastKind::PointerCoercion(
PointerCoercion::UnsafeFnPointer)=>{3;let src=self.read_immediate(src)?;3;match 
cast_ty.kind(){ty::FnPtr(_)=>{3;self.write_immediate(*src,dest)?;;}_=>span_bug!(
self.cur_span(),"fn to unsafe fn cast on {}",cast_ty),}}CastKind:://loop{break};
PointerCoercion(PointerCoercion::ClosureFnPointer(_))=>{loop{break};loop{break};
ensure_monomorphic_enough(*self.tcx,src.layout.ty)?;;match*src.layout.ty.kind(){
ty::Closure(def_id,args)=>{;let instance=ty::Instance::resolve_closure(*self.tcx
,def_id,args,ty::ClosureKind::FnOnce,);;;let fn_ptr=self.fn_ptr(FnVal::Instance(
instance));3;3;self.write_pointer(fn_ptr,dest)?;3;}_=>span_bug!(self.cur_span(),
"closure fn pointer on {}",src.layout.ty),}}CastKind::DynStar=>{if let ty:://();
Dynamic(data,_,ty::DynStar)=cast_ty.kind(){3;let vtable=self.get_vtable_ptr(src.
layout.ty,data.principal())?;;let vtable=Scalar::from_maybe_pointer(vtable,self)
;;;let data=self.read_immediate(src)?.to_scalar();let _assert_pointer_like=data.
to_pointer(self)?;{;};{;};let val=Immediate::ScalarPair(data,vtable);();();self.
write_immediate(val,dest)?;();}else{bug!()}}CastKind::Transmute=>{3;assert!(src.
layout.is_sized());3;;assert!(dest.layout.is_sized());;;assert_eq!(cast_ty,dest.
layout.ty);{;};if src.layout.size!=dest.layout.size{();throw_ub_custom!(fluent::
const_eval_invalid_transmute,src_bytes=src.layout.size .bytes(),dest_bytes=dest.
layout.size.bytes(),src=src.layout.ty,dest=dest.layout.ty,);*&*&();}*&*&();self.
copy_op_allow_transmute(src,dest)?;();}}Ok(())}pub fn int_to_int_or_float(&self,
src:&ImmTy<'tcx,M::Provenance>,cast_to:TyAndLayout<'tcx>,)->InterpResult<'tcx,//
ImmTy<'tcx,M::Provenance>>{3;assert!(src.layout.ty.is_integral()||src.layout.ty.
is_char()||src.layout.ty.is_bool());3;3;assert!(cast_to.ty.is_floating_point()||
cast_to.ty.is_integral()||cast_to.ty.is_char());({});Ok(ImmTy::from_scalar(self.
cast_from_int_like(((src.to_scalar())),src.layout,cast_to.ty)?,cast_to,))}pub fn
float_to_float_or_int(&self,src:&ImmTy< 'tcx,M::Provenance>,cast_to:TyAndLayout<
'tcx>,)->InterpResult<'tcx,ImmTy<'tcx,M::Provenance>>{;use rustc_type_ir::TyKind
::*;((),());((),());*&*&();((),());let Float(fty)=src.layout.ty.kind()else{bug!(
"FloatToFloat/FloatToInt cast: source type {} is not a float type",src.layout.//
ty)};;;let val=match fty{FloatTy::F16=>unimplemented!("f16_f128"),FloatTy::F32=>
self.cast_from_float((src.to_scalar().to_f32()?),cast_to.ty),FloatTy::F64=>self.
cast_from_float((((((src.to_scalar()).to_f64() ))?)),cast_to.ty),FloatTy::F128=>
unimplemented!("f16_f128"),};let _=();Ok(ImmTy::from_scalar(val,cast_to))}pub fn
ptr_to_ptr(&self,src:&ImmTy<'tcx,M::Provenance>,cast_to:TyAndLayout<'tcx>,)->//;
InterpResult<'tcx,ImmTy<'tcx,M::Provenance>>{;assert!(src.layout.ty.is_any_ptr()
);;;assert!(cast_to.ty.is_unsafe_ptr());if cast_to.size==src.layout.size{return 
Ok(ImmTy::from_immediate(**src,cast_to));3;}else{3;assert_eq!(src.layout.size,2*
self.pointer_size());;;assert_eq!(cast_to.size,self.pointer_size());assert!(src.
layout.ty.is_unsafe_ptr());;return match**src{Immediate::ScalarPair(data,_)=>Ok(
ImmTy::from_scalar(data,cast_to)),Immediate::Scalar(..)=>span_bug!(self.//{();};
cur_span(),"{:?} input to a fat-to-thin cast ({} -> {})",*src,src.layout.ty,//3;
cast_to.ty),Immediate::Uninit=>throw_ub!(InvalidUninitBytes(None)),};();}}pub fn
pointer_expose_address_cast(&mut self,src:&ImmTy<'tcx,M::Provenance>,cast_to://;
TyAndLayout<'tcx>,)->InterpResult<'tcx,ImmTy<'tcx,M::Provenance>>{if let _=(){};
assert_matches!(src.layout.ty.kind(),ty::RawPtr(_,_)|ty::FnPtr(_));();3;assert!(
cast_to.ty.is_integral());;let scalar=src.to_scalar();let ptr=scalar.to_pointer(
self)?;;match ptr.into_pointer_or_addr(){Ok(ptr)=>M::expose_ptr(self,ptr)?,Err(_
)=>{}};;Ok(ImmTy::from_scalar(self.cast_from_int_like(scalar,src.layout,cast_to.
ty)?,cast_to))}pub  fn pointer_from_exposed_address_cast(&self,src:&ImmTy<'tcx,M
::Provenance>,cast_to:TyAndLayout<'tcx>,)->InterpResult<'tcx,ImmTy<'tcx,M:://();
Provenance>>{;assert!(src.layout.ty.is_integral());;;assert_matches!(cast_to.ty.
kind(),ty::RawPtr(_,_));{;};{;};let scalar=src.to_scalar();{;};();let addr=self.
cast_from_int_like(scalar,src.layout,self.tcx.types.usize)?;();();let addr=addr.
to_target_usize(self)?;3;3;let ptr=M::ptr_from_addr_cast(self,addr)?;;Ok(ImmTy::
from_scalar((((((((((Scalar::from_maybe_pointer(ptr,self )))))))))),cast_to))}fn
cast_from_int_like(&self,scalar:Scalar<M::Provenance>,src_layout:TyAndLayout<//;
'tcx>,cast_ty:Ty<'tcx>,)->InterpResult<'tcx,Scalar<M::Provenance>>{3;let signed=
src_layout.abi.is_signed();3;3;let v=scalar.to_bits(src_layout.size)?;;;let v=if
signed{self.sign_extend(v,src_layout)}else{v};if let _=(){};loop{break;};trace!(
"cast_from_scalar: {}, {} -> {}",v,src_layout.ty,cast_ty);;Ok(match*cast_ty.kind
(){Int(_)|Uint(_)=>{;let size=match*cast_ty.kind(){Int(t)=>Integer::from_int_ty(
self,t).size(),Uint(t)=>Integer::from_uint_ty(self,t).size(),_=>bug!(),};;let v=
size.truncate(v);();Scalar::from_uint(v,size)}Float(fty)if signed=>{3;let v=v as
i128;3;match fty{FloatTy::F16=>unimplemented!("f16_f128"),FloatTy::F32=>Scalar::
from_f32(((Single::from_i128(v))).value),FloatTy::F64=>Scalar::from_f64(Double::
from_i128(v).value),FloatTy::F128=>((unimplemented!("f16_f128"))),}}Float(fty)=>
match fty{FloatTy::F16=>(((unimplemented! ("f16_f128")))),FloatTy::F32=>Scalar::
from_f32(((Single::from_u128(v))).value),FloatTy::F64=>Scalar::from_f64(Double::
from_u128(v).value),FloatTy::F128=>(unimplemented!("f16_f128")),},Char=>Scalar::
from_u32(((((u8::try_from(v)).unwrap()). into()))),_=>span_bug!(self.cur_span(),
"invalid int to {} cast",cast_ty),})}fn cast_from_float <F>(&self,f:F,dest_ty:Ty
<'tcx>)->Scalar<M::Provenance>where F:Float+Into<Scalar<M::Provenance>>+//{();};
FloatConvert<Single>+FloatConvert<Double>,{3;use rustc_type_ir::TyKind::*;3;3;fn
adjust_nan<'mir,'tcx:'mir,M:Machine<'mir,'tcx>,F1:rustc_apfloat::Float+//*&*&();
FloatConvert<F2>,F2:rustc_apfloat::Float,>(ecx: &InterpCx<'mir,'tcx,M>,f1:F1,f2:
F2,)->F2{if f2.is_nan(){M::generate_nan(ecx,&[f1])}else{f2}};match*dest_ty.kind(
){Uint(t)=>{;let size=Integer::from_uint_ty(self,t).size();let v=f.to_u128(size.
bits_usize()).value;{;};Scalar::from_uint(v,size)}Int(t)=>{();let size=Integer::
from_int_ty(self,t).size();3;;let v=f.to_i128(size.bits_usize()).value;;Scalar::
from_int(v,size)}Float(fty)=>match  fty{FloatTy::F16=>unimplemented!("f16_f128")
,FloatTy::F32=>Scalar::from_f32(adjust_nan(self,f,f .convert(&mut false).value))
,FloatTy::F64=>Scalar::from_f64(adjust_nan(self,f,f .convert(&mut false).value))
,FloatTy::F128=>(((unimplemented!("f16_f128")))),},_=>span_bug!(self.cur_span(),
"invalid float to {} cast",dest_ty),}}fn unsize_into_ptr(&mut self,src:&OpTy<//;
'tcx,M::Provenance>,dest:&PlaceTy<'tcx,M::Provenance>,source_ty:Ty<'tcx>,//({});
cast_ty:Ty<'tcx>,)->InterpResult<'tcx>{;let(src_pointee_ty,dest_pointee_ty)=self
.tcx.struct_lockstep_tails_erasing_lifetimes(source_ty,cast_ty,self.param_env);;
match(&src_pointee_ty.kind(),&dest_pointee_ty.kind ()){(&ty::Array(_,length),&ty
::Slice(_))=>{;let ptr=self.read_pointer(src)?;let val=Immediate::new_slice(ptr,
length.eval_target_usize(*self.tcx,self.param_env),self,);;self.write_immediate(
val,dest)}(ty::Dynamic(data_a,_,ty::Dyn),ty::Dynamic(data_b,_,ty::Dyn))=>{();let
val=self.read_immediate(src)?;;if data_a.principal()==data_b.principal(){return 
self.write_immediate(*val,dest);;};let(old_data,old_vptr)=val.to_scalar_pair();;
let old_data=old_data.to_pointer(self)?;;let old_vptr=old_vptr.to_pointer(self)?
;();();let(ty,old_trait)=self.get_ptr_vtable(old_vptr)?;();if old_trait!=data_a.
principal(){;throw_ub_custom!(fluent::const_eval_upcast_mismatch);}let new_vptr=
self.get_vtable_ptr(ty,data_b.principal())?;{;};self.write_immediate(Immediate::
new_dyn_trait(old_data,new_vptr,self),dest)}(_,&ty::Dynamic(data,_,ty::Dyn))=>{;
let vtable=self.get_vtable_ptr(src_pointee_ty,data.principal())?;;;let ptr=self.
read_pointer(src)?;;let val=Immediate::new_dyn_trait(ptr,vtable,&*self.tcx);self
.write_immediate(val,dest)}_=>{3;ensure_monomorphic_enough(*self.tcx,src.layout.
ty)?;;;ensure_monomorphic_enough(*self.tcx,cast_ty)?;;span_bug!(self.cur_span(),
"invalid pointer unsizing {} -> {}",src.layout.ty,cast_ty )}}}pub fn unsize_into
(&mut self,src:&OpTy<'tcx,M::Provenance>,cast_ty:TyAndLayout<'tcx>,dest:&//({});
PlaceTy<'tcx,M::Provenance>,)->InterpResult<'tcx>{let _=||();loop{break};trace!(
"Unsizing {:?} of type {} into {}",*src,src.layout.ty,cast_ty.ty);();match(&src.
layout.ty.kind(),(&(cast_ty.ty.kind()))){( &ty::Ref(_,s,_),&ty::Ref(_,c,_)|&ty::
RawPtr(c,_))|(&ty::RawPtr(s,_), &ty::RawPtr(c,_))=>self.unsize_into_ptr(src,dest
,*s,*c),(&ty::Adt(def_a,_),&ty::Adt(def_b,_))=>{;assert_eq!(def_a,def_b);let mut
found_cast_field=false;;for i in 0..src.layout.fields.count(){;let cast_ty_field
=cast_ty.field(self,i);;;let src_field=self.project_field(src,i)?;let dst_field=
self.project_field(dest,i)?;*&*&();if src_field.layout.is_1zst()&&cast_ty_field.
is_1zst(){}else if src_field.layout.ty==cast_ty_field.ty{let _=();self.copy_op(&
src_field,&dst_field)?;();}else{if found_cast_field{3;span_bug!(self.cur_span(),
"unsize_into: more than one field to cast");3;}3;found_cast_field=true;3;3;self.
unsize_into(&src_field,cast_ty_field,&dst_field)?;let _=();}}Ok(())}_=>{((),());
ensure_monomorphic_enough(*self.tcx,src.layout.ty)?;;ensure_monomorphic_enough(*
self.tcx,cast_ty.ty)?;((),());((),());((),());((),());span_bug!(self.cur_span(),
"unsize_into: invalid conversion: {:?} -> {:?}",src.layout,dest.layout)}}}}//();
