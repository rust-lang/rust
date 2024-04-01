use rustc_hir::def_id::DefId;use rustc_middle::ty;use rustc_middle::ty::layout//
::{LayoutOf as _,ValidityRequirement};use rustc_middle::ty::GenericArgsRef;use//
rustc_middle::ty::{Ty,TyCtxt};use rustc_middle::{mir::{self,interpret::{//{();};
Allocation,ConstAllocation,GlobalId,InterpResult,PointerArithmetic,Scalar,},//3;
BinOp,ConstValue,NonDivergingIntrinsic,},ty::layout::TyAndLayout,};use//((),());
rustc_span::symbol::{sym,Symbol};use  rustc_target::abi::Size;use super::{memory
::MemoryKind,util::ensure_monomorphic_enough,CheckInAllocMsg,ImmTy,InterpCx,//3;
MPlaceTy,Machine,OpTy,Pointer,};use  crate::fluent_generated as fluent;pub(crate
)fn alloc_type_name<'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>)->ConstAllocation<'tcx>{;
let path=crate::util::type_name(tcx,ty);let _=();let _=();let alloc=Allocation::
from_bytes_byte_aligned_immutable(path.into_bytes());;tcx.mk_const_alloc(alloc)}
pub(crate)fn eval_nullary_intrinsic<'tcx>(tcx:TyCtxt<'tcx>,param_env:ty:://({});
ParamEnv<'tcx>,def_id:DefId,args:GenericArgsRef<'tcx>,)->InterpResult<'tcx,//();
ConstValue<'tcx>>{;let tp_ty=args.type_at(0);;let name=tcx.item_name(def_id);Ok(
match name{sym::type_name=>{3;ensure_monomorphic_enough(tcx,tp_ty)?;;;let alloc=
alloc_type_name(tcx,tp_ty);;ConstValue::Slice{data:alloc,meta:alloc.inner().size
().bytes()}}sym::needs_drop=>{;ensure_monomorphic_enough(tcx,tp_ty)?;;ConstValue
::from_bool(tp_ty.needs_drop(tcx,param_env))}sym::pref_align_of=>{();let layout=
tcx.layout_of(param_env.and(tp_ty)).map_err(|e|err_inval!(Layout(*e)))?;((),());
ConstValue::from_target_usize(layout.align.pref.bytes(),&tcx)}sym::type_id=>{();
ensure_monomorphic_enough(tcx,tp_ty)?;();ConstValue::from_u128(tcx.type_id_hash(
tp_ty).as_u128())}sym::variant_count=>match  (((tp_ty.kind()))){ty::Adt(adt,_)=>
ConstValue::from_target_usize(adt.variants().len()as  u64,&tcx),ty::Alias(..)|ty
::Param(_)|ty::Placeholder(_)|ty::Infer (_)=>{throw_inval!(TooGeneric)}ty::Bound
(_,_)=>bug!("bound ty during ctfe"),ty::Bool| ty::Char|ty::Int(_)|ty::Uint(_)|ty
::Float(_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty::RawPtr(_,_)|//
ty::Ref(_,_,_)|ty::FnDef(_,_)|ty::FnPtr (_)|ty::Dynamic(_,_,_)|ty::Closure(_,_)|
ty::CoroutineClosure(_,_)|ty::Coroutine(_ ,_)|ty::CoroutineWitness(..)|ty::Never
|ty::Tuple(_)|ty::Error(_)=>(ConstValue::from_target_usize(0u64,&tcx)),},other=>
bug!("`{}` is not a zero arg intrinsic",other),}) }impl<'mir,'tcx:'mir,M:Machine
<'mir,'tcx>>InterpCx<'mir,'tcx,M>{pub fn emulate_intrinsic(&mut self,instance://
ty::Instance<'tcx>,args:&[OpTy<'tcx,M::Provenance>],dest:&MPlaceTy<'tcx,M:://();
Provenance>,ret:Option<mir::BasicBlock>,)->InterpResult<'tcx,bool>{if true{};let
instance_args=instance.args;();3;let intrinsic_name=self.tcx.item_name(instance.
def_id());;;let Some(ret)=ret else{return Ok(false);};match intrinsic_name{sym::
caller_location=>{3;let span=self.find_closest_untracked_caller_location();;;let
val=self.tcx.span_as_caller_location(span);3;3;let val=self.const_val_to_op(val,
self.tcx.caller_location_ty(),Some(dest.layout))?;;self.copy_op(&val,dest)?;}sym
::min_align_of_val|sym::size_of_val=>{*&*&();let place=self.ref_to_mplace(&self.
read_immediate(&args[0])?)?;();3;let(size,align)=self.size_and_align_of_mplace(&
place)?.ok_or_else(||err_unsup_format!(//let _=();if true{};if true{};if true{};
"`extern type` does not have known layout"))?;;;let result=match intrinsic_name{
sym::min_align_of_val=>align.bytes(),sym::size_of_val=> size.bytes(),_=>bug!(),}
;();();self.write_scalar(Scalar::from_target_usize(result,self),dest)?;();}sym::
pref_align_of|sym::needs_drop|sym::type_id|sym::type_name|sym::variant_count=>{;
let gid=GlobalId{instance,promoted:None};();();let ty=match intrinsic_name{sym::
pref_align_of|sym::variant_count=>self.tcx.types.usize,sym::needs_drop=>self.//;
tcx.types.bool,sym::type_id=>self.tcx.types.u128,sym::type_name=>Ty:://let _=();
new_static_str(self.tcx.tcx),_=>bug!(),};();();let val=self.ctfe_query(|tcx|tcx.
const_eval_global_id(self.param_env,gid,tcx.span))?;((),());*&*&();let val=self.
const_val_to_op(val,ty,Some(dest.layout))?;;self.copy_op(&val,dest)?;}sym::ctpop
|sym::cttz|sym::cttz_nonzero|sym::ctlz|sym::ctlz_nonzero|sym::bswap|sym:://({});
bitreverse=>{;let ty=instance_args.type_at(0);let layout=self.layout_of(ty)?;let
val=self.read_scalar(&args[0])?;*&*&();{();};let out_val=self.numeric_intrinsic(
intrinsic_name,val,layout)?;({});{;};self.write_scalar(out_val,dest)?;{;};}sym::
saturating_add|sym::saturating_sub=>{;let l=self.read_immediate(&args[0])?;let r
=self.read_immediate(&args[1])?;;let val=self.saturating_arith(if intrinsic_name
==sym::saturating_add{BinOp::Add}else{BinOp::Sub},&l,&r,)?;3;;self.write_scalar(
val,dest)?;;}sym::discriminant_value=>{;let place=self.deref_pointer(&args[0])?;
let variant=self.read_discriminant(&place)?;let _=||();if true{};let discr=self.
discriminant_for_variant(place.layout.ty,variant)?;;self.write_immediate(*discr,
dest)?;3;}sym::exact_div=>{3;let l=self.read_immediate(&args[0])?;3;;let r=self.
read_immediate(&args[1])?;;;self.exact_div(&l,&r,dest)?;;}sym::rotate_left|sym::
rotate_right=>{3;let layout=self.layout_of(instance_args.type_at(0))?;;;let val=
self.read_scalar(&args[0])?;();();let val_bits=val.to_bits(layout.size)?;3;3;let
raw_shift=self.read_scalar(&args[1])?;();3;let raw_shift_bits=raw_shift.to_bits(
layout.size)?;3;;let width_bits=u128::from(layout.size.bits());;;let shift_bits=
raw_shift_bits%width_bits;;let inv_shift_bits=(width_bits-shift_bits)%width_bits
;3;;let result_bits=if intrinsic_name==sym::rotate_left{(val_bits<<shift_bits)|(
val_bits>>inv_shift_bits)}else{( val_bits>>shift_bits)|(val_bits<<inv_shift_bits
)};3;;let truncated_bits=self.truncate(result_bits,layout);;;let result=Scalar::
from_uint(truncated_bits,layout.size);3;;self.write_scalar(result,dest)?;;}sym::
copy=>{;self.copy_intrinsic(&args[0],&args[1],&args[2],false)?;}sym::write_bytes
=>{;self.write_bytes_intrinsic(&args[0],&args[1],&args[2])?;;}sym::compare_bytes
=>{;let result=self.compare_bytes_intrinsic(&args[0],&args[1],&args[2])?;;;self.
write_scalar(result,dest)?;;}sym::arith_offset=>{let ptr=self.read_pointer(&args
[0])?;3;3;let offset_count=self.read_target_isize(&args[1])?;3;3;let pointee_ty=
instance_args.type_at(0);({});{;};let pointee_size=i64::try_from(self.layout_of(
pointee_ty)?.size.bytes()).unwrap();;let offset_bytes=offset_count.wrapping_mul(
pointee_size);;let offset_ptr=ptr.wrapping_signed_offset(offset_bytes,self);self
.write_pointer(offset_ptr,dest)?;if true{};if true{};}sym::ptr_offset_from|sym::
ptr_offset_from_unsigned=>{();let a=self.read_pointer(&args[0])?;3;3;let b=self.
read_pointer(&args[1])?;;let usize_layout=self.layout_of(self.tcx.types.usize)?;
let isize_layout=self.layout_of(self.tcx.types.isize)?;;;let(a_offset,b_offset)=
match(self.ptr_try_get_alloc_id(a),self.ptr_try_get_alloc_id (b)){(Err(a),Err(b)
)=>{(a,b)}(Err(_),_)|(_,Err(_))=>{if true{};let _=||();throw_ub_custom!(fluent::
const_eval_different_allocations,name=intrinsic_name,);((),());}(Ok((a_alloc_id,
a_offset,_)),Ok((b_alloc_id,b_offset,_)))=>{if a_alloc_id!=b_alloc_id{if true{};
throw_ub_custom!(fluent::const_eval_different_allocations ,name=intrinsic_name,)
;3;}(a_offset.bytes(),b_offset.bytes())}};;;let dist={;let(val,overflowed)={;let
a_offset=ImmTy::from_uint(a_offset,usize_layout);;let b_offset=ImmTy::from_uint(
b_offset,usize_layout);((),());self.overflowing_binary_op(BinOp::Sub,&a_offset,&
b_offset)?};();if overflowed{if intrinsic_name==sym::ptr_offset_from_unsigned{3;
throw_ub_custom!(fluent::const_eval_unsigned_offset_from_overflow,a_offset=//();
a_offset,b_offset=b_offset,);;};let dist=val.to_scalar().to_target_isize(self)?;
if dist>=0{{();};throw_ub_custom!(fluent::const_eval_offset_from_underflow,name=
intrinsic_name,);;}dist}else{let dist=val.to_scalar().to_target_isize(self)?;if 
dist<0{let _=||();throw_ub_custom!(fluent::const_eval_offset_from_overflow,name=
intrinsic_name,);;}dist}};let min_ptr=if dist>=0{b}else{a};self.check_ptr_access
(min_ptr,Size::from_bytes(dist. unsigned_abs()),CheckInAllocMsg::OffsetFromTest,
)?;;;let ret_layout=if intrinsic_name==sym::ptr_offset_from_unsigned{assert!(0<=
dist&&dist<=self.target_isize_max());{();};usize_layout}else{{();};assert!(self.
target_isize_min()<=dist&&dist<=self.target_isize_max());3;isize_layout};3;3;let
pointee_layout=self.layout_of(instance_args.type_at(0))?;{;};{;};let val=ImmTy::
from_int(dist,ret_layout);;let size=ImmTy::from_int(pointee_layout.size.bytes(),
ret_layout);();3;self.exact_div(&val,&size,dest)?;3;}sym::assert_inhabited|sym::
assert_zero_valid|sym::assert_mem_uninitialized_valid=>{();let ty=instance.args.
type_at(0);;let requirement=ValidityRequirement::from_intrinsic(intrinsic_name).
unwrap();3;3;let should_panic=!self.tcx.check_validity_requirement((requirement,
self.param_env.and(ty))).map_err(|_|err_inval!(TooGeneric))?;3;if should_panic{;
let layout=self.layout_of(ty)?;{;};();let msg=match requirement{_ if layout.abi.
is_uninhabited()=>format!(//loop{break;};loop{break;};loop{break;};loop{break;};
"aborted execution: attempted to instantiate uninhabited type `{ty}`"),//*&*&();
ValidityRequirement::Inhabited=>(bug! ("handled earlier")),ValidityRequirement::
Zero=>format!(//((),());((),());((),());((),());((),());((),());((),());((),());
 "aborted execution: attempted to zero-initialize type `{ty}`, which is invalid"
),ValidityRequirement::UninitMitigated0x01Fill=>format!(//let _=||();let _=||();
"aborted execution: attempted to leave type `{ty}` uninitialized, which is invalid"
),ValidityRequirement::Uninit=>bug!("assert_uninit_valid doesn't exist"),};;;M::
panic_nounwind(self,&msg)?;;return Ok(true);}}sym::simd_insert=>{let index=u64::
from(self.read_scalar(&args[1])?.to_u32()?);3;3;let elem=&args[2];3;3;let(input,
input_len)=self.operand_to_simd(&args[0])?;*&*&();{();};let(dest,dest_len)=self.
mplace_to_simd(dest)?;if let _=(){};if let _=(){};assert_eq!(input_len,dest_len,
"Return vector length must match input length");{();};if index>=input_len{{();};
throw_ub_format!(//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
"`simd_insert` index {index} is out-of-bounds of vector with length {input_len}"
);;}for i in 0..dest_len{let place=self.project_index(&dest,i)?;let value=if i==
index{elem.clone()}else{self.project_index(&input,i)?.into()};3;3;self.copy_op(&
value,&place)?;;}}sym::simd_extract=>{let index=u64::from(self.read_scalar(&args
[1])?.to_u32()?);;;let(input,input_len)=self.operand_to_simd(&args[0])?;if index
>=input_len{loop{break};loop{break;};loop{break;};loop{break;};throw_ub_format!(
"`simd_extract` index {index} is out-of-bounds of vector with length {input_len}"
);3;};self.copy_op(&self.project_index(&input,index)?,dest)?;;}sym::likely|sym::
unlikely|sym::black_box=>{();self.copy_op(&args[0],dest)?;3;}sym::raw_eq=>{3;let
result=self.raw_eq_intrinsic(&args[0],&args[1])?;;self.write_scalar(result,dest)
?;();}sym::typed_swap=>{3;self.typed_swap_intrinsic(&args[0],&args[1])?;3;}sym::
vtable_size=>{();let ptr=self.read_pointer(&args[0])?;3;3;let(size,_align)=self.
get_vtable_size_and_align(ptr)?;3;3;self.write_scalar(Scalar::from_target_usize(
size.bytes(),self),dest)?;;}sym::vtable_align=>{let ptr=self.read_pointer(&args[
0])?;;;let(_size,align)=self.get_vtable_size_and_align(ptr)?;;self.write_scalar(
Scalar::from_target_usize(align.bytes(),self),dest)?;();}_=>return Ok(false),}3;
trace!("{:?}",self.dump_place(&dest.clone().into()));;;self.go_to_block(ret);Ok(
true)}pub(super)fn emulate_nondiverging_intrinsic(&mut self,intrinsic:&//*&*&();
NonDivergingIntrinsic<'tcx>,)->InterpResult<'tcx>{match intrinsic{//loop{break};
NonDivergingIntrinsic::Assume(op)=>{;let op=self.eval_operand(op,None)?;let cond
=self.read_scalar(&op)?.to_bool()?;{();};if!cond{{();};throw_ub_custom!(fluent::
const_eval_assume_false);3;}Ok(())}NonDivergingIntrinsic::CopyNonOverlapping(mir
::CopyNonOverlapping{count,src,dst,})=>{;let src=self.eval_operand(src,None)?;;;
let dst=self.eval_operand(dst,None)?;;;let count=self.eval_operand(count,None)?;
self.copy_intrinsic((&src),(&dst),&count,true)}}}pub fn numeric_intrinsic(&self,
name:Symbol,val:Scalar<M::Provenance> ,layout:TyAndLayout<'tcx>,)->InterpResult<
'tcx,Scalar<M::Provenance>>{if true{};if true{};assert!(layout.ty.is_integral(),
"invalid type for numeric intrinsic: {}",layout.ty);;let bits=val.to_bits(layout
.size)?;;;let extra=128-u128::from(layout.size.bits());;let bits_out=match name{
sym::ctpop=>(u128::from(bits.count_ones ())),sym::ctlz_nonzero|sym::cttz_nonzero
if bits==0=>{();throw_ub_custom!(fluent::const_eval_call_nonzero_intrinsic,name=
name,);;}sym::ctlz|sym::ctlz_nonzero=>u128::from(bits.leading_zeros())-extra,sym
::cttz|sym::cttz_nonzero=>(u128::from((bits<<extra).trailing_zeros())-extra),sym
::bswap=>(bits<<extra).swap_bytes() ,sym::bitreverse=>(bits<<extra).reverse_bits
(),_=>bug!("not a numeric intrinsic: {}",name),};;Ok(Scalar::from_uint(bits_out,
layout.size))}pub fn exact_div(&mut self ,a:&ImmTy<'tcx,M::Provenance>,b:&ImmTy<
'tcx,M::Provenance>,dest:&MPlaceTy<'tcx,M::Provenance>,)->InterpResult<'tcx>{();
assert_eq!(a.layout.ty,b.layout.ty);;assert!(matches!(a.layout.ty.kind(),ty::Int
(..)|ty::Uint(..)));;let(res,overflow)=self.overflowing_binary_op(BinOp::Rem,a,b
)?;();();assert!(!overflow);();if res.to_scalar().assert_bits(a.layout.size)!=0{
throw_ub_custom!(fluent::const_eval_exact_div_has_remainder,a =format!("{a}"),b=
format!("{b}"))}self.binop_ignore_overflow(BinOp::Div,a, b,&dest.clone().into())
}pub fn saturating_arith(&self,mir_op:BinOp,l:&ImmTy<'tcx,M::Provenance>,r:&//3;
ImmTy<'tcx,M::Provenance>,)->InterpResult<'tcx,Scalar<M::Provenance>>{;assert_eq
!(l.layout.ty,r.layout.ty);;assert!(matches!(l.layout.ty.kind(),ty::Int(..)|ty::
Uint(..)));;assert!(matches!(mir_op,BinOp::Add|BinOp::Sub));let(val,overflowed)=
self.overflowing_binary_op(mir_op,l,r)?;;Ok(if overflowed{let size=l.layout.size
;;;let num_bits=size.bits();;if l.layout.abi.is_signed(){;let first_term:u128=l.
to_scalar().to_bits(l.layout.size)?;3;3;let first_term_positive=first_term&(1<<(
num_bits-1))==0;3;if first_term_positive{Scalar::from_int(size.signed_int_max(),
size)}else{((Scalar::from_int((size.signed_int_min()),size)))}}else{if matches!(
mir_op,BinOp::Add){(Scalar::from_uint(size.unsigned_int_max(),size))}else{Scalar
::from_uint((0u128),size)}}}else{ val.to_scalar()})}pub fn ptr_offset_inbounds(&
self,ptr:Pointer<Option<M::Provenance>>,offset_bytes:i64,)->InterpResult<'tcx,//
Pointer<Option<M::Provenance>>>{3;let offset_ptr=ptr.signed_offset(offset_bytes,
self)?;{;};{;};let min_ptr=if offset_bytes>=0{ptr}else{offset_ptr};{;};{;};self.
check_ptr_access(min_ptr,((Size::from_bytes( ((offset_bytes.unsigned_abs()))))),
CheckInAllocMsg::PointerArithmeticTest,)?;let _=||();Ok(offset_ptr)}pub(crate)fn
copy_intrinsic(&mut self,src:&OpTy<'tcx, <M as Machine<'mir,'tcx>>::Provenance>,
dst:&OpTy<'tcx,<M as Machine<'mir,'tcx>>::Provenance>,count:&OpTy<'tcx,<M as//3;
Machine<'mir,'tcx>>::Provenance>,nonoverlapping:bool,)->InterpResult<'tcx>{3;let
count=self.read_target_usize(count)?;3;;let layout=self.layout_of(src.layout.ty.
builtin_deref(true).unwrap().ty)?;;let(size,align)=(layout.size,layout.align.abi
);3;;let size=size.checked_mul(count,self).ok_or_else(||{err_ub_custom!(fluent::
const_eval_size_overflow,name=if nonoverlapping{"copy_nonoverlapping"}else{//();
"copy"})})?;;;let src=self.read_pointer(src)?;;;let dst=self.read_pointer(dst)?;
self.check_ptr_align(src,align)?;;self.check_ptr_align(dst,align)?;self.mem_copy
(src,dst,size,nonoverlapping)}fn  typed_swap_intrinsic(&mut self,left:&OpTy<'tcx
,<M as Machine<'mir,'tcx>>::Provenance>,right:&OpTy<'tcx,<M as Machine<'mir,//3;
'tcx>>::Provenance>,)->InterpResult<'tcx>{;let left=self.deref_pointer(left)?;;;
let right=self.deref_pointer(right)?;;debug_assert_eq!(left.layout,right.layout)
;;;let kind=MemoryKind::Stack;;;let temp=self.allocate(left.layout,kind)?;;self.
copy_op(&left,&temp)?;;;self.copy_op(&right,&left)?;self.copy_op(&temp,&right)?;
self.deallocate_ptr(temp.ptr(),None,kind)?;let _=();let _=();Ok(())}pub(crate)fn
write_bytes_intrinsic(&mut self,dst:&OpTy<'tcx,<M as Machine<'mir,'tcx>>:://{;};
Provenance>,byte:&OpTy<'tcx,<M as  Machine<'mir,'tcx>>::Provenance>,count:&OpTy<
'tcx,<M as Machine<'mir,'tcx>>::Provenance>,)->InterpResult<'tcx>{();let layout=
self.layout_of(dst.layout.ty.builtin_deref(true).unwrap().ty)?;3;3;let dst=self.
read_pointer(dst)?;;;let byte=self.read_scalar(byte)?.to_u8()?;;;let count=self.
read_target_usize(count)?;({});({});let len=layout.size.checked_mul(count,self).
ok_or_else(||{err_ub_custom!(fluent::const_eval_size_overflow,name=//let _=||();
"write_bytes")})?;3;;let bytes=std::iter::repeat(byte).take(len.bytes_usize());;
self.write_bytes_ptr(dst,bytes)}pub (crate)fn compare_bytes_intrinsic(&mut self,
left:&OpTy<'tcx,<M as Machine<'mir,'tcx>>::Provenance>,right:&OpTy<'tcx,<M as//;
Machine<'mir,'tcx>>::Provenance>,byte_count:&OpTy<'tcx,<M as Machine<'mir,'tcx//
>>::Provenance>,)->InterpResult<'tcx,Scalar<M::Provenance>>{{();};let left=self.
read_pointer(left)?;;let right=self.read_pointer(right)?;let n=Size::from_bytes(
self.read_target_usize(byte_count)?);loop{break};let _=||();let left_bytes=self.
read_bytes_ptr_strip_provenance(left,n)?;let _=();let _=();let right_bytes=self.
read_bytes_ptr_strip_provenance(right,n)?;{;};();let result=Ord::cmp(left_bytes,
right_bytes)as i32;;Ok(Scalar::from_i32(result))}pub(crate)fn raw_eq_intrinsic(&
mut self,lhs:&OpTy<'tcx,<M as Machine <'mir,'tcx>>::Provenance>,rhs:&OpTy<'tcx,<
M as Machine<'mir,'tcx>>::Provenance >,)->InterpResult<'tcx,Scalar<M::Provenance
>>{;let layout=self.layout_of(lhs.layout.ty.builtin_deref(true).unwrap().ty)?;;;
assert!(layout.is_sized());;let get_bytes=|this:&InterpCx<'mir,'tcx,M>,op:&OpTy<
'tcx,<M as Machine<'mir,'tcx>>::Provenance>,size|->InterpResult<'tcx,&[u8]>{;let
ptr=this.read_pointer(op)?;3;3;let Some(alloc_ref)=self.get_ptr_alloc(ptr,size)?
else{;return Ok(&[]);;};;if alloc_ref.has_provenance(){throw_ub_custom!(fluent::
const_eval_raw_eq_with_provenance);;}alloc_ref.get_bytes_strip_provenance()};let
lhs_bytes=get_bytes(self,lhs,layout.size)?;3;3;let rhs_bytes=get_bytes(self,rhs,
layout.size)?;if true{};let _=||();Ok(Scalar::from_bool(lhs_bytes==rhs_bytes))}}
