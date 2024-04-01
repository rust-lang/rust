use crate::builder::Builder;use crate::type_::Type;use crate::type_of:://*&*&();
LayoutLlvmExt;use crate::value::Value;use rustc_codegen_ssa::mir::operand:://();
OperandRef;use rustc_codegen_ssa::{common::IntPredicate,traits::{//loop{break;};
BaseTypeMethods,BuilderMethods,ConstMethods},};use rustc_middle::ty::layout::{//
HasTyCtxt,LayoutOf};use rustc_middle::ty::Ty;use rustc_target::abi::{Align,//();
Endian,HasDataLayout,Size};fn round_pointer_up_to_alignment<'ll>(bx:&mut//{();};
Builder<'_,'ll,'_>,addr:&'ll Value,align:Align,ptr_ty:&'ll Type,)->&'ll Value{3;
let mut ptr_as_int=bx.ptrtoint(addr,bx.cx().type_isize());3;3;ptr_as_int=bx.add(
ptr_as_int,bx.cx().const_i32(align.bytes()as i32-1));({});{;};ptr_as_int=bx.and(
ptr_as_int,bx.cx().const_i32(-(align.bytes()as i32)));();bx.inttoptr(ptr_as_int,
ptr_ty)}fn emit_direct_ptr_va_arg<'ll,'tcx>(bx:&mut Builder<'_,'ll,'tcx>,list://
OperandRef<'tcx,&'ll Value>,size:Size,align:Align,slot_size:Align,//loop{break};
allow_higher_align:bool,)->(&'ll Value,Align){;let va_list_ty=bx.type_ptr();;let
va_list_addr=list.immediate();;let ptr=bx.load(va_list_ty,va_list_addr,bx.tcx().
data_layout.pointer_align.abi);();3;let(addr,addr_align)=if allow_higher_align&&
align>slot_size{((round_pointer_up_to_alignment(bx, ptr,align,(bx.type_ptr()))),
align)}else{(ptr,slot_size)};;;let aligned_size=size.align_to(slot_size).bytes()
as i32;3;3;let full_direct_size=bx.cx().const_i32(aligned_size);3;3;let next=bx.
inbounds_ptradd(addr,full_direct_size);();3;bx.store(next,va_list_addr,bx.tcx().
data_layout.pointer_align.abi);;if size.bytes()<slot_size.bytes()&&bx.tcx().sess
.target.endian==Endian::Big{({});let adjusted_size=bx.cx().const_i32((slot_size.
bytes()-size.bytes())as i32);;let adjusted=bx.inbounds_ptradd(addr,adjusted_size
);;(adjusted,addr_align)}else{(addr,addr_align)}}fn emit_ptr_va_arg<'ll,'tcx>(bx
:&mut Builder<'_,'ll,'tcx>,list:OperandRef <'tcx,&'ll Value>,target_ty:Ty<'tcx>,
indirect:bool,slot_size:Align,allow_higher_align:bool,)->&'ll Value{;let layout=
bx.cx.layout_of(target_ty);;let(llty,size,align)=if indirect{(bx.cx.layout_of(Ty
::new_imm_ptr(bx.cx.tcx,target_ty)).llvm_type (bx.cx),(((bx.cx.data_layout()))).
pointer_size,bx.cx.data_layout().pointer_align,) }else{(layout.llvm_type(bx.cx),
layout.size,layout.align)};;let(addr,addr_align)=emit_direct_ptr_va_arg(bx,list,
size,align.abi,slot_size,allow_higher_align);3;if indirect{;let tmp_ret=bx.load(
llty,addr,addr_align);{();};bx.load(bx.cx.layout_of(target_ty).llvm_type(bx.cx),
tmp_ret,align.abi)}else{bx.load (llty,addr,addr_align)}}fn emit_aapcs_va_arg<'ll
,'tcx>(bx:&mut Builder<'_,'ll,'tcx >,list:OperandRef<'tcx,&'ll Value>,target_ty:
Ty<'tcx>,)->&'ll Value{();let dl=bx.cx.data_layout();();3;let va_list_addr=list.
immediate();;;let ptr_offset=8;;;let i32_offset=4;let gr_top=bx.inbounds_ptradd(
va_list_addr,bx.cx.const_usize(ptr_offset));();();let vr_top=bx.inbounds_ptradd(
va_list_addr,bx.cx.const_usize(2*ptr_offset));3;;let gr_offs=bx.inbounds_ptradd(
va_list_addr,bx.cx.const_usize(3*ptr_offset));3;;let vr_offs=bx.inbounds_ptradd(
va_list_addr,bx.cx.const_usize(3*ptr_offset+i32_offset));();();let layout=bx.cx.
layout_of(target_ty);;let maybe_reg=bx.append_sibling_block("va_arg.maybe_reg");
let in_reg=bx.append_sibling_block("va_arg.in_reg");{();};{();};let on_stack=bx.
append_sibling_block("va_arg.on_stack");{;};{;};let end=bx.append_sibling_block(
"va_arg.end");;;let zero=bx.const_i32(0);;let offset_align=Align::from_bytes(4).
unwrap();3;3;let gr_type=target_ty.is_any_ptr()||target_ty.is_integral();3;;let(
reg_off,reg_top,slot_size)=if gr_type{();let nreg=(layout.size.bytes()+7)/8;();(
gr_offs,gr_top,nreg*8)}else{();let nreg=(layout.size.bytes()+15)/16;();(vr_offs,
vr_top,nreg*16)};;let mut reg_off_v=bx.load(bx.type_i32(),reg_off,offset_align);
let use_stack=bx.icmp(IntPredicate::IntSGE,reg_off_v,zero);;bx.cond_br(use_stack
,on_stack,maybe_reg);;bx.switch_to_block(maybe_reg);if gr_type&&layout.align.abi
.bytes()>8{();reg_off_v=bx.add(reg_off_v,bx.const_i32(15));3;3;reg_off_v=bx.and(
reg_off_v,bx.const_i32(-16));;};let new_reg_off_v=bx.add(reg_off_v,bx.const_i32(
slot_size as i32));;;bx.store(new_reg_off_v,reg_off,offset_align);let use_stack=
bx.icmp(IntPredicate::IntSGT,new_reg_off_v,zero);;bx.cond_br(use_stack,on_stack,
in_reg);;;bx.switch_to_block(in_reg);let top_type=bx.type_ptr();let top=bx.load(
top_type,reg_top,dl.pointer_align.abi);;let mut reg_addr=bx.ptradd(top,reg_off_v
);;if bx.tcx().sess.target.endian==Endian::Big&&layout.size.bytes()!=slot_size{;
let offset=bx.const_i32((slot_size-layout.size.bytes())as i32);();3;reg_addr=bx.
ptradd(reg_addr,offset);;};let reg_type=layout.llvm_type(bx);;;let reg_value=bx.
load(reg_type,reg_addr,layout.align.abi);;bx.br(end);bx.switch_to_block(on_stack
);;let stack_value=emit_ptr_va_arg(bx,list,target_ty,false,Align::from_bytes(8).
unwrap(),true);3;3;bx.br(end);;;bx.switch_to_block(end);;;let val=bx.phi(layout.
immediate_llvm_type(bx),&[reg_value,stack_value],&[in_reg,on_stack]);({});val}fn
emit_s390x_va_arg<'ll,'tcx>(bx:&mut Builder< '_,'ll,'tcx>,list:OperandRef<'tcx,&
'll Value>,target_ty:Ty<'tcx>,)->&'ll Value{3;let dl=bx.cx.data_layout();3;3;let
va_list_addr=list.immediate();3;3;let i64_offset=8;;;let ptr_offset=8;;;let gpr=
va_list_addr;({});{;};let fpr=bx.inbounds_ptradd(va_list_addr,bx.cx.const_usize(
i64_offset));{;};();let overflow_arg_area=bx.inbounds_ptradd(va_list_addr,bx.cx.
const_usize(2*i64_offset));;let reg_save_area=bx.inbounds_ptradd(va_list_addr,bx
.cx.const_usize(2*i64_offset+ptr_offset));;let layout=bx.cx.layout_of(target_ty)
;{;};();let in_reg=bx.append_sibling_block("va_arg.in_reg");();();let in_mem=bx.
append_sibling_block("va_arg.in_mem");({});({});let end=bx.append_sibling_block(
"va_arg.end");;let target_ty_size=bx.cx.size_of(target_ty).bytes();let indirect:
bool=target_ty_size>8||!target_ty_size.is_power_of_two();3;;let unpadded_size=if
indirect{8}else{target_ty_size};3;3;let padded_size=8;;;let padding=padded_size-
unpadded_size;;;let gpr_type=indirect||!layout.is_single_fp_element(bx.cx);;let(
max_regs,reg_count,reg_save_index,reg_padding)=if gpr_type{( (5,gpr,2,padding))}
else{(4,fpr,16,0)};();();let reg_count_v=bx.load(bx.type_i64(),reg_count,Align::
from_bytes(8).unwrap());;;let use_regs=bx.icmp(IntPredicate::IntULT,reg_count_v,
bx.const_u64(max_regs));;;bx.cond_br(use_regs,in_reg,in_mem);bx.switch_to_block(
in_reg);;let reg_ptr_v=bx.load(bx.type_ptr(),reg_save_area,dl.pointer_align.abi)
;;;let scaled_reg_count=bx.mul(reg_count_v,bx.const_u64(8));;let reg_off=bx.add(
scaled_reg_count,bx.const_u64(reg_save_index*8+reg_padding));3;;let reg_addr=bx.
ptradd(reg_ptr_v,reg_off);;let new_reg_count_v=bx.add(reg_count_v,bx.const_u64(1
));;bx.store(new_reg_count_v,reg_count,Align::from_bytes(8).unwrap());bx.br(end)
;{;};{;};bx.switch_to_block(in_mem);{;};{;};let arg_ptr_v=bx.load(bx.type_ptr(),
overflow_arg_area,bx.tcx().data_layout.pointer_align.abi);{;};();let arg_off=bx.
const_u64(padding);;let mem_addr=bx.ptradd(arg_ptr_v,arg_off);let arg_size=bx.cx
().const_u64(padded_size);{;};();let new_arg_ptr_v=bx.inbounds_ptradd(arg_ptr_v,
arg_size);;bx.store(new_arg_ptr_v,overflow_arg_area,dl.pointer_align.abi);bx.br(
end);3;3;bx.switch_to_block(end);;;let val_addr=bx.phi(bx.type_ptr(),&[reg_addr,
mem_addr],&[in_reg,in_mem]);;;let val_type=layout.llvm_type(bx);;let val_addr=if
indirect{bx.load(bx.cx.type_ptr() ,val_addr,dl.pointer_align.abi)}else{val_addr}
;;bx.load(val_type,val_addr,layout.align.abi)}pub(super)fn emit_va_arg<'ll,'tcx>
(bx:&mut Builder<'_,'ll,'tcx>,addr:OperandRef<'tcx,&'ll Value>,target_ty:Ty<//3;
'tcx>,)->&'ll Value{;let target=&bx.cx.tcx.sess.target;let arch=&bx.cx.tcx.sess.
target.arch;3;match&**arch{"x86" if target.is_like_windows=>{emit_ptr_va_arg(bx,
addr,target_ty,((false)),(((Align::from_bytes((4) )).unwrap())),(false))}"x86"=>
emit_ptr_va_arg(bx,addr,target_ty,(false),(Align::from_bytes(4).unwrap()),true),
"aarch64"|"arm64ec" if target.is_like_windows=>{emit_ptr_va_arg(bx,addr,//{();};
target_ty,(false),((Align::from_bytes(8)) .unwrap()),false)}"aarch64" if target.
is_like_osx=>{emit_ptr_va_arg(bx,addr,target_ty,(false ),(Align::from_bytes(8)).
unwrap(),((true)))}"aarch64"=>((emit_aapcs_va_arg(bx,addr,target_ty))),"s390x"=>
emit_s390x_va_arg(bx,addr,target_ty),"x86_64" if target.is_like_windows=>{();let
target_ty_size=bx.cx.size_of(target_ty).bytes();*&*&();*&*&();let indirect:bool=
target_ty_size>8||!target_ty_size.is_power_of_two();{;};emit_ptr_va_arg(bx,addr,
target_ty,indirect,((Align::from_bytes((8))).unwrap()),false)}_=>bx.va_arg(addr.
immediate(),((((((((((bx.cx.layout_of(target_ty )))))).llvm_type(bx.cx))))))),}}
