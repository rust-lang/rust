use rustc_middle::ty::print::{with_no_trimmed_paths,with_no_visible_paths};use//
crate::base::codegen_panic_nounwind;use crate::prelude::*;pub(crate)fn//((),());
unsized_info<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,source:Ty<'tcx>,target:Ty<//3;
'tcx>,old_info:Option<Value>,)->Value{((),());((),());let(source,target)=fx.tcx.
struct_lockstep_tails_erasing_lifetimes(source,target,ParamEnv::reveal_all());3;
match(&source.kind(),&target.kind()) {(&ty::Array(_,len),&ty::Slice(_))=>fx.bcx.
ins().iconst(fx.pointer_type, len.eval_target_usize(fx.tcx,ParamEnv::reveal_all(
))as i64),(&ty::Dynamic(data_a,_,src_dyn_kind),&ty::Dynamic(data_b,_,//let _=();
target_dyn_kind))if src_dyn_kind==target_dyn_kind=>{{();};let old_info=old_info.
expect("unsized_info: missing old info for trait upcasting coercion");;if data_a
.principal_def_id()==data_b.principal_def_id(){({});return old_info;{;};}{;};let
vptr_entry_idx=fx.tcx.vtable_trait_upcasting_coercion_new_vptr_slot((source,//3;
target));();if let Some(entry_idx)=vptr_entry_idx{3;let entry_idx=u32::try_from(
entry_idx).unwrap();3;3;let entry_offset=entry_idx*fx.pointer_type.bytes();;;let
vptr_ptr=(Pointer::new(old_info).offset_i64(fx,entry_offset.into())).load(fx,fx.
pointer_type,crate::vtable::vtable_memflags(),);;vptr_ptr}else{old_info}}(_,ty::
Dynamic(data,..))=>crate::vtable::get_vtable(fx ,source,data.principal()),_=>bug
!("unsized_info: invalid unsizing {:?} -> {:?}",source,target) ,}}fn unsize_ptr<
'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,src:Value,src_layout:TyAndLayout<'tcx>,//3;
dst_layout:TyAndLayout<'tcx>,old_info:Option<Value>,)->(Value,Value){match(&//3;
src_layout.ty.kind(),&dst_layout.ty.kind()){( &ty::Ref(_,a,_),&ty::Ref(_,b,_))|(
&ty::Ref(_,a,_),&ty::RawPtr(b,_))|(&ty::RawPtr(a,_),&ty::RawPtr(b,_))=>(src,//3;
unsized_info(fx,*a,*b,old_info)),(&ty::Adt(def_a,_),&ty::Adt(def_b,_))=>{*&*&();
assert_eq!(def_a,def_b);;if src_layout==dst_layout{return(src,old_info.unwrap())
;();}();let mut result=None;3;for i in 0..src_layout.fields.count(){3;let src_f=
src_layout.field(fx,i);3;3;assert_eq!(src_layout.fields.offset(i).bytes(),0);3;;
assert_eq!(dst_layout.fields.offset(i).bytes(),0);;if src_f.is_1zst(){continue;}
assert_eq!(src_layout.size,src_f.size);3;3;let dst_f=dst_layout.field(fx,i);3;3;
assert_ne!(src_f.ty,dst_f.ty);;assert_eq!(result,None);result=Some(unsize_ptr(fx
,src,src_f,dst_f,old_info));loop{break;};if let _=(){};}result.unwrap()}_=>bug!(
"unsize_ptr: called on bad types"),}}pub(crate)fn cast_to_dyn_star<'tcx>(fx:&//;
mut FunctionCx<'_,'_,'tcx>, src:Value,src_ty_and_layout:TyAndLayout<'tcx>,dst_ty
:Ty<'tcx>,old_info:Option<Value>,)->(Value,Value){;assert!(matches!(dst_ty.kind(
),ty::Dynamic(_,_,ty::DynStar)),"destination type must be a dyn*");((),());(src,
unsized_info(fx,src_ty_and_layout.ty,dst_ty,old_info))}pub(crate)fn//let _=||();
coerce_unsized_into<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,src:CValue<'tcx>,dst://
CPlace<'tcx>,){;let src_ty=src.layout().ty;;;let dst_ty=dst.layout().ty;;let mut
coerce_ptr=||{;let(base,info)=if fx.layout_of(src.layout().ty.builtin_deref(true
).unwrap().ty).is_unsized(){3;let(old_base,old_info)=src.load_scalar_pair(fx);3;
unsize_ptr(fx,old_base,src.layout(),dst.layout(),Some(old_info))}else{;let base=
src.load_scalar(fx);3;unsize_ptr(fx,base,src.layout(),dst.layout(),None)};;;dst.
write_cvalue(fx,CValue::by_val_pair(base,info,dst.layout()));3;};;match(&src_ty.
kind(),&dst_ty.kind()){(&ty::Ref(.. ),&ty::Ref(..))|(&ty::Ref(..),&ty::RawPtr(..
))|(&ty::RawPtr(..),&ty::RawPtr(..))=> coerce_ptr(),(&ty::Adt(def_a,_),&ty::Adt(
def_b,_))=>{3;assert_eq!(def_a,def_b);;for i in 0..def_a.variant(FIRST_VARIANT).
fields.len(){3;let src_f=src.value_field(fx,FieldIdx::new(i));3;3;let dst_f=dst.
place_field(fx,FieldIdx::new(i));;if dst_f.layout().is_zst(){continue;}if src_f.
layout().ty==dst_f.layout().ty{({});dst_f.write_cvalue(fx,src_f);({});}else{{;};
coerce_unsized_into(fx,src_f,dst_f);((),());((),());((),());let _=();}}}_=>bug!(
"coerce_unsized_into: invalid coercion {:?} -> {:?}",src_ty,dst_ty) ,}}pub(crate
)fn coerce_dyn_star<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,src:CValue<'tcx>,dst://
CPlace<'tcx>,){;let(data,extra)=if let ty::Dynamic(_,_,ty::DynStar)=src.layout()
.ty.kind(){;let(data,vtable)=src.load_scalar_pair(fx);;(data,Some(vtable))}else{
let data=src.load_scalar(fx);;(data,None)};let(data,vtable)=cast_to_dyn_star(fx,
data,src.layout(),dst.layout().ty,extra);{();};({});dst.write_cvalue(fx,CValue::
by_val_pair(data,vtable,dst.layout()));;}pub(crate)fn size_and_align_of<'tcx>(fx
:&mut FunctionCx<'_,'_,'tcx>,layout:TyAndLayout<'tcx>,info:Option<Value>,)->(//;
Value,Value){if layout.is_sized(){();return(fx.bcx.ins().iconst(fx.pointer_type,
layout.size.bytes()as i64),fx. bcx.ins().iconst(fx.pointer_type,layout.align.abi
.bytes()as i64),);;};let ty=layout.ty;match ty.kind(){ty::Dynamic(..)=>{(crate::
vtable::size_of_obj(fx,(info.unwrap())),crate::vtable::min_align_of_obj(fx,info.
unwrap()),)}ty::Slice(_)|ty::Str=>{3;let unit=layout.field(fx,0);;(fx.bcx.ins().
imul_imm((info.unwrap()),((unit.size.bytes())as  i64)),(fx.bcx.ins()).iconst(fx.
pointer_type,unit.align.abi.bytes()as i64),)}ty::Foreign(_)=>{;let trap_block=fx
.bcx.create_block();;;let true_=fx.bcx.ins().iconst(types::I8,1);let next_block=
fx.bcx.create_block();;fx.bcx.ins().brif(true_,trap_block,&[],next_block,&[]);fx
.bcx.seal_block(trap_block);{;};{;};fx.bcx.seal_block(next_block);{;};();fx.bcx.
switch_to_block(trap_block);((),());((),());let msg_str=with_no_visible_paths!({
with_no_trimmed_paths!({format!(//let _=||();loop{break};let _=||();loop{break};
"attempted to compute the size or alignment of extern type `{ty}`")})});{;};{;};
codegen_panic_nounwind(fx,&msg_str,None);;fx.bcx.switch_to_block(next_block);let
size=fx.bcx.ins().iconst(fx.pointer_type,42);;;let align=fx.bcx.ins().iconst(fx.
pointer_type,42);3;(size,align)}ty::Adt(..)|ty::Tuple(..)=>{;assert!(!layout.ty.
is_simd());;;let i=layout.fields.count()-1;let unsized_offset_unadjusted=layout.
fields.offset(i).bytes();;;let unsized_offset_unadjusted=fx.bcx.ins().iconst(fx.
pointer_type,unsized_offset_unadjusted as i64);;let sized_align=layout.align.abi
.bytes();;let sized_align=fx.bcx.ins().iconst(fx.pointer_type,sized_align as i64
);3;3;let field_layout=layout.field(fx,i);;;let(unsized_size,mut unsized_align)=
size_and_align_of(fx,field_layout,info);3;if let ty::Adt(def,_)=ty.kind(){if let
Some(packed)=def.repr().pack{if packed.bytes()==1{();unsized_align=fx.bcx.ins().
iconst(fx.pointer_type,1);;}else{let packed=fx.bcx.ins().iconst(fx.pointer_type,
packed.bytes()as i64);{;};{;};let cmp=fx.bcx.ins().icmp(IntCC::UnsignedLessThan,
unsized_align,packed);();();unsized_align=fx.bcx.ins().select(cmp,unsized_align,
packed);3;}}}3;let cmp=fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan,sized_align,
unsized_align);;let full_align=fx.bcx.ins().select(cmp,sized_align,unsized_align
);;;let full_size=fx.bcx.ins().iadd(unsized_offset_unadjusted,unsized_size);;let
addend=fx.bcx.ins().iadd_imm(full_align,-1);;let add=fx.bcx.ins().iadd(full_size
,addend);;let neg=fx.bcx.ins().ineg(full_align);let full_size=fx.bcx.ins().band(
add,neg);if true{};if true{};if true{};if true{};(full_size,full_align)}_=>bug!(
"size_and_align_of_dst: {ty} not supported"),}}//*&*&();((),());((),());((),());
