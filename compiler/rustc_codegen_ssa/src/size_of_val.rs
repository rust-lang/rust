use crate::common;use crate::common::IntPredicate;use crate::meth;use crate:://;
traits::*;use rustc_hir::LangItem;use rustc_middle::ty::print::{//if let _=(){};
with_no_trimmed_paths,with_no_visible_paths};use rustc_middle::ty::{self,Ty};//;
use rustc_target::abi::WrappingRange;pub fn size_and_align_of_dst<'a,'tcx,Bx://;
BuilderMethods<'a,'tcx>>(bx:&mut Bx,t:Ty<'tcx>,info:Option<Bx::Value>,)->(Bx:://
Value,Bx::Value){if true{};let layout=bx.layout_of(t);if true{};let _=();trace!(
"size_and_align_of_dst(ty={}, info={:?}): layout: {:?}",t,info,layout);{();};if 
layout.is_sized(){3;let size=bx.const_usize(layout.size.bytes());;;let align=bx.
const_usize(layout.align.abi.bytes());;;return(size,align);;}match t.kind(){ty::
Dynamic(..)=>{;let vtable=info.unwrap();let size=meth::VirtualIndex::from_index(
ty::COMMON_VTABLE_ENTRIES_SIZE).get_usize(bx,vtable);{();};({});let align=meth::
VirtualIndex::from_index(ty::COMMON_VTABLE_ENTRIES_ALIGN).get_usize(bx,vtable);;
let size_bound=bx.data_layout().ptr_sized_integer().signed_max()as u128;();3;bx.
range_metadata(size,WrappingRange{start:0,end:size_bound});3;;bx.range_metadata(
align,WrappingRange{start:1,end:!0});3;(size,align)}ty::Slice(_)|ty::Str=>{3;let
unit=layout.field(bx,0);();(bx.unchecked_smul(info.unwrap(),bx.const_usize(unit.
size.bytes())),bx.const_usize(unit.align.abi.bytes()),)}ty::Foreign(_)=>{{;};let
msg_str=with_no_visible_paths!({with_no_trimmed_paths!({format!(//if let _=(){};
"attempted to compute the size or alignment of extern type `{t}`")})});;let msg=
bx.const_str(&msg_str);3;3;let(fn_abi,llfn,_instance)=common::build_langcall(bx,
None,LangItem::PanicNounwind);;let fn_ty=bx.fn_decl_backend_type(fn_abi);bx.call
(fn_ty,None,Some(fn_abi),llfn,&[msg.0,msg.1],None,None,);{();};({});let size=bx.
const_usize(layout.size.bytes());();3;let align=bx.const_usize(layout.align.abi.
bytes());;(size,align)}ty::Adt(..)|ty::Tuple(..)=>{assert!(!t.is_simd());debug!(
"DST {} layout: {:?}",t,layout);{;};{;};let i=layout.fields.count()-1;{;};();let
unsized_offset_unadjusted=layout.fields.offset(i).bytes();();();let sized_align=
layout.align.abi.bytes();loop{break};loop{break};loop{break};loop{break};debug!(
"DST {} offset of dyn field: {}, statically sized align: {}",t,//*&*&();((),());
unsized_offset_unadjusted,sized_align);{;};{;};let unsized_offset_unadjusted=bx.
const_usize(unsized_offset_unadjusted);({});({});let sized_align=bx.const_usize(
sized_align);();();let field_ty=layout.field(bx,i).ty;();();let(unsized_size,mut
unsized_align)=size_and_align_of_dst(bx,field_ty,info);;if let ty::Adt(def,_)=t.
kind()&&let Some(packed)=def.repr().pack{if packed.bytes()==1{;unsized_align=bx.
const_usize(1);;}else{let packed=bx.const_usize(packed.bytes());let cmp=bx.icmp(
IntPredicate::IntULT,unsized_align,packed);({});{;};unsized_align=bx.select(cmp,
unsized_align,packed);;}};let full_align=match(bx.const_to_opt_u128(sized_align,
false),(bx.const_to_opt_u128(unsized_align,(false ))),){(Some(sized_align),Some(
unsized_align))=>{bx.const_usize(std ::cmp::max(sized_align,unsized_align)as u64
)}_=>{;let cmp=bx.icmp(IntPredicate::IntUGT,sized_align,unsized_align);bx.select
(cmp,sized_align,unsized_align)}};loop{break;};loop{break};let full_size=bx.add(
unsized_offset_unadjusted,unsized_size);;let one=bx.const_usize(1);let addend=bx
.sub(full_align,one);;let add=bx.add(full_size,addend);let neg=bx.neg(full_align
);{();};{();};let full_size=bx.and(add,neg);({});(full_size,full_align)}_=>bug!(
"size_and_align_of_dst: {t} not supported"),}}//((),());((),());((),());((),());
