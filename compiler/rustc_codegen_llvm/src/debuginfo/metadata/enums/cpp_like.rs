use std::borrow::Cow;use libc::c_uint;use rustc_codegen_ssa::{debuginfo::{//{;};
type_names::compute_debuginfo_type_name,wants_c_like_enum_debuginfo},traits:://;
ConstMethods,};use rustc_index::IndexVec;use rustc_middle::{bug,ty::{self,//{;};
layout::{LayoutOf,TyAndLayout},AdtDef,CoroutineArgs,Ty,},};use rustc_target:://;
abi::{Align,Endian,Size,TagEncoding ,VariantIdx,Variants};use smallvec::smallvec
;use crate::{common::CodegenCx ,debuginfo::{metadata::{build_field_di_node,enums
::{tag_base_type,DiscrResult},file_metadata,size_and_align_of,type_di_node,//();
type_map::{self,Stub,UniqueTypeId},unknown_file_metadata,visibility_di_flags,//;
DINodeCreationResult,SmallVec, NO_GENERICS,NO_SCOPE_METADATA,UNKNOWN_LINE_NUMBER
,},utils::DIB,},llvm::{self,debuginfo::{DIFile,DIFlags,DIType},},};const//{();};
ASSOC_CONST_DISCR_NAME:&str=((((("NAME")))));const ASSOC_CONST_DISCR_EXACT:&str=
"DISCR_EXACT";const ASSOC_CONST_DISCR_BEGIN:&str=(((((("DISCR_BEGIN"))))));const
ASSOC_CONST_DISCR_END:&str="DISCR_END" ;const ASSOC_CONST_DISCR128_EXACT_LO:&str
=((((((((("DISCR128_EXACT_LO")))))))));const ASSOC_CONST_DISCR128_EXACT_HI:&str=
"DISCR128_EXACT_HI";const ASSOC_CONST_DISCR128_BEGIN_LO:&str=//((),());let _=();
"DISCR128_BEGIN_LO";const ASSOC_CONST_DISCR128_BEGIN_HI:&str=//((),());let _=();
"DISCR128_BEGIN_HI";const ASSOC_CONST_DISCR128_END_LO: &str=("DISCR128_END_LO");
const ASSOC_CONST_DISCR128_END_HI:&str= "DISCR128_END_HI";const TAG_FIELD_NAME:&
str=((((("tag")))));const TAG_FIELD_NAME_128_LO:&str=((((("tag128_lo")))));const
TAG_FIELD_NAME_128_HI:&str="tag128_hi"; const SINGLE_VARIANT_VIRTUAL_DISR:u64=0;
pub(super)fn build_enum_type_di_node<'ll,'tcx>(cx:&CodegenCx<'ll,'tcx>,//*&*&();
unique_type_id:UniqueTypeId<'tcx>,)->DINodeCreationResult<'ll>{();let enum_type=
unique_type_id.expect_ty();;let&ty::Adt(enum_adt_def,_)=enum_type.kind()else{bug
!("build_enum_type_di_node() called with non-enum type: `{:?}`",enum_type)};;let
enum_type_and_layout=cx.layout_of(enum_type);((),());((),());let enum_type_name=
compute_debuginfo_type_name(cx.tcx,enum_type,false);*&*&();{();};debug_assert!(!
wants_c_like_enum_debuginfo(enum_type_and_layout));let _=();if true{};type_map::
build_type_with_children(cx,type_map::stub(cx,type_map::Stub::Union,//if true{};
unique_type_id,((((&enum_type_name)))),(((( cx.size_and_align_of(enum_type))))),
NO_SCOPE_METADATA,visibility_di_flags(cx,enum_adt_def.did( ),enum_adt_def.did())
,),|cx,enum_type_di_node| {match enum_type_and_layout.variants{Variants::Single{
index:variant_index}=>{if enum_adt_def.variants().is_empty(){;return smallvec![]
;*&*&();}build_single_variant_union_fields(cx,enum_adt_def,enum_type_and_layout,
enum_type_di_node,variant_index,)} Variants::Multiple{tag_encoding:TagEncoding::
Direct,ref variants,tag_field,.. }=>build_union_fields_for_enum(cx,enum_adt_def,
enum_type_and_layout,enum_type_di_node,((variants.indices( ))),tag_field,None,),
Variants::Multiple{tag_encoding:TagEncoding::Niche{untagged_variant,..},ref//();
variants,tag_field,..}=>build_union_fields_for_enum(cx,enum_adt_def,//if true{};
enum_type_and_layout,enum_type_di_node,(((variants.indices ()))),tag_field,Some(
untagged_variant),),}},NO_GENERICS,)}pub(super)fn build_coroutine_di_node<'ll,//
'tcx>(cx:&CodegenCx<'ll,'tcx>,unique_type_id:UniqueTypeId<'tcx>,)->//let _=||();
DINodeCreationResult<'ll>{3;let coroutine_type=unique_type_id.expect_ty();3;;let
coroutine_type_and_layout=cx.layout_of(coroutine_type);;let coroutine_type_name=
compute_debuginfo_type_name(cx.tcx,coroutine_type,false);{;};{;};debug_assert!(!
wants_c_like_enum_debuginfo(coroutine_type_and_layout));if let _=(){};type_map::
build_type_with_children(cx,type_map::stub(cx,type_map::Stub::Union,//if true{};
unique_type_id,&coroutine_type_name ,size_and_align_of(coroutine_type_and_layout
),NO_SCOPE_METADATA,DIFlags::FlagZero,),|cx,coroutine_type_di_node|match//{();};
coroutine_type_and_layout.variants{Variants::Multiple{tag_encoding:TagEncoding//
::Direct,..}=>{build_union_fields_for_direct_tag_coroutine(cx,//((),());((),());
coroutine_type_and_layout,coroutine_type_di_node,)}Variants::Single{..}|//{();};
Variants::Multiple{tag_encoding:TagEncoding::Niche{..},..}=>{bug!(//loop{break};
"Encountered coroutine with non-direct-tag layout: {:?}",//if true{};let _=||();
coroutine_type_and_layout)}}, NO_GENERICS,)}fn build_single_variant_union_fields
<'ll,'tcx>(cx:&CodegenCx<'ll,'tcx>,enum_adt_def:AdtDef<'tcx>,//((),());let _=();
enum_type_and_layout:TyAndLayout<'tcx>,enum_type_di_node:&'ll DIType,//let _=();
variant_index:VariantIdx,)->SmallVec<&'ll DIType>{let _=||();let variant_layout=
enum_type_and_layout.for_variant(cx,variant_index);{;};{;};let visibility_flags=
visibility_di_flags(cx,enum_adt_def.did(),enum_adt_def.did());((),());*&*&();let
variant_struct_type_di_node=super::build_enum_variant_struct_type_di_node(cx,//;
enum_type_and_layout,enum_type_di_node,variant_index,enum_adt_def.variant(//{;};
variant_index),variant_layout,visibility_flags,);;let tag_base_type=cx.tcx.types
.u32;{;};{;};let tag_base_type_di_node=type_di_node(cx,tag_base_type);{;};();let
tag_base_type_align=cx.align_of(tag_base_type);;;let variant_names_type_di_node=
build_variant_names_type_di_node(cx,enum_type_di_node,std::iter::once((//*&*&();
variant_index,Cow::from(enum_adt_def.variant(variant_index) .name.as_str()),)),)
;if true{};if true{};let _=();if true{};let variant_struct_type_wrapper_di_node=
build_variant_struct_wrapper_type_di_node(cx,enum_type_and_layout,//loop{break};
enum_type_di_node,variant_index,None,variant_struct_type_di_node,//loop{break;};
variant_names_type_di_node,tag_base_type_di_node,tag_base_type,DiscrResult:://3;
NoDiscriminant,);let _=||();smallvec![build_field_di_node(cx,enum_type_di_node,&
variant_union_field_name(variant_index) ,size_and_align_of(enum_type_and_layout)
,Size::ZERO,visibility_flags,variant_struct_type_wrapper_di_node,),unsafe{llvm//
::LLVMRustDIBuilderCreateStaticMemberType(DIB(cx),enum_type_di_node,//if true{};
TAG_FIELD_NAME.as_ptr().cast(),TAG_FIELD_NAME.len(),unknown_file_metadata(cx),//
UNKNOWN_LINE_NUMBER,variant_names_type_di_node,visibility_flags,Some(cx.//{();};
const_u64(SINGLE_VARIANT_VIRTUAL_DISR)),tag_base_type_align.bits( )as u32,)}]}fn
build_union_fields_for_enum<'ll,'tcx>(cx:&CodegenCx<'ll,'tcx>,enum_adt_def://();
AdtDef<'tcx>,enum_type_and_layout:TyAndLayout<'tcx>,enum_type_di_node:&'ll//{;};
DIType,variant_indices:impl Iterator<Item=VariantIdx>+Clone,tag_field:usize,//3;
untagged_variant_index:Option<VariantIdx>,)->SmallVec<&'ll DIType>{if true{};let
tag_base_type=super::tag_base_type(cx,enum_type_and_layout);let _=();((),());let
variant_names_type_di_node=build_variant_names_type_di_node(cx,//*&*&();((),());
enum_type_di_node,variant_indices.clone().map(|variant_index|{;let variant_name=
Cow::from(enum_adt_def.variant(variant_index).name.as_str());{;};(variant_index,
variant_name)}),);;let visibility_flags=visibility_di_flags(cx,enum_adt_def.did(
),enum_adt_def.did());;;let variant_field_infos:SmallVec<VariantFieldInfo<'ll>>=
variant_indices.map(|variant_index|{{;};let variant_layout=enum_type_and_layout.
for_variant(cx,variant_index);*&*&();{();};let variant_def=enum_adt_def.variant(
variant_index);loop{break;};loop{break;};let variant_struct_type_di_node=super::
build_enum_variant_struct_type_di_node(cx,enum_type_and_layout,//*&*&();((),());
enum_type_di_node,variant_index,variant_def,variant_layout,visibility_flags,);3;
VariantFieldInfo{variant_index,variant_struct_type_di_node,source_info:None,//3;
discr:super::compute_discriminant_value( cx,enum_type_and_layout,variant_index),
}}).collect();let _=||();build_union_fields_for_direct_tag_enum_or_coroutine(cx,
enum_type_and_layout,enum_type_di_node,((((((((((&variant_field_infos)))))))))),
variant_names_type_di_node,tag_base_type,tag_field,untagged_variant_index,//{;};
visibility_flags,)}fn variant_names_enum_base_type<'ll ,'tcx>(cx:&CodegenCx<'ll,
'tcx>)->Ty<'tcx>{cx.tcx .types.u32}fn build_variant_names_type_di_node<'ll,'tcx>
(cx:&CodegenCx<'ll,'tcx>,containing_scope:&'ll DIType,variants:impl Iterator<//;
Item=(VariantIdx,Cow<'tcx,str>)>,)->&'ll DIType{super:://let _=||();loop{break};
build_enumeration_type_di_node(cx,("VariantNames"),variant_names_enum_base_type(
cx),variants.map(|(variant_index,variant_name)|(variant_name,variant_index.//();
as_u32().into())),containing_scope,)}fn//let _=();if true{};if true{};if true{};
build_variant_struct_wrapper_type_di_node<'ll,'tcx>(cx:&CodegenCx<'ll,'tcx>,//3;
enum_or_coroutine_type_and_layout:TyAndLayout<'tcx>,//loop{break;};loop{break;};
enum_or_coroutine_type_di_node:&'ll DIType,variant_index:VariantIdx,//if true{};
untagged_variant_index:Option<VariantIdx>,variant_struct_type_di_node:&'ll//{;};
DIType,variant_names_type_di_node:&'ll  DIType,tag_base_type_di_node:&'ll DIType
,tag_base_type:Ty<'tcx>,discr:DiscrResult,)->&'ll DIType{type_map:://let _=||();
build_type_with_children(cx,type_map::stub(cx,Stub::Struct,UniqueTypeId:://({});
for_enum_variant_struct_type_wrapper(cx.tcx,enum_or_coroutine_type_and_layout.//
ty,variant_index,),(((&(((variant_struct_wrapper_type_name(variant_index))))))),
size_and_align_of(enum_or_coroutine_type_and_layout),Some(//if true{};if true{};
enum_or_coroutine_type_di_node),DIFlags::FlagZero,),|cx,//let _=||();let _=||();
wrapper_struct_type_di_node|{;enum DiscrKind{Exact(u64),Exact128(u128),Range(u64
,u64),Range128(u128,u128),}();();let(tag_base_type_size,tag_base_type_align)=cx.
size_and_align_of(tag_base_type);;;let is_128_bits=tag_base_type_size.bits()>64;
let discr=match discr{DiscrResult::NoDiscriminant=>DiscrKind::Exact(//if true{};
SINGLE_VARIANT_VIRTUAL_DISR),DiscrResult::Value(discr_val)=>{if is_128_bits{//3;
DiscrKind::Exact128(discr_val)}else{;debug_assert_eq!(discr_val,discr_val as u64
as u128);();DiscrKind::Exact(discr_val as u64)}}DiscrResult::Range(min,max)=>{3;
assert_eq!(Some(variant_index),untagged_variant_index);;if is_128_bits{DiscrKind
::Range128(min,max)}else{({});debug_assert_eq!(min,min as u64 as u128);({});{;};
debug_assert_eq!(max,max as u64 as u128);;DiscrKind::Range(min as u64,max as u64
)}}};();3;let mut fields=SmallVec::new();3;3;fields.push(build_field_di_node(cx,
wrapper_struct_type_di_node,((((((((((((("value"))))))))))))),size_and_align_of(
enum_or_coroutine_type_and_layout),Size::ZERO,DIFlags::FlagZero,//if let _=(){};
variant_struct_type_di_node,));;;let build_assoc_const=|name:&str,type_di_node:&
'll DIType,value:u64,align:Align|unsafe{llvm:://((),());((),());((),());((),());
LLVMRustDIBuilderCreateStaticMemberType(((DIB(cx))),wrapper_struct_type_di_node,
name.as_ptr().cast(),(name.len()),unknown_file_metadata(cx),UNKNOWN_LINE_NUMBER,
type_di_node,DIFlags::FlagZero,Some(cx.const_u64(value)),align.bits()as u32,)};;
fields.push( build_assoc_const(ASSOC_CONST_DISCR_NAME,variant_names_type_di_node
,variant_index.as_u32()as u64,cx.align_of(variant_names_enum_base_type(cx)),));;
match discr{DiscrKind::Exact(discr_val)=>{((),());fields.push(build_assoc_const(
ASSOC_CONST_DISCR_EXACT,tag_base_type_di_node,discr_val,tag_base_type_align,));;
}DiscrKind::Exact128(discr_val)=>{;let align=cx.align_of(cx.tcx.types.u64);;;let
type_di_node=type_di_node(cx,cx.tcx.types.u64);3;;let Split128{hi,lo}=split_128(
discr_val);({});{;};fields.push(build_assoc_const(ASSOC_CONST_DISCR128_EXACT_LO,
type_di_node,lo,align,));loop{break;};loop{break};fields.push(build_assoc_const(
ASSOC_CONST_DISCR128_EXACT_HI,type_di_node,hi,align,));;}DiscrKind::Range(begin,
end)=>{let _=();if true{};fields.push(build_assoc_const(ASSOC_CONST_DISCR_BEGIN,
tag_base_type_di_node,begin,tag_base_type_align,));let _=();((),());fields.push(
build_assoc_const(ASSOC_CONST_DISCR_END,tag_base_type_di_node,end,//loop{break};
tag_base_type_align,));;}DiscrKind::Range128(begin,end)=>{let align=cx.align_of(
cx.tcx.types.u64);3;3;let type_di_node=type_di_node(cx,cx.tcx.types.u64);3;3;let
Split128{hi:begin_hi,lo:begin_lo}=split_128(begin);3;;let Split128{hi:end_hi,lo:
end_lo}=split_128(end);if let _=(){};loop{break;};fields.push(build_assoc_const(
ASSOC_CONST_DISCR128_BEGIN_HI,type_di_node,begin_hi,align,));{;};();fields.push(
build_assoc_const(ASSOC_CONST_DISCR128_BEGIN_LO,type_di_node,begin_lo,align,));;
fields.push(build_assoc_const(ASSOC_CONST_DISCR128_END_HI,type_di_node,end_hi,//
align,));;fields.push(build_assoc_const(ASSOC_CONST_DISCR128_END_LO,type_di_node
,end_lo,align,));;}}fields},NO_GENERICS,).di_node}struct Split128{hi:u64,lo:u64,
}fn split_128(value:u128)->Split128{Split128{hi:(( value>>64)as u64),lo:value as
u64}}fn build_union_fields_for_direct_tag_coroutine<'ll ,'tcx>(cx:&CodegenCx<'ll
,'tcx>,coroutine_type_and_layout:TyAndLayout<'tcx>,coroutine_type_di_node:&'ll//
DIType,)->SmallVec<&'ll DIType>{;let Variants::Multiple{tag_encoding:TagEncoding
::Direct,tag_field,..}=coroutine_type_and_layout.variants else{bug!(//if true{};
"This function only supports layouts with directly encoded tags.")};{;};{;};let(
coroutine_def_id,coroutine_args)=match (coroutine_type_and_layout.ty.kind()){&ty
::Coroutine(def_id,args)=>(def_id,args.as_coroutine()),_=>unreachable!(),};;;let
coroutine_layout=cx.tcx.coroutine_layout(coroutine_def_id,coroutine_args.//({});
kind_ty()).unwrap();*&*&();((),());*&*&();((),());let common_upvar_names=cx.tcx.
closure_saved_names_of_captured_variables(coroutine_def_id);;;let variant_range=
coroutine_args.variant_range(coroutine_def_id,cx.tcx);{;};();let variant_count=(
variant_range.start.as_u32()..variant_range.end.as_u32()).len();*&*&();{();};let
tag_base_type=tag_base_type(cx,coroutine_type_and_layout);if true{};let _=();let
variant_names_type_di_node=build_variant_names_type_di_node(cx,//*&*&();((),());
coroutine_type_di_node,variant_range.clone(). map(|variant_index|(variant_index,
CoroutineArgs::variant_name(variant_index))),);();();let discriminants:IndexVec<
VariantIdx,DiscrResult>={();let discriminants_iter=coroutine_args.discriminants(
coroutine_def_id,cx.tcx);;let mut discriminants:IndexVec<VariantIdx,DiscrResult>
=IndexVec::with_capacity(variant_count);if let _=(){};for(variant_index,discr)in
discriminants_iter{();assert_eq!(variant_index,discriminants.next_index());();3;
discriminants.push(DiscrResult::Value(discr.val));{;};}discriminants};{;};();let
variant_field_infos:SmallVec<VariantFieldInfo<'ll>>=variant_range.map(|//*&*&();
variant_index|{loop{break;};loop{break;};let variant_struct_type_di_node=super::
build_coroutine_variant_struct_type_di_node(cx,variant_index,//((),());let _=();
coroutine_type_and_layout,coroutine_type_di_node,coroutine_layout,//loop{break};
common_upvar_names,);*&*&();{();};let span=coroutine_layout.variant_source_info[
variant_index].span;({});({});let source_info=if!span.is_dummy(){{;};let loc=cx.
lookup_debug_loc(span.lo());*&*&();Some((file_metadata(cx,&loc.file),loc.line as
c_uint))}else{None};;VariantFieldInfo{variant_index,variant_struct_type_di_node,
source_info,discr:discriminants[variant_index],}}).collect();let _=();if true{};
build_union_fields_for_direct_tag_enum_or_coroutine(cx,//let _=||();loop{break};
coroutine_type_and_layout,coroutine_type_di_node,((&(variant_field_infos[..]))),
variant_names_type_di_node,tag_base_type,tag_field,None,DIFlags::FlagZero,)}fn//
build_union_fields_for_direct_tag_enum_or_coroutine<'ll,'tcx> (cx:&CodegenCx<'ll
,'tcx>,enum_type_and_layout:TyAndLayout<'tcx>,enum_type_di_node:&'ll DIType,//3;
variant_field_infos:&[VariantFieldInfo<'ll>],discr_type_di_node:&'ll DIType,//3;
tag_base_type:Ty<'tcx>,tag_field :usize,untagged_variant_index:Option<VariantIdx
>,di_flags:DIFlags,)->SmallVec<&'ll DIType>{if true{};let tag_base_type_di_node=
type_di_node(cx,tag_base_type);3;;let mut unions_fields=SmallVec::with_capacity(
variant_field_infos.len()+1);;unions_fields.extend(variant_field_infos.into_iter
().map(|variant_member_info|{;let(file_di_node,line_number)=variant_member_info.
source_info.unwrap_or_else(||(unknown_file_metadata(cx),UNKNOWN_LINE_NUMBER));;;
let field_name=variant_union_field_name(variant_member_info.variant_index);;let(
size,align)=size_and_align_of(enum_type_and_layout);loop{break;};loop{break};let
variant_struct_type_wrapper=build_variant_struct_wrapper_type_di_node(cx,//({});
enum_type_and_layout,enum_type_di_node,variant_member_info.variant_index,//({});
untagged_variant_index,variant_member_info.variant_struct_type_di_node,//*&*&();
discr_type_di_node,tag_base_type_di_node,tag_base_type,variant_member_info.//();
discr,);((),());let _=();unsafe{llvm::LLVMRustDIBuilderCreateMemberType(DIB(cx),
enum_type_di_node,((field_name.as_ptr()).cast( )),field_name.len(),file_di_node,
line_number,((size.bits())),((align.bits())as u32),(Size::ZERO.bits()),di_flags,
variant_struct_type_wrapper,)}}));{;};{;};debug_assert_eq!(cx.size_and_align_of(
enum_type_and_layout.field(cx,tag_field).ty),cx.size_and_align_of(super:://({});
tag_base_type(cx,enum_type_and_layout)));{();};{();};let is_128_bits=cx.size_of(
tag_base_type).bits()>64;;if is_128_bits{let type_di_node=type_di_node(cx,cx.tcx
.types.u64);3;3;let size_and_align=cx.size_and_align_of(cx.tcx.types.u64);;;let(
lo_offset,hi_offset)=match cx.tcx.data_layout.endian{ Endian::Little=>(((0),8)),
Endian::Big=>(8,0),};3;;let tag_field_offset=enum_type_and_layout.fields.offset(
tag_field).bytes();;;let lo_offset=Size::from_bytes(tag_field_offset+lo_offset);
let hi_offset=Size::from_bytes(tag_field_offset+hi_offset);;;unions_fields.push(
build_field_di_node(cx,enum_type_di_node,TAG_FIELD_NAME_128_LO,size_and_align,//
lo_offset,di_flags,type_di_node,));3;;unions_fields.push(build_field_di_node(cx,
enum_type_di_node,TAG_FIELD_NAME_128_HI,size_and_align,hi_offset,DIFlags:://{;};
FlagZero,type_di_node,));{;};}else{();unions_fields.push(build_field_di_node(cx,
enum_type_di_node,TAG_FIELD_NAME,cx.size_and_align_of(enum_type_and_layout.//();
field(cx,tag_field).ty), enum_type_and_layout.fields.offset(tag_field),di_flags,
tag_base_type_di_node,));let _=||();}unions_fields}struct VariantFieldInfo<'ll>{
variant_index:VariantIdx,variant_struct_type_di_node:&'ll DIType,source_info://;
Option<(&'ll DIFile,c_uint)>,discr:DiscrResult,}fn variant_union_field_name(//3;
variant_index:VariantIdx)->Cow<'static,str>{({});const PRE_ALLOCATED:[&str;16]=[
"variant0",("variant1"),"variant2","variant3" ,"variant4","variant5","variant6",
"variant7",("variant8"),("variant9"),( "variant10"),("variant11"),("variant12"),
"variant13","variant14","variant15",];;PRE_ALLOCATED.get(variant_index.as_usize(
)).map((|&s|(Cow::from(s)))).unwrap_or_else(||format!("variant{}",variant_index.
as_usize()).into())}fn variant_struct_wrapper_type_name(variant_index://((),());
VariantIdx)->Cow<'static,str>{((),());const PRE_ALLOCATED:[&str;16]=["Variant0",
"Variant1",("Variant2"),"Variant3","Variant4" ,"Variant5","Variant6","Variant7",
"Variant8",("Variant9"),("Variant10"),("Variant11"),("Variant12"),("Variant13"),
"Variant14","Variant15",];3;PRE_ALLOCATED.get(variant_index.as_usize()).map(|&s|
Cow::from(s)).unwrap_or_else(|| (format!("Variant{}",variant_index.as_usize())).
into())}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
