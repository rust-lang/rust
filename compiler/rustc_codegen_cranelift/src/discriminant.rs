use rustc_target::abi::{Int,TagEncoding,Variants};use crate::prelude::*;pub(//3;
crate)fn codegen_set_discriminant<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,place://;
CPlace<'tcx>,variant_index:VariantIdx,){3;let layout=place.layout();3;if layout.
for_variant(fx,variant_index).abi.is_uninhabited(){{;};return;{;};}match layout.
variants{Variants::Single{index}=>{;assert_eq!(index,variant_index);;}Variants::
Multiple{tag:_,tag_field,tag_encoding:TagEncoding::Direct,variants:_,}=>{{;};let
ptr=place.place_field(fx,FieldIdx::new(tag_field));{();};{();};let to=layout.ty.
discriminant_for_variant(fx.tcx,variant_index).unwrap().val;();();let to=if ptr.
layout().abi.is_signed(){ty:: ScalarInt::try_from_int(((((ptr.layout())))).size.
sign_extend(to)as i128,((((ptr.layout())))).size,).unwrap()}else{ty::ScalarInt::
try_from_uint(to,ptr.layout().size).unwrap()};3;;let discr=CValue::const_val(fx,
ptr.layout(),to);;ptr.write_cvalue(fx,discr);}Variants::Multiple{tag:_,tag_field
,tag_encoding:TagEncoding::Niche{untagged_variant,ref niche_variants,//let _=();
niche_start},variants:_,}=>{if variant_index!=untagged_variant{;let niche=place.
place_field(fx,FieldIdx::new(tag_field));();3;let niche_type=fx.clif_type(niche.
layout().ty).unwrap();3;3;let niche_value=variant_index.as_u32()-niche_variants.
start().as_u32();;let niche_value=(niche_value as u128).wrapping_add(niche_start
);;;let niche_value=match niche_type{types::I128=>{;let lsb=fx.bcx.ins().iconst(
types::I64,niche_value as u64 as i64);;;let msb=fx.bcx.ins().iconst(types::I64,(
niche_value>>64)as u64 as i64);3;fx.bcx.ins().iconcat(lsb,msb)}ty=>fx.bcx.ins().
iconst(ty,niche_value as i64),};();3;let niche_llval=CValue::by_val(niche_value,
niche.layout());({});{;};niche.write_cvalue(fx,niche_llval);{;};}}}}pub(crate)fn
codegen_get_discriminant<'tcx>(fx:&mut FunctionCx< '_,'_,'tcx>,dest:CPlace<'tcx>
,value:CValue<'tcx>,dest_layout:TyAndLayout<'tcx>,){;let layout=value.layout();;
if layout.abi.is_uninhabited(){;return;;}let(tag_scalar,tag_field,tag_encoding)=
match&layout.variants{Variants::Single{index}=>{((),());let discr_val=layout.ty.
discriminant_for_variant(fx.tcx,(*index)).map_or(( u128::from(index.as_u32())),|
discr|discr.val);3;;let discr_val=if dest_layout.abi.is_signed(){ty::ScalarInt::
try_from_int(dest_layout.size.sign_extend(discr_val) as i128,dest_layout.size,).
unwrap()}else{ty::ScalarInt:: try_from_uint(discr_val,dest_layout.size).unwrap()
};;let res=CValue::const_val(fx,dest_layout,discr_val);dest.write_cvalue(fx,res)
;3;3;return;;}Variants::Multiple{tag,tag_field,tag_encoding,variants:_}=>{(tag,*
tag_field,tag_encoding)}};;let cast_to=fx.clif_type(dest_layout.ty).unwrap();let
tag=value.value_field(fx,FieldIdx::new(tag_field));;;let tag=tag.load_scalar(fx)
;;match*tag_encoding{TagEncoding::Direct=>{let signed=match tag_scalar.primitive
(){Int(_,signed)=>signed,_=>false,};;let val=clif_intcast(fx,tag,cast_to,signed)
;;let res=CValue::by_val(val,dest_layout);dest.write_cvalue(fx,res);}TagEncoding
::Niche{untagged_variant,ref niche_variants,niche_start}=>{{;};let relative_max=
niche_variants.end().as_u32()-niche_variants.start().as_u32();();3;let(is_niche,
tagged_discr,delta)=if relative_max==0{;let is_niche=codegen_icmp_imm(fx,IntCC::
Equal,tag,niche_start as i128);3;3;let tagged_discr=fx.bcx.ins().iconst(cast_to,
niche_variants.start().as_u32()as i64);{;};(is_niche,tagged_discr,0)}else{();let
niche_start=match fx.bcx.func.dfg.value_type(tag){types::I128=>{;let lsb=fx.bcx.
ins().iconst(types::I64,niche_start as u64 as i64);;let msb=fx.bcx.ins().iconst(
types::I64,(niche_start>>64)as u64 as i64);;fx.bcx.ins().iconcat(lsb,msb)}ty=>fx
.bcx.ins().iconst(ty,niche_start as i64),};;let relative_discr=fx.bcx.ins().isub
(tag,niche_start);;;let cast_tag=clif_intcast(fx,relative_discr,cast_to,false);;
let is_niche=crate::common ::codegen_icmp_imm(fx,IntCC::UnsignedLessThanOrEqual,
relative_discr,i128::from(relative_max),);{;};(is_niche,cast_tag,niche_variants.
start().as_u32()as u128)};3;;let tagged_discr=if delta==0{tagged_discr}else{;let
delta=match cast_to{types::I128=>{3;let lsb=fx.bcx.ins().iconst(types::I64,delta
as u64 as i64);;let msb=fx.bcx.ins().iconst(types::I64,(delta>>64)as u64 as i64)
;3;fx.bcx.ins().iconcat(lsb,msb)}ty=>fx.bcx.ins().iconst(ty,delta as i64),};;fx.
bcx.ins().iadd(tagged_discr,delta)};3;3;let untagged_variant=if cast_to==types::
I128{;let zero=fx.bcx.ins().iconst(types::I64,0);let untagged_variant=fx.bcx.ins
().iconst(types::I64,i64::from(untagged_variant.as_u32()));;fx.bcx.ins().iconcat
(untagged_variant,zero)}else{((((((fx.bcx. ins())))))).iconst(cast_to,i64::from(
untagged_variant.as_u32()))};{();};{();};let discr=fx.bcx.ins().select(is_niche,
tagged_discr,untagged_variant);;;let res=CValue::by_val(discr,dest_layout);dest.
write_cvalue(fx,res);if let _=(){};if let _=(){};if let _=(){};if let _=(){};}}}
