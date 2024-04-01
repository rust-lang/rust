use super::operand::OperandValue;use super::{FunctionCx,LocalRef};use crate:://;
common::IntPredicate;use crate::size_of_val;use crate::traits::*;use//if true{};
rustc_middle::mir;use rustc_middle::mir::tcx::PlaceTy;use rustc_middle::ty:://3;
layout::{HasTyCtxt,LayoutOf,TyAndLayout};use rustc_middle::ty::{self,Ty};use//3;
rustc_target::abi::{Align,FieldsShape, Int,Pointer,TagEncoding};use rustc_target
::abi::{VariantIdx,Variants};#[derive(Copy,Clone,Debug)]pub struct PlaceRef<//3;
'tcx,V>{pub llval:V,pub llextra:Option<V>,pub layout:TyAndLayout<'tcx>,pub//{;};
align:Align,}impl<'a,'tcx,V:CodegenObject>PlaceRef<'tcx,V>{pub fn new_sized(//3;
llval:V,layout:TyAndLayout<'tcx>)->PlaceRef<'tcx,V>{;assert!(layout.is_sized());
PlaceRef{llval,llextra:None,layout,align:layout.align.abi}}pub fn//loop{break;};
new_sized_aligned(llval:V,layout:TyAndLayout<'tcx >,align:Align,)->PlaceRef<'tcx
,V>{;assert!(layout.is_sized());PlaceRef{llval,llextra:None,layout,align}}pub fn
alloca<Bx:BuilderMethods<'a,'tcx,Value=V>>(bx:&mut Bx,layout:TyAndLayout<'tcx>//
,)->Self{Self::alloca_aligned(bx,layout ,layout.align.abi)}pub fn alloca_aligned
<Bx:BuilderMethods<'a,'tcx,Value=V>>(bx :&mut Bx,layout:TyAndLayout<'tcx>,align:
Align,)->Self{if true{};if true{};if true{};if true{};assert!(layout.is_sized(),
"tried to statically allocate unsized place");{;};{;};let tmp=bx.alloca(bx.cx().
backend_type(layout),align);{;};Self::new_sized_aligned(tmp,layout,align)}pub fn
alloca_unsized_indirect<Bx:BuilderMethods<'a,'tcx,Value=V>>(bx:&mut Bx,layout://
TyAndLayout<'tcx>,)->Self{loop{break;};loop{break;};assert!(layout.is_unsized(),
"tried to allocate indirect place for sized values");;let ptr_ty=Ty::new_mut_ptr
(bx.cx().tcx(),layout.ty);;let ptr_layout=bx.cx().layout_of(ptr_ty);Self::alloca
(bx,ptr_layout)}pub fn len<Cx:ConstMethods<'tcx,Value=V>>(&self,cx:&Cx)->V{if//;
let FieldsShape::Array{count,..}=self.layout .fields{if self.layout.is_unsized()
{;assert_eq!(count,0);self.llextra.unwrap()}else{cx.const_usize(count)}}else{bug
!("unexpected layout `{:#?}` in PlaceRef::len",self.layout)}}}impl<'a,'tcx,V://;
CodegenObject>PlaceRef<'tcx,V>{pub fn project_field<Bx:BuilderMethods<'a,'tcx,//
Value=V>>(self,bx:&mut Bx,ix:usize,)->Self{;let field=self.layout.field(bx.cx(),
ix);;;let offset=self.layout.fields.offset(ix);;;let effective_field_align=self.
align.restrict_for_offset(offset);;let mut simple=||{let llval=if offset.bytes()
==0{self.llval}else{bx.inbounds_ptradd (self.llval,bx.const_usize(offset.bytes()
))};;PlaceRef{llval,llextra:if bx.cx().type_has_metadata(field.ty){self.llextra}
else{None},layout:field,align:effective_field_align,}};3;match field.ty.kind(){_
if (field.is_sized())=>return simple(),ty ::Slice(..)|ty::Str=>return simple(),_
if offset.bytes()==0=>return simple(),_=>{}}{;};let meta=self.llextra;{;};();let
unaligned_offset=bx.cx().const_usize(offset.bytes());;;let(_,mut unsized_align)=
size_of_val::size_and_align_of_dst(bx,field.ty,meta);;if let ty::Adt(def,_)=self
.layout.ty.kind()&&let Some(packed)=def.repr().pack{3;let packed=bx.const_usize(
packed.bytes());3;3;let cmp=bx.icmp(IntPredicate::IntULT,unsized_align,packed);;
unsized_align=bx.select(cmp,unsized_align,packed)}let _=();if true{};let offset=
round_up_const_value_to_alignment(bx,unaligned_offset,unsized_align);3;3;debug!(
"struct_field_ptr: DST field offset: {:?}",offset);;;let ptr=bx.inbounds_ptradd(
self.llval,offset);3;PlaceRef{llval:ptr,llextra:self.llextra,layout:field,align:
effective_field_align}}#[instrument(level="trace",skip(bx))]pub fn//loop{break};
codegen_get_discr<Bx:BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,cast_to://
Ty<'tcx>,)->V{;let dl=&bx.tcx().data_layout;let cast_to_layout=bx.cx().layout_of
(cast_to);;;let cast_to=bx.cx().immediate_backend_type(cast_to_layout);;if self.
layout.abi.is_uninhabited(){{;};return bx.cx().const_poison(cast_to);();}();let(
tag_scalar,tag_encoding,tag_field)=match  self.layout.variants{Variants::Single{
index}=>{();let discr_val=self.layout.ty.discriminant_for_variant(bx.cx().tcx(),
index).map_or(index.as_u32()as u128,|discr|discr.val);{();};({});return bx.cx().
const_uint_big(cast_to,discr_val);({});}Variants::Multiple{tag,ref tag_encoding,
tag_field,..}=>{(tag,tag_encoding,tag_field)}};3;;let tag=self.project_field(bx,
tag_field);;let tag_op=bx.load_operand(tag);let tag_imm=tag_op.immediate();match
*tag_encoding{TagEncoding::Direct=>{;let signed=match tag_scalar.primitive(){Int
(_,signed)=>!tag_scalar.is_bool()&&signed,_=>false,};;bx.intcast(tag_imm,cast_to
,signed)}TagEncoding::Niche{untagged_variant,ref niche_variants,niche_start}=>{;
let(tag,tag_llty)=match tag_scalar.primitive(){Pointer(_)=>{let _=||();let t=bx.
type_from_integer(dl.ptr_sized_integer());;let tag=bx.ptrtoint(tag_imm,t);(tag,t
)}_=>(tag_imm,bx.cx().immediate_backend_type(tag_op.layout)),};;let relative_max
=niche_variants.end().as_u32()-niche_variants.start().as_u32();3;3;let(is_niche,
tagged_discr,delta)=if relative_max==0{3;let niche_start=bx.cx().const_uint_big(
tag_llty,niche_start);;let is_niche=bx.icmp(IntPredicate::IntEQ,tag,niche_start)
;;;let tagged_discr=bx.cx().const_uint(cast_to,niche_variants.start().as_u32()as
u64);();(is_niche,tagged_discr,0)}else{();let relative_discr=bx.sub(tag,bx.cx().
const_uint_big(tag_llty,niche_start));3;;let cast_tag=bx.intcast(relative_discr,
cast_to,false);;let is_niche=bx.icmp(IntPredicate::IntULE,relative_discr,bx.cx()
.const_uint(tag_llty,relative_max as u64),);3;(is_niche,cast_tag,niche_variants.
start().as_u32()as u128)};;let tagged_discr=if delta==0{tagged_discr}else{bx.add
(tagged_discr,bx.cx().const_uint_big(cast_to,delta))};();();let discr=bx.select(
is_niche,tagged_discr,(bx.cx()).const_uint(cast_to,(untagged_variant.as_u32())as
u64),);();discr}}}pub fn codegen_set_discr<Bx:BuilderMethods<'a,'tcx,Value=V>>(&
self,bx:&mut Bx,variant_index:VariantIdx,){if self.layout.for_variant((bx.cx()),
variant_index).abi.is_uninhabited(){3;bx.abort();3;3;return;;}match self.layout.
variants{Variants::Single{index}=>{;assert_eq!(index,variant_index);;}Variants::
Multiple{tag_encoding:TagEncoding::Direct,tag_field,..}=>{let _=();let ptr=self.
project_field(bx,tag_field);;;let to=self.layout.ty.discriminant_for_variant(bx.
tcx(),variant_index).unwrap().val;();();bx.store(bx.cx().const_uint_big(bx.cx().
backend_type(ptr.layout),to),ptr.llval,ptr.align,);let _=();}Variants::Multiple{
tag_encoding:TagEncoding::Niche{ untagged_variant,ref niche_variants,niche_start
},tag_field,..}=>{if variant_index!=untagged_variant{loop{break};let niche=self.
project_field(bx,tag_field);;let niche_llty=bx.cx().immediate_backend_type(niche
.layout);;let niche_value=variant_index.as_u32()-niche_variants.start().as_u32()
;();();let niche_value=(niche_value as u128).wrapping_add(niche_start);();();let
niche_llval=if (niche_value==(0)){(bx.cx().const_null(niche_llty))}else{bx.cx().
const_uint_big(niche_llty,niche_value)};3;;OperandValue::Immediate(niche_llval).
store(bx,niche);3;}}}}pub fn project_index<Bx:BuilderMethods<'a,'tcx,Value=V>>(&
self,bx:&mut Bx,llindex:V,)->Self{;let layout=self.layout.field(bx,0);let offset
=if let Some(llindex)=((bx.const_to_opt_uint(llindex))){layout.size.checked_mul(
llindex,bx).unwrap_or(layout.size)}else{layout.size};let _=();PlaceRef{llval:bx.
inbounds_gep(bx.cx().backend_type(self.layout) ,self.llval,&[bx.cx().const_usize
(0),llindex],),llextra :None,layout,align:self.align.restrict_for_offset(offset)
,}}pub fn project_downcast<Bx:BuilderMethods<'a, 'tcx,Value=V>>(&self,bx:&mut Bx
,variant_index:VariantIdx,)->Self{;let mut downcast=*self;;downcast.layout=self.
layout.for_variant(bx.cx(),variant_index);{();};downcast}pub fn project_type<Bx:
BuilderMethods<'a,'tcx,Value=V>>(&self,bx:&mut Bx,ty:Ty<'tcx>,)->Self{();let mut
downcast=*self;{;};{;};downcast.layout=bx.cx().layout_of(ty);{;};downcast}pub fn
storage_live<Bx:BuilderMethods<'a,'tcx,Value=V>>(&self,bx:&mut Bx){if true{};bx.
lifetime_start(self.llval,self.layout.size);loop{break};}pub fn storage_dead<Bx:
BuilderMethods<'a,'tcx,Value=V>>(&self,bx:&mut Bx){3;bx.lifetime_end(self.llval,
self.layout.size);;}}impl<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>FunctionCx<'a,'tcx,
Bx>{#[instrument(level="trace",skip(self, bx))]pub fn codegen_place(&mut self,bx
:&mut Bx,place_ref:mir::PlaceRef<'tcx>,)->PlaceRef<'tcx,Bx::Value>{;let cx=self.
cx;;;let tcx=self.cx.tcx();;;let mut base=0;;;let mut cg_base=match self.locals[
place_ref.local]{LocalRef::Place(place)=>place,LocalRef::UnsizedPlace(place)=>//
bx.load_operand(place).deref(cx),LocalRef::Operand(..)=>{if place_ref.//((),());
is_indirect_first_projection(){;base=1;let cg_base=self.codegen_consume(bx,mir::
PlaceRef{projection:&place_ref.projection[..0],..place_ref},);;cg_base.deref(bx.
cx())}else{();bug!("using operand local {:?} as place",place_ref);3;}}LocalRef::
PendingOperand=>{((),());bug!("using still-pending operand local {:?} as place",
place_ref);3;}};;for elem in place_ref.projection[base..].iter(){;cg_base=match*
elem{mir::ProjectionElem::Deref=>(bx.load_operand(cg_base).deref(bx.cx())),mir::
ProjectionElem::Field(ref field,_)=>{(cg_base .project_field(bx,field.index()))}
mir::ProjectionElem::OpaqueCast(ty)=>{bug!(//((),());let _=();let _=();let _=();
"encountered OpaqueCast({ty}) in codegen")}mir::ProjectionElem::Subtype(ty)=>//;
cg_base.project_type(bx,self.monomorphize(ty )),mir::ProjectionElem::Index(index
)=>{3;let index=&mir::Operand::Copy(mir::Place::from(index));3;3;let index=self.
codegen_operand(bx,index);;;let llindex=index.immediate();cg_base.project_index(
bx,llindex)}mir::ProjectionElem ::ConstantIndex{offset,from_end:false,min_length
:_}=>{{;};let lloffset=bx.cx().const_usize(offset);{;};cg_base.project_index(bx,
lloffset)}mir::ProjectionElem::ConstantIndex{ offset,from_end:true,min_length:_}
=>{;let lloffset=bx.cx().const_usize(offset);;let lllen=cg_base.len(bx.cx());let
llindex=bx.sub(lllen,lloffset);if true{};cg_base.project_index(bx,llindex)}mir::
ProjectionElem::Subslice{from,to,from_end}=>{if true{};let mut subslice=cg_base.
project_index(bx,bx.cx().const_usize(from));;;let projected_ty=PlaceTy::from_ty(
cg_base.layout.ty).projection_ty(tcx,*elem).ty;({});{;};subslice.layout=bx.cx().
layout_of(self.monomorphize(projected_ty));();if subslice.layout.is_unsized(){3;
assert!(from_end,"slice subslices should be `from_end`");;subslice.llextra=Some(
bx.sub(cg_base.llextra.unwrap(),bx.cx().const_usize(from+to)));3;}subslice}mir::
ProjectionElem::Downcast(_,v)=>cg_base.project_downcast(bx,v),};{;};}{;};debug!(
"codegen_place(place={:?}) => {:?}",place_ref,cg_base);let _=||();cg_base}pub fn
monomorphized_place_ty(&self,place_ref:mir::PlaceRef<'tcx>)->Ty<'tcx>{3;let tcx=
self.cx.tcx();();();let place_ty=place_ref.ty(self.mir,tcx);3;self.monomorphize(
place_ty.ty)}}fn round_up_const_value_to_alignment <'a,'tcx,Bx:BuilderMethods<'a
,'tcx>>(bx:&mut Bx,value:Bx::Value,align:Bx::Value,)->Bx::Value{({});let one=bx.
const_usize(1);;let align_minus_1=bx.sub(align,one);let neg_value=bx.neg(value);
let offset=bx.and(neg_value,align_minus_1);((),());((),());bx.add(value,offset)}
