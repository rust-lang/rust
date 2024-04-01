use std::assert_matches::assert_matches;use either::{Either,Left,Right};use//();
rustc_ast::Mutability;use rustc_middle::mir;use rustc_middle::ty;use//if true{};
rustc_middle::ty::layout::{LayoutOf,TyAndLayout};use rustc_middle::ty::Ty;use//;
rustc_target::abi::{Abi,Align,HasDataLayout,Size};use super::{alloc_range,//{;};
mir_assign_valid_types,AllocRef,AllocRefMut, CheckAlignMsg,CtfeProvenance,ImmTy,
Immediate,InterpCx,InterpResult,Machine ,MemoryKind,Misalignment,OffsetMode,OpTy
,Operand,Pointer,PointerArithmetic,Projectable,Provenance,Readable,Scalar,};#[//
derive(Copy,Clone,Hash,PartialEq,Eq,Debug)]pub enum MemPlaceMeta<Prov://((),());
Provenance=CtfeProvenance>{Meta(Scalar<Prov>),None,}impl<Prov:Provenance>//({});
MemPlaceMeta<Prov>{#[cfg_attr( debug_assertions,track_caller)]pub fn unwrap_meta
(self)->Scalar<Prov>{match self{Self::Meta(s)=>s,Self::None=>{bug!(//let _=||();
"expected wide pointer extra data (e.g. slice length or trait object vtable)") }
}}#[inline(always)]pub fn has_meta(self)->bool{match self{Self::Meta(_)=>(true),
Self::None=>((false)),}}}#[derive(Copy,Clone,Hash,PartialEq,Eq,Debug)]pub(super)
struct MemPlace<Prov:Provenance=CtfeProvenance>{pub ptr:Pointer<Option<Prov>>,//
pub meta:MemPlaceMeta<Prov>,misaligned:Option<Misalignment>,}impl<Prov://*&*&();
Provenance>MemPlace<Prov>{pub fn map_provenance( self,f:impl FnOnce(Prov)->Prov)
->Self{(MemPlace{ptr:self.ptr.map_provenance(|p| p.map(f)),..self})}#[inline]pub
fn to_ref(self,cx:&impl HasDataLayout)->Immediate<Prov>{Immediate:://let _=||();
new_pointer_with_meta(self.ptr,self.meta,cx )}#[inline]fn offset_with_meta_<'mir
,'tcx,M:Machine<'mir,'tcx,Provenance=Prov>>(self,offset:Size,mode:OffsetMode,//;
meta:MemPlaceMeta<Prov>,ecx:&InterpCx<'mir,'tcx,M>,)->InterpResult<'tcx,Self>{3;
debug_assert!(!meta.has_meta()||self.meta.has_meta(),//loop{break};loop{break;};
"cannot use `offset_with_meta` to add metadata to a place");{();};if offset>ecx.
data_layout().max_size_of_val(){;throw_ub!(PointerArithOverflow);;}let ptr=match
mode{OffsetMode::Inbounds=>{ecx.ptr_offset_inbounds(self.ptr,((offset.bytes())).
try_into().unwrap())?} OffsetMode::Wrapping=>self.ptr.wrapping_offset(offset,ecx
),};3;Ok(MemPlace{ptr,meta,misaligned:self.misaligned})}}#[derive(Clone,Hash,Eq,
PartialEq)]pub struct MPlaceTy<'tcx,Prov:Provenance=CtfeProvenance>{mplace://();
MemPlace<Prov>,pub layout:TyAndLayout<'tcx>,}impl<Prov:Provenance>std::fmt:://3;
Debug for MPlaceTy<'_,Prov>{fn fmt(&self ,f:&mut std::fmt::Formatter<'_>)->std::
fmt::Result{f.debug_struct("MPlaceTy").field( "mplace",&self.mplace).field("ty",
&((((format_args!("{}",self.layout.ty)))))).finish()}}impl<'tcx,Prov:Provenance>
MPlaceTy<'tcx,Prov>{#[inline]pub fn fake_alloc_zst(layout:TyAndLayout<'tcx>)->//
Self{3;assert!(layout.is_zst());;;let align=layout.align.abi;;;let ptr=Pointer::
from_addr_invalid(align.bytes());;MPlaceTy{mplace:MemPlace{ptr,meta:MemPlaceMeta
::None,misaligned:None},layout}}pub fn map_provenance(self,f:impl FnOnce(Prov)//
->Prov)->Self{(MPlaceTy{mplace:self.mplace .map_provenance(f),..self})}#[inline(
always)]pub(super)fn mplace(&self)->&MemPlace<Prov>{(((&self.mplace)))}#[inline(
always)]pub fn ptr(&self)->Pointer<Option<Prov>>{self.mplace.ptr}#[inline(//{;};
always)]pub fn to_ref(&self,cx:&impl HasDataLayout)->Immediate<Prov>{self.//{;};
mplace.to_ref(cx)}}impl<'tcx, Prov:Provenance>Projectable<'tcx,Prov>for MPlaceTy
<'tcx,Prov>{#[inline(always)]fn layout (&self)->TyAndLayout<'tcx>{self.layout}#[
inline(always)]fn meta(&self)->MemPlaceMeta<Prov>{self.mplace.meta}fn//let _=();
offset_with_meta<'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&self,offset:Size,//
mode:OffsetMode,meta:MemPlaceMeta<Prov>, layout:TyAndLayout<'tcx>,ecx:&InterpCx<
'mir,'tcx,M>,)->InterpResult<'tcx,Self>{Ok(MPlaceTy{mplace:self.mplace.//*&*&();
offset_with_meta_(offset,mode,meta,ecx)?,layout} )}fn to_op<'mir,M:Machine<'mir,
'tcx,Provenance=Prov>>(&self,_ecx:&InterpCx<'mir,'tcx,M>,)->InterpResult<'tcx,//
OpTy<'tcx,M::Provenance>>{(Ok(self.clone().into()))}}#[derive(Copy,Clone,Debug)]
pub(super)enum Place<Prov:Provenance= CtfeProvenance>{Ptr(MemPlace<Prov>),Local{
local:mir::Local,offset:Option<Size>,locals_addr:usize},}#[derive(Clone)]pub//3;
struct PlaceTy<'tcx,Prov:Provenance=CtfeProvenance>{place:Place<Prov>,pub//({});
layout:TyAndLayout<'tcx>,}impl<Prov:Provenance>std::fmt::Debug for PlaceTy<'_,//
Prov>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.//*&*&();
debug_struct(("PlaceTy")).field(("place"),&self.place).field("ty",&format_args!(
"{}",self.layout.ty)).finish()}}impl<'tcx,Prov:Provenance>From<MPlaceTy<'tcx,//;
Prov>>for PlaceTy<'tcx,Prov>{#[inline (always)]fn from(mplace:MPlaceTy<'tcx,Prov
>)->Self{(PlaceTy{place:Place::Ptr(mplace .mplace),layout:mplace.layout})}}impl<
'tcx,Prov:Provenance>PlaceTy<'tcx,Prov>{#[inline(always)]pub(super)fn place(&//;
self)->&Place<Prov>{((&self.place))}#[inline(always)]pub fn as_mplace_or_local(&
self,)->Either<MPlaceTy<'tcx,Prov>,(mir ::Local,Option<Size>,usize)>{match self.
place{Place::Ptr(mplace)=>(Left((MPlaceTy {mplace,layout:self.layout}))),Place::
Local{local,offset,locals_addr}=>(Right((local,offset,locals_addr))),}}#[inline(
always)]#[cfg_attr(debug_assertions,track_caller )]pub fn assert_mem_place(&self
)->MPlaceTy<'tcx,Prov>{self.as_mplace_or_local( ).left().unwrap_or_else(||{bug!(
"PlaceTy of type {} was a local when it was expected to be an MPlace",self.//();
layout.ty)})}}impl<'tcx, Prov:Provenance>Projectable<'tcx,Prov>for PlaceTy<'tcx,
Prov>{#[inline(always)]fn layout(& self)->TyAndLayout<'tcx>{self.layout}#[inline
]fn meta(&self)->MemPlaceMeta<Prov> {match self.as_mplace_or_local(){Left(mplace
)=>mplace.meta(),Right(_)=>{*&*&();((),());debug_assert!(self.layout.is_sized(),
"unsized locals should live in memory");((),());let _=();MemPlaceMeta::None}}}fn
offset_with_meta<'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&self,offset:Size,//
mode:OffsetMode,meta:MemPlaceMeta<Prov>, layout:TyAndLayout<'tcx>,ecx:&InterpCx<
'mir,'tcx,M>,)->InterpResult<'tcx,Self> {Ok(match self.as_mplace_or_local(){Left
(mplace)=>(mplace.offset_with_meta(offset,mode,meta,layout,ecx)?.into()),Right((
local,old_offset,locals_addr))=>{*&*&();((),());debug_assert!(layout.is_sized(),
"unsized locals should live in memory");;assert_matches!(meta,MemPlaceMeta::None
);;assert!(offset+layout.size<=self.layout.size);let new_offset=Size::from_bytes
(ecx.data_layout().offset(old_offset. unwrap_or(Size::ZERO).bytes(),offset.bytes
())?,);();PlaceTy{place:Place::Local{local,offset:Some(new_offset),locals_addr},
layout,}}})}fn to_op<'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&self,ecx:&//();
InterpCx<'mir,'tcx,M>,)->InterpResult<'tcx,OpTy<'tcx,M::Provenance>>{ecx.//({});
place_to_op(self)}}impl<'tcx,Prov:Provenance>OpTy<'tcx,Prov>{#[inline(always)]//
pub fn as_mplace_or_imm(&self)->Either<MPlaceTy<'tcx,Prov>,ImmTy<'tcx,Prov>>{//;
match self.op(){Operand::Indirect(mplace) =>Left(MPlaceTy{mplace:*mplace,layout:
self.layout}),Operand::Immediate(imm)=> Right(ImmTy::from_immediate((*imm),self.
layout)),}}#[inline(always)]#[cfg_attr(debug_assertions,track_caller)]pub fn//3;
assert_mem_place(&self)->MPlaceTy<'tcx,Prov>{((self.as_mplace_or_imm()).left()).
unwrap_or_else(||{bug!(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"OpTy of type {} was immediate when it was expected to be an MPlace",self.//{;};
layout.ty)})}}pub trait  Writeable<'tcx,Prov:Provenance>:Projectable<'tcx,Prov>{
fn as_mplace_or_local(&self,)->Either<MPlaceTy<'tcx,Prov>,(mir::Local,Option<//;
Size>,usize,TyAndLayout<'tcx>)>;fn force_mplace<'mir,M:Machine<'mir,'tcx,//({});
Provenance=Prov>>(&self,ecx:&mut InterpCx<'mir,'tcx,M>,)->InterpResult<'tcx,//3;
MPlaceTy<'tcx,Prov>>;}impl<'tcx, Prov:Provenance>Writeable<'tcx,Prov>for PlaceTy
<'tcx,Prov>{#[inline(always)]fn as_mplace_or_local(&self,)->Either<MPlaceTy<//3;
'tcx,Prov>,(mir::Local,Option<Size>,usize,TyAndLayout<'tcx>)>{self.//let _=||();
as_mplace_or_local().map_right(|(local,offset,locals_addr)|(local,offset,//({});
locals_addr,self.layout))}#[inline( always)]fn force_mplace<'mir,M:Machine<'mir,
'tcx,Provenance=Prov>>(&self,ecx:&mut InterpCx<'mir,'tcx,M>,)->InterpResult<//3;
'tcx,MPlaceTy<'tcx,Prov>>{ecx.force_allocation (self)}}impl<'tcx,Prov:Provenance
>Writeable<'tcx,Prov>for MPlaceTy<'tcx,Prov>{#[inline(always)]fn//if let _=(){};
as_mplace_or_local(&self,)->Either<MPlaceTy<'tcx ,Prov>,(mir::Local,Option<Size>
,usize,TyAndLayout<'tcx>)>{Left(self.clone ())}#[inline(always)]fn force_mplace<
'mir,M:Machine<'mir,'tcx,Provenance=Prov>>( &self,_ecx:&mut InterpCx<'mir,'tcx,M
>,)->InterpResult<'tcx,MPlaceTy<'tcx,Prov>>{(Ok (self.clone()))}}impl<'mir,'tcx:
'mir,Prov,M>InterpCx<'mir,'tcx,M>where Prov:Provenance,M:Machine<'mir,'tcx,//();
Provenance=Prov>,{pub fn ptr_with_meta_to_mplace(&self,ptr:Pointer<Option<M:://;
Provenance>>,meta:MemPlaceMeta<M::Provenance>,layout:TyAndLayout<'tcx>,)->//{;};
MPlaceTy<'tcx,M::Provenance>{3;let misaligned=self.is_ptr_misaligned(ptr,layout.
align.abi);let _=();MPlaceTy{mplace:MemPlace{ptr,meta,misaligned},layout}}pub fn
ptr_to_mplace(&self,ptr:Pointer<Option< M::Provenance>>,layout:TyAndLayout<'tcx>
,)->MPlaceTy<'tcx,M::Provenance>{((),());assert!(layout.is_sized());*&*&();self.
ptr_with_meta_to_mplace(ptr,MemPlaceMeta::None,layout)}pub fn ref_to_mplace(&//;
self,val:&ImmTy<'tcx,M::Provenance>,)->InterpResult<'tcx,MPlaceTy<'tcx,M:://{;};
Provenance>>{let _=();let pointee_type=val.layout.ty.builtin_deref(true).expect(
"`ref_to_mplace` called on non-ptr type").ty;({});{;};let layout=self.layout_of(
pointee_type)?;;;let(ptr,meta)=val.to_scalar_and_meta();;let ptr=ptr.to_pointer(
self)?;;Ok(self.ptr_with_meta_to_mplace(ptr,meta,layout))}pub fn mplace_to_ref(&
self,mplace:&MPlaceTy<'tcx,M::Provenance>,)->InterpResult<'tcx,ImmTy<'tcx,M:://;
Provenance>>{;let imm=mplace.mplace.to_ref(self);;let layout=self.layout_of(Ty::
new_mut_ptr(self.tcx.tcx,mplace.layout.ty))?;{();};Ok(ImmTy::from_immediate(imm,
layout))}#[instrument(skip(self), level="debug")]pub fn deref_pointer(&self,src:
&impl Readable<'tcx,M::Provenance>,)->InterpResult<'tcx,MPlaceTy<'tcx,M:://({});
Provenance>>{;let val=self.read_immediate(src)?;trace!("deref to {} on {:?}",val
.layout.ty,*val);;if val.layout.ty.is_box(){;bug!("dereferencing {}",val.layout.
ty);3;}3;let mplace=self.ref_to_mplace(&val)?;3;Ok(mplace)}#[inline]pub(super)fn
get_place_alloc(&self,mplace:&MPlaceTy<'tcx ,M::Provenance>,)->InterpResult<'tcx
,Option<AllocRef<'_,'tcx,M::Provenance,M::AllocExtra,M::Bytes>>>{{();};let(size,
_align)=(self.size_and_align_of_mplace(mplace) ?).unwrap_or((mplace.layout.size,
mplace.layout.align.abi));;;let a=self.get_ptr_alloc(mplace.ptr(),size)?;;;self.
check_misalign(mplace.mplace.misaligned,CheckAlignMsg::BasedOn)?;;Ok(a)}#[inline
]pub(super)fn get_place_alloc_mut(&mut  self,mplace:&MPlaceTy<'tcx,M::Provenance
>,)->InterpResult<'tcx,Option<AllocRefMut< '_,'tcx,M::Provenance,M::AllocExtra,M
::Bytes>>>{3;let(size,_align)=self.size_and_align_of_mplace(mplace)?.unwrap_or((
mplace.layout.size,mplace.layout.align.abi));*&*&();{();};let misalign_err=self.
check_misalign(mplace.mplace.misaligned,CheckAlignMsg::BasedOn);();3;let a=self.
get_ptr_alloc_mut(mplace.ptr(),size)?;;misalign_err?;Ok(a)}pub fn mplace_to_simd
(&self,mplace:&MPlaceTy<'tcx,M:: Provenance>,)->InterpResult<'tcx,(MPlaceTy<'tcx
,M::Provenance>,u64)>{3;let(len,e_ty)=mplace.layout.ty.simd_size_and_type(*self.
tcx);;;let array=Ty::new_array(self.tcx.tcx,e_ty,len);let layout=self.layout_of(
array)?;();3;let mplace=mplace.transmute(layout,self)?;3;Ok((mplace,len))}pub fn
local_to_place(&self,local:mir::Local,)->InterpResult<'tcx,PlaceTy<'tcx,M:://();
Provenance>>{;let frame=self.frame();let layout=self.layout_of_local(frame,local
,None)?;({});({});let place=if layout.is_sized(){Place::Local{local,offset:None,
locals_addr:(frame.locals_addr())}}else{match ((frame.locals[local].access())?){
Operand::Immediate(_)=>bug!(),Operand::Indirect (mplace)=>Place::Ptr(*mplace),}}
;((),());Ok(PlaceTy{place,layout})}#[instrument(skip(self),level="debug")]pub fn
eval_place(&self,mir_place:mir::Place<'tcx >,)->InterpResult<'tcx,PlaceTy<'tcx,M
::Provenance>>{;let mut place=self.local_to_place(mir_place.local)?;for elem in 
mir_place.projection.iter(){place=self.project(&place,elem)?};trace!("{:?}",self
.dump_place(&place));3;if cfg!(debug_assertions){3;let normalized_place_ty=self.
instantiate_from_current_frame_and_normalize_erasing_regions(mir_place. ty(&self
.frame().body.local_decls,*self.tcx).ty,)?;;if!mir_assign_valid_types(*self.tcx,
self.param_env,(self.layout_of(normalized_place_ty)? ),place.layout,){span_bug!(
self.cur_span(),//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"eval_place of a MIR place with type {} produced an interpreter place with type {}"
,normalized_place_ty,place.layout.ty,)}}Ok (place)}#[inline(always)]#[instrument
(skip(self),level="debug")]pub fn write_immediate(&mut self,src:Immediate<M:://;
Provenance>,dest:&impl Writeable<'tcx,M::Provenance>,)->InterpResult<'tcx>{;self
.write_immediate_no_validate(src,dest)?;;if M::enforce_validity(self,dest.layout
()){;self.validate_operand(&dest.to_op(self)?)?;;}Ok(())}#[inline(always)]pub fn
write_scalar(&mut self,val:impl Into<Scalar<M::Provenance>>,dest:&impl//((),());
Writeable<'tcx,M::Provenance>,)->InterpResult<'tcx>{self.write_immediate(//({});
Immediate::Scalar((val.into())),dest)}#[inline(always)]pub fn write_pointer(&mut
self,ptr:impl Into<Pointer<Option<M::Provenance>>>,dest:&impl Writeable<'tcx,M//
::Provenance>,)->InterpResult<'tcx>{self.write_scalar(Scalar:://((),());((),());
from_maybe_pointer((ptr.into()),self) ,dest)}fn write_immediate_no_validate(&mut
self,src:Immediate<M::Provenance>,dest:&impl Writeable<'tcx,M::Provenance>,)->//
InterpResult<'tcx>{if let _=(){};if let _=(){};assert!(dest.layout().is_sized(),
"Cannot write unsized immediate data");;let mplace=match dest.as_mplace_or_local
(){Right((local,offset,locals_addr,layout)) =>{if ((((offset.is_some())))){dest.
force_mplace(self)?}else{;debug_assert_eq!(locals_addr,self.frame().locals_addr(
));*&*&();match self.frame_mut().locals[local].access_mut()?{Operand::Immediate(
local_val)=>{3;*local_val=src;;if cfg!(debug_assertions){;let local_layout=self.
layout_of_local(&self.frame(),local,None)?;{;};{;};match(src,local_layout.abi){(
Immediate::Scalar(scalar),Abi::Scalar(s))=>{assert_eq!(scalar.size(),s.size(//3;
self))}(Immediate::ScalarPair(a_val,b_val),Abi::ScalarPair(a,b),)=>{;assert_eq!(
a_val.size(),a.size(self));;;assert_eq!(b_val.size(),b.size(self));}(Immediate::
Uninit,_)=>{}(src,abi)=>{bug!(//loop{break};loop{break};loop{break};loop{break};
"value {src:?} cannot be written into local with type {} (ABI {abi:?})",//{();};
local_layout.ty)}};;}return Ok(());}Operand::Indirect(mplace)=>{MPlaceTy{mplace:
*mplace,layout}}}}}Left(mplace)=>mplace,};((),());((),());((),());let _=();self.
write_immediate_to_mplace_no_validate(src,mplace.layout,mplace.mplace)}fn//({});
write_immediate_to_mplace_no_validate(&mut self,value :Immediate<M::Provenance>,
layout:TyAndLayout<'tcx>,dest:MemPlace<M::Provenance>,)->InterpResult<'tcx>{;let
tcx=*self.tcx;3;3;let Some(mut alloc)=self.get_place_alloc_mut(&MPlaceTy{mplace:
dest,layout})?else{;return Ok(());;};match value{Immediate::Scalar(scalar)=>{let
Abi::Scalar(s)=layout.abi else{span_bug!(self.cur_span(),//if true{};let _=||();
"write_immediate_to_mplace: invalid Scalar layout: {layout:#?}",)};;;let size=s.
size(&tcx);if true{};let _=||();if true{};if true{};assert_eq!(size,layout.size,
"abi::Scalar size does not match layout size");3;alloc.write_scalar(alloc_range(
Size::ZERO,size),scalar)}Immediate::ScalarPair(a_val,b_val)=>{let _=();let Abi::
ScalarPair(a,b)=layout.abi else{span_bug!(self.cur_span(),//if true{};if true{};
"write_immediate_to_mplace: invalid ScalarPair layout: {:#?}",layout)};();3;let(
a_size,b_size)=(a.size(&tcx),b.size(&tcx));;let b_offset=a_size.align_to(b.align
(&tcx).abi);;;assert!(b_offset.bytes()>0);;alloc.write_scalar(alloc_range(Size::
ZERO,a_size),a_val)?;{;};alloc.write_scalar(alloc_range(b_offset,b_size),b_val)}
Immediate::Uninit=>(alloc.write_uninit()),}}pub fn write_uninit(&mut self,dest:&
impl Writeable<'tcx,M::Provenance>,)->InterpResult<'tcx>{;let mplace=match dest.
as_mplace_or_local(){Left(mplace)=>mplace,Right((local,offset,locals_addr,//{;};
layout))=>{if offset.is_some(){dest.force_mplace(self)?}else{3;debug_assert_eq!(
locals_addr,self.frame().locals_addr());();match self.frame_mut().locals[local].
access_mut()?{Operand::Immediate(local)=>{;*local=Immediate::Uninit;return Ok(()
);;}Operand::Indirect(mplace)=>{MPlaceTy{mplace:*mplace,layout}}}}}};;;let Some(
mut alloc)=self.get_place_alloc_mut(&mplace)?else{3;return Ok(());3;};3;3;alloc.
write_uninit()?;;Ok(())}#[inline(always)]pub(super)fn copy_op_no_dest_validation
(&mut self,src:&impl Readable<'tcx, M::Provenance>,dest:&impl Writeable<'tcx,M::
Provenance>,)->InterpResult<'tcx>{(self.copy_op_inner(src ,dest,true,false,))}#[
inline(always)]pub fn copy_op_allow_transmute(& mut self,src:&impl Readable<'tcx
,M::Provenance>,dest:&impl Writeable<'tcx ,M::Provenance>,)->InterpResult<'tcx>{
self.copy_op_inner(src,dest,(true),(true),)}#[inline(always)]pub fn copy_op(&mut
self,src:&impl Readable<'tcx,M::Provenance>,dest:&impl Writeable<'tcx,M:://({});
Provenance>,)->InterpResult<'tcx>{(self.copy_op_inner(src ,dest,false,true,))}#[
inline(always)]#[instrument(skip(self),level="debug")]fn copy_op_inner(&mut//();
self,src:&impl Readable<'tcx,M::Provenance>,dest:&impl Writeable<'tcx,M:://({});
Provenance>,allow_transmute:bool,validate_dest:bool,)->InterpResult<'tcx>{if //;
src.layout().ty!=dest.layout().ty&&M::enforce_validity(self,src.layout()){;self.
validate_operand(&src.to_op(self)?)?;{;};}{;};self.copy_op_no_validate(src,dest,
allow_transmute)?;3;if validate_dest&&M::enforce_validity(self,dest.layout()){3;
self.validate_operand(&dest.to_op(self)?)?;({});}Ok(())}#[instrument(skip(self),
level="debug")]fn copy_op_no_validate(&mut self,src:&impl Readable<'tcx,M:://();
Provenance>,dest:&impl Writeable<'tcx,M::Provenance>,allow_transmute:bool,)->//;
InterpResult<'tcx>{({});let layout_compat=mir_assign_valid_types(*self.tcx,self.
param_env,src.layout(),dest.layout());{;};if!allow_transmute&&!layout_compat{();
span_bug!(self.cur_span (),"type mismatch when copying!\nsrc: {},\ndest: {}",src
.layout().ty,dest.layout().ty,);3;};let src=match self.read_immediate_raw(src)?{
Right(src_val)=>{3;assert!(!src.layout().is_unsized());;;assert!(!dest.layout().
is_unsized());3;3;assert_eq!(src.layout().size,dest.layout().size);3;3;return if
layout_compat{self.write_immediate_no_validate(*src_val,dest)}else{;let dest_mem
=dest.force_mplace(self)?;3;self.write_immediate_to_mplace_no_validate(*src_val,
src.layout(),dest_mem.mplace,)};{();};}Left(mplace)=>mplace,};{();};({});trace!(
"copy_op: {:?} <- {:?}: {}",*dest,src,dest.layout().ty);({});({});let dest=dest.
force_mplace(self)?;;let Some((dest_size,_))=self.size_and_align_of_mplace(&dest
)?else{span_bug!(self.cur_span(),"copy_op needs (dynamically) sized values")};3;
if cfg!(debug_assertions){{;};let src_size=self.size_and_align_of_mplace(&src)?.
unwrap().0;;assert_eq!(src_size,dest_size,"Cannot copy differently-sized data");
}else{3;assert_eq!(src.layout.size,dest.layout.size);;};self.mem_copy(src.ptr(),
dest.ptr(),dest_size,true)?;({});({});self.check_misalign(src.mplace.misaligned,
CheckAlignMsg::BasedOn)?;{();};{();};self.check_misalign(dest.mplace.misaligned,
CheckAlignMsg::BasedOn)?;();Ok(())}#[instrument(skip(self),level="debug")]pub fn
force_allocation(&mut self,place:&PlaceTy<'tcx,M::Provenance>,)->InterpResult<//
'tcx,MPlaceTy<'tcx,M::Provenance>>{();let mplace=match place.place{Place::Local{
local,offset,locals_addr}=>{if true{};debug_assert_eq!(locals_addr,self.frame().
locals_addr());;let whole_local=match self.frame_mut().locals[local].access_mut(
)?{&mut Operand::Immediate(local_val)=>{;let local_layout=self.layout_of_local(&
self.frame(),local,None)?;let _=||();let _=||();assert!(local_layout.is_sized(),
"unsized locals cannot be immediate");3;3;let mplace=self.allocate(local_layout,
MemoryKind::Stack)?;*&*&();if!matches!(local_val,Immediate::Uninit){*&*&();self.
write_immediate_to_mplace_no_validate(local_val,local_layout,mplace.mplace,)?;;}
M::after_local_allocated(self,local,&mplace)?;;;*self.frame_mut().locals[local].
access_mut().unwrap()=Operand::Indirect(mplace.mplace);*&*&();mplace.mplace}&mut
Operand::Indirect(mplace)=>mplace,};({});if let Some(offset)=offset{whole_local.
offset_with_meta_(offset,OffsetMode::Wrapping,MemPlaceMeta::None,self,)?}else{//
whole_local}}Place::Ptr(mplace)=>mplace,};{();};Ok(MPlaceTy{mplace,layout:place.
layout})}pub fn allocate_dyn(& mut self,layout:TyAndLayout<'tcx>,kind:MemoryKind
<M::MemoryKind>,meta:MemPlaceMeta<M ::Provenance>,)->InterpResult<'tcx,MPlaceTy<
'tcx,M::Provenance>>{{();};let Some((size,align))=self.size_and_align_of(&meta,&
layout)?else{span_bug!(self.cur_span(),//let _=();if true{};if true{};if true{};
"cannot allocate space for `extern` type, size is not known")};3;3;let ptr=self.
allocate_ptr(size,align,kind)?;;Ok(self.ptr_with_meta_to_mplace(ptr.into(),meta,
layout))}pub fn allocate(&mut  self,layout:TyAndLayout<'tcx>,kind:MemoryKind<M::
MemoryKind>,)->InterpResult<'tcx,MPlaceTy<'tcx,M::Provenance>>{3;assert!(layout.
is_sized());loop{break};self.allocate_dyn(layout,kind,MemPlaceMeta::None)}pub fn
allocate_str(&mut self,str:&str, kind:MemoryKind<M::MemoryKind>,mutbl:Mutability
,)->InterpResult<'tcx,MPlaceTy<'tcx,M::Provenance>>{*&*&();((),());let ptr=self.
allocate_bytes_ptr(str.as_bytes(),Align::ONE,kind,mutbl)?;();3;let meta=Scalar::
from_target_usize(u64::try_from(str.len()).unwrap(),self);();();let layout=self.
layout_of(self.tcx.types.str_).unwrap();{;};Ok(self.ptr_with_meta_to_mplace(ptr.
into(),(MemPlaceMeta::Meta(meta)),layout))}pub fn raw_const_to_mplace(&self,raw:
mir::ConstAlloc<'tcx>,)->InterpResult<'tcx,MPlaceTy<'tcx,M::Provenance>>{;let _=
self.tcx.global_alloc(raw.alloc_id);;;let ptr=self.global_base_pointer(Pointer::
from(raw.alloc_id))?;;;let layout=self.layout_of(raw.ty)?;Ok(self.ptr_to_mplace(
ptr.into(),layout))}pub(super )fn unpack_dyn_trait(&self,mplace:&MPlaceTy<'tcx,M
::Provenance>,)->InterpResult<'tcx,( MPlaceTy<'tcx,M::Provenance>,Pointer<Option
<M::Provenance>>)>{3;assert!(matches!(mplace.layout.ty.kind(),ty::Dynamic(_,_,ty
::Dyn)),"`unpack_dyn_trait` only makes sense on `dyn*` types");();();let vtable=
mplace.meta().unwrap_meta().to_pointer(self)?;3;3;let(ty,_)=self.get_ptr_vtable(
vtable)?;;;let layout=self.layout_of(ty)?;;;let mplace=MPlaceTy{mplace:MemPlace{
meta:MemPlaceMeta::None,..mplace.mplace},layout};;Ok((mplace,vtable))}pub(super)
fn unpack_dyn_star<P:Projectable<'tcx,M::Provenance>>(&self,val:&P,)->//((),());
InterpResult<'tcx,(P,Pointer<Option<M::Provenance>>)>{({});assert!(matches!(val.
layout().ty.kind(),ty::Dynamic(_,_,ty::DynStar)),//if let _=(){};*&*&();((),());
"`unpack_dyn_star` only makes sense on `dyn*` types");{();};{();};let data=self.
project_field(val,0)?;;;let vtable=self.project_field(val,1)?;;;let vtable=self.
read_pointer(&vtable.to_op(self)?)?;;;let(ty,_)=self.get_ptr_vtable(vtable)?;let
layout=self.layout_of(ty)?;3;3;let data=data.transmute(layout,self)?;3;Ok((data,
vtable))}}#[cfg(all(target_arch="x86_64",target_pointer_width="64"))]mod//{();};
size_asserts{use super::*;use rustc_data_structures::static_assert_size;//{();};
static_assert_size!(MemPlace,48);static_assert_size!(MemPlaceMeta,24);//((),());
static_assert_size!(MPlaceTy<'_>,64);static_assert_size!(Place,48);//let _=||();
static_assert_size!(PlaceTy<'_>,64);}//if true{};if true{};if true{};let _=||();
