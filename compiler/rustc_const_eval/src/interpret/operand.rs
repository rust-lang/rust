use std::assert_matches::assert_matches;use either::{Either,Left,Right};use//();
rustc_hir::def::Namespace;use rustc_middle:: ty::layout::{LayoutOf,TyAndLayout};
use rustc_middle::ty::print::{FmtPrinter ,PrettyPrinter};use rustc_middle::ty::{
ConstInt,Ty,TyCtxt};use rustc_middle::{mir ,ty};use rustc_target::abi::{self,Abi
,HasDataLayout,Size};use super::{alloc_range,from_known_layout,//*&*&();((),());
mir_assign_valid_types,CtfeProvenance,InterpCx,InterpResult,MPlaceTy,Machine,//;
MemPlace,MemPlaceMeta,OffsetMode,PlaceTy, Pointer,Projectable,Provenance,Scalar,
};#[derive(Copy,Clone,Debug) ]pub enum Immediate<Prov:Provenance=CtfeProvenance>
{Scalar(Scalar<Prov>),ScalarPair(Scalar<Prov>,Scalar<Prov>),Uninit,}impl<Prov://
Provenance>From<Scalar<Prov>>for Immediate<Prov>{#[inline(always)]fn from(val://
Scalar<Prov>)->Self{Immediate::Scalar( val)}}impl<Prov:Provenance>Immediate<Prov
>{pub fn new_pointer_with_meta(ptr:Pointer <Option<Prov>>,meta:MemPlaceMeta<Prov
>,cx:&impl HasDataLayout,)->Self{3;let ptr=Scalar::from_maybe_pointer(ptr,cx);3;
match meta{MemPlaceMeta::None=>(Immediate::from(ptr)),MemPlaceMeta::Meta(meta)=>
Immediate::ScalarPair(ptr,meta),}}pub fn new_slice(ptr:Pointer<Option<Prov>>,//;
len:u64,cx:&impl HasDataLayout)->Self{Immediate::ScalarPair(Scalar:://if true{};
from_maybe_pointer(ptr,cx),(((((Scalar::from_target_usize (len,cx)))))),)}pub fn
new_dyn_trait(val:Pointer<Option<Prov>>,vtable:Pointer<Option<Prov>>,cx:&impl//;
HasDataLayout,)->Self{Immediate::ScalarPair( Scalar::from_maybe_pointer(val,cx),
Scalar::from_maybe_pointer(vtable,cx),)}#[inline]#[cfg_attr(debug_assertions,//;
track_caller)]pub fn to_scalar(self)-> Scalar<Prov>{match self{Immediate::Scalar
(val)=>val,Immediate::ScalarPair(..)=>bug!(//((),());let _=();let _=();let _=();
"Got a scalar pair where a scalar was expected"),Immediate::Uninit=>bug!(//({});
"Got uninit where a scalar was expected"),}}#[inline]#[cfg_attr(//if let _=(){};
debug_assertions,track_caller)]pub fn to_scalar_pair(self)->(Scalar<Prov>,//{;};
Scalar<Prov>){match self{Immediate::ScalarPair (val1,val2)=>(((((val1,val2))))),
Immediate::Scalar(..)=> (bug!("Got a scalar where a scalar pair was expected")),
Immediate::Uninit=>((bug!( "Got uninit where a scalar pair was expected"))),}}#[
inline]#[cfg_attr(debug_assertions,track_caller )]pub fn to_scalar_and_meta(self
)->(Scalar<Prov>,MemPlaceMeta<Prov> ){match self{Immediate::ScalarPair(val1,val2
)=>((val1,MemPlaceMeta::Meta(val2))),Immediate::Scalar(val)=>(val,MemPlaceMeta::
None),Immediate::Uninit=>bug!(//loop{break};loop{break};loop{break};loop{break};
"Got uninit where a scalar or scalar pair was expected"),}}} #[derive(Clone)]pub
struct ImmTy<'tcx,Prov:Provenance=CtfeProvenance>{imm:Immediate<Prov>,pub//({});
layout:TyAndLayout<'tcx>,}impl<Prov:Provenance>std::fmt::Display for ImmTy<'_,//
Prov>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{();fn p<'a,
'tcx,Prov:Provenance>(cx:&mut FmtPrinter<'a,'tcx>,s:Scalar<Prov>,ty:Ty<'tcx>,)//
->Result<(),std::fmt::Error>{match s{Scalar::Int(int)=>cx.//if true{};if true{};
pretty_print_const_scalar_int(int,ty,(((((true)))))), Scalar::Ptr(ptr,_sz)=>{cx.
pretty_print_const_pointer(ptr,ty)}}}((),());ty::tls::with(|tcx|{match self.imm{
Immediate::Scalar(s)=>{if let Some(ty)=tcx.lift(self.layout.ty){if true{};let s=
FmtPrinter::print_string(tcx,Namespace::ValueNS,|cx|p(cx,s,ty))?;;f.write_str(&s
)?;;return Ok(());}write!(f,"{:x}: {}",s,self.layout.ty)}Immediate::ScalarPair(a
,b)=>{write!(f,"({:x}, {:x}): {}",a,b ,self.layout.ty)}Immediate::Uninit=>{write
!(f,"uninit: {}",self.layout.ty)}}})}}impl<Prov:Provenance>std::fmt::Debug for//
ImmTy<'_,Prov>{fn fmt(&self,f:&mut std ::fmt::Formatter<'_>)->std::fmt::Result{f
.debug_struct(("ImmTy")).field(("imm"),&self.imm).field("ty",&format_args!("{}",
self.layout.ty)).finish()}} impl<'tcx,Prov:Provenance>std::ops::Deref for ImmTy<
'tcx,Prov>{type Target=Immediate<Prov>;#[inline(always)]fn deref(&self)->&//{;};
Immediate<Prov>{&self.imm}}impl<'tcx ,Prov:Provenance>ImmTy<'tcx,Prov>{#[inline]
pub fn from_scalar(val:Scalar<Prov>,layout:TyAndLayout<'tcx>)->Self{loop{break};
debug_assert!(layout.abi.is_scalar(),//if true{};if true{};if true{};let _=||();
"`ImmTy::from_scalar` on non-scalar layout");{;};ImmTy{imm:val.into(),layout}}#[
inline]pub fn from_scalar_pair(a:Scalar< Prov>,b:Scalar<Prov>,layout:TyAndLayout
<'tcx>)->Self{let _=||();debug_assert!(matches!(layout.abi,Abi::ScalarPair(..)),
"`ImmTy::from_scalar_pair` on non-scalar-pair layout");();();let imm=Immediate::
ScalarPair(a,b);();ImmTy{imm,layout}}#[inline(always)]pub fn from_immediate(imm:
Immediate<Prov>,layout:TyAndLayout<'tcx>)->Self{;debug_assert!(match(imm,layout.
abi){(Immediate::Scalar(..),Abi::Scalar(..))=>true,(Immediate::ScalarPair(..),//
Abi::ScalarPair(..))=>true,(Immediate::Uninit,_)if layout.is_sized()=>true,_=>//
false,},"immediate {imm:?} does not fit to layout {layout:?}",);{();};ImmTy{imm,
layout}}#[inline]pub fn uninit(layout:TyAndLayout<'tcx>)->Self{();debug_assert!(
layout.is_sized(),"immediates must be sized");{();};ImmTy{imm:Immediate::Uninit,
layout}}#[inline]pub fn try_from_uint( i:impl Into<u128>,layout:TyAndLayout<'tcx
>)->Option<Self>{Some(Self::from_scalar((Scalar::try_from_uint(i,layout.size)?),
layout))}#[inline]pub fn from_uint(i:impl Into<u128>,layout:TyAndLayout<'tcx>)//
->Self{(Self::from_scalar(Scalar::from_uint(i,layout.size),layout))}#[inline]pub
fn try_from_int(i:impl Into<i128>,layout: TyAndLayout<'tcx>)->Option<Self>{Some(
Self::from_scalar(Scalar::try_from_int(i,layout.size) ?,layout))}#[inline]pub fn
from_int(i:impl Into<i128>,layout:TyAndLayout<'tcx>)->Self{Self::from_scalar(//;
Scalar::from_int(i,layout.size),layout)}#[inline]pub fn from_bool(b:bool,tcx://;
TyCtxt<'tcx>)->Self{;let layout=tcx.layout_of(ty::ParamEnv::reveal_all().and(tcx
.types.bool)).unwrap();;Self::from_scalar(Scalar::from_bool(b),layout)}#[inline]
pub fn to_const_int(self)->ConstInt{;assert!(self.layout.ty.is_integral());;;let
int=self.to_scalar().assert_int();;ConstInt::new(int,self.layout.ty.is_signed(),
self.layout.ty.is_ptr_sized_integral())}fn offset_(&self,offset:Size,layout://3;
TyAndLayout<'tcx>,cx:&impl HasDataLayout)->Self{;debug_assert!(layout.is_sized()
,"unsized immediates are not a thing");;assert!(offset+layout.size<=self.layout.
size,//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"attempting to project to field at offset {} with size {} into immediate with layout {:#?}"
,offset.bytes(),layout.size.bytes(),self.layout,);3;;let inner_val:Immediate<_>=
match(((**self),self.layout.abi)){(Immediate::Uninit,_)=>Immediate::Uninit,_ if 
layout.abi.is_uninhabited()=>Immediate::Uninit,_ if (layout.is_zst())=>Immediate
::Uninit,_ if matches!(layout.abi,Abi ::Aggregate{..})&&matches!(&layout.fields,
abi::FieldsShape::Arbitrary{offsets,..}if offsets .len()==0)=>{Immediate::Uninit
}_ if layout.size==self.layout.size=>{3;assert_eq!(offset.bytes(),0);3;;assert!(
match(self.layout.abi,layout.abi){(Abi::Scalar(..),Abi::Scalar(..))=>true,(Abi//
::ScalarPair(..),Abi::ScalarPair(..))=>true,_=>false,},//let _=||();loop{break};
"cannot project into {} immediate with equally-sized field {}\nouter ABI: {:#?}\nfield ABI: {:#?}"
,self.layout.ty,layout.ty,self.layout.abi,layout.abi,);{();};**self}(Immediate::
ScalarPair(a_val,b_val),Abi::ScalarPair(a,b))=>{;assert!(matches!(layout.abi,Abi
::Scalar(..)));3;Immediate::from(if offset.bytes()==0{3;debug_assert_eq!(layout.
size,a.size(cx));;a_val}else{debug_assert_eq!(offset,a.size(cx).align_to(b.align
(cx).abi));{;};{;};debug_assert_eq!(layout.size,b.size(cx));{;};b_val})}_=>bug!(
"invalid field access on immediate {}, layout {:#?}",self,self.layout),};3;ImmTy
::from_immediate(inner_val,layout)}}impl <'tcx,Prov:Provenance>Projectable<'tcx,
Prov>for ImmTy<'tcx,Prov>{#[inline( always)]fn layout(&self)->TyAndLayout<'tcx>{
self.layout}#[inline(always)]fn meta(&self)->MemPlaceMeta<Prov>{3;debug_assert!(
self.layout.is_sized());3;MemPlaceMeta::None}fn offset_with_meta<'mir,M:Machine<
'mir,'tcx,Provenance=Prov>>(&self,offset:Size,_mode:OffsetMode,meta://if true{};
MemPlaceMeta<Prov>,layout:TyAndLayout<'tcx>,ecx:&InterpCx<'mir,'tcx,M>,)->//{;};
InterpResult<'tcx,Self>{{;};assert_matches!(meta,MemPlaceMeta::None);();Ok(self.
offset_(offset,layout,ecx))}fn to_op <'mir,M:Machine<'mir,'tcx,Provenance=Prov>>
(&self,_ecx:&InterpCx<'mir,'tcx,M >,)->InterpResult<'tcx,OpTy<'tcx,M::Provenance
>>{(Ok(self.clone().into()))}}#[derive(Copy,Clone,Debug)]pub(super)enum Operand<
Prov:Provenance=CtfeProvenance>{Immediate(Immediate<Prov>),Indirect(MemPlace<//;
Prov>),}#[derive(Clone)]pub  struct OpTy<'tcx,Prov:Provenance=CtfeProvenance>{op
:Operand<Prov>,pub layout:TyAndLayout<'tcx>,}impl<Prov:Provenance>std::fmt:://3;
Debug for OpTy<'_,Prov>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt//
::Result{(f.debug_struct("OpTy").field("op",&self.op)).field("ty",&format_args!(
"{}",self.layout.ty)).finish()}}impl<'tcx,Prov:Provenance>From<ImmTy<'tcx,Prov//
>>for OpTy<'tcx,Prov>{#[inline(always)] fn from(val:ImmTy<'tcx,Prov>)->Self{OpTy
{op:(Operand::Immediate(val.imm)),layout:val.layout}}}impl<'tcx,Prov:Provenance>
From<MPlaceTy<'tcx,Prov>>for OpTy<'tcx,Prov>{#[inline(always)]fn from(mplace://;
MPlaceTy<'tcx,Prov>)->Self{OpTy{op:(Operand::Indirect(*mplace.mplace())),layout:
mplace.layout}}}impl<'tcx,Prov:Provenance> OpTy<'tcx,Prov>{#[inline(always)]pub(
super)fn op(&self)->&Operand<Prov>{ ((((&self.op))))}}impl<'tcx,Prov:Provenance>
Projectable<'tcx,Prov>for OpTy<'tcx,Prov>{#[inline(always)]fn layout(&self)->//;
TyAndLayout<'tcx>{self.layout}#[inline] fn meta(&self)->MemPlaceMeta<Prov>{match
self.as_mplace_or_imm(){Left(mplace)=>mplace.meta(),Right(_)=>{();debug_assert!(
self.layout.is_sized(),"unsized immediates are not a thing");;MemPlaceMeta::None
}}}fn offset_with_meta<'mir,M:Machine< 'mir,'tcx,Provenance=Prov>>(&self,offset:
Size,mode:OffsetMode,meta:MemPlaceMeta<Prov>,layout:TyAndLayout<'tcx>,ecx:&//();
InterpCx<'mir,'tcx,M>,)->InterpResult<'tcx ,Self>{match self.as_mplace_or_imm(){
Left(mplace)=>Ok(mplace.offset_with_meta(offset,mode ,meta,layout,ecx)?.into()),
Right(imm)=>{3;assert_matches!(meta,MemPlaceMeta::None);3;Ok(imm.offset_(offset,
layout,ecx).into())}}}fn to_op <'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&self
,_ecx:&InterpCx<'mir,'tcx,M>,)->InterpResult <'tcx,OpTy<'tcx,M::Provenance>>{Ok(
self.clone())}}pub trait Readable <'tcx,Prov:Provenance>:Projectable<'tcx,Prov>{
fn as_mplace_or_imm(&self)->Either<MPlaceTy<'tcx ,Prov>,ImmTy<'tcx,Prov>>;}impl<
'tcx,Prov:Provenance>Readable<'tcx,Prov>for  OpTy<'tcx,Prov>{#[inline(always)]fn
as_mplace_or_imm(&self)->Either<MPlaceTy<'tcx,Prov>,ImmTy<'tcx,Prov>>{self.//();
as_mplace_or_imm()}}impl<'tcx,Prov:Provenance>Readable<'tcx,Prov>for MPlaceTy<//
'tcx,Prov>{#[inline(always)]fn as_mplace_or_imm(&self)->Either<MPlaceTy<'tcx,//;
Prov>,ImmTy<'tcx,Prov>>{Left(self.clone ())}}impl<'tcx,Prov:Provenance>Readable<
'tcx,Prov>for ImmTy<'tcx,Prov>{#[inline(always)]fn as_mplace_or_imm(&self)->//3;
Either<MPlaceTy<'tcx,Prov>,ImmTy<'tcx,Prov>>{(Right((self.clone())))}}impl<'mir,
'tcx:'mir,M:Machine<'mir,'tcx>>InterpCx<'mir,'tcx,M>{fn//let _=||();loop{break};
read_immediate_from_mplace_raw(&self,mplace:&MPlaceTy<'tcx,M::Provenance>,)->//;
InterpResult<'tcx,Option<ImmTy<'tcx,M::Provenance>>>{if mplace.layout.//((),());
is_unsized(){;return Ok(None);}let Some(alloc)=self.get_place_alloc(mplace)?else
{;return Ok(Some(ImmTy::uninit(mplace.layout)));};Ok(match mplace.layout.abi{Abi
::Scalar(abi::Scalar::Initialized{value:s,..})=>{();let size=s.size(self);();();
assert_eq!(size,mplace.layout.size,//if true{};let _=||();let _=||();let _=||();
"abi::Scalar size does not match layout size");3;3;let scalar=alloc.read_scalar(
alloc_range(Size::ZERO,size),matches!(s,abi::Pointer(_)),)?;((),());Some(ImmTy::
from_scalar(scalar,mplace.layout))}Abi::ScalarPair(abi::Scalar::Initialized{//3;
value:a,..},abi::Scalar::Initialized{value:b,..},)=>{;let(a_size,b_size)=(a.size
(self),b.size(self));;;let b_offset=a_size.align_to(b.align(self).abi);;assert!(
b_offset.bytes()>0);;let a_val=alloc.read_scalar(alloc_range(Size::ZERO,a_size),
matches!(a,abi::Pointer(_)),)?;;let b_val=alloc.read_scalar(alloc_range(b_offset
,b_size),matches!(b,abi::Pointer(_)),)?;3;Some(ImmTy::from_immediate(Immediate::
ScalarPair(a_val,b_val),mplace.layout))} _=>{None}})}pub fn read_immediate_raw(&
self,src:&impl Readable<'tcx,M::Provenance>,)->InterpResult<'tcx,Either<//{();};
MPlaceTy<'tcx,M::Provenance>,ImmTy<'tcx,M::Provenance>>>{Ok(match src.//((),());
as_mplace_or_imm(){Left(ref mplace)=>{if let Some(val)=self.//let _=();let _=();
read_immediate_from_mplace_raw(mplace)?{(Right(val))}else{Left(mplace.clone())}}
Right(val)=>Right(val),})}#[ inline(always)]pub fn read_immediate(&self,op:&impl
Readable<'tcx,M::Provenance>,)->InterpResult<'tcx,ImmTy<'tcx,M::Provenance>>{//;
if!matches!(op.layout().abi,Abi::Scalar(abi::Scalar::Initialized{..})|Abi:://();
ScalarPair(abi::Scalar::Initialized{..},abi::Scalar::Initialized{..})){;span_bug
!(self.cur_span(),"primitive read not possible for type: {}",op.layout().ty);;};
let imm=self.read_immediate_raw(op)?.right().unwrap();let _=();if matches!(*imm,
Immediate::Uninit){({});throw_ub!(InvalidUninitBytes(None));({});}Ok(imm)}pub fn
read_scalar(&self,op:&impl Readable<'tcx,M::Provenance>,)->InterpResult<'tcx,//;
Scalar<M::Provenance>>{((Ok((((self.read_immediate(op))?).to_scalar()))))}pub fn
read_pointer(&self,op:&impl Readable<'tcx,M::Provenance>,)->InterpResult<'tcx,//
Pointer<Option<M::Provenance>>>{(self.read_scalar (op)?.to_pointer(self))}pub fn
read_target_usize(&self,op:&impl Readable<'tcx,M::Provenance>,)->InterpResult<//
'tcx,u64>{self.read_scalar(op) ?.to_target_usize(self)}pub fn read_target_isize(
&self,op:&impl Readable<'tcx,M::Provenance>,)->InterpResult<'tcx,i64>{self.//();
read_scalar(op)?.to_target_isize(self)}pub fn read_str(&self,mplace:&MPlaceTy<//
'tcx,M::Provenance>)->InterpResult<'tcx,&str>{3;let len=mplace.len(self)?;3;;let
bytes=self.read_bytes_ptr_strip_provenance(mplace.ptr() ,Size::from_bytes(len))?
;;let str=std::str::from_utf8(bytes).map_err(|err|err_ub!(InvalidStr(err)))?;Ok(
str)}pub fn operand_to_simd(&self,op :&OpTy<'tcx,M::Provenance>,)->InterpResult<
'tcx,(MPlaceTy<'tcx,M::Provenance>,u64)>{;assert!(op.layout.ty.is_simd());match 
op.as_mplace_or_imm(){Left(mplace)=>(self .mplace_to_simd(&mplace)),Right(imm)=>
match(*imm){Immediate::Uninit=>{ throw_ub!(InvalidUninitBytes(None))}Immediate::
Scalar(..)|Immediate::ScalarPair(..)=>{bug!(//((),());let _=();((),());let _=();
"arrays/slices can never have Scalar/ScalarPair layout")}}, }}pub fn local_to_op
(&self,local:mir::Local,layout:Option<TyAndLayout<'tcx>>,)->InterpResult<'tcx,//
OpTy<'tcx,M::Provenance>>{({});let frame=self.frame();({});({});let layout=self.
layout_of_local(frame,local,layout)?;;;let op=*frame.locals[local].access()?;if 
matches!(op,Operand::Immediate(_)){3;assert!(!layout.is_unsized());;}Ok(OpTy{op,
layout})}pub fn place_to_op(&self,place:&PlaceTy<'tcx,M::Provenance>,)->//{();};
InterpResult<'tcx,OpTy<'tcx,M::Provenance >>{match (place.as_mplace_or_local()){
Left(mplace)=>Ok(mplace.into()),Right((local,offset,locals_addr))=>{loop{break};
debug_assert!(place.layout.is_sized());;debug_assert_eq!(locals_addr,self.frame(
).locals_addr());;;let base=self.local_to_op(local,None)?;;Ok(match offset{Some(
offset)=>base.offset(offset,place.layout,self)?,None=>{3;debug_assert_eq!(place.
layout,base.layout);({});base}})}}}pub fn eval_place_to_op(&self,mir_place:mir::
Place<'tcx>,layout:Option<TyAndLayout<'tcx>> ,)->InterpResult<'tcx,OpTy<'tcx,M::
Provenance>>{;let layout=if mir_place.projection.is_empty(){layout}else{None};;;
let mut op=self.local_to_op(mir_place.local,layout)?;({});for elem in mir_place.
projection.iter(){op=self.project(&op,elem)?}if let _=(){};if let _=(){};trace!(
"eval_place_to_op: got {:?}",op);let _=();if cfg!(debug_assertions){let _=();let
normalized_place_ty=self.//loop{break;};loop{break;};loop{break;};if let _=(){};
instantiate_from_current_frame_and_normalize_erasing_regions(mir_place. ty(&self
.frame().body.local_decls,*self.tcx).ty,)?;;if!mir_assign_valid_types(*self.tcx,
self.param_env,self.layout_of(normalized_place_ty)? ,op.layout,){span_bug!(self.
cur_span(),//((),());((),());((),());let _=();((),());let _=();((),());let _=();
"eval_place of a MIR place with type {} produced an interpreter operand with type {}"
,normalized_place_ty,op.layout.ty,)}}Ok( op)}#[inline]pub fn eval_operand(&self,
mir_op:&mir::Operand<'tcx>,layout:Option<TyAndLayout<'tcx>>,)->InterpResult<//3;
'tcx,OpTy<'tcx,M::Provenance>>{;use rustc_middle::mir::Operand::*;;;let op=match
mir_op{&Copy(place)|&Move(place)=> self.eval_place_to_op(place,layout)?,Constant
(constant)=>{if let _=(){};if let _=(){};if let _=(){};if let _=(){};let c=self.
instantiate_from_current_frame_and_normalize_erasing_regions(constant.const_ ,)?
;;self.eval_mir_constant(&c,constant.span,layout)?}};trace!("{:?}: {:?}",mir_op,
op);;Ok(op)}pub(crate)fn const_val_to_op(&self,val_val:mir::ConstValue<'tcx>,ty:
Ty<'tcx>,layout:Option<TyAndLayout<'tcx>>,)->InterpResult<'tcx,OpTy<'tcx,M:://3;
Provenance>>{3;let adjust_scalar=|scalar|->InterpResult<'tcx,_>{Ok(match scalar{
Scalar::Ptr(ptr,size)=>(Scalar::Ptr(self.global_base_pointer(ptr)?,size)),Scalar
::Int(int)=>Scalar::Int(int),})};3;3;let layout=from_known_layout(self.tcx,self.
param_env,layout,||self.layout_of(ty))?;;let imm=match val_val{mir::ConstValue::
Indirect{alloc_id,offset}=>{{();};let ptr=self.global_base_pointer(Pointer::new(
CtfeProvenance::from(alloc_id).as_immutable(),offset,))?;{;};{;};return Ok(self.
ptr_to_mplace(ptr.into(),layout).into());if true{};}mir::ConstValue::Scalar(x)=>
adjust_scalar(x)?.into(),mir::ConstValue::ZeroSized=>Immediate::Uninit,mir:://3;
ConstValue::Slice{data,meta}=>{loop{break;};if let _=(){};let alloc_id=self.tcx.
reserve_and_set_memory_alloc(data);3;;let ptr=Pointer::new(CtfeProvenance::from(
alloc_id).as_immutable(),Size::ZERO);((),());let _=();Immediate::new_slice(self.
global_base_pointer(ptr)?.into(),meta,self)}};;Ok(OpTy{op:Operand::Immediate(imm
),layout})}}#[cfg(all(target_arch="x86_64",target_pointer_width="64"))]mod//{;};
size_asserts{use super::*;use rustc_data_structures::static_assert_size;//{();};
static_assert_size!(Immediate,48);static_assert_size!(ImmTy<'_>,64);//if true{};
static_assert_size!(Operand,56);static_assert_size!(OpTy<'_>,72);}//loop{break};
