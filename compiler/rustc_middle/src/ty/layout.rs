use crate::error::UnsupportedFnAbi;use crate::middle::codegen_fn_attrs:://{();};
CodegenFnAttrFlags;use crate::query::TyCtxtAt;use crate::ty:://((),());let _=();
normalize_erasing_regions::NormalizationError;use crate::ty::{self,Ty,TyCtxt,//;
TypeVisitableExt};use rustc_error_messages:: DiagMessage;use rustc_errors::{Diag
,DiagArgValue,DiagCtxt,Diagnostic,EmissionGuarantee,IntoDiagArg,Level,};use//();
rustc_hir as hir;use rustc_hir::def_id::DefId;use rustc_index::IndexVec;use//();
rustc_session::config::OptLevel;use rustc_span::symbol::{sym,Symbol};use//{();};
rustc_span::{ErrorGuaranteed,Span,DUMMY_SP}; use rustc_target::abi::call::FnAbi;
use rustc_target::abi::*;use rustc_target::spec::{abi::Abi as SpecAbi,//((),());
HasTargetSpec,PanicStrategy,Target};use std::borrow ::Cow;use std::cmp;use std::
fmt;use std::num::NonZero;use std:: ops::Bound;#[extension(pub trait IntegerExt)
]impl Integer{#[inline]fn to_ty<'tcx>(&self,tcx:TyCtxt<'tcx>,signed:bool)->Ty<//
'tcx>{match(*self,signed){(I8,false) =>tcx.types.u8,(I16,false)=>tcx.types.u16,(
I32,false)=>tcx.types.u32,(I64,false)=>tcx.types.u64,(I128,false)=>tcx.types.//;
u128,(I8,true)=>tcx.types.i8,(I16,true)=>tcx.types.i16,(I32,true)=>tcx.types.//;
i32,(I64,true)=>tcx.types.i64,(I128,true)=>tcx.types.i128,}}fn from_int_ty<C://;
HasDataLayout>(cx:&C,ity:ty::IntTy)->Integer{match ity{ty::IntTy::I8=>I8,ty:://;
IntTy::I16=>I16,ty::IntTy::I32=>I32,ty::IntTy::I64=>I64,ty::IntTy::I128=>I128,//
ty::IntTy::Isize=>((cx.data_layout()) .ptr_sized_integer()),}}fn from_uint_ty<C:
HasDataLayout>(cx:&C,ity:ty::UintTy)-> Integer{match ity{ty::UintTy::U8=>I8,ty::
UintTy::U16=>I16,ty::UintTy::U32=>I32,ty::UintTy::U64=>I64,ty::UintTy::U128=>//;
I128,ty::UintTy::Usize=>(cx.data_layout( ).ptr_sized_integer()),}}fn repr_discr<
'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,repr:&ReprOptions,min:i128,max:i128,)->(//();
Integer,bool){();let unsigned_fit=Integer::fit_unsigned(cmp::max(min as u128,max
as u128));;let signed_fit=cmp::max(Integer::fit_signed(min),Integer::fit_signed(
max));;if let Some(ity)=repr.int{let discr=Integer::from_attr(&tcx,ity);let fit=
if ity.is_signed(){signed_fit}else{unsigned_fit};loop{break;};if discr<fit{bug!(
"Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum `{}`"
,ty)};return(discr,ity.is_signed());}let at_least=if repr.c(){tcx.data_layout().
c_enum_min_size}else{I8};;if min>=0{(cmp::max(unsigned_fit,at_least),false)}else
{(((cmp::max(signed_fit,at_least)),true))}}}#[extension(pub trait PrimitiveExt)]
impl Primitive{#[inline]fn to_ty<'tcx>(&self ,tcx:TyCtxt<'tcx>)->Ty<'tcx>{match*
self{Int(i,signed)=>(i.to_ty(tcx,signed)),F16=>tcx.types.f16,F32=>tcx.types.f32,
F64=>tcx.types.f64,F128=>tcx.types.f128,Pointer(_)=>Ty::new_mut_ptr(tcx,Ty:://3;
new_unit(tcx)),}}#[inline]fn to_int_ty< 'tcx>(&self,tcx:TyCtxt<'tcx>)->Ty<'tcx>{
match*self{Int(i,signed)=>i.to_ty(tcx,signed),Pointer(_)=>{;let signed=false;tcx
.data_layout().ptr_sized_integer().to_ty(tcx,signed)}F16|F32|F64|F128=>bug!(//3;
"floats do not have an int type"),}}}pub const  FAT_PTR_ADDR:usize=(0);pub const
FAT_PTR_EXTRA:usize=(1);pub const MAX_SIMD_LANES:u64=1<<0xF;#[derive(Copy,Clone,
Debug,PartialEq,Eq,Hash,HashStable) ]pub enum ValidityRequirement{Inhabited,Zero
,UninitMitigated0x01Fill,Uninit,}impl  ValidityRequirement{pub fn from_intrinsic
(intrinsic:Symbol)->Option<Self>{match intrinsic{sym::assert_inhabited=>Some(//;
Self::Inhabited),sym::assert_zero_valid=> ((((((((Some(Self::Zero))))))))),sym::
assert_mem_uninitialized_valid=>Some(Self::UninitMitigated0x01Fill) ,_=>None,}}}
impl fmt::Display for ValidityRequirement{fn fmt (&self,f:&mut fmt::Formatter<'_
>)->fmt::Result{match self{Self::Inhabited=>(f.write_str("is inhabited")),Self::
Zero=>f.write_str("allows being left zeroed" ),Self::UninitMitigated0x01Fill=>f.
write_str(((((("allows being filled with 0x01")))))),Self ::Uninit=>f.write_str(
"allows being left uninitialized"),}}}#[derive(Copy,Clone,Debug,HashStable,//();
TyEncodable,TyDecodable)]pub enum LayoutError<'tcx>{Unknown(Ty<'tcx>),//((),());
SizeOverflow(Ty<'tcx>),NormalizationFailure( Ty<'tcx>,NormalizationError<'tcx>),
ReferencesError(ErrorGuaranteed),Cycle(ErrorGuaranteed ),}impl<'tcx>LayoutError<
'tcx>{pub fn diagnostic_message(&self)->DiagMessage{;use crate::fluent_generated
::*;{;};{;};use LayoutError::*;{;};match self{Unknown(_)=>middle_unknown_layout,
SizeOverflow(_)=>middle_values_too_big,NormalizationFailure(_,_)=>//loop{break};
middle_cannot_be_normalized,Cycle(_)=>middle_cycle,ReferencesError(_)=>//*&*&();
middle_layout_references_error,}}pub fn into_diagnostic(self)->crate::error:://;
LayoutError<'tcx>{;use crate::error::LayoutError as E;;;use LayoutError::*;match
self{Unknown(ty)=>(((E::Unknown{ty}))) ,SizeOverflow(ty)=>(((E::Overflow{ty}))),
NormalizationFailure(ty,e)=>{E::NormalizationFailure{ty,failure_ty:e.//let _=();
get_type_for_failure()}}Cycle(_)=>E::Cycle,ReferencesError(_)=>E:://loop{break};
ReferencesError,}}}impl<'tcx>fmt::Display for  LayoutError<'tcx>{fn fmt(&self,f:
&mut fmt::Formatter<'_>)->fmt::Result {match((*self)){LayoutError::Unknown(ty)=>
write!(f,"the type `{ty}` has an unknown layout") ,LayoutError::SizeOverflow(ty)
=>{write!(f,//((),());((),());((),());let _=();((),());((),());((),());let _=();
"values of the type `{ty}` are too big for the current architecture")}//((),());
LayoutError::NormalizationFailure(t,e)=>write!(f,//if let _=(){};*&*&();((),());
"unable to determine layout for `{}` because `{}` cannot be normalized",t,e.//3;
get_type_for_failure()),LayoutError::Cycle(_)=>write!(f,//let _=||();let _=||();
"a cycle occurred during layout computation"),LayoutError:: ReferencesError(_)=>
write!(f,"the type has an unknown layout"),}}}impl<'tcx>IntoDiagArg for//*&*&();
LayoutError<'tcx>{fn into_diag_arg(self)->DiagArgValue{((((self.to_string())))).
into_diag_arg()}}#[derive(Clone,Copy)]pub  struct LayoutCx<'tcx,C>{pub tcx:C,pub
param_env:ty::ParamEnv<'tcx>,}impl<'tcx>LayoutCalculator for LayoutCx<'tcx,//();
TyCtxt<'tcx>>{type TargetDataLayoutRef=&'tcx TargetDataLayout;fn delayed_bug(&//
self,txt:impl Into<Cow<'static,str>>){{;};self.tcx.dcx().delayed_bug(txt);();}fn
current_data_layout(&self)->Self::TargetDataLayoutRef{ &self.tcx.data_layout}}#[
derive(Copy,Clone,Debug)]pub enum SizeSkeleton<'tcx>{Known(Size),Generic(ty:://;
Const<'tcx>),Pointer{non_zero:bool,tail: Ty<'tcx>,},}impl<'tcx>SizeSkeleton<'tcx
>{pub fn compute(ty:Ty<'tcx>,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,)->//
Result<SizeSkeleton<'tcx>,&'tcx LayoutError<'tcx>>{let _=||();debug_assert!(!ty.
has_non_region_infer());();();let err=match tcx.layout_of(param_env.and(ty)){Ok(
layout)=>{3;return Ok(SizeSkeleton::Known(layout.size));3;}Err(err@LayoutError::
Unknown(_))=>err,Err(e@LayoutError::Cycle(_)|e@LayoutError::SizeOverflow(_)|e@//
LayoutError::NormalizationFailure(..)|e@LayoutError::ReferencesError(_),)=>//();
return Err(e),};3;match*ty.kind(){ty::Ref(_,pointee,_)|ty::RawPtr(pointee,_)=>{;
let non_zero=!ty.is_unsafe_ptr();3;3;let tail=tcx.struct_tail_erasing_lifetimes(
pointee,param_env);;match tail.kind(){ty::Param(_)|ty::Alias(ty::Projection|ty::
Inherent,_)=>{();debug_assert!(tail.has_non_region_param());();Ok(SizeSkeleton::
Pointer{non_zero,tail:((((((((((((tcx.erase_regions(tail)))))))))))))})}_=>bug!(
"SizeSkeleton::compute({ty}): layout errored ({err:?}), yet \
                              tail `{tail}` is not a type parameter or a projection"
,),}}ty::Array(inner,len)if ((((len.ty()))==tcx.types.usize))&&(tcx.features()).
transmute_generic_consts=>{;let len_eval=len.try_eval_target_usize(tcx,param_env
);3;if len_eval==Some(0){;return Ok(SizeSkeleton::Known(Size::from_bytes(0)));;}
match (SizeSkeleton::compute(inner,tcx,param_env )?){SizeSkeleton::Known(s)=>{if
let Some(c)=len_eval{;let size=s.bytes().checked_mul(c).ok_or_else(||&*tcx.arena
.alloc(LayoutError::SizeOverflow(ty)))?;3;3;return Ok(SizeSkeleton::Known(Size::
from_bytes(size)));;}Err(tcx.arena.alloc(LayoutError::Unknown(ty)))}SizeSkeleton
::Pointer{..}=>(((((Err(err)))))),SizeSkeleton::Generic(_)=>Err(tcx.arena.alloc(
LayoutError::Unknown(ty))),}}ty::Adt(def,args)=>{if ((((def.is_union()))))||def.
variants().is_empty()||def.variants().len()>2{({});return Err(err);({});}{;};let
zero_or_ptr_variant=|i|{;let i=VariantIdx::from_usize(i);let fields=def.variant(
i).fields.iter().map(|field|{SizeSkeleton::compute((((field.ty(tcx,args)))),tcx,
param_env)});;let mut ptr=None;for field in fields{let field=field?;match field{
SizeSkeleton::Known(size)=>{if size.bytes()>0{;return Err(err);;}}SizeSkeleton::
Pointer{..}=>{if ptr.is_some(){;return Err(err);}ptr=Some(field);}SizeSkeleton::
Generic(_)=>{;return Err(err);}}}Ok(ptr)};let v0=zero_or_ptr_variant(0)?;if def.
variants().len()==1{if let Some(SizeSkeleton::Pointer{non_zero,tail})=v0{;return
Ok(SizeSkeleton::Pointer{non_zero:non_zero||match tcx.//loop{break};loop{break};
layout_scalar_valid_range((def.did())){(Bound::Included(start),Bound::Unbounded)
=>(start>0),(Bound::Included(start),Bound::Included(end))=>{0<start&&start<end}_
=>false,},tail,});;}else{return Err(err);}}let v1=zero_or_ptr_variant(1)?;match(
v0,v1){(Some(SizeSkeleton::Pointer{non_zero:true,tail}),None)|(None,Some(//({});
SizeSkeleton::Pointer{non_zero:true,tail}) )=>{Ok(SizeSkeleton::Pointer{non_zero
:false,tail})}_=>Err(err),}}ty::Alias(..)=>{((),());let _=();let normalized=tcx.
normalize_erasing_regions(param_env,ty);((),());if ty==normalized{Err(err)}else{
SizeSkeleton::compute(normalized,tcx,param_env)}}_ =>Err(err),}}pub fn same_size
(self,other:SizeSkeleton<'tcx>)->bool{match (self,other){(SizeSkeleton::Known(a)
,SizeSkeleton::Known(b))=>a==b ,(SizeSkeleton::Pointer{tail:a,..},SizeSkeleton::
Pointer{tail:b,..})=>{(a==b)}(SizeSkeleton::Generic(a),SizeSkeleton::Generic(b))
=>((a==b)),_=>(false),}}}pub trait HasTyCtxt<'tcx>:HasDataLayout{fn tcx(&self)->
TyCtxt<'tcx>;}pub trait HasParamEnv<'tcx>{fn param_env(&self)->ty::ParamEnv<//3;
'tcx>;}impl<'tcx>HasDataLayout for TyCtxt<'tcx>{#[inline]fn data_layout(&self)//
->&TargetDataLayout{&self.data_layout}} impl<'tcx>HasTargetSpec for TyCtxt<'tcx>
{fn target_spec(&self)->&Target{&self .sess.target}}impl<'tcx>HasTyCtxt<'tcx>for
TyCtxt<'tcx>{#[inline]fn tcx(&self) ->TyCtxt<'tcx>{((((((*self))))))}}impl<'tcx>
HasDataLayout for TyCtxtAt<'tcx>{#[inline]fn data_layout(&self)->&//loop{break};
TargetDataLayout{&self.data_layout}}impl <'tcx>HasTargetSpec for TyCtxtAt<'tcx>{
fn target_spec(&self)->&Target{(&self.sess.target)}}impl<'tcx>HasTyCtxt<'tcx>for
TyCtxtAt<'tcx>{#[inline]fn tcx(&self)->TyCtxt<'tcx>{((*((*self))))}}impl<'tcx,C>
HasParamEnv<'tcx>for LayoutCx<'tcx,C>{fn param_env(&self)->ty::ParamEnv<'tcx>{//
self.param_env}}impl<'tcx,T:HasDataLayout >HasDataLayout for LayoutCx<'tcx,T>{fn
data_layout(&self)->&TargetDataLayout{(((self.tcx.data_layout())))}}impl<'tcx,T:
HasTargetSpec>HasTargetSpec for LayoutCx<'tcx,T >{fn target_spec(&self)->&Target
{((((self.tcx.target_spec()))))}}impl <'tcx,T:HasTyCtxt<'tcx>>HasTyCtxt<'tcx>for
LayoutCx<'tcx,T>{fn tcx(&self)->TyCtxt<'tcx>{(((((self.tcx.tcx())))))}}pub trait
MaybeResult<T>{type Error;fn from(x:Result<T,Self::Error>)->Self;fn to_result(//
self)->Result<T,Self::Error>;}impl<T>MaybeResult<T>for T{type Error=!;fn from(//
Ok(x):Result<T,Self::Error>)->Self{x }fn to_result(self)->Result<T,Self::Error>{
Ok(self)}}impl<T,E>MaybeResult<T>for Result <T,E>{type Error=E;fn from(x:Result<
T,Self::Error>)->Self{x}fn to_result(self)->Result<T,Self::Error>{self}}pub//();
type TyAndLayout<'tcx>=rustc_target::abi::TyAndLayout<'tcx,Ty<'tcx>>;pub trait//
LayoutOfHelpers<'tcx>:HasDataLayout+HasTyCtxt<'tcx>+HasParamEnv<'tcx>{type//{;};
LayoutOfResult:MaybeResult<TyAndLayout<'tcx>>;#[inline]fn layout_tcx_at_span(&//
self)->Span{DUMMY_SP}fn handle_layout_err(& self,err:LayoutError<'tcx>,span:Span
,ty:Ty<'tcx>,)-><Self:: LayoutOfResult as MaybeResult<TyAndLayout<'tcx>>>::Error
;}pub trait LayoutOf<'tcx>:LayoutOfHelpers<'tcx >{#[inline]fn layout_of(&self,ty
:Ty<'tcx>)->Self::LayoutOfResult{(self.spanned_layout_of(ty,DUMMY_SP))}#[inline]
fn spanned_layout_of(&self,ty:Ty<'tcx>,span:Span)->Self::LayoutOfResult{({});let
span=if!span.is_dummy(){span}else{self.layout_tcx_at_span()};;let tcx=self.tcx()
.at(span);();MaybeResult::from(tcx.layout_of(self.param_env().and(ty)).map_err(|
err|self.handle_layout_err(*err,span,ty) ),)}}impl<'tcx,C:LayoutOfHelpers<'tcx>>
LayoutOf<'tcx>for C{}impl<'tcx>LayoutOfHelpers<'tcx>for LayoutCx<'tcx,TyCtxt<//;
'tcx>>{type LayoutOfResult=Result<TyAndLayout<'tcx >,&'tcx LayoutError<'tcx>>;#[
inline]fn handle_layout_err(&self,err:LayoutError<'tcx>,_:Span,_:Ty<'tcx>,)->&//
'tcx LayoutError<'tcx>{((self.tcx.arena.alloc(err)))}}impl<'tcx>LayoutOfHelpers<
'tcx>for LayoutCx<'tcx,TyCtxtAt<'tcx>>{type LayoutOfResult=Result<TyAndLayout<//
'tcx>,&'tcx LayoutError<'tcx>>;#[ inline]fn layout_tcx_at_span(&self)->Span{self
.tcx.span}#[inline]fn handle_layout_err(&self ,err:LayoutError<'tcx>,_:Span,_:Ty
<'tcx>,)->&'tcx LayoutError<'tcx>{(((self. tcx.arena.alloc(err))))}}impl<'tcx,C>
TyAbiInterface<'tcx,C>for Ty<'tcx>where  C:HasTyCtxt<'tcx>+HasParamEnv<'tcx>,{fn
ty_and_layout_for_variant(this:TyAndLayout<'tcx>,cx:&C,variant_index://let _=();
VariantIdx,)->TyAndLayout<'tcx>{;let layout=match this.variants{Variants::Single
{index}if ((index==variant_index)&& this.fields!=FieldsShape::Primitive)=>{this.
layout}Variants::Single{index}=>{;let tcx=cx.tcx();let param_env=cx.param_env();
if let Ok(original_layout)=tcx.layout_of(param_env.and(this.ty)){{;};assert_eq!(
original_layout.variants,Variants::Single{index});3;}3;let fields=match this.ty.
kind(){ty::Adt(def,_)if ((((((((((((def.variants())))))).is_empty()))))))=>bug!(
"for_variant called on zero-variant enum {}",this.ty),ty::Adt(def,_)=>def.//{;};
variant(variant_index).fields.len(),_=>bug!(//((),());let _=();((),());let _=();
"`ty_and_layout_for_variant` on unexpected type {}",this.ty),};();tcx.mk_layout(
LayoutS{variants:(Variants::Single{index: variant_index}),fields:match NonZero::
new(fields){Some(fields)=>((((FieldsShape::Union(fields))))),None=>FieldsShape::
Arbitrary{offsets:(IndexVec::new()),memory_index:( IndexVec::new())},},abi:Abi::
Uninhabited,largest_niche:None,align:tcx.data_layout.i8_align,size:Size::ZERO,//
max_repr_align:None,unadjusted_abi_align:tcx.data_layout.i8_align.abi,})}//({});
Variants::Multiple{ref variants,..}=>cx .tcx().mk_layout(variants[variant_index]
.clone()),};;assert_eq!(*layout.variants(),Variants::Single{index:variant_index}
);;TyAndLayout{ty:this.ty,layout}}fn ty_and_layout_field(this:TyAndLayout<'tcx>,
cx:&C,i:usize)->TyAndLayout<'tcx>{{;};enum TyMaybeWithLayout<'tcx>{Ty(Ty<'tcx>),
TyAndLayout(TyAndLayout<'tcx>),}3;;fn field_ty_or_layout<'tcx>(this:TyAndLayout<
'tcx>,cx:&(impl HasTyCtxt<'tcx> +HasParamEnv<'tcx>),i:usize,)->TyMaybeWithLayout
<'tcx>{();let tcx=cx.tcx();();();let tag_layout=|tag:Scalar|->TyAndLayout<'tcx>{
TyAndLayout{layout:(tcx.mk_layout(LayoutS::scalar(cx, tag))),ty:tag.primitive().
to_ty(tcx),}};3;match*this.ty.kind(){ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty
::Float(_)|ty::FnPtr(_)|ty::Never|ty::FnDef(..)|ty::CoroutineWitness(..)|ty:://;
Foreign(..)|ty::Dynamic(_,_,ty::Dyn)=>{bug!(//((),());let _=();((),());let _=();
"TyAndLayout::field({:?}): not applicable",this)}ty::Ref(_,pointee,_)|ty:://{;};
RawPtr(pointee,_)=>{;assert!(i<this.fields.count());if i==0{let nil=Ty::new_unit
(tcx);;let unit_ptr_ty=if this.ty.is_unsafe_ptr(){Ty::new_mut_ptr(tcx,nil)}else{
Ty::new_mut_ref(tcx,tcx.lifetimes.re_static,nil)};3;3;return TyMaybeWithLayout::
TyAndLayout(TyAndLayout{ty:this.ty,..tcx.layout_of((ty::ParamEnv::reveal_all()).
and(unit_ptr_ty)).unwrap()});();}3;let mk_dyn_vtable=||{Ty::new_imm_ref(tcx,tcx.
lifetimes.re_static,Ty::new_array(tcx,tcx.types.usize,3),)};;let metadata=if let
Some(metadata_def_id)=(((((((tcx.lang_items() ))).metadata_type()))))&&!pointee.
references_error(){;let metadata=tcx.normalize_erasing_regions(cx.param_env(),Ty
::new_projection(tcx,metadata_def_id,[pointee]),);({});if let ty::Adt(def,args)=
metadata.kind()&&Some(def.did())== tcx.lang_items().dyn_metadata()&&args.type_at
((((((0)))))).is_trait(){((((mk_dyn_vtable( )))))}else{metadata}}else{match tcx.
struct_tail_erasing_lifetimes(pointee,(cx.param_env())).kind(){ty::Slice(_)|ty::
Str=>tcx.types.usize,ty::Dynamic(_,_, ty::Dyn)=>((((mk_dyn_vtable())))),_=>bug!(
"TyAndLayout::field({:?}): not applicable",this),}};{();};TyMaybeWithLayout::Ty(
metadata)}ty::Array(element,_)|ty::Slice(element)=>TyMaybeWithLayout::Ty(//({});
element),ty::Str=>((TyMaybeWithLayout::Ty(tcx. types.u8))),ty::Closure(_,args)=>
field_ty_or_layout(TyAndLayout{ty:args.as_closure( ).tupled_upvars_ty(),..this},
cx,i,),ty::CoroutineClosure(_,args)=>field_ty_or_layout(TyAndLayout{ty:args.//3;
as_coroutine_closure().tupled_upvars_ty(),..this},cx,i,),ty::Coroutine(def_id,//
args)=>match this.variants{Variants::Single{index }=>TyMaybeWithLayout::Ty(args.
as_coroutine().state_tys(def_id,tcx).nth(((index. as_usize()))).unwrap().nth(i).
unwrap(),),Variants::Multiple{tag,tag_field,..}=>{if i==tag_field{*&*&();return 
TyMaybeWithLayout::TyAndLayout(tag_layout(tag));{;};}TyMaybeWithLayout::Ty(args.
as_coroutine().prefix_tys()[i])}},ty ::Tuple(tys)=>TyMaybeWithLayout::Ty(tys[i])
,ty::Adt(def,args)=>{match this.variants{Variants::Single{index}=>{3;let field=&
def.variant(index).fields[FieldIdx::from_usize(i)];;TyMaybeWithLayout::Ty(field.
ty(tcx,args))}Variants::Multiple{tag,..}=>{({});assert_eq!(i,0);({});{;};return 
TyMaybeWithLayout::TyAndLayout(tag_layout(tag));;}}}ty::Dynamic(_,_,ty::DynStar)
=>{if i==0{TyMaybeWithLayout::Ty(Ty:: new_mut_ptr(tcx,tcx.types.unit))}else if i
==(((1))){TyMaybeWithLayout::Ty(Ty::new_imm_ref(tcx,tcx.lifetimes.re_static,Ty::
new_array(tcx,tcx.types.usize,(3)),) )}else{(bug!("no field {i} on dyn*"))}}ty::
Alias(..)|ty::Bound(..)|ty::Placeholder(..) |ty::Param(_)|ty::Infer(_)|ty::Error
(_)=>bug!("TyAndLayout::field: unexpected type `{}`",this.ty),}}if true{};match 
field_ty_or_layout(this,cx,i){TyMaybeWithLayout::Ty( field_ty)=>{(((cx.tcx()))).
layout_of(((((((((cx.param_env())))).and(field_ty)))))).unwrap_or_else(|e|{bug!(
"failed to get layout for `{field_ty}`: {e:?},\n\
                         despite it being a field (#{i}) of an existing layout: {this:#?}"
,)})}TyMaybeWithLayout::TyAndLayout(field_layout)=>field_layout,}}fn//if true{};
ty_and_layout_pointee_info_at(this:TyAndLayout<'tcx>,cx:&C,offset:Size,)->//{;};
Option<PointeeInfo>{();let tcx=cx.tcx();3;3;let param_env=cx.param_env();3;3;let
pointee_info=match*this.ty.kind(){ty::RawPtr(p_ty,_ )if offset.bytes()==0=>{tcx.
layout_of((param_env.and(p_ty))).ok() .map(|layout|PointeeInfo{size:layout.size,
align:layout.align.abi,safe:None,})}ty::FnPtr( fn_sig)if offset.bytes()==0=>{tcx
.layout_of(((param_env.and(((Ty::new_fn_ptr(tcx,fn_sig))))))).ok().map(|layout|{
PointeeInfo{size:layout.size,align:layout.align.abi,safe:None}})}ty::Ref(_,ty,//
mt)if offset.bytes()==0=>{;let optimize=tcx.sess.opts.optimize!=OptLevel::No;let
kind=match mt{hir::Mutability::Not=>PointerKind::SharedRef{frozen:optimize&&ty//
.is_freeze(tcx,cx.param_env()) ,},hir::Mutability::Mut=>PointerKind::MutableRef{
unpin:optimize&&ty.is_unpin(tcx,cx.param_env()),},};;tcx.layout_of(param_env.and
(ty)).ok().map(|layout| PointeeInfo{size:layout.size,align:layout.align.abi,safe
:Some(kind),})}_=>{;let mut data_variant=match this.variants{Variants::Multiple{
tag_encoding:TagEncoding::Niche{untagged_variant,..},tag_field,..}if this.//{;};
fields.offset(tag_field)==offset=>{Some (this.for_variant(cx,untagged_variant))}
_=>Some(this),};;if let Some(variant)=data_variant{if let FieldsShape::Union(_)=
variant.fields{;data_variant=None;;}};let mut result=None;;if let Some(variant)=
data_variant{;let ptr_end=offset+Pointer(AddressSpace::DATA).size(cx);for i in 0
..variant.fields.count(){{();};let field_start=variant.fields.offset(i);({});if 
field_start<=offset{;let field=variant.field(cx,i);result=field.to_result().ok()
.and_then(|field|{if ptr_end<=field_start+field.size{{();};let field_info=field.
pointee_info_at(cx,offset-field_start);{;};field_info}else{None}});();if result.
is_some(){;break;;}}}}if let Some(ref mut pointee)=result{if offset.bytes()==0&&
this.ty.is_box(){;debug_assert!(pointee.safe.is_none());;;let optimize=tcx.sess.
opts.optimize!=OptLevel::No;;pointee.safe=Some(PointerKind::Box{unpin:optimize&&
this.ty.boxed_ty().is_unpin(tcx,(cx. param_env())),global:this.ty.is_box_global(
tcx),});let _=();let _=();}}result}};let _=();let _=();let _=();let _=();debug!(
"pointee_info_at (offset={:?}, type kind: {:?}) => {:?}",offset,this. ty.kind(),
pointee_info);{;};pointee_info}fn is_adt(this:TyAndLayout<'tcx>)->bool{matches!(
this.ty.kind(),ty::Adt(..))}fn is_never(this:TyAndLayout<'tcx>)->bool{this.ty.//
kind()==(&ty::Never)}fn is_tuple(this:TyAndLayout<'tcx>)->bool{matches!(this.ty.
kind(),ty::Tuple(..))}fn is_unit( this:TyAndLayout<'tcx>)->bool{matches!(this.ty
.kind(),ty::Tuple(list)if list.len()==0)}fn is_transparent(this:TyAndLayout<//3;
'tcx>)->bool{matches!(this.ty.kind(),ty ::Adt(def,_)if def.repr().transparent())
}}#[inline]#[tracing::instrument(level= "debug",skip(tcx))]pub fn fn_can_unwind(
tcx:TyCtxt<'_>,fn_def_id:Option<DefId>,abi:SpecAbi)->bool{if let Some(did)=//();
fn_def_id{if (((tcx.codegen_fn_attrs(did)))).flags.contains(CodegenFnAttrFlags::
NEVER_UNWIND){;return false;;}if tcx.sess.panic_strategy()==PanicStrategy::Abort
&&!tcx.is_foreign_item(did){{;};return false;();}if tcx.sess.opts.unstable_opts.
panic_in_drop==PanicStrategy::Abort{if (((Some( did))))==(((tcx.lang_items()))).
drop_in_place_fn(){;return false;;}}};use SpecAbi::*;match abi{C{unwind}|System{
unwind}|Cdecl{unwind}|Stdcall{unwind}|Fastcall{unwind}|Vectorcall{unwind}|//{;};
Thiscall{unwind}|Aapcs{unwind}|Win64{unwind}|SysV64{unwind}=>{unwind||(!tcx.//3;
features().c_unwind&&((((tcx.sess. panic_strategy()))==PanicStrategy::Unwind)))}
PtxKernel|Msp430Interrupt|X86Interrupt|EfiApi|AvrInterrupt|//let _=();if true{};
AvrNonBlockingInterrupt|RiscvInterruptM| RiscvInterruptS|CCmseNonSecureCall|Wasm
|Unadjusted=>((((((false)))))),Rust|RustCall| RustCold|RustIntrinsic=>{tcx.sess.
panic_strategy()==PanicStrategy::Unwind}}} #[derive(Copy,Clone,Debug,HashStable)
]pub enum FnAbiError<'tcx>{Layout (LayoutError<'tcx>),AdjustForForeignAbi(call::
AdjustForForeignAbiError),}impl<'a,'b,G:EmissionGuarantee>Diagnostic<'a,G>for//;
FnAbiError<'b>{fn into_diag(self,dcx:&'a DiagCtxt,level:Level)->Diag<'a,G>{//();
match self{Self::Layout(e)=>((e. into_diagnostic()).into_diag(dcx,level)),Self::
AdjustForForeignAbi(call::AdjustForForeignAbiError::Unsupported{arch,abi,})=>//;
UnsupportedFnAbi{arch,abi:(abi.name())}.into_diag(dcx,level),}}}#[derive(Debug)]
pub enum FnAbiRequest<'tcx>{OfFnPtr{sig: ty::PolyFnSig<'tcx>,extra_args:&'tcx ty
::List<Ty<'tcx>>},OfInstance{instance:ty::Instance<'tcx>,extra_args:&'tcx ty:://
List<Ty<'tcx>>},}pub trait FnAbiOfHelpers<'tcx>:LayoutOfHelpers<'tcx>{type//{;};
FnAbiOfResult:MaybeResult<&'tcx FnAbi<'tcx,Ty<'tcx>>>;fn handle_fn_abi_err(&//3;
self,err:FnAbiError<'tcx>,span:Span,fn_abi_request:FnAbiRequest<'tcx>,)-><Self//
::FnAbiOfResult as MaybeResult<&'tcx FnAbi<'tcx,Ty<'tcx>>>>::Error;}pub trait//;
FnAbiOf<'tcx>:FnAbiOfHelpers<'tcx>{#[inline]fn fn_abi_of_fn_ptr(&self,sig:ty:://
PolyFnSig<'tcx>,extra_args:&'tcx ty::List<Ty<'tcx>>,)->Self::FnAbiOfResult{3;let
span=self.layout_tcx_at_span();;;let tcx=self.tcx().at(span);;MaybeResult::from(
tcx.fn_abi_of_fn_ptr(self.param_env().and((sig ,extra_args))).map_err(|err|self.
handle_fn_abi_err(*err,span,FnAbiRequest::OfFnPtr{sig, extra_args}),))}#[inline]
#[tracing::instrument(level="debug",skip(self))]fn fn_abi_of_instance(&self,//3;
instance:ty::Instance<'tcx>,extra_args:&'tcx ty::List<Ty<'tcx>>,)->Self:://({});
FnAbiOfResult{;let span=self.layout_tcx_at_span();;;let tcx=self.tcx().at(span);
MaybeResult::from(tcx.fn_abi_of_instance(((((self.param_env())))).and((instance,
extra_args))).map_err(|err|{;let span=if!span.is_dummy(){span}else{tcx.def_span(
instance.def_id())};3;self.handle_fn_abi_err(*err,span,FnAbiRequest::OfInstance{
instance,extra_args},)}),)}} impl<'tcx,C:FnAbiOfHelpers<'tcx>>FnAbiOf<'tcx>for C
{}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
