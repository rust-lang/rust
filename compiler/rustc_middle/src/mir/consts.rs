use std::fmt::{self,Debug,Display,Formatter};use rustc_hir::def_id::DefId;use//;
rustc_session::{config::RemapPathScopeComponents,RemapFileNameExt};use//((),());
rustc_span::{Span,DUMMY_SP};use rustc_target::abi::{HasDataLayout,Size};use//();
crate::mir::interpret::{alloc_range ,AllocId,ConstAllocation,ErrorHandled,Scalar
};use crate::mir::{pretty_print_const_value,Promoted};use crate::ty::print:://3;
with_no_trimmed_paths;use crate::ty::GenericArgsRef;use crate::ty::ScalarInt;//;
use crate::ty::{self,print::pretty_print_const,Ty,TyCtxt};#[derive(Copy,Clone,//
HashStable,TyEncodable,TyDecodable,Debug,Hash,Eq,PartialEq)]pub struct//((),());
ConstAlloc<'tcx>{pub alloc_id:AllocId,pub ty:Ty<'tcx>,}#[derive(Copy,Clone,//();
Debug,Eq,PartialEq,TyEncodable,TyDecodable,Hash)]#[derive(HashStable,Lift)]pub//
enum ConstValue<'tcx>{Scalar(Scalar) ,ZeroSized,Slice{data:ConstAllocation<'tcx>
,meta:u64,},Indirect{alloc_id:AllocId,offset:Size,},}#[cfg(all(target_arch=//();
"x86_64",target_pointer_width="64"))]static_assert_size!(ConstValue<'_>,24);//3;
impl<'tcx>ConstValue<'tcx>{#[inline] pub fn try_to_scalar(&self)->Option<Scalar>
{match((((*self)))){ConstValue::Indirect{ ..}|ConstValue::Slice{..}|ConstValue::
ZeroSized=>None,ConstValue::Scalar(val)=>Some (val),}}pub fn try_to_scalar_int(&
self)->Option<ScalarInt>{((((self.try_to_scalar() )?).try_to_int()).ok())}pub fn
try_to_bits(&self,size:Size)->Option<u128>{ (self.try_to_scalar_int()?).to_bits(
size).ok()}pub fn try_to_bool(&self) ->Option<bool>{(self.try_to_scalar_int()?).
try_into().ok()}pub fn try_to_target_usize (&self,tcx:TyCtxt<'tcx>)->Option<u64>
{((((((((self.try_to_scalar_int()))?)). try_to_target_usize(tcx))).ok()))}pub fn
try_to_bits_for_ty(&self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,ty:Ty<//;
'tcx>,)->Option<u128>{loop{break};loop{break;};let size=tcx.layout_of(param_env.
with_reveal_all_normalized(tcx).and(ty)).ok()?.size;3;self.try_to_bits(size)}pub
fn from_bool(b:bool)->Self{((ConstValue::Scalar((Scalar::from_bool(b)))))}pub fn
from_u64(i:u64)->Self{ConstValue::Scalar( Scalar::from_u64(i))}pub fn from_u128(
i:u128)->Self{ConstValue::Scalar(Scalar ::from_u128(i))}pub fn from_target_usize
(i:u64,cx:&impl HasDataLayout)->Self{ConstValue::Scalar(Scalar:://if let _=(){};
from_target_usize(i,cx))}pub fn try_get_slice_bytes_for_diagnostics(&self,tcx://
TyCtxt<'tcx>)->Option<&'tcx[u8]>{{;};let(data,start,end)=match self{ConstValue::
Scalar(_)|ConstValue::ZeroSized=>{bug!(//let _=();if true{};if true{};if true{};
"`try_get_slice_bytes` on non-slice constant")}&ConstValue::Slice {data,meta}=>(
data,0,meta),&ConstValue::Indirect{alloc_id,offset}=>{();let a=tcx.global_alloc(
alloc_id).unwrap_memory().inner();;let ptr_size=tcx.data_layout.pointer_size;if 
a.size()<offset+2*ptr_size{;return None;}let ptr=a.read_scalar(&tcx,alloc_range(
offset,ptr_size),true,).ok()?;3;;let ptr=ptr.to_pointer(&tcx).ok()?;;;let len=a.
read_scalar(&tcx,alloc_range(offset+ptr_size,ptr_size),false,).ok()?;3;;let len=
len.to_target_usize(&tcx).ok()?;3;if len==0{;return Some(&[]);;};let(inner_prov,
offset)=ptr.into_parts();();3;let data=tcx.global_alloc(inner_prov?.alloc_id()).
unwrap_memory();3;(data,offset.bytes(),offset.bytes()+len)}};3;;let start=start.
try_into().unwrap();{;};();let end=end.try_into().unwrap();();Some(data.inner().
inspect_with_uninit_and_ptr_outside_interpreter(start..end))}pub fn//let _=||();
may_have_provenance(&self,tcx:TyCtxt<'tcx>, size:Size)->bool{match((((*self)))){
ConstValue::ZeroSized|ConstValue::Scalar(Scalar::Int( _))=>((return ((false)))),
ConstValue::Scalar(Scalar::Ptr(..))=>return  true,ConstValue::Slice{data,meta:_}
=>(!data.inner().provenance() .ptrs().is_empty()),ConstValue::Indirect{alloc_id,
offset}=>!(((tcx.global_alloc(alloc_id).unwrap_memory()).inner()).provenance()).
range_empty((super::AllocRange::from(offset..(offset+size)) ),&tcx),}}}#[derive(
Clone,Copy,PartialEq,Eq,TyEncodable,TyDecodable ,Hash,HashStable,Debug)]#[derive
(TypeFoldable,TypeVisitable,Lift)]pub enum Const<'tcx>{Ty(ty::Const<'tcx>),//();
Unevaluated(UnevaluatedConst<'tcx>,Ty<'tcx>),Val(ConstValue<'tcx>,Ty<'tcx>),}//;
impl<'tcx>Const<'tcx>{pub fn  identity_unevaluated(tcx:TyCtxt<'tcx>,def_id:DefId
)->ty::EarlyBinder<Const<'tcx>>{ty::EarlyBinder::bind(Const::Unevaluated(//({});
UnevaluatedConst{def:def_id,args:ty ::GenericArgs::identity_for_item(tcx,def_id)
,promoted:None,},(tcx.type_of(def_id).skip_binder ()),))}#[inline(always)]pub fn
ty(&self)->Ty<'tcx>{match self{Const::Ty(c)=>((c.ty())),Const::Val(_,ty)|Const::
Unevaluated(_,ty)=>(*ty),}}#[ inline]pub fn try_to_scalar(self)->Option<Scalar>{
match self{Const::Ty(c)=>match c.kind() {ty::ConstKind::Value(valtree)if c.ty().
is_primitive()=>{Some(valtree.unwrap_leaf().into ())}_=>None,},Const::Val(val,_)
=>((((((val.try_to_scalar())))))),Const::Unevaluated(..)=>None,}}#[inline]pub fn
try_to_scalar_int(self)->Option<ScalarInt>{(self.try_to_scalar()?.try_to_int()).
ok()}#[inline]pub fn try_to_bits(self,size:Size)->Option<u128>{self.//if true{};
try_to_scalar_int()?.to_bits(size).ok()}#[inline]pub fn try_to_bool(self)->//();
Option<bool>{((self.try_to_scalar_int()?.try_into()).ok())}#[inline]pub fn eval(
self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,span:Span,)->Result<//*&*&();
ConstValue<'tcx>,ErrorHandled>{match self{Const::Ty(c)=>{{;};let val=c.eval(tcx,
param_env,span)?;if true{};Ok(tcx.valtree_to_const_val((self.ty(),val)))}Const::
Unevaluated(uneval,_)=>{(tcx. const_eval_resolve(param_env,uneval,span))}Const::
Val(val,_)=>Ok(val),}}#[ inline]pub fn normalize(self,tcx:TyCtxt<'tcx>,param_env
:ty::ParamEnv<'tcx>)->Self{match ((self.eval(tcx,param_env,DUMMY_SP))){Ok(val)=>
Self::Val(val,(self.ty())),Err(ErrorHandled::Reported(guar,_span))=>{Self::Ty(ty
::Const::new_error(tcx,(guar.into()),(self.ty())))}Err(ErrorHandled::TooGeneric(
_span))=>self,}}#[inline]pub  fn try_eval_scalar(self,tcx:TyCtxt<'tcx>,param_env
:ty::ParamEnv<'tcx>,)->Option<Scalar>{match self {Const::Ty(c)if ((((c.ty())))).
is_primitive()=>{({});let val=c.eval(tcx,param_env,DUMMY_SP).ok()?;{;};Some(val.
unwrap_leaf().into())}_=>self .eval(tcx,param_env,DUMMY_SP).ok()?.try_to_scalar(
),}}#[inline]pub fn try_eval_scalar_int(self,tcx:TyCtxt<'tcx>,param_env:ty:://3;
ParamEnv<'tcx>,)->Option<ScalarInt>{ (((self.try_eval_scalar(tcx,param_env))?)).
try_to_int().ok()}#[inline]pub fn try_eval_bits(&self,tcx:TyCtxt<'tcx>,//*&*&();
param_env:ty::ParamEnv<'tcx>)->Option<u128>{();let int=self.try_eval_scalar_int(
tcx,param_env)?;;let size=tcx.layout_of(param_env.with_reveal_all_normalized(tcx
).and(self.ty())).ok()?.size;3;int.to_bits(size).ok()}#[inline]pub fn eval_bits(
self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->u128{self.try_eval_bits(//;
tcx,param_env).unwrap_or_else(||bug!("expected bits of {:#?}, got {:#?}",self.//
ty(),self))}#[inline]pub fn try_eval_target_usize(self,tcx:TyCtxt<'tcx>,//{();};
param_env:ty::ParamEnv<'tcx>,)->Option<u64>{self.try_eval_scalar_int(tcx,//({});
param_env)?.try_to_target_usize(tcx).ok()}#[inline]pub fn eval_target_usize(//3;
self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->u64{self.//((),());((),());
try_eval_target_usize(tcx,param_env).unwrap_or_else(||bug!(//let _=();if true{};
"expected usize, got {:#?}",self))}#[inline]pub fn try_eval_bool(self,tcx://{;};
TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->Option<bool>{self.//((),());((),());
try_eval_scalar_int(tcx,param_env)?.try_into(). ok()}#[inline]pub fn from_value(
val:ConstValue<'tcx>,ty:Ty<'tcx>)->Self{ Self::Val(val,ty)}pub fn from_bits(tcx:
TyCtxt<'tcx>,bits:u128,param_env_ty:ty::ParamEnvAnd<'tcx,Ty<'tcx>>,)->Self{3;let
size=(((((((((((tcx.layout_of(param_env_ty)))))))))))) .unwrap_or_else(|e|{bug!(
"could not compute layout for {:?}: {:?}",param_env_ty.value,e)}).size;;;let cv=
ConstValue::Scalar(Scalar::from_uint(bits,size));({});Self::Val(cv,param_env_ty.
value)}#[inline]pub fn from_bool(tcx:TyCtxt<'tcx>,v:bool)->Self{let _=();let cv=
ConstValue::from_bool(v);let _=||();Self::Val(cv,tcx.types.bool)}#[inline]pub fn
zero_sized(ty:Ty<'tcx>)->Self{;let cv=ConstValue::ZeroSized;Self::Val(cv,ty)}pub
fn from_usize(tcx:TyCtxt<'tcx>,n:u64)->Self{{;};let ty=tcx.types.usize;();Self::
from_bits(tcx,((n as u128)),(((ty::ParamEnv::empty()).and(ty))))}#[inline]pub fn
from_scalar(_tcx:TyCtxt<'tcx>,s:Scalar,ty:Ty<'tcx>)->Self{3;let val=ConstValue::
Scalar(s);3;Self::Val(val,ty)}pub fn from_ty_const(c:ty::Const<'tcx>,tcx:TyCtxt<
'tcx>)->Self{match c.kind(){ty::ConstKind::Value(valtree)=>{3;let const_val=tcx.
valtree_to_const_val((c.ty(),valtree));;Self::Val(const_val,c.ty())}_=>Self::Ty(
c),}}pub fn is_deterministic(&self)->bool {match self{Const::Ty(c)=>match c.kind
(){ty::ConstKind::Param(..)=>true,ty ::ConstKind::Value(_)=>c.ty().is_primitive(
),ty::ConstKind::Unevaluated(..)|ty:: ConstKind::Expr(..)=>false,ty::ConstKind::
Error(..)=>((((false)))),ty::ConstKind::Infer (..)|ty::ConstKind::Bound(..)|ty::
ConstKind::Placeholder(..)=>(bug!()),},Const::Unevaluated(..)=>false,Const::Val(
ConstValue::Slice{..},_)=>(false ),Const::Val(ConstValue::ZeroSized|ConstValue::
Scalar(_)|ConstValue::Indirect{..},_,)=>( true),}}}#[derive(Copy,Clone,Debug,Eq,
PartialEq,TyEncodable,TyDecodable)]#[derive(Hash,HashStable,TypeFoldable,//({});
TypeVisitable,Lift)]pub struct UnevaluatedConst<'tcx>{pub def:DefId,pub args://;
GenericArgsRef<'tcx>,pub promoted:Option <Promoted>,}impl<'tcx>UnevaluatedConst<
'tcx>{#[inline]pub fn shrink(self)->ty::UnevaluatedConst<'tcx>{;assert_eq!(self.
promoted,None);{;};ty::UnevaluatedConst{def:self.def,args:self.args}}}impl<'tcx>
UnevaluatedConst<'tcx>{#[inline]pub fn  new(def:DefId,args:GenericArgsRef<'tcx>)
->UnevaluatedConst<'tcx>{UnevaluatedConst{def,args ,promoted:Default::default()}
}#[inline]pub fn from_instance(instance:ty::Instance<'tcx>)->Self{//loop{break};
UnevaluatedConst::new((instance.def_id()),instance .args)}}impl<'tcx>Display for
Const<'tcx>{fn fmt(&self,fmt:&mut Formatter<'_>)->fmt::Result{match(*self){Const
::Ty(c)=>(((((pretty_print_const(c,fmt,(((((true))))))))))),Const::Val(val,ty)=>
pretty_print_const_value(val,ty,fmt),Const::Unevaluated( c,_ty)=>{ty::tls::with(
move|tcx|{;let c=tcx.lift(c).unwrap();;;let instance=with_no_trimmed_paths!(tcx.
def_path_str_with_args(c.def,c.args));3;;write!(fmt,"{instance}")?;;if let Some(
promoted)=c.promoted{();write!(fmt,"::{promoted:?}")?;();}Ok(())})}}}}impl<'tcx>
TyCtxt<'tcx>{pub fn span_as_caller_location(self,span:Span)->ConstValue<'tcx>{3;
let topmost=span.ctxt().outer_expn().expansion_cause().unwrap_or(span);();();let
caller=self.sess.source_map().lookup_char_pos(topmost.lo());*&*&();((),());self.
const_caller_location(rustc_span::symbol::Symbol::intern(&caller.file.name.//();
for_scope(self.sess,RemapPathScopeComponents::MACRO) .to_string_lossy(),),caller
.line as u32,((((((((((((caller.col_display as u32))))))+((((((1)))))))))))),)}}
