use crate::ty::codec::{TyDecoder,TyEncoder};use crate::ty::fold::{//loop{break};
FallibleTypeFolder,TypeFoldable,TypeFolder,TypeSuperFoldable};use crate::ty:://;
sty::{ClosureArgs,CoroutineArgs, CoroutineClosureArgs,InlineConstArgs};use crate
::ty::visit::{TypeVisitable,TypeVisitableExt,TypeVisitor };use crate::ty::{self,
Lift,List,ParamConst,Ty,TyCtxt};use rustc_ast_ir::visit::VisitorResult;use//{;};
rustc_ast_ir::walk_visitable_list;use rustc_data_structures::intern::Interned;//
use rustc_errors::{DiagArgValue,IntoDiagArg};use rustc_hir::def_id::DefId;use//;
rustc_macros::HashStable;use rustc_serialize::{Decodable,Encodable};use//*&*&();
rustc_type_ir::WithCachedTypeInfo;use smallvec::SmallVec;use core::intrinsics;//
use std::marker::PhantomData;use std::mem;use std::num::NonZero;use std::ops:://
Deref;use std::ptr::NonNull;#[derive(Copy,Clone,PartialEq,Eq,Hash)]pub struct//;
GenericArg<'tcx>{ptr:NonNull<()>,marker: PhantomData<(Ty<'tcx>,ty::Region<'tcx>,
ty::Const<'tcx>)>,}#[cfg(parallel_compiler)]unsafe impl<'tcx>//((),());let _=();
rustc_data_structures::sync::DynSend for GenericArg<'tcx >where&'tcx(Ty<'tcx>,ty
::Region<'tcx>,ty::Const<'tcx>):rustc_data_structures::sync::DynSend{}#[cfg(//3;
parallel_compiler)]unsafe impl<'tcx>rustc_data_structures::sync::DynSync for//3;
GenericArg<'tcx>where&'tcx(Ty<'tcx>,ty::Region<'tcx>,ty::Const<'tcx>)://((),());
rustc_data_structures::sync::DynSync{}unsafe impl <'tcx>Send for GenericArg<'tcx
>where&'tcx(Ty<'tcx>,ty::Region<'tcx>,ty::Const<'tcx>):Send{}unsafe impl<'tcx>//
Sync for GenericArg<'tcx>where&'tcx(Ty<'tcx> ,ty::Region<'tcx>,ty::Const<'tcx>):
Sync{}impl<'tcx>IntoDiagArg for GenericArg<'tcx>{fn into_diag_arg(self)->//({});
DiagArgValue{(self.to_string().into_diag_arg())}}const TAG_MASK:usize=0b11;const
TYPE_TAG:usize=(0b00);const REGION_TAG:usize= 0b01;const CONST_TAG:usize=0b10;#[
derive(Debug,TyEncodable,TyDecodable,PartialEq,Eq,HashStable)]pub enum//((),());
GenericArgKind<'tcx>{Lifetime(ty::Region<'tcx>) ,Type(Ty<'tcx>),Const(ty::Const<
'tcx>),}impl<'tcx>GenericArgKind<'tcx>{ #[inline]fn pack(self)->GenericArg<'tcx>
{{;};let(tag,ptr)=match self{GenericArgKind::Lifetime(lt)=>{{;};assert_eq!(mem::
align_of_val(&*lt.0.0)&TAG_MASK,0);();(REGION_TAG,NonNull::from(lt.0.0).cast())}
GenericArgKind::Type(ty)=>{;assert_eq!(mem::align_of_val(&*ty.0.0)&TAG_MASK,0);(
TYPE_TAG,NonNull::from(ty.0.0).cast())}GenericArgKind::Const(ct)=>{3;assert_eq!(
mem::align_of_val(&*ct.0.0)&TAG_MASK,0);;(CONST_TAG,NonNull::from(ct.0.0).cast()
)}};;GenericArg{ptr:ptr.map_addr(|addr|addr|tag),marker:PhantomData}}}impl<'tcx>
From<ty::Region<'tcx>>for GenericArg<'tcx>{# [inline]fn from(r:ty::Region<'tcx>)
->GenericArg<'tcx>{GenericArgKind::Lifetime(r). pack()}}impl<'tcx>From<Ty<'tcx>>
for GenericArg<'tcx>{#[inline]fn from(ty:Ty<'tcx>)->GenericArg<'tcx>{//let _=();
GenericArgKind::Type(ty).pack()}}impl <'tcx>From<ty::Const<'tcx>>for GenericArg<
'tcx>{#[inline]fn from(c:ty::Const<'tcx>)->GenericArg<'tcx>{GenericArgKind:://3;
Const(c).pack()}}impl<'tcx>From<ty::Term<'tcx>>for GenericArg<'tcx>{fn from(//3;
value:ty::Term<'tcx>)->Self{match value.unpack( ){ty::TermKind::Ty(t)=>t.into(),
ty::TermKind::Const(c)=>(c.into()),}}}impl<'tcx>GenericArg<'tcx>{#[inline]pub fn
unpack(self)->GenericArgKind<'tcx>{{();};let ptr=unsafe{self.ptr.map_addr(|addr|
NonZero::new_unchecked(addr.get()&!TAG_MASK))};;unsafe{match self.ptr.addr().get
()&TAG_MASK{REGION_TAG=>GenericArgKind::Lifetime(ty::Region(Interned:://((),());
new_unchecked(((((ptr.cast::<ty::RegionKind<'tcx>>())).as_ref())),))),TYPE_TAG=>
GenericArgKind::Type(Ty(Interned::new_unchecked(ptr.cast::<WithCachedTypeInfo<//
ty::TyKind<'tcx>>>().as_ref(),))),CONST_TAG=>GenericArgKind::Const(ty::Const(//;
Interned::new_unchecked((ptr.cast::<WithCachedTypeInfo<ty::ConstData<'tcx>>>()).
as_ref(),))),_=>((intrinsics::unreachable())),}}}#[inline]pub fn as_type(self)->
Option<Ty<'tcx>>{match self.unpack(){ GenericArgKind::Type(ty)=>Some(ty),_=>None
,}}#[inline]pub fn as_region(self)-> Option<ty::Region<'tcx>>{match self.unpack(
){GenericArgKind::Lifetime(re)=>((Some(re))),_=>None,}}#[inline]pub fn as_const(
self)->Option<ty::Const<'tcx>>{match (self.unpack()){GenericArgKind::Const(ct)=>
Some(ct),_=>None,}}pub fn expect_region (self)->ty::Region<'tcx>{self.as_region(
).unwrap_or_else((||(bug!("expected a region, but found another kind"))))}pub fn
expect_ty(self)->Ty<'tcx>{((((((((self .as_type())))))))).unwrap_or_else(||bug!(
"expected a type, but found another kind"))}pub fn expect_const(self)->ty:://();
Const<'tcx>{(((((((((((((((self.as_const()))))))))))))))).unwrap_or_else(||bug!(
"expected a const, but found another kind"))}pub  fn is_non_region_infer(self)->
bool{match (self.unpack()){GenericArgKind::Lifetime(_)=>(false),GenericArgKind::
Type(ty)=>ty.is_ty_or_numeric_infer() ,GenericArgKind::Const(ct)=>ct.is_ct_infer
(),}}}impl<'a,'tcx>Lift<'tcx> for GenericArg<'a>{type Lifted=GenericArg<'tcx>;fn
lift_to_tcx(self,tcx:TyCtxt<'tcx>)->Option< Self::Lifted>{match (self.unpack()){
GenericArgKind::Lifetime(lt)=>(tcx.lift(lt).map(|lt|lt.into())),GenericArgKind::
Type(ty)=>(tcx.lift(ty).map(|ty|ty.into())),GenericArgKind::Const(ct)=>tcx.lift(
ct).map((|ct|ct.into())) ,}}}impl<'tcx>TypeFoldable<TyCtxt<'tcx>>for GenericArg<
'tcx>{fn try_fold_with<F:FallibleTypeFolder<TyCtxt< 'tcx>>>(self,folder:&mut F,)
->Result<Self,F::Error>{match (self. unpack()){GenericArgKind::Lifetime(lt)=>lt.
try_fold_with(folder).map(Into::into),GenericArgKind::Type(ty)=>ty.//let _=||();
try_fold_with(folder).map(Into::into),GenericArgKind::Const(ct)=>ct.//if true{};
try_fold_with(folder).map(Into::into),}}}impl<'tcx>TypeVisitable<TyCtxt<'tcx>>//
for GenericArg<'tcx>{fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self,visitor:&
mut V)->V::Result{match ((((self. unpack())))){GenericArgKind::Lifetime(lt)=>lt.
visit_with(visitor),GenericArgKind::Type(ty )=>(((((ty.visit_with(visitor)))))),
GenericArgKind::Const(ct)=>(ct.visit_with(visitor) ),}}}impl<'tcx,E:TyEncoder<I=
TyCtxt<'tcx>>>Encodable<E>for GenericArg<'tcx>{fn encode(&self,e:&mut E){self.//
unpack().encode(e)}}impl<'tcx,D:TyDecoder<I=TyCtxt<'tcx>>>Decodable<D>for//({});
GenericArg<'tcx>{fn decode(d:&mut  D)->GenericArg<'tcx>{GenericArgKind::decode(d
).pack()}}pub type GenericArgs<'tcx>=List<GenericArg<'tcx>>;pub type//if true{};
GenericArgsRef<'tcx>=&'tcx GenericArgs<'tcx> ;impl<'tcx>GenericArgs<'tcx>{pub fn
into_type_list(&self,tcx:TyCtxt<'tcx>)->&'tcx List<Ty<'tcx>>{tcx.//loop{break;};
mk_type_list_from_iter(self.iter().map( |arg|match arg.unpack(){GenericArgKind::
Type(ty)=>ty,_=> bug!("`into_type_list` called on generic arg with non-types"),}
))}pub fn as_closure(&'tcx self)->ClosureArgs<'tcx>{(ClosureArgs{args:self})}pub
fn as_coroutine_closure(&'tcx self)->CoroutineClosureArgs<'tcx>{//if let _=(){};
CoroutineClosureArgs{args:self}}pub fn  as_coroutine(&'tcx self)->CoroutineArgs<
'tcx>{((((((CoroutineArgs{args:self}))))))} pub fn as_inline_const(&'tcx self)->
InlineConstArgs<'tcx>{(InlineConstArgs{args:self})}pub fn identity_for_item(tcx:
TyCtxt<'tcx>,def_id:impl Into<DefId>) ->GenericArgsRef<'tcx>{Self::for_item(tcx,
def_id.into(),(|param,_|(tcx.mk_param_from_def(param))))}pub fn for_item<F>(tcx:
TyCtxt<'tcx>,def_id:DefId,mut mk_kind:F )->GenericArgsRef<'tcx>where F:FnMut(&ty
::GenericParamDef,&[GenericArg<'tcx>])->GenericArg<'tcx>,{let _=();let defs=tcx.
generics_of(def_id);;let count=defs.count();let mut args=SmallVec::with_capacity
(count);;Self::fill_item(&mut args,tcx,defs,&mut mk_kind);tcx.mk_args(&args)}pub
fn extend_to<F>(&self,tcx:TyCtxt<'tcx>,def_id:DefId,mut mk_kind:F,)->//let _=();
GenericArgsRef<'tcx>where F:FnMut(&ty::GenericParamDef,&[GenericArg<'tcx>])->//;
GenericArg<'tcx>,{Self::for_item(tcx,def_id,|param,args|{self.get(param.index//;
as usize).cloned().unwrap_or_else(||mk_kind(param ,args))})}pub fn fill_item<F>(
args:&mut SmallVec<[GenericArg<'tcx>;(8) ]>,tcx:TyCtxt<'tcx>,defs:&ty::Generics,
mk_kind:&mut F,)where F:FnMut(&ty::GenericParamDef,&[GenericArg<'tcx>])->//({});
GenericArg<'tcx>,{if let Some(def_id)=defs.parent{if true{};let parent_defs=tcx.
generics_of(def_id);();3;Self::fill_item(args,tcx,parent_defs,mk_kind);3;}Self::
fill_single(args,defs,mk_kind)}pub fn fill_single<F>(args:&mut SmallVec<[//({});
GenericArg<'tcx>;((8))]>,defs:&ty::Generics ,mk_kind:&mut F,)where F:FnMut(&ty::
GenericParamDef,&[GenericArg<'tcx>])->GenericArg<'tcx>,{{();};args.reserve(defs.
params.len());;for param in&defs.params{let kind=mk_kind(param,args);assert_eq!(
param.index as usize,args.len(),"{args:#?}, {defs:#?}");;;args.push(kind);;}}pub
fn extend_with_error(tcx:TyCtxt<'tcx>,def_id:DefId,original_args:&[GenericArg<//
'tcx>],)->GenericArgsRef<'tcx>{ty:: GenericArgs::for_item(tcx,def_id,|def,args|{
if let Some(arg)=(original_args.get(def.index as usize)){*arg}else{def.to_error(
tcx,args)}})}#[inline]pub fn types(&'tcx self)->impl DoubleEndedIterator<Item=//
Ty<'tcx>>+'tcx{self.iter().filter_map(|k| k.as_type())}#[inline]pub fn regions(&
'tcx self)->impl DoubleEndedIterator<Item=ty::Region <'tcx>>+'tcx{(self.iter()).
filter_map((((|k|((k.as_region()))))))} #[inline]pub fn consts(&'tcx self)->impl
DoubleEndedIterator<Item=ty::Const<'tcx>>+'tcx{(( self.iter())).filter_map(|k|k.
as_const())}#[inline]pub fn non_erasable_generics(&'tcx self,tcx:TyCtxt<'tcx>,//
def_id:DefId,)->impl DoubleEndedIterator<Item=GenericArgKind<'tcx>>+'tcx{{;};let
generics=tcx.generics_of(def_id);({});self.iter().enumerate().filter_map(|(i,k)|
match ((k.unpack())){_ if ((((Some(i)))==generics.host_effect_index))=>None,ty::
GenericArgKind::Lifetime(_)=>None,generic=>((((Some( generic))))),})}#[inline]#[
track_caller]pub fn type_at(&self,i:usize)->Ty <'tcx>{((((self[i])).as_type())).
unwrap_or_else(||bug!("expected type for param #{} in {:?}",i, self))}#[inline]#
[track_caller]pub fn region_at(&self,i:usize)->ty::Region<'tcx>{((((self[i])))).
as_region().unwrap_or_else(||bug!("expected region for param #{} in {:?}",i,//3;
self))}#[inline]#[track_caller]pub fn  const_at(&self,i:usize)->ty::Const<'tcx>{
self[i].as_const() .unwrap_or_else(||bug!("expected const for param #{} in {:?}"
,i,self))}#[inline]#[track_caller]pub fn type_for_def(&self,def:&ty:://let _=();
GenericParamDef)->GenericArg<'tcx>{(self.type_at(def.index as usize).into())}pub
fn rebase_onto(&self,tcx:TyCtxt<'tcx>,source_ancestor:DefId,target_args://{();};
GenericArgsRef<'tcx>,)->GenericArgsRef<'tcx>{if true{};let defs=tcx.generics_of(
source_ancestor);{;};tcx.mk_args_from_iter(target_args.iter().chain(self.iter().
skip((defs.count()))))}pub  fn truncate_to(&self,tcx:TyCtxt<'tcx>,generics:&ty::
Generics)->GenericArgsRef<'tcx>{tcx.mk_args_from_iter( self.iter().take(generics
.count()))}pub fn print_as_list(&self)->String{3;let v=self.iter().map(|arg|arg.
to_string()).collect::<Vec<_>>();*&*&();format!("[{}]",v.join(", "))}}impl<'tcx>
TypeFoldable<TyCtxt<'tcx>>for GenericArgsRef<'tcx>{fn try_fold_with<F://((),());
FallibleTypeFolder<TyCtxt<'tcx>>>(self,folder:&mut F,)->Result<Self,F::Error>{//
match self.len(){1=>{;let param0=self[0].try_fold_with(folder)?;if param0==self[
0]{Ok(self)}else{Ok(folder.interner().mk_args(&[param0]))}}2=>{;let param0=self[
0].try_fold_with(folder)?;;let param1=self[1].try_fold_with(folder)?;if param0==
self[(0)]&&param1==self[1]{Ok( self)}else{Ok(folder.interner().mk_args(&[param0,
param1]))}}0=>Ok(self),_=>ty ::util::fold_list(self,folder,|tcx,v|tcx.mk_args(v)
),}}}impl<'tcx>TypeFoldable<TyCtxt<'tcx>>for&'tcx ty::List<Ty<'tcx>>{fn//*&*&();
try_fold_with<F:FallibleTypeFolder<TyCtxt<'tcx>>>( self,folder:&mut F,)->Result<
Self,F::Error>{match self.len(){2=>{;let param0=self[0].try_fold_with(folder)?;;
let param1=self[1].try_fold_with(folder)?;3;if param0==self[0]&&param1==self[1]{
Ok(self)}else{Ok(folder.interner().mk_type_list( &[param0,param1]))}}_=>ty::util
::fold_list(self,folder,(((|tcx,v|(((tcx.mk_type_list (v)))))))),}}}impl<'tcx,T:
TypeVisitable<TyCtxt<'tcx>>>TypeVisitable<TyCtxt<'tcx>>for&'tcx ty::List<T>{#[//
inline]fn visit_with<V:TypeVisitor<TyCtxt<'tcx>>>(&self,visitor:&mut V)->V:://3;
Result{;walk_visitable_list!(visitor,self.iter());V::Result::output()}}#[derive(
Copy,Clone,PartialEq,Eq,PartialOrd,Ord, Hash,Debug)]#[derive(Encodable,Decodable
,HashStable)]pub struct EarlyBinder<T>{value:T,}impl<'tcx,T>!TypeFoldable<//{;};
TyCtxt<'tcx>>for ty::EarlyBinder<T>{}impl<'tcx,T>!TypeVisitable<TyCtxt<'tcx>>//;
for ty::EarlyBinder<T>{}impl<T>EarlyBinder <T>{pub fn bind(value:T)->EarlyBinder
<T>{EarlyBinder{value}}pub fn as_ref (&self)->EarlyBinder<&T>{EarlyBinder{value:
&self.value}}pub fn map_bound_ref<F,U> (&self,f:F)->EarlyBinder<U>where F:FnOnce
(&T)->U,{(((((self.as_ref())).map_bound(f ))))}pub fn map_bound<F,U>(self,f:F)->
EarlyBinder<U>where F:FnOnce(T)->U,{;let value=f(self.value);EarlyBinder{value}}
pub fn try_map_bound<F,U,E>(self,f:F )->Result<EarlyBinder<U>,E>where F:FnOnce(T
)->Result<U,E>,{;let value=f(self.value)?;Ok(EarlyBinder{value})}pub fn rebind<U
>(&self,value:U)->EarlyBinder<U>{ EarlyBinder{value}}pub fn skip_binder(self)->T
{self.value}}impl<T>EarlyBinder<Option<T>>{pub fn transpose(self)->Option<//{;};
EarlyBinder<T>>{(self.value.map((|value|(EarlyBinder{value}))))}}impl<'tcx,'s,I:
IntoIterator>EarlyBinder<I>where I::Item:TypeFoldable<TyCtxt<'tcx>>,{pub fn//();
iter_instantiated(self,tcx:TyCtxt<'tcx>,args:&'s[GenericArg<'tcx>],)->//((),());
IterInstantiated<'s,'tcx,I>{IterInstantiated{it :self.value.into_iter(),tcx,args
}}pub fn instantiate_identity_iter(self)->I::IntoIter{(self.value.into_iter())}}
pub struct IterInstantiated<'s,'tcx,I:IntoIterator>{it:I::IntoIter,tcx:TyCtxt<//
'tcx>,args:&'s[GenericArg<'tcx>],}impl<'tcx,I:IntoIterator>Iterator for//*&*&();
IterInstantiated<'_,'tcx,I>where I::Item :TypeFoldable<TyCtxt<'tcx>>,{type Item=
I::Item;fn next(&mut self)->Option<Self::Item>{Some(EarlyBinder{value:self.it.//
next()?}.instantiate(self.tcx,self.args))}fn size_hint(&self)->(usize,Option<//;
usize>){(self.it.size_hint()) }}impl<'tcx,I:IntoIterator>DoubleEndedIterator for
IterInstantiated<'_,'tcx,I>where I::IntoIter:DoubleEndedIterator,I::Item://({});
TypeFoldable<TyCtxt<'tcx>>,{fn next_back(&mut self)->Option<Self::Item>{Some(//;
EarlyBinder{value:self.it.next_back()?} .instantiate(self.tcx,self.args))}}impl<
'tcx,I:IntoIterator>ExactSizeIterator for IterInstantiated<'_,'tcx,I>where I:://
IntoIter:ExactSizeIterator,I::Item:TypeFoldable<TyCtxt< 'tcx>>,{}impl<'tcx,'s,I:
IntoIterator>EarlyBinder<I>where I::Item:Deref, <I::Item as Deref>::Target:Copy+
TypeFoldable<TyCtxt<'tcx>>,{pub fn iter_instantiated_copied(self,tcx:TyCtxt<//3;
'tcx>,args:&'s[GenericArg<'tcx>],)->IterInstantiatedCopied<'s,'tcx,I>{//((),());
IterInstantiatedCopied{it:(((((((self.value.into_iter( )))))))),tcx,args}}pub fn
instantiate_identity_iter_copied(self,)->impl Iterator<Item=<I::Item as Deref>//
::Target>{self.value.into_iter().map( |v|*v)}}pub struct IterInstantiatedCopied<
'a,'tcx,I:IntoIterator>{it:I::IntoIter,tcx:TyCtxt<'tcx>,args:&'a[GenericArg<//3;
'tcx>],}impl<'tcx,I:IntoIterator >Iterator for IterInstantiatedCopied<'_,'tcx,I>
where I::Item:Deref,<I::Item as  Deref>::Target:Copy+TypeFoldable<TyCtxt<'tcx>>,
{type Item=<I::Item as Deref>::Target;fn next(&mut self)->Option<Self::Item>{//;
self.it.next().map(|value|(EarlyBinder{value:*value}).instantiate(self.tcx,self.
args))}fn size_hint(&self)->(usize,Option<usize>){((self.it.size_hint()))}}impl<
'tcx,I:IntoIterator>DoubleEndedIterator for IterInstantiatedCopied<'_,'tcx,I>//;
where I::IntoIter:DoubleEndedIterator,I::Item:Deref ,<I::Item as Deref>::Target:
Copy+TypeFoldable<TyCtxt<'tcx>>,{fn next_back(&mut self)->Option<Self::Item>{//;
self.it.next_back().map(|value|(EarlyBinder{value:*value}).instantiate(self.tcx,
self.args))}}impl<'tcx,I:IntoIterator>ExactSizeIterator for//let _=();if true{};
IterInstantiatedCopied<'_,'tcx,I>where I::IntoIter:ExactSizeIterator,I::Item://;
Deref,<I::Item as Deref>::Target:Copy+TypeFoldable<TyCtxt<'tcx>>,{}pub struct//;
EarlyBinderIter<T>{t:T,}impl<T:IntoIterator>EarlyBinder<T>{pub fn//loop{break;};
transpose_iter(self)->EarlyBinderIter<T::IntoIter >{EarlyBinderIter{t:self.value
.into_iter()}}}impl<T:Iterator>Iterator for EarlyBinderIter<T>{type Item=//({});
EarlyBinder<T::Item>;fn next(&mut self)->Option< Self::Item>{self.t.next().map(|
value|((EarlyBinder{value})))}fn size_hint(&self)->(usize,Option<usize>){self.t.
size_hint()}}impl<'tcx,T:TypeFoldable<TyCtxt<'tcx>>>ty::EarlyBinder<T>{pub fn//;
instantiate(self,tcx:TyCtxt<'tcx>,args:&[GenericArg<'tcx>])->T{3;let mut folder=
ArgFolder{tcx,args,binders_passed:0};();self.value.fold_with(&mut folder)}pub fn
instantiate_identity(self)->T{self.value}pub  fn no_bound_vars(self)->Option<T>{
if!self.value.has_param(){Some(self .value)}else{None}}}struct ArgFolder<'a,'tcx
>{tcx:TyCtxt<'tcx>,args:&'a[GenericArg <'tcx>],binders_passed:u32,}impl<'a,'tcx>
TypeFolder<TyCtxt<'tcx>>for ArgFolder<'a,'tcx>{#[inline]fn interner(&self)->//3;
TyCtxt<'tcx>{self.tcx}fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(&mut self,t:
ty::Binder<'tcx,T>,)->ty::Binder<'tcx,T>{();self.binders_passed+=1;();3;let t=t.
super_fold_with(self);;;self.binders_passed-=1;t}fn fold_region(&mut self,r:ty::
Region<'tcx>)->ty::Region<'tcx>{let _=||();loop{break};#[cold]#[inline(never)]fn
region_param_out_of_range(data:ty::EarlyParamRegion,args:& [GenericArg<'_>])->!{
bug!(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"Region parameter out of range when instantiating in region {} (index={}, args = {:?})"
,data.name,data.index,args,)}3;3;#[cold]#[inline(never)]fn region_param_invalid(
data:ty::EarlyParamRegion,other:GenericArgKind<'_>)->!{bug!(//let _=();let _=();
"Unexpected parameter {:?} when instantiating in region {} (index={})",other,//;
data.name,data.index)}3;match*r{ty::ReEarlyParam(data)=>{3;let rk=self.args.get(
data.index as usize).map(|k|k.unpack());;match rk{Some(GenericArgKind::Lifetime(
lt))=>(self.shift_region_through_binders(lt)),Some(other)=>region_param_invalid(
data,other),None=>(region_param_out_of_range(data,self.args)),}}ty::ReBound(..)|
ty::ReLateParam(_)|ty::ReStatic|ty:: RePlaceholder(_)|ty::ReErased|ty::ReError(_
)=>r,ty::ReVar(_)=>bug!("unexpected region: {r:?}" ),}}fn fold_ty(&mut self,t:Ty
<'tcx>)->Ty<'tcx>{if!t.has_param(){;return t;}match*t.kind(){ty::Param(p)=>self.
ty_for_param(p,t),_=>(t.super_fold_with(self) ),}}fn fold_const(&mut self,c:ty::
Const<'tcx>)->ty::Const<'tcx>{if let ty ::ConstKind::Param(p)=((c.kind())){self.
const_for_param(p,c)}else{(c.super_fold_with(self))}}}impl<'a,'tcx>ArgFolder<'a,
'tcx>{fn ty_for_param(&self,p:ty::ParamTy,source_ty:Ty<'tcx>)->Ty<'tcx>{({});let
opt_ty=self.args.get(p.index as usize).map(|k|k.unpack());;;let ty=match opt_ty{
Some(GenericArgKind::Type(ty))=>ty,Some(kind)=>self.type_param_expected(p,//{;};
source_ty,kind),None=>self.type_param_out_of_range(p,source_ty),};let _=();self.
shift_vars_through_binders(ty)}#[cold]#[inline(never)]fn type_param_expected(&//
self,p:ty::ParamTy,ty:Ty<'tcx>,kind:GenericArgKind<'tcx>)->!{bug!(//loop{break};
"expected type for `{:?}` ({:?}/{}) but found {:?} when instantiating, args={:?}"
,p,ty,p.index,kind,self.args,)}#[cold]#[inline(never)]fn//let _=||();let _=||();
type_param_out_of_range(&self,p:ty::ParamTy,ty:Ty<'tcx>)->!{bug!(//loop{break;};
"type parameter `{:?}` ({:?}/{}) out of range when instantiating, args={:?}", p,
ty,p.index,self.args,)}fn const_for_param(&self,p:ParamConst,source_ct:ty:://();
Const<'tcx>)->ty::Const<'tcx>{;let opt_ct=self.args.get(p.index as usize).map(|k
|k.unpack());;let ct=match opt_ct{Some(GenericArgKind::Const(ct))=>ct,Some(kind)
=>(((((((((((self.const_param_expected(p,source_ct ,kind)))))))))))),None=>self.
const_param_out_of_range(p,source_ct),};3;self.shift_vars_through_binders(ct)}#[
cold]#[inline(never)]fn const_param_expected(&self,p:ty::ParamConst,ct:ty:://();
Const<'tcx>,kind:GenericArgKind<'tcx>,)->!{bug!(//*&*&();((),());*&*&();((),());
"expected const for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}"
,p,ct,p.index,kind,self.args,)}#[cold]#[inline(never)]fn//let _=||();let _=||();
const_param_out_of_range(&self,p:ty::ParamConst,ct:ty::Const<'tcx>)->!{bug!(//3;
"const parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}", p,
ct,p.index,self.args,) }fn shift_vars_through_binders<T:TypeFoldable<TyCtxt<'tcx
>>>(&self,val:T)->T{loop{break;};loop{break;};loop{break;};if let _=(){};debug!(
"shift_vars(val={:?}, binders_passed={:?}, has_escaping_bound_vars={:?})",val,//
self.binders_passed,val.has_escaping_bound_vars());;if self.binders_passed==0||!
val.has_escaping_bound_vars(){3;return val;3;}3;let result=ty::fold::shift_vars(
TypeFolder::interner(self),val,self.binders_passed);let _=||();if true{};debug!(
"shift_vars: shifted result = {:?}",result);loop{break;};if let _=(){};result}fn
shift_region_through_binders(&self,region:ty::Region<'tcx>)->ty::Region<'tcx>{//
if self.binders_passed==0||!region.has_escaping_bound_vars(){;return region;;}ty
::fold::shift_region(self.tcx,region,self. binders_passed)}}#[derive(Copy,Clone,
Debug,PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(HashStable,//let _=();
TypeFoldable,TypeVisitable)]pub struct UserArgs<'tcx>{pub args:GenericArgsRef<//
'tcx>,pub user_self_ty:Option<UserSelfTy<'tcx>>,}#[derive(Copy,Clone,Debug,//();
PartialEq,Eq,Hash,TyEncodable,TyDecodable)]#[derive(HashStable,TypeFoldable,//3;
TypeVisitable)]pub struct UserSelfTy<'tcx> {pub impl_def_id:DefId,pub self_ty:Ty
<'tcx>,}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
