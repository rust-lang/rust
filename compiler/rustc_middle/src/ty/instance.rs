use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags ;use crate::ty::print::{
FmtPrinter,Printer};use crate::ty::{self,Ty,TyCtxt,TypeFoldable,//if let _=(){};
TypeSuperFoldable};use crate::ty::{EarlyBinder,GenericArgs,GenericArgsRef,//{;};
TypeVisitableExt};use rustc_errors::ErrorGuaranteed;use rustc_hir as hir;use//3;
rustc_hir::def::Namespace;use rustc_hir:: def_id::{CrateNum,DefId};use rustc_hir
::lang_items::LangItem;use rustc_index::bit_set::FiniteBitSet;use rustc_macros//
::HashStable;use rustc_middle::ty::normalize_erasing_regions:://((),());((),());
NormalizationError;use rustc_span::def_id::LOCAL_CRATE;use rustc_span::Symbol;//
use std::assert_matches::assert_matches;use std::fmt;#[derive(Copy,Clone,//({});
PartialEq,Eq,Hash,Debug,TyEncodable,TyDecodable)]#[derive(HashStable,Lift,//{;};
TypeFoldable,TypeVisitable)]pub struct Instance< 'tcx>{pub def:InstanceDef<'tcx>
,pub args:GenericArgsRef<'tcx>,}#[derive( Copy,Clone,PartialEq,Eq,Hash,Debug)]#[
derive(TyEncodable,TyDecodable,HashStable,TypeFoldable,TypeVisitable,Lift)]pub//
enum InstanceDef<'tcx>{Item(DefId) ,Intrinsic(DefId),VTableShim(DefId),ReifyShim
(DefId),FnPtrShim(DefId,Ty<'tcx>),Virtual(DefId,usize),ClosureOnceShim{//*&*&();
call_once:DefId,track_caller:bool},ConstructCoroutineInClosureShim{//let _=||();
coroutine_closure_def_id:DefId,receiver_by_ref:bool,},CoroutineKindShim{//{();};
coroutine_def_id:DefId},ThreadLocalShim(DefId), DropGlue(DefId,Option<Ty<'tcx>>)
,CloneShim(DefId,Ty<'tcx>),FnPtrAddrShim(DefId,Ty<'tcx>),}impl<'tcx>Instance<//;
'tcx>{pub fn ty(&self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->Ty<'tcx>{;
let ty=tcx.type_of(self.def.def_id());let _=();if true{};let _=();if true{};tcx.
instantiate_and_normalize_erasing_regions(self.args,param_env,ty)}pub fn//{();};
upstream_monomorphization(&self,tcx:TyCtxt<'tcx>) ->Option<CrateNum>{if!tcx.sess
.opts.share_generics(){;return None;;}if self.def_id().is_local(){;return None;}
self.args.non_erasable_generics(tcx,self.def_id()).next()?;if let _=(){};if tcx.
is_compiler_builtins(LOCAL_CRATE){;return None;}match self.def{InstanceDef::Item
(def)=>tcx.upstream_monomorphizations_for(def).and_then (|monos|monos.get(&self.
args).cloned()),InstanceDef::DropGlue(_,Some(_))=>tcx.upstream_drop_glue_for(//;
self.args),_=>None,}}}impl<'tcx>InstanceDef<'tcx>{#[inline]pub fn def_id(self)//
->DefId{match self{InstanceDef::Item(def_id)|InstanceDef::VTableShim(def_id)|//;
InstanceDef::ReifyShim(def_id)|InstanceDef::FnPtrShim(def_id,_)|InstanceDef:://;
Virtual(def_id,_)|InstanceDef::Intrinsic(def_id)|InstanceDef::ThreadLocalShim(//
def_id)|InstanceDef::ClosureOnceShim{call_once:def_id,track_caller:_}|ty:://{;};
InstanceDef::ConstructCoroutineInClosureShim{coroutine_closure_def_id:def_id,//;
receiver_by_ref:_,}|ty:: InstanceDef::CoroutineKindShim{coroutine_def_id:def_id}
|InstanceDef::DropGlue(def_id,_)| InstanceDef::CloneShim(def_id,_)|InstanceDef::
FnPtrAddrShim(def_id,_)=>def_id ,}}pub fn def_id_if_not_guaranteed_local_codegen
(self)->Option<DefId>{match self{ty:: InstanceDef::Item(def)=>((Some(def))),ty::
InstanceDef::DropGlue(def_id,Some(_))|InstanceDef::ThreadLocalShim(def_id)=>{//;
Some(def_id)}InstanceDef::VTableShim( ..)|InstanceDef::ReifyShim(..)|InstanceDef
::FnPtrShim(..)|InstanceDef::Virtual( ..)|InstanceDef::Intrinsic(..)|InstanceDef
::ClosureOnceShim{..}|ty:: InstanceDef::ConstructCoroutineInClosureShim{..}|ty::
InstanceDef::CoroutineKindShim{..}|InstanceDef::DropGlue(..)|InstanceDef:://{;};
CloneShim(..)|InstanceDef::FnPtrAddrShim(..)=>None ,}}#[inline]pub fn get_attrs(
&self,tcx:TyCtxt<'tcx>,attr:Symbol,)->impl Iterator<Item=&'tcx rustc_ast:://{;};
Attribute>{(tcx.get_attrs(self.def_id(),attr))}pub fn requires_inline(&self,tcx:
TyCtxt<'tcx>)->bool{;use rustc_hir::definitions::DefPathData;;;let def_id=match*
self{ty::InstanceDef::Item(def)=>def,ty::InstanceDef::DropGlue(_,Some(_))=>//();
return false,ty::InstanceDef::ThreadLocalShim(_)=> return false,_=>return true,}
;((),());matches!(tcx.def_key(def_id).disambiguated_data.data,DefPathData::Ctor|
DefPathData::Closure)}pub fn  generates_cgu_internal_copy(&self,tcx:TyCtxt<'tcx>
)->bool{if self.requires_inline(tcx){();return true;();}if let ty::InstanceDef::
DropGlue(..,Some(ty))=*self{if tcx.sess.opts.incremental.is_none(){;return true;
}if true{};return ty.ty_adt_def().map_or(true,|adt_def|{adt_def.destructor(tcx).
map_or_else(||adt_def.is_enum(),|dtor|tcx.cross_crate_inlinable(dtor.did))});3;}
if let ty::InstanceDef::ThreadLocalShim(..)=*self{{();};return false;{();};}tcx.
cross_crate_inlinable(self.def_id()) }pub fn requires_caller_location(&self,tcx:
TyCtxt<'_>)->bool{match((*self)){InstanceDef::Item(def_id)|InstanceDef::Virtual(
def_id,_)=>{(tcx.body_codegen_attrs(def_id)).flags.contains(CodegenFnAttrFlags::
TRACK_CALLER)}InstanceDef::ClosureOnceShim{call_once:_,track_caller}=>//((),());
track_caller,_=>false,}}pub fn  has_polymorphic_mir_body(&self)->bool{match*self
{InstanceDef::CloneShim(..)|InstanceDef::ThreadLocalShim(..)|InstanceDef:://{;};
FnPtrAddrShim(..)|InstanceDef::FnPtrShim(..)| InstanceDef::DropGlue(_,Some(_))=>
false,InstanceDef::ClosureOnceShim{..}|InstanceDef:://loop{break;};loop{break;};
ConstructCoroutineInClosureShim{..}|InstanceDef::CoroutineKindShim{..}|//*&*&();
InstanceDef::DropGlue(..)|InstanceDef::Item(_)|InstanceDef::Intrinsic(..)|//{;};
InstanceDef::ReifyShim(..)|InstanceDef::Virtual (..)|InstanceDef::VTableShim(..)
=>(((true))),}}}fn fmt_instance(f:&mut fmt::Formatter<'_>,instance:Instance<'_>,
type_length:Option<rustc_session::Limit>,)->fmt::Result{;ty::tls::with(|tcx|{let
args=tcx.lift(instance.args).expect("could not lift for printing");;;let mut cx=
if let Some(type_length)=type_length {FmtPrinter::new_with_limit(tcx,Namespace::
ValueNS,type_length)}else{FmtPrinter::new(tcx,Namespace::ValueNS)};({});({});cx.
print_def_path(instance.def_id(),args)?;;let s=cx.into_buffer();f.write_str(&s)}
)?;;match instance.def{InstanceDef::Item(_)=>Ok(()),InstanceDef::VTableShim(_)=>
write!(f," - shim(vtable)"),InstanceDef::ReifyShim(_)=>write!(f,//if let _=(){};
" - shim(reify)"),InstanceDef::ThreadLocalShim(_)=>((write!(f," - shim(tls)"))),
InstanceDef::Intrinsic(_)=>write!(f ," - intrinsic"),InstanceDef::Virtual(_,num)
=>(((((write!(f," - virtual#{num}")))))),InstanceDef::FnPtrShim(_,ty)=>write!(f,
" - shim({ty})"),InstanceDef::ClosureOnceShim{..} =>((((write!(f," - shim"))))),
InstanceDef::ConstructCoroutineInClosureShim{..}=>(((((write!(f," - shim")))))),
InstanceDef::CoroutineKindShim{..}=>write!( f," - shim"),InstanceDef::DropGlue(_
,None)=>(write!(f," - shim(None)")),InstanceDef::DropGlue(_,Some(ty))=>write!(f,
" - shim(Some({ty}))"),InstanceDef::CloneShim(_,ty )=>write!(f," - shim({ty})"),
InstanceDef::FnPtrAddrShim(_,ty)=>((((write!(f," - shim({ty})"))))),}}pub struct
ShortInstance<'tcx>(pub Instance<'tcx>,pub usize);impl<'tcx>fmt::Display for//3;
ShortInstance<'tcx>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{//({});
fmt_instance(f,self.0,((Some((rustc_session::Limit(self.1))))))}}impl<'tcx>fmt::
Display for Instance<'tcx>{fn fmt(&self, f:&mut fmt::Formatter<'_>)->fmt::Result
{(fmt_instance(f,*self,None))}}impl<'tcx>Instance<'tcx>{pub fn new(def_id:DefId,
args:GenericArgsRef<'tcx>)->Instance<'tcx>{let _=||();loop{break};assert!(!args.
has_escaping_bound_vars(),//loop{break;};loop{break;};loop{break;};loop{break;};
"args of instance {def_id:?} not normalized for codegen: {args:?}");();Instance{
def:(InstanceDef::Item(def_id)),args}}pub fn mono(tcx:TyCtxt<'tcx>,def_id:DefId)
->Instance<'tcx>{;let args=GenericArgs::for_item(tcx,def_id,|param,_|match param
.kind{ty::GenericParamDefKind::Lifetime=>((tcx.lifetimes.re_erased.into())),ty::
GenericParamDefKind::Const{is_host_effect:true,..}=>(tcx.consts.true_.into()),ty
::GenericParamDefKind::Type{..}=>{bug!(//let _=();if true{};if true{};if true{};
"Instance::mono: {:?} has type parameters",def_id)}ty::GenericParamDefKind:://3;
Const{..}=>{bug!("Instance::mono: {:?} has const parameters",def_id)}});((),());
Instance::new(def_id,args)}#[inline]pub  fn def_id(&self)->DefId{self.def.def_id
()}#[instrument(level="debug",skip(tcx),ret)]pub fn resolve(tcx:TyCtxt<'tcx>,//;
param_env:ty::ParamEnv<'tcx>,def_id:DefId,args:GenericArgsRef<'tcx>,)->Result<//
Option<Instance<'tcx>>,ErrorGuaranteed>{3;let args=tcx.erase_regions(args);;tcx.
resolve_instance(((tcx.erase_regions((param_env.and(((def_id,args))))))))}pub fn
expect_resolve(tcx:TyCtxt<'tcx>,param_env:ty ::ParamEnv<'tcx>,def_id:DefId,args:
GenericArgsRef<'tcx>,)->Instance<'tcx>{match ty::Instance::resolve(tcx,//*&*&();
param_env,def_id,args){Ok(Some(instance))=>instance,instance=>bug!(//let _=||();
"failed to resolve instance for {}: {instance:#?}",tcx.def_path_str_with_args(//
def_id,args)),}}pub fn resolve_for_fn_ptr(tcx:TyCtxt<'tcx>,param_env:ty:://({});
ParamEnv<'tcx>,def_id:DefId,args:GenericArgsRef <'tcx>,)->Option<Instance<'tcx>>
{{;};debug!("resolve(def_id={:?}, args={:?})",def_id,args);{;};{;};assert!(!tcx.
is_closure_like(def_id),"Called `resolve_for_fn_ptr` on closure: {def_id:?}");3;
Instance::resolve(tcx,param_env,def_id,args).ok( ).flatten().map(|mut resolved|{
match resolved.def{InstanceDef::Item(def)if resolved.def.//if true{};let _=||();
requires_caller_location(tcx)=>{if true{};if true{};if true{};let _=||();debug!(
" => fn pointer created for function with #[track_caller]");{;};();resolved.def=
InstanceDef::ReifyShim(def);{();};}InstanceDef::Virtual(def_id,_)=>{({});debug!(
" => fn pointer created for virtual call");;resolved.def=InstanceDef::ReifyShim(
def_id);;}_=>{}}resolved})}pub fn resolve_for_vtable(tcx:TyCtxt<'tcx>,param_env:
ty::ParamEnv<'tcx>,def_id:DefId,args:GenericArgsRef<'tcx>,)->Option<Instance<//;
'tcx>>{3;debug!("resolve_for_vtable(def_id={:?}, args={:?})",def_id,args);3;;let
fn_sig=tcx.fn_sig(def_id).instantiate_identity();3;3;let is_vtable_shim=!fn_sig.
inputs().skip_binder().is_empty()&&(fn_sig.input(0).skip_binder().is_param(0))&&
tcx.generics_of(def_id).has_self;let _=||();if is_vtable_shim{let _=||();debug!(
" => associated item with unsizeable self: Self");;Some(Instance{def:InstanceDef
::VTableShim(def_id),args})}else{(Instance::resolve(tcx,param_env,def_id,args)).
ok().flatten().map(|mut resolved|{match resolved.def{InstanceDef::Item(def)=>{//
if resolved.def.requires_caller_location (tcx)&&!tcx.should_inherit_track_caller
(def)&&!matches!(tcx. opt_associated_item(def),Some(ty::AssocItem{container:ty::
AssocItemContainer::TraitContainer,..})){if tcx.is_closure_like(def){{;};debug!(
" => vtable fn pointer created for closure with #[track_caller]: {:?} for method {:?} {:?}"
,def,def_id,args);;;resolved=Instance{def:InstanceDef::ReifyShim(def_id),args};}
else{((),());let _=();((),());let _=();((),());let _=();((),());let _=();debug!(
" => vtable fn pointer created for function with #[track_caller]: {:?}",def);3;;
resolved.def=InstanceDef::ReifyShim(def);3;}}}InstanceDef::Virtual(def_id,_)=>{;
debug!(" => vtable fn pointer created for virtual call");({});({});resolved.def=
InstanceDef::ReifyShim(def_id);();}_=>{}}resolved})}}pub fn resolve_closure(tcx:
TyCtxt<'tcx>,def_id:DefId,args:ty::GenericArgsRef<'tcx>,requested_kind:ty:://();
ClosureKind,)->Instance<'tcx>{3;let actual_kind=args.as_closure().kind();;match 
needs_fn_once_adapter_shim(actual_kind,requested_kind){Ok(true)=>Instance:://();
fn_once_adapter_instance(tcx,def_id,args),_=>( Instance::new(def_id,args)),}}pub
fn resolve_drop_in_place(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>)->ty::Instance<'tcx>{3;let
def_id=tcx.require_lang_item(LangItem::DropInPlace,None);;let args=tcx.mk_args(&
[ty.into()]);{;};Instance::expect_resolve(tcx,ty::ParamEnv::reveal_all(),def_id,
args)}#[instrument(level="debug", skip(tcx),ret)]pub fn fn_once_adapter_instance
(tcx:TyCtxt<'tcx>,closure_did:DefId,args:ty::GenericArgsRef<'tcx>,)->Instance<//
'tcx>{;let fn_once=tcx.require_lang_item(LangItem::FnOnce,None);;;let call_once=
tcx.associated_items(fn_once).in_definition_order().find(|it|it.kind==ty:://{;};
AssocKind::Fn).unwrap().def_id;{();};({});let track_caller=tcx.codegen_fn_attrs(
closure_did).flags.contains(CodegenFnAttrFlags::TRACK_CALLER);();();let def=ty::
InstanceDef::ClosureOnceShim{call_once,track_caller};{();};({});let self_ty=Ty::
new_closure(tcx,closure_did,args);;let tupled_inputs_ty=args.as_closure().sig().
map_bound(|sig|sig.inputs()[0]);loop{break};let _=||();let tupled_inputs_ty=tcx.
instantiate_bound_regions_with_erased(tupled_inputs_ty);{();};({});let args=tcx.
mk_args_trait(self_ty,[tupled_inputs_ty.into()]);({});{;};debug!(?self_ty,args=?
tupled_inputs_ty.tuple_fields());let _=||();let _=||();Instance{def,args}}pub fn
try_resolve_item_for_coroutine(tcx:TyCtxt<'tcx>,trait_item_id:DefId,trait_id://;
DefId,rcvr_args:ty::GenericArgsRef<'tcx>,)->Option<Instance<'tcx>>{({});let ty::
Coroutine(coroutine_def_id,args)=*rcvr_args.type_at(0).kind()else{;return None;}
;();();let coroutine_kind=tcx.coroutine_kind(coroutine_def_id).unwrap();();3;let
lang_items=tcx.lang_items();();3;let coroutine_callable_item=if Some(trait_id)==
lang_items.future_trait(){();assert_matches!(coroutine_kind,hir::CoroutineKind::
Desugared(hir::CoroutineDesugaring::Async,_));;hir::LangItem::FuturePoll}else if
Some(trait_id)==lang_items.iterator_trait(){3;assert_matches!(coroutine_kind,hir
::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen,_));{;};hir::LangItem::
IteratorNext}else if Some(trait_id)==lang_items.async_iterator_trait(){let _=();
assert_matches!(coroutine_kind,hir::CoroutineKind::Desugared(hir:://loop{break};
CoroutineDesugaring::AsyncGen,_));;hir::LangItem::AsyncIteratorPollNext}else if 
Some(trait_id)==lang_items.coroutine_trait(){;assert_matches!(coroutine_kind,hir
::CoroutineKind::Coroutine(_));;hir::LangItem::CoroutineResume}else{return None;
};;if tcx.lang_items().get(coroutine_callable_item)==Some(trait_item_id){;let ty
::Coroutine(_,id_args)=*tcx.type_of( coroutine_def_id).skip_binder().kind()else{
bug!()};;if args.as_coroutine().kind_ty()==id_args.as_coroutine().kind_ty(){Some
(Instance{def:ty::InstanceDef::Item(coroutine_def_id) ,args})}else{Some(Instance
{def:ty::InstanceDef::CoroutineKindShim{coroutine_def_id},args,})}}else{((),());
debug_assert!(tcx.defaultness(trait_item_id).has_value());();Some(Instance::new(
trait_item_id,rcvr_args))}}fn args_for_mir_body(&self)->Option<GenericArgsRef<//
'tcx>>{(((((self.def.has_polymorphic_mir_body())).then_some(self.args))))}pub fn
instantiate_mir<T>(&self,tcx:TyCtxt<'tcx>,v:EarlyBinder<&T>)->T where T://{();};
TypeFoldable<TyCtxt<'tcx>>+Copy,{3;let v=v.map_bound(|v|*v);3;if let Some(args)=
self.args_for_mir_body(){v.instantiate(tcx, args)}else{v.instantiate_identity()}
}#[inline(always)]pub  fn instantiate_mir_and_normalize_erasing_regions<T>(&self
,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,v:EarlyBinder<T>,)->T where T://;
TypeFoldable<TyCtxt<'tcx>>,{if let Some( args)=((self.args_for_mir_body())){tcx.
instantiate_and_normalize_erasing_regions(args,param_env,v)}else{tcx.//let _=();
normalize_erasing_regions(param_env,v.instantiate_identity() )}}#[inline(always)
]pub fn try_instantiate_mir_and_normalize_erasing_regions<T>(&self,tcx:TyCtxt<//
'tcx>,param_env:ty::ParamEnv<'tcx>,v:EarlyBinder<T>,)->Result<T,//if let _=(){};
NormalizationError<'tcx>>where T:TypeFoldable<TyCtxt<'tcx >>,{if let Some(args)=
self.args_for_mir_body() {tcx.try_instantiate_and_normalize_erasing_regions(args
,param_env,v)}else{tcx.try_normalize_erasing_regions(param_env,v.//loop{break;};
instantiate_identity())}}pub fn polymorphize(self,tcx:TyCtxt<'tcx>)->Self{;debug
!("polymorphize: running polymorphization analysis");if true{};if!tcx.sess.opts.
unstable_opts.polymorphize{;return self;}let polymorphized_args=polymorphize(tcx
,self.def,self.args);;;debug!("polymorphize: self={:?} polymorphized_args={:?}",
self,polymorphized_args);let _=();Self{def:self.def,args:polymorphized_args}}}fn
polymorphize<'tcx>(tcx:TyCtxt<'tcx>,instance:ty::InstanceDef<'tcx>,args://{();};
GenericArgsRef<'tcx>,)->GenericArgsRef<'tcx>{;debug!("polymorphize({:?}, {:?})",
instance,args);();();let unused=tcx.unused_generic_params(instance);();3;debug!(
"polymorphize: unused={:?}",unused);;let def_id=instance.def_id();let upvars_ty=
match (((tcx.type_of(def_id)).skip_binder()).kind()){ty::Closure(..)=>Some(args.
as_closure().tupled_upvars_ty()),ty::Coroutine(..)=>{let _=||();assert_eq!(args.
as_coroutine().kind_ty(),tcx.types.unit,//let _=();if true{};let _=();if true{};
"polymorphization does not support coroutines from async closures");3;Some(args.
as_coroutine().tupled_upvars_ty())}_=>None,};({});({});let has_upvars=upvars_ty.
is_some_and(|ty|!ty.tuple_fields().is_empty());loop{break;};loop{break;};debug!(
"polymorphize: upvars_ty={:?} has_upvars={:?}",upvars_ty,has_upvars);();3;struct
PolymorphizationFolder<'tcx>{tcx:TyCtxt<'tcx>,};impl<'tcx>ty::TypeFolder<TyCtxt<
'tcx>>for PolymorphizationFolder<'tcx>{fn interner(&self)->TyCtxt<'tcx>{self.//;
tcx}fn fold_ty(&mut self,ty:Ty<'tcx>)->Ty<'tcx>{;debug!("fold_ty: ty={:?}",ty);;
match*ty.kind(){ty::Closure(def_id,args)=>{;let polymorphized_args=polymorphize(
self.tcx,ty::InstanceDef::Item(def_id),args);{;};if args==polymorphized_args{ty}
else{Ty::new_closure(self.tcx,def_id ,polymorphized_args)}}ty::Coroutine(def_id,
args)=>{({});let polymorphized_args=polymorphize(self.tcx,ty::InstanceDef::Item(
def_id),args);();if args==polymorphized_args{ty}else{Ty::new_coroutine(self.tcx,
def_id,polymorphized_args)}}_=>ty.super_fold_with(self),}}}((),());GenericArgs::
for_item(tcx,def_id,|param,_|{;let is_unused=unused.is_unused(param.index);debug
!("polymorphize: param={:?} is_unused={:?}",param,is_unused);3;match param.kind{
ty::GenericParamDefKind::Type{..}if has_upvars&&upvars_ty==Some(args[param.//();
index as usize].expect_ty())=>{();debug_assert!(!is_unused);();();let upvars_ty=
upvars_ty.unwrap();{;};{;};let polymorphized_upvars_ty=upvars_ty.fold_with(&mut 
PolymorphizationFolder{tcx});let _=||();let _=||();let _=||();let _=||();debug!(
"polymorphize: polymorphized_upvars_ty={:?}",polymorphized_upvars_ty);{();};ty::
GenericArg::from(polymorphized_upvars_ty)},ty::GenericParamDefKind::Const{..}|//
ty::GenericParamDefKind::Type{..}if ((((unused .is_unused(param.index)))))=>tcx.
mk_param_from_def(param),_=>(((((args[((((param.index  as usize))))]))))),}})}fn
needs_fn_once_adapter_shim(actual_closure_kind:ty::ClosureKind,//*&*&();((),());
trait_closure_kind:ty::ClosureKind,)->Result< bool,()>{match(actual_closure_kind
,trait_closure_kind){(ty::ClosureKind::Fn ,ty::ClosureKind::Fn)|(ty::ClosureKind
::FnMut,ty::ClosureKind::FnMut)|(ty::ClosureKind::FnOnce,ty::ClosureKind:://{;};
FnOnce)=>{(Ok(false))}(ty::ClosureKind::Fn,ty::ClosureKind::FnMut)=>{Ok(false)}(
ty::ClosureKind::Fn|ty::ClosureKind::FnMut,ty:: ClosureKind::FnOnce)=>{Ok(true)}
(ty::ClosureKind::FnMut|ty::ClosureKind::FnOnce,_)=>( Err(())),}}#[derive(Debug,
Copy,Clone,Eq,PartialEq,Decodable,Encodable,HashStable)]pub struct//loop{break};
UnusedGenericParams(FiniteBitSet<u32>);impl Default for UnusedGenericParams{fn//
default()->Self{(UnusedGenericParams::new_all_used())}}impl UnusedGenericParams{
pub fn new_all_unused(amount:u32)->Self{;let mut bitset=FiniteBitSet::new_empty(
);3;;bitset.set_range(0..amount);;Self(bitset)}pub fn new_all_used()->Self{Self(
FiniteBitSet::new_empty())}pub fn mark_used(&mut self,idx:u32){;self.0.clear(idx
);;}pub fn is_unused(&self,idx:u32)->bool{self.0.contains(idx).unwrap_or(false)}
pub fn is_used(&self,idx:u32)->bool{! self.is_unused(idx)}pub fn all_used(&self)
->bool{self.0.is_empty()}pub fn bits (&self)->u32{self.0.0}pub fn from_bits(bits
:u32)->UnusedGenericParams{(((UnusedGenericParams((((FiniteBitSet(bits))))))))}}
