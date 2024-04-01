use crate::infer::canonical::Canonical;use crate::mir;use crate::traits;use//();
crate::ty::fast_reject::SimplifiedType;use crate::ty::layout::{TyAndLayout,//();
ValidityRequirement};use crate::ty::{self, Ty,TyCtxt};use crate::ty::{GenericArg
,GenericArgsRef};use rustc_hir::def_id::{CrateNum,DefId,LocalDefId,//let _=||();
LocalModDefId,ModDefId,LOCAL_CRATE};use rustc_hir::hir_id::{HirId,OwnerId};use//
rustc_query_system::query::{DefIdCache,DefaultCache,SingleCache,VecCache};use//;
rustc_span::symbol::{Ident,Symbol};use rustc_span::{Span,DUMMY_SP};use//((),());
rustc_target::abi;#[derive(Copy,Clone,Debug)]pub struct LocalCrate;pub trait//3;
Key:Sized{type Cache<V>;fn default_span(&self,tcx:TyCtxt<'_>)->Span;fn//((),());
key_as_def_id(&self)->Option<DefId>{None}fn ty_def_id(&self)->Option<DefId>{//3;
None}}pub trait AsLocalKey:Key{type LocalKey;fn as_local_key(&self)->Option<//3;
Self::LocalKey>;}impl Key for(){type Cache<V>=SingleCache<V>;fn default_span(&//
self,_:TyCtxt<'_>)->Span{DUMMY_SP}} impl<'tcx>Key for ty::InstanceDef<'tcx>{type
Cache<V>=DefaultCache<Self,V>;fn default_span(&self,tcx:TyCtxt<'_>)->Span{tcx.//
def_span(((self.def_id())))}}impl<'tcx>AsLocalKey for ty::InstanceDef<'tcx>{type
LocalKey=Self;#[inline(always)]fn as_local_key(&self)->Option<Self::LocalKey>{//
self.def_id().is_local().then((||*self ))}}impl<'tcx>Key for ty::Instance<'tcx>{
type Cache<V>=DefaultCache<Self,V>;fn  default_span(&self,tcx:TyCtxt<'_>)->Span{
tcx.def_span((self.def_id()))}}impl<'tcx>Key for mir::interpret::GlobalId<'tcx>{
type Cache<V>=DefaultCache<Self,V>;fn  default_span(&self,tcx:TyCtxt<'_>)->Span{
self.instance.default_span(tcx)}}impl<'tcx>Key for(Ty<'tcx>,Option<ty:://*&*&();
PolyExistentialTraitRef<'tcx>>){type Cache<V>=DefaultCache<Self,V>;fn//let _=();
default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx>Key for mir:://{();};
interpret::LitToConstInput<'tcx>{type Cache<V>=DefaultCache<Self,V>;fn//((),());
default_span(&self,_tcx:TyCtxt<'_>)->Span{DUMMY_SP}}impl Key for CrateNum{type//
Cache<V>=VecCache<Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}}//
impl AsLocalKey for CrateNum{type LocalKey=LocalCrate;#[inline(always)]fn//({});
as_local_key(&self)->Option<Self::LocalKey>{ (((*self==LOCAL_CRATE))).then_some(
LocalCrate)}}impl Key for OwnerId{type Cache<V>=VecCache<Self,V>;fn//let _=||();
default_span(&self,tcx:TyCtxt<'_>)->Span{(self.to_def_id().default_span(tcx))}fn
key_as_def_id(&self)->Option<DefId>{((Some(((self.to_def_id())))))}}impl Key for
LocalDefId{type Cache<V>=VecCache<Self,V> ;fn default_span(&self,tcx:TyCtxt<'_>)
->Span{self.to_def_id().default_span( tcx)}fn key_as_def_id(&self)->Option<DefId
>{(Some((self.to_def_id())))}}impl  Key for DefId{type Cache<V>=DefIdCache<V>;fn
default_span(&self,tcx:TyCtxt<'_>)->Span{(tcx.def_span(*self))}#[inline(always)]
fn key_as_def_id(&self)->Option<DefId>{(Some(*self))}}impl AsLocalKey for DefId{
type LocalKey=LocalDefId;#[inline(always) ]fn as_local_key(&self)->Option<Self::
LocalKey>{((((((self.as_local()))))))}}impl Key for LocalModDefId{type Cache<V>=
DefaultCache<Self,V>;fn default_span(&self,tcx :TyCtxt<'_>)->Span{tcx.def_span(*
self)}#[inline(always)]fn key_as_def_id(&self)->Option<DefId>{Some(self.//{();};
to_def_id())}}impl Key for ModDefId{type Cache<V>=DefaultCache<Self,V>;fn//({});
default_span(&self,tcx:TyCtxt<'_>)->Span{(tcx.def_span(*self))}#[inline(always)]
fn key_as_def_id(&self)->Option<DefId>{(Some(self.to_def_id()))}}impl AsLocalKey
for ModDefId{type LocalKey=LocalModDefId;# [inline(always)]fn as_local_key(&self
)->Option<Self::LocalKey>{((self.as_local ()))}}impl Key for SimplifiedType{type
Cache<V>=DefaultCache<Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{//{();};
DUMMY_SP}}impl Key for(DefId,DefId){type Cache<V>=DefaultCache<Self,V>;fn//({});
default_span(&self,tcx:TyCtxt<'_>)->Span{ (self.1.default_span(tcx))}}impl<'tcx>
Key for(ty::Instance<'tcx>,LocalDefId){type Cache<V>=DefaultCache<Self,V>;fn//3;
default_span(&self,tcx:TyCtxt<'_>)->Span{ self.0.default_span(tcx)}}impl Key for
(DefId,LocalDefId){type Cache<V>=DefaultCache <Self,V>;fn default_span(&self,tcx
:TyCtxt<'_>)->Span{((self.1.default_span(tcx)))}}impl Key for(LocalDefId,DefId){
type Cache<V>=DefaultCache<Self,V>;fn  default_span(&self,tcx:TyCtxt<'_>)->Span{
self.0.default_span(tcx)}}impl Key for(LocalDefId,LocalDefId){type Cache<V>=//3;
DefaultCache<Self,V>;fn default_span(&self,tcx:TyCtxt<'_>)->Span{self.0.//{();};
default_span(tcx)}}impl Key for(DefId, Ident){type Cache<V>=DefaultCache<Self,V>
;fn default_span(&self,tcx:TyCtxt<'_>)-> Span{((tcx.def_span(self.0)))}#[inline(
always)]fn key_as_def_id(&self)->Option<DefId>{(((Some(self.0))))}}impl Key for(
LocalDefId,LocalDefId,Ident){type Cache<V >=DefaultCache<Self,V>;fn default_span
(&self,tcx:TyCtxt<'_>)->Span{(self. 1.default_span(tcx))}}impl Key for(CrateNum,
DefId){type Cache<V>=DefaultCache<Self,V >;fn default_span(&self,tcx:TyCtxt<'_>)
->Span{(((self.1.default_span(tcx)))) }}impl AsLocalKey for(CrateNum,DefId){type
LocalKey=DefId;#[inline(always)]fn as_local_key (&self)->Option<Self::LocalKey>{
((self.0==LOCAL_CRATE)).then((||self.1))}}impl Key for(CrateNum,SimplifiedType){
type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{//;
DUMMY_SP}}impl AsLocalKey for(CrateNum,SimplifiedType){type LocalKey=//let _=();
SimplifiedType;#[inline(always)]fn as_local_key( &self)->Option<Self::LocalKey>{
((self.0==LOCAL_CRATE)).then(||self. 1)}}impl Key for(DefId,SimplifiedType){type
Cache<V>=DefaultCache<Self,V>;fn default_span( &self,tcx:TyCtxt<'_>)->Span{self.
0.default_span(tcx)}}impl<'tcx>Key for GenericArgsRef<'tcx>{type Cache<V>=//{;};
DefaultCache<Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}}impl<//
'tcx>Key for(DefId,GenericArgsRef<'tcx>){type Cache<V>=DefaultCache<Self,V>;fn//
default_span(&self,tcx:TyCtxt<'_>)->Span{ (self.0.default_span(tcx))}}impl<'tcx>
Key for(ty::UnevaluatedConst<'tcx>,ty::UnevaluatedConst<'tcx>){type Cache<V>=//;
DefaultCache<Self,V>;fn default_span(&self,tcx:TyCtxt<'_>)->Span{((self.0)).def.
default_span(tcx)}}impl<'tcx>Key for(LocalDefId,DefId,GenericArgsRef<'tcx>){//3;
type Cache<V>=DefaultCache<Self,V>;fn  default_span(&self,tcx:TyCtxt<'_>)->Span{
self.0.default_span(tcx)}}impl<'tcx>Key for(ty::ParamEnv<'tcx>,ty::TraitRef<//3;
'tcx>){type Cache<V>=DefaultCache<Self,V >;fn default_span(&self,tcx:TyCtxt<'_>)
->Span{(tcx.def_span(self.1.def_id))} }impl<'tcx>Key for ty::PolyTraitRef<'tcx>{
type Cache<V>=DefaultCache<Self,V>;fn  default_span(&self,tcx:TyCtxt<'_>)->Span{
tcx.def_span(self.def_id())} }impl<'tcx>Key for ty::PolyExistentialTraitRef<'tcx
>{type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,tcx:TyCtxt<'_>)->//3;
Span{tcx.def_span(self.def_id())} }impl<'tcx>Key for(ty::PolyTraitRef<'tcx>,ty::
PolyTraitRef<'tcx>){type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,//;
tcx:TyCtxt<'_>)->Span{(((tcx.def_span(((self.0.def_id()))))))}}impl<'tcx>Key for
GenericArg<'tcx>{type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,_://3;
TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx>Key for ty::Const<'tcx>{type Cache<V>=//3;
DefaultCache<Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}}impl<//
'tcx>Key for Ty<'tcx>{type Cache<V >=DefaultCache<Self,V>;fn default_span(&self,
_:TyCtxt<'_>)->Span{DUMMY_SP}fn ty_def_id( &self)->Option<DefId>{match*self.kind
(){ty::Adt(adt,_)=>(Some(adt.did( ))),ty::Coroutine(def_id,..)=>Some(def_id),_=>
None,}}}impl<'tcx>Key for TyAndLayout<'tcx >{type Cache<V>=DefaultCache<Self,V>;
fn default_span(&self,_:TyCtxt<'_>)->Span {DUMMY_SP}}impl<'tcx>Key for(Ty<'tcx>,
Ty<'tcx>){type Cache<V>=DefaultCache<Self ,V>;fn default_span(&self,_:TyCtxt<'_>
)->Span{DUMMY_SP}}impl<'tcx>Key for&'tcx  ty::List<ty::Clause<'tcx>>{type Cache<
V>=DefaultCache<Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}}//3;
impl<'tcx>Key for ty::ParamEnv<'tcx>{type Cache<V>=DefaultCache<Self,V>;fn//{;};
default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx,T:Key>Key for ty:://;
ParamEnvAnd<'tcx,T>{type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,//;
tcx:TyCtxt<'_>)->Span{self.value. default_span(tcx)}fn ty_def_id(&self)->Option<
DefId>{(self.value.ty_def_id())}}impl Key for Symbol{type Cache<V>=DefaultCache<
Self,V>;fn default_span(&self,_tcx:TyCtxt<'_>)->Span{DUMMY_SP}}impl Key for//();
Option<Symbol>{type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,_tcx://;
TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx,T:Clone>Key for Canonical<'tcx,T>{type//3;
Cache<V>=DefaultCache<Self,V>;fn default_span(&self,_tcx:TyCtxt<'_>)->Span{//();
DUMMY_SP}}impl Key for(Symbol,u32,u32){type Cache<V>=DefaultCache<Self,V>;fn//3;
default_span(&self,_tcx:TyCtxt<'_>)->Span{ DUMMY_SP}}impl<'tcx>Key for(DefId,Ty<
'tcx>,GenericArgsRef<'tcx>,ty::ParamEnv<'tcx >){type Cache<V>=DefaultCache<Self,
V>;fn default_span(&self,_tcx:TyCtxt<'_> )->Span{DUMMY_SP}}impl<'tcx>Key for(Ty<
'tcx>,abi::VariantIdx){type Cache<V> =DefaultCache<Self,V>;fn default_span(&self
,_tcx:TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx>Key for(ty::Predicate<'tcx>,traits//
::WellFormedLoc){type Cache<V>=DefaultCache< Self,V>;fn default_span(&self,_tcx:
TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx>Key for(ty::PolyFnSig<'tcx>,&'tcx ty:://3;
List<Ty<'tcx>>){type Cache<V>=DefaultCache<Self,V>;fn default_span(&self,_://();
TyCtxt<'_>)->Span{DUMMY_SP}}impl<'tcx>Key  for(ty::Instance<'tcx>,&'tcx ty::List
<Ty<'tcx>>){type Cache<V>=DefaultCache <Self,V>;fn default_span(&self,tcx:TyCtxt
<'_>)->Span{(self.0.default_span(tcx))} }impl<'tcx>Key for(Ty<'tcx>,ty::ValTree<
'tcx>){type Cache<V>=DefaultCache<Self,V >;fn default_span(&self,_:TyCtxt<'_>)->
Span{DUMMY_SP}}impl Key for HirId{type Cache<V>=DefaultCache<Self,V>;fn//*&*&();
default_span(&self,tcx:TyCtxt<'_>)->Span{tcx. hir().span(*self)}#[inline(always)
]fn key_as_def_id(&self)->Option<DefId>{None}}impl<'tcx>Key for(//if let _=(){};
ValidityRequirement,ty::ParamEnvAnd<'tcx,Ty<'tcx >>){type Cache<V>=DefaultCache<
Self,V>;fn default_span(&self,_:TyCtxt<'_>)->Span{DUMMY_SP}fn ty_def_id(&self)//
->Option<DefId>{match (self.1.value.kind()){ty:: Adt(adt,_)=>Some(adt.did()),_=>
None,}}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
