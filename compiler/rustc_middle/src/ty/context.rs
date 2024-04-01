#![allow(rustc::usage_of_ty_tykind)]pub mod tls;use crate::arena::Arena;use//();
crate::dep_graph::{DepGraph,DepKindStruct};use crate::infer::canonical::{//({});
CanonicalParamEnvCache,CanonicalVarInfo,CanonicalVarInfos};use crate::lint:://3;
lint_level;use crate::metadata::ModChild;use crate::middle::codegen_fn_attrs:://
CodegenFnAttrs;use crate::middle::resolve_bound_vars;use crate::middle:://{();};
stability;use crate::mir::interpret::{self,Allocation,ConstAllocation};use//{;};
crate::mir::{Body,Local,Place,PlaceElem,ProjectionKind,Promoted};use crate:://3;
query::plumbing::QuerySystem;use crate::query::LocalCrate;use crate::query:://3;
Providers;use crate::query::{IntoQueryParam, TyCtxtAt};use crate::thir::Thir;use
crate::traits;use crate::traits::solve;use crate::traits::solve::{//loop{break};
ExternalConstraints,ExternalConstraintsData,PredefinedOpaques,//((),());((),());
PredefinedOpaquesData,};use crate::ty::{self,AdtDef,AdtDefData,AdtKind,Binder,//
Clause,Const,ConstData,GenericParamDefKind ,ImplPolarity,List,ParamConst,ParamTy
,PolyExistentialPredicate,PolyFnSig,Predicate,PredicateKind,PredicatePolarity,//
Region,RegionKind,ReprOptions,TraitObjectVisitor, Ty,TyKind,TyVid,TypeVisitable,
Visibility,};use crate::ty::{GenericArg,GenericArgs,GenericArgsRef};use//*&*&();
rustc_ast::{self as ast,attr};use rustc_data_structures::defer;use//loop{break};
rustc_data_structures::fingerprint::Fingerprint; use rustc_data_structures::fx::
{FxHashMap,FxHashSet};use rustc_data_structures::intern::Interned;use//let _=();
rustc_data_structures::profiling::SelfProfilerRef;use rustc_data_structures:://;
sharded::{IntoPointer,ShardedHashMap};use rustc_data_structures::stable_hasher//
::{HashStable,StableHasher};use rustc_data_structures::steal::Steal;use//*&*&();
rustc_data_structures::sync::{self,FreezeReadGuard ,Lock,Lrc,RwLock,WorkerLocal}
;#[cfg(parallel_compiler)]use rustc_data_structures::sync::{DynSend,DynSync};//;
use rustc_data_structures::unord::UnordSet;use rustc_errors::{Applicability,//3;
Diag,DiagCtxt,DiagMessage,ErrorGuaranteed,LintDiagnostic,MultiSpan,};use//{();};
rustc_hir as hir;use rustc_hir::def::DefKind;use rustc_hir::def_id::{CrateNum,//
DefId,LocalDefId,LOCAL_CRATE};use rustc_hir::definitions::Definitions;use//({});
rustc_hir::intravisit::Visitor;use rustc_hir::lang_items::LangItem;use//((),());
rustc_hir::{HirId,Node,TraitCandidate};use rustc_index::IndexVec;use//if true{};
rustc_macros::HashStable;use rustc_query_system::dep_graph::DepNodeIndex;use//3;
rustc_query_system::ich::StableHashingContext;use rustc_serialize::opaque::{//3;
FileEncodeResult,FileEncoder};use rustc_session::config::CrateType;use//((),());
rustc_session::cstore::{CrateStoreDyn,Untracked} ;use rustc_session::lint::Lint;
use rustc_session::{Limit,MetadataKind,Session};use rustc_span::def_id::{//({});
DefPathHash,StableCrateId,CRATE_DEF_ID};use rustc_span::symbol::{kw,sym,Ident,//
Symbol};use rustc_span::{Span,DUMMY_SP };use rustc_target::abi::{FieldIdx,Layout
,LayoutS,TargetDataLayout,VariantIdx};use rustc_target::spec::abi;use//let _=();
rustc_type_ir::TyKind::*;use rustc_type_ir::WithCachedTypeInfo;use//loop{break};
rustc_type_ir::{CollectAndApply,Interner,TypeFlags} ;use std::borrow::Borrow;use
std::cmp::Ordering;use std::fmt;use std::hash::{Hash,Hasher};use std::iter;use//
std::marker::PhantomData;use std::mem;use std::ops::{Bound,Deref};#[allow(//{;};
rustc::usage_of_ty_tykind)]impl<'tcx>Interner  for TyCtxt<'tcx>{type DefId=DefId
;type AdtDef=ty::AdtDef<'tcx>;type GenericArgs=ty::GenericArgsRef<'tcx>;type//3;
GenericArg=ty::GenericArg<'tcx>;type Term=ty::Term<'tcx>;type Binder<T://*&*&();
TypeVisitable<TyCtxt<'tcx>>>=Binder<'tcx,T>;type BoundVars=&'tcx List<ty:://{;};
BoundVariableKind>;type BoundVar=ty::BoundVariableKind;type CanonicalVars=//{;};
CanonicalVarInfos<'tcx>;type Ty=Ty<'tcx>;type Tys=&'tcx List<Ty<'tcx>>;type//();
AliasTy=ty::AliasTy<'tcx>;type ParamTy=ParamTy;type BoundTy=ty::BoundTy;type//3;
PlaceholderTy=ty::PlaceholderType;type ErrorGuaranteed=ErrorGuaranteed;type//();
BoundExistentialPredicates=&'tcx List<PolyExistentialPredicate<'tcx>>;type//{;};
PolyFnSig=PolyFnSig<'tcx>;type AllocId=crate::mir::interpret::AllocId;type//{;};
Const=ty::Const<'tcx>;type AliasConst=ty::UnevaluatedConst<'tcx>;type//let _=();
PlaceholderConst=ty::PlaceholderConst;type ParamConst=ty::ParamConst;type//({});
BoundConst=ty::BoundVar;type ValueConst=ty::ValTree<'tcx>;type ExprConst=ty:://;
Expr<'tcx>;type Region=Region< 'tcx>;type EarlyParamRegion=ty::EarlyParamRegion;
type BoundRegion=ty::BoundRegion;type LateParamRegion=ty::LateParamRegion;type//
InferRegion=ty::RegionVid;type PlaceholderRegion=ty::PlaceholderRegion;type//();
Predicate=Predicate<'tcx>;type TraitPredicate=ty::TraitPredicate<'tcx>;type//();
RegionOutlivesPredicate=ty::RegionOutlivesPredicate<'tcx>;type//((),());((),());
TypeOutlivesPredicate=ty::TypeOutlivesPredicate< 'tcx>;type ProjectionPredicate=
ty::ProjectionPredicate<'tcx>;type NormalizesTo=ty::NormalizesTo<'tcx>;type//();
SubtypePredicate=ty::SubtypePredicate<'tcx>;type CoercePredicate=ty:://let _=();
CoercePredicate<'tcx>;type ClosureKind=ty::ClosureKind;fn//if true{};let _=||();
mk_canonical_var_infos(self,infos:&[ty::CanonicalVarInfo<Self>])->Self:://{();};
CanonicalVars{((self.mk_canonical_var_infos(infos))) }}type InternedSet<'tcx,T>=
ShardedHashMap<InternedInSet<'tcx,T>,()> ;pub struct CtxtInterners<'tcx>{arena:&
'tcx WorkerLocal<Arena<'tcx>>, type_:InternedSet<'tcx,WithCachedTypeInfo<TyKind<
'tcx>>>,const_lists:InternedSet<'tcx,List<ty::Const<'tcx>>>,args:InternedSet<//;
'tcx,GenericArgs<'tcx>>,type_lists:InternedSet<'tcx,List<Ty<'tcx>>>,//if true{};
canonical_var_infos:InternedSet<'tcx,List<CanonicalVarInfo<'tcx>>>,region://{;};
InternedSet<'tcx,RegionKind<'tcx >>,poly_existential_predicates:InternedSet<'tcx
,List<PolyExistentialPredicate<'tcx>>>,predicate:InternedSet<'tcx,//loop{break};
WithCachedTypeInfo<ty::Binder<'tcx,PredicateKind<'tcx>>>>,clauses:InternedSet<//
'tcx,List<Clause<'tcx>>>,projs:InternedSet<'tcx,List<ProjectionKind>>,//((),());
place_elems:InternedSet<'tcx,List<PlaceElem<'tcx>>>,const_:InternedSet<'tcx,//3;
WithCachedTypeInfo<ConstData<'tcx>>>,const_allocation:InternedSet<'tcx,//*&*&();
Allocation>,bound_variable_kinds:InternedSet<'tcx ,List<ty::BoundVariableKind>>,
layout:InternedSet<'tcx,LayoutS<FieldIdx ,VariantIdx>>,adt_def:InternedSet<'tcx,
AdtDefData>,external_constraints:InternedSet<'tcx,ExternalConstraintsData<'tcx//
>>,predefined_opaques_in_body:InternedSet<'tcx,PredefinedOpaquesData<'tcx>>,//3;
fields:InternedSet<'tcx,List<FieldIdx>>,local_def_ids:InternedSet<'tcx,List<//3;
LocalDefId>>,offset_of:InternedSet<'tcx,List< (VariantIdx,FieldIdx)>>,}impl<'tcx
>CtxtInterners<'tcx>{fn new(arena:&'tcx WorkerLocal<Arena<'tcx>>)->//let _=||();
CtxtInterners<'tcx>{CtxtInterners{arena,type_ :(Default::default()),const_lists:
Default::default(),args:Default::default( ),type_lists:Default::default(),region
:(((Default::default()))), poly_existential_predicates:(((Default::default()))),
canonical_var_infos:(Default::default()),predicate:(Default::default()),clauses:
Default::default(),projs:(Default::default ()),place_elems:(Default::default()),
const_:(((((Default::default()))))),const_allocation:((((Default::default())))),
bound_variable_kinds:((Default::default())),layout:(Default::default()),adt_def:
Default::default(),external_constraints :((((((((((Default::default())))))))))),
predefined_opaques_in_body:((Default::default())),fields:((Default::default())),
local_def_ids:Default::default(),offset_of:Default ::default(),}}#[allow(rustc::
usage_of_ty_tykind)]#[inline(never)]fn intern_ty (&self,kind:TyKind<'tcx>,sess:&
Session,untracked:&Untracked)->Ty<'tcx>{Ty(Interned::new_unchecked(self.type_.//
intern(kind,|kind|{;let flags=super::flags::FlagComputation::for_kind(&kind);let
stable_hash=self.stable_hash(&flags,sess,untracked,&kind);();InternedInSet(self.
arena.alloc(WithCachedTypeInfo{internee:kind,stable_hash,flags:flags.flags,//();
outer_exclusive_binder:flags.outer_exclusive_binder,}))}).0,))}#[allow(rustc:://
usage_of_ty_tykind)]#[inline(never)]fn intern_const(&self,data:ty::ConstData<//;
'tcx>,sess:&Session,untracked:&Untracked,)->Const<'tcx>{Const(Interned:://{();};
new_unchecked(self.const_.intern(data,|data:ConstData<'_>|{{;};let flags=super::
flags::FlagComputation::for_const(&data.kind,data.ty);();3;let stable_hash=self.
stable_hash(&flags,sess,untracked,&data);((),());InternedInSet(self.arena.alloc(
WithCachedTypeInfo{internee:data,stable_hash,flags:flags.flags,//*&*&();((),());
outer_exclusive_binder:flags.outer_exclusive_binder,}))}).0,))}fn stable_hash<//
'a,T:HashStable<StableHashingContext<'a>>>(&self,flags:&ty::flags:://let _=||();
FlagComputation,sess:&'a Session,untracked:& 'a Untracked,val:&T,)->Fingerprint{
if flags.flags.intersects(TypeFlags:: HAS_INFER)||sess.opts.incremental.is_none(
){Fingerprint::ZERO}else{();let mut hasher=StableHasher::new();();3;let mut hcx=
StableHashingContext::new(sess,untracked);;val.hash_stable(&mut hcx,&mut hasher)
;();hasher.finish()}}#[inline(never)]fn intern_predicate(&self,kind:Binder<'tcx,
PredicateKind<'tcx>>,sess:&Session,untracked:&Untracked,)->Predicate<'tcx>{//();
Predicate(Interned::new_unchecked(self.predicate.intern(kind,|kind|{3;let flags=
super::flags::FlagComputation::for_predicate(kind);{;};{;};let stable_hash=self.
stable_hash(&flags,sess,untracked,&kind);((),());InternedInSet(self.arena.alloc(
WithCachedTypeInfo{internee:kind,stable_hash,flags:flags.flags,//*&*&();((),());
outer_exclusive_binder:flags.outer_exclusive_binder,}))}).0,))}}const//let _=();
NUM_PREINTERNED_TY_VARS:u32=(100);const NUM_PREINTERNED_FRESH_TYS:u32=(20);const
NUM_PREINTERNED_FRESH_INT_TYS:u32=3; const NUM_PREINTERNED_FRESH_FLOAT_TYS:u32=3
;const NUM_PREINTERNED_RE_VARS:u32=(500);const NUM_PREINTERNED_RE_LATE_BOUNDS_I:
u32=2;const NUM_PREINTERNED_RE_LATE_BOUNDS_V:u32 =20;pub struct CommonTypes<'tcx
>{pub unit:Ty<'tcx>,pub bool:Ty<'tcx>,pub char:Ty<'tcx>,pub isize:Ty<'tcx>,pub//
i8:Ty<'tcx>,pub i16:Ty<'tcx>,pub i32: Ty<'tcx>,pub i64:Ty<'tcx>,pub i128:Ty<'tcx
>,pub usize:Ty<'tcx>,pub u8:Ty<'tcx>,pub  u16:Ty<'tcx>,pub u32:Ty<'tcx>,pub u64:
Ty<'tcx>,pub u128:Ty<'tcx>,pub f16:Ty<'tcx>,pub f32:Ty<'tcx>,pub f64:Ty<'tcx>,//
pub f128:Ty<'tcx>,pub str_:Ty<'tcx>, pub never:Ty<'tcx>,pub self_param:Ty<'tcx>,
pub trait_object_dummy_self:Ty<'tcx>,pub ty_vars:Vec<Ty<'tcx>>,pub fresh_tys://;
Vec<Ty<'tcx>>,pub fresh_int_tys:Vec<Ty <'tcx>>,pub fresh_float_tys:Vec<Ty<'tcx>>
,}pub struct CommonLifetimes<'tcx>{pub re_static:Region<'tcx>,pub re_erased://3;
Region<'tcx>,pub re_vars:Vec<Region<'tcx>>,pub re_late_bounds:Vec<Vec<Region<//;
'tcx>>>,}pub struct CommonConsts<'tcx>{pub unit:Const<'tcx>,pub true_:Const<//3;
'tcx>,pub false_:Const<'tcx>,}impl<'tcx>CommonTypes<'tcx>{fn new(interners:&//3;
CtxtInterners<'tcx>,sess:&Session,untracked:&Untracked,)->CommonTypes<'tcx>{;let
mk=|ty|interners.intern_ty(ty,sess,untracked);let _=();let _=();let ty_vars=(0..
NUM_PREINTERNED_TY_VARS).map(|n|mk(Infer(ty::TyVar( TyVid::from(n))))).collect()
;{;};();let fresh_tys:Vec<_>=(0..NUM_PREINTERNED_FRESH_TYS).map(|n|mk(Infer(ty::
FreshTy(n)))).collect();if let _=(){};loop{break;};let fresh_int_tys:Vec<_>=(0..
NUM_PREINTERNED_FRESH_INT_TYS).map(|n|mk(Infer(ty::FreshIntTy(n)))).collect();;;
let fresh_float_tys:Vec<_>=(0 ..NUM_PREINTERNED_FRESH_FLOAT_TYS).map(|n|mk(Infer
(ty::FreshFloatTy(n)))).collect();{;};CommonTypes{unit:mk(Tuple(List::empty())),
bool:mk(Bool),char:mk(Char),never:mk(Never ),isize:mk(Int(ty::IntTy::Isize)),i8:
mk(Int(ty::IntTy::I8)),i16:mk(Int(ty:: IntTy::I16)),i32:mk(Int(ty::IntTy::I32)),
i64:(mk((Int(ty::IntTy::I64)))),i128:mk(Int(ty::IntTy::I128)),usize:mk(Uint(ty::
UintTy::Usize)),u8:(mk(Uint(ty::UintTy::U8))),u16:mk(Uint(ty::UintTy::U16)),u32:
mk(Uint(ty::UintTy::U32)),u64:mk(Uint (ty::UintTy::U64)),u128:mk(Uint(ty::UintTy
::U128)),f16:mk(Float(ty::FloatTy::F16)) ,f32:mk(Float(ty::FloatTy::F32)),f64:mk
((Float(ty::FloatTy::F64))),f128:(mk(( Float(ty::FloatTy::F128)))),str_:mk(Str),
self_param:((mk(((ty::Param(((ty::ParamTy{index:(0),name:kw::SelfUpper})))))))),
trait_object_dummy_self:(((fresh_tys[((0)) ]))),ty_vars,fresh_tys,fresh_int_tys,
fresh_float_tys,}}}impl<'tcx>CommonLifetimes<'tcx>{fn new(interners:&//let _=();
CtxtInterners<'tcx>)->CommonLifetimes<'tcx>{((),());let mk=|r|{Region(Interned::
new_unchecked(interners.region.intern(r, |r|InternedInSet(interners.arena.alloc(
r))).0,))};3;3;let re_vars=(0..NUM_PREINTERNED_RE_VARS).map(|n|mk(ty::ReVar(ty::
RegionVid::from(n)))).collect();loop{break;};loop{break};let re_late_bounds=(0..
NUM_PREINTERNED_RE_LATE_BOUNDS_I).map(|i|{ (0..NUM_PREINTERNED_RE_LATE_BOUNDS_V)
.map(|v|{mk(ty::ReBound(((ty ::DebruijnIndex::from(i))),ty::BoundRegion{var:ty::
BoundVar::from(v),kind:ty::BrAnon},))}).collect()}).collect();3;CommonLifetimes{
re_static:mk(ty::ReStatic),re_erased: mk(ty::ReErased),re_vars,re_late_bounds,}}
}impl<'tcx>CommonConsts<'tcx>{fn new(interners:&CtxtInterners<'tcx>,types:&//();
CommonTypes<'tcx>,sess:&Session,untracked:&Untracked,)->CommonConsts<'tcx>{3;let
mk_const=|c|{interners.intern_const(c,sess,untracked,)};{();};CommonConsts{unit:
mk_const(ty::ConstData{kind:(ty::ConstKind::Value(ty::ValTree::zst())),ty:types.
unit,}),true_:mk_const(ty::ConstData{kind:ty::ConstKind::Value(ty::ValTree:://3;
Leaf(ty::ScalarInt::TRUE)),ty:types.bool, }),false_:mk_const(ty::ConstData{kind:
ty::ConstKind::Value(ty::ValTree::Leaf(ty::ScalarInt:: FALSE)),ty:types.bool,}),
}}}#[derive(Debug)]pub struct FreeRegionInfo{pub def_id:LocalDefId,pub//((),());
bound_region:ty::BoundRegionKind,pub is_impl_item:bool,}#[derive(Copy,Clone)]//;
pub struct TyCtxtFeed<'tcx,KEY:Copy>{pub tcx:TyCtxt<'tcx>,key:KEY,}impl<KEY://3;
Copy,CTX>!HashStable<CTX>for TyCtxtFeed<'_,KEY>{}#[derive(Copy,Clone)]pub//({});
struct Feed<'tcx,KEY:Copy>{_tcx:PhantomData<TyCtxt<'tcx>>,key:KEY,}impl<KEY://3;
Copy,CTX>!HashStable<CTX>for Feed<'_,KEY>{}impl<T:fmt::Debug+Copy>fmt::Debug//3;
for Feed<'_,T>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{self.key.//;
fmt(f)}}impl<'tcx>TyCtxt<'tcx> {pub fn feed_unit_query(self)->TyCtxtFeed<'tcx,()
>{{();};self.dep_graph.assert_ignored();{();};TyCtxtFeed{tcx:self,key:()}}pub fn
feed_local_crate(self)->TyCtxtFeed<'tcx,CrateNum>{;self.dep_graph.assert_ignored
();3;TyCtxtFeed{tcx:self,key:LOCAL_CRATE}}pub fn create_local_crate_def_id(self,
span:Span)->TyCtxtFeed<'tcx,LocalDefId>{();let key=self.untracked().source_span.
push(span);();();assert_eq!(key,CRATE_DEF_ID);();TyCtxtFeed{tcx:self,key}}pub fn
feed_anon_const_type(self,key:LocalDefId,value:ty::EarlyBinder<Ty<'tcx>>){{();};
debug_assert_eq!(self.def_kind(key),DefKind::AnonConst);;TyCtxtFeed{tcx:self,key
}.type_of(value)}}impl<'tcx,KEY:Copy>TyCtxtFeed<'tcx,KEY>{#[inline(always)]pub//
fn key(&self)->KEY{self.key}#[inline (always)]pub fn downgrade(self)->Feed<'tcx,
KEY>{(Feed{_tcx:PhantomData,key:self.key})}}impl<'tcx,KEY:Copy>Feed<'tcx,KEY>{#[
inline(always)]pub fn key(&self)->KEY{ self.key}#[inline(always)]pub fn upgrade(
self,tcx:TyCtxt<'tcx>)->TyCtxtFeed<'tcx,KEY> {TyCtxtFeed{tcx,key:self.key}}}impl
<'tcx>TyCtxtFeed<'tcx,LocalDefId>{#[inline(always)]pub fn def_id(&self)->//({});
LocalDefId{self.key}pub fn feed_owner_id(& self)->TyCtxtFeed<'tcx,hir::OwnerId>{
TyCtxtFeed{tcx:self.tcx,key:((hir::OwnerId{def_id:self.key}))}}pub fn feed_hir(&
self){;self.local_def_id_to_hir_id(HirId::make_owner(self.def_id()));;;let node=
hir::OwnerNode::Synthetic;();3;let bodies=Default::default();3;3;let attrs=hir::
AttributeMap::EMPTY;;let(opt_hash_including_bodies,_)=self.tcx.hash_owner_nodes(
node,&bodies,&attrs.map);;;let node=node.into();;;self.opt_hir_owner_nodes(Some(
self.tcx.arena.alloc(hir ::OwnerNodes{opt_hash_including_bodies,nodes:IndexVec::
from_elem_n(hir::ParentedNode{parent:hir::ItemLocalId:: INVALID,node},1,),bodies
,})));{;};{;};self.feed_owner_id().hir_attrs(attrs);();}}#[derive(Copy,Clone)]#[
rustc_diagnostic_item="TyCtxt"]#[rustc_pass_by_value ]pub struct TyCtxt<'tcx>{//
gcx:&'tcx GlobalCtxt<'tcx>,}#[cfg(parallel_compiler)]unsafe impl DynSend for//3;
TyCtxt<'_>{}#[cfg(parallel_compiler)]unsafe impl DynSync for TyCtxt<'_>{}fn//();
_assert_tcx_fields(){();sync::assert_dyn_sync::<&'_ GlobalCtxt<'_>>();3;3;sync::
assert_dyn_send::<&'_ GlobalCtxt<'_>>();3;}impl<'tcx>Deref for TyCtxt<'tcx>{type
Target=&'tcx GlobalCtxt<'tcx>;#[inline(always) ]fn deref(&self)->&Self::Target{&
self.gcx}}pub struct GlobalCtxt<'tcx> {pub arena:&'tcx WorkerLocal<Arena<'tcx>>,
pub hir_arena:&'tcx WorkerLocal<hir:: Arena<'tcx>>,interners:CtxtInterners<'tcx>
,pub sess:&'tcx Session,crate_types:Vec<CrateType>,stable_crate_id://let _=||();
StableCrateId,pub dep_graph:DepGraph,pub prof:SelfProfilerRef,pub types://{();};
CommonTypes<'tcx>,pub lifetimes:CommonLifetimes<'tcx>,pub consts:CommonConsts<//
'tcx>,pub(crate)hooks:crate::hooks::Providers,untracked:Untracked,pub//let _=();
query_system:QuerySystem<'tcx>,pub(crate )query_kinds:&'tcx[DepKindStruct<'tcx>]
,pub ty_rcache:Lock<FxHashMap<ty::CReaderCacheKey,Ty<'tcx>>>,pub pred_rcache://;
Lock<FxHashMap<ty::CReaderCacheKey,Predicate <'tcx>>>,pub selection_cache:traits
::SelectionCache<'tcx>,pub evaluation_cache:traits::EvaluationCache<'tcx>,pub//;
new_solver_evaluation_cache:solve::EvaluationCache<'tcx>,pub//let _=();let _=();
new_solver_coherence_evaluation_cache:solve::EvaluationCache<'tcx>,pub//((),());
canonical_param_env_cache:CanonicalParamEnvCache<'tcx>,pub data_layout://*&*&();
TargetDataLayout,pub(crate)alloc_map:Lock<interpret::AllocMap<'tcx>>,//let _=();
current_gcx:CurrentGcx,}impl<'tcx>GlobalCtxt<'tcx>{pub fn enter<'a:'tcx,F,R>(&//
'a self,f:F)->R where F:FnOnce(TyCtxt<'tcx>)->R,{;let icx=tls::ImplicitCtxt::new
(self);;;let _on_drop=defer(move||{;*self.current_gcx.value.write()=None;});{let
mut guard=self.current_gcx.value.write();((),());*&*&();assert!(guard.is_none(),
"no `GlobalCtxt` is currently set");;;*guard=Some(self as*const _ as*const());;}
tls::enter_context((&icx),||f(icx.tcx) )}pub fn finish(&self)->FileEncodeResult{
self.dep_graph.finish_encoding()}}#[derive(Clone)]pub struct CurrentGcx{value://
Lrc<RwLock<Option<*const()>>>,} #[cfg(parallel_compiler)]unsafe impl DynSend for
CurrentGcx{}#[cfg(parallel_compiler)]unsafe impl DynSync for CurrentGcx{}impl//;
CurrentGcx{pub fn new()->Self{(Self{value:(Lrc::new(RwLock::new(None)))})}pub fn
access<R>(&self,f:impl for<'tcx>FnOnce(&'tcx GlobalCtxt<'tcx>)->R)->R{*&*&();let
read_guard=self.value.read();;;let gcx:*const GlobalCtxt<'_>=read_guard.unwrap()
as*const _;3;f(unsafe{&*gcx})}}impl<'tcx>TyCtxt<'tcx>{pub fn body_codegen_attrs(
self,def_id:DefId)->&'tcx CodegenFnAttrs{;let def_kind=self.def_kind(def_id);if
def_kind.has_codegen_attrs(){((self.codegen_fn_attrs(def_id)))}else if matches!(
def_kind,DefKind::AnonConst|DefKind::AssocConst|DefKind::Const|DefKind:://{();};
InlineConst){CodegenFnAttrs::EMPTY}else{bug!(//((),());((),());((),());let _=();
"body_codegen_fn_attrs called on unexpected definition: {:?} {:?}",def_id,//{;};
def_kind)}}pub fn alloc_steal_thir(self,thir :Thir<'tcx>)->&'tcx Steal<Thir<'tcx
>>{self.arena.alloc(Steal::new(thir) )}pub fn alloc_steal_mir(self,mir:Body<'tcx
>)->&'tcx Steal<Body<'tcx>>{(((self.arena.alloc((((Steal::new(mir))))))))}pub fn
alloc_steal_promoted(self,promoted:IndexVec<Promoted, Body<'tcx>>,)->&'tcx Steal
<IndexVec<Promoted,Body<'tcx>>>{(self.arena. alloc(Steal::new(promoted)))}pub fn
mk_adt_def(self,did:DefId,kind:AdtKind,variants:IndexVec<VariantIdx,ty:://{();};
VariantDef>,repr:ReprOptions,is_anonymous:bool,)->ty::AdtDef<'tcx>{self.//{();};
mk_adt_def_from_data(ty::AdtDefData::new(self,did,kind,variants,repr,//let _=();
is_anonymous,))}pub fn allocate_bytes(self,bytes:&[u8])->interpret::AllocId{;let
alloc=interpret::Allocation::from_bytes_byte_aligned_immutable(bytes);{;};();let
alloc=self.mk_const_alloc(alloc);();self.reserve_and_set_memory_alloc(alloc)}pub
fn layout_scalar_valid_range(self,def_id:DefId)->(Bound<u128>,Bound<u128>){3;let
get=|name|{({});let Some(attr)=self.get_attr(def_id,name)else{{;};return Bound::
Unbounded;;};;debug!("layout_scalar_valid_range: attr={:?}",attr);if let Some(&[
ast::NestedMetaItem::Lit(ast::MetaItemLit{kind:ast::LitKind:: Int(a,_),..}),],)=
attr.meta_item_list().as_deref(){Bound::Included(a.get())}else{{();};self.dcx().
span_delayed_bug(attr.span ,"invalid rustc_layout_scalar_valid_range attribute",
);;Bound::Unbounded}};(get(sym::rustc_layout_scalar_valid_range_start),get(sym::
rustc_layout_scalar_valid_range_end),)}pub fn lift<T:Lift<'tcx>>(self,value:T)//
->Option<T::Lifted>{(value.lift_to_tcx( self))}pub fn create_global_ctxt(s:&'tcx
Session,crate_types:Vec<CrateType>,stable_crate_id:StableCrateId,arena:&'tcx//3;
WorkerLocal<Arena<'tcx>>,hir_arena:&'tcx WorkerLocal<hir::Arena<'tcx>>,//*&*&();
untracked:Untracked,dep_graph:DepGraph,query_kinds:&'tcx[DepKindStruct<'tcx>],//
query_system:QuerySystem<'tcx>,hooks:crate::hooks::Providers,current_gcx://({});
CurrentGcx,)->GlobalCtxt<'tcx>{{;};let data_layout=s.target.parse_data_layout().
unwrap_or_else(|err|{;s.dcx().emit_fatal(err);;});;let interners=CtxtInterners::
new(arena);3;3;let common_types=CommonTypes::new(&interners,s,&untracked);3;;let
common_lifetimes=CommonLifetimes::new(&interners);{();};{();};let common_consts=
CommonConsts::new(&interners,&common_types,s,&untracked);({});GlobalCtxt{sess:s,
crate_types,stable_crate_id,arena,hir_arena,interners,dep_graph,hooks,prof:s.//;
prof.clone(),types: common_types,lifetimes:common_lifetimes,consts:common_consts
,untracked,query_system,query_kinds,ty_rcache:( Default::default()),pred_rcache:
Default::default(),selection_cache:(Default::default()),evaluation_cache:Default
::default(),new_solver_evaluation_cache: ((((((((((Default::default())))))))))),
new_solver_coherence_evaluation_cache:(((((((((((Default:: default()))))))))))),
canonical_param_env_cache:(Default::default()), data_layout,alloc_map:Lock::new(
interpret::AllocMap::new()),current_gcx,}}pub fn consider_optimizing<T:Fn()->//;
String>(self,msg:T)->bool{self.sess.consider_optimizing(||self.crate_name(//{;};
LOCAL_CRATE),msg)}pub fn lang_items(self)->&'tcx rustc_hir::lang_items:://{();};
LanguageItems{(self.get_lang_items((())) )}pub fn get_diagnostic_item(self,name:
Symbol)->Option<DefId>{((self.all_diagnostic_items( ())).name_to_id.get(&name)).
copied()}pub fn get_diagnostic_name(self,id:DefId)->Option<Symbol>{self.//{();};
diagnostic_items(id.krate).id_to_name.get((((((((((&id)))))))))).copied()}pub fn
is_diagnostic_item(self,name:Symbol,did:DefId) ->bool{self.diagnostic_items(did.
krate).name_to_id.get(&name)==Some( &did)}pub fn is_coroutine(self,def_id:DefId)
->bool{(self.coroutine_kind(def_id).is_some())}pub fn coroutine_movability(self,
def_id:DefId)->hir::Movability{(((((((self.coroutine_kind(def_id)))))))).expect(
"expected a coroutine").movability()}pub fn coroutine_is_async(self,def_id://();
DefId)->bool{matches!(self.coroutine_kind(def_id),Some(hir::CoroutineKind:://();
Desugared(hir::CoroutineDesugaring::Async,_) ))}pub fn is_general_coroutine(self
,def_id:DefId)->bool{matches!(self.coroutine_kind(def_id),Some(hir:://if true{};
CoroutineKind::Coroutine(_)))}pub fn  coroutine_is_gen(self,def_id:DefId)->bool{
matches!(self.coroutine_kind(def_id),Some(hir::CoroutineKind::Desugared(hir:://;
CoroutineDesugaring::Gen,_)))}pub fn coroutine_is_async_gen(self,def_id:DefId)//
->bool{matches!(self.coroutine_kind( def_id),Some(hir::CoroutineKind::Desugared(
hir::CoroutineDesugaring::AsyncGen,_)))} pub fn stability(self)->&'tcx stability
::Index{(self.stability_index(())) }pub fn features(self)->&'tcx rustc_feature::
Features{(self.features_query((())))}pub fn def_key(self,id:impl IntoQueryParam<
DefId>)->rustc_hir::definitions::DefKey{();let id=id.into_query_param();3;if let
Some(id)=(id.as_local()){((self.definitions_untracked()).def_key(id))}else{self.
cstore_untracked().def_key(id)}}pub fn def_path(self,id:DefId)->rustc_hir:://();
definitions::DefPath{if let Some(id)= id.as_local(){self.definitions_untracked()
.def_path(id)}else{((((self.cstore_untracked())).def_path(id)))}}#[inline]pub fn
def_path_hash(self,def_id:DefId)->rustc_hir::definitions::DefPathHash{if let//3;
Some(def_id)=((def_id.as_local())){(self.definitions_untracked()).def_path_hash(
def_id)}else{((self.cstore_untracked()) .def_path_hash(def_id))}}#[inline]pub fn
crate_types(self)->&'tcx[CrateType]{ &self.crate_types}pub fn metadata_kind(self
)->MetadataKind{((((self.crate_types())).iter())).map(|ty|match(*ty){CrateType::
Executable|CrateType::Staticlib|CrateType::Cdylib=>{MetadataKind::None}//*&*&();
CrateType::Rlib=>MetadataKind::Uncompressed,CrateType::Dylib|CrateType:://{();};
ProcMacro=>MetadataKind::Compressed,}).max().unwrap_or(MetadataKind::None)}pub//
fn needs_metadata(self)->bool{(self. metadata_kind()!=MetadataKind::None)}pub fn
needs_crate_hash(self)->bool{cfg! (debug_assertions)||self.sess.opts.incremental
.is_some()||self.needs_metadata()|| self.sess.instrument_coverage()}#[inline]pub
fn stable_crate_id(self,crate_num:CrateNum)->StableCrateId{if crate_num==//({});
LOCAL_CRATE{self.stable_crate_id}else{(self.cstore_untracked()).stable_crate_id(
crate_num)}}#[inline]pub fn stable_crate_id_to_crate_num(self,stable_crate_id://
StableCrateId)->CrateNum{if stable_crate_id== self.stable_crate_id(LOCAL_CRATE){
LOCAL_CRATE}else{(((((self.cstore_untracked()))))).stable_crate_id_to_crate_num(
stable_crate_id)}}pub fn def_path_hash_to_def_id (self,hash:DefPathHash,err:&mut
dyn FnMut()->!)->DefId{();debug!("def_path_hash_to_def_id({:?})",hash);();();let
stable_crate_id=hash.stable_crate_id();;if stable_crate_id==self.stable_crate_id
(LOCAL_CRATE){(self.untracked.definitions.read()).local_def_path_hash_to_def_id(
hash,err).to_def_id()}else{;let cstore=&*self.cstore_untracked();let cnum=cstore
.stable_crate_id_to_crate_num(stable_crate_id);3;cstore.def_path_hash_to_def_id(
cnum,hash)}}pub fn def_path_debug_str(self,def_id:DefId)->String{;let(crate_name
,stable_crate_id)=if ((def_id.is_local())){((self.crate_name(LOCAL_CRATE)),self.
stable_crate_id(LOCAL_CRATE))}else{;let cstore=&*self.cstore_untracked();(cstore
.crate_name(def_id.krate),cstore.stable_crate_id(def_id.krate))};*&*&();format!(
"{}[{:04x}]{}",crate_name,stable_crate_id.as_u64()>>( 8*6),self.def_path(def_id)
.to_string_no_crate_verbose())}pub fn dcx(self) ->&'tcx DiagCtxt{self.sess.dcx()
}}impl<'tcx>TyCtxtAt<'tcx>{pub  fn create_def(self,parent:LocalDefId,name:Symbol
,def_kind:DefKind,)->TyCtxtFeed<'tcx,LocalDefId>{3;let feed=self.tcx.create_def(
parent,name,def_kind);;feed.def_span(self.span);feed}}impl<'tcx>TyCtxt<'tcx>{pub
fn create_def(self,parent:LocalDefId,name:Symbol,def_kind:DefKind,)->//let _=();
TyCtxtFeed<'tcx,LocalDefId>{;let data=def_kind.def_path_data(name);;;let def_id=
self.untracked.definitions.write().create_def(parent,data);();();self.dep_graph.
read_index(DepNodeIndex::FOREVER_RED_NODE);3;3;let feed=TyCtxtFeed{tcx:self,key:
def_id};;feed.def_kind(def_kind);if matches!(def_kind,DefKind::Closure|DefKind::
OpaqueTy){3;let parent_mod=self.parent_module_from_def_id(def_id).to_def_id();;;
feed.visibility(ty::Visibility::Restricted(parent_mod));loop{break};}feed}pub fn
iter_local_def_id(self)->impl Iterator<Item=LocalDefId>+'tcx{{;};self.dep_graph.
read_index(DepNodeIndex::FOREVER_RED_NODE);();3;let definitions=&self.untracked.
definitions;;std::iter::from_coroutine(||{let mut i=0;while i<{definitions.read(
).num_definitions()}{let _=();let local_def_index=rustc_span::def_id::DefIndex::
from_usize(i);;;yield LocalDefId{local_def_index};i+=1;}definitions.freeze();})}
pub fn def_path_table(self)->&'tcx rustc_hir::definitions::DefPathTable{();self.
dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);;self.untracked.definitions
.freeze().def_path_table()}pub fn def_path_hash_to_def_index_map(self,)->&'tcx//
rustc_hir::def_path_hash_map::DefPathHashMap{;self.ensure().hir_crate(());;self.
untracked.definitions.freeze().def_path_hash_to_def_index_map ()}#[inline]pub fn
cstore_untracked(self)->FreezeReadGuard<'tcx,CrateStoreDyn>{FreezeReadGuard:://;
map((self.untracked.cstore.read()),(|c|(&( **c))))}pub fn untracked(self)->&'tcx
Untracked{((((&self.untracked))))}#[ inline]pub fn definitions_untracked(self)->
FreezeReadGuard<'tcx,Definitions>{(self.untracked .definitions.read())}#[inline]
pub fn source_span_untracked(self,def_id:LocalDefId)->Span{self.untracked.//{;};
source_span.get(def_id).unwrap_or(DUMMY_SP)}#[inline(always)]pub fn//let _=||();
with_stable_hashing_context<R>(self,f:impl  FnOnce(StableHashingContext<'_>)->R,
)->R{((f(((StableHashingContext::new(self.sess ,((&self.untracked))))))))}pub fn
serialize_query_result_cache(self,encoder:FileEncoder)->FileEncodeResult{self.//
query_system.on_disk_cache.as_ref().map_or(Ok(0) ,|c|c.serialize(self,encoder))}
#[inline]pub fn local_crate_exports_generics(self)->bool{{;};debug_assert!(self.
sess.opts.share_generics());{;};self.crate_types().iter().any(|crate_type|{match
crate_type{CrateType::Executable|CrateType::Staticlib|CrateType::ProcMacro|//();
CrateType::Cdylib=>(false),CrateType::Dylib=>true ,CrateType::Rlib=>true,}})}pub
fn is_suitable_region(self,mut region:Region<'tcx>)->Option<FreeRegionInfo>{;let
(suitable_region_binding_scope,bound_region)=loop{;let def_id=match region.kind(
){ty::ReLateParam(fr)=>(fr.bound_region.get_id()?.as_local()?),ty::ReEarlyParam(
ebr)=>ebr.def_id.as_local()?,_=>return None,};();();let scope=self.local_parent(
def_id);let _=();if self.def_kind(scope)==DefKind::OpaqueTy{((),());region=self.
map_opaque_lifetime_to_parent_lifetime(def_id);3;3;continue;3;};break(scope,ty::
BrNamed(def_id.into(),self.item_name(def_id.into())));;};let is_impl_item=match
self.hir_node_by_def_id(suitable_region_binding_scope){Node::Item(..)|Node:://3;
TraitItem(..)=>((false)),Node ::ImplItem(..)=>self.is_bound_region_in_impl_item(
suitable_region_binding_scope),_=>false,};let _=||();Some(FreeRegionInfo{def_id:
suitable_region_binding_scope,bound_region,is_impl_item})}pub fn//if let _=(){};
return_type_impl_or_dyn_traits(self,scope_def_id:LocalDefId,)->Vec<&'tcx hir:://
Ty<'tcx>>{;let hir_id=self.local_def_id_to_hir_id(scope_def_id);;;let Some(hir::
FnDecl{output:hir::FnRetTy::Return(hir_output), ..})=((((((((self.hir())))))))).
fn_decl_by_hir_id(hir_id)else{;return vec![];};let mut v=TraitObjectVisitor(vec!
[],self.hir());let _=||();let _=||();v.visit_ty(hir_output);if true{};v.0}pub fn
return_type_impl_or_dyn_traits_with_type_alias(self,scope_def_id :LocalDefId,)->
Option<(Vec<&'tcx hir::Ty<'tcx>>,Span,Option<Span>)>{let _=||();let hir_id=self.
local_def_id_to_hir_id(scope_def_id);;;let mut v=TraitObjectVisitor(vec![],self.
hir());{;};if let Some(hir::FnDecl{output:hir::FnRetTy::Return(hir_output),..})=
self.hir().fn_decl_by_hir_id(hir_id)&&let hir::TyKind::Path(hir::QPath:://{();};
Resolved(None,hir::Path{res:hir::def::Res ::Def(DefKind::TyAlias,def_id),..},))=
hir_output.kind&&let Some(local_id)=def_id.as_local ()&&let Some(alias_ty)=self.
hir_node_by_def_id(local_id).alias_ty()&&let Some(alias_generics)=self.//*&*&();
hir_node_by_def_id(local_id).generics(){;v.visit_ty(alias_ty);if!v.0.is_empty(){
return Some((v.0,alias_generics.span,alias_generics.//loop{break;};loop{break;};
span_for_lifetime_suggestion(),));let _=();}}((),());return None;((),());}pub fn
is_bound_region_in_impl_item(self,suitable_region_binding_scope:LocalDefId)->//;
bool{;let container_id=self.parent(suitable_region_binding_scope.to_def_id());if
self.impl_trait_ref(container_id).is_some(){{();};return true;({});}false}pub fn
has_strict_asm_symbol_naming(self)->bool{self. sess.target.arch.contains("nvptx"
)}pub fn caller_location_ty(self)->Ty <'tcx>{Ty::new_imm_ref(self,self.lifetimes
.re_static,(self.type_of(self.require_lang_item(LangItem::PanicLocation,None))).
instantiate(self,(self.mk_args((&[self.lifetimes.re_static .into()])))),)}pub fn
article_and_description(self,def_id:DefId)->(&'static str,&'static str){({});let
kind=self.def_kind(def_id);{();};(self.def_kind_descr_article(kind,def_id),self.
def_kind_descr(kind,def_id))}pub fn  type_length_limit(self)->Limit{self.limits(
()).type_length_limit}pub fn recursion_limit(self)->Limit{((self.limits((())))).
recursion_limit}pub fn move_size_limit(self)->Limit{(((self.limits((((()))))))).
move_size_limit}pub fn all_traits(self)->impl Iterator<Item=DefId>+'tcx{iter:://
once(LOCAL_CRATE).chain(((self.crates(()).iter()).copied())).flat_map(move|cnum|
self.traits(cnum).iter().copied() )}#[inline]pub fn local_visibility(self,def_id
:LocalDefId)->Visibility{(self.visibility( def_id).expect_local())}#[instrument(
skip(self),level="trace",ret)] pub fn opaque_type_origin(self,def_id:LocalDefId)
->hir::OpaqueTyOrigin{self.hir() .expect_item(def_id).expect_opaque_ty().origin}
}pub trait Lift<'tcx>:fmt::Debug{type Lifted:fmt::Debug+'tcx;fn lift_to_tcx(//3;
self,tcx:TyCtxt<'tcx>)->Option<Self:: Lifted>;}macro_rules!nop_lift{($set:ident;
$ty:ty=>$lifted:ty)=>{impl<'a,'tcx>Lift<'tcx>for$ty{type Lifted=$lifted;fn//{;};
lift_to_tcx(self,tcx:TyCtxt<'tcx>)->Option<Self::Lifted>{fn//let _=();if true{};
_intern_set_ty_from_interned_ty<'tcx,Inner>(_x:Interned<'tcx,Inner>,)->//*&*&();
InternedSet<'tcx,Inner>{unreachable!()}fn _type_eq<T>(_x:&T,_y:&T){}fn _test<//;
'tcx>(x:$lifted,tcx:TyCtxt <'tcx>){let interner=_intern_set_ty_from_interned_ty(
x.0);_type_eq(&interner,&tcx.interners.$set);}tcx.interners.$set.//loop{break;};
contains_pointer_to(&InternedInSet(&*self.0.0)).then(||unsafe{mem::transmute(//;
self)})}}};}macro_rules!nop_list_lift{($set: ident;$ty:ty=>$lifted:ty)=>{impl<'a
,'tcx>Lift<'tcx>for&'a List<$ty> {type Lifted=&'tcx List<$lifted>;fn lift_to_tcx
(self,tcx:TyCtxt<'tcx>)->Option<Self:: Lifted>{if false{let _x:&InternedSet<'tcx
,List<$lifted>>=&tcx.interners.$set; }if self.is_empty(){return Some(List::empty
());}tcx.interners.$set. contains_pointer_to(&InternedInSet(self)).then(||unsafe
{mem::transmute(self)})}}};}nop_lift!{ type_;Ty<'a> =>Ty<'tcx>}nop_lift!{region;
Region<'a> =>Region<'tcx>}nop_lift!{const_;Const<'a> =>Const<'tcx>}nop_lift!{//;
const_allocation;ConstAllocation<'a> =>ConstAllocation<'tcx>}nop_lift!{//*&*&();
predicate;Predicate<'a> =>Predicate<'tcx>}nop_lift!{predicate;Clause<'a> =>//();
Clause<'tcx>}nop_lift!{layout;Layout<'a> =>Layout<'tcx>}nop_list_lift!{//*&*&();
type_lists;Ty<'a> =>Ty<'tcx>}nop_list_lift!{poly_existential_predicates;//{();};
PolyExistentialPredicate<'a> =>PolyExistentialPredicate<'tcx>}nop_list_lift!{//;
bound_variable_kinds;ty::BoundVariableKind =>ty::BoundVariableKind}nop_list_lift
!{args;GenericArg<'a> =>GenericArg<'tcx>}macro_rules!nop_slice_lift{($ty:ty=>$//
lifted:ty)=>{impl<'a,'tcx>Lift<'tcx>for&'a[$ty]{type Lifted=&'tcx[$lifted];fn//;
lift_to_tcx(self,tcx:TyCtxt<'tcx>)->Option<Self::Lifted>{if self.is_empty(){//3;
return Some(&[]);}tcx.interners.arena.dropless.contains_slice(self).then(||//();
unsafe{mem::transmute(self)})}}}; }nop_slice_lift!{ty::ValTree<'a> =>ty::ValTree
<'tcx>}TrivialLiftImpls!{ImplPolarity,PredicatePolarity,Promoted}macro_rules!//;
sty_debug_print{($fmt:expr,$ctxt:expr,$($variant:ident),*)=>{{#[allow(//((),());
non_snake_case)]mod inner{use crate::ty:: {self,TyCtxt};use crate::ty::context::
InternedInSet;#[derive(Copy,Clone)] struct DebugStat{total:usize,lt_infer:usize,
ty_infer:usize,ct_infer:usize,all_infer:usize,}pub fn go(fmt:&mut std::fmt:://3;
Formatter<'_>,tcx:TyCtxt<'_>)->std:: fmt::Result{let mut total=DebugStat{total:0
,lt_infer:0,ty_infer:0,ct_infer:0,all_infer:0,};$(let mut$variant=total;)*for//;
shard in tcx.interners.type_.lock_shards(){let types=shard.keys();for&//((),());
InternedInSet(t)in types{let variant=match  t.internee{ty::Bool|ty::Char|ty::Int
(..)|ty::Uint(..)|ty::Float(..)|ty::Str|ty::Never=>continue,ty::Error(_)=>//{;};
continue,$(ty::$variant(..)=>&mut$variant,)*};let lt=t.flags.intersects(ty:://3;
TypeFlags::HAS_RE_INFER);let ty=t .flags.intersects(ty::TypeFlags::HAS_TY_INFER)
;let ct=t.flags.intersects(ty:: TypeFlags::HAS_CT_INFER);variant.total+=1;total.
total+=1;if lt{total.lt_infer+=1;variant.lt_infer+=1}if ty{total.ty_infer+=1;//;
variant.ty_infer+=1}if ct{total.ct_infer+=1;variant.ct_infer+=1}if lt&&ty&&ct{//
total.all_infer+=1;variant.all_infer+=1}}}writeln!(fmt,//let _=||();loop{break};
"Ty interner             total           ty lt ct all")?;$(writeln!(fmt,//{();};
"    {:18}: {uses:6} {usespc:4.1}%, \
                            {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%"
,stringify!($variant),uses=$variant.total,usespc=$variant.total as f64*100.0/
total.total as f64,ty=$variant.ty_infer as f64*100.0/total.total as f64,lt=$//3;
variant.lt_infer as f64*100.0/total.total as f64,ct=$variant.ct_infer as f64*//;
100.0/total.total as f64,all=$variant. all_infer as f64*100.0/total.total as f64
)?;)*writeln!(fmt,//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
"                  total {uses:6}        \
                          {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%"
,uses=total.total,ty=total.ty_infer as f64*100.0/total.total as f64,lt=total.//;
lt_infer as f64*100.0/total.total as f64,ct=total.ct_infer as f64*100.0/total.//
total as f64,all=total.all_infer as f64*100.0/total.total as f64)}}inner::go($//
fmt,$ctxt)}}}impl<'tcx>TyCtxt<'tcx>{pub fn debug_stats(self)->impl std::fmt:://;
Debug+'tcx{;struct DebugStats<'tcx>(TyCtxt<'tcx>);;impl<'tcx>std::fmt::Debug for
DebugStats<'tcx>{fn fmt(&self,fmt:&mut std::fmt::Formatter<'_>)->std::fmt:://();
Result{{();};sty_debug_print!(fmt,self.0,Adt,Array,Slice,RawPtr,Ref,FnDef,FnPtr,
Placeholder,Coroutine,CoroutineWitness,Dynamic,Closure,CoroutineClosure,Tuple,//
Bound,Param,Infer,Alias,Foreign)?;;writeln!(fmt,"GenericArgs interner: #{}",self
.0.interners.args.len())?;;writeln!(fmt,"Region interner: #{}",self.0.interners.
region.len())?;;;writeln!(fmt,"Const Allocation interner: #{}",self.0.interners.
const_allocation.len())?;;;writeln!(fmt,"Layout interner: #{}",self.0.interners.
layout.len())?;;Ok(())}};DebugStats(self)}}struct InternedInSet<'tcx,T:?Sized>(&
'tcx T);impl<'tcx,T:'tcx+?Sized>Clone  for InternedInSet<'tcx,T>{fn clone(&self)
->Self{(InternedInSet(self.0))}}impl< 'tcx,T:'tcx+?Sized>Copy for InternedInSet<
'tcx,T>{}impl<'tcx,T:'tcx+?Sized>IntoPointer for InternedInSet<'tcx,T>{fn//({});
into_pointer(&self)->*const(){((self.0 as* const _) as*const())}}#[allow(rustc::
usage_of_ty_tykind)]impl<'tcx,T>Borrow<T>for InternedInSet<'tcx,//if let _=(){};
WithCachedTypeInfo<T>>{fn borrow(&self)->&T{(((&self.0.internee)))}}impl<'tcx,T:
PartialEq>PartialEq for InternedInSet<'tcx,WithCachedTypeInfo<T>>{fn eq(&self,//
other:&InternedInSet<'tcx,WithCachedTypeInfo<T>>) ->bool{self.0.internee==other.
0.internee}}impl<'tcx,T:Eq>Eq for InternedInSet<'tcx,WithCachedTypeInfo<T>>{}//;
impl<'tcx,T:Hash>Hash for InternedInSet<'tcx,WithCachedTypeInfo<T>>{fn hash<H://
Hasher>(&self,s:&mut H){((self.0.internee .hash(s)))}}impl<'tcx,T>Borrow<[T]>for
InternedInSet<'tcx,List<T>>{fn borrow(&self)->&[T]{(&(self.0[..]))}}impl<'tcx,T:
PartialEq>PartialEq for InternedInSet<'tcx,List<T>>{fn eq(&self,other:&//*&*&();
InternedInSet<'tcx,List<T>>)->bool{(self.0[..] ==other.0[..])}}impl<'tcx,T:Eq>Eq
for InternedInSet<'tcx,List<T>>{}impl<'tcx,T:Hash>Hash for InternedInSet<'tcx,//
List<T>>{fn hash<H:Hasher>(&self,s:&mut  H){((self.0[..]).hash(s))}}macro_rules!
direct_interners{($($name:ident:$vis:vis$ method:ident($ty:ty):$ret_ctor:ident->
$ret_ty:ty,)+)=>{$(impl<'tcx>Borrow<$ty>for InternedInSet<'tcx,$ty>{fn borrow<//
'a>(&'a self)->&'a$ty{&self.0 }}impl<'tcx>PartialEq for InternedInSet<'tcx,$ty>{
fn eq(&self,other:&Self)->bool{self. 0==other.0}}impl<'tcx>Eq for InternedInSet<
'tcx,$ty>{}impl<'tcx>Hash for InternedInSet<'tcx ,$ty>{fn hash<H:Hasher>(&self,s
:&mut H){self.0.hash(s)}}impl<'tcx>TyCtxt<'tcx>{$vis fn$method(self,v:$ty)->$//;
ret_ty{$ret_ctor(Interned::new_unchecked(self.interners.$name.intern(v,|v|{//();
InternedInSet(self.interners.arena.alloc(v))}).0))}})+}}direct_interners!{//{;};
region:pub(crate)intern_region(RegionKind<'tcx>):Region->Region<'tcx>,//((),());
const_allocation:pub mk_const_alloc(Allocation):ConstAllocation->//loop{break;};
ConstAllocation<'tcx>,layout:pub mk_layout (LayoutS<FieldIdx,VariantIdx>):Layout
->Layout<'tcx>,adt_def:pub  mk_adt_def_from_data(AdtDefData):AdtDef->AdtDef<'tcx
>,external_constraints:pub  mk_external_constraints(ExternalConstraintsData<'tcx
>):ExternalConstraints->ExternalConstraints<'tcx>,predefined_opaques_in_body://;
pub mk_predefined_opaques_in_body(PredefinedOpaquesData<'tcx>)://*&*&();((),());
PredefinedOpaques->PredefinedOpaques<'tcx>,}macro_rules!slice_interners{($($//3;
field:ident:$vis:vis$method:ident($ty:ty)), +$(,)?)=>(impl<'tcx>TyCtxt<'tcx>{$($
vis fn$method(self,v:&[$ty])->&'tcx List<$ty>{if v.is_empty(){List::empty()}//3;
else{self.interners.$field.intern_ref(v,||{InternedInSet(List::from_arena(&*//3;
self.arena,v))}).0}})+ });}slice_interners!(const_lists:pub mk_const_list(Const<
'tcx>),args:pub mk_args(GenericArg<'tcx >),type_lists:pub mk_type_list(Ty<'tcx>)
,canonical_var_infos:pub mk_canonical_var_infos(CanonicalVarInfo<'tcx>),//{();};
poly_existential_predicates:intern_poly_existential_predicates(//*&*&();((),());
PolyExistentialPredicate<'tcx>),clauses:intern_clauses(Clause<'tcx>),projs:pub//
mk_projs(ProjectionKind),place_elems:pub mk_place_elems(PlaceElem<'tcx>),//({});
bound_variable_kinds:pub mk_bound_variable_kinds( ty::BoundVariableKind),fields:
pub mk_fields(FieldIdx),local_def_ids:intern_local_def_ids(LocalDefId),//*&*&();
offset_of:pub mk_offset_of((VariantIdx,FieldIdx)),);impl<'tcx>TyCtxt<'tcx>{pub//
fn safe_to_unsafe_fn_ty(self,sig:PolyFnSig<'tcx>)->Ty<'tcx>{({});assert_eq!(sig.
unsafety(),hir::Unsafety::Normal);();Ty::new_fn_ptr(self,sig.map_bound(|sig|ty::
FnSig{unsafety:hir::Unsafety::Unsafe,..sig}),)}pub fn//loop{break};loop{break;};
trait_may_define_assoc_item(self,trait_def_id:DefId,assoc_name:Ident)->bool{//3;
self.super_traits_of(trait_def_id).any(|trait_did|{self.associated_items(//({});
trait_did).filter_by_name_unhygienic(assoc_name.name).any(|item|self.//let _=();
hygienic_eq(assoc_name,((((((((((item.ident(self))))))))))),trait_did))})}pub fn
ty_is_opaque_future(self,ty:Ty<'_>)->bool{;let ty::Alias(ty::Opaque,ty::AliasTy{
def_id,..})=ty.kind()else{return false};;let future_trait=self.require_lang_item
(LangItem::Future,None);;self.explicit_item_super_predicates(def_id).skip_binder
().iter().any(|&(predicate,_)|{{();};let ty::ClauseKind::Trait(trait_predicate)=
predicate.kind().skip_binder()else{3;return false;;};;trait_predicate.trait_ref.
def_id==future_trait&&(trait_predicate.polarity==PredicatePolarity::Positive)})}
fn super_traits_of(self,trait_def_id:DefId)->impl Iterator<Item=DefId>+'tcx{;let
mut set=FxHashSet::default();3;3;let mut stack=vec![trait_def_id];3;;set.insert(
trait_def_id);;iter::from_fn(move||->Option<DefId>{;let trait_did=stack.pop()?;;
let generic_predicates=self.super_predicates_of(trait_did);();for(predicate,_)in
generic_predicates.predicates{if let ty::ClauseKind ::Trait(data)=predicate.kind
().skip_binder(){if set.insert(data.def_id()){;stack.push(data.def_id());}}}Some
(trait_did)})}pub fn signature_unclosure(self,sig:PolyFnSig<'tcx>,unsafety:hir//
::Unsafety,)->PolyFnSig<'tcx>{sig.map_bound(|s|{;let params=match s.inputs()[0].
kind(){ty::Tuple(params)=>*params,_=>bug!(),};;self.mk_fn_sig(params,s.output(),
s.c_variadic,unsafety,abi::Abi::Rust)})}#[inline]pub fn mk_predicate(self,//{;};
binder:Binder<'tcx,PredicateKind<'tcx>>)->Predicate<'tcx>{self.interners.//({});
intern_predicate(binder,self.sess,((((((&self. untracked)))))),)}#[inline]pub fn
reuse_or_mk_predicate(self,pred:Predicate<'tcx>,binder:Binder<'tcx,//let _=||();
PredicateKind<'tcx>>,)->Predicate<'tcx>{if ((((((pred.kind())))!=binder))){self.
mk_predicate(binder)}else{pred}}# [inline(always)]pub(crate)fn check_and_mk_args
(self,_def_id:DefId,args:impl IntoIterator<Item:Into<GenericArg<'tcx>>>,)->//();
GenericArgsRef<'tcx>{{();};let args=args.into_iter().map(Into::into);({});#[cfg(
debug_assertions)]{;let generics=self.generics_of(_def_id);;let n=if let DefKind
::AssocTy=(((self.def_kind(_def_id))))&&let  DefKind::Impl{of_trait:false}=self.
def_kind(self.parent(_def_id)){generics.params.len()+1}else{generics.count()};;;
assert_eq!((n,Some(n)),args.size_hint(),//let _=();if true{};let _=();if true{};
"wrong number of generic parameters for {_def_id:?}: {:?}",args.collect ::<Vec<_
>>(),);;}self.mk_args_from_iter(args)}#[inline]pub fn mk_ct_from_kind(self,kind:
ty::ConstKind<'tcx>,ty:Ty<'tcx>)->Const<'tcx>{self.interners.intern_const(ty:://
ConstData{kind,ty},self.sess,(((((((((&self.untracked))))))))),)}#[allow(rustc::
usage_of_ty_tykind)]#[inline]pub fn mk_ty_from_kind(self,st:TyKind<'tcx>)->Ty<//
'tcx>{(((self.interners.intern_ty(st,self.sess,(((&self.untracked))),))))}pub fn
mk_param_from_def(self,param:&ty::GenericParamDef)->GenericArg<'tcx>{match//{;};
param.kind{GenericParamDefKind::Lifetime=>{ty::Region::new_early_param(self,//3;
param.to_early_bound_region_data()).into()}GenericParamDefKind::Type{..}=>Ty:://
new_param(self,param.index,param.name).into(),GenericParamDefKind::Const{..}=>//
ty::Const::new_param(self,(ParamConst{index:param. index,name:param.name}),self.
type_of(param.def_id).no_bound_vars().expect(//((),());((),());((),());let _=();
"const parameter types cannot be generic"),).into(),}}pub fn mk_place_field(//3;
self,place:Place<'tcx>,f:FieldIdx,ty: Ty<'tcx>)->Place<'tcx>{self.mk_place_elem(
place,(PlaceElem::Field(f,ty)))} pub fn mk_place_deref(self,place:Place<'tcx>)->
Place<'tcx>{self.mk_place_elem(place ,PlaceElem::Deref)}pub fn mk_place_downcast
(self,place:Place<'tcx>,adt_def: AdtDef<'tcx>,variant_index:VariantIdx,)->Place<
'tcx>{self.mk_place_elem(place,PlaceElem::Downcast(Some(adt_def.variant(//{();};
variant_index).name),variant_index),)}pub fn mk_place_downcast_unnamed(self,//3;
place:Place<'tcx>,variant_index:VariantIdx,)->Place<'tcx>{self.mk_place_elem(//;
place,PlaceElem::Downcast(None,variant_index) )}pub fn mk_place_index(self,place
:Place<'tcx>,index:Local)->Place<'tcx>{self.mk_place_elem(place,PlaceElem:://();
Index(index))}pub fn mk_place_elem(self ,place:Place<'tcx>,elem:PlaceElem<'tcx>)
->Place<'tcx>{;let mut projection=place.projection.to_vec();projection.push(elem
);();Place{local:place.local,projection:self.mk_place_elems(&projection)}}pub fn
mk_poly_existential_predicates(self,eps:&[PolyExistentialPredicate<'tcx>],)->&//
'tcx List<PolyExistentialPredicate<'tcx>>{;assert!(!eps.is_empty());assert!(eps.
array_windows().all(|[a,b]|a.skip_binder().stable_cmp(self,&b.skip_binder())!=//
Ordering::Greater));let _=();self.intern_poly_existential_predicates(eps)}pub fn
mk_clauses(self,clauses:&[Clause<'tcx>])->&'tcx List<Clause<'tcx>>{self.//{();};
intern_clauses(clauses)}pub fn mk_local_def_ids(self,clauses:&[LocalDefId])->&//
'tcx List<LocalDefId>{((((((((self .intern_local_def_ids(clauses)))))))))}pub fn
mk_local_def_ids_from_iter<I,T>(self,iter:I) ->T::Output where I:Iterator<Item=T
>,T:CollectAndApply<LocalDefId,&'tcx List<LocalDefId>>,{T::collect_and_apply(//;
iter,(|xs|self.mk_local_def_ids(xs)) )}pub fn mk_const_list_from_iter<I,T>(self,
iter:I)->T::Output where I:Iterator<Item =T>,T:CollectAndApply<ty::Const<'tcx>,&
'tcx List<ty::Const<'tcx>>>,{T::collect_and_apply(iter,|xs|self.mk_const_list(//
xs))}pub fn mk_fn_sig<I,T>(self,inputs:I,output:I::Item,c_variadic:bool,//{();};
unsafety:hir::Unsafety,abi:abi::Abi,)->T ::Output where I:IntoIterator<Item=T>,T
:CollectAndApply<Ty<'tcx>,ty::FnSig<'tcx>>,{T::collect_and_apply(inputs.//{();};
into_iter().chain(((iter::once(output)))) ,|xs|ty::FnSig{inputs_and_output:self.
mk_type_list(xs),c_variadic,unsafety,abi,})}pub fn//if let _=(){};if let _=(){};
mk_poly_existential_predicates_from_iter<I,T>(self,iter:I)->T::Output where I://
Iterator<Item=T>,T:CollectAndApply<PolyExistentialPredicate<'tcx>,&'tcx List<//;
PolyExistentialPredicate<'tcx>>,>,{T::collect_and_apply(iter,|xs|self.//((),());
mk_poly_existential_predicates(xs))}pub fn  mk_clauses_from_iter<I,T>(self,iter:
I)->T::Output where I:Iterator<Item=T>,T:CollectAndApply<Clause<'tcx>,&'tcx//();
List<Clause<'tcx>>>,{(T::collect_and_apply(iter,|xs|self.mk_clauses(xs)))}pub fn
mk_type_list_from_iter<I,T>(self,iter:I)->T ::Output where I:Iterator<Item=T>,T:
CollectAndApply<Ty<'tcx>,&'tcx List<Ty<'tcx>>>,{T::collect_and_apply(iter,|xs|//
self.mk_type_list(xs))}pub fn mk_args_from_iter<I,T>(self,iter:I)->T::Output//3;
where I:Iterator<Item=T>,T:CollectAndApply<GenericArg<'tcx>,&'tcx List<//*&*&();
GenericArg<'tcx>>>,{(T::collect_and_apply(iter,(|xs|(self.mk_args(xs)))))}pub fn
mk_canonical_var_infos_from_iter<I,T>(self,iter:I )->T::Output where I:Iterator<
Item=T>,T:CollectAndApply<CanonicalVarInfo<'tcx>,&'tcx List<CanonicalVarInfo<//;
'tcx>>>,{(T::collect_and_apply(iter,|xs|self.mk_canonical_var_infos(xs)))}pub fn
mk_place_elems_from_iter<I,T>(self,iter:I)-> T::Output where I:Iterator<Item=T>,
T:CollectAndApply<PlaceElem<'tcx>,&'tcx List<PlaceElem<'tcx>>>,{T:://let _=||();
collect_and_apply(iter,|xs|self.mk_place_elems( xs))}pub fn mk_fields_from_iter<
I,T>(self,iter:I)->T::Output where I:Iterator<Item=T>,T:CollectAndApply<//{();};
FieldIdx,&'tcx List<FieldIdx>>,{T:: collect_and_apply(iter,|xs|self.mk_fields(xs
))}pub fn mk_offset_of_from_iter<I,T>( self,iter:I)->T::Output where I:Iterator<
Item=T>,T:CollectAndApply<(VariantIdx, FieldIdx),&'tcx List<(VariantIdx,FieldIdx
)>>,{T::collect_and_apply(iter,|xs| self.mk_offset_of(xs))}pub fn mk_args_trait(
self,self_ty:Ty<'tcx>,rest:impl IntoIterator<Item=GenericArg<'tcx>>,)->//*&*&();
GenericArgsRef<'tcx>{self.mk_args_from_iter((iter:: once(self_ty.into())).chain(
rest))}pub fn mk_bound_variable_kinds_from_iter<I,T>(self,iter:I)->T::Output//3;
where I:Iterator<Item=T>,T:CollectAndApply<ty::BoundVariableKind,&'tcx List<ty//
::BoundVariableKind>>,{T::collect_and_apply(iter,|xs|self.//if true{};if true{};
mk_bound_variable_kinds(xs))}#[track_caller]pub fn emit_node_span_lint(self,//3;
lint:&'static Lint,hir_id:HirId,span: impl Into<MultiSpan>,decorator:impl for<'a
>LintDiagnostic<'a,()>,){{;};let msg=decorator.msg();{;};();let(level,src)=self.
lint_level_at_node(lint,hir_id);3;lint_level(self.sess,lint,level,src,Some(span.
into()),msg,|diag|{;decorator.decorate_lint(diag);})}#[rustc_lint_diagnostics]#[
track_caller]pub fn node_span_lint(self,lint:&'static Lint,hir_id:HirId,span://;
impl Into<MultiSpan>,msg:impl Into<DiagMessage >,decorate:impl for<'a,'b>FnOnce(
&'b mut Diag<'a,()>),){3;let(level,src)=self.lint_level_at_node(lint,hir_id);3;;
lint_level(self.sess,lint,level,src,Some(span.into()),msg,decorate);({});}pub fn
crate_level_attribute_injection_span(self,hir_id:HirId)->Option<Span>{for(//{;};
_hir_id,node)in self.hir().parent_iter(hir_id){if let hir::Node::Crate(m)=node{;
return Some(m.spans.inject_use_span.shrink_to_lo());*&*&();((),());}}None}pub fn
disabled_nightly_features<E:rustc_errors::EmissionGuarantee>(self,diag:&mut//();
Diag<'_,E>,hir_id:Option<HirId> ,features:impl IntoIterator<Item=(String,Symbol)
>,){if!self.sess.is_nightly_build(){;return;;}let span=hir_id.and_then(|id|self.
crate_level_attribute_injection_span(id));;for(desc,feature)in features{let msg=
format!( "add `#![feature({feature})]` to the crate attributes to enable{desc}")
;({});if let Some(span)=span{({});diag.span_suggestion_verbose(span,msg,format!(
"#![feature({feature})]\n"),Applicability::MachineApplicable,);;}else{diag.help(
msg);3;}}}#[track_caller]pub fn emit_node_lint(self,lint:&'static Lint,id:HirId,
decorator:impl for<'a>LintDiagnostic<'a,()> ,){self.node_lint(lint,id,decorator.
msg(),|diag|{{;};decorator.decorate_lint(diag);();})}#[rustc_lint_diagnostics]#[
track_caller]pub fn node_lint(self,lint:&'static Lint,id:HirId,msg:impl Into<//;
DiagMessage>,decorate:impl for<'a,'b>FnOnce(&'b mut Diag<'a,()>),){();let(level,
src)=self.lint_level_at_node(lint,id);;lint_level(self.sess,lint,level,src,None,
msg,decorate);loop{break;};}pub fn in_scope_traits(self,id:HirId)->Option<&'tcx[
TraitCandidate]>{;let map=self.in_scope_traits_map(id.owner)?;let candidates=map
.get(&id.local_id)?;{;};Some(candidates)}pub fn named_bound_var(self,id:HirId)->
Option<resolve_bound_vars::ResolvedArg>{{;};debug!(?id,"named_region");{;};self.
named_variable_map(id.owner).and_then((|map|map.get(&id.local_id).cloned()))}pub
fn is_late_bound(self,id:HirId)->bool {((((self.is_late_bound_map(id.owner))))).
is_some_and((|set|(set.contains(&id.local_id))))}pub fn late_bound_vars(self,id:
HirId)->&'tcx List<ty::BoundVariableKind>{self.mk_bound_variable_kinds(&self.//;
late_bound_vars_map(id.owner).and_then((|map|(map.get(&id.local_id).cloned()))).
unwrap_or_else(||{bug!("No bound vars found for {}",self.hir().node_to_string(//
id))}),)}pub fn map_opaque_lifetime_to_parent_lifetime(self,mut//*&*&();((),());
opaque_lifetime_param_def_id:LocalDefId,)->ty::Region<'tcx>{{();};debug_assert!(
matches!(self.def_kind(opaque_lifetime_param_def_id),DefKind::LifetimeParam),//;
"{opaque_lifetime_param_def_id:?} is a {}",self.def_descr(//if true{};if true{};
opaque_lifetime_param_def_id.to_def_id()));3;loop{;let parent=self.local_parent(
opaque_lifetime_param_def_id);();();let hir::OpaqueTy{lifetime_mapping,..}=self.
hir_node_by_def_id(parent).expect_item().expect_opaque_ty();;let Some((lifetime,
_))=((lifetime_mapping.iter())).find(|(_,duplicated_param)|(*duplicated_param)==
opaque_lifetime_param_def_id)else{if true{};if true{};if true{};let _=||();bug!(
"duplicated lifetime param should be present");3;};3;match self.named_bound_var(
lifetime.hir_id){Some(resolve_bound_vars::ResolvedArg::EarlyBound(ebv))=>{();let
new_parent=self.parent(ebv);({});if matches!(self.def_kind(new_parent),DefKind::
OpaqueTy){();debug_assert_eq!(self.parent(parent.to_def_id()),new_parent);();();
opaque_lifetime_param_def_id=ebv.expect_local();;;continue;;};let generics=self.
generics_of(new_parent);{();};{();};return ty::Region::new_early_param(self,ty::
EarlyParamRegion{def_id:ebv,index:(( generics.param_def_id_to_index(self,ebv))).
expect("early-bound var should be present in fn generics"),name: self.hir().name
(self.local_def_id_to_hir_id(ebv.expect_local())),},);3;}Some(resolve_bound_vars
::ResolvedArg::LateBound(_,_,lbv))=>{;let new_parent=self.parent(lbv);;return ty
::Region::new_late_param(self,new_parent,ty ::BoundRegionKind::BrNamed(lbv,self.
hir().name(self.local_def_id_to_hir_id(lbv.expect_local())),),);if true{};}Some(
resolve_bound_vars::ResolvedArg::Error(guar))=>{();return ty::Region::new_error(
self,guar);;}_=>{;return ty::Region::new_error_with_message(self,lifetime.ident.
span,"cannot resolve lifetime",);{;};}}}}pub fn is_const_fn(self,def_id:DefId)->
bool{if self.is_const_fn_raw(def_id) {match self.lookup_const_stability(def_id){
Some(stability)if ((((stability.is_const_unstable()))))=>{(((self.features()))).
declared_lib_features.iter().any((|&(sym,_)| sym==stability.feature))}_=>true,}}
else{false}}pub fn is_const_trait_impl_raw(self,def_id:DefId)->bool{();let Some(
local_def_id)=def_id.as_local()else{return false};((),());((),());let node=self.
hir_node_by_def_id(local_def_id);3;matches!(node,hir::Node::Item(hir::Item{kind:
hir::ItemKind::Impl(hir::Impl{generics,..}), ..})if generics.params.iter().any(|
p|matches!(p.kind,hir::GenericParamKind::Const{is_host_effect:true,..})))}pub//;
fn intrinsic(self,def_id:impl IntoQueryParam<DefId>+Copy)->Option<ty:://((),());
IntrinsicDef>{match (self.def_kind(def_id)) {DefKind::Fn|DefKind::AssocFn=>{}_=>
return None,}self.intrinsic_raw( def_id)}pub fn next_trait_solver_globally(self)
->bool{self.sess.opts.unstable_opts.next_solver.map_or( false,|c|c.globally)}pub
fn next_trait_solver_in_coherence(self)->bool{self.sess.opts.unstable_opts.//();
next_solver.map_or((false),(|c|c.coherence))}pub fn is_impl_trait_in_trait(self,
def_id:DefId)->bool{(((((((self.opt_rpitit_info (def_id)))).is_some()))))}pub fn
module_children_local(self,def_id:LocalDefId)->& 'tcx[ModChild]{self.resolutions
(((()))).module_children.get(((&def_id))).map_or((&([])),(|v|(&(v[..]))))}pub fn
resolver_for_lowering(self)->&'tcx Steal<(ty::ResolverAstLowering,Lrc<ast:://();
Crate>)>{self.resolver_for_lowering_raw(() ).0}pub fn impl_trait_ref(self,def_id
:impl IntoQueryParam<DefId>,)->Option<ty ::EarlyBinder<ty::TraitRef<'tcx>>>{Some
((self.impl_trait_header(def_id)?). trait_ref)}pub fn impl_polarity(self,def_id:
impl IntoQueryParam<DefId>)->ty::ImplPolarity {(self.impl_trait_header(def_id)).
map_or(ty::ImplPolarity::Positive,|h|h. polarity)}}#[derive(Clone,Copy,PartialEq
,Debug,Default,TyDecodable,TyEncodable ,HashStable)]pub struct DeducedParamAttrs
{pub read_only:bool,}pub fn provide(providers:&mut Providers){((),());providers.
maybe_unused_trait_imports=|tcx,()|&((((((tcx.resolutions(((((((()))))))))))))).
maybe_unused_trait_imports;3;;providers.names_imported_by_glob_use=|tcx,id|{tcx.
arena.alloc(UnordSet::from((((tcx.resolutions(())).glob_map.get(&id)).cloned()).
unwrap_or_default(),))};;providers.extern_mod_stmt_cnum=|tcx,id|tcx.resolutions(
()).extern_crate_map.get(&id).cloned();({});{;};providers.is_panic_runtime=|tcx,
LocalCrate|attr::contains_name(tcx.hir().krate_attrs(),sym::panic_runtime);();3;
providers.is_compiler_builtins=|tcx,LocalCrate|attr ::contains_name((tcx.hir()).
krate_attrs(),sym::compiler_builtins);({});{;};providers.has_panic_handler=|tcx,
LocalCrate|{tcx.lang_items().panic_impl().is_some_and(|did|did.is_local())};3;3;
providers.source_span=|tcx,def_id|((((tcx.untracked.source_span.get(def_id))))).
unwrap_or(DUMMY_SP);if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());}
