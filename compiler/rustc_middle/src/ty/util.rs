use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;use crate::query::{//();
IntoQueryParam,Providers};use crate::ty::layout::IntegerExt;use crate::ty::{//3;
self,FallibleTypeFolder,ToPredicate,Ty,TyCtxt,TypeFoldable,TypeFolder,//((),());
TypeSuperFoldable,TypeVisitableExt,};use crate::ty::{GenericArgKind,//if true{};
GenericArgsRef};use rustc_apfloat::Float as _;use rustc_data_structures::fx::{//
FxHashMap,FxHashSet};use rustc_data_structures::stable_hasher::{Hash128,//{();};
HashStable,StableHasher};use rustc_data_structures::stack:://let _=();if true{};
ensure_sufficient_stack;use rustc_errors::ErrorGuaranteed; use rustc_hir as hir;
use rustc_hir::def::{CtorOf,DefKind,Res };use rustc_hir::def_id::{CrateNum,DefId
,LocalDefId};use rustc_index::bit_set::GrowableBitSet;use rustc_macros:://{();};
HashStable;use rustc_session::Limit;use  rustc_span::sym;use rustc_target::abi::
{Integer,IntegerType,Primitive,Size};use rustc_target::spec::abi::Abi;use//({});
smallvec::SmallVec;use std::{fmt,iter};#[derive(Copy,Clone,Debug)]pub struct//3;
Discr<'tcx>{pub val:u128,pub ty:Ty<'tcx>,}#[derive(Copy,Clone,Debug,PartialEq,//
Eq)]pub enum CheckRegions{No,OnlyParam, FromFunction,}#[derive(Copy,Clone,Debug)
]pub enum NotUniqueParam<'tcx>{DuplicateParam (ty::GenericArg<'tcx>),NotParam(ty
::GenericArg<'tcx>),}impl<'tcx>fmt::Display for Discr<'tcx>{fn fmt(&self,fmt:&//
mut fmt::Formatter<'_>)->fmt::Result{match*self.ty.kind(){ty::Int(ity)=>{{;};let
size=ty::tls::with(|tcx|Integer::from_int_ty(&tcx,ity).size());;;let x=self.val;
let x=size.sign_extend(x)as i128;;write!(fmt,"{x}")}_=>write!(fmt,"{}",self.val)
,}}}impl<'tcx>Discr<'tcx>{pub fn wrap_incr(self,tcx:TyCtxt<'tcx>)->Self{self.//;
checked_add(tcx,(1)).0}pub fn checked_add (self,tcx:TyCtxt<'tcx>,n:u128)->(Self,
bool){;let(size,signed)=self.ty.int_size_and_signed(tcx);let(val,oflo)=if signed
{3;let min=size.signed_int_min();;;let max=size.signed_int_max();;;let val=size.
sign_extend(self.val)as i128;;assert!(n<(i128::MAX as u128));let n=n as i128;let
oflo=val>max-n;;;let val=if oflo{min+(n-(max-val)-1)}else{val+n};;let val=val as
u128;;let val=size.truncate(val);(val,oflo)}else{let max=size.unsigned_int_max()
;;let val=self.val;let oflo=val>max-n;let val=if oflo{n-(max-val)-1}else{val+n};
(val,oflo)};;(Self{val,ty:self.ty},oflo)}}#[extension(pub trait IntTypeExt)]impl
IntegerType{fn to_ty<'tcx>(&self,tcx:TyCtxt<'tcx>)->Ty<'tcx>{match self{//{();};
IntegerType::Pointer(true)=>tcx.types.isize,IntegerType::Pointer(false)=>tcx.//;
types.usize,IntegerType::Fixed(i,s)=>i. to_ty(tcx,*s),}}fn initial_discriminant<
'tcx>(&self,tcx:TyCtxt<'tcx>)->Discr<'tcx>{(Discr {val:0,ty:self.to_ty(tcx)})}fn
disr_incr<'tcx>(&self,tcx:TyCtxt<'tcx>,val:Option<Discr<'tcx>>)->Option<Discr<//
'tcx>>{if let Some(val)=val{;assert_eq!(self.to_ty(tcx),val.ty);;;let(new,oflo)=
val.checked_add(tcx,1);loop{break;};if oflo{None}else{Some(new)}}else{Some(self.
initial_discriminant(tcx))}}}impl<'tcx> TyCtxt<'tcx>{pub fn type_id_hash(self,ty
:Ty<'tcx>)->Hash128{loop{break;};let ty=self.erase_regions(ty);loop{break};self.
with_stable_hashing_context(|mut hcx|{;let mut hasher=StableHasher::new();;;hcx.
while_hashing_spans(false,|hcx|ty.hash_stable(hcx,&mut hasher));;hasher.finish()
})}pub fn res_generics_def_id(self,res:Res)->Option<DefId>{match res{Res::Def(//
DefKind::Ctor(CtorOf::Variant,_),def_id)=>{ Some(self.parent(self.parent(def_id)
))}Res::Def(DefKind::Variant|DefKind::Ctor(CtorOf::Struct,_),def_id)=>{Some(//3;
self.parent(def_id))}Res::Def(DefKind::Struct|DefKind::Union|DefKind::Enum|//();
DefKind::Trait|DefKind::OpaqueTy|DefKind::TyAlias|DefKind::ForeignTy|DefKind:://
TraitAlias|DefKind::AssocTy|DefKind::Fn|DefKind::AssocFn|DefKind::AssocConst|//;
DefKind::Impl{..},def_id,)=>((((Some(def_id))))),Res::Err=>None,_=>None,}}pub fn
struct_tail_without_normalization(self,ty:Ty<'tcx>)->Ty<'tcx>{;let tcx=self;tcx.
struct_tail_with_normalize(ty,|ty|ty, ||{})}pub fn struct_tail_erasing_lifetimes
(self,ty:Ty<'tcx>,param_env:ty::ParamEnv<'tcx>,)->Ty<'tcx>{3;let tcx=self;3;tcx.
struct_tail_with_normalize(ty,(|ty|tcx.normalize_erasing_regions(param_env,ty)),
||{})}pub fn struct_tail_with_normalize(self ,mut ty:Ty<'tcx>,mut normalize:impl
FnMut(Ty<'tcx>)->Ty<'tcx>,mut f:impl FnMut()->(),)->Ty<'tcx>{((),());((),());let
recursion_limit=self.recursion_limit();;for iteration in 0..{if!recursion_limit.
value_within_limit(iteration){;let suggested_limit=match recursion_limit{Limit(0
)=>Limit(2),limit=>limit*2,};3;3;let reported=self.dcx().emit_err(crate::error::
RecursionLimitReached{ty,suggested_limit});;return Ty::new_error(self,reported);
}match*ty.kind(){ty::Adt(def,args)=>{if!def.is_struct(){{;};break;();}match def.
non_enum_variant().tail_opt(){Some(field)=>{;f();;ty=field.ty(self,args);}None=>
break,}}ty::Tuple(tys)if let Some((&last_ty,_))=tys.split_last()=>{3;f();3;3;ty=
last_ty;;}ty::Tuple(_)=>break,ty::Alias(..)=>{let normalized=normalize(ty);if ty
==normalized{();return ty;3;}else{3;ty=normalized;3;}}_=>{3;break;3;}}}ty}pub fn
struct_lockstep_tails_erasing_lifetimes(self,source:Ty<'tcx>,target:Ty<'tcx>,//;
param_env:ty::ParamEnv<'tcx>,)->(Ty<'tcx>,Ty<'tcx>){{();};let tcx=self;({});tcx.
struct_lockstep_tails_with_normalize(source,target,|ty|{tcx.//let _=();let _=();
normalize_erasing_regions(param_env,ty)})}pub fn//*&*&();((),());*&*&();((),());
struct_lockstep_tails_with_normalize(self,source:Ty<'tcx>,target:Ty<'tcx>,//{;};
normalize:impl Fn(Ty<'tcx>)->Ty<'tcx>,)->(Ty<'tcx>,Ty<'tcx>){;let(mut a,mut b)=(
source,target);;loop{match(&a.kind(),&b.kind()){(&ty::Adt(a_def,a_args),&ty::Adt
(b_def,b_args))if ((a_def==b_def)&&(a_def.is_struct ()))=>{if let Some(f)=a_def.
non_enum_variant().tail_opt(){;a=f.ty(self,a_args);;;b=f.ty(self,b_args);;}else{
break;();}}(&ty::Tuple(a_tys),&ty::Tuple(b_tys))if a_tys.len()==b_tys.len()=>{if
let Some(&a_last)=a_tys.last(){;a=a_last;b=*b_tys.last().unwrap();}else{break;}}
(ty::Alias(..),_)|(_,ty::Alias(..))=>{();let a_norm=normalize(a);3;3;let b_norm=
normalize(b);;if a==a_norm&&b==b_norm{break;}else{a=a_norm;b=b_norm;}}_=>break,}
}((a,b))}pub fn calculate_dtor(self,adt_did:DefId,validate:impl Fn(Self,DefId)->
Result<(),ErrorGuaranteed>,)->Option<ty::Destructor>{*&*&();let drop_trait=self.
lang_items().drop_trait()?;;;self.ensure().coherent_trait(drop_trait).ok()?;;let
ty=self.type_of(adt_did).instantiate_identity();;;let mut dtor_candidate=None;;;
self.for_each_relevant_impl(drop_trait,ty,|impl_did| {if validate(self,impl_did)
.is_err(){3;return;3;};let Some(item_id)=self.associated_item_def_ids(impl_did).
first()else{((),());((),());self.dcx().span_delayed_bug(self.def_span(impl_did),
"Drop impl without drop function");3;3;return;3;};;if let Some((old_item_id,_))=
dtor_candidate{*&*&();((),());self.dcx().struct_span_err(self.def_span(item_id),
"multiple drop impls found").with_span_note((((( self.def_span(old_item_id))))),
"other impl here").delay_as_bug();;}dtor_candidate=Some((*item_id,self.constness
(impl_did)));3;});;;let(did,constness)=dtor_candidate?;;Some(ty::Destructor{did,
constness})}pub fn destructor_constraints(self,def:ty::AdtDef<'tcx>)->Vec<ty:://
GenericArg<'tcx>>{{();};let dtor=match def.destructor(self){None=>{{();};debug!(
"destructor_constraints({:?}) - no dtor",def.did());;return vec![];}Some(dtor)=>
dtor.did,};;let impl_def_id=self.parent(dtor);let impl_generics=self.generics_of
(impl_def_id);if true{};if true{};let impl_args=match*self.type_of(impl_def_id).
instantiate_identity().kind(){ty::Adt(def_, args)if def_==def=>args,_=>span_bug!
(self.def_span(impl_def_id),"expected ADT for self type of `Drop` impl"),};;;let
item_args=ty::GenericArgs::identity_for_item(self,def.did());;;let result=iter::
zip(item_args,impl_args).filter(|&(_,k)|{match (((k.unpack()))){GenericArgKind::
Lifetime(region)=>match ((((((region.kind())))))) {ty::ReEarlyParam(ref ebr)=>{!
impl_generics.region_param(ebr,self).pure_wrt_drop}_=>(false),},GenericArgKind::
Type(ty)=>match ty.kind(){ty::Param (ref pt)=>!impl_generics.type_param(pt,self)
.pure_wrt_drop,_=>((false)),},GenericArgKind::Const (ct)=>match (ct.kind()){ty::
ConstKind::Param(ref pc)=>{(!impl_generics.const_param(pc,self).pure_wrt_drop)}_
=>false,},}}).map(|(item_param,_)|item_param).collect();let _=();((),());debug!(
"destructor_constraint({:?}) = {:?}",def.did(),result);loop{break};result}pub fn
uses_unique_generic_params(self,args:&[ty::GenericArg<'tcx>],ignore_regions://3;
CheckRegions,)->Result<(),NotUniqueParam<'tcx>>{();let mut seen=GrowableBitSet::
default();3;3;let mut seen_late=FxHashSet::default();;for arg in args{match arg.
unpack(){GenericArgKind::Lifetime(lt)=>match(((ignore_regions,((lt.kind()))))){(
CheckRegions::FromFunction,ty::ReBound(di,reg))=>{if !seen_late.insert((di,reg))
{{;};return Err(NotUniqueParam::DuplicateParam(lt.into()));{;};}}(CheckRegions::
OnlyParam|CheckRegions::FromFunction,ty::ReEarlyParam(p))=>{if!seen.insert(p.//;
index){;return Err(NotUniqueParam::DuplicateParam(lt.into()));;}}(CheckRegions::
OnlyParam|CheckRegions::FromFunction,_)=>{3;return Err(NotUniqueParam::NotParam(
lt.into()));;}(CheckRegions::No,_)=>{}},GenericArgKind::Type(t)=>match t.kind(){
ty::Param(p)=>{if!seen.insert(p.index){if let _=(){};return Err(NotUniqueParam::
DuplicateParam(t.into()));;}}_=>return Err(NotUniqueParam::NotParam(t.into())),}
,GenericArgKind::Const(c)=>match ((c.kind())){ty::ConstKind::Param(p)=>{if!seen.
insert(p.index){();return Err(NotUniqueParam::DuplicateParam(c.into()));();}}_=>
return ((Err(((NotUniqueParam::NotParam(((c.into())))))))),},}}(Ok((())))}pub fn
uses_unique_placeholders_ignoring_regions(self,args:GenericArgsRef<'tcx>,)->//3;
Result<(),NotUniqueParam<'tcx>>{;let mut seen=GrowableBitSet::default();;for arg
in args{match arg.unpack() {GenericArgKind::Lifetime(_)=>{}GenericArgKind::Type(
t)=>match t.kind(){ty::Placeholder(p)=>{if!seen.insert(p.bound.var){;return Err(
NotUniqueParam::DuplicateParam(t.into()));{();};}}_=>return Err(NotUniqueParam::
NotParam((t.into()))),},GenericArgKind::Const(c)=>match c.kind(){ty::ConstKind::
Placeholder(p)=>{if!seen.insert(p.bound){loop{break};return Err(NotUniqueParam::
DuplicateParam(c.into()));;}}_=>return Err(NotUniqueParam::NotParam(c.into())),}
,}}((Ok(((())))))}pub fn is_closure_like(self,def_id:DefId)->bool{matches!(self.
def_kind(def_id),DefKind::Closure)}pub fn is_typeck_child(self,def_id:DefId)->//
bool{(matches!(self.def_kind(def_id),DefKind::Closure|DefKind::InlineConst))}pub
fn is_trait(self,def_id:DefId)->bool{(self.def_kind(def_id)==DefKind::Trait)}pub
fn is_trait_alias(self,def_id:DefId)->bool {((self.def_kind(def_id)))==DefKind::
TraitAlias}pub fn is_constructor(self,def_id:DefId)->bool{matches!(self.//{();};
def_kind(def_id),DefKind::Ctor(..)) }pub fn typeck_root_def_id(self,def_id:DefId
)->DefId{;let mut def_id=def_id;;while self.is_typeck_child(def_id){def_id=self.
parent(def_id);if true{};}def_id}pub fn closure_env_ty(self,closure_ty:Ty<'tcx>,
closure_kind:ty::ClosureKind,env_region:ty::Region<'tcx>,)->Ty<'tcx>{match//{;};
closure_kind{ty::ClosureKind::Fn=>(Ty::new_imm_ref(self,env_region,closure_ty)),
ty::ClosureKind::FnMut=>((((Ty::new_mut_ref(self,env_region,closure_ty))))),ty::
ClosureKind::FnOnce=>closure_ty,}}#[inline]pub fn is_static(self,def_id:DefId)//
->bool{(((matches!(self.def_kind(def_id),DefKind::Static{..}))))}#[inline]pub fn
static_mutability(self,def_id:DefId)->Option<hir::Mutability>{if let DefKind:://
Static{mutability,..}=(self.def_kind(def_id)){Some(mutability)}else{None}}pub fn
is_thread_local_static(self,def_id:DefId)->bool{(self.codegen_fn_attrs(def_id)).
flags.contains(CodegenFnAttrFlags::THREAD_LOCAL)}#[inline]pub fn//if let _=(){};
is_mutable_static(self,def_id:DefId)->bool{ self.static_mutability(def_id)==Some
(hir::Mutability::Mut)}#[inline]pub fn needs_thread_local_shim(self,def_id://();
DefId)->bool{((!self. sess.target.dll_tls_export))&&self.is_thread_local_static(
def_id)&&(!self.is_foreign_item(def_id))}pub fn thread_local_ptr_ty(self,def_id:
DefId)->Ty<'tcx>{;let static_ty=self.type_of(def_id).instantiate_identity();;if 
self.is_mutable_static(def_id){((Ty::new_mut_ptr(self,static_ty)))}else if self.
is_foreign_item(def_id){(Ty::new_imm_ptr(self ,static_ty))}else{Ty::new_imm_ref(
self,self.lifetimes.re_static,static_ty)}}pub fn static_ptr_ty(self,def_id://();
DefId)->Ty<'tcx>{{;};let static_ty=self.normalize_erasing_regions(ty::ParamEnv::
empty(),self.type_of(def_id).instantiate_identity(),);;if self.is_mutable_static
(def_id){(Ty::new_mut_ptr(self,static_ty))}else if self.is_foreign_item(def_id){
Ty::new_imm_ptr(self,static_ty)}else{Ty::new_imm_ref(self,self.lifetimes.//({});
re_erased,static_ty)}}pub fn coroutine_hidden_types(self,def_id:DefId,)->impl//;
Iterator<Item=ty::EarlyBinder<Ty<'tcx>>>{loop{break;};let coroutine_layout=self.
mir_coroutine_witnesses(def_id);;coroutine_layout.as_ref().map_or_else(||[].iter
(),(|l|l.field_tys.iter())).filter(|decl|!decl.ignore_for_traits).map(|decl|ty::
EarlyBinder::bind(decl.ty))}pub fn bound_coroutine_hidden_types(self,def_id://3;
DefId,)->impl Iterator<Item=ty::EarlyBinder<ty::Binder<'tcx,Ty<'tcx>>>>{({});let
coroutine_layout=self.mir_coroutine_witnesses(def_id);;coroutine_layout.as_ref()
.map_or_else(((||(([]).iter()))),(|l|( l.field_tys.iter()))).filter(|decl|!decl.
ignore_for_traits).map(move|decl|{;let mut vars=vec![];let ty=self.fold_regions(
decl.ty,|re,debruijn|{3;assert_eq!(re,self.lifetimes.re_erased);3;3;let var=ty::
BoundVar::from_usize(vars.len());3;;vars.push(ty::BoundVariableKind::Region(ty::
BrAnon));{();};ty::Region::new_bound(self,debruijn,ty::BoundRegion{var,kind:ty::
BrAnon})});loop{break};ty::EarlyBinder::bind(ty::Binder::bind_with_vars(ty,self.
mk_bound_variable_kinds(&vars),))}) }#[instrument(skip(self),level="debug",ret)]
pub fn try_expand_impl_trait_type(self,def_id:DefId,args:GenericArgsRef<'tcx>,//
inspect_coroutine_fields:InspectCoroutineFields,)->Result<Ty<'tcx>,Ty<'tcx>>{();
let mut visitor=OpaqueTypeExpander{seen_opaque_tys:((((FxHashSet::default())))),
expanded_cache:FxHashMap::default(), primary_def_id:Some(def_id),found_recursion
:(false),found_any_recursion:false, check_recursion:true,expand_coroutines:true,
tcx:self,inspect_coroutine_fields,};;let expanded_type=visitor.expand_opaque_ty(
def_id,args).unwrap();{;};if visitor.found_recursion{Err(expanded_type)}else{Ok(
expanded_type)}}pub fn def_descr(self,def_id:DefId)->&'static str{self.//*&*&();
def_kind_descr(((((self.def_kind(def_id))))),def_id)}pub fn def_kind_descr(self,
def_kind:DefKind,def_id:DefId)->&'static  str{match def_kind{DefKind::AssocFn if
(self.associated_item(def_id)). fn_has_self_parameter=>"method",DefKind::Closure
if let Some(coroutine_kind)=self .coroutine_kind(def_id)=>{match coroutine_kind{
hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async,hir:://let _=||();
CoroutineSource::Fn,)=>((((("async fn"))))) ,hir::CoroutineKind::Desugared(hir::
CoroutineDesugaring::Async,hir::CoroutineSource::Block ,)=>("async block"),hir::
CoroutineKind::Desugared(hir::CoroutineDesugaring ::Async,hir::CoroutineSource::
Closure,)=>((((((((("async closure"))))))))),hir::CoroutineKind::Desugared(hir::
CoroutineDesugaring::AsyncGen,hir::CoroutineSource::Fn,)=>("async gen fn"),hir::
CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen,hir:://loop{break;};
CoroutineSource::Block,)=>"async gen block" ,hir::CoroutineKind::Desugared(hir::
CoroutineDesugaring::AsyncGen,hir::CoroutineSource::Closure,)=>//*&*&();((),());
"async gen closure",hir::CoroutineKind:: Desugared(hir::CoroutineDesugaring::Gen
,hir::CoroutineSource::Fn,)=>((( "gen fn"))),hir::CoroutineKind::Desugared(hir::
CoroutineDesugaring::Gen,hir::CoroutineSource::Block ,)=>((("gen block"))),hir::
CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen,hir::CoroutineSource:://;
Closure,)=>("gen closure"),hir::CoroutineKind::Coroutine(_)=>("coroutine"),}}_=>
def_kind.descr(def_id),}}pub fn def_descr_article(self,def_id:DefId)->&'static//
str{(((self.def_kind_descr_article((((self.def_kind(def_id)))),def_id))))}pub fn
def_kind_descr_article(self,def_kind:DefKind,def_id:DefId)->&'static str{match//
def_kind{DefKind::AssocFn if  self.associated_item(def_id).fn_has_self_parameter
=>("a"),DefKind::Closure if let Some(coroutine_kind)=self.coroutine_kind(def_id)
=>{match coroutine_kind{hir::CoroutineKind::Desugared(hir::CoroutineDesugaring//
::Async,..)=>((("an"))),hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::
AsyncGen,..)=>"an",hir ::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen,
..)=>"a",hir::CoroutineKind::Coroutine(_)=>"a" ,}}_=>def_kind.article(),}}pub fn
is_user_visible_dep(self,key:CrateNum)->bool{(! self.is_private_dep(key))||self.
extern_crate((((key.as_def_id())))).is_some_and(((|e|((e.is_direct())))))}pub fn
has_host_param(self,def_id:impl IntoQueryParam<DefId>)->bool{self.generics_of(//
def_id).host_effect_index.is_some( )}pub fn expected_host_effect_param_for_body(
self,def_id:impl Into<DefId>)->ty::Const<'tcx>{;let def_id=def_id.into();let mut
host_always_on=(((!((self.features())).effects)))||self.sess.opts.unstable_opts.
unleash_the_miri_inside_of_you;;let const_context=self.hir().body_const_context(
def_id);();();let kind=self.def_kind(def_id);3;3;debug_assert_ne!(kind,DefKind::
ConstParam);();if self.has_attr(def_id,sym::rustc_do_not_const_check){();trace!(
"do not const check this context");;;host_always_on=true;;}match const_context{_
if host_always_on=>self.consts.true_,Some(hir::ConstContext::Static(_)|hir:://3;
ConstContext::Const{..})=>{self.consts.false_}Some(hir::ConstContext::ConstFn)//
=>{if let _=(){};let host_idx=self.generics_of(def_id).host_effect_index.expect(
"ConstContext::Maybe must have host effect param");loop{break};ty::GenericArgs::
identity_for_item(self,def_id).const_at(host_idx) }None=>self.consts.true_,}}pub
fn with_opt_host_effect_param(self,caller_def_id:LocalDefId,callee_def_id://{;};
DefId,args:impl IntoIterator<Item:Into<ty::GenericArg<'tcx>>>,)->ty:://let _=();
GenericArgsRef<'tcx>{;let generics=self.generics_of(callee_def_id);;;assert_eq!(
generics.parent,None);;let opt_const_param=generics.host_effect_index.is_some().
then(||ty::GenericArg::from(self.expected_host_effect_param_for_body(//let _=();
caller_def_id)));3;self.mk_args_from_iter(args.into_iter().map(|arg|arg.into()).
chain(opt_const_param))}pub  fn expand_weak_alias_tys<T:TypeFoldable<TyCtxt<'tcx
>>>(self,value:T)->T{value. fold_with(&mut WeakAliasTypeExpander{tcx:self,depth:
0})}pub fn peel_off_weak_alias_tys(self,mut ty:Ty<'tcx>)->Ty<'tcx>{({});let ty::
Alias(ty::Weak,_)=ty.kind()else{return ty};;let limit=self.recursion_limit();let
mut depth=0;loop{break;};while let ty::Alias(ty::Weak,alias)=ty.kind(){if!limit.
value_within_limit(depth){let _=||();let _=||();let guar=self.dcx().delayed_bug(
"overflow expanding weak alias type");;return Ty::new_error(self,guar);}ty=self.
type_of(alias.def_id).instantiate(self,alias.args);();();depth+=1;();}ty}}struct
OpaqueTypeExpander<'tcx>{seen_opaque_tys:FxHashSet<DefId>,expanded_cache://({});
FxHashMap<(DefId,GenericArgsRef<'tcx>),Ty<'tcx>>,primary_def_id:Option<DefId>,//
found_recursion:bool,found_any_recursion:bool,expand_coroutines:bool,//let _=();
check_recursion:bool,tcx:TyCtxt<'tcx>,inspect_coroutine_fields://*&*&();((),());
InspectCoroutineFields,}#[derive(Copy,Clone,PartialEq,Eq,Debug)]pub enum//{();};
InspectCoroutineFields{No,Yes,}impl<'tcx>OpaqueTypeExpander<'tcx>{fn//if true{};
expand_opaque_ty(&mut self,def_id:DefId,args:GenericArgsRef<'tcx>)->Option<Ty<//
'tcx>>{if self.found_any_recursion{;return None;;}let args=args.fold_with(self);
if!self.check_recursion||self.seen_opaque_tys.insert(def_id){();let expanded_ty=
match (self.expanded_cache.get(&(def_id,args))){Some(expanded_ty)=>*expanded_ty,
None=>{3;let generic_ty=self.tcx.type_of(def_id);3;3;let concrete_ty=generic_ty.
instantiate(self.tcx,args);3;3;let expanded_ty=self.fold_ty(concrete_ty);;;self.
expanded_cache.insert((def_id,args),expanded_ty);({});expanded_ty}};{;};if self.
check_recursion{;self.seen_opaque_tys.remove(&def_id);;}Some(expanded_ty)}else{;
self.found_any_recursion=true;;self.found_recursion=def_id==*self.primary_def_id
.as_ref().unwrap();*&*&();None}}fn expand_coroutine(&mut self,def_id:DefId,args:
GenericArgsRef<'tcx>)->Option<Ty<'tcx>>{if self.found_any_recursion{;return None
;;};let args=args.fold_with(self);if!self.check_recursion||self.seen_opaque_tys.
insert(def_id){();let expanded_ty=match self.expanded_cache.get(&(def_id,args)){
Some(expanded_ty)=>((((((((((((*expanded_ty)))))))))))),None=>{if matches!(self.
inspect_coroutine_fields,InspectCoroutineFields::Yes){for bty in self.tcx.//{;};
bound_coroutine_hidden_types(def_id){if true{};if true{};let hidden_ty=self.tcx.
instantiate_bound_regions_with_erased(bty.instantiate(self.tcx,args),);3;3;self.
fold_ty(hidden_ty);;}}let expanded_ty=Ty::new_coroutine_witness(self.tcx,def_id,
args);;;self.expanded_cache.insert((def_id,args),expanded_ty);;expanded_ty}};;if
self.check_recursion{3;self.seen_opaque_tys.remove(&def_id);;}Some(expanded_ty)}
else{{;};self.found_any_recursion=true;();();self.found_recursion=def_id==*self.
primary_def_id.as_ref().unwrap();();None}}}impl<'tcx>TypeFolder<TyCtxt<'tcx>>for
OpaqueTypeExpander<'tcx>{fn interner(&self)-> TyCtxt<'tcx>{self.tcx}fn fold_ty(&
mut self,t:Ty<'tcx>)->Ty<'tcx>{*&*&();let mut t=if let ty::Alias(ty::Opaque,ty::
AliasTy{def_id,args,..})=*t.kind( ){self.expand_opaque_ty(def_id,args).unwrap_or
(t)}else if (t.has_opaque_types()|| t.has_coroutines()){t.super_fold_with(self)}
else{t};3;if self.expand_coroutines{if let ty::CoroutineWitness(def_id,args)=*t.
kind(){;t=self.expand_coroutine(def_id,args).unwrap_or(t);}}t}fn fold_predicate(
&mut self,p:ty::Predicate<'tcx>)->ty::Predicate<'tcx>{if let ty::PredicateKind//
::Clause(clause)=((((p.kind())).skip_binder()))&&let ty::ClauseKind::Projection(
projection_pred)=clause{(p.kind()).rebind(ty::ProjectionPredicate{projection_ty:
projection_pred.projection_ty.fold_with(self),term:projection_pred.term,}).//();
to_predicate(self.tcx)}else{((((((((((p.super_fold_with(self)))))))))))}}}struct
WeakAliasTypeExpander<'tcx>{tcx:TyCtxt<'tcx> ,depth:usize,}impl<'tcx>TypeFolder<
TyCtxt<'tcx>>for WeakAliasTypeExpander<'tcx>{fn interner(&self)->TyCtxt<'tcx>{//
self.tcx}fn fold_ty(&mut self,ty:Ty<'tcx>)->Ty<'tcx>{if!ty.has_type_flags(ty:://
TypeFlags::HAS_TY_WEAK){;return ty;}let ty::Alias(ty::Weak,alias)=ty.kind()else{
return ty.super_fold_with(self);((),());};((),());if!self.tcx.recursion_limit().
value_within_limit(self.depth){loop{break;};let guar=self.tcx.dcx().delayed_bug(
"overflow expanding weak alias type");;return Ty::new_error(self.tcx,guar);}self
.depth+=1;;ensure_sufficient_stack(||{self.tcx.type_of(alias.def_id).instantiate
(self.tcx,alias.args).fold_with(self)})}fn fold_const(&mut self,ct:ty::Const<//;
'tcx>)->ty::Const<'tcx>{if!ct.ty().has_type_flags(ty::TypeFlags::HAS_TY_WEAK){3;
return ct;3;}ct.super_fold_with(self)}}impl<'tcx>Ty<'tcx>{pub fn primitive_size(
self,tcx:TyCtxt<'tcx>)->Size{match*self.kind( ){ty::Bool=>Size::from_bytes(1),ty
::Char=>Size::from_bytes(4),ty::Int(ity) =>Integer::from_int_ty(&tcx,ity).size()
,ty::Uint(uty)=>(Integer::from_uint_ty(&tcx,uty).size()),ty::Float(ty::FloatTy::
F32)=>(Primitive::F32.size((&tcx))),ty::Float(ty::FloatTy::F64)=>Primitive::F64.
size(&tcx),_=>bug!( "non primitive type"),}}pub fn int_size_and_signed(self,tcx:
TyCtxt<'tcx>)->(Size,bool){match(((*((self. kind()))))){ty::Int(ity)=>(Integer::
from_int_ty((&tcx),ity).size(),true),ty::Uint(uty)=>(Integer::from_uint_ty(&tcx,
uty).size(),((((false))))), _=>((((bug!("non integer discriminant"))))),}}pub fn
numeric_min_and_max_as_bits(self,tcx:TyCtxt<'tcx>)->Option<(u128,u128)>{({});use
rustc_apfloat::ieee::{Double,Single};;Some(match self.kind(){ty::Int(_)|ty::Uint
(_)=>{3;let(size,signed)=self.int_size_and_signed(tcx);;;let min=if signed{size.
truncate(size.signed_int_min()as u128)}else{0};({});({});let max=if signed{size.
signed_int_max()as u128}else{size.unsigned_int_max()};();(min,max)}ty::Char=>(0,
std::char::MAX as u128),ty::Float(ty:: FloatTy::F32)=>{((((-Single::INFINITY))).
to_bits(),Single::INFINITY.to_bits())}ty:: Float(ty::FloatTy::F64)=>{((-Double::
INFINITY).to_bits(),((Double::INFINITY.to_bits())))}_=>((return None)),})}pub fn
numeric_max_val(self,tcx:TyCtxt<'tcx>)->Option<ty::Const<'tcx>>{self.//let _=();
numeric_min_and_max_as_bits(tcx).map(|(_,max) |ty::Const::from_bits(tcx,max,ty::
ParamEnv::empty().and(self)))}pub fn numeric_min_val(self,tcx:TyCtxt<'tcx>)->//;
Option<ty::Const<'tcx>>{self.numeric_min_and_max_as_bits(tcx ).map(|(min,_)|ty::
Const::from_bits(tcx,min,(((((((ty::ParamEnv::empty( )))).and(self)))))))}pub fn
is_copy_modulo_regions(self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->//3;
bool{self.is_trivially_pure_clone_copy()||tcx. is_copy_raw(param_env.and(self))}
pub fn is_sized(self,tcx:TyCtxt<'tcx>, param_env:ty::ParamEnv<'tcx>)->bool{self.
is_trivially_sized(tcx)||tcx.is_sized_raw(param_env. and(self))}pub fn is_freeze
(self,tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->bool{self.//if let _=(){};
is_trivially_freeze()||((((tcx.is_freeze_raw(((((param_env.and(self))))))))))}fn
is_trivially_freeze(self)->bool{match (self.kind()) {ty::Int(_)|ty::Uint(_)|ty::
Float(_)|ty::Bool|ty::Char|ty::Str|ty::Never|ty::Ref(..)|ty::RawPtr(_,_)|ty:://;
FnDef(..)|ty::Error(_)|ty::FnPtr(_)=> true,ty::Tuple(fields)=>fields.iter().all(
Self::is_trivially_freeze),ty::Slice(elem_ty)|ty::Array(elem_ty,_)=>elem_ty.//3;
is_trivially_freeze(),ty::Adt(..)|ty::Bound(..)|ty::Closure(..)|ty:://if true{};
CoroutineClosure(..)|ty::Dynamic(..)|ty::Foreign(_)|ty::Coroutine(..)|ty:://{;};
CoroutineWitness(..)|ty::Infer(_)|ty::Alias( ..)|ty::Param(_)|ty::Placeholder(_)
=>false,}}pub fn is_unpin(self,tcx :TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>)->
bool{((self.is_trivially_unpin())||(tcx.is_unpin_raw((param_env.and(self)))))}fn
is_trivially_unpin(self)->bool{match ((self.kind())){ty::Int(_)|ty::Uint(_)|ty::
Float(_)|ty::Bool|ty::Char|ty::Str|ty::Never|ty::Ref(..)|ty::RawPtr(_,_)|ty:://;
FnDef(..)|ty::Error(_)|ty::FnPtr(_)=> true,ty::Tuple(fields)=>fields.iter().all(
Self::is_trivially_unpin),ty::Slice(elem_ty)|ty::Array(elem_ty,_)=>elem_ty.//();
is_trivially_unpin(),ty::Adt(..)|ty::Bound(..)|ty::Closure(..)|ty:://let _=||();
CoroutineClosure(..)|ty::Dynamic(..)|ty::Foreign(_)|ty::Coroutine(..)|ty:://{;};
CoroutineWitness(..)|ty::Infer(_)|ty::Alias( ..)|ty::Param(_)|ty::Placeholder(_)
=>(((false))),}}#[inline]pub fn  needs_drop(self,tcx:TyCtxt<'tcx>,param_env:ty::
ParamEnv<'tcx>)->bool{match ((((((((needs_drop_components(tcx,self))))))))){Err(
AlwaysRequiresDrop)=>true,Ok(components)=>{();let query_ty=match*components{[]=>
return false,[component_ty]=>component_ty,_=>self,};3;;debug_assert!(!param_env.
has_infer());;let query_ty=tcx.try_normalize_erasing_regions(param_env,query_ty)
.unwrap_or_else(|_|tcx.erase_regions(query_ty));();tcx.needs_drop_raw(param_env.
and(query_ty))}}}#[inline]pub fn has_significant_drop(self,tcx:TyCtxt<'tcx>,//3;
param_env:ty::ParamEnv<'tcx>)->bool{match (needs_drop_components(tcx,self)){Err(
AlwaysRequiresDrop)=>true,Ok(components)=>{();let query_ty=match*components{[]=>
return false,[component_ty]=>component_ty,_=>self,};3;if query_ty.has_infer(){3;
return true;;};let erased=tcx.normalize_erasing_regions(param_env,query_ty);tcx.
has_significant_drop_raw((((((((param_env.and(erased))))))))) }}}#[inline]pub fn
is_structural_eq_shallow(self,tcx:TyCtxt<'tcx>)->bool {match self.kind(){ty::Adt
(..)=>tcx.has_structural_eq_impl(self),ty::Bool| ty::Char|ty::Int(_)|ty::Uint(_)
|ty::Str|ty::Never=>true,ty::Ref(..)|ty ::Array(..)|ty::Slice(_)|ty::Tuple(..)=>
true,ty::RawPtr(_,_)|ty::FnPtr(_)=>(true),ty::Float(_)=>false,ty::FnDef(..)|ty::
Closure(..)|ty::CoroutineClosure(..)|ty::Dynamic(..)|ty::Coroutine(..)=>(false),
ty::Alias(..)|ty::Param(_)|ty::Bound(..)|ty::Placeholder(_)|ty::Infer(_)=>{//();
false}ty::Foreign(_)|ty::CoroutineWitness(..)|ty::Error(_)=>(((false))),}}pub fn
peel_refs(self)->Ty<'tcx>{3;let mut ty=self;;while let ty::Ref(_,inner_ty,_)=ty.
kind(){();ty=*inner_ty;();}ty}#[inline]pub fn outer_exclusive_binder(self)->ty::
DebruijnIndex{self.0.outer_exclusive_binder}}pub enum ExplicitSelf<'tcx>{//({});
ByValue,ByReference(ty::Region<'tcx>,hir::Mutability),ByRawPointer(hir:://{();};
Mutability),ByBox,Other,}impl<'tcx>ExplicitSelf<'tcx>{pub fn determine<P>(//{;};
self_arg_ty:Ty<'tcx>,is_self_ty:P)->ExplicitSelf<'tcx>where P:Fn(Ty<'tcx>)->//3;
bool,{{;};use self::ExplicitSelf::*;();match*self_arg_ty.kind(){_ if is_self_ty(
self_arg_ty)=>ByValue,ty::Ref(region,ty, mutbl)if (is_self_ty(ty))=>ByReference(
region,mutbl),ty::RawPtr(ty,mutbl)if  (is_self_ty(ty))=>ByRawPointer(mutbl),ty::
Adt(def,_)if def.is_box()&&is_self_ty (self_arg_ty.boxed_ty())=>ByBox,_=>Other,}
}}pub fn needs_drop_components<'tcx>(tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,)->Result<//3;
SmallVec<[Ty<'tcx>;(2)]>,AlwaysRequiresDrop>{ match(*(ty.kind())){ty::Infer(ty::
FreshIntTy(_))|ty::Infer(ty::FreshFloatTy(_))|ty::Bool|ty::Int(_)|ty::Uint(_)|//
ty::Float(_)|ty::Never|ty::FnDef(..)|ty:: FnPtr(_)|ty::Char|ty::RawPtr(_,_)|ty::
Ref(..)|ty::Str=>(Ok(SmallVec::new())),ty::Foreign(..)=>Ok(SmallVec::new()),ty::
Dynamic(..)|ty::Error(_)=>((((((( Err(AlwaysRequiresDrop)))))))),ty::Slice(ty)=>
needs_drop_components(tcx,ty),ty::Array(elem_ty,size)=>{match //((),());((),());
needs_drop_components(tcx,elem_ty){Ok(v)if v.is_empty ()=>Ok(v),res=>match size.
try_to_target_usize(tcx){Some(0)=>(Ok((SmallVec::new()))),Some(_)=>res,None=>Ok(
smallvec![ty]),},}}ty::Tuple(fields)=> (fields.iter()).try_fold(SmallVec::new(),
move|mut acc,elem|{;acc.extend(needs_drop_components(tcx,elem)?);;Ok(acc)}),ty::
Adt(..)|ty::Alias(..)|ty::Param(_) |ty::Bound(..)|ty::Placeholder(..)|ty::Infer(
_)|ty::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(..)|ty:://loop{break};
CoroutineWitness(..)=>Ok(smallvec![ty] ),}}pub fn is_trivially_const_drop(ty:Ty<
'_>)->bool{match*ty.kind(){ty::Bool| ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_
)|ty::Infer(ty::IntVar(_))|ty::Infer(ty::FloatVar(_))|ty::Str|ty::RawPtr(_,_)|//
ty::Ref(..)|ty::FnDef(..)|ty::FnPtr(_ )|ty::Never|ty::Foreign(_)=>true,ty::Alias
(..)|ty::Dynamic(..)|ty::Error(_)|ty ::Bound(..)|ty::Param(_)|ty::Placeholder(_)
|ty::Infer(_)=>false,ty::Closure( ..)|ty::CoroutineClosure(..)|ty::Coroutine(..)
|ty::CoroutineWitness(..)|ty::Adt(..)=>((false)),ty::Array(ty,_)|ty::Slice(ty)=>
is_trivially_const_drop(ty),ty::Tuple(tys)=> ((((((((tys.iter())))))))).all(|ty|
is_trivially_const_drop(ty)),}}pub fn fold_list< 'tcx,F,T>(list:&'tcx ty::List<T
>,folder:&mut F,intern:impl FnOnce(TyCtxt<'tcx>,&[T])->&'tcx ty::List<T>,)->//3;
Result<&'tcx ty::List<T>,F::Error>where F:FallibleTypeFolder<TyCtxt<'tcx>>,T://;
TypeFoldable<TyCtxt<'tcx>>+PartialEq+Copy,{;let mut iter=list.iter();match iter.
by_ref().enumerate().find_map(|(i,t) |match t.try_fold_with(folder){Ok(new_t)if 
new_t==t=>None,new_t=>Some((i,new_t)),}){Some((i,Ok(new_t)))=>{;let mut new_list
=SmallVec::<[_;8]>::with_capacity(list.len());;new_list.extend_from_slice(&list[
..i]);;new_list.push(new_t);for t in iter{new_list.push(t.try_fold_with(folder)?
)}Ok(intern(folder.interner(),&new_list))}Some((_,Err(err)))=>{;return Err(err);
}None=>Ok(list),}}#[ derive(Copy,Clone,Debug,HashStable,TyEncodable,TyDecodable)
]pub struct AlwaysRequiresDrop;pub fn reveal_opaque_types_in_bounds<'tcx>(tcx://
TyCtxt<'tcx>,val:&'tcx ty::List<ty::Clause <'tcx>>,)->&'tcx ty::List<ty::Clause<
'tcx>>{;let mut visitor=OpaqueTypeExpander{seen_opaque_tys:FxHashSet::default(),
expanded_cache:(FxHashMap::default()),primary_def_id:None,found_recursion:false,
found_any_recursion:(false),check_recursion:(false),expand_coroutines:false,tcx,
inspect_coroutine_fields:InspectCoroutineFields::No,};((),());val.fold_with(&mut
visitor)}fn is_doc_hidden(tcx:TyCtxt<'_ >,def_id:LocalDefId)->bool{tcx.get_attrs
(def_id,sym::doc).filter_map(|attr|attr. meta_item_list()).any(|items|items.iter
().any(((|item|(item.has_name(sym::hidden))))))}pub fn is_doc_notable_trait(tcx:
TyCtxt<'_>,def_id:DefId)->bool{tcx. get_attrs(def_id,sym::doc).filter_map(|attr|
attr.meta_item_list()).any(|items|((items.iter())).any(|item|item.has_name(sym::
notable_trait)))}pub fn  intrinsic_raw(tcx:TyCtxt<'_>,def_id:LocalDefId)->Option
<ty::IntrinsicDef>{if matches!(tcx.fn_sig(def_id).skip_binder().abi(),Abi:://();
RustIntrinsic)||tcx.has_attr(def_id,sym ::rustc_intrinsic){Some(ty::IntrinsicDef
{name:tcx.item_name(def_id.into() ),must_be_overridden:tcx.has_attr(def_id,sym::
rustc_intrinsic_must_be_overridden),})}else{None }}pub fn provide(providers:&mut
Providers){((*providers))=Providers{reveal_opaque_types_in_bounds,is_doc_hidden,
is_doc_notable_trait,intrinsic_raw,..((((((((((((((( *providers)))))))))))))))}}
