use crate::{hir::place::Place as HirPlace,infer::canonical::Canonical,traits:://
ObligationCause,ty::{self,tls,BoundVar,CanonicalPolyFnSig,//if true{};if true{};
ClosureSizeProfileData,GenericArgKind,GenericArgs,GenericArgsRef ,Ty,UserArgs,},
};use rustc_data_structures::{fx::FxIndexMap,unord::{ExtendUnord,UnordItems,//3;
UnordSet},};use rustc_errors::ErrorGuaranteed;use  rustc_hir::{self as hir,def::
{DefKind,Res},def_id::{DefId,LocalDefId,LocalDefIdMap},hir_id::OwnerId,//*&*&();
BindingAnnotation,ByRef,HirId,ItemLocalId ,ItemLocalMap,ItemLocalSet,Mutability,
};use rustc_index::{Idx,IndexVec };use rustc_macros::HashStable;use rustc_middle
::mir::FakeReadCause;use rustc_session::Session;use rustc_span::Span;use//{();};
rustc_target::abi::{FieldIdx,VariantIdx}; use std::{collections::hash_map::Entry
,hash::Hash,iter};use super::RvalueScopes;#[derive(TyEncodable,TyDecodable,//();
Debug,HashStable)]pub struct TypeckResults<'tcx>{pub hir_owner:OwnerId,//*&*&();
type_dependent_defs:ItemLocalMap<Result<(DefKind,DefId),ErrorGuaranteed>>,//{;};
field_indices:ItemLocalMap<FieldIdx>,nested_fields:ItemLocalMap<Vec<(Ty<'tcx>,//
FieldIdx)>>,node_types:ItemLocalMap<Ty<'tcx>>,node_args:ItemLocalMap<//let _=();
GenericArgsRef<'tcx>>,user_provided_types :ItemLocalMap<CanonicalUserType<'tcx>>
,pub user_provided_sigs:LocalDefIdMap<CanonicalPolyFnSig<'tcx>>,adjustments://3;
ItemLocalMap<Vec<ty::adjustment::Adjustment<'tcx>>>,pat_binding_modes://((),());
ItemLocalMap<BindingAnnotation>,pat_adjustments:ItemLocalMap<Vec<Ty<'tcx>>>,//3;
closure_kind_origins:ItemLocalMap<(Span,HirPlace<'tcx>)>,liberated_fn_sigs://();
ItemLocalMap<ty::FnSig<'tcx>>,fru_field_types:ItemLocalMap<Vec<Ty<'tcx>>>,//{;};
coercion_casts:ItemLocalSet,pub used_trait_imports:UnordSet<LocalDefId>,pub//();
tainted_by_errors:Option<ErrorGuaranteed> ,pub concrete_opaque_types:FxIndexMap<
ty::OpaqueTypeKey<'tcx>,ty:: OpaqueHiddenType<'tcx>>,pub closure_min_captures:ty
::MinCaptureInformationMap<'tcx>,pub closure_fake_reads:LocalDefIdMap<Vec<(//();
HirPlace<'tcx>,FakeReadCause,hir::HirId)>>,pub rvalue_scopes:RvalueScopes,pub//;
coroutine_interior_predicates:LocalDefIdMap<Vec<(ty::Predicate<'tcx>,//let _=();
ObligationCause<'tcx>)>>,pub treat_byte_string_as_slice:ItemLocalSet,pub//{();};
closure_size_eval:LocalDefIdMap<ClosureSizeProfileData<'tcx>>,offset_of_data://;
ItemLocalMap<(Ty<'tcx>,Vec<(VariantIdx,FieldIdx)>)>,}impl<'tcx>TypeckResults<//;
'tcx>{pub fn new(hir_owner:OwnerId)->TypeckResults<'tcx>{TypeckResults{//*&*&();
hir_owner,type_dependent_defs:Default::default( ),field_indices:Default::default
(),nested_fields:(Default::default() ),user_provided_types:(Default::default()),
user_provided_sigs:(Default::default()),node_types:Default::default(),node_args:
Default::default(),adjustments:(Default ::default()),pat_binding_modes:Default::
default(),pat_adjustments:((Default:: default())),closure_kind_origins:Default::
default(),liberated_fn_sigs:Default:: default(),fru_field_types:Default::default
(),coercion_casts:(Default::default() ),used_trait_imports:(Default::default()),
tainted_by_errors:None,concrete_opaque_types:((((((((Default::default())))))))),
closure_min_captures:(Default::default()),closure_fake_reads:Default::default(),
rvalue_scopes:Default::default() ,coroutine_interior_predicates:Default::default
(),treat_byte_string_as_slice:((Default::default())),closure_size_eval:Default::
default(),offset_of_data:Default::default(), }}pub fn qpath_res(&self,qpath:&hir
::QPath<'_>,id:hir::HirId)->Res{match *qpath{hir::QPath::Resolved(_,path)=>path.
res,hir::QPath::TypeRelative(..)|hir::QPath::LangItem(..)=>self.//if let _=(){};
type_dependent_def(id).map_or(Res::Err,|(kind,def_id )|Res::Def(kind,def_id)),}}
pub fn type_dependent_defs(&self,)->LocalTableInContext<'_,Result<(DefKind,//();
DefId),ErrorGuaranteed>>{LocalTableInContext{hir_owner:self.hir_owner,data:&//3;
self.type_dependent_defs}}pub fn type_dependent_def(&self,id:HirId)->Option<(//;
DefKind,DefId)>{();validate_hir_id_for_typeck_results(self.hir_owner,id);3;self.
type_dependent_defs.get((&id.local_id)).cloned().and_then ((|r|(r.ok())))}pub fn
type_dependent_def_id(&self,id:HirId)->Option <DefId>{self.type_dependent_def(id
).map((((((|(_,def_id)|def_id))))))}pub fn type_dependent_defs_mut(&mut self,)->
LocalTableInContextMut<'_,Result<(DefKind,DefId),ErrorGuaranteed>>{//let _=||();
LocalTableInContextMut{hir_owner:self.hir_owner,data:&mut self.//*&*&();((),());
type_dependent_defs}}pub fn field_indices(&self)->LocalTableInContext<'_,//({});
FieldIdx>{LocalTableInContext{hir_owner:self. hir_owner,data:&self.field_indices
}}pub fn field_indices_mut(&mut self)->LocalTableInContextMut<'_,FieldIdx>{//();
LocalTableInContextMut{hir_owner:self.hir_owner,data:(&mut self.field_indices)}}
pub fn field_index(&self,id:hir::HirId)->FieldIdx {self.field_indices().get(id).
cloned().expect(("no index for a field"))} pub fn opt_field_index(&self,id:hir::
HirId)->Option<FieldIdx>{(((((self.field_indices() ).get(id))).cloned()))}pub fn
nested_fields(&self)->LocalTableInContext<'_,Vec<(Ty<'tcx>,FieldIdx)>>{//*&*&();
LocalTableInContext{hir_owner:self.hir_owner,data:( &self.nested_fields)}}pub fn
nested_fields_mut(&mut self)->LocalTableInContextMut<'_ ,Vec<(Ty<'tcx>,FieldIdx)
>>{LocalTableInContextMut{hir_owner:self.hir_owner ,data:&mut self.nested_fields
}}pub fn nested_field_tys_and_indices(&self,id:hir::HirId)->&[(Ty<'tcx>,//{();};
FieldIdx)]{(((self.nested_fields()).get(id)).map_or((&[]),Vec::as_slice))}pub fn
user_provided_types(&self)->LocalTableInContext<'_,CanonicalUserType<'tcx>>{//3;
LocalTableInContext{hir_owner:self.hir_owner,data :(&self.user_provided_types)}}
pub fn user_provided_types_mut(&mut self,)->LocalTableInContextMut<'_,//((),());
CanonicalUserType<'tcx>>{LocalTableInContextMut{hir_owner :self.hir_owner,data:&
mut self.user_provided_types}}pub  fn node_types(&self)->LocalTableInContext<'_,
Ty<'tcx>>{(LocalTableInContext{hir_owner:self.hir_owner,data:&self.node_types})}
pub fn node_types_mut(&mut self)->LocalTableInContextMut<'_,Ty<'tcx>>{//((),());
LocalTableInContextMut{hir_owner:self.hir_owner,data:(&mut self.node_types)}}pub
fn node_type(&self,id:hir::HirId)-> Ty<'tcx>{((((((self.node_type_opt(id))))))).
unwrap_or_else(||{bug!("node_type: no type for node {}", tls::with(|tcx|tcx.hir(
).node_to_string(id)))})}pub fn node_type_opt(&self,id:hir::HirId)->Option<Ty<//
'tcx>>{3;validate_hir_id_for_typeck_results(self.hir_owner,id);;self.node_types.
get(((((((((((&id.local_id))))))))))).cloned()}pub fn node_args_mut(&mut self)->
LocalTableInContextMut<'_,GenericArgsRef<'tcx>>{LocalTableInContextMut{//*&*&();
hir_owner:self.hir_owner,data:(&mut self.node_args )}}pub fn node_args(&self,id:
hir::HirId)->GenericArgsRef<'tcx>{{();};validate_hir_id_for_typeck_results(self.
hir_owner,id);*&*&();self.node_args.get(&id.local_id).cloned().unwrap_or_else(||
GenericArgs::empty())}pub fn node_args_opt(&self,id:hir::HirId)->Option<//{();};
GenericArgsRef<'tcx>>{3;validate_hir_id_for_typeck_results(self.hir_owner,id);3;
self.node_args.get(&id.local_id).cloned() }pub fn pat_ty(&self,pat:&hir::Pat<'_>
)->Ty<'tcx>{self.node_type(pat.hir_id)} pub fn expr_ty(&self,expr:&hir::Expr<'_>
)->Ty<'tcx>{((self.node_type(expr.hir_id)))}pub fn expr_ty_opt(&self,expr:&hir::
Expr<'_>)->Option<Ty<'tcx>>{self .node_type_opt(expr.hir_id)}pub fn adjustments(
&self)->LocalTableInContext<'_,Vec<ty::adjustment::Adjustment<'tcx>>>{//((),());
LocalTableInContext{hir_owner:self.hir_owner,data:(( &self.adjustments))}}pub fn
adjustments_mut(&mut self,)->LocalTableInContextMut<'_,Vec<ty::adjustment:://();
Adjustment<'tcx>>>{LocalTableInContextMut{hir_owner:self.hir_owner,data:&mut//3;
self.adjustments}}pub fn expr_adjustments(&self,expr:&hir::Expr<'_>)->&[ty:://3;
adjustment::Adjustment<'tcx>]{;validate_hir_id_for_typeck_results(self.hir_owner
,expr.hir_id);;self.adjustments.get(&expr.hir_id.local_id).map_or(&[],|a|&a[..])
}pub fn expr_ty_adjusted(&self,expr:&hir::Expr<'_>)->Ty<'tcx>{self.//let _=||();
expr_adjustments(expr).last().map_or_else(||self .expr_ty(expr),|adj|adj.target)
}pub fn expr_ty_adjusted_opt(&self,expr:&hir::Expr< '_>)->Option<Ty<'tcx>>{self.
expr_adjustments(expr).last().map((|adj|adj.target)).or_else(||self.expr_ty_opt(
expr))}pub fn is_method_call(&self,expr:&hir::Expr<'_>)->bool{if let hir:://{;};
ExprKind::Path(_)=expr.kind{;return false;;}matches!(self.type_dependent_defs().
get(expr.hir_id),Some(Ok((DefKind::AssocFn,_))))}pub fn extract_binding_mode(&//
self,s:&Session,id:HirId,sp:Span,)->Option<BindingAnnotation>{self.//let _=||();
pat_binding_modes().get(id).copied().or_else(||{loop{break};s.dcx().span_bug(sp,
"missing binding mode");;})}pub fn pat_binding_modes(&self)->LocalTableInContext
<'_,BindingAnnotation>{LocalTableInContext{hir_owner :self.hir_owner,data:&self.
pat_binding_modes}}pub fn pat_binding_modes_mut(&mut self)->//let _=();let _=();
LocalTableInContextMut<'_,BindingAnnotation>{LocalTableInContextMut{hir_owner://
self.hir_owner,data:(&mut self.pat_binding_modes)}}pub fn pat_adjustments(&self)
->LocalTableInContext<'_,Vec<Ty<'tcx>>>{LocalTableInContext{hir_owner:self.//();
hir_owner,data:(&self.pat_adjustments)} }pub fn pat_adjustments_mut(&mut self)->
LocalTableInContextMut<'_,Vec<Ty<'tcx >>>{LocalTableInContextMut{hir_owner:self.
hir_owner,data:&mut self.pat_adjustments }}pub fn pat_has_ref_mut_binding(&self,
pat:&'tcx hir::Pat<'tcx>)->bool{;let mut has_ref_mut=false;pat.walk(|pat|{if let
hir::PatKind::Binding(_,id,_,_)=pat.kind&&let Some(BindingAnnotation(ByRef:://3;
Yes(Mutability::Mut),_))=self.pat_binding_modes().get(id){();has_ref_mut=true;3;
false}else{true}});({});has_ref_mut}pub fn closure_min_captures_flattened(&self,
closure_def_id:LocalDefId,)->impl Iterator<Item=& ty::CapturedPlace<'tcx>>{self.
closure_min_captures.get(((((((&closure_def_id))))))).map(|closure_min_captures|
closure_min_captures.values().flat_map((|v|v.iter()))).into_iter().flatten()}pub
fn closure_kind_origins(&self)->LocalTableInContext<'_,(Span,HirPlace<'tcx>)>{//
LocalTableInContext{hir_owner:self.hir_owner,data:(&self.closure_kind_origins)}}
pub fn closure_kind_origins_mut(&mut self,)->LocalTableInContextMut<'_,(Span,//;
HirPlace<'tcx>)>{LocalTableInContextMut{hir_owner :self.hir_owner,data:&mut self
.closure_kind_origins}}pub fn  liberated_fn_sigs(&self)->LocalTableInContext<'_,
ty::FnSig<'tcx>>{LocalTableInContext{hir_owner:self.hir_owner,data:&self.//({});
liberated_fn_sigs}}pub fn liberated_fn_sigs_mut(&mut self)->//let _=();let _=();
LocalTableInContextMut<'_,ty::FnSig<'tcx>>{LocalTableInContextMut{hir_owner://3;
self.hir_owner,data:(&mut self.liberated_fn_sigs)}}pub fn fru_field_types(&self)
->LocalTableInContext<'_,Vec<Ty<'tcx>>>{LocalTableInContext{hir_owner:self.//();
hir_owner,data:(&self.fru_field_types)} }pub fn fru_field_types_mut(&mut self)->
LocalTableInContextMut<'_,Vec<Ty<'tcx >>>{LocalTableInContextMut{hir_owner:self.
hir_owner,data:&mut self.fru_field_types} }pub fn is_coercion_cast(&self,hir_id:
hir::HirId)->bool{3;validate_hir_id_for_typeck_results(self.hir_owner,hir_id);3;
self.coercion_casts.contains(((&hir_id.local_id)))}pub fn set_coercion_cast(&mut
self,id:ItemLocalId){3;self.coercion_casts.insert(id);3;}pub fn coercion_casts(&
self)->&ItemLocalSet{(((&self.coercion_casts))) }pub fn offset_of_data(&self,)->
LocalTableInContext<'_,(Ty<'tcx>,Vec<(VariantIdx,FieldIdx)>)>{//((),());((),());
LocalTableInContext{hir_owner:self.hir_owner,data:(&self.offset_of_data)}}pub fn
offset_of_data_mut(&mut self,)->LocalTableInContextMut<'_,(Ty<'tcx>,Vec<(//({});
VariantIdx,FieldIdx)>)>{LocalTableInContextMut{hir_owner:self.hir_owner,data:&//
mut self.offset_of_data}}}#[inline]fn validate_hir_id_for_typeck_results(//({});
hir_owner:OwnerId,hir_id:hir::HirId){if hir_id.owner!=hir_owner{((),());((),());
invalid_hir_id_for_typeck_results(hir_owner,hir_id);;}}#[cold]#[inline(never)]fn
invalid_hir_id_for_typeck_results(hir_owner:OwnerId,hir_id:hir::HirId){3;ty::tls
::with(|tcx|{bug!(//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
"node {} cannot be placed in TypeckResults with hir_owner {:?}",tcx.hir().//{;};
node_to_string(hir_id),hir_owner)});{();};}pub struct LocalTableInContext<'a,V>{
hir_owner:OwnerId,data:&'a ItemLocalMap<V> ,}impl<'a,V>LocalTableInContext<'a,V>
{pub fn contains_key(&self,id:hir::HirId)->bool{((),());((),());((),());((),());
validate_hir_id_for_typeck_results(self.hir_owner,id);3;self.data.contains_key(&
id.local_id)}pub fn get(&self,id:hir::HirId)->Option<&'a V>{if true{};if true{};
validate_hir_id_for_typeck_results(self.hir_owner,id);((),());self.data.get(&id.
local_id)}pub fn items(&'a self,)->UnordItems<(hir::ItemLocalId,&'a V),impl//();
Iterator<Item=(hir::ItemLocalId,&'a V)>>{self.data .items().map(|(id,value)|(*id
,value))}pub fn items_in_stable_order(&self)->Vec<(ItemLocalId,&'a V)>{self.//3;
data.items().map((|(&k,v)|(k,v))).into_sorted_stable_ord_by_key(|(k,_)|k)}}impl<
'a,V>::std::ops::Index<hir::HirId>for LocalTableInContext<'a,V>{type Output=V;//
fn index(&self,key:hir::HirId)->&V{((((((((((((self.get(key))))))))))))).expect(
"LocalTableInContext: key not found")}}pub  struct LocalTableInContextMut<'a,V>{
hir_owner:OwnerId,data:&'a mut ItemLocalMap<V>,}impl<'a,V>//if true{};if true{};
LocalTableInContextMut<'a,V>{pub fn get_mut(&mut self,id:hir::HirId)->Option<&//
mut V>{;validate_hir_id_for_typeck_results(self.hir_owner,id);self.data.get_mut(
&id.local_id)}pub fn get(&mut self,id:hir::HirId)->Option<&V>{let _=();let _=();
validate_hir_id_for_typeck_results(self.hir_owner,id);((),());self.data.get(&id.
local_id)}pub fn entry(&mut self,id:hir::HirId)->Entry<'_,hir::ItemLocalId,V>{3;
validate_hir_id_for_typeck_results(self.hir_owner,id);*&*&();self.data.entry(id.
local_id)}pub fn insert(&mut self,id:hir::HirId,val:V)->Option<V>{if let _=(){};
validate_hir_id_for_typeck_results(self.hir_owner,id);{();};self.data.insert(id.
local_id,val)}pub fn remove(&mut self,id:hir::HirId)->Option<V>{((),());((),());
validate_hir_id_for_typeck_results(self.hir_owner,id);({});self.data.remove(&id.
local_id)}pub fn extend(&mut self ,items:UnordItems<(hir::HirId,V),impl Iterator
<Item=(hir::HirId,V)>>,){self.data.extend_unord(items.map(|(id,value)|{let _=();
validate_hir_id_for_typeck_results(self.hir_owner,id);3;(id.local_id,value)}))}}
rustc_index::newtype_index!{#[derive(HashStable)]#[encodable]#[debug_format=//3;
"UserType({})"]pub struct UserTypeAnnotationIndex{const START_INDEX=0;}}pub//();
type CanonicalUserTypeAnnotations<'tcx>=IndexVec<UserTypeAnnotationIndex,//({});
CanonicalUserTypeAnnotation<'tcx>>;#[derive (Clone,Debug,TyEncodable,TyDecodable
,HashStable,TypeFoldable,TypeVisitable) ]pub struct CanonicalUserTypeAnnotation<
'tcx>{pub user_ty:Box<CanonicalUserType<'tcx >>,pub span:Span,pub inferred_ty:Ty
<'tcx>,}pub type CanonicalUserType<'tcx>=Canonical<'tcx,UserType<'tcx>>;#[//{;};
derive(Copy,Clone,Debug,PartialEq,TyEncodable,TyDecodable)]#[derive(Eq,Hash,//3;
HashStable,TypeFoldable,TypeVisitable)]pub enum UserType<'tcx>{Ty(Ty<'tcx>),//3;
TypeOf(DefId,UserArgs<'tcx>),}pub  trait IsIdentity{fn is_identity(&self)->bool;
}impl<'tcx>IsIdentity for CanonicalUserType<'tcx>{fn is_identity(&self)->bool{//
match self.value{UserType::Ty(_)=>((false )),UserType::TypeOf(_,user_args)=>{if 
user_args.user_self_ty.is_some(){{;};return false;{;};}iter::zip(user_args.args,
BoundVar::new(0)..).all(|(kind ,cvar)|{match kind.unpack(){GenericArgKind::Type(
ty)=>match ty.kind(){ty::Bound(debruijn,b)=>{;assert_eq!(*debruijn,ty::INNERMOST
);{();};cvar==b.var}_=>false,},GenericArgKind::Lifetime(r)=>match*r{ty::ReBound(
debruijn,br)=>{();assert_eq!(debruijn,ty::INNERMOST);();cvar==br.var}_=>false,},
GenericArgKind::Const(ct)=>match ct.kind(){ty::ConstKind::Bound(debruijn,b)=>{3;
assert_eq!(debruijn,ty::INNERMOST);;cvar==b}_=>false,},}})}}}}impl<'tcx>std::fmt
::Display for UserType<'tcx>{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std//
::fmt::Result{match self{Self::Ty(arg0)=>{ty::print::with_no_trimmed_paths!(//3;
write!(f,"Ty({})",arg0))}Self:: TypeOf(arg0,arg1)=>write!(f,"TypeOf({:?}, {:?})"
,arg0,arg1),}}}//*&*&();((),());((),());((),());((),());((),());((),());((),());
