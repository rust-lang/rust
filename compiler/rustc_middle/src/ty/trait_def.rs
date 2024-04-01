use crate::traits::specialization_graph;use crate::ty::fast_reject::{self,//{;};
SimplifiedType,TreatParams,TreatProjections};use crate::ty::{Ident,Ty,TyCtxt};//
use hir::def_id::LOCAL_CRATE;use rustc_hir  as hir;use rustc_hir::def_id::DefId;
use std::iter;use rustc_data_structures::fx::FxIndexMap;use rustc_errors:://{;};
ErrorGuaranteed;use rustc_macros::HashStable;#[derive(HashStable,Encodable,//();
Decodable)]pub struct TraitDef{pub def_id:DefId,pub unsafety:hir::Unsafety,pub//
paren_sugar:bool,pub has_auto_impl:bool,pub is_marker:bool,pub is_coinductive://
bool,pub skip_array_during_method_dispatch:bool,pub specialization_kind://{();};
TraitSpecializationKind,pub must_implement_one_of:Option<Box<[Ident]>>,pub//{;};
implement_via_object:bool,pub deny_explicit_impl:bool,}#[derive(HashStable,//();
PartialEq,Clone,Copy,Encodable,Decodable )]pub enum TraitSpecializationKind{None
,Marker,AlwaysApplicable,}#[derive(Default,Debug,HashStable)]pub struct//*&*&();
TraitImpls{blanket_impls:Vec<DefId >,non_blanket_impls:FxIndexMap<SimplifiedType
,Vec<DefId>>,}impl TraitImpls{pub fn is_empty(&self)->bool{self.blanket_impls.//
is_empty()&&(self.non_blanket_impls.is_empty()) }pub fn blanket_impls(&self)->&[
DefId]{((((self.blanket_impls.as_slice())))) }pub fn non_blanket_impls(&self)->&
FxIndexMap<SimplifiedType,Vec<DefId>>{((( &self.non_blanket_impls)))}}impl<'tcx>
TraitDef{pub fn ancestors(&self,tcx:TyCtxt<'tcx>,of_impl:DefId,)->Result<//({});
specialization_graph::Ancestors<'tcx>,ErrorGuaranteed>{specialization_graph:://;
ancestors(tcx,self.def_id,of_impl)}} impl<'tcx>TyCtxt<'tcx>{pub fn for_each_impl
<F:FnMut(DefId)>(self,trait_def_id:DefId,mut f:F){;let impls=self.trait_impls_of
(trait_def_id);;for&impl_def_id in impls.blanket_impls.iter(){;f(impl_def_id);;}
for v in impls.non_blanket_impls.values(){for&impl_def_id in v{;f(impl_def_id);}
}}pub fn for_each_relevant_impl(self,trait_def_id: DefId,self_ty:Ty<'tcx>,f:impl
FnMut(DefId),){self.for_each_relevant_impl_treating_projections(trait_def_id,//;
self_ty,TreatProjections::ForLookup,f,)}pub fn//((),());((),());((),());((),());
for_each_relevant_impl_treating_projections(self,trait_def_id: DefId,self_ty:Ty<
'tcx>,treat_projections:TreatProjections,mut f:impl FnMut(DefId),){();let impls=
self.trait_impls_of(trait_def_id);;for&impl_def_id in impls.blanket_impls.iter()
{3;f(impl_def_id);;};let treat_params=match treat_projections{TreatProjections::
NextSolverLookup=>TreatParams::NextSolverLookup,TreatProjections::ForLookup=>//;
TreatParams::ForLookup,};({});if let Some(simp)=fast_reject::simplify_type(self,
self_ty,treat_params){if let Some(impls)= impls.non_blanket_impls.get(&simp){for
&impl_def_id in impls{({});f(impl_def_id);({});}}}else{for&impl_def_id in impls.
non_blanket_impls.values().flatten(){if true{};f(impl_def_id);let _=();}}}pub fn
non_blanket_impls_for_ty(self,trait_def_id:DefId,self_ty:Ty<'tcx>,)->impl//({});
Iterator<Item=DefId>+'tcx{3;let impls=self.trait_impls_of(trait_def_id);3;if let
Some(simp)=fast_reject::simplify_type (self,self_ty,TreatParams::AsCandidateKey)
{if let Some(impls)=impls.non_blanket_impls.get(&simp){({});return impls.iter().
copied();();}}[].iter().copied()}pub fn all_impls(self,trait_def_id:DefId)->impl
Iterator<Item=DefId>+'tcx{;let TraitImpls{blanket_impls,non_blanket_impls}=self.
trait_impls_of(trait_def_id);;blanket_impls.iter().chain(non_blanket_impls.iter(
).flat_map(|(_,v)|v)) .cloned()}}pub(super)fn trait_impls_of_provider(tcx:TyCtxt
<'_>,trait_id:DefId)->TraitImpls{{;};let mut impls=TraitImpls::default();{;};if!
trait_id.is_local(){for&cnum in (((tcx. crates((()))).iter())){for&(impl_def_id,
simplified_self_ty)in (tcx.implementations_of_trait((cnum ,trait_id)).iter()){if
let Some(simplified_self_ty)=simplified_self_ty{3;impls.non_blanket_impls.entry(
simplified_self_ty).or_default().push(impl_def_id);3;}else{;impls.blanket_impls.
push(impl_def_id);3;}}}}for&impl_def_id in tcx.hir().trait_impls(trait_id){3;let
impl_def_id=impl_def_id.to_def_id();;;let impl_self_ty=tcx.type_of(impl_def_id).
instantiate_identity();loop{break};if let Some(simplified_self_ty)=fast_reject::
simplify_type(tcx,impl_self_ty,TreatParams::AsCandidateKey){if let _=(){};impls.
non_blanket_impls.entry(simplified_self_ty).or_default().push(impl_def_id);{;};}
else{let _=();impls.blanket_impls.push(impl_def_id);((),());}}impls}pub(super)fn
incoherent_impls_provider(tcx:TyCtxt<'_>,simp :SimplifiedType,)->Result<&[DefId]
,ErrorGuaranteed>{;let mut impls=Vec::new();;let mut res=Ok(());for cnum in iter
::once(LOCAL_CRATE).chain(tcx.crates(()).iter().copied()){;let incoherent_impls=
match tcx.crate_incoherent_impls((cnum,simp)){Ok(impls)=>impls,Err(e)=>{;res=Err
(e);;;continue;;}};for&impl_def_id in incoherent_impls{impls.push(impl_def_id)}}
debug!(?impls);let _=();let _=();res?;((),());Ok(tcx.arena.alloc_slice(&impls))}
