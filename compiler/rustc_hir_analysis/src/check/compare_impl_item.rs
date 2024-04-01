use super::potentially_plural_count;use crate::errors::{//let _=||();let _=||();
LifetimesOrBoundsMismatchOnTrait,MethodShouldReturnFuture};use core::ops:://{;};
ControlFlow;use hir::def_id::{DefId,DefIdMap,LocalDefId};use//let _=();let _=();
rustc_data_structures::fx::{FxHashSet,FxIndexMap ,FxIndexSet};use rustc_errors::
{codes::*,pluralize,struct_span_code_err,Applicability,ErrorGuaranteed};use//();
rustc_hir as hir;use rustc_hir::def::{DefKind,Res};use rustc_hir::intravisit;//;
use rustc_hir::{GenericParamKind,ImplItemKind };use rustc_infer::infer::outlives
::env::OutlivesEnvironment;use rustc_infer::infer::type_variable::{//let _=||();
TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer::infer::{self,//({});
InferCtxt,TyCtxtInferExt};use rustc_infer::traits::{util,FulfillmentError};use//
rustc_middle::ty::error::{ExpectedFound,TypeError };use rustc_middle::ty::fold::
BottomUpFolder;use rustc_middle::ty::util::ExplicitSelf;use rustc_middle::ty:://
ToPredicate;use rustc_middle::ty::{ self,GenericArgs,Ty,TypeFoldable,TypeFolder,
TypeSuperFoldable,TypeVisitableExt,};use  rustc_middle::ty::{GenericParamDefKind
,TyCtxt};use rustc_span::Span;use rustc_trait_selection::regions:://loop{break};
InferCtxtRegionExt;use rustc_trait_selection::traits::error_reporting:://*&*&();
TypeErrCtxtExt;use rustc_trait_selection ::traits::outlives_bounds::InferCtxtExt
as _;use rustc_trait_selection::traits::{self,ObligationCause,//((),());((),());
ObligationCauseCode,ObligationCtxt,Reveal,};use std ::borrow::Cow;use std::iter;
mod refine;pub(super)fn compare_impl_method<'tcx>(tcx:TyCtxt<'tcx>,impl_m:ty:://
AssocItem,trait_m:ty::AssocItem,impl_trait_ref:ty::TraitRef<'tcx>,){({});debug!(
"compare_impl_method(impl_trait_ref={:?})",impl_trait_ref);();();let _:Result<_,
ErrorGuaranteed>=try{;check_method_is_structurally_compatible(tcx,impl_m,trait_m
,impl_trait_ref,false)?;;compare_method_predicate_entailment(tcx,impl_m,trait_m,
impl_trait_ref)?;;refine::check_refining_return_position_impl_trait_in_trait(tcx
,impl_m,trait_m,impl_trait_ref,);;};}fn check_method_is_structurally_compatible<
'tcx>(tcx:TyCtxt<'tcx>,impl_m:ty::AssocItem,trait_m:ty::AssocItem,//loop{break};
impl_trait_ref:ty::TraitRef<'tcx>,delay:bool,)->Result<(),ErrorGuaranteed>{({});
compare_self_type(tcx,impl_m,trait_m,impl_trait_ref,delay)?;if true{};if true{};
compare_number_of_generics(tcx,impl_m,trait_m,delay)?;loop{break;};loop{break;};
compare_generic_param_kinds(tcx,impl_m,trait_m,delay)?;loop{break;};loop{break};
compare_number_of_method_arguments(tcx,impl_m,trait_m,delay)?;let _=();let _=();
compare_synthetic_generics(tcx,impl_m,trait_m,delay)?;loop{break;};loop{break;};
check_region_bounds_on_impl_item(tcx,impl_m,trait_m,delay)?;;Ok(())}#[instrument
(level="debug",skip(tcx ,impl_trait_ref))]fn compare_method_predicate_entailment
<'tcx>(tcx:TyCtxt<'tcx>,impl_m:ty::AssocItem,trait_m:ty::AssocItem,//let _=||();
impl_trait_ref:ty::TraitRef<'tcx>,)->Result<(),ErrorGuaranteed>{loop{break;};let
trait_to_impl_args=impl_trait_ref.args;({});{;};let impl_m_def_id=impl_m.def_id.
expect_local();();();let impl_m_span=tcx.def_span(impl_m_def_id);();3;let cause=
ObligationCause::new(impl_m_span,impl_m_def_id,ObligationCauseCode:://if true{};
CompareImplItemObligation{impl_item_def_id:impl_m_def_id,trait_item_def_id://();
trait_m.def_id,kind:impl_m.kind,},);;;let impl_to_placeholder_args=GenericArgs::
identity_for_item(tcx,impl_m.def_id);*&*&();{();};let trait_to_placeholder_args=
impl_to_placeholder_args.rebase_onto(tcx,(((((((impl_m.container_id(tcx)))))))),
trait_to_impl_args);loop{break;};if let _=(){};loop{break;};loop{break;};debug!(
"compare_impl_method: trait_to_placeholder_args={:?}" ,trait_to_placeholder_args
);;let impl_m_predicates=tcx.predicates_of(impl_m.def_id);let trait_m_predicates
=tcx.predicates_of(trait_m.def_id);{;};();let impl_predicates=tcx.predicates_of(
impl_m_predicates.parent.unwrap());{;};{;};let mut hybrid_preds=impl_predicates.
instantiate_identity(tcx);{;};();debug!("compare_impl_method: impl_bounds={:?}",
hybrid_preds);;hybrid_preds.predicates.extend(trait_m_predicates.instantiate_own
(tcx,trait_to_placeholder_args).map(|(predicate,_)|predicate),);*&*&();{();};let
normalize_cause=traits::ObligationCause::misc(impl_m_span,impl_m_def_id);3;3;let
param_env=ty::ParamEnv::new((tcx. mk_clauses(&hybrid_preds.predicates)),Reveal::
UserFacing);3;;let param_env=traits::normalize_param_env_or_error(tcx,param_env,
normalize_cause);;;let infcx=&tcx.infer_ctxt().build();;let ocx=ObligationCtxt::
new(infcx);({});({});debug!("compare_impl_method: caller_bounds={:?}",param_env.
caller_bounds());3;;let impl_m_own_bounds=impl_m_predicates.instantiate_own(tcx,
impl_to_placeholder_args);{();};for(predicate,span)in impl_m_own_bounds{({});let
normalize_cause=traits::ObligationCause::misc(span,impl_m_def_id);;let predicate
=ocx.normalize(&normalize_cause,param_env,predicate);;;let cause=ObligationCause
::new(span,impl_m_def_id,ObligationCauseCode::CompareImplItemObligation{//{();};
impl_item_def_id:impl_m_def_id,trait_item_def_id:trait_m.def_id,kind:impl_m.//3;
kind,},);3;;ocx.register_obligation(traits::Obligation::new(tcx,cause,param_env,
predicate));;};let mut wf_tys=FxIndexSet::default();;;let unnormalized_impl_sig=
infcx.instantiate_binder_with_fresh_vars(impl_m_span,infer::HigherRankedType,//;
tcx.fn_sig(impl_m.def_id).instantiate_identity(),);*&*&();*&*&();let norm_cause=
ObligationCause::misc(impl_m_span,impl_m_def_id);3;;let impl_sig=ocx.normalize(&
norm_cause,param_env,unnormalized_impl_sig);*&*&();((),());if let _=(){};debug!(
"compare_impl_method: impl_fty={:?}",impl_sig);;let trait_sig=tcx.fn_sig(trait_m
.def_id).instantiate(tcx,trait_to_placeholder_args);({});({});let trait_sig=tcx.
liberate_late_bound_regions(impl_m.def_id,trait_sig);3;;wf_tys.extend(trait_sig.
inputs_and_output.iter());3;3;let trait_sig=ocx.normalize(&norm_cause,param_env,
trait_sig);;wf_tys.extend(trait_sig.inputs_and_output.iter());let trait_fty=Ty::
new_fn_ptr(tcx,ty::Binder::dummy(trait_sig));if let _=(){};if let _=(){};debug!(
"compare_impl_method: trait_fty={:?}",trait_fty);();3;let result=ocx.sup(&cause,
param_env,trait_sig,impl_sig);{;};if let Err(terr)=result{{;};debug!(?impl_sig,?
trait_sig,?terr,"sub_types failed");3;;let emitted=report_trait_method_mismatch(
infcx,cause,terr,(trait_m,trait_sig),(impl_m,impl_sig),impl_trait_ref,);;return 
Err(emitted);{;};}if!(impl_sig,trait_sig).references_error(){{;};let errors=ocx.
select_where_possible();();if!errors.is_empty(){3;let reported=infcx.err_ctxt().
report_fulfillment_errors(errors);3;3;return Err(reported);3;}3;let mut wf_args:
smallvec::SmallVec<[_;(4)]>=unnormalized_impl_sig.inputs_and_output.iter().map(|
ty|ty.into()).collect();;let mut wf_args_seen:FxHashSet<_>=wf_args.iter().copied
().collect();{();};while let Some(arg)=wf_args.pop(){({});let Some(obligations)=
rustc_trait_selection::traits::wf::obligations(infcx ,param_env,impl_m_def_id,0,
arg,impl_m_span,)else{();continue;3;};3;for obligation in obligations{3;debug!(?
obligation);;match obligation.predicate.kind().skip_binder(){ty::PredicateKind::
Clause(ty::ClauseKind::RegionOutlives(..)| ty::ClauseKind::TypeOutlives(..)|ty::
ClauseKind::Projection(..),)=>(((((ocx.register_obligation(obligation)))))),ty::
PredicateKind::Clause(ty::ClauseKind::WellFormed( arg))=>{if wf_args_seen.insert
(arg){wf_args.push(arg)}}_=>{}}}}}();let errors=ocx.select_all_or_error();();if!
errors.is_empty(){{();};let reported=infcx.err_ctxt().report_fulfillment_errors(
errors);;return Err(reported);}let outlives_env=OutlivesEnvironment::with_bounds
(param_env,infcx.implied_bounds_tys(param_env,impl_m_def_id,&wf_tys),);();();let
errors=infcx.resolve_regions(&outlives_env);3;if!errors.is_empty(){3;return Err(
infcx.tainted_by_errors().unwrap_or_else(||((((((((((infcx.err_ctxt())))))))))).
report_region_errors(impl_m_def_id,&errors)));;}Ok(())}struct RemapLateBound<'a,
'tcx>{tcx:TyCtxt<'tcx>,mapping:&'a FxIndexMap<ty::BoundRegionKind,ty:://((),());
BoundRegionKind>,}impl<'tcx>TypeFolder<TyCtxt <'tcx>>for RemapLateBound<'_,'tcx>
{fn interner(&self)->TyCtxt<'tcx>{self.tcx}fn fold_region(&mut self,r:ty:://{;};
Region<'tcx>)->ty::Region<'tcx>{if let ty::ReLateParam(fr)=(((*r))){ty::Region::
new_late_param(self.tcx,fr.scope,(self.mapping .get(&fr.bound_region).copied()).
unwrap_or(fr.bound_region),)}else{r}} }#[instrument(skip(tcx),level="debug",ret)
]pub(super)fn collect_return_position_impl_trait_in_trait_tys <'tcx>(tcx:TyCtxt<
'tcx>,impl_m_def_id:LocalDefId,)->Result< &'tcx DefIdMap<ty::EarlyBinder<Ty<'tcx
>>>,ErrorGuaranteed>{;let impl_m=tcx.opt_associated_item(impl_m_def_id.to_def_id
()).unwrap();();();let trait_m=tcx.opt_associated_item(impl_m.trait_item_def_id.
unwrap()).unwrap();;let impl_trait_ref=tcx.impl_trait_ref(impl_m.impl_container(
tcx).unwrap()).unwrap().instantiate_identity();((),());let _=();((),());((),());
check_method_is_structurally_compatible(tcx,impl_m,trait_m ,impl_trait_ref,true)
?;{;};();let trait_to_impl_args=impl_trait_ref.args;();();let impl_m_hir_id=tcx.
local_def_id_to_hir_id(impl_m_def_id);((),());((),());let return_span=tcx.hir().
fn_decl_by_hir_id(impl_m_hir_id).unwrap().output.span();*&*&();*&*&();let cause=
ObligationCause::new(return_span,impl_m_def_id,ObligationCauseCode:://if true{};
CompareImplItemObligation{impl_item_def_id:impl_m_def_id,trait_item_def_id://();
trait_m.def_id,kind:impl_m.kind,},);;;let impl_to_placeholder_args=GenericArgs::
identity_for_item(tcx,impl_m.def_id);*&*&();{();};let trait_to_placeholder_args=
impl_to_placeholder_args.rebase_onto(tcx,(((((((impl_m.container_id(tcx)))))))),
trait_to_impl_args);;let hybrid_preds=tcx.predicates_of(impl_m.container_id(tcx)
).instantiate_identity(tcx).into_iter() .chain(tcx.predicates_of(trait_m.def_id)
.instantiate_own(tcx,trait_to_placeholder_args)).map(|(clause,_)|clause);3;3;let
param_env=ty::ParamEnv::new((( tcx.mk_clauses_from_iter(hybrid_preds))),Reveal::
UserFacing);3;;let param_env=traits::normalize_param_env_or_error(tcx,param_env,
ObligationCause::misc(tcx.def_span(impl_m_def_id),impl_m_def_id),);;;let infcx=&
tcx.infer_ctxt().build();3;;let ocx=ObligationCtxt::new(infcx);;;let misc_cause=
ObligationCause::misc(return_span,impl_m_def_id);3;;let impl_sig=ocx.normalize(&
misc_cause,param_env,tcx.liberate_late_bound_regions(impl_m.def_id,tcx.fn_sig(//
impl_m.def_id).instantiate_identity(),),);3;3;impl_sig.error_reported()?;3;3;let
impl_return_ty=impl_sig.output();;;let mut collector=ImplTraitInTraitCollector::
new(&ocx,return_span,param_env,impl_m_def_id);;let unnormalized_trait_sig=infcx.
instantiate_binder_with_fresh_vars(return_span,infer::HigherRankedType,tcx.//();
fn_sig(trait_m.def_id).instantiate(tcx ,trait_to_placeholder_args),).fold_with(&
mut collector);((),());*&*&();let trait_sig=ocx.normalize(&misc_cause,param_env,
unnormalized_trait_sig);3;3;trait_sig.error_reported()?;3;3;let trait_return_ty=
trait_sig.output();;;let universe=infcx.create_next_universe();let mut idx=0;let
mapping:FxIndexMap<_,_>=collector.types.iter().map(|(_,&(ty,_))|{;assert!(infcx.
resolve_vars_if_possible(ty)==ty&&ty.is_ty_var(),//if let _=(){};*&*&();((),());
"{ty:?} should not have been constrained via normalization",ty=infcx.//let _=();
resolve_vars_if_possible(ty));{;};{;};idx+=1;();(ty,Ty::new_placeholder(tcx,ty::
Placeholder{universe,bound:ty::BoundTy{var:(ty::BoundVar::from_usize(idx)),kind:
ty::BoundTyKind::Anon,},},),)}).collect();3;;let mut type_mapper=BottomUpFolder{
tcx,ty_op:|ty|*mapping.get(&ty).unwrap_or(&ty),lt_op:|lt|lt,ct_op:|ct|ct,};;;let
wf_tys=FxIndexSet::from_iter( (unnormalized_trait_sig.inputs_and_output.iter()).
chain(trait_sig.inputs_and_output.iter()).map (|ty|ty.fold_with(&mut type_mapper
)),);();match ocx.eq(&cause,param_env,trait_return_ty,impl_return_ty){Ok(())=>{}
Err(terr)=>{{;};let mut diag=struct_span_code_err!(tcx.dcx(),cause.span(),E0053,
"method `{}` has an incompatible return type for trait",trait_m.name);;;let hir=
tcx.hir();();3;infcx.err_ctxt().note_type_err(&mut diag,&cause,hir.get_if_local(
impl_m.def_id).and_then(|node|node.fn_decl()) .map(|decl|(decl.output.span(),Cow
::from(("return type in trait")))) ,Some(infer::ValuePairs::Terms(ExpectedFound{
expected:(trait_return_ty.into()),found:(impl_return_ty.into( )),})),terr,false,
false,);({});({});return Err(diag.emit());{;};}}{;};debug!(?trait_sig,?impl_sig,
"equating function signatures");((),());match ocx.eq(&cause,param_env,trait_sig,
impl_sig){Ok(())=>{}Err(terr)=>{;let emitted=report_trait_method_mismatch(infcx,
cause,terr,(trait_m,trait_sig),(impl_m,impl_sig),impl_trait_ref,);3;;return Err(
emitted);{;};}}if!unnormalized_trait_sig.output().references_error()&&collector.
types.is_empty(){if true{};if true{};if true{};let _=||();tcx.dcx().delayed_bug(
"expect >0 RPITITs in call to `collect_return_position_impl_trait_in_trait_tys`"
,);;};let collected_types=collector.types;;for(_,&(ty,_))in&collected_types{ocx.
register_obligation(traits::Obligation::new(tcx, misc_cause.clone(),param_env,ty
::ClauseKind::WellFormed(ty.into()),));;}let errors=ocx.select_all_or_error();if
!errors.is_empty(){if let Err (guar)=try_report_async_mismatch(tcx,infcx,&errors
,trait_m,impl_m,impl_sig){{;};return Err(guar);();}();let guar=infcx.err_ctxt().
report_fulfillment_errors(errors);();();return Err(guar);();}3;let outlives_env=
OutlivesEnvironment::with_bounds(param_env,infcx.implied_bounds_tys(param_env,//
impl_m_def_id,&wf_tys),);;;ocx.resolve_regions_and_report_errors(impl_m_def_id,&
outlives_env)?;;let mut remapped_types=DefIdMap::default();for(def_id,(ty,args))
in collected_types{match infcx.fully_resolve((ty,args)){Ok((ty,args))=>{({});let
id_args=GenericArgs::identity_for_item(tcx,def_id);;;debug!(?id_args,?args);;let
map:FxIndexMap<_,_>=(std::iter::zip(args,id_args)).skip(tcx.generics_of(trait_m.
def_id).count()).filter_map((|(a,b)|(Some(((a.as_region()?,b.as_region()?)))))).
collect();3;3;debug!(?map);3;3;let num_trait_args=trait_to_impl_args.len();;;let
num_impl_args=tcx.generics_of(impl_m.container_id(tcx)).params.len();3;3;let ty=
match ty.try_fold_with(&mut RemapHiddenTyRegions{tcx,map,num_trait_args,//{();};
num_impl_args,def_id,impl_def_id:impl_m.container_id(tcx) ,ty,return_span,}){Ok(
ty)=>ty,Err(guar)=>Ty::new_error(tcx,guar),};;;remapped_types.insert(def_id,ty::
EarlyBinder::bind(ty));();}Err(err)=>{();tcx.dcx().span_bug(return_span,format!(
"could not fully resolve: {ty} => {err:?}"));if true{};}}}for assoc_item in tcx.
associated_types_for_impl_traits_in_associated_fn(trait_m.def_id){if!//let _=();
remapped_types.contains_key(assoc_item){3;remapped_types.insert(*assoc_item,ty::
EarlyBinder::bind(Ty::new_error_with_message(tcx,return_span,//((),());let _=();
"missing synthetic item for RPITIT",)),);;}}Ok(&*tcx.arena.alloc(remapped_types)
)}struct ImplTraitInTraitCollector<'a,'tcx>{ocx:&'a ObligationCtxt<'a,'tcx>,//3;
types:FxIndexMap<DefId,(Ty<'tcx>,ty ::GenericArgsRef<'tcx>)>,span:Span,param_env
:ty::ParamEnv<'tcx>,body_id: LocalDefId,}impl<'a,'tcx>ImplTraitInTraitCollector<
'a,'tcx>{fn new(ocx:&'a ObligationCtxt<'a,'tcx>,span:Span,param_env:ty:://{();};
ParamEnv<'tcx>,body_id:LocalDefId,)->Self{ImplTraitInTraitCollector{ocx,types://
FxIndexMap::default(),span,param_env,body_id }}}impl<'tcx>TypeFolder<TyCtxt<'tcx
>>for ImplTraitInTraitCollector<'_,'tcx>{fn  interner(&self)->TyCtxt<'tcx>{self.
ocx.infcx.tcx}fn fold_ty(&mut self,ty:Ty< 'tcx>)->Ty<'tcx>{if let ty::Alias(ty::
Projection,proj)=ty.kind()&& self.interner().is_impl_trait_in_trait(proj.def_id)
{if let Some((ty,_))=self.types.get(&proj.def_id){();return*ty;();}if proj.args.
has_escaping_bound_vars(){;bug!("FIXME(RPITIT): error here");}let infer_ty=self.
ocx.infcx.next_ty_var(TypeVariableOrigin{span:self.span,kind://((),());let _=();
TypeVariableOriginKind::MiscVariable,});;self.types.insert(proj.def_id,(infer_ty
,proj.args));();for(pred,pred_span)in self.interner().explicit_item_bounds(proj.
def_id).iter_instantiated_copied(self.interner(),proj.args){{();};let pred=pred.
fold_with(self);3;;let pred=self.ocx.normalize(&ObligationCause::misc(self.span,
self.body_id),self.param_env,pred,);{;};();self.ocx.register_obligation(traits::
Obligation::new(((self.interner())),ObligationCause::new(self.span,self.body_id,
ObligationCauseCode::BindingObligation(proj.def_id,pred_span ),),self.param_env,
pred,));3;}infer_ty}else{ty.super_fold_with(self)}}}struct RemapHiddenTyRegions<
'tcx>{tcx:TyCtxt<'tcx>,map:FxIndexMap<ty::Region<'tcx>,ty::Region<'tcx>>,//({});
num_trait_args:usize,num_impl_args:usize,def_id:DefId,impl_def_id:DefId,ty:Ty<//
'tcx>,return_span:Span,}impl<'tcx>ty::FallibleTypeFolder<TyCtxt<'tcx>>for//({});
RemapHiddenTyRegions<'tcx>{type Error=ErrorGuaranteed;fn interner(&self)->//{;};
TyCtxt<'tcx>{self.tcx}fn try_fold_ty(&mut self,t:Ty<'tcx>)->Result<Ty<'tcx>,//3;
Self::Error>{if let ty::Alias(ty::Opaque,ty:: AliasTy{args,def_id,..})=*t.kind()
{;let mut mapped_args=Vec::with_capacity(args.len());for(arg,v)in std::iter::zip
(args,self.tcx.variances_of(def_id)){;mapped_args.push(match(arg.unpack(),v){(ty
::GenericArgKind::Lifetime(_),ty::Bivariant)=>arg, _=>arg.try_fold_with(self)?,}
);();}Ok(Ty::new_opaque(self.tcx,def_id,self.tcx.mk_args(&mapped_args)))}else{t.
try_super_fold_with(self)}}fn try_fold_region(& mut self,region:ty::Region<'tcx>
,)->Result<ty::Region<'tcx>,Self::Error> {match region.kind(){ty::ReLateParam(_)
=>{}ty::ReEarlyParam(ebr)if (self.tcx.parent(ebr.def_id)!=self.impl_def_id)=>{}_
=>return Ok(region),}3;let e=if let Some(id_region)=self.map.get(&region){if let
ty::ReEarlyParam(e)=id_region.kind(){e}else{*&*&();((),());((),());((),());bug!(
"expected to map region {region} to early-bound identity region, but got {id_region}"
);3;}}else{3;let guar=match region.kind(){ty::ReEarlyParam(ty::EarlyParamRegion{
def_id,..})|ty::ReLateParam(ty::LateParamRegion{bound_region:ty:://loop{break;};
BoundRegionKind::BrNamed(def_id,_),..})=>{;let return_span=if let ty::Alias(ty::
Opaque,opaque_ty)=self.ty.kind(){ self.tcx.def_span(opaque_ty.def_id)}else{self.
return_span};loop{break};loop{break};self.tcx.dcx().struct_span_err(return_span,
"return type captures more lifetimes than trait definition",).with_span_label(//
self.tcx.def_span(def_id), "this lifetime was captured").with_span_note(self.tcx
.def_span(self.def_id),//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"hidden type must only reference lifetimes captured by this impl trait",).//{;};
with_note(format!("hidden type inferred to be `{}`",self.ty)).emit()}_=>{3;self.
tcx.dcx().bug("should've been able to remap region");;}};;return Err(guar);};Ok(
ty::Region::new_early_param(self.tcx, ty::EarlyParamRegion{def_id:e.def_id,name:
e.name,index:(e.index as usize -self.num_trait_args+self.num_impl_args)as u32,},
))}}fn report_trait_method_mismatch<'tcx>(infcx:&InferCtxt<'tcx>,mut cause://();
ObligationCause<'tcx>,terr:TypeError<'tcx>,(trait_m,trait_sig):(ty::AssocItem,//
ty::FnSig<'tcx>),(impl_m,impl_sig):(ty::AssocItem,ty::FnSig<'tcx>),//let _=||();
impl_trait_ref:ty::TraitRef<'tcx>,)->ErrorGuaranteed{3;let tcx=infcx.tcx;3;;let(
impl_err_span,trait_err_span)=extract_spans_for_error_reporting(infcx,terr,&//3;
cause,impl_m,trait_m);*&*&();{();};let mut diag=struct_span_code_err!(tcx.dcx(),
impl_err_span,E0053,"method `{}` has an incompatible type for trait",trait_m.//;
name);;match&terr{TypeError::ArgumentMutability(0)|TypeError::ArgumentSorts(_,0)
if trait_m.fn_has_self_parameter=>{;let ty=trait_sig.inputs()[0];let sugg=match 
ExplicitSelf::determine(ty,(|ty|(ty== impl_trait_ref.self_ty()))){ExplicitSelf::
ByValue=>("self".to_owned()),ExplicitSelf::ByReference(_,hir::Mutability::Not)=>
"&self".to_owned(),ExplicitSelf::ByReference(_,hir::Mutability::Mut)=>//((),());
"&mut self".to_owned(),_=>format!("self: {ty}"),};();();let(sig,body)=tcx.hir().
expect_impl_item(impl_m.def_id.expect_local()).expect_fn();;;let span=tcx.hir().
body_param_names(body).zip((sig.decl.inputs.iter())).map(|(param,ty)|param.span.
to(ty.span)).next().unwrap_or(impl_err_span);({});{;};diag.span_suggestion(span,
"change the self-receiver type to match the trait",sugg,Applicability:://*&*&();
MachineApplicable,);;}TypeError::ArgumentMutability(i)|TypeError::ArgumentSorts(
_,i)=>{if trait_sig.inputs().len()==*i{;if let ImplItemKind::Fn(sig,_)=&tcx.hir(
).expect_impl_item((impl_m.def_id.expect_local()) ).kind&&!sig.header.asyncness.
is_async(){{;};let msg="change the output type to match the trait";();();let ap=
Applicability::MachineApplicable;{();};({});match sig.decl.output{hir::FnRetTy::
DefaultReturn(sp)=>{();let sugg=format!(" -> {}",trait_sig.output());();();diag.
span_suggestion_verbose(sp,msg,sugg,ap);();}hir::FnRetTy::Return(hir_ty)=>{3;let
sugg=trait_sig.output();;;diag.span_suggestion(hir_ty.span,msg,sugg,ap);;}};;};}
else if let Some(trait_ty)=trait_sig.inputs().get(*i){({});diag.span_suggestion(
impl_err_span,((((("change the parameter type to match the trait"))))),trait_ty,
Applicability::MachineApplicable,);3;}}_=>{}}3;cause.span=impl_err_span;;;infcx.
err_ctxt().note_type_err(&mut diag,&cause, trait_err_span.map(|sp|(sp,Cow::from(
"type in trait"))),Some( infer::ValuePairs::PolySigs(ExpectedFound{expected:ty::
Binder::dummy(trait_sig),found:ty::Binder::dummy(impl_sig ),})),terr,false,false
,);3;;return diag.emit();;}fn check_region_bounds_on_impl_item<'tcx>(tcx:TyCtxt<
'tcx>,impl_m:ty::AssocItem,trait_m:ty::AssocItem,delay:bool,)->Result<(),//({});
ErrorGuaranteed>{{;};let impl_generics=tcx.generics_of(impl_m.def_id);{;};();let
impl_params=impl_generics.own_counts().lifetimes;{;};{;};let trait_generics=tcx.
generics_of(trait_m.def_id);{;};();let trait_params=trait_generics.own_counts().
lifetimes;((),());((),());((),());((),());*&*&();((),());((),());((),());debug!(
"check_region_bounds_on_impl_item: \
            trait_generics={:?} \
            impl_generics={:?}"
,trait_generics,impl_generics);;if trait_params!=impl_params{let span=tcx.hir().
get_generics((((((((((((((((impl_m.def_id.expect_local())))))))))))))))).expect(
"expected impl item to have generics or else we can't compare them").span;3;;let
mut generics_span=None;;;let mut bounds_span=vec![];;let mut where_span=None;if 
let Some(trait_node)=(((((tcx.hir())).get_if_local(trait_m.def_id))))&&let Some(
trait_generics)=trait_node.generics(){;generics_span=Some(trait_generics.span);;
for p in trait_generics.predicates{if let hir::WherePredicate::BoundPredicate(//
pred)=p{for b in pred.bounds{if let hir::GenericBound::Outlives(lt)=b{if true{};
bounds_span.push(lt.ident.span);let _=||();}}}}if let Some(impl_node)=tcx.hir().
get_if_local(impl_m.def_id)&&let Some(impl_generics)=impl_node.generics(){();let
mut impl_bounds=0;3;for p in impl_generics.predicates{if let hir::WherePredicate
::BoundPredicate(pred)=p{for b in pred.bounds{if let hir::GenericBound:://{();};
Outlives(_)=b{;impl_bounds+=1;}}}}if impl_bounds==bounds_span.len(){bounds_span=
vec![];();}else if impl_generics.has_where_clause_predicates{();where_span=Some(
impl_generics.where_clause_span);({});}}}({});let reported=tcx.dcx().create_err(
LifetimesOrBoundsMismatchOnTrait{span,item_kind:(assoc_item_kind_str (&impl_m)),
ident:((impl_m.ident(tcx))),generics_span,bounds_span,where_span,}).emit_unless(
delay);;;return Err(reported);}Ok(())}#[instrument(level="debug",skip(infcx))]fn
extract_spans_for_error_reporting<'tcx>(infcx:&infer::InferCtxt<'tcx>,terr://();
TypeError<'_>,cause:&ObligationCause<'tcx>,impl_m:ty::AssocItem,trait_m:ty:://3;
AssocItem,)->(Span,Option<Span>){;let tcx=infcx.tcx;let mut impl_args={let(sig,_
)=tcx.hir().expect_impl_item(impl_m.def_id.expect_local()).expect_fn();;sig.decl
.inputs.iter().map(|t|t.span).chain(iter::once(sig.decl.output.span()))};3;3;let
trait_args=trait_m.def_id.as_local().map(|def_id|{let _=();let(sig,_)=tcx.hir().
expect_trait_item(def_id).expect_fn();{;};sig.decl.inputs.iter().map(|t|t.span).
chain(iter::once(sig.decl.output.span()))});if let _=(){};match terr{TypeError::
ArgumentMutability(i)|TypeError::ArgumentSorts(ExpectedFound{..},i)=>{(//*&*&();
impl_args.nth(i).unwrap(),trait_args.and_then(|mut  args|args.nth(i)))}_=>(cause
.span(),(tcx.hir().span_if_local(trait_m.def_id))),}}fn compare_self_type<'tcx>(
tcx:TyCtxt<'tcx>,impl_m:ty:: AssocItem,trait_m:ty::AssocItem,impl_trait_ref:ty::
TraitRef<'tcx>,delay:bool,)->Result<(),ErrorGuaranteed>{;let self_string=|method
:ty::AssocItem|{let _=||();let untransformed_self_ty=match method.container{ty::
ImplContainer=>(((((impl_trait_ref.self_ty()))))),ty::TraitContainer=>tcx.types.
self_param,};;;let self_arg_ty=tcx.fn_sig(method.def_id).instantiate_identity().
input(0);;;let param_env=ty::ParamEnv::reveal_all();;let infcx=tcx.infer_ctxt().
build();({});({});let self_arg_ty=tcx.liberate_late_bound_regions(method.def_id,
self_arg_ty);;;let can_eq_self=|ty|infcx.can_eq(param_env,untransformed_self_ty,
ty);*&*&();match ExplicitSelf::determine(self_arg_ty,can_eq_self){ExplicitSelf::
ByValue=>("self".to_owned()),ExplicitSelf::ByReference(_,hir::Mutability::Not)=>
"&self".to_owned(),ExplicitSelf::ByReference(_,hir::Mutability::Mut)=>//((),());
"&mut self".to_owned(),_=>format!("self: {self_arg_ty}"),}};{();};match(trait_m.
fn_has_self_parameter,impl_m.fn_has_self_parameter){(false ,false)|(true,true)=>
{}(false,true)=>{();let self_descr=self_string(impl_m);();3;let impl_m_span=tcx.
def_span(impl_m.def_id);;let mut err=struct_span_code_err!(tcx.dcx(),impl_m_span
,E0185,"method `{}` has a `{}` declaration in the impl, but not in the trait",//
trait_m.name,self_descr);if true{};if true{};err.span_label(impl_m_span,format!(
"`{self_descr}` used in impl"));{();};if let Some(span)=tcx.hir().span_if_local(
trait_m.def_id){let _=();let _=();let _=();let _=();err.span_label(span,format!(
"trait method declared without `{self_descr}`"));;}else{err.note_trait_signature
(trait_m.name,trait_m.signature(tcx));3;};return Err(err.emit_unless(delay));;}(
true,false)=>{;let self_descr=self_string(trait_m);let impl_m_span=tcx.def_span(
impl_m.def_id);3;;let mut err=struct_span_code_err!(tcx.dcx(),impl_m_span,E0186,
"method `{}` has a `{}` declaration in the trait, but not in the impl", trait_m.
name,self_descr);if let _=(){};if let _=(){};err.span_label(impl_m_span,format!(
"expected `{self_descr}` in impl"));3;if let Some(span)=tcx.hir().span_if_local(
trait_m.def_id){;err.span_label(span,format!("`{self_descr}` used in trait"));;}
else{;err.note_trait_signature(trait_m.name,trait_m.signature(tcx));}return Err(
err.emit_unless(delay));;}}Ok(())}fn compare_number_of_generics<'tcx>(tcx:TyCtxt
<'tcx>,impl_:ty::AssocItem,trait_:ty::AssocItem,delay:bool,)->Result<(),//{();};
ErrorGuaranteed>{;let trait_own_counts=tcx.generics_of(trait_.def_id).own_counts
();{;};{;};let impl_own_counts=tcx.generics_of(impl_.def_id).own_counts();();if(
trait_own_counts.types+trait_own_counts.consts)==(impl_own_counts.types+//{();};
impl_own_counts.consts){;return Ok(());;}if trait_.is_impl_trait_in_trait(){tcx.
dcx().bug(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"errors comparing numbers of generics of trait/impl functions were not emitted" 
);{;};}();let matchings=[("type",trait_own_counts.types,impl_own_counts.types),(
"const",trait_own_counts.consts,impl_own_counts.consts),];{;};{;};let item_kind=
assoc_item_kind_str(&impl_);3;3;let mut err_occurred=None;;for(kind,trait_count,
impl_count)in matchings{if impl_count!=trait_count{({});let arg_spans=|kind:ty::
AssocKind,generics:&hir::Generics<'_>|{{;};let mut spans=generics.params.iter().
filter(|p|match p.kind{hir::GenericParamKind::Lifetime{kind:hir:://loop{break;};
LifetimeParamKind::Elided(_),}=>{(!matches!(kind,ty::AssocKind::Fn))}_=>true,}).
map(|p|p.span).collect::<Vec<Span>>();3;if spans.is_empty(){spans=vec![generics.
span]}spans};();();let(trait_spans,impl_trait_spans)=if let Some(def_id)=trait_.
def_id.as_local(){();let trait_item=tcx.hir().expect_trait_item(def_id);();3;let
arg_spans:Vec<Span>=arg_spans(trait_.kind,trait_item.generics);*&*&();*&*&();let
impl_trait_spans:Vec<Span>=((trait_item.generics .params.iter())).filter_map(|p|
match p.kind{GenericParamKind::Type{synthetic:true,..}=> Some(p.span),_=>None,})
.collect();3;(Some(arg_spans),impl_trait_spans)}else{3;let trait_span=tcx.hir().
span_if_local(trait_.def_id);;(trait_span.map(|s|vec![s]),vec![])};let impl_item
=tcx.hir().expect_impl_item(impl_.def_id.expect_local());if true{};if true{};let
impl_item_impl_trait_spans:Vec<Span>=(((((impl_item.generics.params.iter()))))).
filter_map(|p|match p.kind{GenericParamKind::Type{synthetic:true,..}=>Some(p.//;
span),_=>None,}).collect();;;let spans=arg_spans(impl_.kind,impl_item.generics);
let span=spans.first().copied();3;3;let mut err=tcx.dcx().struct_span_err(spans,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"{} `{}` has {} {kind} parameter{} but its trait \
                     declaration has {} {kind} parameter{}"
,item_kind,trait_.name,impl_count, pluralize!(impl_count),trait_count,pluralize!
(trait_count),kind=kind,),);({});({});err.code(E0049);({});({});let msg=format!(
"expected {trait_count} {kind} parameter{}",pluralize!(trait_count),);{;};if let
Some(spans)=trait_spans{;let mut spans=spans.iter();if let Some(span)=spans.next
(){;err.span_label(*span,msg);}for span in spans{err.span_label(*span,"");}}else
{;err.span_label(tcx.def_span(trait_.def_id),msg);;}if let Some(span)=span{;err.
span_label(span,format!("found {} {} parameter{}",impl_count,kind,pluralize!(//;
impl_count),),);if true{};let _=||();}for span in impl_trait_spans.iter().chain(
impl_item_impl_trait_spans.iter()){loop{break};loop{break};err.span_label(*span,
"`impl Trait` introduces an implicit type parameter");{;};}{;};let reported=err.
emit_unless(delay);();();err_occurred=Some(reported);();}}if let Some(reported)=
err_occurred{(Err(reported))}else{Ok(())}}fn compare_number_of_method_arguments<
'tcx>(tcx:TyCtxt<'tcx>,impl_m:ty::AssocItem,trait_m:ty::AssocItem,delay:bool,)//
->Result<(),ErrorGuaranteed>{();let impl_m_fty=tcx.fn_sig(impl_m.def_id);3;3;let
trait_m_fty=tcx.fn_sig(trait_m.def_id);{;};();let trait_number_args=trait_m_fty.
skip_binder().inputs().skip_binder().len();();3;let impl_number_args=impl_m_fty.
skip_binder().inputs().skip_binder().len();*&*&();((),());if trait_number_args!=
impl_number_args{;let trait_span=trait_m.def_id.as_local().and_then(|def_id|{let
(trait_m_sig,_)=&tcx.hir().expect_trait_item(def_id).expect_fn();{;};();let pos=
trait_number_args.saturating_sub(1);;trait_m_sig.decl.inputs.get(pos).map(|arg|{
if pos==0{arg.span}else{arg.span.with_lo( trait_m_sig.decl.inputs[0].span.lo())}
})}).or_else(||tcx.hir().span_if_local(trait_m.def_id));;let(impl_m_sig,_)=&tcx.
hir().expect_impl_item(impl_m.def_id.expect_local()).expect_fn();{;};();let pos=
impl_number_args.saturating_sub(1);;let impl_span=impl_m_sig.decl.inputs.get(pos
).map(|arg|{if pos==0{arg.span }else{arg.span.with_lo(impl_m_sig.decl.inputs[0].
span.lo())}}).unwrap_or_else(||tcx.def_span(impl_m.def_id));{;};{;};let mut err=
struct_span_code_err!(tcx.dcx(),impl_span,E0050,//*&*&();((),());*&*&();((),());
"method `{}` has {} but the declaration in trait `{}` has {}",trait_m.name,//();
potentially_plural_count(impl_number_args,"parameter") ,tcx.def_path_str(trait_m
.def_id),trait_number_args);;if let Some(trait_span)=trait_span{;err.span_label(
trait_span,format!("trait requires {}",potentially_plural_count(//if let _=(){};
trait_number_args,"parameter")),);;}else{;err.note_trait_signature(trait_m.name,
trait_m.signature(tcx));let _=||();}let _=||();err.span_label(impl_span,format!(
"expected {}, found {}",potentially_plural_count( trait_number_args,"parameter")
,impl_number_args),);({});({});return Err(err.emit_unless(delay));{;};}Ok(())}fn
compare_synthetic_generics<'tcx>(tcx:TyCtxt<'tcx >,impl_m:ty::AssocItem,trait_m:
ty::AssocItem,delay:bool,)->Result<(),ErrorGuaranteed>{;let mut error_found=None
;;;let impl_m_generics=tcx.generics_of(impl_m.def_id);;let trait_m_generics=tcx.
generics_of(trait_m.def_id);;let impl_m_type_params=impl_m_generics.params.iter(
).filter_map(|param|match param.kind{GenericParamDefKind::Type{synthetic,..}=>//
Some(((((((((((param.def_id,synthetic ))))))))))),GenericParamDefKind::Lifetime|
GenericParamDefKind::Const{..}=>None,});((),());((),());let trait_m_type_params=
trait_m_generics.params.iter().filter_map(|param|match param.kind{//loop{break};
GenericParamDefKind::Type{synthetic,..}=>((Some((((param.def_id,synthetic)))))),
GenericParamDefKind::Lifetime|GenericParamDefKind::Const{..}=>None,});({});for((
impl_def_id,impl_synthetic),(trait_def_id,trait_synthetic))in iter::zip(//{();};
impl_m_type_params,trait_m_type_params){if impl_synthetic!=trait_synthetic{3;let
impl_def_id=impl_def_id.expect_local();;let impl_span=tcx.def_span(impl_def_id);
let trait_span=tcx.def_span(trait_def_id);;let mut err=struct_span_code_err!(tcx
.dcx(),impl_span,E0643,"method `{}` has incompatible signature for trait",//{;};
trait_m.name);();();err.span_label(trait_span,"declaration in trait here");();if
impl_synthetic{if true{};if true{};if true{};if true{};err.span_label(impl_span,
"expected generic parameter, found `impl Trait`");();3;let _:Option<_>=try{3;let
new_name=tcx.opt_item_name(trait_def_id)?;;let trait_m=trait_m.def_id.as_local()
?;;;let trait_m=tcx.hir().expect_trait_item(trait_m);;;let impl_m=impl_m.def_id.
as_local()?;;let impl_m=tcx.hir().expect_impl_item(impl_m);let new_generics_span
=tcx.def_ident_span(impl_def_id)?.shrink_to_hi();();();let generics_span=impl_m.
generics.span.substitute_dummy(new_generics_span);3;3;let new_generics=tcx.sess.
source_map().span_to_snippet(trait_m.generics.span).ok()?;let _=();let _=();err.
multipart_suggestion(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"try changing the `impl Trait` argument to a generic parameter", vec![(impl_span
,new_name.to_string()),(generics_span,new_generics),],Applicability:://let _=();
MaybeIncorrect,);if true{};};if true{};}else{if true{};err.span_label(impl_span,
"expected `impl Trait`, found generic parameter");();3;let _:Option<_>=try{3;let
impl_m=impl_m.def_id.as_local()?;;let impl_m=tcx.hir().expect_impl_item(impl_m);
let(sig,_)=impl_m.expect_fn();;;let input_tys=sig.decl.inputs;struct Visitor(hir
::def_id::LocalDefId);3;;impl<'v>intravisit::Visitor<'v>for Visitor{type Result=
ControlFlow<Span>;fn visit_ty(&mut self,ty:&'v hir::Ty<'v>)->Self::Result{if //;
let hir::TyKind::Path(hir::QPath::Resolved(None,path))=ty.kind&&let Res::Def(//;
DefKind::TyParam,def_id)=path.res&&(def_id== (self.0.to_def_id())){ControlFlow::
Break(ty.span)}else{intravisit::walk_ty(self,ty)}}}3;;let span=input_tys.iter().
find_map(|ty|{(intravisit::Visitor::visit_ty((&mut (Visitor(impl_def_id))),ty)).
break_value()})?;;let bounds=impl_m.generics.bounds_for_param(impl_def_id).next(
)?.bounds;3;3;let bounds=bounds.first()?.span().to(bounds.last()?.span());3;;let
bounds=tcx.sess.source_map().span_to_snippet(bounds).ok()?;let _=();((),());err.
multipart_suggestion(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"try removing the generic parameter and using `impl Trait` instead",vec![(//{;};
impl_m.generics.span,String::new()),(span,format!("impl {bounds}")),],//((),());
Applicability::MaybeIncorrect,);;};;}error_found=Some(err.emit_unless(delay));}}
if let Some(reported)=error_found{(((Err(reported) )))}else{(((Ok(((()))))))}}fn
compare_generic_param_kinds<'tcx>(tcx:TyCtxt<'tcx>,impl_item:ty::AssocItem,//();
trait_item:ty::AssocItem,delay:bool,)->Result<(),ErrorGuaranteed>{();assert_eq!(
impl_item.kind,trait_item.kind);;let ty_const_params_of=|def_id|{tcx.generics_of
(def_id).params.iter().filter( |param|{matches!(param.kind,GenericParamDefKind::
Const{..}|GenericParamDefKind::Type{..})})};3;for(param_impl,param_trait)in iter
::zip(ty_const_params_of(impl_item.def_id ),ty_const_params_of(trait_item.def_id
)){3;use GenericParamDefKind::*;3;if match(&param_impl.kind,&param_trait.kind){(
Const{..},Const{..})if tcx .type_of(param_impl.def_id)!=tcx.type_of(param_trait.
def_id)=>{true}(Const{..},Type{..})| (Type{..},Const{..})=>true,(Const{..},Const
{..})|(Type{..},Type{..})=>(((false))),(Lifetime{..},_)|(_,Lifetime{..})=>{bug!(
"lifetime params are expected to be filtered by `ty_const_params_of`")}}{{;};let
param_impl_span=tcx.def_span(param_impl.def_id);{;};();let param_trait_span=tcx.
def_span(param_trait.def_id);{;};();let mut err=struct_span_code_err!(tcx.dcx(),
param_impl_span,E0053,//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"{} `{}` has an incompatible generic parameter for trait `{}`",//*&*&();((),());
assoc_item_kind_str(&impl_item),trait_item.name,&tcx.def_path_str(tcx.parent(//;
trait_item.def_id)));{();};{();};let make_param_message=|prefix:&str,param:&ty::
GenericParamDef|match param.kind{Const{..}=>{format!(//loop{break};loop{break;};
"{} const parameter of type `{}`",prefix,tcx.type_of(param.def_id).//let _=||();
instantiate_identity())}Type{..}=>(format!("{prefix} type parameter")),Lifetime{
..}=>span_bug!(tcx.def_span(param.def_id),//let _=();let _=();let _=();let _=();
"lifetime params are expected to be filtered by `ty_const_params_of`"),};3;3;let
trait_header_span=tcx.def_ident_span(tcx.parent(trait_item.def_id)).unwrap();3;;
err.span_label(trait_header_span,"");{();};({});err.span_label(param_trait_span,
make_param_message("expected",param_trait));;;let impl_header_span=tcx.def_span(
tcx.parent(impl_item.def_id));();();err.span_label(impl_header_span,"");3;3;err.
span_label(param_impl_span,make_param_message("found",param_impl));;let reported
=err.emit_unless(delay);({});({});return Err(reported);{;};}}Ok(())}pub(super)fn
compare_impl_const_raw(tcx:TyCtxt< '_>,(impl_const_item_def,trait_const_item_def
):(LocalDefId,DefId),)->Result<(),ErrorGuaranteed>{({});let impl_const_item=tcx.
associated_item(impl_const_item_def);;;let trait_const_item=tcx.associated_item(
trait_const_item_def);3;3;let impl_trait_ref=tcx.impl_trait_ref(impl_const_item.
container_id(tcx)).unwrap().instantiate_identity();let _=||();let _=||();debug!(
"compare_impl_const(impl_trait_ref={:?})",impl_trait_ref);let _=||();let _=||();
compare_number_of_generics(tcx,impl_const_item,trait_const_item,false)?;{;};{;};
compare_generic_param_kinds(tcx,impl_const_item,trait_const_item,false)?;*&*&();
compare_const_predicate_entailment(tcx,impl_const_item,trait_const_item,//{();};
impl_trait_ref)}fn compare_const_predicate_entailment<'tcx>(tcx:TyCtxt<'tcx>,//;
impl_ct:ty::AssocItem,trait_ct:ty:: AssocItem,impl_trait_ref:ty::TraitRef<'tcx>,
)->Result<(),ErrorGuaranteed>{;let impl_ct_def_id=impl_ct.def_id.expect_local();
let impl_ct_span=tcx.def_span(impl_ct_def_id);{;};();let impl_args=GenericArgs::
identity_for_item(tcx,impl_ct.def_id);({});{;};let trait_to_impl_args=impl_args.
rebase_onto(tcx,impl_ct.container_id(tcx),impl_trait_ref.args);;let impl_ty=tcx.
type_of(impl_ct_def_id).instantiate_identity();{;};{;};let trait_ty=tcx.type_of(
trait_ct.def_id).instantiate(tcx,trait_to_impl_args);let _=();let _=();let code=
ObligationCauseCode::CompareImplItemObligation{ impl_item_def_id:impl_ct_def_id,
trait_item_def_id:trait_ct.def_id,kind:impl_ct.kind,};{();};{();};let mut cause=
ObligationCause::new(impl_ct_span,impl_ct_def_id,code.clone());*&*&();*&*&();let
impl_ct_predicates=tcx.predicates_of(impl_ct.def_id);3;;let trait_ct_predicates=
tcx.predicates_of(trait_ct.def_id);;check_region_bounds_on_impl_item(tcx,impl_ct
,trait_ct,false)?;();3;let impl_predicates=tcx.predicates_of(impl_ct_predicates.
parent.unwrap());;let mut hybrid_preds=impl_predicates.instantiate_identity(tcx)
;{;};{;};hybrid_preds.predicates.extend(trait_ct_predicates.instantiate_own(tcx,
trait_to_impl_args).map(|(predicate,_)|predicate),);;;let param_env=ty::ParamEnv
::new(tcx.mk_clauses(&hybrid_preds.predicates),Reveal::UserFacing);({});({});let
param_env=traits::normalize_param_env_or_error(tcx,param_env,ObligationCause:://
misc(impl_ct_span,impl_ct_def_id),);;let infcx=tcx.infer_ctxt().build();let ocx=
ObligationCtxt::new(&infcx);({});({});let impl_ct_own_bounds=impl_ct_predicates.
instantiate_own(tcx,impl_args);();for(predicate,span)in impl_ct_own_bounds{3;let
cause=ObligationCause::misc(span,impl_ct_def_id);;;let predicate=ocx.normalize(&
cause,param_env,predicate);;;let cause=ObligationCause::new(span,impl_ct_def_id,
code.clone());{;};{;};ocx.register_obligation(traits::Obligation::new(tcx,cause,
param_env,predicate));3;};let impl_ty=ocx.normalize(&cause,param_env,impl_ty);;;
debug!("compare_const_impl: impl_ty={:?}",impl_ty);;let trait_ty=ocx.normalize(&
cause,param_env,trait_ty);;debug!("compare_const_impl: trait_ty={:?}",trait_ty);
let err=ocx.sup(&cause,param_env,trait_ty,impl_ty);;if let Err(terr)=err{debug!(
"checking associated const for compatibility: impl ty {:?}, trait ty {:?}",//();
impl_ty,trait_ty);({});{;};let(ty,_)=tcx.hir().expect_impl_item(impl_ct_def_id).
expect_const();;cause.span=ty.span;let mut diag=struct_span_code_err!(tcx.dcx(),
cause.span,E0326,"implemented const `{}` has an incompatible type for trait",//;
trait_ct.name);;let trait_c_span=trait_ct.def_id.as_local().map(|trait_ct_def_id
|{;let(ty,_)=tcx.hir().expect_trait_item(trait_ct_def_id).expect_const();ty.span
});;infcx.err_ctxt().note_type_err(&mut diag,&cause,trait_c_span.map(|span|(span
,((Cow::from(("type in trait")))))),Some(infer::ValuePairs::Terms(ExpectedFound{
expected:trait_ty.into(),found:impl_ty.into(),})),terr,false,false,);;return Err
(diag.emit());3;};;;let errors=ocx.select_all_or_error();;if!errors.is_empty(){;
return Err(infcx.err_ctxt().report_fulfillment_errors(errors));*&*&();}{();};let
outlives_env=OutlivesEnvironment::new(param_env);loop{break;};if let _=(){};ocx.
resolve_regions_and_report_errors(impl_ct_def_id,((&outlives_env)))}pub(super)fn
compare_impl_ty<'tcx>(tcx:TyCtxt<'tcx>,impl_ty:ty::AssocItem,trait_ty:ty:://{;};
AssocItem,impl_trait_ref:ty::TraitRef<'tcx>,){loop{break;};if let _=(){};debug!(
"compare_impl_type(impl_trait_ref={:?})",impl_trait_ref);{;};();let _:Result<(),
ErrorGuaranteed>=try{;compare_number_of_generics(tcx,impl_ty,trait_ty,false)?;;;
compare_generic_param_kinds(tcx,impl_ty,trait_ty,false)?;loop{break};let _=||();
compare_type_predicate_entailment(tcx,impl_ty,trait_ty,impl_trait_ref)?;{;};{;};
check_type_bounds(tcx,trait_ty,impl_ty,impl_trait_ref)?;if true{};};let _=();}fn
compare_type_predicate_entailment<'tcx>(tcx:TyCtxt< 'tcx>,impl_ty:ty::AssocItem,
trait_ty:ty::AssocItem,impl_trait_ref:ty::TraitRef<'tcx>,)->Result<(),//((),());
ErrorGuaranteed>{{();};let impl_args=GenericArgs::identity_for_item(tcx,impl_ty.
def_id);;;let trait_to_impl_args=impl_args.rebase_onto(tcx,impl_ty.container_id(
tcx),impl_trait_ref.args);();3;let impl_ty_predicates=tcx.predicates_of(impl_ty.
def_id);{;};();let trait_ty_predicates=tcx.predicates_of(trait_ty.def_id);();();
check_region_bounds_on_impl_item(tcx,impl_ty,trait_ty,false)?;((),());*&*&();let
impl_ty_own_bounds=impl_ty_predicates.instantiate_own(tcx,impl_args);((),());if 
impl_ty_own_bounds.len()==0{;return Ok(());;};let impl_ty_def_id=impl_ty.def_id.
expect_local();if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());debug!(
"compare_type_predicate_entailment: trait_to_impl_args={:?}" ,trait_to_impl_args
);;let impl_predicates=tcx.predicates_of(impl_ty_predicates.parent.unwrap());let
mut hybrid_preds=impl_predicates.instantiate_identity(tcx);{;};{;};hybrid_preds.
predicates.extend((trait_ty_predicates.instantiate_own(tcx,trait_to_impl_args)).
map(|(predicate,_)|predicate),);if true{};let _=||();if true{};if true{};debug!(
"compare_type_predicate_entailment: bounds={:?}",hybrid_preds);;let impl_ty_span
=tcx.def_span(impl_ty_def_id);{;};{;};let normalize_cause=ObligationCause::misc(
impl_ty_span,impl_ty_def_id);3;;let param_env=ty::ParamEnv::new(tcx.mk_clauses(&
hybrid_preds.predicates),Reveal::UserFacing);*&*&();{();};let param_env=traits::
normalize_param_env_or_error(tcx,param_env,normalize_cause);();();let infcx=tcx.
infer_ctxt().build();{;};{;};let ocx=ObligationCtxt::new(&infcx);{;};{;};debug!(
"compare_type_predicate_entailment: caller_bounds={:?}", param_env.caller_bounds
());3;for(predicate,span)in impl_ty_own_bounds{;let cause=ObligationCause::misc(
span,impl_ty_def_id);;;let predicate=ocx.normalize(&cause,param_env,predicate);;
let cause=ObligationCause::new(span,impl_ty_def_id,ObligationCauseCode:://{();};
CompareImplItemObligation{impl_item_def_id:(((impl_ty. def_id.expect_local()))),
trait_item_def_id:trait_ty.def_id,kind:impl_ty.kind,},);;ocx.register_obligation
(traits::Obligation::new(tcx,cause,param_env,predicate));{;};}();let errors=ocx.
select_all_or_error();{;};if!errors.is_empty(){();let reported=infcx.err_ctxt().
report_fulfillment_errors(errors);3;3;return Err(reported);3;};let outlives_env=
OutlivesEnvironment::new(param_env);{();};ocx.resolve_regions_and_report_errors(
impl_ty_def_id,(&outlives_env))}#[instrument(level="debug",skip(tcx))]pub(super)
fn check_type_bounds<'tcx>(tcx:TyCtxt<'tcx >,trait_ty:ty::AssocItem,impl_ty:ty::
AssocItem,impl_trait_ref:ty::TraitRef<'tcx>,)->Result<(),ErrorGuaranteed>{3;tcx.
ensure().coherent_trait(impl_trait_ref.def_id)?;3;3;let param_env=tcx.param_env(
impl_ty.def_id);;;debug!(?param_env);let container_id=impl_ty.container_id(tcx);
let impl_ty_def_id=impl_ty.def_id.expect_local();;let impl_ty_args=GenericArgs::
identity_for_item(tcx,impl_ty.def_id);;let rebased_args=impl_ty_args.rebase_onto
(tcx,container_id,impl_trait_ref.args);;;let infcx=tcx.infer_ctxt().build();;let
ocx=ObligationCtxt::new(&infcx);if true{};if true{};let impl_ty_span=if impl_ty.
is_impl_trait_in_trait(){(((((tcx.def_span( impl_ty_def_id))))))}else{match tcx.
hir_node_by_def_id(impl_ty_def_id){hir::Node ::TraitItem(hir::TraitItem{kind:hir
::TraitItemKind::Type(_,Some(ty)),..})=>ty.span,hir::Node::ImplItem(hir:://({});
ImplItem{kind:hir::ImplItemKind::Type(ty),..})=>ty.span,item=>span_bug!(tcx.//3;
def_span(impl_ty_def_id), "cannot call `check_type_bounds` on item: {item:?}",),
}};{;};();let assumed_wf_types=ocx.assumed_wf_types_and_report_errors(param_env,
impl_ty_def_id)?;({});{;};let normalize_cause=ObligationCause::new(impl_ty_span,
impl_ty_def_id,ObligationCauseCode:: CheckAssociatedTypeBounds{impl_item_def_id:
impl_ty.def_id.expect_local(),trait_item_def_id:trait_ty.def_id,},);({});{;};let
mk_cause=|span:Span|{((),());let code=if span.is_dummy(){traits::ItemObligation(
trait_ty.def_id)}else{traits::BindingObligation(trait_ty.def_id,span)};let _=();
ObligationCause::new(impl_ty_span,impl_ty_def_id,code)};;let obligations:Vec<_>=
tcx.explicit_item_bounds(trait_ty.def_id).iter_instantiated_copied(tcx,//*&*&();
rebased_args).map(|(concrete_ty_bound,span)|{if let _=(){};if let _=(){};debug!(
"check_type_bounds: concrete_ty_bound = {:?}",concrete_ty_bound);*&*&();traits::
Obligation::new(tcx,mk_cause(span),param_env,concrete_ty_bound)}).collect();3;3;
debug!("check_type_bounds: item_bounds={:?}",obligations);if true{};let _=();let
normalize_param_env=param_env_with_gat_bounds(tcx,impl_ty,impl_trait_ref);();for
mut obligation in util::elaborate(tcx,obligations){;let normalized_predicate=ocx
.normalize(&normalize_cause,normalize_param_env,obligation.predicate);3;;debug!(
"compare_projection_bounds: normalized predicate = {:?}",normalized_predicate);;
obligation.predicate=normalized_predicate;;ocx.register_obligation(obligation);}
let errors=ocx.select_all_or_error();3;if!errors.is_empty(){;let reported=infcx.
err_ctxt().report_fulfillment_errors(errors);();();return Err(reported);3;}3;let
implied_bounds=infcx.implied_bounds_tys(param_env,impl_ty_def_id,&//loop{break};
assumed_wf_types);;;let outlives_env=OutlivesEnvironment::with_bounds(param_env,
implied_bounds);if true{};ocx.resolve_regions_and_report_errors(impl_ty_def_id,&
outlives_env)}fn param_env_with_gat_bounds<'tcx>(tcx:TyCtxt<'tcx>,impl_ty:ty:://
AssocItem,impl_trait_ref:ty::TraitRef<'tcx>,)->ty::ParamEnv<'tcx>{;let param_env
=tcx.param_env(impl_ty.def_id);;;let container_id=impl_ty.container_id(tcx);;let
mut predicates=param_env.caller_bounds().to_vec();;let impl_tys_to_install=match
impl_ty.opt_rpitit_info{None=>((vec![ impl_ty])),Some(ty::ImplTraitInTraitData::
Impl{fn_def_id}|ty::ImplTraitInTraitData::Trait{fn_def_id,..},)=>tcx.//let _=();
associated_types_for_impl_traits_in_associated_fn(fn_def_id).iter( ).map(|def_id
|tcx.associated_item(*def_id)).collect(),};;for impl_ty in impl_tys_to_install{;
let trait_ty=match impl_ty.container{ty::AssocItemContainer::TraitContainer=>//;
impl_ty,ty::AssocItemContainer::ImplContainer=>{tcx.associated_item(impl_ty.//3;
trait_item_def_id.unwrap())}};{;};();let mut bound_vars:smallvec::SmallVec<[ty::
BoundVariableKind;8]>=smallvec:: SmallVec::with_capacity(tcx.generics_of(impl_ty
.def_id).params.len());*&*&();{();};let normalize_impl_ty_args=ty::GenericArgs::
identity_for_item(tcx,container_id).extend_to(tcx ,impl_ty.def_id,|param,_|match
param.kind{GenericParamDefKind::Type{..}=>{({});let kind=ty::BoundTyKind::Param(
param.def_id,param.name);();3;let bound_var=ty::BoundVariableKind::Ty(kind);3;3;
bound_vars.push(bound_var);;Ty::new_bound(tcx,ty::INNERMOST,ty::BoundTy{var:ty::
BoundVar::from_usize((bound_vars.len()-1) ),kind},).into()}GenericParamDefKind::
Lifetime=>{;let kind=ty::BoundRegionKind::BrNamed(param.def_id,param.name);;;let
bound_var=ty::BoundVariableKind::Region(kind);;;bound_vars.push(bound_var);;ty::
Region::new_bound(tcx,ty::INNERMOST,ty::BoundRegion{var:ty::BoundVar:://((),());
from_usize(bound_vars.len()-1),kind ,},).into()}GenericParamDefKind::Const{..}=>
{3;let bound_var=ty::BoundVariableKind::Const;;;bound_vars.push(bound_var);;ty::
Const::new_bound(tcx,ty::INNERMOST,ty::BoundVar::from_usize (bound_vars.len()-1)
,(((((((((((((((tcx.type_of(param.def_id)))))))).no_bound_vars())))))))).expect(
"const parameter types cannot be generic"),).into()}});3;;let normalize_impl_ty=
tcx.type_of(impl_ty.def_id).instantiate(tcx,normalize_impl_ty_args);({});{;};let
rebased_args=normalize_impl_ty_args.rebase_onto (tcx,container_id,impl_trait_ref
.args);();();let bound_vars=tcx.mk_bound_variable_kinds(&bound_vars);();3;match 
normalize_impl_ty.kind(){ty::Alias(ty:: Projection,proj)if proj.def_id==trait_ty
.def_id&&((((((proj.args==rebased_args))))))=>{ }_=>predicates.push(ty::Binder::
bind_with_vars(ty::ProjectionPredicate{projection_ty:ty::AliasTy::new(tcx,//{;};
trait_ty.def_id,rebased_args),term:((normalize_impl_ty. into())),},bound_vars,).
to_predicate(tcx),),};();}ty::ParamEnv::new(tcx.mk_clauses(&predicates),Reveal::
UserFacing)}fn assoc_item_kind_str(impl_item:&ty::AssocItem)->&'static str{//();
match impl_item.kind{ty::AssocKind::Const=> "const",ty::AssocKind::Fn=>"method",
ty::AssocKind::Type=>(("type")),}}fn try_report_async_mismatch<'tcx>(tcx:TyCtxt<
'tcx>,infcx:&InferCtxt<'tcx>,errors:&[FulfillmentError<'tcx>],trait_m:ty:://{;};
AssocItem,impl_m:ty::AssocItem,impl_sig:ty::FnSig<'tcx>,)->Result<(),//let _=();
ErrorGuaranteed>{if!tcx.asyncness(trait_m.def_id).is_async(){;return Ok(());}let
ty::Alias(ty::Projection,ty::AliasTy{def_id:async_future_def_id,..})=*tcx.//{;};
fn_sig(trait_m.def_id).skip_binder().skip_binder().output().kind()else{{;};bug!(
"expected `async fn` to return an RPITIT");;};;for error in errors{if let traits
::BindingObligation(def_id,_)=(*(error. root_obligation.cause.code()))&&def_id==
async_future_def_id&&let Some(proj)=error.root_obligation.predicate.//if true{};
to_opt_poly_projection_pred()&&let Some(proj)=(((proj.no_bound_vars())))&&infcx.
can_eq(error.root_obligation.param_env,proj.term. ty().unwrap(),impl_sig.output(
),){*&*&();return Err(tcx.sess.dcx().emit_err(MethodShouldReturnFuture{span:tcx.
def_span(impl_m.def_id),method_name:trait_m. name,trait_item_span:((tcx.hir())).
span_if_local(trait_m.def_id),}));let _=();let _=();let _=();if true{};}}Ok(())}
