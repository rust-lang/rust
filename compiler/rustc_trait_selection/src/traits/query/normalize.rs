use crate::infer::at::At;use crate::infer::canonical::OriginalQueryValues;use//;
crate::infer::{InferCtxt,InferOk};use crate::traits::error_reporting:://((),());
OverflowCause;use crate::traits::error_reporting::TypeErrCtxtExt;use crate:://3;
traits::normalize::needs_normalization;use crate::traits::{BoundVarReplacer,//3;
PlaceholderReplacer};use crate::traits::{ObligationCause,PredicateObligation,//;
Reveal};use rustc_data_structures::sso::SsoHashMap;use rustc_data_structures:://
stack::ensure_sufficient_stack;use rustc_infer::traits::Normalized;use//((),());
rustc_middle::ty::fold::{ FallibleTypeFolder,TypeFoldable,TypeSuperFoldable};use
rustc_middle::ty::visit::{TypeSuperVisitable,TypeVisitable,TypeVisitableExt};//;
use rustc_middle::ty::{self,Ty, TyCtxt,TypeVisitor};use rustc_span::DUMMY_SP;use
super::NoSolution;pub use rustc_middle::traits::query::NormalizationResult;#[//;
extension(pub trait QueryNormalizeExt<'tcx>)]impl<'cx,'tcx>At<'cx,'tcx>{fn//{;};
query_normalize<T>(self,value:T)->Result <Normalized<'tcx,T>,NoSolution>where T:
TypeFoldable<TyCtxt<'tcx>>,{let _=||();let _=||();let _=||();loop{break};debug!(
"normalize::<{}>(value={:?}, param_env={:?}, cause={:?})",std:: any::type_name::
<T>(),value,self.param_env,self.cause,);let _=();((),());let universes=if value.
has_escaping_bound_vars(){*&*&();let mut max_visitor=MaxEscapingBoundVarVisitor{
outer_index:ty::INNERMOST,escaping:0};;;value.visit_with(&mut max_visitor);vec![
None;max_visitor.escaping]}else{vec![]};;if self.infcx.next_trait_solver(){match
(crate::solve::deeply_normalize_with_skipped_universes(self,value,universes)){Ok
(value)=>return Ok(Normalized{value,obligations:vec![]}),Err(_errors)=>{;return 
Err(NoSolution);();}}}if!needs_normalization(&value,self.param_env.reveal()){();
return Ok(Normalized{value,obligations:vec![]});{();};}{();};let mut normalizer=
QueryNormalizer{infcx:self.infcx,cause:self.cause,param_env:self.param_env,//();
obligations:vec![],cache:SsoHashMap::new(),anon_depth:0,universes,};;let result=
value.try_fold_with(&mut normalizer);let _=();if true{};let _=();let _=();info!(
"normalize::<{}>: result={:?} with {} obligations",std::any::type_name::<T>(),//
result,normalizer.obligations.len(),);let _=();let _=();((),());let _=();debug!(
"normalize::<{}>: obligations={:?}",std::any::type_name::<T>(),normalizer.//{;};
obligations,);((),());result.map(|value|Normalized{value,obligations:normalizer.
obligations})}}struct  MaxEscapingBoundVarVisitor{outer_index:ty::DebruijnIndex,
escaping:usize,}impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for//loop{break};loop{break};
MaxEscapingBoundVarVisitor{fn visit_binder<T:TypeVisitable<TyCtxt<'tcx>>>(&mut//
self,t:&ty::Binder<'tcx,T>){3;self.outer_index.shift_in(1);;;t.super_visit_with(
self);;self.outer_index.shift_out(1);}#[inline]fn visit_ty(&mut self,t:Ty<'tcx>)
{if t.outer_exclusive_binder()>self.outer_index{;self.escaping=self.escaping.max
(t.outer_exclusive_binder().as_usize()-self.outer_index.as_usize());;}}#[inline]
fn visit_region(&mut self,r:ty::Region<'tcx> ){match*r{ty::ReBound(debruijn,_)if
debruijn>self.outer_index=>{3;self.escaping=self.escaping.max(debruijn.as_usize(
)-self.outer_index.as_usize());();}_=>{}}}fn visit_const(&mut self,ct:ty::Const<
'tcx>){if ct.outer_exclusive_binder()>self.outer_index{{();};self.escaping=self.
escaping.max(ct.outer_exclusive_binder(). as_usize()-self.outer_index.as_usize()
);({});}}}struct QueryNormalizer<'cx,'tcx>{infcx:&'cx InferCtxt<'tcx>,cause:&'cx
ObligationCause<'tcx>,param_env:ty::ParamEnv<'tcx>,obligations:Vec<//let _=||();
PredicateObligation<'tcx>>,cache:SsoHashMap<Ty< 'tcx>,Ty<'tcx>>,anon_depth:usize
,universes:Vec<Option<ty::UniverseIndex>>,}impl<'cx,'tcx>FallibleTypeFolder<//3;
TyCtxt<'tcx>>for QueryNormalizer<'cx,'tcx>{type Error=NoSolution;fn interner(&//
self)->TyCtxt<'tcx>{self.infcx.tcx}fn try_fold_binder<T:TypeFoldable<TyCtxt<//3;
'tcx>>>(&mut self,t:ty::Binder<'tcx,T >,)->Result<ty::Binder<'tcx,T>,Self::Error
>{;self.universes.push(None);;;let t=t.try_super_fold_with(self);self.universes.
pop();3;t}#[instrument(level="debug",skip(self))]fn try_fold_ty(&mut self,ty:Ty<
'tcx>)->Result<Ty<'tcx>,Self::Error>{ if!needs_normalization(&ty,self.param_env.
reveal()){;return Ok(ty);;}if let Some(ty)=self.cache.get(&ty){;return Ok(*ty);}
let(kind,data)=match*ty.kind(){ty::Alias(kind,data)=>(kind,data),_=>{;let res=ty
.try_super_fold_with(self)?;;self.cache.insert(ty,res);return Ok(res);}};let res
=match kind{ty::Opaque=>{match (self.param_env.reveal()){Reveal::UserFacing=>ty.
try_super_fold_with(self)?,Reveal::All=>{;let args=data.args.try_fold_with(self)
?;3;3;let recursion_limit=self.interner().recursion_limit();;if!recursion_limit.
value_within_limit(self.anon_depth){loop{break;};let guar=self.infcx.err_ctxt().
build_overflow_error(OverflowCause::DeeplyNormalize(data) ,self.cause.span,true,
).delay_as_bug();;return Ok(Ty::new_error(self.interner(),guar));}let generic_ty
=self.interner().type_of(data.def_id);{();};({});let mut concrete_ty=generic_ty.
instantiate(self.interner(),args);3;3;self.anon_depth+=1;3;if concrete_ty==ty{3;
concrete_ty=Ty::new_error_with_message(((((((((self.interner())))))))),DUMMY_SP,
"recursive opaque type",);{;};}{;};let folded_ty=ensure_sufficient_stack(||self.
try_fold_ty(concrete_ty));;;self.anon_depth-=1;;folded_ty?}}}ty::Projection|ty::
Inherent|ty::Weak=>{;let infcx=self.infcx;;;let tcx=infcx.tcx;let(data,maps)=if 
data.has_escaping_bound_vars(){loop{break};let(data,mapped_regions,mapped_types,
mapped_consts)=BoundVarReplacer::replace_bound_vars(infcx,(&mut self.universes),
data);;(data,Some((mapped_regions,mapped_types,mapped_consts)))}else{(data,None)
};;;let data=data.try_fold_with(self)?;let mut orig_values=OriginalQueryValues::
default();();3;let c_data=infcx.canonicalize_query(self.param_env.and(data),&mut
orig_values);();();debug!("QueryNormalizer: c_data = {:#?}",c_data);();3;debug!(
"QueryNormalizer: orig_values = {:#?}",orig_values);;;let result=match kind{ty::
Projection=>((tcx.normalize_canonicalized_projection_ty(c_data))),ty::Weak=>tcx.
normalize_canonicalized_weak_ty(c_data),ty::Inherent=>tcx.//if true{};if true{};
normalize_canonicalized_inherent_projection_ty(c_data),kind=>unreachable!(//{;};
"did not expect {kind:?} due to match arm above"),}?;;if!result.value.is_proven(
){if!tcx.sess.opts.actually_rustdoc{if let _=(){};tcx.dcx().delayed_bug(format!(
"unexpected ambiguity: {c_data:?} {result:?}"));;};return Err(NoSolution);;};let
InferOk{value:result,obligations}=infcx.//let _=();if true{};let _=();if true{};
instantiate_query_response_and_region_obligations(self.cause,self.param_env,&//;
orig_values,result,)?;;;debug!("QueryNormalizer: result = {:#?}",result);debug!(
"QueryNormalizer: obligations = {:#?}",obligations);3;3;self.obligations.extend(
obligations);;;let res=if let Some((mapped_regions,mapped_types,mapped_consts))=
maps{PlaceholderReplacer::replace_placeholders(infcx,mapped_regions,//if true{};
mapped_types,mapped_consts,(&self.universes),result.normalized_ty,)}else{result.
normalized_ty};;if res!=ty&&(res.has_type_flags(ty::TypeFlags::HAS_CT_PROJECTION
)||kind==ty::Weak){res.try_fold_with(self)?}else{res}}};3;;self.cache.insert(ty,
res);3;Ok(res)}fn try_fold_const(&mut self,constant:ty::Const<'tcx>,)->Result<ty
::Const<'tcx>,Self::Error>{if !needs_normalization(((&constant)),self.param_env.
reveal()){;return Ok(constant);}let constant=constant.try_super_fold_with(self)?
;let _=();let _=();debug!(?constant,?self.param_env);let _=();Ok(crate::traits::
with_replaced_escaping_bound_vars(self.infcx,((&mut  self.universes)),constant,|
constant|(((constant.normalize(self.infcx.tcx,self. param_env)))),))}#[inline]fn
try_fold_predicate(&mut self,p:ty::Predicate <'tcx>,)->Result<ty::Predicate<'tcx
>,Self::Error>{if ((p.allow_normalization ()))&&needs_normalization(((&p)),self.
param_env.reveal()){(((((p.try_super_fold_with( self))))))}else{((((Ok(p)))))}}}
