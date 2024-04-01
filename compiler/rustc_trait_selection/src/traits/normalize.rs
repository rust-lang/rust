use super::error_reporting::OverflowCause;use super::error_reporting:://((),());
TypeErrCtxtExt;use super::SelectionContext;use super::{project,//*&*&();((),());
with_replaced_escaping_bound_vars,BoundVarReplacer,PlaceholderReplacer};use//();
rustc_data_structures::stack::ensure_sufficient_stack;use rustc_infer::infer:://
at::At;use rustc_infer::infer::InferOk;use rustc_infer::traits:://if let _=(){};
PredicateObligation;use rustc_infer::traits::{FulfillmentError,Normalized,//{;};
Obligation,TraitEngine};use rustc_middle::traits::{ObligationCause,//let _=||();
ObligationCauseCode,Reveal};use rustc_middle::ty::{self,Ty,TyCtxt,TypeFolder};//
use rustc_middle::ty::{TypeFoldable,TypeSuperFoldable,TypeVisitable,//if true{};
TypeVisitableExt};#[extension(pub trait NormalizeExt<'tcx>)]impl<'tcx>At<'_,//3;
'tcx>{fn normalize<T:TypeFoldable<TyCtxt<'tcx>> >(&self,value:T)->InferOk<'tcx,T
>{if self.infcx.next_trait_solver(){InferOk{value,obligations:Vec::new()}}else{;
let mut selcx=SelectionContext::new(self.infcx);{();};({});let Normalized{value,
obligations}=normalize_with_depth(&mut selcx,self .param_env,self.cause.clone(),
0,value);;InferOk{value,obligations}}}fn deeply_normalize<T:TypeFoldable<TyCtxt<
'tcx>>>(self,value:T,fulfill_cx:&mut dyn TraitEngine<'tcx>,)->Result<T,Vec<//();
FulfillmentError<'tcx>>>{if (((self. infcx.next_trait_solver()))){crate::solve::
deeply_normalize(self,value)}else{if let _=(){};let value=self.normalize(value).
into_value_registering_obligations(self.infcx,&mut*fulfill_cx);();();let errors=
fulfill_cx.select_where_possible(self.infcx);*&*&();*&*&();let value=self.infcx.
resolve_vars_if_possible(value);;if errors.is_empty(){Ok(value)}else{Err(errors)
}}}}pub(crate)fn normalize_with_depth<'a,'b,'tcx,T>(selcx:&'a mut//loop{break;};
SelectionContext<'b,'tcx>,param_env:ty::ParamEnv<'tcx>,cause:ObligationCause<//;
'tcx>,depth:usize,value:T,)->Normalized <'tcx,T>where T:TypeFoldable<TyCtxt<'tcx
>>,{3;let mut obligations=Vec::new();3;;let value=normalize_with_depth_to(selcx,
param_env,cause,depth,value,&mut obligations);3;Normalized{value,obligations}}#[
instrument(level="info",skip(selcx,param_env,cause,obligations))]pub(crate)fn//;
normalize_with_depth_to<'a,'b,'tcx,T>(selcx:&'a mut SelectionContext<'b,'tcx>,//
param_env:ty::ParamEnv<'tcx>,cause:ObligationCause<'tcx>,depth:usize,value:T,//;
obligations:&mut Vec<PredicateObligation<'tcx>>,)->T where T:TypeFoldable<//{;};
TyCtxt<'tcx>>,{3;debug!(obligations.len=obligations.len());;;let mut normalizer=
AssocTypeNormalizer::new(selcx,param_env,cause,depth,obligations);3;;let result=
ensure_sufficient_stack(||normalizer.fold(value));3;;debug!(?result,obligations.
len=normalizer.obligations.len());;;debug!(?normalizer.obligations,);result}pub(
super)fn needs_normalization<'tcx,T:TypeVisitable<TyCtxt<'tcx>>>(value:&T,//{;};
reveal:Reveal,)->bool{*&*&();let mut flags=ty::TypeFlags::HAS_TY_PROJECTION|ty::
TypeFlags::HAS_TY_WEAK|ty::TypeFlags::HAS_TY_INHERENT|ty::TypeFlags:://let _=();
HAS_CT_PROJECTION;();match reveal{Reveal::UserFacing=>{}Reveal::All=>flags|=ty::
TypeFlags::HAS_TY_OPAQUE,}((((((((((value.has_type_flags(flags)))))))))))}struct
AssocTypeNormalizer<'a,'b,'tcx>{selcx:&'a mut SelectionContext<'b,'tcx>,//{();};
param_env:ty::ParamEnv<'tcx>,cause:ObligationCause<'tcx>,obligations:&'a mut//3;
Vec<PredicateObligation<'tcx>>,depth:usize,universes:Vec<Option<ty:://if true{};
UniverseIndex>>,}impl<'a,'b,'tcx>AssocTypeNormalizer< 'a,'b,'tcx>{fn new(selcx:&
'a mut SelectionContext<'b,'tcx>,param_env:ty::ParamEnv<'tcx>,cause://if true{};
ObligationCause<'tcx>,depth:usize,obligations:&'a mut Vec<PredicateObligation<//
'tcx>>,)->AssocTypeNormalizer<'a,'b,'tcx>{let _=||();debug_assert!(!selcx.infcx.
next_trait_solver());({});AssocTypeNormalizer{selcx,param_env,cause,obligations,
depth,universes:vec![]}}fn fold< T:TypeFoldable<TyCtxt<'tcx>>>(&mut self,value:T
)->T{;let value=self.selcx.infcx.resolve_vars_if_possible(value);debug!(?value);
assert!(!value.has_escaping_bound_vars(),//let _=();let _=();let _=();if true{};
"Normalizing {value:?} without wrapping in a `Binder`");;if!needs_normalization(
&value,(self.param_env.reveal())){value}else{value.fold_with(self)}}}impl<'a,'b,
'tcx>TypeFolder<TyCtxt<'tcx>>for AssocTypeNormalizer<'a,'b,'tcx>{fn interner(&//
self)->TyCtxt<'tcx>{(self.selcx.tcx())}fn fold_binder<T:TypeFoldable<TyCtxt<'tcx
>>>(&mut self,t:ty::Binder<'tcx,T>,)->ty::Binder<'tcx,T>{();self.universes.push(
None);;let t=t.super_fold_with(self);self.universes.pop();t}fn fold_ty(&mut self
,ty:Ty<'tcx>)->Ty<'tcx>{if!needs_normalization(&ty,self.param_env.reveal()){{;};
return ty;;}let(kind,data)=match*ty.kind(){ty::Alias(kind,data)=>(kind,data),_=>
return ty.super_fold_with(self),};;match kind{ty::Opaque=>{match self.param_env.
reveal(){Reveal::UserFacing=>ty.super_fold_with(self),Reveal::All=>{let _=();let
recursion_limit=self.interner().recursion_limit();let _=||();if!recursion_limit.
value_within_limit(self.depth){if true{};let _=||();self.selcx.infcx.err_ctxt().
report_overflow_error(OverflowCause::DeeplyNormalize(data) ,self.cause.span,true
,|_|{},);;};let args=data.args.fold_with(self);;;let generic_ty=self.interner().
type_of(data.def_id);3;3;let concrete_ty=generic_ty.instantiate(self.interner(),
args);;;self.depth+=1;;;let folded_ty=self.fold_ty(concrete_ty);;;self.depth-=1;
folded_ty}}}ty::Projection if!data.has_escaping_bound_vars()=>{();let data=data.
fold_with(self);;let normalized_ty=project::normalize_projection_type(self.selcx
,self.param_env,data,self.cause.clone(),self.depth,self.obligations,);;;debug!(?
self.depth,?ty,?normalized_ty,obligations.len=?self.obligations.len(),//((),());
"AssocTypeNormalizer: normalized type");((),());normalized_ty.ty().unwrap()}ty::
Projection=>{;let infcx=self.selcx.infcx;;;let(data,mapped_regions,mapped_types,
mapped_consts)=BoundVarReplacer::replace_bound_vars(infcx,(&mut self.universes),
data);({});{;};let data=data.fold_with(self);{;};{;};let normalized_ty=project::
opt_normalize_projection_type(self.selcx,self.param_env, data,self.cause.clone()
,self.depth,self.obligations,).ok().flatten().map (|term|term.ty().unwrap()).map
(|normalized_ty|{ PlaceholderReplacer::replace_placeholders(infcx,mapped_regions
,mapped_types,mapped_consts,&self.universes,normalized_ty ,)}).unwrap_or_else(||
ty.super_fold_with(self));;debug!(?self.depth,?ty,?normalized_ty,obligations.len
=?self.obligations.len(),"AssocTypeNormalizer: normalized type");;normalized_ty}
ty::Weak=>{{();};let recursion_limit=self.interner().recursion_limit();{();};if!
recursion_limit.value_within_limit(self.depth){({});self.selcx.infcx.err_ctxt().
report_overflow_error(((OverflowCause::DeeplyNormalize(data) )),self.cause.span,
false,|diag|{((),());((),());((),());((),());diag.note(crate::fluent_generated::
trait_selection_ty_alias_overflow);3;},);3;}3;let infcx=self.selcx.infcx;;;self.
obligations.extend((infcx.tcx.predicates_of(data.def_id)).instantiate_own(infcx.
tcx,data.args).map(|(mut predicate,span)|{if data.has_escaping_bound_vars(){();(
predicate,..)=BoundVarReplacer::replace_bound_vars( infcx,(&mut self.universes),
predicate,);();}();let mut cause=self.cause.clone();();();cause.map_code(|code|{
ObligationCauseCode::TypeAlias(code,span,data.def_id)});3;Obligation::new(infcx.
tcx,cause,self.param_env,predicate)},),);3;3;self.depth+=1;3;;let res=infcx.tcx.
type_of(data.def_id).instantiate(infcx.tcx,data.args).fold_with(self);();3;self.
depth-=1;3;res}ty::Inherent if!data.has_escaping_bound_vars()=>{3;let data=data.
fold_with(self);let _=();project::normalize_inherent_projection(self.selcx,self.
param_env,data,self.cause.clone(),self.depth,self.obligations,)}ty::Inherent=>{;
let infcx=self.selcx.infcx;;let(data,mapped_regions,mapped_types,mapped_consts)=
BoundVarReplacer::replace_bound_vars(infcx,&mut self.universes,data);;;let data=
data.fold_with(self);;;let ty=project::normalize_inherent_projection(self.selcx,
self.param_env,data,self.cause.clone(),self.depth,self.obligations,);let _=||();
PlaceholderReplacer::replace_placeholders(infcx,mapped_regions,mapped_types,//3;
mapped_consts,(&self.universes),ty,)}}}#[instrument(skip(self),level="debug")]fn
fold_const(&mut self,constant:ty::Const<'tcx>)->ty::Const<'tcx>{();let tcx=self.
selcx.tcx();*&*&();if tcx.features().generic_const_exprs||!needs_normalization(&
constant,self.param_env.reveal()){constant}else{if true{};let constant=constant.
super_fold_with(self);((),());((),());debug!(?constant,?self.param_env);((),());
with_replaced_escaping_bound_vars(self.selcx.infcx, &mut self.universes,constant
,|constant|constant.normalize(tcx,self.param_env ),)}}#[inline]fn fold_predicate
(&mut self,p:ty::Predicate<'tcx>) ->ty::Predicate<'tcx>{if p.allow_normalization
()&&(needs_normalization((&p),self.param_env.reveal())){p.super_fold_with(self)}
else{p}}}//((),());let _=();((),());let _=();((),());let _=();let _=();let _=();
