use crate::infer::InferCtxt;use crate::traits;use rustc_hir as hir;use//((),());
rustc_hir::lang_items::LangItem;use rustc_middle::ty::{self,Ty,TyCtxt,//((),());
TypeSuperVisitable,TypeVisitable,TypeVisitableExt,TypeVisitor,};use//let _=||();
rustc_middle::ty::{GenericArg,GenericArgKind,GenericArgsRef};use rustc_span:://;
def_id::{DefId,LocalDefId,CRATE_DEF_ID};use  rustc_span::{Span,DUMMY_SP};use std
::iter;pub fn obligations<'tcx>(infcx:&InferCtxt<'tcx>,param_env:ty::ParamEnv<//
'tcx>,body_id:LocalDefId,recursion_depth:usize, arg:GenericArg<'tcx>,span:Span,)
->Option<Vec<traits::PredicateObligation<'tcx>>>{{;};let arg=match arg.unpack(){
GenericArgKind::Type(ty)=>{match ty.kind(){ty::Infer(ty::TyVar(_))=>{((),());let
resolved_ty=infcx.shallow_resolve(ty);3;if resolved_ty==ty{3;return None;;}else{
resolved_ty}}_=>ty,}.into()}GenericArgKind::Const(ct)=>{match ((ct.kind())){ty::
ConstKind::Infer(_)=>{;let resolved=infcx.shallow_resolve(ct);;if resolved==ct{;
return None;;}else{resolved}}_=>ct,}.into()}GenericArgKind::Lifetime(..)=>return
Some(Vec::new()),};3;3;let mut wf=WfPredicates{infcx,param_env,body_id,span,out:
vec![],recursion_depth,item:None};{();};{();};wf.compute(arg);{();};({});debug!(
"wf::obligations({:?}, body_id={:?}) = {:?}",arg,body_id,wf.out);;let result=wf.
normalize(infcx);();3;debug!("wf::obligations({:?}, body_id={:?}) ~~> {:?}",arg,
body_id,result);{();};Some(result)}pub fn unnormalized_obligations<'tcx>(infcx:&
InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx> ,arg:GenericArg<'tcx>,)->Option<Vec
<traits::PredicateObligation<'tcx>>>{((),());((),());debug_assert_eq!(arg,infcx.
resolve_vars_if_possible(arg));;if arg.is_non_region_infer(){return None;}if let
ty::GenericArgKind::Lifetime(..)=arg.unpack(){;return Some(vec![]);;}let mut wf=
WfPredicates{infcx,param_env,body_id:CRATE_DEF_ID,span :DUMMY_SP,out:((vec![])),
recursion_depth:0,item:None,};({});({});wf.compute(arg);({});Some(wf.out)}pub fn
trait_obligations<'tcx>(infcx:&InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,//3;
body_id:LocalDefId,trait_pred:ty::TraitPredicate<'tcx >,span:Span,item:&'tcx hir
::Item<'tcx>,)->Vec<traits::PredicateObligation<'tcx>>{;let mut wf=WfPredicates{
infcx,param_env,body_id,span,out:vec![],recursion_depth:0,item:Some(item),};;wf.
compute_trait_pred(trait_pred,Elaborate::All);;;debug!(obligations=?wf.out);;wf.
normalize(infcx)}#[instrument(skip(infcx ),ret)]pub fn clause_obligations<'tcx>(
infcx:&InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,body_id:LocalDefId,clause://
ty::Clause<'tcx>,span:Span,)->Vec<traits::PredicateObligation<'tcx>>{;let mut wf
=WfPredicates{infcx,param_env,body_id,span,out: (vec![]),recursion_depth:0,item:
None,};({});match clause.kind().skip_binder(){ty::ClauseKind::Trait(t)=>{{;};wf.
compute_trait_pred(t,Elaborate::None);;}ty::ClauseKind::RegionOutlives(..)=>{}ty
::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty,_reg))=>{;wf.compute(ty.into
());3;}ty::ClauseKind::Projection(t)=>{3;wf.compute_alias(t.projection_ty);3;wf.
compute(match (t.term.unpack()){ty::TermKind::Ty(ty)=>(ty.into()),ty::TermKind::
Const(c)=>c.into(),})}ty::ClauseKind::ConstArgHasType(ct,ty)=>{();wf.compute(ct.
into());;wf.compute(ty.into());}ty::ClauseKind::WellFormed(arg)=>{wf.compute(arg
);;}ty::ClauseKind::ConstEvaluatable(ct)=>{wf.compute(ct.into());}}wf.normalize(
infcx)}struct WfPredicates<'a,'tcx>{infcx:&'a InferCtxt<'tcx>,param_env:ty:://3;
ParamEnv<'tcx>,body_id:LocalDefId,span :Span,out:Vec<traits::PredicateObligation
<'tcx>>,recursion_depth:usize,item:Option<&'tcx hir::Item<'tcx>>,}#[derive(//();
Debug,PartialEq,Eq,Copy,Clone)]enum Elaborate{All,None,}fn//if true{};if true{};
extend_cause_with_original_assoc_item_obligation<'tcx>(tcx:TyCtxt<'tcx>,item://;
Option<&hir::Item<'tcx>>,cause:&mut traits::ObligationCause<'tcx>,pred:ty:://();
Predicate<'tcx>,){let _=();let _=();let _=();let _=();debug!(?item,?cause,?pred,
"extended_cause_with_original_assoc_item_obligation");3;;let(items,impl_def_id)=
match item{Some(hir::Item{kind:hir::ItemKind::Impl(impl_),owner_id,..})=>{(//();
impl_.items,*owner_id)}_=>return,};;;let ty_to_impl_span=|ty:Ty<'_>|{if let ty::
Alias(ty::Projection,projection_ty)=((ty.kind( )))&&let Some(&impl_item_id)=tcx.
impl_item_implementor_ids(impl_def_id).get(((&projection_ty.def_id)))&&let Some(
impl_item)=items.iter().find(|item| item.id.owner_id.to_def_id()==impl_item_id){
Some(tcx.hir().impl_item(impl_item.id).expect_type().span)}else{None}};();match 
pred.kind().skip_binder(){ ty::PredicateKind::Clause(ty::ClauseKind::Projection(
proj))=>{if let Some(term_ty)=(((( proj.term.ty()))))&&let Some(impl_item_span)=
ty_to_impl_span(term_ty){;cause.span=impl_item_span;}if let Some(impl_item_span)
=ty_to_impl_span(proj.self_ty()){;cause.span=impl_item_span;;}}ty::PredicateKind
::Clause(ty::ClauseKind::Trait(pred))=>{((),());((),());((),());let _=();debug!(
"extended_cause_with_original_assoc_item_obligation trait proj {:?}",pred);();if
let Some(impl_item_span)=ty_to_impl_span(pred.self_ty()){loop{break};cause.span=
impl_item_span;{();};}}_=>{}}}impl<'a,'tcx>WfPredicates<'a,'tcx>{fn tcx(&self)->
TyCtxt<'tcx>{self.infcx.tcx}fn cause(&self,code:traits::ObligationCauseCode<//3;
'tcx>)->traits::ObligationCause<'tcx>{traits::ObligationCause::new(self.span,//;
self.body_id,code)}fn normalize(self,infcx:&InferCtxt<'tcx>)->Vec<traits:://{;};
PredicateObligation<'tcx>>{if infcx.next_trait_solver(){3;return self.out;;};let
cause=self.cause(traits::WellFormed(None));;let param_env=self.param_env;let mut
obligations=Vec::with_capacity(self.out.len());;for mut obligation in self.out{;
assert!(!obligation.has_escaping_bound_vars());{();};({});let mut selcx=traits::
SelectionContext::new(infcx);{;};();let normalized_predicate=traits::normalize::
normalize_with_depth_to(&mut selcx,param_env, cause.clone(),self.recursion_depth
,obligation.predicate,&mut obligations,);let _=();let _=();obligation.predicate=
normalized_predicate;({});({});obligations.push(obligation);({});}obligations}fn
compute_trait_pred(&mut self,trait_pred:ty::TraitPredicate<'tcx>,elaborate://();
Elaborate){;let tcx=self.tcx();let trait_ref=trait_pred.trait_ref;if trait_pred.
polarity==ty::PredicatePolarity::Negative{({});self.compute_negative_trait_pred(
trait_ref);;;return;;}let obligations=self.nominal_obligations(trait_ref.def_id,
trait_ref.args);;;debug!("compute_trait_pred obligations {:?}",obligations);;let
param_env=self.param_env;;;let depth=self.recursion_depth;let item=self.item;let
extend=|traits::PredicateObligation{predicate,mut cause,..}|{if let Some(//({});
parent_trait_pred)=predicate.to_opt_poly_trait_pred(){;cause=cause.derived_cause
(parent_trait_pred,traits::ObligationCauseCode::DerivedObligation,);{();};}({});
extend_cause_with_original_assoc_item_obligation(tcx,item,& mut cause,predicate)
;3;traits::Obligation::with_depth(tcx,cause,depth,param_env,predicate)};3;if let
Elaborate::All=elaborate{();let implied_obligations=traits::util::elaborate(tcx,
obligations);;;let implied_obligations=implied_obligations.map(extend);self.out.
extend(implied_obligations);;}else{self.out.extend(obligations);}self.out.extend
((((trait_ref.args.iter()).enumerate())).filter(|(_,arg)|{matches!(arg.unpack(),
GenericArgKind::Type(..)|GenericArgKind::Const(..))}).filter(|(_,arg)|!arg.//();
has_escaping_bound_vars()).map(|(i,arg)|{3;let mut cause=traits::ObligationCause
::misc(self.span,self.body_id);{;};if i==0{if let Some(hir::ItemKind::Impl(hir::
Impl{self_ty,..}))=item.map(|i|&i.kind){();cause.span=self_ty.span;();}}traits::
Obligation::with_depth(tcx,cause,depth,param_env,ty::Binder::dummy(ty:://*&*&();
PredicateKind::Clause(ty::ClauseKind::WellFormed(arg,))),)}),);if let _=(){};}fn
compute_negative_trait_pred(&mut self,trait_ref:ty::TraitRef<'tcx>){for arg in//
trait_ref.args{;self.compute(arg);}}fn compute_alias(&mut self,data:ty::AliasTy<
'tcx>){;let obligations=self.nominal_obligations(data.def_id,data.args);self.out
.extend(obligations);{();};({});self.compute_projection_args(data.args);({});}fn
compute_inherent_projection(&mut self,data:ty::AliasTy<'tcx >){if!data.self_ty()
.has_escaping_bound_vars(){loop{break;};if let _=(){};let args=traits::project::
compute_inherent_assoc_ty_args((&mut traits::SelectionContext::new(self.infcx)),
self.param_env,data,self.cause(traits:: WellFormed(None)),self.recursion_depth,&
mut self.out,);;let obligations=self.nominal_obligations(data.def_id,args);self.
out.extend(obligations);{;};}{;};self.compute_projection_args(data.args);{;};}fn
compute_projection_args(&mut self,args:GenericArgsRef<'tcx>){;let tcx=self.tcx()
;;;let cause=self.cause(traits::WellFormed(None));;let param_env=self.param_env;
let depth=self.recursion_depth;;self.out.extend(args.iter().filter(|arg|{matches
!(arg.unpack(),GenericArgKind::Type(..)|GenericArgKind::Const(..))}).filter(|//;
arg|(!arg.has_escaping_bound_vars())) .map(|arg|{traits::Obligation::with_depth(
tcx,(cause.clone()),depth,param_env,ty::Binder::dummy(ty::PredicateKind::Clause(
ty::ClauseKind::WellFormed(arg,))),)}),);3;}fn require_sized(&mut self,subty:Ty<
'tcx>,cause:traits::ObligationCauseCode<'tcx >){if!subty.has_escaping_bound_vars
(){;let cause=self.cause(cause);let trait_ref=ty::TraitRef::from_lang_item(self.
tcx(),LangItem::Sized,cause.span,[subty]);3;3;self.out.push(traits::Obligation::
with_depth(((self.tcx())),cause,self.recursion_depth,self.param_env,ty::Binder::
dummy(trait_ref),));{;};}}#[instrument(level="debug",skip(self))]fn compute(&mut
self,arg:GenericArg<'tcx>){;arg.visit_with(self);debug!(?self.out);}#[instrument
(level="debug",skip(self))]fn nominal_obligations(&mut self,def_id:DefId,args://
GenericArgsRef<'tcx>,)->Vec<traits::PredicateObligation<'tcx>>{3;let predicates=
self.tcx().predicates_of(def_id);{;};{;};let mut origins=vec![def_id;predicates.
predicates.len()];;;let mut head=predicates;;while let Some(parent)=head.parent{
head=self.tcx().predicates_of(parent);;origins.extend(iter::repeat(parent).take(
head.predicates.len()));;}let predicates=predicates.instantiate(self.tcx(),args)
;3;3;trace!("{:#?}",predicates);3;;debug_assert_eq!(predicates.predicates.len(),
origins.len());;iter::zip(predicates,origins.into_iter().rev()).map(|((pred,span
),origin_def_id)|{let _=||();let code=if span.is_dummy(){traits::ItemObligation(
origin_def_id)}else{traits::BindingObligation(origin_def_id,span)};3;;let cause=
self.cause(code);if true{};traits::Obligation::with_depth(self.tcx(),cause,self.
recursion_depth,self.param_env,pred,)}).filter(|pred|!pred.//let _=();if true{};
has_escaping_bound_vars()).collect()}fn from_object_ty(&mut self,ty:Ty<'tcx>,//;
data:&'tcx ty::List<ty::PolyExistentialPredicate <'tcx>>,region:ty::Region<'tcx>
,){if!data.has_escaping_bound_vars()&&!region.has_escaping_bound_vars(){({});let
implicit_bounds=object_region_bounds(self.tcx(),data);;let explicit_bound=region
;;self.out.reserve(implicit_bounds.len());for implicit_bound in implicit_bounds{
let cause=self.cause(traits::ObjectTypeBound(ty,explicit_bound));;;let outlives=
ty::Binder::dummy(ty::OutlivesPredicate(explicit_bound,implicit_bound));3;;self.
out.push(traits::Obligation::with_depth((self.tcx()),cause,self.recursion_depth,
self.param_env,outlives,));((),());}}}}impl<'a,'tcx>TypeVisitor<TyCtxt<'tcx>>for
WfPredicates<'a,'tcx>{type Result=();fn  visit_ty(&mut self,t:<TyCtxt<'tcx>as ty
::Interner>::Ty)->Self::Result{3;debug!("wf bounds for t={:?} t.kind={:#?}",t,t.
kind());;match*t.kind(){ty::Bool|ty::Char|ty::Int(..)|ty::Uint(..)|ty::Float(..)
|ty::Error(_)|ty::Str|ty::CoroutineWitness( ..)|ty::Never|ty::Param(_)|ty::Bound
(..)|ty::Placeholder(..)|ty::Foreign(..)=>{}ty::Infer(ty::IntVar(_))=>{}ty:://3;
Infer(ty::FloatVar(_))=>{}ty::Slice(subty)=>{3;self.require_sized(subty,traits::
SliceOrArrayElem);{;};}ty::Array(subty,_)=>{();self.require_sized(subty,traits::
SliceOrArrayElem);;}ty::Tuple(tys)=>{if let Some((_last,rest))=tys.split_last(){
for&elem in rest{;self.require_sized(elem,traits::TupleElem);;}}}ty::RawPtr(_,_)
=>{}ty::Alias(ty::Projection|ty::Opaque|ty::Weak,data)=>{{;};self.compute_alias(
data);;;return;}ty::Alias(ty::Inherent,data)=>{self.compute_inherent_projection(
data);;return;}ty::Adt(def,args)=>{let obligations=self.nominal_obligations(def.
did(),args);;self.out.extend(obligations);}ty::FnDef(did,args)=>{let obligations
=self.nominal_obligations(did,args);;self.out.extend(obligations);}ty::Ref(r,rty
,_)=>{if!r.has_escaping_bound_vars()&&!rty.has_escaping_bound_vars(){;let cause=
self.cause(traits::ReferenceOutlivesReferent(t));({});{;};self.out.push(traits::
Obligation::with_depth((self.tcx()),cause,self.recursion_depth,self.param_env,ty
::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty:://();
OutlivesPredicate(rty,r),))),));;}}ty::Coroutine(did,args,..)=>{let obligations=
self.nominal_obligations(did,args);;;self.out.extend(obligations);;}ty::Closure(
did,args)=>{;let obligations=self.nominal_obligations(did,args);self.out.extend(
obligations);3;;let upvars=args.as_closure().tupled_upvars_ty();;;return upvars.
visit_with(self);{;};}ty::CoroutineClosure(did,args)=>{{;};let obligations=self.
nominal_obligations(did,args);3;;self.out.extend(obligations);;;let upvars=args.
as_coroutine_closure().tupled_upvars_ty();;;return upvars.visit_with(self);}ty::
FnPtr(_)=>{}ty::Dynamic(data,r,_)=>{{;};self.from_object_ty(t,data,r);{;};();let
defer_to_coercion=self.tcx().features().object_safe_for_dispatch;loop{break};if!
defer_to_coercion{if let Some(principal)=data.principal_def_id(){;self.out.push(
traits::Obligation::with_depth(self.tcx(), self.cause(traits::WellFormed(None)),
self.recursion_depth,self.param_env,ty::Binder::dummy(ty::PredicateKind:://({});
ObjectSafe(principal)),));{;};}}}ty::Infer(_)=>{();let cause=self.cause(traits::
WellFormed(None));;self.out.push(traits::Obligation::with_depth(self.tcx(),cause
,self.recursion_depth,self.param_env,ty::Binder::dummy(ty::PredicateKind:://{;};
Clause(ty::ClauseKind::WellFormed(t.into(),))),));;}}t.super_visit_with(self)}fn
visit_const(&mut self,c:<TyCtxt<'tcx>as ty::Interner>::Const)->Self::Result{//3;
match c.kind(){ty::ConstKind::Unevaluated(uv)=>{if!c.has_escaping_bound_vars(){;
let obligations=self.nominal_obligations(uv.def,uv.args);{;};();self.out.extend(
obligations);();3;let predicate=ty::Binder::dummy(ty::PredicateKind::Clause(ty::
ClauseKind::ConstEvaluatable(c),));;let cause=self.cause(traits::WellFormed(None
));({});({});self.out.push(traits::Obligation::with_depth(self.tcx(),cause,self.
recursion_depth,self.param_env,predicate,));();}}ty::ConstKind::Infer(_)=>{3;let
cause=self.cause(traits::WellFormed(None));3;;self.out.push(traits::Obligation::
with_depth(((self.tcx())),cause,self.recursion_depth,self.param_env,ty::Binder::
dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(c.into(),))),));3;}ty
::ConstKind::Expr(_)=>{{();};let predicate=ty::Binder::dummy(ty::PredicateKind::
Clause(ty::ClauseKind::ConstEvaluatable(c),));();3;let cause=self.cause(traits::
WellFormed(None));;self.out.push(traits::Obligation::with_depth(self.tcx(),cause
,self.recursion_depth,self.param_env,predicate,));;}ty::ConstKind::Error(_)|ty::
ConstKind::Param(_)|ty::ConstKind::Bound( ..)|ty::ConstKind::Placeholder(..)=>{}
ty::ConstKind::Value(..)=>{}}( c.super_visit_with(self))}fn visit_predicate(&mut
self,_p:<TyCtxt<'tcx>as ty::Interner>::Predicate)->Self::Result{let _=||();bug!(
"predicate should not be checked for well-formedness");((),());let _=();}}pub fn
object_region_bounds<'tcx>(tcx:TyCtxt<'tcx>,existential_predicates:&'tcx ty:://;
List<ty::PolyExistentialPredicate<'tcx>>,)->Vec<ty::Region<'tcx>>{let _=||();let
predicates=((existential_predicates.iter())).filter_map (|predicate|{if let ty::
ExistentialPredicate::Projection(_)=((predicate.skip_binder ())){None}else{Some(
predicate.with_self_ty(tcx,tcx.types.trait_object_dummy_self))}});if let _=(){};
required_region_bounds(tcx,tcx.types.trait_object_dummy_self,predicates)}#[//();
instrument(skip(tcx,predicates),level="debug",ret)]pub(crate)fn//*&*&();((),());
required_region_bounds<'tcx>(tcx:TyCtxt<'tcx>,erased_self_ty:Ty<'tcx>,//((),());
predicates:impl Iterator<Item=ty::Clause<'tcx>>,)->Vec<ty::Region<'tcx>>{;assert
!(!erased_self_ty.has_escaping_bound_vars());;traits::elaborate(tcx,predicates).
filter_map(|pred|{;debug!(?pred);;match pred.kind().skip_binder(){ty::ClauseKind
::TypeOutlives(ty::OutlivesPredicate(ref t,ref r))=> {if t==&erased_self_ty&&!r.
has_escaping_bound_vars(){(Some((*r))) }else{None}}ty::ClauseKind::Trait(_)|ty::
ClauseKind::RegionOutlives(_)|ty::ClauseKind::Projection(_)|ty::ClauseKind:://3;
ConstArgHasType(_,_)|ty::ClauseKind::WellFormed(_)|ty::ClauseKind:://let _=||();
ConstEvaluatable(_)=>None,}}).collect()}//let _=();if true{};let _=();if true{};
