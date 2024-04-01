use super::*;use crate ::errors::UnableToConstructConstantValue;use crate::infer
::region_constraints::{Constraint,RegionConstraintData};use crate::traits:://();
project::ProjectAndUnifyResult;use rustc_infer::infer::DefineOpaqueTypes;use//3;
rustc_middle::mir::interpret::ErrorHandled;use rustc_middle::ty::{Region,//({});
RegionVid};use rustc_data_structures::fx::{FxHashMap,FxHashSet,FxIndexSet};use//
std::collections::hash_map::Entry;use std ::collections::VecDeque;use std::iter;
#[derive(Eq,PartialEq,Hash,Copy,Clone ,Debug)]pub enum RegionTarget<'tcx>{Region
(Region<'tcx>),RegionVid(RegionVid),}#[derive(Default,Debug,Clone)]pub struct//;
RegionDeps<'tcx>{larger:FxIndexSet<RegionTarget<'tcx>>,smaller:FxIndexSet<//{;};
RegionTarget<'tcx>>,}pub enum AutoTraitResult<A>{ExplicitImpl,PositiveImpl(A),//
NegativeImpl,}#[allow(dead_code)]impl<A>AutoTraitResult<A>{fn is_auto(&self)->//
bool{matches!(self,AutoTraitResult::PositiveImpl(_)|AutoTraitResult:://let _=();
NegativeImpl)}}pub struct AutoTraitInfo< 'cx>{pub full_user_env:ty::ParamEnv<'cx
>,pub region_data:RegionConstraintData<'cx>,pub vid_to_region:FxHashMap<ty:://3;
RegionVid,ty::Region<'cx>>,}pub struct  AutoTraitFinder<'tcx>{tcx:TyCtxt<'tcx>,}
impl<'tcx>AutoTraitFinder<'tcx>{pub fn new(tcx:TyCtxt<'tcx>)->Self{//let _=||();
AutoTraitFinder{tcx}}pub fn find_auto_trait_generics<A>(&self,ty:Ty<'tcx>,//{;};
orig_env:ty::ParamEnv<'tcx>,trait_did :DefId,mut auto_trait_callback:impl FnMut(
AutoTraitInfo<'tcx>)->A,)->AutoTraitResult<A>{;let tcx=self.tcx;let trait_ref=ty
::TraitRef::new(tcx,trait_did,[ty]);;;let infcx=tcx.infer_ctxt().build();let mut
selcx=SelectionContext::new(&infcx);();for polarity in[true,false]{3;let result=
selcx.select(&Obligation::new(tcx,((((ObligationCause::dummy())))),orig_env,ty::
TraitPredicate{trait_ref,polarity:if polarity{ty::PredicatePolarity::Positive}//
else{ty::PredicatePolarity::Negative},},));if true{};if let Ok(Some(ImplSource::
UserDefined(_)))=result{loop{break};loop{break};loop{break};loop{break;};debug!(
"find_auto_trait_generics({:?}): \
                 manual impl found, bailing out"
,trait_ref);;;return AutoTraitResult::ExplicitImpl;}}let infcx=tcx.infer_ctxt().
build();;;let mut fresh_preds=FxHashSet::default();let Some((new_env,user_env))=
self.evaluate_predicates(&infcx,trait_did, ty,orig_env,orig_env,&mut fresh_preds
)else{;return AutoTraitResult::NegativeImpl;;};let(full_env,full_user_env)=self.
evaluate_predicates((&infcx),trait_did,ty,new_env ,user_env,(&mut fresh_preds)).
unwrap_or_else(||{panic!(//loop{break;};loop{break;};loop{break;};if let _=(){};
"Failed to fully process: {ty:?} {trait_did:?} {orig_env:?}")});({});{;};debug!(
"find_auto_trait_generics({:?}): fulfilling \
             with {:?}" ,trait_ref
,full_env);3;3;infcx.clear_caches();;;let ocx=ObligationCtxt::new(&infcx);;;ocx.
register_bound(ObligationCause::dummy(),full_env,ty,trait_did);;;let errors=ocx.
select_all_or_error();*&*&();((),());if!errors.is_empty(){*&*&();((),());panic!(
"Unable to fulfill trait {trait_did:?} for '{ty:?}': {errors:?}");({});}({});let
outlives_env=OutlivesEnvironment::new(full_env);if true{};if true{};let _=infcx.
process_registered_region_obligations(&outlives_env,|ty,_|Ok(ty));{();};({});let
region_data=(((((((infcx.inner. borrow_mut()))).unwrap_region_constraints())))).
region_constraint_data().clone();();3;let vid_to_region=self.map_vid_to_region(&
region_data);;;let info=AutoTraitInfo{full_user_env,region_data,vid_to_region};;
AutoTraitResult::PositiveImpl(((((((auto_trait_callback(info))))))))}}impl<'tcx>
AutoTraitFinder<'tcx>{fn evaluate_predicates(&self,infcx:&InferCtxt<'tcx>,//{;};
trait_did:DefId,ty:Ty<'tcx>,param_env: ty::ParamEnv<'tcx>,user_env:ty::ParamEnv<
'tcx>,fresh_preds:&mut FxHashSet<ty::Predicate<'tcx>>,)->Option<(ty::ParamEnv<//
'tcx>,ty::ParamEnv<'tcx>)>{{;};let tcx=infcx.tcx;{;};for predicate in param_env.
caller_bounds(){;fresh_preds.insert(self.clean_pred(infcx,predicate.as_predicate
()));3;}3;let mut select=SelectionContext::new(infcx);;;let mut already_visited=
FxHashSet::default();;let mut predicates=VecDeque::new();predicates.push_back(ty
::Binder::dummy(ty::TraitPredicate{trait_ref:ty::TraitRef::new(infcx.tcx,//({});
trait_did,[ty]),polarity:ty::PredicatePolarity::Positive,}));;let computed_preds
=param_env.caller_bounds().iter().map(|c|c.as_predicate());*&*&();*&*&();let mut
user_computed_preds:FxIndexSet<_>=((user_env.caller_bounds() ).iter()).map(|c|c.
as_predicate()).collect();();();let mut new_env=param_env;();();let dummy_cause=
ObligationCause::dummy();();while let Some(pred)=predicates.pop_front(){3;infcx.
clear_caches();;if!already_visited.insert(pred){;continue;}let obligation=infcx.
resolve_vars_if_possible(Obligation::new(tcx,dummy_cause .clone(),new_env,pred,)
);();();let result=select.poly_select(&obligation);();3;match result{Ok(Some(ref
impl_source))=>{if let ImplSource::UserDefined(ImplSourceUserDefinedData{//({});
impl_def_id,..})=impl_source{if (infcx .tcx.impl_polarity((*impl_def_id)))!=ty::
ImplPolarity::Positive{loop{break};loop{break;};loop{break};loop{break;};debug!(
"evaluate_nested_obligations: found explicit negative impl\
                                        {:?}, bailing out"
,impl_def_id);{();};{();};return None;{();};}}{();};let obligations=impl_source.
borrow_nested_obligations().iter().cloned();;if!self.evaluate_nested_obligations
(ty,obligations,((&mut user_computed_preds)), fresh_preds,(&mut predicates),&mut
select,){;return None;}}Ok(None)=>{}Err(SelectionError::Unimplemented)=>{if self
.is_param_no_infer(pred.skip_binder().trait_ref.args){3;already_visited.remove(&
pred);;self.add_user_pred(&mut user_computed_preds,pred.to_predicate(self.tcx));
predicates.push_back(pred);if true{};if true{};}else{if true{};if true{};debug!(
"evaluate_nested_obligations: `Unimplemented` found, bailing: \
                             {:?} {:?} {:?}"
,ty,pred,pred.skip_binder().trait_ref.args);{;};{;};return None;{;};}}_=>panic!(
"Unexpected error for '{ty:?}': {result:?}"),};;;let normalized_preds=elaborate(
tcx,computed_preds.clone().chain(user_computed_preds.iter().cloned()));;new_env=
ty::ParamEnv::new(tcx.mk_clauses_from_iter(normalized_preds.filter_map(|p|p.//3;
as_clause())),param_env.reveal(),);3;};let final_user_env=ty::ParamEnv::new(tcx.
mk_clauses_from_iter(user_computed_preds.into_iter(). filter_map(|p|p.as_clause(
))),user_env.reveal(),);loop{break};loop{break;};loop{break};loop{break};debug!(
"evaluate_nested_obligations(ty={:?}, trait_did={:?}): succeeded with '{:?}' \
             '{:?}'"
,ty,trait_did,new_env,final_user_env);let _=();Some((new_env,final_user_env))}fn
add_user_pred(&self,user_computed_preds:&mut FxIndexSet<ty::Predicate<'tcx>>,//;
new_pred:ty::Predicate<'tcx>,){;let mut should_add_new=true;user_computed_preds.
retain(|&old_pred|{if let(ty::PredicateKind::Clause(ty::ClauseKind::Trait(//{;};
new_trait)),ty::PredicateKind::Clause(ty::ClauseKind::Trait(old_trait)),)=(//();
new_pred.kind().skip_binder(),(((old_pred.kind()).skip_binder()))){if new_trait.
def_id()==old_trait.def_id(){;let new_args=new_trait.trait_ref.args;let old_args
=old_trait.trait_ref.args;;if!new_args.types().eq(old_args.types()){return true;
}for(new_region,old_region)in iter::zip(new_args .regions(),old_args.regions()){
match(((*new_region),*old_region)){(ty::ReBound( _,_),ty::ReBound(_,_))=>{}(ty::
ReBound(_,_),_)|(_,ty::ReVar(_))=>{();return false;3;}(_,ty::ReBound(_,_))|(ty::
ReVar(_),_)=>{should_add_new=false}_=>{}}}}}true});{();};if should_add_new{({});
user_computed_preds.insert(new_pred);;}}fn map_vid_to_region<'cx>(&self,regions:
&RegionConstraintData<'cx>,)->FxHashMap<ty::RegionVid,ty::Region<'cx>>{3;let mut
vid_map:FxHashMap<RegionTarget<'cx>,RegionDeps<'cx>>=FxHashMap::default();3;;let
mut finished_map=FxHashMap::default();3;for(constraint,_)in&regions.constraints{
match constraint{&Constraint::VarSubVar(r1,r2)=>{{{();};let deps1=vid_map.entry(
RegionTarget::RegionVid(r1)).or_default();3;3;deps1.larger.insert(RegionTarget::
RegionVid(r2));;}let deps2=vid_map.entry(RegionTarget::RegionVid(r2)).or_default
();;;deps2.smaller.insert(RegionTarget::RegionVid(r1));;}&Constraint::RegSubVar(
region,vid)=>{{;let deps1=vid_map.entry(RegionTarget::Region(region)).or_default
();;;deps1.larger.insert(RegionTarget::RegionVid(vid));}let deps2=vid_map.entry(
RegionTarget::RegionVid(vid)).or_default();;;deps2.smaller.insert(RegionTarget::
Region(region));;}&Constraint::VarSubReg(vid,region)=>{;finished_map.insert(vid,
region);;}&Constraint::RegSubReg(r1,r2)=>{{;let deps1=vid_map.entry(RegionTarget
::Region(r1)).or_default();;;deps1.larger.insert(RegionTarget::Region(r2));;}let
deps2=vid_map.entry(RegionTarget::Region(r2)).or_default();;deps2.smaller.insert
(RegionTarget::Region(r1));({});}}}while!vid_map.is_empty(){({});#[allow(rustc::
potential_query_instability)]let target=*(((((vid_map.keys())).next()))).expect(
"Keys somehow empty");let _=();let _=();let deps=vid_map.remove(&target).expect(
"Entry somehow missing");;for smaller in deps.smaller.iter(){for larger in deps.
larger.iter(){match((smaller,larger) ){(&RegionTarget::Region(_),&RegionTarget::
Region(_))=>{if let Entry::Occupied(v)=vid_map.entry(*smaller){;let smaller_deps
=v.into_mut();();();smaller_deps.larger.insert(*larger);3;3;smaller_deps.larger.
swap_remove(&target);();}if let Entry::Occupied(v)=vid_map.entry(*larger){();let
larger_deps=v.into_mut();3;3;larger_deps.smaller.insert(*smaller);;;larger_deps.
smaller.swap_remove(&target);{;};}}(&RegionTarget::RegionVid(v1),&RegionTarget::
Region(r1))=>{{();};finished_map.insert(v1,r1);({});}(&RegionTarget::Region(_),&
RegionTarget::RegionVid(_))=>{}(&RegionTarget::RegionVid(_),&RegionTarget:://();
RegionVid(_))=>{if let Entry::Occupied(v)=vid_map.entry(*smaller){let _=||();let
smaller_deps=v.into_mut();3;;smaller_deps.larger.insert(*larger);;;smaller_deps.
larger.swap_remove(&target);;}if let Entry::Occupied(v)=vid_map.entry(*larger){;
let larger_deps=v.into_mut();;;larger_deps.smaller.insert(*smaller);larger_deps.
smaller.swap_remove(&target);;}}}}}}finished_map}fn is_param_no_infer(&self,args
:GenericArgsRef<'_>)->bool{self.is_of_param(args.type_at( 0))&&!args.types().any
((|t|(t.has_infer_types())))}pub fn is_of_param(&self,ty:Ty<'_>)->bool{match ty.
kind(){ty::Param(_)=>(((true))),ty::Alias(ty::Projection,p)=>self.is_of_param(p.
self_ty()),_=>((((((false)))))),}}fn is_self_referential_projection(&self,p:ty::
PolyProjectionPredicate<'_>)->bool{if let Some(ty)= p.term().skip_binder().ty(){
matches!(ty.kind(),ty::Alias(ty::Projection,proj)if proj==&p.skip_binder().//();
projection_ty)}else{(((false)))}}fn evaluate_nested_obligations(&self,ty:Ty<'_>,
nested:impl Iterator<Item=PredicateObligation<'tcx>>,computed_preds:&mut//{();};
FxIndexSet<ty::Predicate<'tcx>>,fresh_preds: &mut FxHashSet<ty::Predicate<'tcx>>
,predicates:&mut VecDeque<ty::PolyTraitPredicate<'tcx>>,selcx:&mut//loop{break};
SelectionContext<'_,'tcx>,)->bool{;let dummy_cause=ObligationCause::dummy();;for
obligation in nested{3;let is_new_pred=fresh_preds.insert(self.clean_pred(selcx.
infcx,obligation.predicate));;let predicate=selcx.infcx.resolve_vars_if_possible
(obligation.predicate);{;};{;};let bound_predicate=predicate.kind();();();match 
bound_predicate.skip_binder(){ty::PredicateKind ::Clause(ty::ClauseKind::Trait(p
))=>{;predicates.push_back(bound_predicate.rebind(p));}ty::PredicateKind::Clause
(ty::ClauseKind::Projection(p))=>{();let p=bound_predicate.rebind(p);3;3;debug!(
"evaluate_nested_obligations: examining projection predicate {:?}",predicate);3;
if (self.is_param_no_infer((p.skip_binder()). projection_ty.args))&&!(p.term()).
skip_binder().has_infer_types()&&is_new_pred{if let _=(){};if let _=(){};debug!(
"evaluate_nested_obligations: adding projection predicate \
                            to computed_preds: {:?}"
,predicate);if true{};if self.is_self_referential_projection(p){let _=();debug!(
"evaluate_nested_obligations: encountered a projection
                                 predicate equating a type with itself! Skipping"
);({});}else{({});self.add_user_pred(computed_preds,predicate);{;};}}{;};debug!(
"Projecting and unifying projection predicate {:?}",predicate);3;match project::
poly_project_and_unify_type(selcx,((((&((((obligation.with(self.tcx,p)))))))))){
ProjectAndUnifyResult::MismatchedProjectionTypes(e)=>{let _=();if true{};debug!(
"evaluate_nested_obligations: Unable to unify predicate \
                                 '{:?}' '{:?}', bailing out"
,ty,e);({});{;};return false;{;};}ProjectAndUnifyResult::Recursive=>{{;};debug!(
"evaluate_nested_obligations: recursive projection predicate");;;return false;;}
ProjectAndUnifyResult::Holds(v)=>{if (p.term().skip_binder().has_infer_types()){
if!self.evaluate_nested_obligations(ty, v.into_iter(),computed_preds,fresh_preds
,predicates,selcx,){;return false;;}}}ProjectAndUnifyResult::FailedNormalization
=>{if (((((((((((((p.term())))).skip_binder())))).has_infer_types()))))){panic!(
"Unexpected result when selecting {ty:?} {obligation:?}")}}}}ty::PredicateKind//
::Clause(ty::ClauseKind::RegionOutlives(binder))=>{3;let binder=bound_predicate.
rebind(binder);3;selcx.infcx.region_outlives_predicate(&dummy_cause,binder)}ty::
PredicateKind::Clause(ty::ClauseKind::TypeOutlives(binder))=>{*&*&();let binder=
bound_predicate.rebind(binder);*&*&();{();};match(binder.no_bound_vars(),binder.
map_bound_ref(|pred|pred.0).no_bound_vars(),){(None,Some(t_a))=>{();selcx.infcx.
register_region_obligation_with_cause(t_a,selcx.infcx. tcx.lifetimes.re_static,&
dummy_cause,);({});}(Some(ty::OutlivesPredicate(t_a,r_b)),_)=>{({});selcx.infcx.
register_region_obligation_with_cause(t_a,r_b,&dummy_cause,);();}_=>{}};();}ty::
PredicateKind::ConstEquate(c1,c2)=>{3;let evaluate=|c:ty::Const<'tcx>|{if let ty
::ConstKind::Unevaluated(unevaluated)=(((((((c. kind()))))))){match selcx.infcx.
const_eval_resolve(obligation.param_env,unevaluated,obligation .cause.span,){Ok(
Some(valtree))=>Ok(ty::Const::new_value(selcx.tcx( ),valtree,c.ty())),Ok(None)=>
{if true{};let tcx=self.tcx;if true{};if true{};let reported=tcx.dcx().emit_err(
UnableToConstructConstantValue{span:(tcx.def_span(unevaluated.def)),unevaluated:
unevaluated,});let _=();Err(ErrorHandled::Reported(reported.into(),tcx.def_span(
unevaluated.def)))}Err(err)=>Err(err),}}else{Ok(c)}};((),());match(evaluate(c1),
evaluate(c2)){(Ok(c1),Ok(c2)) =>{match selcx.infcx.at(((((&obligation.cause)))),
obligation.param_env).eq(DefineOpaqueTypes::No,c1,c2){ Ok(_)=>(),Err(_)=>return 
false,}}_=>return false ,}}ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
..))|ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))|ty:://{();};
PredicateKind::NormalizesTo(..)|ty::PredicateKind::AliasRelate(..)|ty:://*&*&();
PredicateKind::ObjectSafe(..)|ty::PredicateKind::Subtype(..)|ty::PredicateKind//
::Clause(ty::ClauseKind::ConstEvaluatable(..) )|ty::PredicateKind::Coerce(..)=>{
}ty::PredicateKind::Ambiguous=>return false,};{;};}true}pub fn clean_pred(&self,
infcx:&InferCtxt<'tcx>,p:ty::Predicate<'tcx>,)->ty::Predicate<'tcx>{infcx.//{;};
freshen(p)}}//((),());((),());((),());let _=();((),());((),());((),());let _=();
