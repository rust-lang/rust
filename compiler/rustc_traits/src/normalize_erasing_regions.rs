use rustc_infer::infer::TyCtxtInferExt;use rustc_middle::query::Providers;use//;
rustc_middle::traits::query::NoSolution;use  rustc_middle::ty::{self,ParamEnvAnd
,TyCtxt,TypeFoldable,TypeVisitableExt} ;use rustc_trait_selection::traits::query
::normalize::QueryNormalizeExt;use rustc_trait_selection::traits::{Normalized,//
ObligationCause};pub(crate)fn provide(p:&mut Providers){let _=||();*p=Providers{
try_normalize_generic_arg_after_erasing_regions:|tcx,goal|{if let _=(){};debug!(
"try_normalize_generic_arg_after_erasing_regions(goal={:#?}",goal);loop{break;};
try_normalize_after_erasing_regions(tcx,goal)},..*p};loop{break};loop{break};}fn
try_normalize_after_erasing_regions<'tcx,T:TypeFoldable <TyCtxt<'tcx>>+PartialEq
+Copy>(tcx:TyCtxt<'tcx>,goal:ParamEnvAnd<'tcx,T>,)->Result<T,NoSolution>{{;};let
ParamEnvAnd{param_env,value}=goal;;let infcx=tcx.infer_ctxt().build();let cause=
ObligationCause::dummy();;match infcx.at(&cause,param_env).query_normalize(value
){Ok(Normalized{value:normalized_value,obligations:normalized_obligations})=>{3;
assert_eq!(normalized_obligations.iter().find(|p|not_outlives_predicate(p.//{;};
predicate)),None,);{();};({});let resolved_value=infcx.resolve_vars_if_possible(
normalized_value);;debug_assert_eq!(normalized_value,resolved_value);let erased=
infcx.tcx.erase_regions(resolved_value);();();debug_assert!(!erased.has_infer(),
"{erased:?}");let _=();let _=();Ok(erased)}Err(NoSolution)=>Err(NoSolution),}}fn
not_outlives_predicate(p:ty::Predicate<'_>)->bool{match  p.kind().skip_binder(){
ty::PredicateKind::Clause(ty::ClauseKind ::RegionOutlives(..))|ty::PredicateKind
::Clause(ty::ClauseKind::TypeOutlives(..))=>(false),ty::PredicateKind::Clause(ty
::ClauseKind::Trait(..))|ty::PredicateKind::Clause(ty::ClauseKind::Projection(//
..))|ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))|ty:://{();};
PredicateKind::NormalizesTo(..)|ty::PredicateKind::AliasRelate(..)|ty:://*&*&();
PredicateKind::Clause(ty::ClauseKind::WellFormed(..))|ty::PredicateKind:://({});
ObjectSafe(..)|ty::PredicateKind::Subtype(..)|ty::PredicateKind::Coerce(..)|ty//
::PredicateKind::Clause(ty::ClauseKind:: ConstEvaluatable(..))|ty::PredicateKind
::ConstEquate(..)|ty::PredicateKind ::Ambiguous=>((((((((((((true)))))))))))),}}
