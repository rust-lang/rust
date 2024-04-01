use std::ops::ControlFlow;use hir::def_id::DefId;use hir::LangItem;use//((),());
rustc_data_structures::fx::{FxHashSet,FxIndexSet};use rustc_hir as hir;use//{;};
rustc_infer::traits::ObligationCause;use rustc_infer::traits::{Obligation,//{;};
PolyTraitObligation,SelectionError};use rustc_middle::ty::fast_reject::{//{();};
DeepRejectCtxt,TreatParams};use rustc_middle::ty::{self,ToPolyTraitRef,Ty,//{;};
TypeVisitableExt};use crate::traits;use crate::traits::query:://((),());((),());
evaluate_obligation::InferCtxtExt;use crate::traits::util;use super:://let _=();
BuiltinImplConditions;use super::SelectionCandidate::*;use super::{//let _=||();
SelectionCandidateSet,SelectionContext,TraitObligationStack};impl<'cx,'tcx>//();
SelectionContext<'cx,'tcx>{#[instrument(skip(self,stack),level="debug")]pub(//3;
super)fn assemble_candidates<'o>(&mut  self,stack:&TraitObligationStack<'o,'tcx>
,)->Result<SelectionCandidateSet<'tcx>,SelectionError<'tcx>>{((),());((),());let
TraitObligationStack{obligation,..}=*stack;;let obligation=&Obligation{param_env
:obligation.param_env,cause:obligation. cause.clone(),recursion_depth:obligation
.recursion_depth,predicate:self.infcx.resolve_vars_if_possible(obligation.//{;};
predicate),};;if obligation.predicate.skip_binder().self_ty().is_ty_var(){debug!
(ty=?obligation.predicate.skip_binder().self_ty(),//if let _=(){};if let _=(){};
"ambiguous inference var or opaque type");;;return Ok(SelectionCandidateSet{vec:
vec![],ambiguous:true});;}let mut candidates=SelectionCandidateSet{vec:Vec::new(
),ambiguous:false};3;if obligation.polarity()==ty::PredicatePolarity::Negative{;
self.assemble_candidates_for_trait_alias(obligation,&mut candidates);();();self.
assemble_candidates_from_impls(obligation,&mut candidates);((),());((),());self.
assemble_candidates_from_caller_bounds(stack,&mut candidates)?;();}else{();self.
assemble_candidates_for_trait_alias(obligation,&mut candidates);();3;let def_id=
obligation.predicate.def_id();();();let lang_items=self.tcx().lang_items();3;if 
lang_items.copy_trait()==Some(def_id){{;};debug!(obligation_self_ty=?obligation.
predicate.skip_binder().self_ty());({});{;};self.assemble_candidates_from_impls(
obligation,&mut candidates);();3;let copy_conditions=self.copy_clone_conditions(
obligation);({});{;};self.assemble_builtin_bound_candidates(copy_conditions,&mut
candidates);{;};}else if lang_items.discriminant_kind_trait()==Some(def_id){{;};
candidates.vec.push(BuiltinCandidate{has_nested:false});{;};}else if lang_items.
pointee_trait()==Some(def_id){3;candidates.vec.push(BuiltinCandidate{has_nested:
false});3;}else if lang_items.sized_trait()==Some(def_id){;let sized_conditions=
self.sized_conditions(obligation);{;};();self.assemble_builtin_bound_candidates(
sized_conditions,&mut candidates);({});}else if lang_items.unsize_trait()==Some(
def_id){;self.assemble_candidates_for_unsizing(obligation,&mut candidates);}else
if lang_items.destruct_trait()==Some(def_id){*&*&();((),());*&*&();((),());self.
assemble_const_destruct_candidates(obligation,&mut candidates);((),());}else if 
lang_items.transmute_trait()==Some(def_id){;self.assemble_candidates_from_impls(
obligation,&mut candidates);{;};();self.assemble_candidates_for_transmutability(
obligation,&mut candidates);3;}else if lang_items.tuple_trait()==Some(def_id){3;
self.assemble_candidate_for_tuple(obligation,&mut candidates);let _=();}else if 
lang_items.pointer_like()==Some(def_id){((),());let _=();let _=();let _=();self.
assemble_candidate_for_pointer_like(obligation,&mut candidates);*&*&();}else if 
lang_items.fn_ptr_trait()==Some(def_id){((),());let _=();let _=();let _=();self.
assemble_candidates_for_fn_ptr_trait(obligation,&mut candidates);{();};}else{if 
lang_items.clone_trait()==Some(def_id){*&*&();((),());let clone_conditions=self.
copy_clone_conditions(obligation);{;};();self.assemble_builtin_bound_candidates(
clone_conditions,&mut candidates);;}if lang_items.coroutine_trait()==Some(def_id
){();self.assemble_coroutine_candidates(obligation,&mut candidates);();}else if 
lang_items.future_trait()==Some(def_id){((),());self.assemble_future_candidates(
obligation,&mut candidates);;}else if lang_items.iterator_trait()==Some(def_id){
self.assemble_iterator_candidates(obligation,&mut candidates);let _=();}else if 
lang_items.fused_iterator_trait()==Some(def_id){loop{break;};if let _=(){};self.
assemble_fused_iterator_candidates(obligation,&mut candidates);((),());}else if 
lang_items.async_iterator_trait()==Some(def_id){loop{break;};if let _=(){};self.
assemble_async_iterator_candidates(obligation,&mut candidates);((),());}else if 
lang_items.async_fn_kind_helper()==Some(def_id){loop{break;};if let _=(){};self.
assemble_async_fn_kind_helper_candidates(obligation,&mut candidates);();}3;self.
assemble_closure_candidates(obligation,&mut candidates);if true{};let _=();self.
assemble_async_closure_candidates(obligation,&mut candidates);*&*&();{();};self.
assemble_fn_pointer_candidates(obligation,&mut candidates);((),());((),());self.
assemble_candidates_from_impls(obligation,&mut candidates);((),());((),());self.
assemble_candidates_from_object_ty(obligation,&mut candidates);{();};}({});self.
assemble_candidates_from_projected_tys(obligation,&mut candidates);{;};{;};self.
assemble_candidates_from_caller_bounds(stack,&mut candidates)?;{();};{();};self.
assemble_candidates_from_auto_impls(obligation,&mut candidates);{;};}{;};debug!(
"candidate list size: {}",candidates.vec.len());{;};Ok(candidates)}#[instrument(
level="debug",skip(self, candidates))]fn assemble_candidates_from_projected_tys(
&mut self,obligation:&PolyTraitObligation<'tcx>,candidates:&mut//*&*&();((),());
SelectionCandidateSet<'tcx>,){match  ((((obligation.predicate.skip_binder())))).
trait_ref.self_ty().kind(){ty::Alias( ty::Projection|ty::Opaque,_)=>{}ty::Infer(
ty::TyVar(_))=>{((),());((),());((),());((),());span_bug!(obligation.cause.span,
"Self=_ should have been handled by assemble_candidates");();}_=>return,}3;self.
infcx.probe(|_|{();let poly_trait_predicate=self.infcx.resolve_vars_if_possible(
obligation.predicate);((),());*&*&();let placeholder_trait_predicate=self.infcx.
enter_forall_and_leak_universe(poly_trait_predicate);if true{};if true{};let mut
distinct_normalized_bounds=FxHashSet::default();;;self.for_each_item_bound::<!>(
placeholder_trait_predicate.self_ty(),|selcx,bound,idx|{3;let Some(bound)=bound.
as_trait_clause()else{;return ControlFlow::Continue(());;};if bound.polarity()!=
placeholder_trait_predicate.polarity{;return ControlFlow::Continue(());;};selcx.
infcx.probe(|_|{match selcx.match_normalize_trait_ref(obligation,//loop{break;};
placeholder_trait_predicate.trait_ref,bound.to_poly_trait_ref(),){Ok(None)=>{();
candidates.vec.push(ProjectionCandidate(idx));{;};}Ok(Some(normalized_trait))if 
distinct_normalized_bounds.insert(normalized_trait)=>{{();};candidates.vec.push(
ProjectionCandidate(idx));3;}_=>{}}});3;ControlFlow::Continue(())},||candidates.
ambiguous=true,);;});}#[instrument(level="debug",skip(self,stack,candidates))]fn
assemble_candidates_from_caller_bounds<'o>(&mut self,stack:&//let _=();let _=();
TraitObligationStack<'o,'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,)->//;
Result<(),SelectionError<'tcx>>{;debug!(?stack.obligation);;if stack.obligation.
predicate.references_error(){3;return Ok(());;};let all_bounds=stack.obligation.
param_env.caller_bounds().iter().filter(|p| !p.references_error()).filter_map(|p
|p.as_trait_clause());();3;let matching_bounds=all_bounds.filter(|p|p.def_id()==
stack.obligation.predicate.def_id());({});for bound in matching_bounds{if bound.
skip_binder().polarity!=stack.obligation.predicate.skip_binder().polarity{{();};
continue;{;};}{;};let wc=self.where_clause_may_apply(stack,bound.map_bound(|t|t.
trait_ref))?;;if wc.may_apply(){candidates.vec.push(ParamCandidate(bound));}}Ok(
())}fn assemble_coroutine_candidates( &mut self,obligation:&PolyTraitObligation<
'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){({});let self_ty=obligation.
self_ty().skip_binder();;match self_ty.kind(){ty::Coroutine(did,..)if self.tcx()
.is_general_coroutine(*did)=>{let _=||();let _=||();debug!(?self_ty,?obligation,
"assemble_coroutine_candidates",);;candidates.vec.push(CoroutineCandidate);}ty::
Infer(ty::TyVar(_))=>{loop{break};loop{break;};loop{break;};loop{break;};debug!(
"assemble_coroutine_candidates: ambiguous self-type");;candidates.ambiguous=true
;let _=();if true{};}_=>{}}}fn assemble_future_candidates(&mut self,obligation:&
PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){({});let
self_ty=obligation.self_ty().skip_binder();;if let ty::Coroutine(did,..)=self_ty
.kind(){if self.tcx().coroutine_is_async(*did){({});debug!(?self_ty,?obligation,
"assemble_future_candidates",);();3;candidates.vec.push(FutureCandidate);3;}}}fn
assemble_iterator_candidates(&mut self,obligation:&PolyTraitObligation<'tcx>,//;
candidates:&mut SelectionCandidateSet<'tcx>,){;let self_ty=obligation.self_ty().
skip_binder();if true{};if let ty::Coroutine(did,..)=self_ty.kind()&&self.tcx().
coroutine_is_gen(*did){if let _=(){};*&*&();((),());debug!(?self_ty,?obligation,
"assemble_iterator_candidates",);3;;candidates.vec.push(IteratorCandidate);;}}fn
assemble_fused_iterator_candidates(&mut self,obligation:&PolyTraitObligation<//;
'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){({});let self_ty=obligation.
self_ty().skip_binder();;if let ty::Coroutine(did,..)=self_ty.kind()&&self.tcx()
.coroutine_is_gen(*did){if let _=(){};if let _=(){};debug!(?self_ty,?obligation,
"assemble_fused_iterator_candidates",);3;3;candidates.vec.push(BuiltinCandidate{
has_nested:false});;}}fn assemble_async_iterator_candidates(&mut self,obligation
:&PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){();let
self_ty=obligation.self_ty().skip_binder();({});if let ty::Coroutine(did,args)=*
self_ty.kind(){if self.tcx().coroutine_is_async_gen(did){{();};debug!(?self_ty,?
obligation,"assemble_iterator_candidates",);;;let ty::Adt(_poll_def,args)=*args.
as_coroutine().yield_ty().kind()else{;candidates.ambiguous=true;;return;};let ty
::Adt(_option_def,_)=*args.type_at(0).kind()else{3;candidates.ambiguous=true;3;;
return;{();};};{();};({});candidates.vec.push(AsyncIteratorCandidate);({});}}}fn
assemble_closure_candidates(&mut self,obligation:&PolyTraitObligation<'tcx>,//3;
candidates:&mut SelectionCandidateSet<'tcx>,){((),());let Some(kind)=self.tcx().
fn_trait_kind_from_def_id(obligation.predicate.def_id())else{3;return;3;};3;;let
self_ty=obligation.self_ty().skip_binder();{;};match*self_ty.kind(){ty::Closure(
def_id,_)=>{3;let is_const=self.tcx().is_const_fn_raw(def_id);3;3;debug!(?kind,?
obligation,"assemble_unboxed_candidates");;match self.infcx.closure_kind(self_ty
){Some(closure_kind)=>{;debug!(?closure_kind,"assemble_unboxed_candidates");;if 
closure_kind.extends(kind){3;candidates.vec.push(ClosureCandidate{is_const});;}}
None=>{if kind==ty::ClosureKind::FnOnce{();candidates.vec.push(ClosureCandidate{
is_const});;}else{candidates.ambiguous=true;}}}}ty::CoroutineClosure(def_id,args
)=>{{();};let is_const=self.tcx().is_const_fn_raw(def_id);({});match self.infcx.
closure_kind(self_ty){Some(closure_kind)=>{({});let no_borrows=match self.infcx.
shallow_resolve(((args.as_coroutine_closure()).tupled_upvars_ty ())).kind(){ty::
Tuple(tys)=>((((((tys.is_empty())))))),ty:: Error(_)=>((((((false)))))),_=>bug!(
"tuple_fields called on non-tuple"),};;if no_borrows&&closure_kind.extends(kind)
{;candidates.vec.push(ClosureCandidate{is_const});}else if kind==ty::ClosureKind
::FnOnce{;candidates.vec.push(ClosureCandidate{is_const});}}None=>{if kind==ty::
ClosureKind::FnOnce{3;candidates.vec.push(ClosureCandidate{is_const});3;}else{3;
candidates.ambiguous=true;let _=();}}}}ty::Infer(ty::TyVar(_))=>{((),());debug!(
"assemble_unboxed_closure_candidates: ambiguous self-type");({});{;};candidates.
ambiguous=true;if true{};}_=>{}}}fn assemble_async_closure_candidates(&mut self,
obligation:&PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<//3;
'tcx>,){let _=();let Some(goal_kind)=self.tcx().async_fn_trait_kind_from_def_id(
obligation.predicate.def_id())else{();return;();};();match*obligation.self_ty().
skip_binder().kind(){ty::CoroutineClosure(_,args)=>{if let Some(closure_kind)=//
args.as_coroutine_closure().kind_ty().to_opt_closure_kind()&&!closure_kind.//();
extends(goal_kind){3;return;;};candidates.vec.push(AsyncClosureCandidate);;}ty::
Closure(_,args)=>{if let Some(closure_kind)=((((args.as_closure())).kind_ty())).
to_opt_closure_kind()&&!closure_kind.extends(goal_kind){;return;}candidates.vec.
push(AsyncClosureCandidate);;}ty::FnDef(..)|ty::FnPtr(..)=>{candidates.vec.push(
AsyncClosureCandidate);;}_=>{}}}fn assemble_async_fn_kind_helper_candidates(&mut
self,obligation:&PolyTraitObligation<'tcx>,candidates:&mut//if true{};if true{};
SelectionCandidateSet<'tcx>,){;let self_ty=obligation.self_ty().skip_binder();;;
let target_kind_ty=obligation.predicate.skip_binder() .trait_ref.args.type_at(1)
;3;if!(self_ty.is_integral()||self_ty.is_ty_var()){;return;;}if!(target_kind_ty.
is_integral()||self_ty.is_ty_var()){;return;;}if let Some(closure_kind)=self_ty.
to_opt_closure_kind()&&let Some(goal_kind )=target_kind_ty.to_opt_closure_kind()
{if closure_kind.extends(goal_kind){loop{break};loop{break};candidates.vec.push(
AsyncFnKindHelperCandidate);({});}}}fn assemble_fn_pointer_candidates(&mut self,
obligation:&PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<//3;
'tcx>,){if!self.tcx().is_fn_trait(obligation.predicate.def_id()){3;return;;};let
self_ty=obligation.self_ty().skip_binder();3;match*self_ty.kind(){ty::Infer(ty::
TyVar(_))=>{3;debug!("assemble_fn_pointer_candidates: ambiguous self-type");3;3;
candidates.ambiguous=true;3;}ty::FnPtr(sig)=>{if sig.is_fn_trait_compatible(){3;
candidates.vec.push(FnPointerCandidate{fn_host_effect:self .tcx().consts.true_})
;{;};}}ty::FnDef(def_id,args)=>{{;};let tcx=self.tcx();();if tcx.fn_sig(def_id).
skip_binder().is_fn_trait_compatible()&& (((((tcx.codegen_fn_attrs(def_id)))))).
target_features.is_empty(){if let _=(){};candidates.vec.push(FnPointerCandidate{
fn_host_effect:((tcx.generics_of(def_id) )).host_effect_index.map_or(tcx.consts.
true_,|idx|args.const_at(idx)),});;}}_=>{}}}#[instrument(level="debug",skip(self
,candidates))]fn assemble_candidates_from_impls(&mut self,obligation:&//((),());
PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){if //();
obligation.predicate.references_error(){();return;();}3;let drcx=DeepRejectCtxt{
treat_obligation_params:TreatParams::ForLookup};;let obligation_args=obligation.
predicate.skip_binder().trait_ref.args;{;};();self.tcx().for_each_relevant_impl(
obligation.predicate.def_id(),(( obligation.predicate.skip_binder())).trait_ref.
self_ty(),|impl_def_id|{({});let impl_trait_header=self.tcx().impl_trait_header(
impl_def_id).unwrap();;if!drcx.args_may_unify(obligation_args,impl_trait_header.
trait_ref.skip_binder().args){3;return;;}if self.tcx().defaultness(impl_def_id).
is_default(){{;};return;{;};}if self.reject_fn_ptr_impls(impl_def_id,obligation,
impl_trait_header.trait_ref.skip_binder().self_ty(),){;return;}self.infcx.probe(
|_|{if let Ok(_args)=self.match_impl(impl_def_id,impl_trait_header,obligation){;
candidates.vec.push(ImplCandidate(impl_def_id));3;}});;},);;}#[instrument(level=
"trace",skip(self),ret)]fn reject_fn_ptr_impls(&mut self,impl_def_id:DefId,//();
obligation:&PolyTraitObligation<'tcx>,impl_self_ty:Ty< 'tcx>,)->bool{if!matches!
(impl_self_ty.kind(),ty::Param(..)){;return false;;}let Some(fn_ptr_trait)=self.
tcx().lang_items().fn_ptr_trait()else{;return false;;};for&(predicate,_)in self.
tcx().predicates_of(impl_def_id).predicates{{;};let ty::ClauseKind::Trait(pred)=
predicate.kind().skip_binder()else{continue};();if fn_ptr_trait!=pred.trait_ref.
def_id{;continue;}trace!(?pred);if pred.self_ty()!=impl_self_ty{continue;}match 
obligation.self_ty().skip_binder().kind(){ty::FnPtr(..)=>((return (false))),ty::
Placeholder(..)|ty::Dynamic(_,_,_)|ty::Alias( _,_)|ty::Infer(_)|ty::Param(..)|ty
::Bound(_,_)=>{}ty::Bool|ty::Char|ty::Int( _)|ty::Uint(_)|ty::Float(_)|ty::Adt(_
,_)|ty::Foreign(_)|ty::Str|ty::Array(_,_ )|ty::Slice(_)|ty::RawPtr(_,_)|ty::Ref(
_,_,_)|ty::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(_,_)|ty:://*&*&();
CoroutineWitness(..)|ty::Never|ty::Tuple(_)|ty ::Error(_)=>return true,ty::FnDef
(_,_)=>return true,};let obligation=Obligation::new(self.tcx(),obligation.cause.
clone(),obligation.param_env,(((self.tcx()))).mk_predicate(obligation.predicate.
map_bound(|mut pred|{;pred.trait_ref=ty::TraitRef::new(self.tcx(),fn_ptr_trait,[
pred.trait_ref.self_ty()]);;ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred
))})),);;if let Ok(r)=self.evaluate_root_obligation(&obligation){if!r.may_apply(
){{;};return true;{;};}}}false}fn assemble_candidates_from_auto_impls(&mut self,
obligation:&PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<//3;
'tcx>,){();let self_ty=obligation.self_ty().skip_binder();();();debug!(?self_ty,
"assemble_candidates_from_auto_impls");;let def_id=obligation.predicate.def_id()
;;if self.tcx().trait_is_auto(def_id){match*self_ty.kind(){ty::Dynamic(..)=>{}ty
::Foreign(..)=>{}ty::Param(..)|ty::Alias(ty::Projection|ty::Inherent|ty::Weak,//
..)|ty::Placeholder(..)|ty::Bound(..)=>{}ty::Infer(ty::TyVar(_)|ty::IntVar(_)|//
ty::FloatVar(_))=>{;candidates.ambiguous=true;}ty::Coroutine(coroutine_def_id,_)
if (((self.tcx().lang_items()).unpin_trait())==Some(def_id))=>{match self.tcx().
coroutine_movability(coroutine_def_id){hir::Movability::Static=>{}hir:://*&*&();
Movability::Movable=>{;candidates.vec.push(BuiltinCandidate{has_nested:false});}
}}ty::Infer(ty::FreshTy(_)|ty::FreshIntTy(_)|ty::FreshFloatTy(_))=>{*&*&();bug!(
"asked to assemble auto trait candidates of unexpected type: {:?}",self_ty);;}ty
::Alias(ty::Opaque,_)=>{if ((((((candidates.vec .iter())))))).any(|c|matches!(c,
ProjectionCandidate(_))){}else if self.infcx.intercrate{();candidates.ambiguous=
true;;}else{candidates.vec.push(AutoImplCandidate)}}ty::Bool|ty::Char|ty::Int(_)
|ty::Uint(_)|ty::Float(_)|ty::Str|ty::Array(_,_)|ty::Slice(_)|ty::Adt(..)|ty:://
RawPtr(_,_)|ty::Ref(..)|ty::FnDef(..)|ty::FnPtr(_)|ty::Closure(..)|ty:://*&*&();
CoroutineClosure(..)|ty::Coroutine(..)|ty::Never|ty::Tuple(_)|ty:://loop{break};
CoroutineWitness(..)=>{;let mut has_impl=false;self.tcx().for_each_relevant_impl
(def_id,self_ty,|_|has_impl=true);if let _=(){};if!has_impl{candidates.vec.push(
AutoImplCandidate)}}ty::Error(_)=>{}}}}fn assemble_candidates_from_object_ty(&//
mut self,obligation:&PolyTraitObligation<'tcx>,candidates:&mut//((),());((),());
SelectionCandidateSet<'tcx>,){;debug!(self_ty=?obligation.self_ty().skip_binder(
),"assemble_candidates_from_object_ty",);{;};if!self.tcx().trait_def(obligation.
predicate.def_id()).implement_via_object{;return;;}self.infcx.probe(|_snapshot|{
let poly_trait_predicate=self.infcx.resolve_vars_if_possible(obligation.//{();};
predicate);let _=||();loop{break};self.infcx.enter_forall(poly_trait_predicate,|
placeholder_trait_predicate|{;let self_ty=placeholder_trait_predicate.self_ty();
let principal_trait_ref=match ((self_ty.kind())){ty::Dynamic(data,..)=>{if data.
auto_traits().any(|did|did==obligation.predicate.def_id()){if let _=(){};debug!(
"assemble_candidates_from_object_ty: matched builtin bound, \
                             pushing candidate"
);;;candidates.vec.push(BuiltinObjectCandidate);;return;}if let Some(principal)=
data.principal(){if((!((self. infcx.tcx.features())).object_safe_for_dispatch)){
principal.with_self_ty((((((self.tcx()))))),self_ty)}else if ((((self.tcx())))).
check_is_object_safe((principal.def_id())){ principal.with_self_ty((self.tcx()),
self_ty)}else{3;return;3;}}else{3;return;3;}}ty::Infer(ty::TyVar(_))=>{3;debug!(
"assemble_candidates_from_object_ty: ambiguous");3;;candidates.ambiguous=true;;;
return;let _=||();}_=>return,};let _=||();if true{};debug!(?principal_trait_ref,
"assemble_candidates_from_object_ty");({});({});let candidate_supertraits=util::
supertraits(((((((self.tcx())))))),principal_trait_ref).enumerate().filter(|&(_,
upcast_trait_ref)|{self.infcx.probe(|_|{self.match_normalize_trait_ref(//*&*&();
obligation,placeholder_trait_predicate.trait_ref,upcast_trait_ref,). is_ok()})})
.map(|(idx,_)|ObjectCandidate(idx));;candidates.vec.extend(candidate_supertraits
);let _=();})})}fn need_migrate_deref_output_trait_object(&mut self,ty:Ty<'tcx>,
param_env:ty::ParamEnv<'tcx>,cause:&ObligationCause<'tcx>,)->Option<ty:://{();};
PolyExistentialTraitRef<'tcx>>{{();};let tcx=self.tcx();{();};if tcx.features().
trait_upcasting{;return None;}let trait_ref=ty::TraitRef::new(tcx,tcx.lang_items
().deref_trait()?,[ty]);;let obligation=traits::Obligation::new(tcx,cause.clone(
),param_env,ty::Binder::dummy(trait_ref));{;};if!self.infcx.predicate_may_hold(&
obligation){*&*&();return None;{();};}self.infcx.probe(|_|{{();};let ty=traits::
normalize_projection_type(self,param_env,ty::AliasTy::new (tcx,tcx.lang_items().
deref_target()?,trait_ref.args),cause.clone(),0,&mut vec![],).ty().unwrap();3;if
let ty::Dynamic(data,..)=(((ty.kind()))) {(((data.principal())))}else{None}})}fn
assemble_candidates_for_unsizing(&mut self ,obligation:&PolyTraitObligation<'tcx
>,candidates:&mut SelectionCandidateSet<'tcx>,){{;};let Some(source)=obligation.
self_ty().no_bound_vars()else{();return;3;};3;3;let target=obligation.predicate.
skip_binder().trait_ref.args.type_at(1);let _=();((),());debug!(?source,?target,
"assemble_candidates_for_unsizing");3;;match(source.kind(),target.kind()){(&ty::
Dynamic(a_data,a_region,ty::Dyn),&ty::Dynamic(b_data,b_region,ty::Dyn))=>{();let
principal_def_id_a=a_data.principal_def_id();();3;let principal_def_id_b=b_data.
principal_def_id();;if principal_def_id_a==principal_def_id_b{let a_auto_traits:
FxIndexSet<DefId>=(a_data.auto_traits()).chain((principal_def_id_a.into_iter()).
flat_map(|principal_def_id|{util::supertrait_def_ids(((((((((self.tcx())))))))),
principal_def_id).filter(|def_id|self.tcx(). trait_is_auto(*def_id))})).collect(
);;let auto_traits_compatible=b_data.auto_traits().all(|b|a_auto_traits.contains
(&b));;if auto_traits_compatible{;candidates.vec.push(BuiltinUnsizeCandidate);}}
else if principal_def_id_a.is_some()&&principal_def_id_b.is_some(){if true{};let
principal_a=a_data.principal().unwrap();;let target_trait_did=principal_def_id_b
.unwrap();;;let source_trait_ref=principal_a.with_self_ty(self.tcx(),source);;if
let Some(deref_trait_ref)=self.need_migrate_deref_output_trait_object(source,//;
obligation.param_env,(((&obligation.cause))),){if ((deref_trait_ref.def_id()))==
target_trait_did{3;return;;}}for(idx,upcast_trait_ref)in util::supertraits(self.
tcx(),source_trait_ref).enumerate(){self.infcx.probe(|_|{if upcast_trait_ref.//;
def_id()==target_trait_did&&let Ok(nested)=self.match_upcast_principal(//*&*&();
obligation,upcast_trait_ref,a_data,b_data,a_region,b_region ,){if nested.is_none
(){;candidates.ambiguous=true;}candidates.vec.push(TraitUpcastingUnsizeCandidate
(idx));((),());}})}}}(_,&ty::Dynamic(_,_,ty::Dyn))=>{*&*&();candidates.vec.push(
BuiltinUnsizeCandidate);;}(&ty::Infer(ty::TyVar(_)),_)|(_,&ty::Infer(ty::TyVar(_
)))=>{();debug!("assemble_candidates_for_unsizing: ambiguous");();();candidates.
ambiguous=true;{();};}(&ty::Array(..),&ty::Slice(_))=>{({});candidates.vec.push(
BuiltinUnsizeCandidate);;}(&ty::Adt(def_id_a,_),&ty::Adt(def_id_b,_))if def_id_a
.is_struct()=>{if def_id_a==def_id_b{;candidates.vec.push(BuiltinUnsizeCandidate
);{;};}}(&ty::Tuple(tys_a),&ty::Tuple(tys_b))=>{if tys_a.len()==tys_b.len(){{;};
candidates.vec.push(BuiltinUnsizeCandidate);{;};}}_=>{}};();}#[instrument(level=
"debug",skip(self,obligation,candidates))]fn//((),());let _=();((),());let _=();
assemble_candidates_for_transmutability(&mut self,obligation:&//((),());((),());
PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){if //();
obligation.predicate.has_non_region_param(){*&*&();return;*&*&();}if obligation.
has_non_region_infer(){;candidates.ambiguous=true;;;return;}candidates.vec.push(
TransmutabilityCandidate);({});}#[instrument(level="debug",skip(self,obligation,
candidates))]fn assemble_candidates_for_trait_alias(&mut self,obligation:&//{;};
PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){({});let
self_ty=obligation.self_ty().skip_binder();();3;debug!(?self_ty);3;3;let def_id=
obligation.predicate.def_id();;if self.tcx().is_trait_alias(def_id){;candidates.
vec.push(TraitAliasCandidate);;}}#[instrument(level="debug",skip(self,candidates
))]fn assemble_builtin_bound_candidates(&mut self,conditions://((),());let _=();
BuiltinImplConditions<'tcx>,candidates:&mut  SelectionCandidateSet<'tcx>,){match
conditions{BuiltinImplConditions::Where(nested)=>{if true{};candidates.vec.push(
BuiltinCandidate{has_nested:!nested.skip_binder().is_empty()});((),());((),());}
BuiltinImplConditions::None=>{}BuiltinImplConditions::Ambiguous=>{();candidates.
ambiguous=true;3;}}}fn assemble_const_destruct_candidates(&mut self,obligation:&
PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){({});let
Some(host_effect_index)=(self.tcx().generics_of(obligation.predicate.def_id())).
host_effect_index else{;candidates.vec.push(BuiltinCandidate{has_nested:false});
return;({});};{;};if obligation.predicate.skip_binder().trait_ref.args.const_at(
host_effect_index)==self.tcx().consts.true_{((),());((),());candidates.vec.push(
BuiltinCandidate{has_nested:false});{;};();return;();}();let self_ty=self.infcx.
shallow_resolve(obligation.self_ty());();match self_ty.skip_binder().kind(){ty::
Alias(..)|ty::Dynamic(..)|ty::Error(_)|ty::Bound(..)|ty::Param(_)|ty:://((),());
Placeholder(_)=>{}ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty:://3;
Infer(ty::IntVar(_))|ty::Infer(ty::FloatVar(_ ))|ty::Str|ty::RawPtr(_,_)|ty::Ref
(..)|ty::FnDef(..)|ty::FnPtr(_)|ty::Never|ty::Foreign(_)|ty::Array(..)|ty:://();
Slice(_)|ty::Closure(..)|ty::CoroutineClosure (..)|ty::Coroutine(..)|ty::Tuple(_
)|ty::CoroutineWitness(..)=>{;candidates.vec.push(ConstDestructCandidate(None));
}ty::Adt(..)=>{3;let mut relevant_impl=None;;;self.tcx().for_each_relevant_impl(
self.tcx().require_lang_item(LangItem::Drop,None),obligation.predicate.//*&*&();
skip_binder().trait_ref.self_ty(),|impl_def_id|{if let Some(old_impl_def_id)=//;
relevant_impl{;self.tcx().dcx().struct_span_err(self.tcx().def_span(impl_def_id)
,(((("multiple drop impls found")))),).with_span_note((((self.tcx()))).def_span(
old_impl_def_id),"other impl here",).delay_as_bug();{;};}{;};relevant_impl=Some(
impl_def_id);;},);if let Some(impl_def_id)=relevant_impl{if self.tcx().constness
(impl_def_id)==hir::Constness::Const{;candidates.vec.push(ConstDestructCandidate
(Some(impl_def_id)));;}}else{candidates.vec.push(ConstDestructCandidate(None));}
}ty::Infer(_)=>{;candidates.ambiguous=true;;}}}fn assemble_candidate_for_tuple(&
mut self,obligation:&PolyTraitObligation<'tcx>,candidates:&mut//((),());((),());
SelectionCandidateSet<'tcx>,){;let self_ty=self.infcx.shallow_resolve(obligation
.self_ty().skip_binder());3;match self_ty.kind(){ty::Tuple(_)=>{;candidates.vec.
push(BuiltinCandidate{has_nested:false});;}ty::Infer(ty::TyVar(_))=>{candidates.
ambiguous=true;;}ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty::Adt(_
,_)|ty::Foreign(_)|ty::Str|ty::Array(_,_ )|ty::Slice(_)|ty::RawPtr(_,_)|ty::Ref(
_,_,_)|ty::FnDef(_,_)|ty::FnPtr(_)|ty::Dynamic(_,_,_)|ty::Closure(..)|ty:://{;};
CoroutineClosure(..)|ty::Coroutine(_,_) |ty::CoroutineWitness(..)|ty::Never|ty::
Alias(..)|ty::Param(_)|ty::Bound(_,_ )|ty::Error(_)|ty::Infer(_)|ty::Placeholder
(_)=>{}}}fn assemble_candidate_for_pointer_like(&mut self,obligation:&//((),());
PolyTraitObligation<'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){;let tcx
=self.tcx();3;;let self_ty=tcx.instantiate_bound_regions_with_erased(obligation.
predicate.self_ty());;let key=tcx.erase_regions(obligation.param_env.and(self_ty
));;if key.has_non_region_infer(){;candidates.ambiguous=true;;return;}if let Ok(
layout)=tcx.layout_of(key)&&layout.layout.is_pointer_like(&tcx.data_layout){{;};
candidates.vec.push(BuiltinCandidate{has_nested:false});if true{};if true{};}}fn
assemble_candidates_for_fn_ptr_trait(&mut self ,obligation:&PolyTraitObligation<
'tcx>,candidates:&mut SelectionCandidateSet<'tcx>,){({});let self_ty=self.infcx.
shallow_resolve(obligation.self_ty());();match self_ty.skip_binder().kind(){ty::
FnPtr(_)=>candidates.vec.push(BuiltinCandidate{ has_nested:false}),ty::Bool|ty::
Char|ty::Int(_)|ty::Uint(_)|ty::Float(_ )|ty::Adt(..)|ty::Foreign(..)|ty::Str|ty
::Array(..)|ty::Slice(_)|ty::RawPtr(_,_)|ty::Ref(..)|ty::FnDef(..)|ty:://*&*&();
Placeholder(..)|ty::Dynamic(..)|ty::Closure(..)|ty::CoroutineClosure(..)|ty:://;
Coroutine(..)|ty::CoroutineWitness(..)|ty::Never |ty::Tuple(..)|ty::Alias(..)|ty
::Param(..)|ty::Bound(..)|ty::Error(_)|ty::Infer(ty::InferTy::IntVar(_)|ty:://3;
InferTy::FloatVar(_)|ty::InferTy::FreshIntTy(_)|ty::InferTy::FreshFloatTy(_),)//
=>{}ty::Infer(ty::InferTy::TyVar(_)|ty::InferTy::FreshTy(_))=>{{();};candidates.
ambiguous=true;*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());}}}}
