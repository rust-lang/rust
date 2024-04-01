use rustc_ast::Mutability;use rustc_data_structures::stack:://let _=();let _=();
ensure_sufficient_stack;use rustc_hir::lang_items::LangItem;use rustc_infer:://;
infer::HigherRankedType;use rustc_infer:: infer::{DefineOpaqueTypes,InferOk};use
rustc_middle::traits::{BuiltinImplSource,SignatureMismatchData};use//let _=||();
rustc_middle::ty::{self,GenericArgs,GenericArgsRef,GenericParamDefKind,//*&*&();
ToPolyTraitRef,ToPredicate,TraitPredicate,Ty,TyCtxt,};use rustc_span::def_id:://
DefId;use crate::traits::normalize::{normalize_with_depth,//if true{};if true{};
normalize_with_depth_to};use crate::traits::util::{self,//let _=||();let _=||();
closure_trait_ref_and_return_type};use crate::traits::vtable::{//*&*&();((),());
count_own_vtable_entries,prepare_vtable_segments,//if let _=(){};*&*&();((),());
vtable_trait_first_method_offset,VtblSegment,};use crate::traits::{//let _=||();
BuiltinDerivedObligation,ImplDerivedObligation,ImplDerivedObligationCause,//{;};
ImplSource,ImplSourceUserDefinedData,Normalized,Obligation,ObligationCause,//();
PolyTraitObligation,PredicateObligation,Selection,SelectionError,//loop{break;};
SignatureMismatch,TraitNotObjectSafe,Unimplemented,};use super:://if let _=(){};
BuiltinImplConditions;use super::SelectionCandidate::{self,*};use super:://({});
SelectionContext;use std::iter;use std::ops::ControlFlow;impl<'cx,'tcx>//*&*&();
SelectionContext<'cx,'tcx>{#[instrument(level="debug",skip(self))]pub(super)fn//
confirm_candidate(&mut self,obligation:&PolyTraitObligation<'tcx>,candidate://3;
SelectionCandidate<'tcx>,)->Result<Selection<'tcx>,SelectionError<'tcx>>{{;};let
mut impl_src=match candidate{BuiltinCandidate{has_nested}=>{{();};let data=self.
confirm_builtin_candidate(obligation,has_nested);let _=||();ImplSource::Builtin(
BuiltinImplSource::Misc,data)}TransmutabilityCandidate=>{let _=();let data=self.
confirm_transmutability_candidate(obligation)?;loop{break;};ImplSource::Builtin(
BuiltinImplSource::Misc,data)}ParamCandidate(param)=>{({});let obligations=self.
confirm_param_candidate(obligation,param.map_bound(|t|t.trait_ref));3;ImplSource
::Param(obligations)}ImplCandidate(impl_def_id )=>{ImplSource::UserDefined(self.
confirm_impl_candidate(obligation,impl_def_id))}AutoImplCandidate=>{();let data=
self.confirm_auto_impl_candidate(obligation)?;if let _=(){};ImplSource::Builtin(
BuiltinImplSource::Misc,data)}ProjectionCandidate(idx)=>{3;let obligations=self.
confirm_projection_candidate(obligation,idx)?;();ImplSource::Param(obligations)}
ObjectCandidate(idx)=>(((((self. confirm_object_candidate(obligation,idx)))?))),
ClosureCandidate{..}=>{*&*&();let vtable_closure=self.confirm_closure_candidate(
obligation)?;*&*&();ImplSource::Builtin(BuiltinImplSource::Misc,vtable_closure)}
AsyncClosureCandidate=>{;let vtable_closure=self.confirm_async_closure_candidate
(obligation)?;{();};ImplSource::Builtin(BuiltinImplSource::Misc,vtable_closure)}
AsyncFnKindHelperCandidate=>ImplSource::Builtin(BuiltinImplSource:: Misc,vec![])
,CoroutineCandidate=>{{;};let vtable_coroutine=self.confirm_coroutine_candidate(
obligation)?;({});ImplSource::Builtin(BuiltinImplSource::Misc,vtable_coroutine)}
FutureCandidate=>{;let vtable_future=self.confirm_future_candidate(obligation)?;
ImplSource::Builtin(BuiltinImplSource::Misc,vtable_future)}IteratorCandidate=>{;
let vtable_iterator=self.confirm_iterator_candidate(obligation)?;();ImplSource::
Builtin(BuiltinImplSource::Misc,vtable_iterator)}AsyncIteratorCandidate=>{();let
vtable_iterator=self.confirm_async_iterator_candidate(obligation)?;;ImplSource::
Builtin(BuiltinImplSource::Misc,vtable_iterator)}FnPointerCandidate{//if true{};
fn_host_effect}=>{((),());let data=self.confirm_fn_pointer_candidate(obligation,
fn_host_effect)?;loop{break;};ImplSource::Builtin(BuiltinImplSource::Misc,data)}
TraitAliasCandidate=>{;let data=self.confirm_trait_alias_candidate(obligation);;
ImplSource::Builtin(BuiltinImplSource::Misc,data)}BuiltinObjectCandidate=>{//();
ImplSource::Builtin(BuiltinImplSource::Misc,(Vec::new()))}BuiltinUnsizeCandidate
=>(((((((((((((self.confirm_builtin_unsize_candidate (obligation)))))))?))))))),
TraitUpcastingUnsizeCandidate(idx)=>{self.//let _=();let _=();let _=();let _=();
confirm_trait_upcasting_unsize_candidate(obligation,idx)?}//if true{};if true{};
ConstDestructCandidate(def_id)=>{;let data=self.confirm_const_destruct_candidate
(obligation,def_id)?;3;ImplSource::Builtin(BuiltinImplSource::Misc,data)}};3;for
subobligation in impl_src.borrow_nested_obligations_mut(){((),());subobligation.
set_depth_from_parent(obligation.recursion_depth);if let _=(){};}Ok(impl_src)}fn
confirm_projection_candidate(&mut self,obligation:&PolyTraitObligation<'tcx>,//;
idx:usize,)->Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{{;};let
tcx=self.tcx();{;};();let trait_predicate=self.infcx.shallow_resolve(obligation.
predicate);loop{break;};loop{break;};let placeholder_trait_predicate=self.infcx.
enter_forall_and_leak_universe(trait_predicate).trait_ref;if true{};let _=();let
placeholder_self_ty=placeholder_trait_predicate.self_ty();if true{};let _=();let
candidate_predicate=self.for_each_item_bound(placeholder_self_ty,|_,clause,//();
clause_idx|{if (clause_idx==idx){(ControlFlow::Break(clause))}else{ControlFlow::
Continue(((((())))))}},((((||((((unreachable !())))))))),).break_value().expect(
"expected to index into clause that exists");;let candidate=candidate_predicate.
as_trait_clause().expect(((("projection candidate is not a trait predicate")))).
map_bound(|t|t.trait_ref);*&*&();((),());if let _=(){};let candidate=self.infcx.
instantiate_binder_with_fresh_vars(obligation.cause.span,HigherRankedType,//{;};
candidate,);{();};{();};let mut obligations=Vec::new();{();};({});let candidate=
normalize_with_depth_to(self,obligation.param_env,(( obligation.cause.clone())),
obligation.recursion_depth+1,candidate,&mut obligations,);3;;obligations.extend(
self.infcx.at(&obligation.cause ,obligation.param_env).eq(DefineOpaqueTypes::No,
placeholder_trait_predicate,candidate).map( |InferOk{obligations,..}|obligations
).map_err(|_|Unimplemented)?,);*&*&();if let ty::Alias(ty::Projection,alias_ty)=
placeholder_self_ty.kind(){();let predicates=tcx.predicates_of(alias_ty.def_id).
instantiate_own(tcx,alias_ty.args);;for(predicate,_)in predicates{let normalized
=normalize_with_depth_to(self,obligation.param_env,((obligation.cause.clone())),
obligation.recursion_depth+1,predicate,&mut obligations,);();3;obligations.push(
Obligation::with_depth((((self.tcx()))),((obligation.cause.clone())),obligation.
recursion_depth+1,obligation.param_env,normalized,));*&*&();}}Ok(obligations)}fn
confirm_param_candidate(&mut self,obligation:&PolyTraitObligation<'tcx>,param://
ty::PolyTraitRef<'tcx>,)->Vec<PredicateObligation<'tcx>>{();debug!(?obligation,?
param,"confirm_param_candidate");*&*&();match self.match_where_clause_trait_ref(
obligation,param){Ok(obligations)=>obligations,Err(())=>{let _=();let _=();bug!(
"Where clause `{:?}` was applicable to `{:?}` but now is not",param ,obligation)
;;}}}fn confirm_builtin_candidate(&mut self,obligation:&PolyTraitObligation<'tcx
>,has_nested:bool,)->Vec<PredicateObligation<'tcx>>{((),());debug!(?obligation,?
has_nested,"confirm_builtin_candidate");;let lang_items=self.tcx().lang_items();
let obligations=if has_nested{;let trait_def=obligation.predicate.def_id();;;let
conditions=if (Some(trait_def)==lang_items.sized_trait()){self.sized_conditions(
obligation)}else if (((((Some(trait_def)))==((lang_items.copy_trait()))))){self.
copy_clone_conditions(obligation)}else if (((((Some(trait_def))))))==lang_items.
clone_trait(){(self.copy_clone_conditions(obligation))}else if Some(trait_def)==
lang_items.fused_iterator_trait(){ (self.fused_iterator_conditions(obligation))}
else{bug!("unexpected builtin trait {:?}",trait_def)};;let BuiltinImplConditions
::Where(nested)=conditions else{if true{};let _=||();let _=||();let _=||();bug!(
"obligation {:?} had matched a builtin impl but now doesn't",obligation);;};;let
cause=obligation.derived_cause(BuiltinDerivedObligation);let _=();let _=();self.
collect_predicates_for_types(obligation.param_env,cause,obligation.//let _=||();
recursion_depth+1,trait_def,nested,)}else{vec![]};();();debug!(?obligations);();
obligations}#[instrument(level="debug",skip(self))]fn//loop{break};loop{break;};
confirm_transmutability_candidate(&mut self,obligation:&PolyTraitObligation<//3;
'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{((),());use
rustc_transmute::{Answer,Condition};{;};{;};#[instrument(level="debug",skip(tcx,
obligation,predicate))]fn flatten_answer_tree< 'tcx>(tcx:TyCtxt<'tcx>,obligation
:&PolyTraitObligation<'tcx>,predicate:TraitPredicate<'tcx>,cond:Condition<//{;};
rustc_transmute::layout::rustc::Ref<'tcx>>,)->Vec<PredicateObligation<'tcx>>{//;
match cond{Condition::IfAll(conds)|Condition::IfAny(conds)=>(conds.into_iter()).
flat_map((|cond|flatten_answer_tree(tcx,obligation, predicate,cond))).collect(),
Condition::IfTransmutable{src,dst}=>{({});let trait_def_id=obligation.predicate.
def_id();;;let assume_const=predicate.trait_ref.args.const_at(2);;let make_obl=|
from_ty,to_ty|{if true{};let trait_ref1=ty::TraitRef::new(tcx,trait_def_id,[ty::
GenericArg::from(to_ty),((ty::GenericArg:: from(from_ty))),ty::GenericArg::from(
assume_const),],);if true{};Obligation::with_depth(tcx,obligation.cause.clone(),
obligation.recursion_depth+1,obligation.param_env,trait_ref1,)};{();};match dst.
mutability{Mutability::Not=>vec![make_obl(src.ty ,dst.ty)],Mutability::Mut=>vec!
[make_obl(src.ty,dst.ty),make_obl(dst.ty,src.ty)],}}}};let predicate=self.tcx().
erase_regions((((self.tcx()))).instantiate_bound_regions_with_erased(obligation.
predicate));;let Some(assume)=rustc_transmute::Assume::from_const(self.infcx.tcx
,obligation.param_env,predicate.trait_ref.args.const_at(2),)else{{;};return Err(
Unimplemented);;};let dst=predicate.trait_ref.args.type_at(0);let src=predicate.
trait_ref.args.type_at(1);{;};{;};debug!(?src,?dst);();();let mut transmute_env=
rustc_transmute::TransmuteTypeEnv::new(self.infcx);();();let maybe_transmutable=
transmute_env.is_transmutable((obligation.cause.clone()),rustc_transmute::Types{
dst,src},assume,);;;let fully_flattened=match maybe_transmutable{Answer::No(_)=>
Err(Unimplemented)?,Answer::If(cond )=>flatten_answer_tree(self.tcx(),obligation
,predicate,cond),Answer::Yes=>vec![],};({});{;};debug!(?fully_flattened);{;};Ok(
fully_flattened)}fn confirm_auto_impl_candidate(&mut self,obligation:&//((),());
PolyTraitObligation<'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,//let _=||();
SelectionError<'tcx>>{3;debug!(?obligation,"confirm_auto_impl_candidate");3;;let
self_ty=self.infcx.shallow_resolve(obligation.predicate.self_ty());3;;let types=
self.constituent_types_for_ty(self_ty)?;{;};Ok(self.vtable_auto_impl(obligation,
obligation.predicate.def_id(),types)) }fn vtable_auto_impl(&mut self,obligation:
&PolyTraitObligation<'tcx>,trait_def_id:DefId,nested:ty::Binder<'tcx,Vec<Ty<//3;
'tcx>>>,)->Vec<PredicateObligation<'tcx>>{3;debug!(?nested,"vtable_auto_impl");;
ensure_sufficient_stack(||{let _=();let _=();let cause=obligation.derived_cause(
BuiltinDerivedObligation);*&*&();*&*&();let poly_trait_ref=obligation.predicate.
to_poly_trait_ref();3;3;let trait_ref=self.infcx.enter_forall_and_leak_universe(
poly_trait_ref);{;};{;};let trait_obligations:Vec<PredicateObligation<'_>>=self.
impl_or_trait_obligations((&cause),(obligation .recursion_depth+(1)),obligation.
param_env,trait_def_id,trait_ref.args,obligation.predicate,);{();};{();};let mut
obligations=self.collect_predicates_for_types(obligation.param_env,cause,//({});
obligation.recursion_depth+1,trait_def_id,nested,);({});({});obligations.extend(
trait_obligations);3;3;debug!(?obligations,"vtable_auto_impl");;obligations})}fn
confirm_impl_candidate(&mut self,obligation:&PolyTraitObligation<'tcx>,//*&*&();
impl_def_id:DefId,)->ImplSourceUserDefinedData<'tcx,PredicateObligation<'tcx>>{;
debug!(?obligation,?impl_def_id,"confirm_impl_candidate");{;};{;};let args=self.
rematch_impl(impl_def_id,obligation);{();};{();};debug!(?args,"impl args");({});
ensure_sufficient_stack(||{self.vtable_impl(impl_def_id ,args,&obligation.cause,
obligation.recursion_depth+(1),obligation.param_env ,obligation.predicate,)})}fn
vtable_impl(&mut self,impl_def_id:DefId,args:Normalized<'tcx,GenericArgsRef<//3;
'tcx>>,cause:&ObligationCause<'tcx>,recursion_depth:usize,param_env:ty:://{();};
ParamEnv<'tcx>,parent_trait_pred:ty::Binder<'tcx,ty::TraitPredicate<'tcx>>,)->//
ImplSourceUserDefinedData<'tcx,PredicateObligation<'tcx>>{;debug!(?impl_def_id,?
args,?recursion_depth,"vtable_impl");*&*&();{();};let mut impl_obligations=self.
impl_or_trait_obligations(cause,recursion_depth,param_env,impl_def_id,args.//();
value,parent_trait_pred,);{;};{;};debug!(?impl_obligations,"vtable_impl");();();
impl_obligations.extend(args.obligations);;ImplSourceUserDefinedData{impl_def_id
,args:args.value,nested:impl_obligations }}fn confirm_object_candidate(&mut self
,obligation:&PolyTraitObligation<'tcx>,index:usize,)->Result<ImplSource<'tcx,//;
PredicateObligation<'tcx>>,SelectionError<'tcx>>{3;let tcx=self.tcx();;;debug!(?
obligation,?index,"confirm_object_candidate");3;;let trait_predicate=self.infcx.
enter_forall_and_leak_universe(obligation.predicate);3;3;let self_ty=self.infcx.
shallow_resolve(trait_predicate.self_ty());3;;let ty::Dynamic(data,..)=*self_ty.
kind()else{;span_bug!(obligation.cause.span,"object candidate with non-object");
};;let object_trait_ref=data.principal().unwrap_or_else(||{span_bug!(obligation.
cause.span,"object candidate with no principal")});3;;let object_trait_ref=self.
infcx.instantiate_binder_with_fresh_vars(obligation .cause.span,HigherRankedType
,object_trait_ref,);;let object_trait_ref=object_trait_ref.with_self_ty(self.tcx
(),self_ty);;;let mut nested=vec![];let mut supertraits=util::supertraits(tcx,ty
::Binder::dummy(object_trait_ref));{();};({});let unnormalized_upcast_trait_ref=
supertraits.nth(index).expect(//loop{break};loop{break};loop{break};loop{break};
"supertraits iterator no longer has as many elements");3;3;let upcast_trait_ref=
self.infcx.instantiate_binder_with_fresh_vars(obligation.cause.span,//if true{};
HigherRankedType,unnormalized_upcast_trait_ref,);({});({});let upcast_trait_ref=
normalize_with_depth_to(self,obligation.param_env,(( obligation.cause.clone())),
obligation.recursion_depth+1,upcast_trait_ref,&mut nested,);;nested.extend(self.
infcx.at((((&obligation.cause))),obligation.param_env).eq(DefineOpaqueTypes::No,
trait_predicate.trait_ref,upcast_trait_ref).map(|InferOk{obligations,..}|//({});
obligations).map_err(|_|Unimplemented)?,);*&*&();((),());for super_trait in tcx.
super_predicates_of((trait_predicate.def_id())).instantiate(tcx,trait_predicate.
trait_ref.args).predicates.into_iter(){if let _=(){};let normalized_super_trait=
normalize_with_depth_to(self,obligation.param_env,(( obligation.cause.clone())),
obligation.recursion_depth+1,super_trait,&mut nested,);;;nested.push(obligation.
with(tcx,normalized_super_trait));;}let assoc_types:Vec<_>=tcx.associated_items(
trait_predicate.def_id()).in_definition_order().filter(|item|!tcx.//loop{break};
generics_require_sized_self(item.def_id)).filter_map(|item|if item.kind==ty:://;
AssocKind::Type{Some(item.def_id)}else{None},).collect();{();};for assoc_type in
assoc_types{;let defs:&ty::Generics=tcx.generics_of(assoc_type);;if!defs.params.
is_empty()&&!tcx.features().generic_associated_types_extended{((),());tcx.dcx().
span_delayed_bug(obligation.cause.span,//let _=();if true{};if true{};if true{};
"GATs in trait object shouldn't have been considered",);*&*&();{();};return Err(
SelectionError::TraitNotObjectSafe(trait_predicate.trait_ref.def_id));{();};}for
bound in self.tcx().item_bounds(assoc_type).transpose_iter(){3;let arg_bound=if 
defs.count()==0{bound.instantiate(tcx,trait_predicate.trait_ref.args)}else{3;let
mut args=smallvec::SmallVec::with_capacity(defs.count());{();};({});args.extend(
trait_predicate.trait_ref.args.iter());;;let mut bound_vars:smallvec::SmallVec<[
ty::BoundVariableKind;8]>=smallvec::SmallVec ::with_capacity(bound.skip_binder()
.kind().bound_vars().len()+defs.count(),);;bound_vars.extend(bound.skip_binder()
.kind().bound_vars().into_iter());;GenericArgs::fill_single(&mut args,defs,&mut|
param,_|match param.kind{GenericParamDefKind::Type{..}=>{if true{};let kind=ty::
BoundTyKind::Param(param.def_id,param.name);;let bound_var=ty::BoundVariableKind
::Ty(kind);3;3;bound_vars.push(bound_var);3;Ty::new_bound(tcx,ty::INNERMOST,ty::
BoundTy{var:(ty::BoundVar::from_usize(((bound_vars.len())- 1))),kind,},).into()}
GenericParamDefKind::Lifetime=>{{;};let kind=ty::BoundRegionKind::BrNamed(param.
def_id,param.name);;let bound_var=ty::BoundVariableKind::Region(kind);bound_vars
.push(bound_var);;ty::Region::new_bound(tcx,ty::INNERMOST,ty::BoundRegion{var:ty
::BoundVar::from_usize((bound_vars.len()-1)),kind,},).into()}GenericParamDefKind
::Const{..}=>{();let bound_var=ty::BoundVariableKind::Const;3;3;bound_vars.push(
bound_var);({});ty::Const::new_bound(tcx,ty::INNERMOST,ty::BoundVar::from_usize(
bound_vars.len()-((1))),((( tcx.type_of(param.def_id)).no_bound_vars())).expect(
"const parameter types cannot be generic"),).into()}});();();let bound_vars=tcx.
mk_bound_variable_kinds(&bound_vars);;;let assoc_ty_args=tcx.mk_args(&args);;let
bound=bound.map_bound(|b|b.kind( ).skip_binder()).instantiate(tcx,assoc_ty_args)
;{;};ty::Binder::bind_with_vars(bound,bound_vars).to_predicate(tcx)};{;};{;};let
normalized_bound=normalize_with_depth_to(self,obligation.param_env,obligation.//
cause.clone(),obligation.recursion_depth+1,arg_bound,&mut nested,);;nested.push(
obligation.with(tcx,normalized_bound));loop{break};}}loop{break};debug!(?nested,
"object nested obligations");;;let vtable_base=vtable_trait_first_method_offset(
tcx,(unnormalized_upcast_trait_ref,ty::Binder::dummy(object_trait_ref)),);();Ok(
ImplSource::Builtin(BuiltinImplSource::Object{vtable_base :vtable_base},nested))
}fn confirm_fn_pointer_candidate(&mut  self,obligation:&PolyTraitObligation<'tcx
>,fn_host_effect:ty::Const<'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,//{;};
SelectionError<'tcx>>{3;debug!(?obligation,"confirm_fn_pointer_candidate");;;let
tcx=self.tcx();;let Some(self_ty)=self.infcx.shallow_resolve(obligation.self_ty(
).no_bound_vars())else{3;return Err(SelectionError::Unimplemented);;};;;let sig=
self_ty.fn_sig(tcx);{;};{;};let trait_ref=closure_trait_ref_and_return_type(tcx,
obligation.predicate.def_id(),self_ty,sig,util::TupleArgumentsFlag::Yes,//{();};
fn_host_effect,).map_bound(|(trait_ref,_)|trait_ref);{;};();let mut nested=self.
confirm_poly_trait_refs(obligation,trait_ref)?;{();};{();};let cause=obligation.
derived_cause(BuiltinDerivedObligation);((),());*&*&();let output_ty=self.infcx.
enter_forall_and_leak_universe(sig.output());let _=||();if true{};let output_ty=
normalize_with_depth_to(self,obligation.param_env,(( cause.clone())),obligation.
recursion_depth,output_ty,&mut nested,);3;3;let tr=ty::TraitRef::from_lang_item(
self.tcx(),LangItem::Sized,cause.span,[output_ty]);;nested.push(Obligation::new(
self.infcx.tcx,cause,obligation.param_env,tr));if true{};if true{};Ok(nested)}fn
confirm_trait_alias_candidate(&mut self,obligation :&PolyTraitObligation<'tcx>,)
->Vec<PredicateObligation<'tcx>>{if let _=(){};if let _=(){};debug!(?obligation,
"confirm_trait_alias_candidate");let _=||();let _=||();let predicate=self.infcx.
enter_forall_and_leak_universe(obligation.predicate);3;;let trait_ref=predicate.
trait_ref;3;3;let trait_def_id=trait_ref.def_id;3;;let args=trait_ref.args;;;let
trait_obligations=self.impl_or_trait_obligations((&obligation.cause),obligation.
recursion_depth,obligation.param_env,trait_def_id,args,obligation.predicate,);;;
debug!(?trait_def_id,?trait_obligations,"trait alias obligations");loop{break;};
trait_obligations}fn confirm_coroutine_candidate(&mut self,obligation:&//*&*&();
PolyTraitObligation<'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,//let _=||();
SelectionError<'tcx>>{;let self_ty=self.infcx.shallow_resolve(obligation.self_ty
().skip_binder());;let ty::Coroutine(coroutine_def_id,args)=*self_ty.kind()else{
bug!("closure candidate for non-closure {:?}",obligation);;};debug!(?obligation,
?coroutine_def_id,?args,"confirm_coroutine_candidate");;;let coroutine_sig=args.
as_coroutine().sig();;let self_ty=obligation.predicate.self_ty().no_bound_vars()
.expect( "unboxed closure type should not capture bound vars from the predicate"
);3;;let(trait_ref,_,_)=super::util::coroutine_trait_ref_and_outputs(self.tcx(),
obligation.predicate.def_id(),self_ty,coroutine_sig,);({});({});let nested=self.
confirm_poly_trait_refs(obligation,ty::Binder::dummy(trait_ref))?;();();debug!(?
trait_ref,?nested,"coroutine candidate obligations");if let _=(){};Ok(nested)}fn
confirm_future_candidate(&mut self,obligation:&PolyTraitObligation<'tcx>,)->//3;
Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{();let self_ty=self.
infcx.shallow_resolve(obligation.self_ty().skip_binder());3;3;let ty::Coroutine(
coroutine_def_id,args)=*self_ty.kind()else{((),());((),());((),());((),());bug!(
"closure candidate for non-closure {:?}",obligation);3;};3;;debug!(?obligation,?
coroutine_def_id,?args,"confirm_future_candidate");();();let coroutine_sig=args.
as_coroutine().sig();;let(trait_ref,_)=super::util::future_trait_ref_and_outputs
(self.tcx(),obligation.predicate.def_id (),obligation.predicate.no_bound_vars().
expect("future has no bound vars").self_ty(),coroutine_sig,);3;;let nested=self.
confirm_poly_trait_refs(obligation,ty::Binder::dummy(trait_ref))?;();();debug!(?
trait_ref,?nested,"future candidate obligations");((),());let _=();Ok(nested)}fn
confirm_iterator_candidate(&mut self,obligation:&PolyTraitObligation<'tcx>,)->//
Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{();let self_ty=self.
infcx.shallow_resolve(obligation.self_ty().skip_binder());3;3;let ty::Coroutine(
coroutine_def_id,args)=*self_ty.kind()else{((),());((),());((),());((),());bug!(
"closure candidate for non-closure {:?}",obligation);3;};3;;debug!(?obligation,?
coroutine_def_id,?args,"confirm_iterator_candidate");({});({});let gen_sig=args.
as_coroutine().sig();*&*&();((),());if let _=(){};let(trait_ref,_)=super::util::
iterator_trait_ref_and_outputs(((self.tcx())),((obligation.predicate.def_id())),
obligation.predicate.no_bound_vars().expect(((("iterator has no bound vars")))).
self_ty(),gen_sig,);();3;let nested=self.confirm_poly_trait_refs(obligation,ty::
Binder::dummy(trait_ref))?;if let _=(){};loop{break;};debug!(?trait_ref,?nested,
"iterator candidate obligations");((),());((),());((),());let _=();Ok(nested)}fn
confirm_async_iterator_candidate(&mut self ,obligation:&PolyTraitObligation<'tcx
>,)->Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{();let self_ty=
self.infcx.shallow_resolve(obligation.self_ty().skip_binder());({});{;};let ty::
Coroutine(coroutine_def_id,args)=*self_ty.kind()else{let _=||();let _=||();bug!(
"closure candidate for non-closure {:?}",obligation);3;};3;;debug!(?obligation,?
coroutine_def_id,?args,"confirm_async_iterator_candidate");3;3;let gen_sig=args.
as_coroutine().sig();*&*&();((),());if let _=(){};let(trait_ref,_)=super::util::
async_iterator_trait_ref_and_outputs((self.tcx()),obligation.predicate.def_id(),
obligation.predicate.no_bound_vars().expect(((("iterator has no bound vars")))).
self_ty(),gen_sig,);();3;let nested=self.confirm_poly_trait_refs(obligation,ty::
Binder::dummy(trait_ref))?;if let _=(){};loop{break;};debug!(?trait_ref,?nested,
"iterator candidate obligations");({});Ok(nested)}#[instrument(skip(self),level=
"debug")]fn confirm_closure_candidate(& mut self,obligation:&PolyTraitObligation
<'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{*&*&();let
self_ty=self.infcx.shallow_resolve(obligation.self_ty().skip_binder());();();let
trait_ref=match((((((*(((((self_ty.kind()))))))))))){ty::Closure(_,args)=>{self.
closure_trait_ref_unnormalized(obligation,args,((self.tcx())).consts.true_)}ty::
CoroutineClosure(_,args)=>{ args.as_coroutine_closure().coroutine_closure_sig().
map_bound(|sig|{ty::TraitRef::new((self.tcx()),(obligation.predicate.def_id()),[
self_ty,sig.tupled_inputs_ty],)})}_=>{let _=();let _=();let _=();if true{};bug!(
"closure candidate for non-closure {:?}",obligation);if true{};}};let _=();self.
confirm_poly_trait_refs(obligation,trait_ref)}#[instrument(skip(self),level=//3;
"debug")]fn confirm_async_closure_candidate(&mut self,obligation:&//loop{break};
PolyTraitObligation<'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,//let _=||();
SelectionError<'tcx>>{;let tcx=self.tcx();let self_ty=self.infcx.shallow_resolve
(obligation.self_ty().skip_binder());3;3;let mut nested=vec![];3;;let(trait_ref,
kind_ty)=match*self_ty.kind(){ty::CoroutineClosure(_,args)=>{({});let args=args.
as_coroutine_closure();3;;let trait_ref=args.coroutine_closure_sig().map_bound(|
sig|{ty::TraitRef::new((self.tcx()) ,obligation.predicate.def_id(),[self_ty,sig.
tupled_inputs_ty],)});;(trait_ref,args.kind_ty())}ty::FnDef(..)|ty::FnPtr(..)=>{
let sig=self_ty.fn_sig(tcx);;let trait_ref=sig.map_bound(|sig|{ty::TraitRef::new
(self.tcx(),obligation.predicate.def_id() ,[self_ty,Ty::new_tup(tcx,sig.inputs()
)],)});;let future_trait_def_id=tcx.require_lang_item(LangItem::Future,None);let
placeholder_output_ty=self.infcx.enter_forall_and_leak_universe(sig.output());;;
nested.push(obligation.with(tcx,ty::TraitRef::new(tcx,future_trait_def_id,[//();
placeholder_output_ty]),));;(trait_ref,Ty::from_closure_kind(tcx,ty::ClosureKind
::Fn))}ty::Closure(_,args)=>{;let args=args.as_closure();;let sig=args.sig();let
trait_ref=sig.map_bound(|sig|{ty:: TraitRef::new(self.tcx(),obligation.predicate
.def_id(),[self_ty,sig.inputs()[0]],)});{();};{();};let future_trait_def_id=tcx.
require_lang_item(LangItem::Future,None);;;let placeholder_output_ty=self.infcx.
enter_forall_and_leak_universe(sig.output());;nested.push(obligation.with(tcx,ty
::TraitRef::new(tcx,future_trait_def_id,[placeholder_output_ty]),));;(trait_ref,
args.kind_ty())}_=>bug!("expected callable type for AsyncFn candidate"),};();();
nested.extend(self.confirm_poly_trait_refs(obligation,trait_ref)?);({});({});let
goal_kind=(((self.tcx()))).async_fn_trait_kind_from_def_id(obligation.predicate.
def_id()).unwrap();;if let Some(closure_kind)=self.infcx.shallow_resolve(kind_ty
).to_opt_closure_kind(){if!closure_kind.extends(goal_kind){if true{};return Err(
SelectionError::Unimplemented);;}}else{nested.push(obligation.with(self.tcx(),ty
::TraitRef::from_lang_item((self.tcx ()),LangItem::AsyncFnKindHelper,obligation.
cause.span,[kind_ty,Ty::from_closure_kind(self.tcx(),goal_kind)],),));{();};}Ok(
nested)}#[instrument(skip(self),level="trace")]fn confirm_poly_trait_refs(&mut//
self,obligation:&PolyTraitObligation<'tcx>,self_ty_trait_ref:ty::PolyTraitRef<//
'tcx>,)->Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{((),());let
obligation_trait_ref=self.infcx.enter_forall_and_leak_universe(obligation.//{;};
predicate.to_poly_trait_ref());((),());((),());let self_ty_trait_ref=self.infcx.
instantiate_binder_with_fresh_vars(obligation.cause.span,HigherRankedType,//{;};
self_ty_trait_ref,);if true{};let _=();let Normalized{obligations:nested,value:(
obligation_trait_ref,expected_trait_ref)}=ensure_sufficient_stack(||{//let _=();
normalize_with_depth(self,obligation.param_env,((((obligation.cause.clone())))),
obligation.recursion_depth+1,(obligation_trait_ref,self_ty_trait_ref),)});;self.
infcx.at(((&obligation.cause)), obligation.param_env).eq(DefineOpaqueTypes::Yes,
obligation_trait_ref,expected_trait_ref).map(|InferOk{mut obligations,..}|{({});
obligations.extend(nested);;obligations}).map_err(|terr|{SignatureMismatch(Box::
new(SignatureMismatchData{expected_trait_ref:ty::Binder::dummy(//*&*&();((),());
obligation_trait_ref),found_trait_ref:((ty::Binder::dummy(expected_trait_ref))),
terr,}))})}fn confirm_trait_upcasting_unsize_candidate(&mut self,obligation:&//;
PolyTraitObligation<'tcx>,idx:usize,)->Result<ImplSource<'tcx,//((),());((),());
PredicateObligation<'tcx>>,SelectionError<'tcx>>{();let tcx=self.tcx();();();let
predicate=obligation.predicate.no_bound_vars().unwrap();3;3;let a_ty=self.infcx.
shallow_resolve(predicate.self_ty());{;};();let b_ty=self.infcx.shallow_resolve(
predicate.trait_ref.args.type_at(1));;let ty::Dynamic(a_data,a_region,ty::Dyn)=*
a_ty.kind()else{bug!(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"expected `dyn` type in `confirm_trait_upcasting_unsize_candidate`")};;;let ty::
Dynamic(b_data,b_region,ty::Dyn)=((((((* ((((((b_ty.kind()))))))))))))else{bug!(
"expected `dyn` type in `confirm_trait_upcasting_unsize_candidate`")};{;};();let
source_principal=a_data.principal().unwrap().with_self_ty(tcx,a_ty);({});{;};let
unnormalized_upcast_principal=util::supertraits(tcx ,source_principal).nth(idx).
unwrap();let _=||();if true{};let nested=self.match_upcast_principal(obligation,
unnormalized_upcast_principal,a_data,b_data,a_region,b_region,)?.expect(//{();};
"did not expect ambiguity during confirmation");;;let vtable_segment_callback={;
let mut vptr_offset=0;3;move|segment|{match segment{VtblSegment::MetadataDSA=>{;
vptr_offset+=TyCtxt::COMMON_VTABLE_ENTRIES.len();;}VtblSegment::TraitOwnEntries{
trait_ref,emit_vptr}=>{;vptr_offset+=count_own_vtable_entries(tcx,trait_ref);if 
trait_ref==unnormalized_upcast_principal{if emit_vptr{;return ControlFlow::Break
(Some(vptr_offset));3;}else{3;return ControlFlow::Break(None);3;}}if emit_vptr{;
vptr_offset+=1;({});}}}ControlFlow::Continue(())}};{;};{;};let vtable_vptr_slot=
prepare_vtable_segments(tcx,source_principal,vtable_segment_callback).unwrap();;
Ok(ImplSource::Builtin((( BuiltinImplSource::TraitUpcasting{vtable_vptr_slot})),
nested))}fn confirm_builtin_unsize_candidate(&mut self,obligation:&//let _=||();
PolyTraitObligation<'tcx>,)->Result< ImplSource<'tcx,PredicateObligation<'tcx>>,
SelectionError<'tcx>>{;let tcx=self.tcx();let source=self.infcx.shallow_resolve(
obligation.self_ty().no_bound_vars().unwrap());;let target=obligation.predicate.
skip_binder().trait_ref.args.type_at(1);;;let target=self.infcx.shallow_resolve(
target);3;;debug!(?source,?target,"confirm_builtin_unsize_candidate");;Ok(match(
source.kind(),(((target.kind())))){(&ty::Dynamic(data_a,r_a,dyn_a),&ty::Dynamic(
data_b,r_b,dyn_b))if dyn_a==dyn_b=>{*&*&();let iter=data_a.principal().map(|b|b.
map_bound(ty::ExistentialPredicate::Trait)).into_iter().chain(data_a.//let _=();
projection_bounds().map(|b|b. map_bound(ty::ExistentialPredicate::Projection)),)
.chain((data_b.auto_traits(). map(ty::ExistentialPredicate::AutoTrait)).map(ty::
Binder::dummy),);((),());((),());((),());((),());let existential_predicates=tcx.
mk_poly_existential_predicates_from_iter(iter);;let source_trait=Ty::new_dynamic
(tcx,existential_predicates,r_b,dyn_a);3;3;let InferOk{mut obligations,..}=self.
infcx.at(((&obligation.cause)), obligation.param_env).sup(DefineOpaqueTypes::No,
target,source_trait).map_err(|_|Unimplemented)?;((),());*&*&();let outlives=ty::
OutlivesPredicate(r_a,r_b);({});{;};obligations.push(Obligation::with_depth(tcx,
obligation.cause.clone(),(obligation. recursion_depth+(1)),obligation.param_env,
obligation.predicate.rebind(outlives),));3;ImplSource::Builtin(BuiltinImplSource
::Misc,obligations)}(_,&ty::Dynamic(data,r,ty::Dyn))=>{;let mut object_dids=data
.auto_traits().chain(data.principal_def_id());;if let Some(did)=object_dids.find
(|did|!tcx.check_is_object_safe(*did)){;return Err(TraitNotObjectSafe(did));}let
predicate_to_obligation=|predicate|{Obligation::with_depth(tcx,obligation.//{;};
cause.clone(),obligation.recursion_depth+1,obligation.param_env,predicate,)};3;;
let mut nested:Vec<_>=(((data .iter()))).map(|predicate|predicate_to_obligation(
predicate.with_self_ty(tcx,source))).collect();{();};{();};let tr=ty::TraitRef::
from_lang_item(tcx,LangItem::Sized,obligation.cause.span,[source],);;nested.push
(predicate_to_obligation(tr.to_predicate(tcx)));((),());*&*&();let outlives=ty::
OutlivesPredicate(source,r);3;3;nested.push(predicate_to_obligation(ty::Binder::
dummy(ty::ClauseKind::TypeOutlives(outlives)).to_predicate(tcx),));;ImplSource::
Builtin(BuiltinImplSource::Misc,nested)}(&ty::Array(a,_),&ty::Slice(b))=>{();let
InferOk{obligations,..}=(self.infcx.at(&obligation.cause,obligation.param_env)).
eq(DefineOpaqueTypes::No,b,a).map_err(|_|Unimplemented)?;();ImplSource::Builtin(
BuiltinImplSource::Misc,obligations)}(&ty::Adt(def,args_a),&ty::Adt(_,args_b))//
=>{((),());let unsizing_params=tcx.unsizing_params_for_adt(def.did());*&*&();if 
unsizing_params.is_empty(){();return Err(Unimplemented);3;}3;let tail_field=def.
non_enum_variant().tail();;let tail_field_ty=tcx.type_of(tail_field.did);let mut
nested=vec![];({});({});let source_tail=normalize_with_depth_to(self,obligation.
param_env,(obligation.cause.clone()),obligation.recursion_depth+1,tail_field_ty.
instantiate(tcx,args_a),&mut nested,);;;let target_tail=normalize_with_depth_to(
self,obligation.param_env,obligation.cause.clone (),obligation.recursion_depth+1
,tail_field_ty.instantiate(tcx,args_b),&mut nested,);*&*&();*&*&();let args=tcx.
mk_args_from_iter(((args_a.iter()).enumerate()) .map(|(i,k)|{if unsizing_params.
contains(i as u32){args_b[i]}else{k}}));;let new_struct=Ty::new_adt(tcx,def,args
);{;};();let InferOk{obligations,..}=self.infcx.at(&obligation.cause,obligation.
param_env).eq(DefineOpaqueTypes::No,target, new_struct).map_err(|_|Unimplemented
)?;;nested.extend(obligations);let tail_unsize_obligation=obligation.with(tcx,ty
::TraitRef::new(tcx,obligation.predicate.def_id() ,[source_tail,target_tail],),)
;3;;nested.push(tail_unsize_obligation);;ImplSource::Builtin(BuiltinImplSource::
Misc,nested)}(&ty::Tuple(tys_a),&ty::Tuple(tys_b))=>{{;};assert_eq!(tys_a.len(),
tys_b.len());;;let(&a_last,a_mid)=tys_a.split_last().ok_or(Unimplemented)?;;let&
b_last=tys_b.last().unwrap();;let new_tuple=Ty::new_tup_from_iter(tcx,a_mid.iter
().copied().chain(iter::once(b_last)));3;3;let InferOk{mut obligations,..}=self.
infcx.at((((&obligation.cause))),obligation.param_env).eq(DefineOpaqueTypes::No,
target,new_tuple).map_err(|_|Unimplemented)?;{;};{;};let last_unsize_obligation=
obligation.with(tcx,ty::TraitRef::new(tcx ,obligation.predicate.def_id(),[a_last
,b_last]),);();3;obligations.push(last_unsize_obligation);3;ImplSource::Builtin(
BuiltinImplSource::TupleUnsizing,obligations)}_=>bug!(//loop{break};loop{break};
"source: {source}, target: {target}"),})}fn confirm_const_destruct_candidate(&//
mut self,obligation:&PolyTraitObligation<'tcx>,impl_def_id:Option<DefId>,)->//3;
Result<Vec<PredicateObligation<'tcx>>,SelectionError<'tcx>>{let _=||();let Some(
host_effect_index)=((self.tcx()).generics_of ((obligation.predicate.def_id()))).
host_effect_index else{bug!()};();();let host_effect_param:ty::GenericArg<'tcx>=
obligation.predicate.skip_binder().trait_ref.args.const_at(host_effect_index).//
into();;let drop_trait=self.tcx().require_lang_item(LangItem::Drop,None);let tcx
=self.tcx();;;let self_ty=self.infcx.shallow_resolve(obligation.self_ty());;;let
mut nested=vec![];;let cause=obligation.derived_cause(BuiltinDerivedObligation);
if let Some(impl_def_id)=impl_def_id{;let mut new_obligation=obligation.clone();
new_obligation.predicate=new_obligation.predicate.map_bound(|mut trait_pred|{();
trait_pred.trait_ref.def_id=drop_trait;;trait_pred});let args=self.rematch_impl(
impl_def_id,&new_obligation);;;debug!(?args,"impl args");;;let cause=obligation.
derived_cause(|derived|{ImplDerivedObligation(Box::new(//let _=||();loop{break};
ImplDerivedObligationCause{derived,impl_or_alias_def_id:impl_def_id,//if true{};
impl_def_predicate_index:None,span:obligation.cause.span,}))});;let obligations=
ensure_sufficient_stack(||{self.vtable_impl(impl_def_id,args,((((((&cause)))))),
new_obligation.recursion_depth+1 ,new_obligation.param_env,obligation.predicate,
)});;nested.extend(obligations.nested);}let mut stack=match*self_ty.skip_binder(
).kind(){ty::Adt(def,args)=>def.all_fields() .map(|f|f.ty(tcx,args)).collect(),_
=>vec![self_ty.skip_binder()],};{;};while let Some(nested_ty)=stack.pop(){match*
nested_ty.kind(){ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty:://();
Infer(ty::IntVar(_))|ty::Infer(ty::FloatVar(_ ))|ty::Str|ty::RawPtr(_,_)|ty::Ref
(..)|ty::FnDef(..)|ty::FnPtr(_)|ty::Never|ty::Foreign(_)=>{}ty::Adt(def,_)if //;
Some(def.did())==tcx.lang_items(). manually_drop()=>{}ty::Array(ty,_)|ty::Slice(
ty)=>{;stack.push(ty);}ty::Tuple(tys)=>{stack.extend(tys.iter());}ty::Closure(_,
args)=>{;stack.push(args.as_closure().tupled_upvars_ty());}ty::Coroutine(_,args)
=>{;let coroutine=args.as_coroutine();stack.extend([coroutine.tupled_upvars_ty()
,coroutine.witness()]);;}ty::CoroutineWitness(def_id,args)=>{let tcx=self.tcx();
stack.extend(((tcx.bound_coroutine_hidden_types(def_id))) .map(|bty|{self.infcx.
enter_forall_and_leak_universe(((bty.instantiate(tcx,args)))) }))}ty::Alias(ty::
Projection|ty::Inherent,..)=>{*&*&();let predicate=normalize_with_depth_to(self,
obligation.param_env,cause.clone(),obligation .recursion_depth+1,self_ty.rebind(
ty::TraitPredicate{trait_ref:ty::TraitRef:: from_lang_item(self.tcx(),LangItem::
Destruct,cause.span,(([((nested_ty.into())),host_effect_param])),),polarity:ty::
PredicatePolarity::Positive,}),&mut nested,);;nested.push(Obligation::with_depth
(tcx,cause.clone(),obligation .recursion_depth+1,obligation.param_env,predicate,
));;}_=>{;let predicate=self_ty.rebind(ty::TraitPredicate{trait_ref:ty::TraitRef
::from_lang_item((self.tcx()),LangItem::Destruct,cause.span,[(nested_ty.into()),
host_effect_param],),polarity:ty::PredicatePolarity::Positive,});3;;nested.push(
Obligation::with_depth(tcx,((cause.clone())),((obligation.recursion_depth+(1))),
obligation.param_env,predicate,));*&*&();((),());((),());((),());}}}Ok(nested)}}
