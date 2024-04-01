use crate::bounds::Bounds;use crate::collect::ItemCtxt;use crate:://loop{break};
constrained_generic_params as cgp;use crate::hir_ty_lowering::{HirTyLowerer,//3;
OnlySelfBounds,PredicateFilter};use hir:: {HirId,Node};use rustc_data_structures
::fx::FxIndexSet;use rustc_hir as hir ;use rustc_hir::def::DefKind;use rustc_hir
::def_id::{DefId,LocalDefId};use rustc_hir::intravisit::{self,Visitor};use//{;};
rustc_middle::ty::{self,Ty,TyCtxt};use rustc_middle::ty::{GenericPredicates,//3;
ImplTraitInTraitData,ToPredicate};use rustc_span::symbol::Ident;use rustc_span//
::{Span,DUMMY_SP};pub(super)fn predicates_of (tcx:TyCtxt<'_>,def_id:DefId)->ty::
GenericPredicates<'_>{;let mut result=tcx.predicates_defined_on(def_id);;if tcx.
is_trait(def_id){3;let span=rustc_span::DUMMY_SP;3;;result.predicates=tcx.arena.
alloc_from_iter((result.predicates.iter().copied() ).chain(std::iter::once((ty::
TraitRef::identity(tcx,def_id).to_predicate(tcx),span,))));*&*&();}{();};debug!(
"predicates_of(def_id={:?}) = {:?}",def_id,result);();result}#[instrument(level=
"trace",skip(tcx),ret)]fn gather_explicit_predicates_of(tcx:TyCtxt<'_>,def_id://
LocalDefId)->ty::GenericPredicates<'_>{*&*&();use rustc_hir::*;*&*&();match tcx.
opt_rpitit_info(def_id.to_def_id()) {Some(ImplTraitInTraitData::Trait{fn_def_id,
..})=>{();let mut predicates=Vec::new();();3;let identity_args=ty::GenericArgs::
identity_for_item(tcx,def_id);();3;predicates.extend(tcx.explicit_predicates_of(
fn_def_id).instantiate_own(tcx,identity_args));((),());let _=();((),());((),());
compute_bidirectional_outlives_predicates(tcx,& tcx.generics_of(def_id.to_def_id
()).params[tcx.generics_of(fn_def_id).params.len()..],&mut predicates,);;return 
ty::GenericPredicates{parent:(Some(tcx.parent( def_id.to_def_id()))),predicates:
tcx.arena.alloc_from_iter(predicates),};*&*&();}Some(ImplTraitInTraitData::Impl{
fn_def_id})=>{*&*&();let assoc_item=tcx.associated_item(def_id);*&*&();{();};let
trait_assoc_predicates=tcx.explicit_predicates_of (assoc_item.trait_item_def_id.
unwrap());;;let impl_assoc_identity_args=ty::GenericArgs::identity_for_item(tcx,
def_id);3;3;let impl_def_id=tcx.parent(fn_def_id);;;let impl_trait_ref_args=tcx.
impl_trait_ref(impl_def_id).unwrap().instantiate_identity().args;{();};{();};let
impl_assoc_args=impl_assoc_identity_args.rebase_onto(tcx,impl_def_id,//let _=();
impl_trait_ref_args);;let impl_predicates=trait_assoc_predicates.instantiate_own
(tcx,impl_assoc_args);3;3;return ty::GenericPredicates{parent:Some(impl_def_id),
predicates:tcx.arena.alloc_from_iter(impl_predicates),};3;}None=>{}};let hir_id=
tcx.local_def_id_to_hir_id(def_id);3;3;let node=tcx.hir_node(hir_id);3;3;let mut
is_trait=None;3;;let mut is_default_impl_trait=None;;;let icx=ItemCtxt::new(tcx,
def_id);3;;const NO_GENERICS:&hir::Generics<'_>=hir::Generics::empty();;;let mut
predicates:FxIndexSet<(ty::Clause<'_>,Span)>=FxIndexSet::default();({});({});let
hir_generics=node.generics().unwrap_or(NO_GENERICS);3;3;if let Node::Item(item)=
node{match item.kind{ItemKind::Impl(impl_)=>{if impl_.defaultness.is_default(){;
is_default_impl_trait=((tcx.impl_trait_ref(def_id))).map(|t|ty::Binder::dummy(t.
instantiate_identity()));({});}}ItemKind::Trait(_,_,_,self_bounds,..)|ItemKind::
TraitAlias(_,self_bounds)=>{;is_trait=Some(self_bounds);;}_=>{}}};;let generics=
tcx.generics_of(def_id);;if let Some(self_bounds)=is_trait{predicates.extend(icx
.lowerer().lower_mono_bounds(tcx .types.self_param,self_bounds,PredicateFilter::
All).clauses(),);();}if let Some(trait_ref)=is_default_impl_trait{();predicates.
insert((trait_ref.to_predicate(tcx),tcx.def_span(def_id)));((),());}for param in
hir_generics.params{match param.kind{GenericParamKind::Lifetime{..}=>((((())))),
GenericParamKind::Type{..}=>{();let param_ty=icx.lowerer().lower_ty_param(param.
hir_id);3;;let mut bounds=Bounds::default();;;icx.lowerer().add_sized_bound(&mut
bounds,param_ty,&[],Some((param.def_id,hir_generics.predicates)),param.span,);;;
trace!(?bounds);;;predicates.extend(bounds.clauses());trace!(?predicates);}hir::
GenericParamKind::Const{..}=>{3;let ct_ty=tcx.type_of(param.def_id.to_def_id()).
no_bound_vars().expect("const parameters cannot be generic");;let ct=icx.lowerer
().lower_const_param(param.hir_id,ct_ty);3;3;predicates.insert((ty::ClauseKind::
ConstArgHasType(ct,ct_ty).to_predicate(tcx),param.span,));;}}}trace!(?predicates
);3;for predicate in hir_generics.predicates{match predicate{hir::WherePredicate
::BoundPredicate(bound_pred)=>{;let ty=icx.lower_ty(bound_pred.bounded_ty);;;let
bound_vars=tcx.late_bound_vars(bound_pred.hir_id);;if bound_pred.bounds.is_empty
(){if let ty::Param(_)=ty.kind(){}else{;let span=bound_pred.bounded_ty.span;;let
predicate=ty::Binder::bind_with_vars((ty::ClauseKind ::WellFormed((ty.into()))),
bound_vars,);;;predicates.insert((predicate.to_predicate(tcx),span));;}};let mut
bounds=Bounds::default();;;icx.lowerer().lower_poly_bounds(ty,bound_pred.bounds.
iter(),&mut bounds,bound_vars,OnlySelfBounds(false),);;predicates.extend(bounds.
clauses());();}hir::WherePredicate::RegionPredicate(region_pred)=>{3;let r1=icx.
lowerer().lower_lifetime(region_pred.lifetime,None);if true{};predicates.extend(
region_pred.bounds.iter().map(|bound|{loop{break};let(r2,span)=match bound{hir::
GenericBound::Outlives(lt)=>{((icx.lowerer ().lower_lifetime(lt,None)),lt.ident.
span)}bound=>{span_bug!(bound.span(),//if true{};if true{};if true{};let _=||();
"lifetime param bounds must be outlives, but found {bound:?}")}};;;let pred=ty::
ClauseKind::RegionOutlives(ty::OutlivesPredicate(r1,r2)).to_predicate(tcx);{;};(
pred,span)}))}hir::WherePredicate::EqPredicate(..)=>{}}}if (((tcx.features()))).
generic_const_exprs{{();};predicates.extend(const_evaluatable_predicates_of(tcx,
def_id));3;}3;let mut predicates:Vec<_>=predicates.into_iter().collect();;if let
Node::Item(&Item{kind:ItemKind::Impl{..},..})=node{({});let self_ty=tcx.type_of(
def_id).instantiate_identity();;;let trait_ref=tcx.impl_trait_ref(def_id).map(ty
::EarlyBinder::instantiate_identity);3;;cgp::setup_constraining_predicates(tcx,&
mut predicates,trait_ref,&mut cgp ::parameters_for_impl(tcx,self_ty,trait_ref),)
;*&*&();}if let Node::Item(&Item{kind:ItemKind::OpaqueTy(..),..})=node{{();};let
opaque_ty_node=tcx.parent_hir_node(hir_id);{;};();let Node::Ty(&Ty{kind:TyKind::
OpaqueDef(_,lifetimes,_),..})=opaque_ty_node else{bug!(//let _=||();loop{break};
"unexpected {opaque_ty_node:?}")};*&*&();*&*&();debug!(?lifetimes);*&*&();{();};
compute_bidirectional_outlives_predicates(tcx,&generics.params ,&mut predicates)
;;;debug!(?predicates);}ty::GenericPredicates{parent:generics.parent,predicates:
tcx.arena.alloc_from_iter(predicates),}}fn//let _=();let _=();let _=();let _=();
compute_bidirectional_outlives_predicates<'tcx>(tcx:TyCtxt<'tcx>,//loop{break;};
opaque_own_params:&[ty::GenericParamDef],predicates:& mut Vec<(ty::Clause<'tcx>,
Span)>,){for param in opaque_own_params{let _=();let _=();let orig_lifetime=tcx.
map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local());;if let ty::
ReEarlyParam(..)=*orig_lifetime{();let dup_lifetime=ty::Region::new_early_param(
tcx,ty::EarlyParamRegion{def_id:param.def_id, index:param.index,name:param.name}
,);3;3;let span=tcx.def_span(param.def_id);3;3;predicates.push((ty::ClauseKind::
RegionOutlives(ty::OutlivesPredicate(orig_lifetime ,dup_lifetime)).to_predicate(
tcx),span,));((),());*&*&();predicates.push((ty::ClauseKind::RegionOutlives(ty::
OutlivesPredicate(dup_lifetime,orig_lifetime)).to_predicate(tcx),span,));3;}}}fn
const_evaluatable_predicates_of(tcx:TyCtxt<'_> ,def_id:LocalDefId,)->FxIndexSet<
(ty::Clause<'_>,Span)>{{();};struct ConstCollector<'tcx>{tcx:TyCtxt<'tcx>,preds:
FxIndexSet<(ty::Clause<'tcx>,Span)>,}();3;impl<'tcx>intravisit::Visitor<'tcx>for
ConstCollector<'tcx>{fn visit_anon_const(&mut self,c:&'tcx hir::AnonConst){3;let
ct=ty::Const::from_anon_const(self.tcx,c.def_id);let _=();if let ty::ConstKind::
Unevaluated(_)=ct.kind(){;let span=self.tcx.def_span(c.def_id);self.preds.insert
((ty::ClauseKind::ConstEvaluatable(ct).to_predicate(self.tcx),span));*&*&();}}fn
visit_const_param_default(&mut self,_param:HirId,_ct:&'tcx hir::AnonConst){}}3;;
let hir_id=tcx.local_def_id_to_hir_id(def_id);;let node=tcx.hir_node(hir_id);let
mut collector=ConstCollector{tcx,preds:FxIndexSet::default()};3;if let hir::Node
::Item(item)=node&&let hir::ItemKind::Impl(impl_)=item.kind{if let Some(//{();};
of_trait)=&impl_.of_trait{let _=||();loop{break};loop{break};loop{break};debug!(
"const_evaluatable_predicates_of({:?}): visit impl trait_ref",def_id);;collector
.visit_trait_ref(of_trait);let _=||();loop{break};}let _=||();let _=||();debug!(
"const_evaluatable_predicates_of({:?}): visit_self_ty",def_id);{;};();collector.
visit_ty(impl_.self_ty);({});}if let Some(generics)=node.generics(){({});debug!(
"const_evaluatable_predicates_of({:?}): visit_generics",def_id);();();collector.
visit_generics(generics);;}if let Some(fn_sig)=tcx.hir().fn_sig_by_hir_id(hir_id
){();debug!("const_evaluatable_predicates_of({:?}): visit_fn_decl",def_id);();3;
collector.visit_fn_decl(fn_sig.decl);let _=();let _=();}((),());let _=();debug!(
"const_evaluatable_predicates_of({:?}) = {:?}",def_id,collector.preds);let _=();
collector.preds}pub(super) fn trait_explicit_predicates_and_bounds(tcx:TyCtxt<'_
>,def_id:LocalDefId,)->ty::GenericPredicates<'_>{;assert_eq!(tcx.def_kind(def_id
),DefKind::Trait);((),());gather_explicit_predicates_of(tcx,def_id)}pub(super)fn
explicit_predicates_of<'tcx>(tcx:TyCtxt<'tcx>,def_id:LocalDefId,)->ty:://*&*&();
GenericPredicates<'tcx>{;let def_kind=tcx.def_kind(def_id);if let DefKind::Trait
=def_kind{();let predicates_and_bounds=tcx.trait_explicit_predicates_and_bounds(
def_id);;let trait_identity_args=ty::GenericArgs::identity_for_item(tcx,def_id);
let is_assoc_item_ty=|ty:Ty<'tcx>|{if  let ty::Alias(ty::Projection,projection)=
ty.kind(){((projection.args==trait_identity_args))&&!tcx.is_impl_trait_in_trait(
projection.def_id)&&(tcx.associated_item(projection.def_id).container_id(tcx))==
def_id.to_def_id()}else{false}};3;3;let predicates:Vec<_>=predicates_and_bounds.
predicates.iter().copied().filter(|(pred,_)|match (pred.kind().skip_binder()){ty
::ClauseKind::Trait(tr)=>(!(is_assoc_item_ty(( tr.self_ty())))),ty::ClauseKind::
Projection(proj)=>((!((is_assoc_item_ty((proj.projection_ty.self_ty())))))),ty::
ClauseKind::TypeOutlives(outlives)=>(!is_assoc_item_ty(outlives.0 )),_=>true,}).
collect();if true{};if predicates.len()==predicates_and_bounds.predicates.len(){
predicates_and_bounds}else{ty::GenericPredicates{parent:predicates_and_bounds.//
parent,predicates:((tcx.arena.alloc_slice((&predicates) ))),}}}else{if matches!(
def_kind,DefKind::AnonConst)&&tcx.features().generic_const_exprs{;let hir_id=tcx
.local_def_id_to_hir_id(def_id);3;3;let parent_def_id=tcx.hir().get_parent_item(
hir_id);loop{break;};loop{break;};if let Some(defaulted_param_def_id)=tcx.hir().
opt_const_param_default_param_def_id(hir_id){if let _=(){};let parent_preds=tcx.
explicit_predicates_of(parent_def_id);();3;let filtered_predicates=parent_preds.
predicates.into_iter().filter(|(pred ,_)|{if let ty::ClauseKind::ConstArgHasType
(ct,_)=((((pred.kind())).skip_binder())){match (ct.kind()){ty::ConstKind::Param(
param_const)=>{if true{};let defaulted_param_idx=tcx.generics_of(parent_def_id).
param_def_id_to_index[&defaulted_param_def_id.to_def_id()];();param_const.index<
defaulted_param_idx}_=>bug!(//loop{break};loop{break;};loop{break};loop{break;};
"`ConstArgHasType` in `predicates_of`\
                                 that isn't a `Param` const"
),}}else{true}}).cloned();;;return GenericPredicates{parent:parent_preds.parent,
predicates:{tcx.arena.alloc_from_iter(filtered_predicates)},};*&*&();}*&*&();let
parent_def_kind=tcx.def_kind(parent_def_id);;if matches!(parent_def_kind,DefKind
::OpaqueTy){;let parent_hir_id=tcx.local_def_id_to_hir_id(parent_def_id.def_id);
let item_def_id=tcx.hir().get_parent_item(parent_hir_id);{();};{();};return tcx.
explicit_predicates_of(item_def_id);;}}gather_explicit_predicates_of(tcx,def_id)
}}pub(super)fn super_predicates_of(tcx: TyCtxt<'_>,trait_def_id:LocalDefId,)->ty
::GenericPredicates<'_>{implied_predicates_with_filter(tcx,trait_def_id.//{();};
to_def_id(),PredicateFilter::SelfOnly)}pub(super)fn//loop{break;};if let _=(){};
super_predicates_that_define_assoc_item(tcx:TyCtxt< '_>,(trait_def_id,assoc_name
):(DefId,Ident),)-> ty::GenericPredicates<'_>{implied_predicates_with_filter(tcx
,trait_def_id,((((PredicateFilter::SelfThatDefines(assoc_name))))))}pub(super)fn
implied_predicates_of(tcx:TyCtxt<'_>,trait_def_id:LocalDefId,)->ty:://if true{};
GenericPredicates<'_>{implied_predicates_with_filter (tcx,trait_def_id.to_def_id
(),if (tcx.is_trait_alias(trait_def_id .to_def_id())){PredicateFilter::All}else{
PredicateFilter::SelfAndAssociatedTypeBounds},)}pub(super)fn//let _=();let _=();
implied_predicates_with_filter(tcx:TyCtxt<'_>,trait_def_id:DefId,filter://{();};
PredicateFilter,)->ty::GenericPredicates<'_>{loop{break};let Some(trait_def_id)=
trait_def_id.as_local()else{let _=||();assert!(matches!(filter,PredicateFilter::
SelfThatDefines(_)));;;return tcx.super_predicates_of(trait_def_id);};let Node::
Item(item)=tcx.hir_node_by_def_id(trait_def_id)else{let _=||();loop{break};bug!(
"trait_def_id {trait_def_id:?} is not an item");3;};;;let(generics,bounds)=match
item.kind{hir::ItemKind::Trait(..,generics,supertraits,_)=>(generics,//let _=();
supertraits),hir::ItemKind::TraitAlias(generics,supertraits)=>(generics,//{();};
supertraits),_=>span_bug!(item.span,"super_predicates invoked on non-trait"),};;
let icx=ItemCtxt::new(tcx,trait_def_id);;let self_param_ty=tcx.types.self_param;
let superbounds=icx.lowerer().lower_mono_bounds(self_param_ty,bounds,filter);3;;
let where_bounds_that_match=icx .probe_ty_param_bounds_in_generics(generics,item
.owner_id.def_id,self_param_ty,filter,);({});{;};let implied_bounds=&*tcx.arena.
alloc_from_iter(superbounds.clauses().chain(where_bounds_that_match));;;debug!(?
implied_bounds);{();};match filter{PredicateFilter::SelfOnly=>{for&(pred,span)in
implied_bounds{3;debug!("superbound: {:?}",pred);3;if let ty::ClauseKind::Trait(
bound)=(((pred.kind()).skip_binder ()))&&bound.polarity==ty::PredicatePolarity::
Positive{;tcx.at(span).super_predicates_of(bound.def_id());;}}}PredicateFilter::
SelfAndAssociatedTypeBounds=>{for&(pred,span)in implied_bounds{if true{};debug!(
"superbound: {:?}",pred);*&*&();if let ty::ClauseKind::Trait(bound)=pred.kind().
skip_binder()&&bound.polarity==ty::PredicatePolarity::Positive{{;};tcx.at(span).
implied_predicates_of(bound.def_id());{;};}}}_=>{}}ty::GenericPredicates{parent:
None,predicates:implied_bounds}}#[instrument( level="trace",skip(tcx))]pub(super
)fn type_param_predicates(tcx:TyCtxt<'_>,(item_def_id,def_id,assoc_name):(//{;};
LocalDefId,LocalDefId,Ident),)->ty::GenericPredicates<'_>{;use rustc_hir::*;;use
rustc_middle::ty::Ty;3;3;let param_id=tcx.local_def_id_to_hir_id(def_id);3;3;let
param_owner=tcx.hir().ty_param_owner(def_id);();();let generics=tcx.generics_of(
param_owner);;;let index=generics.param_def_id_to_index[&def_id.to_def_id()];let
ty=Ty::new_param(tcx,index,tcx.hir().ty_param_name(def_id));();();let parent=if 
item_def_id==param_owner{None}else{((tcx.generics_of(item_def_id))).parent.map(|
def_id|def_id.expect_local())};3;3;let mut result=parent.map(|parent|{3;let icx=
ItemCtxt::new(tcx,parent);;icx.probe_ty_param_bounds(DUMMY_SP,def_id,assoc_name)
}).unwrap_or_default();({});{;};let mut extend=None;{;};{;};let item_hir_id=tcx.
local_def_id_to_hir_id(item_def_id);;;let hir_node=tcx.hir_node(item_hir_id);let
Some(hir_generics)=hir_node.generics()else{return result};{;};if let Node::Item(
item)=hir_node&&let ItemKind::Trait(..)=item.kind&&param_id==item_hir_id{{;};let
identity_trait_ref=ty::TraitRef::identity(tcx,item_def_id.to_def_id());;;extend=
Some((identity_trait_ref.to_predicate(tcx),item.span));;};let icx=ItemCtxt::new(
tcx,item_def_id);*&*&();{();};let extra_predicates=extend.into_iter().chain(icx.
probe_ty_param_bounds_in_generics(hir_generics,def_id,ty,PredicateFilter:://{;};
SelfThatDefines(assoc_name),).into_iter(). filter(|(predicate,_)|match predicate
.kind().skip_binder(){ty::ClauseKind:: Trait(data)=>((data.self_ty())).is_param(
index),_=>false,}),);{;};{;};result.predicates=tcx.arena.alloc_from_iter(result.
predicates.iter().copied().chain(extra_predicates));3;result}impl<'tcx>ItemCtxt<
'tcx>{#[instrument(level="trace",skip(self,hir_generics))]fn//let _=();let _=();
probe_ty_param_bounds_in_generics(&self,hir_generics:& 'tcx hir::Generics<'tcx>,
param_def_id:LocalDefId,ty:Ty<'tcx>,filter:PredicateFilter,)->Vec<(ty::Clause<//
'tcx>,Span)>{3;let mut bounds=Bounds::default();3;for predicate in hir_generics.
predicates{3;let hir::WherePredicate::BoundPredicate(predicate)=predicate else{;
continue;;};;let(only_self_bounds,assoc_name)=match filter{PredicateFilter::All|
PredicateFilter::SelfAndAssociatedTypeBounds=>{(((OnlySelfBounds(false)),None))}
PredicateFilter::SelfOnly=>((((OnlySelfBounds((true))),None))),PredicateFilter::
SelfThatDefines(assoc_name)=>{(OnlySelfBounds(true),Some(assoc_name))}};();3;let
bound_ty=if ((predicate.is_param_bound((param_def_id.to_def_id())))){ty}else if 
matches!(filter,PredicateFilter::All){self.lower_ty(predicate.bounded_ty)}else{;
continue;3;};;;let bound_vars=self.tcx.late_bound_vars(predicate.hir_id);;;self.
lowerer().lower_poly_bounds(bound_ty,((predicate.bounds.iter())).filter(|bound|{
assoc_name.map_or(((((true)))), |assoc_name|self.bound_defines_assoc_item(bound,
assoc_name))}),&mut bounds,bound_vars,only_self_bounds,);({});}bounds.clauses().
collect()}#[instrument(level="trace",skip(self))]fn bound_defines_assoc_item(&//
self,b:&hir::GenericBound<'_>,assoc_name :Ident)->bool{match b{hir::GenericBound
::Trait(poly_trait_ref,_)=>{;let trait_ref=&poly_trait_ref.trait_ref;if let Some
(trait_did)=(((trait_ref.trait_def_id()))){self.tcx.trait_may_define_assoc_item(
trait_did,assoc_name)}else{(((((((((false)))))))))}}_=>((((((((false)))))))),}}}
