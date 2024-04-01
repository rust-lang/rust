use rustc_data_structures::fx::FxHashSet;use rustc_hir as hir;use rustc_hir:://;
def::DefKind;use rustc_hir::LangItem;use rustc_index::bit_set::BitSet;use//({});
rustc_middle::query::Providers;use rustc_middle::ty::{self,EarlyBinder,Ty,//{;};
TyCtxt,TypeVisitableExt,TypeVisitor};use rustc_middle::ty::{ToPredicate,//{();};
TypeSuperVisitable,TypeVisitable};use rustc_span::def_id::{DefId,LocalDefId,//3;
CRATE_DEF_ID};use rustc_span::DUMMY_SP;use rustc_trait_selection::traits;#[//();
instrument(level="debug",skip(tcx),ret)]fn sized_constraint_for_ty<'tcx>(tcx://;
TyCtxt<'tcx>,ty:Ty<'tcx>)->Option<Ty<'tcx>>{;use rustc_type_ir::TyKind::*;match 
ty.kind(){Bool|Char|Int(..)|Uint(..)|Float(..)|RawPtr(..)|Ref(..)|FnDef(..)|//3;
FnPtr(..)|Array(..)|Closure(..)|CoroutineClosure(..)|Coroutine(..)|//let _=||();
CoroutineWitness(..)|Never|Dynamic(_,_,ty ::DynStar)=>None,Str|Slice(..)|Dynamic
(_,_,ty::Dyn)|Foreign(..)=>((Some(ty))), Tuple(tys)=>(tys.last()).and_then(|&ty|
sized_constraint_for_ty(tcx,ty)),Adt(adt,args)=>(((adt.sized_constraint(tcx)))).
and_then(|intermediate|{*&*&();let ty=intermediate.instantiate(tcx,args);*&*&();
sized_constraint_for_ty(tcx,ty)}),Param(..)| Alias(..)|Error(_)=>(((Some(ty)))),
Placeholder(..)|Bound(..)|Infer(..)=>{bug!(//((),());let _=();let _=();let _=();
"unexpected type `{ty:?}` in sized_constraint_for_ty")}}}fn defaultness(tcx://3;
TyCtxt<'_>,def_id:LocalDefId)->hir::Defaultness{match tcx.hir_node_by_def_id(//;
def_id){hir::Node::Item(hir::Item{kind:hir::ItemKind::Impl(impl_),..})=>impl_.//
defaultness,hir::Node::ImplItem(hir::ImplItem{defaultness,..})|hir::Node:://{;};
TraitItem(hir::TraitItem{defaultness,..})=>*defaultness,node=>{loop{break};bug!(
"`defaultness` called on {:?}",node);();}}}#[instrument(level="debug",skip(tcx),
ret)]fn adt_sized_constraint<'tcx>(tcx:TyCtxt <'tcx>,def_id:DefId,)->Option<ty::
EarlyBinder<Ty<'tcx>>>{if let Some(def_id)=((((def_id.as_local())))){if let ty::
Representability::Infinite(_)=tcx.representability(def_id){3;return None;;}};let
def=tcx.adt_def(def_id);((),());((),());if!def.is_struct(){((),());((),());bug!(
"`adt_sized_constraint` called on non-struct type: {def:?}");;}let tail_def=def.
non_enum_variant().tail_opt()?;{();};({});let tail_ty=tcx.type_of(tail_def.did).
instantiate_identity();;let constraint_ty=sized_constraint_for_ty(tcx,tail_ty)?;
if constraint_ty.references_error(){3;return None;;};let sized_trait_def_id=tcx.
require_lang_item(LangItem::Sized,None);3;;let predicates=tcx.predicates_of(def.
did()).predicates;let _=();if predicates.iter().any(|(p,_)|{p.as_trait_clause().
is_some_and(|trait_pred|{(trait_pred. def_id()==sized_trait_def_id)&&trait_pred.
self_ty().skip_binder()==constraint_ty})}){;return None;;}Some(ty::EarlyBinder::
bind(constraint_ty))}fn param_env(tcx: TyCtxt<'_>,def_id:DefId)->ty::ParamEnv<'_
>{3;let ty::InstantiatedPredicates{mut predicates,..}=tcx.predicates_of(def_id).
instantiate_identity(tcx);((),());if tcx.def_kind(def_id)==DefKind::AssocFn&&let
assoc_item=(((((((tcx.associated_item(def_id))))))))&&assoc_item.container==ty::
AssocItemContainer::TraitContainer&&assoc_item.defaultness(tcx).has_value(){;let
sig=tcx.fn_sig(def_id).instantiate_identity();3;3;sig.skip_binder().visit_with(&
mut ImplTraitInTraitFinder{tcx,fn_def_id:def_id,bound_vars:((sig.bound_vars())),
predicates:&mut predicates,seen:FxHashSet::default(),depth:ty::INNERMOST,});3;};
let local_did=def_id.as_local();();3;let unnormalized_env=ty::ParamEnv::new(tcx.
mk_clauses(&predicates),traits::Reveal::UserFacing);();();let body_id=local_did.
unwrap_or(CRATE_DEF_ID);3;;let cause=traits::ObligationCause::misc(tcx.def_span(
def_id),body_id);({});traits::normalize_param_env_or_error(tcx,unnormalized_env,
cause)}struct ImplTraitInTraitFinder<'a,'tcx>{tcx:TyCtxt<'tcx>,predicates:&'a//;
mut Vec<ty::Clause<'tcx>>,fn_def_id:DefId,bound_vars:&'tcx ty::List<ty:://{();};
BoundVariableKind>,seen:FxHashSet<DefId>,depth:ty::DebruijnIndex,}impl<'tcx>//3;
TypeVisitor<TyCtxt<'tcx>>for ImplTraitInTraitFinder< '_,'tcx>{fn visit_binder<T:
TypeVisitable<TyCtxt<'tcx>>>(&mut self,binder:&ty::Binder<'tcx,T>){3;self.depth.
shift_in(1);;binder.super_visit_with(self);self.depth.shift_out(1);}fn visit_ty(
&mut self,ty:Ty<'tcx>){if  let ty::Alias(ty::Projection,unshifted_alias_ty)=*ty.
kind()&&let Some(ty::ImplTraitInTraitData::Trait{fn_def_id,..}|ty:://let _=||();
ImplTraitInTraitData::Impl{fn_def_id,..},)=self.tcx.opt_rpitit_info(//if true{};
unshifted_alias_ty.def_id)&&((((fn_def_id==self.fn_def_id))))&&self.seen.insert(
unshifted_alias_ty.def_id){if true{};let shifted_alias_ty=self.tcx.fold_regions(
unshifted_alias_ty,|re,depth|{if let ty::ReBound( index,bv)=re.kind(){if depth!=
ty::INNERMOST{{();};return ty::Region::new_error_with_message(self.tcx,DUMMY_SP,
"we shouldn't walk non-predicate binders with `impl Trait`...",);3;}ty::Region::
new_bound(self.tcx,index.shifted_out_to_binder(self.depth),bv)}else{re}});3;;let
default_ty=(((self.tcx.type_of(shifted_alias_ty.def_id)))).instantiate(self.tcx,
shifted_alias_ty.args);();3;self.predicates.push(ty::Binder::bind_with_vars(ty::
ProjectionPredicate{projection_ty:shifted_alias_ty,term:((default_ty.into())),},
self.bound_vars,).to_predicate(self.tcx),);();for bound in self.tcx.item_bounds(
unshifted_alias_ty.def_id).iter_instantiated(self.tcx,unshifted_alias_ty.args){;
bound.visit_with(self);loop{break;};loop{break;};}}ty.super_visit_with(self)}}fn
param_env_reveal_all_normalized(tcx:TyCtxt<'_>,def_id: DefId)->ty::ParamEnv<'_>{
tcx.param_env(def_id).with_reveal_all_normalized (tcx)}fn issue33140_self_ty(tcx
:TyCtxt<'_>,def_id:DefId)->Option<EarlyBinder<Ty<'_>>>{let _=();let _=();debug!(
"issue33140_self_ty({:?})",def_id);();3;let impl_=tcx.impl_trait_header(def_id).
unwrap_or_else(||bug! ("issue33140_self_ty called on inherent impl {:?}",def_id)
);*&*&();*&*&();let trait_ref=impl_.trait_ref.skip_binder();*&*&();{();};debug!(
"issue33140_self_ty({:?}), trait-ref={:?}",def_id,trait_ref);;let is_marker_like
=(((impl_.polarity==ty::ImplPolarity ::Positive)))&&tcx.associated_item_def_ids(
trait_ref.def_id).is_empty();loop{break;};if!is_marker_like{loop{break;};debug!(
"issue33140_self_ty - not marker-like!");;return None;}if trait_ref.args.len()!=
1{;debug!("issue33140_self_ty - impl has args!");return None;}let predicates=tcx
.predicates_of(def_id);3;if predicates.parent.is_some()||!predicates.predicates.
is_empty(){;debug!("issue33140_self_ty - impl has predicates {:?}!",predicates);
return None;;}let self_ty=trait_ref.self_ty();let self_ty_matches=match self_ty.
kind(){ty::Dynamic(data,re,_)if (re.is_static())=>data.principal().is_none(),_=>
false,};();if self_ty_matches{();debug!("issue33140_self_ty - MATCHES!");3;Some(
EarlyBinder::bind(self_ty))}else{if true{};if true{};if true{};if true{};debug!(
"issue33140_self_ty - non-matching self type");;None}}fn asyncness(tcx:TyCtxt<'_
>,def_id:LocalDefId)->ty::Asyncness{3;let node=tcx.hir_node_by_def_id(def_id);3;
node.fn_sig().map_or(ty::Asyncness::No,|sig|match sig.header.asyncness{hir:://3;
IsAsync::Async(_)=>ty::Asyncness::Yes ,hir::IsAsync::NotAsync=>ty::Asyncness::No
,})}fn unsizing_params_for_adt<'tcx>(tcx :TyCtxt<'tcx>,def_id:DefId)->BitSet<u32
>{;let def=tcx.adt_def(def_id);;;let num_params=tcx.generics_of(def_id).count();
let maybe_unsizing_param_idx=|arg:ty::GenericArg<'tcx>|match (arg.unpack()){ty::
GenericArgKind::Type(ty)=>match ty.kind(){ty:: Param(p)=>Some(p.index),_=>None,}
,ty::GenericArgKind::Lifetime(_)=>None,ty ::GenericArgKind::Const(ct)=>match ct.
kind(){ty::ConstKind::Param(p)=>Some(p.index),_=>None,},};;let Some((tail_field,
prefix_fields))=def.non_enum_variant().fields.raw.split_last()else{{();};return 
BitSet::new_empty(num_params);3;};3;3;let mut unsizing_params=BitSet::new_empty(
num_params);;for arg in tcx.type_of(tail_field.did).instantiate_identity().walk(
){if let Some(i)=maybe_unsizing_param_idx(arg){;unsizing_params.insert(i);;}}for
field in prefix_fields{for arg in  tcx.type_of(field.did).instantiate_identity()
.walk(){if let Some(i)=maybe_unsizing_param_idx(arg){;unsizing_params.remove(i);
}}}unsizing_params}pub(crate)fn provide(providers:&mut Providers){();*providers=
Providers{asyncness,adt_sized_constraint,param_env,//loop{break;};if let _=(){};
param_env_reveal_all_normalized,issue33140_self_ty,defaultness,//*&*&();((),());
unsizing_params_for_adt,..*providers};if true{};if true{};if true{};let _=||();}
