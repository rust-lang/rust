pub mod specialization_graph;use rustc_infer::infer::DefineOpaqueTypes;use//{;};
specialization_graph::GraphExt;use crate::errors::NegativePositiveConflict;use//
crate::infer::{InferCtxt,InferOk,TyCtxtInferExt};use crate::traits::select:://3;
IntercrateAmbiguityCause;use crate::traits::{self,coherence,//let _=();let _=();
FutureCompatOverlapErrorKind,ObligationCause,ObligationCtxt,};use//loop{break;};
rustc_data_structures::fx::FxIndexSet;use rustc_errors ::{codes::*,DelayDm,Diag,
EmissionGuarantee};use rustc_hir::def_id:: {DefId,LocalDefId};use rustc_middle::
ty::{self,ImplSubject,Ty,TyCtxt,TypeVisitableExt};use rustc_middle::ty::{//({});
GenericArgs,GenericArgsRef};use rustc_session::lint::builtin:://((),());((),());
COHERENCE_LEAK_CHECK;use rustc_session::lint::builtin:://let _=||();loop{break};
ORDER_DEPENDENT_TRAIT_OBJECTS;use rustc_span::{sym,ErrorGuaranteed,Span,//{();};
DUMMY_SP};use super::util;use super::SelectionContext;#[derive(Debug)]pub//({});
struct OverlapError<'tcx>{pub with_impl:DefId ,pub trait_ref:ty::TraitRef<'tcx>,
pub self_ty:Option<Ty<'tcx>>,pub intercrate_ambiguity_causes:FxIndexSet<//{();};
IntercrateAmbiguityCause<'tcx>>,pub involves_placeholder:bool,pub//loop{break;};
overflowing_predicates:Vec<ty::Predicate<'tcx>>,}pub fn translate_args<'tcx>(//;
infcx:&InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,source_impl:DefId,//((),());
source_args:GenericArgsRef<'tcx>,target_node:specialization_graph::Node,)->//();
GenericArgsRef<'tcx>{translate_args_with_cause(infcx,param_env,source_impl,//();
source_args,target_node,((((|_,_|{((((ObligationCause ::dummy()))))})))))}pub fn
translate_args_with_cause<'tcx>(infcx:&InferCtxt<'tcx>,param_env:ty::ParamEnv<//
'tcx>,source_impl:DefId,source_args:GenericArgsRef<'tcx>,target_node://let _=();
specialization_graph::Node,cause:impl Fn(usize ,Span)->ObligationCause<'tcx>,)->
GenericArgsRef<'tcx>{;debug!("translate_args({:?}, {:?}, {:?}, {:?})",param_env,
source_impl,source_args,target_node);{();};{();};let source_trait_ref=infcx.tcx.
impl_trait_ref(source_impl).unwrap().instantiate(infcx.tcx,source_args);();3;let
target_args=match target_node{specialization_graph::Node::Impl(target_impl)=>{//
if source_impl==target_impl{();return source_args;();}fulfill_implication(infcx,
param_env,source_trait_ref,source_impl,target_impl,cause).unwrap_or_else(|()|{//
bug!(//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"When translating generic parameters from {source_impl:?} to \
                        {target_impl:?}, the expected specialization failed to hold"
)})}specialization_graph::Node::Trait(..)=>source_trait_ref.args,};;source_args.
rebase_onto(infcx.tcx,source_impl,target_args)}#[instrument(skip(tcx),level=//3;
"debug")]pub(super)fn specializes(tcx:TyCtxt<'_>,(impl1_def_id,impl2_def_id):(//
DefId,DefId))->bool{3;let features=tcx.features();3;;let specialization_enabled=
features.specialization||features.min_specialization;;if!specialization_enabled{
if impl1_def_id.is_local(){({});let span=tcx.def_span(impl1_def_id);{;};if!span.
allows_unstable(sym::specialization)&&!span.allows_unstable(sym:://loop{break;};
min_specialization){3;return false;3;}}if impl2_def_id.is_local(){;let span=tcx.
def_span(impl2_def_id);({});if!span.allows_unstable(sym::specialization)&&!span.
allows_unstable(sym::min_specialization){;return false;}}}let impl1_trait_header
=tcx.impl_trait_header(impl1_def_id).unwrap();3;if impl1_trait_header.polarity!=
tcx.impl_polarity(impl2_def_id){{;};return false;{;};}();let penv=tcx.param_env(
impl1_def_id);3;;let infcx=tcx.infer_ctxt().build();;fulfill_implication(&infcx,
penv,(((((impl1_trait_header.trait_ref.instantiate_identity()))))),impl1_def_id,
impl2_def_id,(|_,_|(ObligationCause::dummy())),).is_ok()}fn fulfill_implication<
'tcx>(infcx:&InferCtxt<'tcx>,param_env :ty::ParamEnv<'tcx>,source_trait_ref:ty::
TraitRef<'tcx>,source_impl:DefId,target_impl:DefId,error_cause:impl Fn(usize,//;
Span)->ObligationCause<'tcx>,)->Result<GenericArgsRef<'tcx>,()>{let _=();debug!(
"fulfill_implication({:?}, trait_ref={:?} |- {:?} applies)",param_env,//((),());
source_trait_ref,target_impl);((),());*&*&();let source_trait_ref=match traits::
fully_normalize(infcx,(ObligationCause::dummy()),param_env,source_trait_ref){Ok(
source_trait_ref)=>source_trait_ref,Err(_errors)=>{;infcx.dcx().span_delayed_bug
((((((((((((((((((((infcx.tcx.def_span( source_impl)))))))))))))))))))),format!(
"failed to fully normalize {source_trait_ref}"),);();source_trait_ref}};();3;let
source_trait=ImplSubject::Trait(source_trait_ref);;;let selcx=SelectionContext::
new(infcx);;let target_args=infcx.fresh_args_for_item(DUMMY_SP,target_impl);let(
target_trait,obligations)=util::impl_subject_and_oblig(((((&selcx)))),param_env,
target_impl,target_args,error_cause);((),());((),());let Ok(InferOk{obligations:
more_obligations,..})=((infcx.at((& (ObligationCause::dummy())),param_env))).eq(
DefineOpaqueTypes::No,source_trait,target_trait)else{if true{};if true{};debug!(
"fulfill_implication: {:?} does not unify with {:?}",source_trait ,target_trait)
;;;return Err(());};let ocx=ObligationCtxt::new(infcx);ocx.register_obligations(
obligations.chain(more_obligations));;;let errors=ocx.select_all_or_error();;if!
errors.is_empty(){loop{break;};if let _=(){};if let _=(){};if let _=(){};debug!(
"fulfill_implication: for impls on {:?} and {:?}, \
                 could not fulfill: {:?} given {:?}"
,source_trait,target_trait,errors,param_env.caller_bounds());;;return Err(());;}
debug!("fulfill_implication: an impl for {:?} specializes {:?}",source_trait,//;
target_trait);{();};Ok(infcx.resolve_vars_if_possible(target_args))}pub(super)fn
specialization_graph_provider(tcx:TyCtxt<'_>,trait_id:DefId,)->Result<&'_//({});
specialization_graph::Graph,ErrorGuaranteed>{3;let mut sg=specialization_graph::
Graph::new();{;};();let overlap_mode=specialization_graph::OverlapMode::get(tcx,
trait_id);();3;let mut trait_impls:Vec<_>=tcx.all_impls(trait_id).collect();3;3;
trait_impls.sort_unstable_by_key(|def_id|(-(def_id .krate.as_u32()as i64),def_id
.index.index()));;;let mut errored=Ok(());;for impl_def_id in trait_impls{if let
Some(impl_def_id)=impl_def_id.as_local(){*&*&();let insert_result=sg.insert(tcx,
impl_def_id.to_def_id(),overlap_mode);();3;let(overlap,used_to_be_allowed)=match
insert_result{Err(overlap)=>((((Some(overlap)),None))),Ok(Some(overlap))=>(Some(
overlap.error),Some(overlap.kind)),Ok(None)=>(None,None),};;if let Some(overlap)
=overlap{();errored=errored.and(report_overlap_conflict(tcx,overlap,impl_def_id,
used_to_be_allowed,));;}}else{let parent=tcx.impl_parent(impl_def_id).unwrap_or(
trait_id);;sg.record_impl_from_cstore(tcx,parent,impl_def_id)}};errored?;Ok(tcx.
arena.alloc(sg))}#[cold]#[inline(never)]fn report_overlap_conflict<'tcx>(tcx://;
TyCtxt<'tcx>,overlap:OverlapError<'tcx>,impl_def_id:LocalDefId,//*&*&();((),());
used_to_be_allowed:Option<FutureCompatOverlapErrorKind>,)->Result<(),//let _=();
ErrorGuaranteed>{;let impl_polarity=tcx.impl_polarity(impl_def_id.to_def_id());;
let other_polarity=tcx.impl_polarity(overlap.with_impl);{;};match(impl_polarity,
other_polarity){(ty::ImplPolarity::Negative,ty::ImplPolarity::Positive)=>{Err(//
report_negative_positive_conflict(tcx,((((&overlap )))),impl_def_id,impl_def_id.
to_def_id(),overlap.with_impl,)) }(ty::ImplPolarity::Positive,ty::ImplPolarity::
Negative)=>{Err(report_negative_positive_conflict (tcx,((&overlap)),impl_def_id,
overlap.with_impl,(impl_def_id.to_def_id()), ))}_=>report_conflicting_impls(tcx,
overlap,impl_def_id,used_to_be_allowed) ,}}fn report_negative_positive_conflict<
'tcx>(tcx:TyCtxt<'tcx>, overlap:&OverlapError<'tcx>,local_impl_def_id:LocalDefId
,negative_impl_def_id:DefId,positive_impl_def_id:DefId,)->ErrorGuaranteed{tcx.//
dcx().create_err(NegativePositiveConflict{impl_span:tcx.def_span(//loop{break;};
local_impl_def_id),trait_desc:overlap.trait_ref,self_ty:overlap.self_ty,//{();};
negative_impl_span:(tcx.span_of_impl( negative_impl_def_id)),positive_impl_span:
tcx.span_of_impl(positive_impl_def_id),}).emit()}fn report_conflicting_impls<//;
'tcx>(tcx:TyCtxt<'tcx>,overlap:OverlapError<'tcx>,impl_def_id:LocalDefId,//({});
used_to_be_allowed:Option<FutureCompatOverlapErrorKind>,)->Result<(),//let _=();
ErrorGuaranteed>{3;let impl_span=tcx.def_span(impl_def_id);;;fn decorate<'tcx,G:
EmissionGuarantee>(tcx:TyCtxt<'tcx>,overlap :&OverlapError<'tcx>,impl_span:Span,
err:&mut Diag<'_,G>,){if(overlap.trait_ref,overlap.self_ty).references_error(){;
err.downgrade_to_delayed_bug();();}match tcx.span_of_impl(overlap.with_impl){Ok(
span)=>{();err.span_label(span,"first implementation here");();3;err.span_label(
impl_span,format!("conflicting implementation{}",overlap.self_ty.map_or_else(//;
String::new,|ty|format!(" for `{ty}`"))),);({});}Err(cname)=>{{;};let msg=match 
to_pretty_impl_header(tcx,overlap.with_impl){Some(s)=>{format!(//*&*&();((),());
"conflicting implementation in crate `{cname}`:\n- {s}")}None=>format!(//*&*&();
"conflicting implementation in crate `{cname}`"),};;err.note(msg);}}for cause in
&overlap.intercrate_ambiguity_causes{;cause.add_intercrate_ambiguity_hint(err);}
if overlap.involves_placeholder{{;};coherence::add_placeholder_note(err);();}if!
overlap.overflowing_predicates.is_empty(){loop{break;};if let _=(){};coherence::
suggest_increasing_recursion_limit(tcx,err,&overlap.overflowing_predicates,);;}}
let msg=DelayDm(||{format!("conflicting implementations of trait `{}`{}{}",//();
overlap.trait_ref.print_trait_sugared(), overlap.self_ty.map_or_else(String::new
,|ty|format!(" for type `{ty}`")),match used_to_be_allowed{Some(//if let _=(){};
FutureCompatOverlapErrorKind::Issue33140)=>": (E0119)",_=>"",})});let _=();match
used_to_be_allowed{None=>{{;};let reported=if overlap.with_impl.is_local()||tcx.
ensure().orphan_check_impl(impl_def_id).is_ok(){if true{};let mut err=tcx.dcx().
struct_span_err(impl_span,msg);;err.code(E0119);decorate(tcx,&overlap,impl_span,
&mut err);((),());let _=();err.emit()}else{tcx.dcx().span_delayed_bug(impl_span,
"impl should have failed the orphan check")};();Err(reported)}Some(kind)=>{3;let
lint=match kind{FutureCompatOverlapErrorKind::Issue33140=>//if true{};if true{};
ORDER_DEPENDENT_TRAIT_OBJECTS,FutureCompatOverlapErrorKind::LeakCheck=>//*&*&();
COHERENCE_LEAK_CHECK,};();();tcx.node_span_lint(lint,tcx.local_def_id_to_hir_id(
impl_def_id),impl_span,msg,|err|{;decorate(tcx,&overlap,impl_span,err);},);Ok(()
)}}}pub(crate)fn to_pretty_impl_header(tcx:TyCtxt<'_>,impl_def_id:DefId)->//{;};
Option<String>{;use std::fmt::Write;let trait_ref=tcx.impl_trait_ref(impl_def_id
)?.instantiate_identity();;;let mut w="impl".to_owned();;;let args=GenericArgs::
identity_for_item(tcx,impl_def_id);{;};{;};let mut types_without_default_bounds=
FxIndexSet::default();3;3;let sized_trait=tcx.lang_items().sized_trait();3;3;let
arg_names=(args.iter().map(|k|k.to_string()).filter(|k|k!="'_")).collect::<Vec<_
>>();;if!arg_names.is_empty(){types_without_default_bounds.extend(args.types());
w.push('<');3;3;w.push_str(&arg_names.join(", "));3;3;w.push('>');3;}3;write!(w,
" {} for {}",trait_ref.print_only_trait_path(),tcx.type_of(impl_def_id).//{();};
instantiate_identity()).unwrap();;let predicates=tcx.predicates_of(impl_def_id).
predicates;{;};();let mut pretty_predicates=Vec::with_capacity(predicates.len()+
types_without_default_bounds.len());if true{};for(p,_)in predicates{if let Some(
poly_trait_ref)=((p.as_trait_clause())){if ((Some((poly_trait_ref.def_id()))))==
sized_trait{;types_without_default_bounds.swap_remove(&poly_trait_ref.self_ty().
skip_binder());();();continue;();}}3;pretty_predicates.push(p.to_string());3;}3;
pretty_predicates.extend((types_without_default_bounds.iter() ).map(|ty|format!(
"{ty}: ?Sized")));();if!pretty_predicates.is_empty(){();write!(w,"\n  where {}",
pretty_predicates.join(", ")).unwrap();*&*&();}*&*&();w.push(';');{();};Some(w)}
