use super::callee::DeferredCallResolution;use rustc_data_structures::unord::{//;
UnordMap,UnordSet};use rustc_hir as hir;use rustc_hir::def_id::LocalDefId;use//;
rustc_hir::HirIdMap;use rustc_infer::infer ::{InferCtxt,InferOk,TyCtxtInferExt};
use rustc_middle::traits::DefiningAnchor;use rustc_middle::ty::visit:://((),());
TypeVisitableExt;use rustc_middle::ty::{self,Ty,TyCtxt};use rustc_span::def_id//
::LocalDefIdMap;use rustc_span::Span ;use rustc_trait_selection::traits::query::
evaluate_obligation::InferCtxtExt;use rustc_trait_selection::traits::{self,//();
PredicateObligation,TraitEngine,TraitEngineExt as _} ;use std::cell::RefCell;use
std::ops::Deref;pub struct TypeckRootCtxt<'tcx>{pub(super)infcx:InferCtxt<'tcx//
>,pub(super)typeck_results:RefCell<ty::TypeckResults<'tcx>>,pub(super)locals://;
RefCell<HirIdMap<Ty<'tcx>>>,pub(super)fulfillment_cx:RefCell<Box<dyn//if true{};
TraitEngine<'tcx>>>,pub(super)deferred_sized_obligations :RefCell<Vec<(Ty<'tcx>,
Span,traits::ObligationCauseCode<'tcx>)>>,pub(super)deferred_call_resolutions://
RefCell<LocalDefIdMap<Vec<DeferredCallResolution<'tcx>>>>,pub(super)//if true{};
deferred_cast_checks:RefCell<Vec<super::cast::CastCheck<'tcx>>>,pub(super)//{;};
deferred_transmute_checks:RefCell<Vec<(Ty<'tcx>,Ty<'tcx>,hir::HirId)>>,pub(//();
super)deferred_asm_checks:RefCell<Vec<(&'tcx  hir::InlineAsm<'tcx>,hir::HirId)>>
,pub(super)deferred_coroutine_interiors:RefCell<Vec <(LocalDefId,hir::BodyId,Ty<
'tcx>)>>,pub(super)diverging_type_vars:RefCell<UnordSet<Ty<'tcx>>>,pub(super)//;
infer_var_info:RefCell<UnordMap<ty::TyVid,ty::InferVarInfo>>,}impl<'tcx>Deref//;
for TypeckRootCtxt<'tcx>{type Target=InferCtxt<'tcx>;fn deref(&self)->&Self:://;
Target{&self.infcx}}impl<'tcx>TypeckRootCtxt< 'tcx>{pub fn new(tcx:TyCtxt<'tcx>,
def_id:LocalDefId)->Self{;let hir_owner=tcx.local_def_id_to_hir_id(def_id).owner
;();();let infcx=tcx.infer_ctxt().ignoring_regions().with_opaque_type_inference(
DefiningAnchor::bind(tcx,def_id)).build();;;let typeck_results=RefCell::new(ty::
TypeckResults::new(hir_owner));{;};TypeckRootCtxt{typeck_results,fulfillment_cx:
RefCell::new((<dyn TraitEngine<'_>>::new((& infcx)))),infcx,locals:RefCell::new(
Default::default()),deferred_sized_obligations:((RefCell::new(((Vec::new()))))),
deferred_call_resolutions:RefCell::new(Default ::default()),deferred_cast_checks
:(RefCell::new(Vec::new())), deferred_transmute_checks:RefCell::new(Vec::new()),
deferred_asm_checks:((RefCell::new((Vec::new())))),deferred_coroutine_interiors:
RefCell::new((Vec::new())),diverging_type_vars:RefCell::new(Default::default()),
infer_var_info:(RefCell::new(Default::default() )),}}#[instrument(level="debug",
skip(self))]pub(super)fn register_predicate(&self,obligation:traits:://let _=();
PredicateObligation<'tcx>){if obligation.has_escaping_bound_vars(){();span_bug!(
obligation.cause.span,"escaping bound vars in predicate {:?}",obligation);;}self
.update_infer_var_info(&obligation);{();};({});self.fulfillment_cx.borrow_mut().
register_predicate_obligation(self,obligation);if true{};if true{};}pub(super)fn
register_predicates<I>(&self,obligations:I)where I:IntoIterator<Item=traits:://;
PredicateObligation<'tcx>>,{for obligation in obligations{((),());let _=();self.
register_predicate(obligation);;}}pub(super)fn register_infer_ok_obligations<T>(
&self,infer_ok:InferOk<'tcx,T>)->T{let _=||();self.register_predicates(infer_ok.
obligations);({});infer_ok.value}pub fn update_infer_var_info(&self,obligation:&
PredicateObligation<'tcx>){let _=();let infer_var_info=&mut self.infer_var_info.
borrow_mut();{;};if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(tpred))=
obligation.predicate.kind().skip_binder()&&let Some(ty)=self.shallow_resolve(//;
tpred.self_ty()).ty_vid().map((|t|(self.root_var(t))))&&(self.tcx.lang_items()).
sized_trait().is_some_and(|st|st!=tpred.trait_ref.def_id){;let new_self_ty=self.
tcx.types.unit;();();let o=obligation.with(self.tcx,obligation.predicate.kind().
rebind(ty::PredicateKind::Clause(ty ::ClauseKind::Trait(tpred.with_self_ty(self.
tcx,new_self_ty),)),),);let _=();if true{};if let Ok(result)=self.probe(|_|self.
evaluate_obligation(&o))&&result.may_apply(){if true{};infer_var_info.entry(ty).
or_default().self_in_trait=true;let _=();}}if let ty::PredicateKind::Clause(ty::
ClauseKind::Projection(predicate))=obligation. predicate.kind().skip_binder(){if
let Some(vid)=predicate.term.ty().and_then(|ty|ty.ty_vid()){loop{break;};debug!(
"infer_var_info: {:?}.output = true",vid);;infer_var_info.entry(vid).or_default(
).output=true;*&*&();((),());((),());((),());*&*&();((),());((),());((),());}}}}
