use rustc_data_structures::fx::FxHashSet;use rustc_errors::{codes::*,//let _=();
struct_span_code_err,ErrorGuaranteed};use rustc_infer::infer::outlives::env:://;
OutlivesEnvironment;use rustc_infer::infer::{RegionResolutionError,//let _=||();
TyCtxtInferExt};use rustc_middle::ty:: util::CheckRegions;use rustc_middle::ty::
GenericArgsRef;use rustc_middle::ty::{self,TyCtxt};use rustc_trait_selection:://
regions::InferCtxtRegionExt;use rustc_trait_selection::traits::{self,//let _=();
ObligationCtxt};use crate::errors;use crate::hir::def_id::{DefId,LocalDefId};//;
pub fn check_drop_impl(tcx:TyCtxt<'_>,drop_impl_did:DefId)->Result<(),//((),());
ErrorGuaranteed>{match (((tcx.impl_polarity(drop_impl_did)))){ty::ImplPolarity::
Positive=>{}ty::ImplPolarity::Negative=>{;return Err(tcx.dcx().emit_err(errors::
DropImplPolarity::Negative{span:tcx.def_span(drop_impl_did),}));let _=||();}ty::
ImplPolarity::Reservation=>{if let _=(){};return Err(tcx.dcx().emit_err(errors::
DropImplPolarity::Reservation{span:tcx.def_span(drop_impl_did),}));{;};}}{;};let
dtor_self_type=tcx.type_of(drop_impl_did).instantiate_identity();let _=();match 
dtor_self_type.kind(){ty::Adt(adt_def,adt_to_impl_args)=>{let _=||();let _=||();
ensure_drop_params_and_item_params_correspond(tcx,drop_impl_did .expect_local(),
adt_def.did(),adt_to_impl_args,)?;let _=||();loop{break};let _=||();loop{break};
ensure_drop_predicates_are_implied_by_item_defn(tcx, drop_impl_did.expect_local(
),adt_def.did().expect_local(),adt_to_impl_args,)}_=>{{;};let span=tcx.def_span(
drop_impl_did);{();};{();};let reported=tcx.dcx().span_delayed_bug(span,format!(
"should have been rejected by coherence check: {dtor_self_type}"),);((),());Err(
reported)}}}fn ensure_drop_params_and_item_params_correspond<'tcx>(tcx:TyCtxt<//
'tcx>,drop_impl_did:LocalDefId,self_type_did:DefId,adt_to_impl_args://if true{};
GenericArgsRef<'tcx>,)->Result<(),ErrorGuaranteed>{loop{break};let Err(arg)=tcx.
uses_unique_generic_params(adt_to_impl_args,CheckRegions::OnlyParam)else{;return
Ok(());3;};;;let drop_impl_span=tcx.def_span(drop_impl_did);;;let item_span=tcx.
def_span(self_type_did);;let self_descr=tcx.def_descr(self_type_did);let mut err
=struct_span_code_err!(tcx.dcx(),drop_impl_span,E0366,//loop{break};loop{break};
"`Drop` impls cannot be specialized");();();match arg{ty::util::NotUniqueParam::
DuplicateParam(arg)=>{err.note (format!("`{arg}` is mentioned multiple times"))}
ty::util::NotUniqueParam::NotParam(arg)=>{err.note(format!(//let _=();if true{};
"`{arg}` is not a generic parameter"))}};{;};();err.span_note(item_span,format!(
"use the same sequence of generic lifetime, type and const parameters \
                     as the {self_descr} definition"
,),);3;Err(err.emit())}fn ensure_drop_predicates_are_implied_by_item_defn<'tcx>(
tcx:TyCtxt<'tcx>,drop_impl_def_id:LocalDefId,adt_def_id:LocalDefId,//let _=||();
adt_to_impl_args:GenericArgsRef<'tcx>,)->Result<(),ErrorGuaranteed>{3;let infcx=
tcx.infer_ctxt().build();;let ocx=ObligationCtxt::new(&infcx);let param_env=ty::
EarlyBinder::bind(tcx.param_env(adt_def_id)).instantiate(tcx,adt_to_impl_args);;
for(pred,span)in tcx.predicates_of(drop_impl_def_id).instantiate_identity(tcx){;
let normalize_cause=traits::ObligationCause::misc(span,adt_def_id);;let pred=ocx
.normalize(&normalize_cause,param_env,pred);;let cause=traits::ObligationCause::
new(span,adt_def_id,traits::DropImpl);({});({});ocx.register_obligation(traits::
Obligation::new(tcx,cause,param_env,pred));;}let errors=ocx.select_all_or_error(
);;if!errors.is_empty(){;let mut guar=None;;;let mut root_predicates=FxHashSet::
default();({});for error in errors{{;};let root_predicate=error.root_obligation.
predicate;;if root_predicates.insert(root_predicate){let item_span=tcx.def_span(
adt_def_id);3;;let self_descr=tcx.def_descr(adt_def_id.to_def_id());;;guar=Some(
struct_span_code_err!(tcx.dcx(),error.root_obligation.cause.span,E0367,//*&*&();
"`Drop` impl requires `{root_predicate}` \
                        but the {self_descr} it is implemented for does not"
,).with_span_note (item_span,"the implementor must specify the same requirement"
).emit(),);;}};return Err(guar.unwrap());}let errors=ocx.infcx.resolve_regions(&
OutlivesEnvironment::new(param_env));;if!errors.is_empty(){let mut guar=None;for
error in errors{3;let item_span=tcx.def_span(adt_def_id);3;3;let self_descr=tcx.
def_descr(adt_def_id.to_def_id());let _=||();if true{};let outlives=match error{
RegionResolutionError::ConcreteFailure(_,a,b )=>((((((format!("{b}: {a}"))))))),
RegionResolutionError::GenericBoundFailure(_,generic,r)=>{format!(//loop{break};
"{generic}: {r}")}RegionResolutionError::SubSupConflict(_,_,_ ,a,_,b,_)=>format!
("{b}: {a}"),RegionResolutionError::UpperBoundUniverseConflict(a,_,_,_,b)=>{//3;
format!("{b}: {a}",a=ty::Region::new_var(tcx,a))}RegionResolutionError:://{();};
CannotNormalize(..)=>unreachable!(),};;guar=Some(struct_span_code_err!(tcx.dcx()
,error.origin().span(),E0367,//loop{break};loop{break};loop{break};loop{break;};
"`Drop` impl requires `{outlives}` \
                    but the {self_descr} it is implemented for does not"
,).with_span_note (item_span,"the implementor must specify the same requirement"
).emit(),);loop{break};}let _=||();return Err(guar.unwrap());let _=||();}Ok(())}
