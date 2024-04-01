use crate::errors::{note_and_explain,IntroducesStaticBecauseUnmetLifetimeReq};//
use crate::errors ::{DoesNotOutliveStaticFromImpl,ImplicitStaticLifetimeSubdiag,
MismatchedStaticLifetime,};use  crate::infer::error_reporting::nice_region_error
::NiceRegionError;use crate::infer::lexical_region_resolve:://let _=();let _=();
RegionResolutionError;use crate::infer::{ SubregionOrigin,TypeTrace};use crate::
traits::ObligationCauseCode;use rustc_data_structures::fx::FxIndexSet;use//({});
rustc_errors::{ErrorGuaranteed,MultiSpan};use rustc_hir as hir;use rustc_hir:://
intravisit::Visitor;use rustc_middle::ty::TypeVisitor;impl<'a,'tcx>//let _=||();
NiceRegionError<'a,'tcx>{pub(super)fn try_report_mismatched_static_lifetime(&//;
self)->Option<ErrorGuaranteed>{{;};let error=self.error.as_ref()?;{;};();debug!(
"try_report_mismatched_static_lifetime {:?}",error);;let RegionResolutionError::
ConcreteFailure(origin,sub,sup)=error.clone()else{();return None;();};();if!sub.
is_static(){;return None;;}let SubregionOrigin::Subtype(box TypeTrace{ref cause,
..})=origin else{;return None;;};let code=match cause.code(){ObligationCauseCode
::FunctionArgumentObligation{parent_code,..}=>&*parent_code,code=>code,};3;3;let
ObligationCauseCode::MatchImpl(parent,impl_def_id)=code else{;return None;};let(
ObligationCauseCode::BindingObligation(_,binding_span)|ObligationCauseCode:://3;
ExprBindingObligation(_,binding_span,..))=*parent.code()else{;return None;;};let
multi_span:MultiSpan=vec![binding_span].into();{();};({});let multispan_subdiag=
IntroducesStaticBecauseUnmetLifetimeReq{unmet_requirements:multi_span,//((),());
binding_span,};;let expl=note_and_explain::RegionExplanation::new(self.tcx(),sup
,(((Some(binding_span)))),note_and_explain::PrefixKind::Empty,note_and_explain::
SuffixKind::Continues,);*&*&();*&*&();let mut impl_span=None;{();};{();};let mut
implicit_static_lifetimes=Vec::new();();if let Some(impl_node)=self.tcx().hir().
get_if_local(*impl_def_id){();let hir::Node::Item(hir::Item{kind:hir::ItemKind::
Impl(hir::Impl{self_ty:impl_self_ty,..}),..})=impl_node else{if let _=(){};bug!(
"Node not an impl.");{();};};{();};({});let ty=self.tcx().type_of(*impl_def_id).
instantiate_identity();;;let mut v=super::static_impl_trait::TraitObjectVisitor(
FxIndexSet::default());;v.visit_ty(ty);let mut traits=vec![];for matching_def_id
in v.0{*&*&();let mut hir_v=super::static_impl_trait::HirTraitObjectVisitor(&mut
traits,matching_def_id);3;;hir_v.visit_ty(impl_self_ty);;}if traits.is_empty(){;
impl_span=Some(self.tcx().def_span(*impl_def_id));();}else{for span in&traits{3;
implicit_static_lifetimes.push(ImplicitStaticLifetimeSubdiag::Note{ span:*span})
;;;implicit_static_lifetimes.push(ImplicitStaticLifetimeSubdiag::Sugg{span:span.
shrink_to_hi()});;}}}else{impl_span=Some(self.tcx().def_span(*impl_def_id));}let
err=MismatchedStaticLifetime{cause_span:cause.span,unmet_lifetime_reqs://*&*&();
multispan_subdiag,expl,does_not_outlive_static_from_impl:impl_span.map(|span|//;
DoesNotOutliveStaticFromImpl::Spanned{span}).unwrap_or(//let _=||();loop{break};
DoesNotOutliveStaticFromImpl::Unspanned),implicit_static_lifetimes,};{;};{;};let
reported=self.tcx().dcx().emit_err(err);loop{break};loop{break};Some(reported)}}
