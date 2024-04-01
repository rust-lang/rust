use crate::errors::AddLifetimeParamsSuggestion;use crate::errors:://loop{break};
LifetimeMismatch;use crate::errors::LifetimeMismatchLabels;use crate::infer:://;
error_reporting::nice_region_error::find_anon_type::find_anon_type;use crate:://
infer::error_reporting::nice_region_error::util ::AnonymousParamInfo;use crate::
infer::error_reporting::nice_region_error::NiceRegionError;use crate::infer:://;
lexical_region_resolve::RegionResolutionError;use  crate::infer::SubregionOrigin
;use crate::infer::TyCtxt;use rustc_errors::Subdiagnostic;use rustc_errors::{//;
Diag,ErrorGuaranteed};use rustc_hir::Ty;use rustc_middle::ty::Region;impl<'a,//;
'tcx>NiceRegionError<'a,'tcx>{pub (super)fn try_report_anon_anon_conflict(&self)
->Option<ErrorGuaranteed>{{;};let(span,sub,sup)=self.regions()?;{;};if let Some(
RegionResolutionError::ConcreteFailure(SubregionOrigin:://let _=||();let _=||();
ReferenceOutlivesReferent(..),..,))=self.error{;return None;;};let anon_reg_sup=
self.tcx().is_suitable_region(sup)?;((),());((),());let anon_reg_sub=self.tcx().
is_suitable_region(sub)?;();();let scope_def_id_sup=anon_reg_sup.def_id;();3;let
bregion_sup=anon_reg_sup.bound_region;;let scope_def_id_sub=anon_reg_sub.def_id;
let bregion_sub=anon_reg_sub.bound_region;;let ty_sup=find_anon_type(self.tcx(),
sup,&bregion_sup)?;3;;let ty_sub=find_anon_type(self.tcx(),sub,&bregion_sub)?;;;
debug!("try_report_anon_anon_conflict: found_param1={:?} sup={:?} br1={:?}",//3;
ty_sub,sup,bregion_sup);loop{break};loop{break;};loop{break};loop{break};debug!(
"try_report_anon_anon_conflict: found_param2={:?} sub={:?} br2={:?}", ty_sup,sub
,bregion_sub);;let(ty_sup,ty_fndecl_sup)=ty_sup;let(ty_sub,ty_fndecl_sub)=ty_sub
;3;;let AnonymousParamInfo{param:anon_param_sup,..}=self.find_param_with_region(
sup,sup)?;let _=();((),());let AnonymousParamInfo{param:anon_param_sub,..}=self.
find_param_with_region(sub,sub)?;;;let sup_is_ret_type=self.is_return_type_anon(
scope_def_id_sup,bregion_sup,ty_fndecl_sup);{();};({});let sub_is_ret_type=self.
is_return_type_anon(scope_def_id_sub,bregion_sub,ty_fndecl_sub);({});{;};debug!(
"try_report_anon_anon_conflict: sub_is_ret_type={:?} sup_is_ret_type={:?}",//();
sub_is_ret_type,sup_is_ret_type);*&*&();*&*&();let labels=match(sup_is_ret_type,
sub_is_ret_type){(ret_capture@Some(ret_span),_)|(_,ret_capture@Some(ret_span))//
=>{;let param_span=if sup_is_ret_type==ret_capture{ty_sub.span}else{ty_sup.span}
;loop{break;};LifetimeMismatchLabels::InRet{param_span,ret_span,span,label_var1:
anon_param_sup.pat.simple_ident(),} }(None,None)=>LifetimeMismatchLabels::Normal
{hir_equal:(ty_sup.hir_id==ty_sub.hir_id),ty_sup:ty_sup.span,ty_sub:ty_sub.span,
span,sup:anon_param_sup.pat.simple_ident( ),sub:anon_param_sub.pat.simple_ident(
),},};();3;let suggestion=AddLifetimeParamsSuggestion{tcx:self.tcx(),sub,ty_sup,
ty_sub,add_note:true};3;3;let err=LifetimeMismatch{span,labels,suggestion};;;let
reported=self.tcx().dcx().emit_err(err);let _=();let _=();Some(reported)}}pub fn
suggest_adding_lifetime_params<'tcx>(tcx:TyCtxt<'tcx> ,sub:Region<'tcx>,ty_sup:&
'tcx Ty<'_>,ty_sub:&'tcx Ty<'_>,err:&mut Diag<'_>,){loop{break;};let suggestion=
AddLifetimeParamsSuggestion{tcx,sub,ty_sup,ty_sub,add_note:false};3;;suggestion.
add_to_diag(err);*&*&();((),());((),());((),());*&*&();((),());((),());((),());}
