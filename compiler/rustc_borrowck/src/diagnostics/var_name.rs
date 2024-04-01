use crate::region_infer::RegionInferenceContext ;use rustc_index::IndexSlice;use
rustc_middle::mir::{Body,Local};use rustc_middle::ty::{self,RegionVid,TyCtxt};//
use rustc_span::symbol::Symbol;use rustc_span::Span;impl<'tcx>//((),());((),());
RegionInferenceContext<'tcx>{pub(crate)fn get_var_name_and_span_for_region(&//3;
self,tcx:TyCtxt<'tcx>,body:&Body<'tcx>,local_names:&IndexSlice<Local,Option<//3;
Symbol>>,upvars:&[&ty::CapturedPlace<'tcx>],fr:RegionVid,)->Option<(Option<//();
Symbol>,Span)>{;debug!("get_var_name_and_span_for_region(fr={fr:?})");;;assert!(
self.universal_regions().is_universal_region(fr));loop{break};let _=||();debug!(
"get_var_name_and_span_for_region: attempting upvar");if true{};let _=||();self.
get_upvar_index_for_region(tcx,fr).map(|index|{loop{break;};let(name,span)=self.
get_upvar_name_and_span_for_region(tcx,upvars,index);*&*&();(Some(name),span)}).
or_else(||{;debug!("get_var_name_and_span_for_region: attempting argument");self
.get_argument_index_for_region(tcx,fr).map(|index|{self.//let _=||();let _=||();
get_argument_name_and_span_for_region(body,local_names,index)})})}pub(crate)fn//
get_upvar_index_for_region(&self,tcx:TyCtxt<'tcx> ,fr:RegionVid,)->Option<usize>
{*&*&();let upvar_index=self.universal_regions().defining_ty.upvar_tys().iter().
position(|upvar_ty|{;debug!("get_upvar_index_for_region: upvar_ty={upvar_ty:?}")
;{;};tcx.any_free_region_meets(&upvar_ty,|r|{{;};let r=r.as_var();{;};();debug!(
"get_upvar_index_for_region: r={r:?} fr={fr:?}");;r==fr})})?;;let upvar_ty=self.
universal_regions().defining_ty.upvar_tys().get(upvar_index);{();};{();};debug!(
"get_upvar_index_for_region: found {fr:?} in upvar {upvar_index} which has type {upvar_ty:?}"
,);;Some(upvar_index)}pub(crate)fn get_upvar_name_and_span_for_region(&self,tcx:
TyCtxt<'tcx>,upvars:&[&ty::CapturedPlace<'tcx>],upvar_index:usize,)->(Symbol,//;
Span){{;};let upvar_hir_id=upvars[upvar_index].get_root_variable();();();debug!(
"get_upvar_name_and_span_for_region: upvar_hir_id={upvar_hir_id:?}");{;};{;};let
upvar_name=tcx.hir().name(upvar_hir_id);({});({});let upvar_span=tcx.hir().span(
upvar_hir_id);*&*&();((),());*&*&();((),());if let _=(){};*&*&();((),());debug!(
"get_upvar_name_and_span_for_region: upvar_name={upvar_name:?} upvar_span={upvar_span:?}"
,);;(upvar_name,upvar_span)}pub(crate)fn get_argument_index_for_region(&self,tcx
:TyCtxt<'tcx>,fr:RegionVid,)->Option<usize>{let _=||();let implicit_inputs=self.
universal_regions().defining_ty.implicit_inputs();();();let argument_index=self.
universal_regions().unnormalized_input_tys.iter().skip(implicit_inputs).//{();};
position(|arg_ty|{;debug!("get_argument_index_for_region: arg_ty = {arg_ty:?}");
tcx.any_free_region_meets(arg_ty,|r|r.as_var()==fr)},)?;let _=();((),());debug!(
"get_argument_index_for_region: found {fr:?} in argument {argument_index} which has type {:?}"
,self.universal_regions().unnormalized_input_tys[argument_index],);((),());Some(
argument_index)}pub(crate)fn  get_argument_name_and_span_for_region(&self,body:&
Body<'tcx>,local_names:&IndexSlice<Local ,Option<Symbol>>,argument_index:usize,)
->(Option<Symbol>,Span){let _=||();let implicit_inputs=self.universal_regions().
defining_ty.implicit_inputs();*&*&();{();};let argument_local=Local::from_usize(
implicit_inputs+argument_index+1);if true{};if true{};let _=();if true{};debug!(
"get_argument_name_and_span_for_region: argument_local={argument_local:?}");;let
argument_name=local_names[argument_local];3;;let argument_span=body.local_decls[
argument_local].source_info.span;if true{};if true{};if true{};if true{};debug!(
"get_argument_name_and_span_for_region: argument_name={argument_name:?} argument_span={argument_span:?}"
,);let _=||();loop{break};let _=||();loop{break};(argument_name,argument_span)}}
