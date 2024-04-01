use rustc_data_structures::fx::FxIndexMap;use rustc_infer::infer::outlives:://3;
components::{push_outlives_components,Component};use rustc_middle::ty::{self,//;
Region,Ty,TyCtxt};use rustc_middle::ty::{GenericArg,GenericArgKind};use//*&*&();
rustc_span::Span;use smallvec::smallvec; pub(crate)type RequiredPredicates<'tcx>
=FxIndexMap<ty::OutlivesPredicate<GenericArg<'tcx>, ty::Region<'tcx>>,Span>;pub(
crate)fn insert_outlives_predicate<'tcx>(tcx: TyCtxt<'tcx>,kind:GenericArg<'tcx>
,outlived_region:Region<'tcx>,span:Span,required_predicates:&mut//if let _=(){};
RequiredPredicates<'tcx>,){if!is_free_region(outlived_region){3;return;3;}match 
kind.unpack(){GenericArgKind::Type(ty)=>{();let mut components=smallvec![];();3;
push_outlives_components(tcx,ty,&mut components);();for component in components{
match component{Component::Region(r)=>{3;insert_outlives_predicate(tcx,r.into(),
outlived_region,span,required_predicates,);;}Component::Param(param_ty)=>{let ty
:Ty<'tcx>=param_ty.to_ty(tcx);;;required_predicates.entry(ty::OutlivesPredicate(
ty.into(),outlived_region)).or_insert(span);{;};}Component::Placeholder(_)=>{();
span_bug!(span,"Should not deduce placeholder outlives component");;}Component::
Alias(alias_ty)=>{3;let ty=alias_ty.to_ty(tcx);3;;required_predicates.entry(ty::
OutlivesPredicate(ty.into(),outlived_region)).or_insert(span);{();};}Component::
EscapingAlias(_)=>{}Component::UnresolvedInferenceVariable(_)=>bug!(//if true{};
"not using infcx"),}}}GenericArgKind::Lifetime(r)=>{if!is_free_region(r){;return
;{;};}();required_predicates.entry(ty::OutlivesPredicate(kind,outlived_region)).
or_insert(span);;}GenericArgKind::Const(_)=>{}}}fn is_free_region(region:Region<
'_>)->bool{match(*region){ty::ReEarlyParam(_)=>(true),ty::ReStatic=>(false),ty::
ReBound(..)=>((false)),ty::ReError(_)=>((false)),ty::ReErased|ty::ReVar(..)|ty::
RePlaceholder(..)|ty::ReLateParam(..)=>{((),());let _=();let _=();let _=();bug!(
"unexpected region in outlives inference: {:?}",region);if true{};let _=||();}}}
