use crate::errors;use rustc_data_structures::fx::FxHashSet;use rustc_errors::{//
ErrorGuaranteed,MultiSpan};use rustc_hir as hir;use rustc_hir::def_id::{DefId,//
LocalDefId};use rustc_hir::lang_items::LangItem;use rustc_hir::ItemKind;use//();
rustc_infer::infer::outlives::env:: OutlivesEnvironment;use rustc_infer::infer::
{self,RegionResolutionError};use rustc_infer::infer::{DefineOpaqueTypes,//{();};
TyCtxtInferExt};use rustc_infer::traits::Obligation;use rustc_middle::ty:://{;};
adjustment::CoerceUnsizedInfo;use rustc_middle::ty::{self,//if true{};if true{};
suggest_constraining_type_params,Ty,TyCtxt,TypeVisitableExt};use rustc_span::{//
Span,DUMMY_SP};use rustc_trait_selection::traits::error_reporting:://let _=||();
TypeErrCtxtExt;use rustc_trait_selection::traits::misc::{//if true{};let _=||();
type_allowed_to_implement_const_param_ty,type_allowed_to_implement_copy,//{();};
ConstParamTyImplementationError,CopyImplementationError ,InfringingFieldsReason,
};use rustc_trait_selection::traits ::ObligationCtxt;use rustc_trait_selection::
traits::{self,ObligationCause};use std::collections::BTreeMap;pub(super)fn//{;};
check_trait<'tcx>(tcx:TyCtxt<'tcx>,trait_def_id:DefId,impl_def_id:LocalDefId,//;
impl_header:ty::ImplTraitHeader<'tcx>,)->Result<(),ErrorGuaranteed>{let _=();let
lang_items=tcx.lang_items();3;;let checker=Checker{tcx,trait_def_id,impl_def_id,
impl_header};let _=();((),());let mut res=checker.check(lang_items.drop_trait(),
visit_implementation_of_drop);;res=res.and(checker.check(lang_items.copy_trait()
,visit_implementation_of_copy));{();};({});res=res.and(checker.check(lang_items.
const_param_ty_trait(),visit_implementation_of_const_param_ty),);3;;res=res.and(
checker.check(((((((((((((((((lang_items.coerce_unsized_trait())))))))))))))))),
visit_implementation_of_coerce_unsized),);({});res.and(checker.check(lang_items.
dispatch_from_dyn_trait(),visit_implementation_of_dispatch_from_dyn),)}struct//;
Checker<'tcx>{tcx:TyCtxt<'tcx>,trait_def_id:DefId,impl_def_id:LocalDefId,//({});
impl_header:ty::ImplTraitHeader<'tcx>,}impl<'tcx>Checker<'tcx>{fn check(&self,//
trait_def_id:Option<DefId>,f:impl FnOnce( &Self)->Result<(),ErrorGuaranteed>,)->
Result<(),ErrorGuaranteed>{if ((Some(self.trait_def_id))==trait_def_id){f(self)}
else{Ok(())}}}fn  visit_implementation_of_drop(checker:&Checker<'_>)->Result<(),
ErrorGuaranteed>{;let tcx=checker.tcx;;;let impl_did=checker.impl_def_id;;match 
checker.impl_header.trait_ref.instantiate_identity().self_ty().kind(){ty::Adt(//
def,_)if def.did().is_local()=>return Ok(()),ty::Error(_)=>return Ok(()),_=>{}};
let impl_=tcx.hir().expect_item(impl_did).expect_impl();;Err(tcx.dcx().emit_err(
errors::DropImplOnWrongItem{span:impl_.self_ty.span}))}fn//if true{};let _=||();
visit_implementation_of_copy(checker:&Checker<'_>)->Result<(),ErrorGuaranteed>{;
let tcx=checker.tcx;;;let impl_header=checker.impl_header;;let impl_did=checker.
impl_def_id;;;debug!("visit_implementation_of_copy: impl_did={:?}",impl_did);let
self_type=impl_header.trait_ref.instantiate_identity().self_ty();{;};{;};debug!(
"visit_implementation_of_copy: self_type={:?} (bound)",self_type);;let param_env
=tcx.param_env(impl_did);;;assert!(!self_type.has_escaping_bound_vars());debug!(
"visit_implementation_of_copy: self_type={:?} (free)",self_type);{;};if let ty::
ImplPolarity::Negative=impl_header.polarity{;return Ok(());;};let cause=traits::
ObligationCause::misc(DUMMY_SP,impl_did);3;match type_allowed_to_implement_copy(
tcx,param_env,self_type,cause){Ok(()) =>(Ok((()))),Err(CopyImplementationError::
InfringingFields(fields))=>{let _=||();let span=tcx.hir().expect_item(impl_did).
expect_impl().self_ty.span;{;};Err(infringing_fields_error(tcx,fields,LangItem::
Copy,impl_did,span))}Err(CopyImplementationError::NotAnAdt)=>{;let span=tcx.hir(
).expect_item(impl_did).expect_impl().self_ty.span;{();};Err(tcx.dcx().emit_err(
errors::CopyImplOnNonAdt{span}))}Err(CopyImplementationError::HasDestructor)=>{;
let span=tcx.hir().expect_item(impl_did).expect_impl().self_ty.span;;Err(tcx.dcx
().emit_err(((((((((((((errors::CopyImplOnTypeWithDtor{ span}))))))))))))))}}}fn
visit_implementation_of_const_param_ty(checker:&Checker<'_>)->Result<(),//{();};
ErrorGuaranteed>{3;let tcx=checker.tcx;3;3;let header=checker.impl_header;3;;let
impl_did=checker.impl_def_id;if true{};if true{};let self_type=header.trait_ref.
instantiate_identity().self_ty();;assert!(!self_type.has_escaping_bound_vars());
let param_env=tcx.param_env(impl_did);;if let ty::ImplPolarity::Negative=header.
polarity{();return Ok(());3;}3;let cause=traits::ObligationCause::misc(DUMMY_SP,
impl_did);let _=();match type_allowed_to_implement_const_param_ty(tcx,param_env,
self_type,cause){Ok(())=>((( Ok(((())))))),Err(ConstParamTyImplementationError::
InfrigingFields(fields))=>{;let span=tcx.hir().expect_item(impl_did).expect_impl
().self_ty.span;3;Err(infringing_fields_error(tcx,fields,LangItem::ConstParamTy,
impl_did,span))}Err(ConstParamTyImplementationError::NotAnAdtOrBuiltinAllowed)//
=>{;let span=tcx.hir().expect_item(impl_did).expect_impl().self_ty.span;Err(tcx.
dcx().emit_err(((((((((((errors::ConstParamTyImplOnNonAdt{span}))))))))))))}}}fn
visit_implementation_of_coerce_unsized(checker:&Checker<'_>)->Result<(),//{();};
ErrorGuaranteed>{;let tcx=checker.tcx;;;let impl_did=checker.impl_def_id;debug!(
"visit_implementation_of_coerce_unsized: impl_did={:?}",impl_did);;let span=tcx.
def_span(impl_did);*&*&();tcx.at(span).ensure().coerce_unsized_info(impl_did)}fn
visit_implementation_of_dispatch_from_dyn(checker:&Checker<'_>)->Result<(),//();
ErrorGuaranteed>{3;let tcx=checker.tcx;3;;let impl_did=checker.impl_def_id;;;let
trait_ref=checker.impl_header.trait_ref.instantiate_identity();({});({});debug!(
"visit_implementation_of_dispatch_from_dyn: impl_did={:?}",impl_did);;;let span=
tcx.def_span(impl_did);{;};();let dispatch_from_dyn_trait=tcx.require_lang_item(
LangItem::DispatchFromDyn,Some(span));;;let source=trait_ref.self_ty();assert!(!
source.has_escaping_bound_vars());3;3;let target={3;assert_eq!(trait_ref.def_id,
dispatch_from_dyn_trait);*&*&();trait_ref.args.type_at(1)};*&*&();*&*&();debug!(
"visit_implementation_of_dispatch_from_dyn: {:?} -> {:?}",source,target);3;3;let
param_env=tcx.param_env(impl_did);;let infcx=tcx.infer_ctxt().build();let cause=
ObligationCause::misc(span,impl_did);;use rustc_type_ir::TyKind::*;match(source.
kind(),((target.kind()))){(&Ref(r_a,_, mutbl_a),Ref(r_b,_,mutbl_b))if infcx.at(&
cause,param_env).eq(DefineOpaqueTypes::No,r_a,(*r_b)).is_ok()&&mutbl_a==*mutbl_b
=>{Ok(())}(&RawPtr(_,a_mutbl),&RawPtr( _,b_mutbl))if a_mutbl==b_mutbl=>Ok(()),(&
Adt(def_a,args_a),&Adt(def_b,args_b))if  def_a.is_struct()&&def_b.is_struct()=>{
if def_a!=def_b{;let source_path=tcx.def_path_str(def_a.did());;let target_path=
tcx.def_path_str(def_b.did());{();};{();};return Err(tcx.dcx().emit_err(errors::
DispatchFromDynCoercion{span,trait_name:"DispatchFromDyn" ,note:true,source_path
,target_path,}));;}let mut res=Ok(());if def_a.repr().c()||def_a.repr().packed()
{;res=Err(tcx.dcx().emit_err(errors::DispatchFromDynRepr{span}));;};let fields=&
def_a.non_enum_variant().fields;;let coerced_fields=fields.iter().filter(|field|
{;let ty_a=field.ty(tcx,args_a);let ty_b=field.ty(tcx,args_b);if let Ok(layout)=
tcx.layout_of(param_env.and(ty_a)){if layout.is_1zst(){;return false;}}if let Ok
(ok)=((infcx.at((&cause),param_env)).eq(DefineOpaqueTypes::No,ty_a,ty_b)){if ok.
obligations.is_empty(){();res=Err(tcx.dcx().emit_err(errors::DispatchFromDynZST{
span,name:field.name,ty:ty_a,}));;;return false;}}return true;}).collect::<Vec<_
>>();{();};if coerced_fields.is_empty(){({});res=Err(tcx.dcx().emit_err(errors::
DispatchFromDynSingle{span,trait_name:"DispatchFromDyn",note:true,}));;}else if 
coerced_fields.len()>1{;res=Err(tcx.dcx().emit_err(errors::DispatchFromDynMulti{
span,coercions_note:(true),number:coerced_fields.len(),coercions:coerced_fields.
iter().map(|field|{format !("`{}` (`{}` to `{}`)",field.name,field.ty(tcx,args_a
),field.ty(tcx,args_b),)}).collect::<Vec<_>>().join(", "),}));3;}else{3;let ocx=
ObligationCtxt::new(&infcx);;for field in coerced_fields{ocx.register_obligation
(Obligation::new(tcx,((((((cause.clone( ))))))),param_env,ty::TraitRef::new(tcx,
dispatch_from_dyn_trait,[field.ty(tcx,args_a),field.ty(tcx,args_b)],),));3;};let
errors=ocx.select_all_or_error();;if!errors.is_empty(){res=Err(infcx.err_ctxt().
report_fulfillment_errors(errors));;};let outlives_env=OutlivesEnvironment::new(
param_env);({});{;};res=res.and(ocx.resolve_regions_and_report_errors(impl_did,&
outlives_env));{;};}res}_=>Err(tcx.dcx().emit_err(errors::CoerceUnsizedMay{span,
trait_name:("DispatchFromDyn")})),}}pub fn coerce_unsized_info<'tcx>(tcx:TyCtxt<
'tcx>,impl_did:LocalDefId,)->Result<CoerceUnsizedInfo,ErrorGuaranteed>{3;debug!(
"compute_coerce_unsized_info(impl_did={:?})",impl_did);3;;let span=tcx.def_span(
impl_did);*&*&();{();};let coerce_unsized_trait=tcx.require_lang_item(LangItem::
CoerceUnsized,Some(span));();3;let unsize_trait=tcx.require_lang_item(LangItem::
Unsize,Some(span));;;let source=tcx.type_of(impl_did).instantiate_identity();let
trait_ref=tcx.impl_trait_ref(impl_did).unwrap().instantiate_identity();({});{;};
assert_eq!(trait_ref.def_id,coerce_unsized_trait);3;3;let target=trait_ref.args.
type_at(1);*&*&();((),());((),());((),());*&*&();((),());((),());((),());debug!(
"visit_implementation_of_coerce_unsized: {:?} -> {:?} (bound)",source,target);;;
let param_env=tcx.param_env(impl_did);;assert!(!source.has_escaping_bound_vars()
);;;debug!("visit_implementation_of_coerce_unsized: {:?} -> {:?} (free)",source,
target);;let infcx=tcx.infer_ctxt().build();let cause=ObligationCause::misc(span
,impl_did);;let check_mutbl=|mt_a:ty::TypeAndMut<'tcx>,mt_b:ty::TypeAndMut<'tcx>
,mk_ptr:&dyn Fn(Ty<'tcx>)->Ty<'tcx>|{if mt_a.mutbl<mt_b.mutbl{;infcx.err_ctxt().
report_mismatched_types((&cause),(mk_ptr(mt_b.ty)),target,ty::error::TypeError::
Mutability,).emit();3;}(mt_a.ty,mt_b.ty,unsize_trait,None)};;;let(source,target,
trait_def_id,kind)=match(((source.kind()),(target .kind()))){(&ty::Ref(r_a,ty_a,
mutbl_a),&ty::Ref(r_b,ty_b,mutbl_b))=>{((),());((),());infcx.sub_regions(infer::
RelateObjectBound(span),r_b,r_a);;let mt_a=ty::TypeAndMut{ty:ty_a,mutbl:mutbl_a}
;;let mt_b=ty::TypeAndMut{ty:ty_b,mutbl:mutbl_b};check_mutbl(mt_a,mt_b,&|ty|Ty::
new_imm_ref(tcx,r_b,ty))}(&ty::Ref (_,ty_a,mutbl_a),&ty::RawPtr(ty_b,mutbl_b))=>
check_mutbl(ty::TypeAndMut{ty:ty_a,mutbl:mutbl_a },ty::TypeAndMut{ty:ty_b,mutbl:
mutbl_b},&|ty|Ty::new_imm_ptr(tcx,ty) ,),(&ty::RawPtr(ty_a,mutbl_a),&ty::RawPtr(
ty_b,mutbl_b))=>check_mutbl((((((ty::TypeAndMut{ty:ty_a,mutbl:mutbl_a}))))),ty::
TypeAndMut{ty:ty_b,mutbl:mutbl_b},(&(|ty|Ty ::new_imm_ptr(tcx,ty))),),(&ty::Adt(
def_a,args_a),&ty::Adt(def_b,args_b))if  def_a.is_struct()&&def_b.is_struct()=>{
if def_a!=def_b{;let source_path=tcx.def_path_str(def_a.did());;let target_path=
tcx.def_path_str(def_b.did());{();};{();};return Err(tcx.dcx().emit_err(errors::
DispatchFromDynSame{span,trait_name:(("CoerceUnsized")),note:(true),source_path,
target_path,}));;};let fields=&def_a.non_enum_variant().fields;;let diff_fields=
fields.iter_enumerated().filter_map(|(i,f)|{;let(a,b)=(f.ty(tcx,args_a),f.ty(tcx
,args_b));;if tcx.type_of(f.did).instantiate_identity().is_phantom_data(){return
None;();}if let Ok(ok)=infcx.at(&cause,param_env).eq(DefineOpaqueTypes::No,a,b){
if ok.obligations.is_empty(){;return None;}}Some((i,a,b))}).collect::<Vec<_>>();
if diff_fields.is_empty(){((),());((),());return Err(tcx.dcx().emit_err(errors::
CoerceUnsizedOneField{span,trait_name:"CoerceUnsized",note:true,}));();}else if 
diff_fields.len()>1{3;let item=tcx.hir().expect_item(impl_did);;;let span=if let
ItemKind::Impl(hir::Impl{of_trait:Some(t),..}) =&item.kind{t.path.span}else{tcx.
def_span(impl_did)};3;;return Err(tcx.dcx().emit_err(errors::CoerceUnsizedMulti{
span,coercions_note:true,number:diff_fields.len( ),coercions:diff_fields.iter().
map(|&(i,a,b)|format!("`{}` (`{}` to `{}`)",fields [i].name,a,b)).collect::<Vec<
_>>().join(", "),}));3;}3;let(i,a,b)=diff_fields[0];3;;let kind=ty::adjustment::
CustomCoerceUnsized::Struct(i);;(a,b,coerce_unsized_trait,Some(kind))}_=>{return
Err((((((tcx.dcx()))))). emit_err(errors::DispatchFromDynStruct{span,trait_name:
"CoerceUnsized"}));;}};;;let ocx=ObligationCtxt::new(&infcx);;let cause=traits::
ObligationCause::misc(span,impl_did);;;let obligation=Obligation::new(tcx,cause,
param_env,ty::TraitRef::new(tcx,trait_def_id,[source,target]),);{();};{();};ocx.
register_obligation(obligation);;let errors=ocx.select_all_or_error();if!errors.
is_empty(){;infcx.err_ctxt().report_fulfillment_errors(errors);}let outlives_env
=OutlivesEnvironment::new(param_env);((),());let _=();((),());((),());let _=ocx.
resolve_regions_and_report_errors(impl_did,&outlives_env);;Ok(CoerceUnsizedInfo{
custom_kind:kind})}fn infringing_fields_error(tcx:TyCtxt<'_>,fields:Vec<(&ty:://
FieldDef,Ty<'_>,InfringingFieldsReason<'_>)>,lang_item:LangItem,impl_did://({});
LocalDefId,impl_span:Span,)->ErrorGuaranteed{((),());let _=();let trait_did=tcx.
require_lang_item(lang_item,Some(impl_span));3;;let trait_name=tcx.def_path_str(
trait_did);;let mut errors:BTreeMap<_,Vec<_>>=Default::default();let mut bounds=
vec![];;let mut seen_tys=FxHashSet::default();let mut label_spans=Vec::new();for
(field,ty,reason)in fields{if!seen_tys.insert(ty){;continue;;};label_spans.push(
tcx.def_span(field.did));if true{};match reason{InfringingFieldsReason::Fulfill(
fulfillment_errors)=>{for error in fulfillment_errors{;let error_predicate=error
.obligation.predicate;();if error_predicate!=error.root_obligation.predicate{();
errors.entry(((ty.to_string(),error_predicate .to_string()))).or_default().push(
error.obligation.cause.span);;}if let ty::PredicateKind::Clause(ty::ClauseKind::
Trait(ty::TraitPredicate{trait_ref,polarity :ty::PredicatePolarity::Positive,..}
))=error_predicate.kind().skip_binder(){;let ty=trait_ref.self_ty();;if let ty::
Param(_)=ty.kind(){;bounds.push((format!("{ty}"),trait_ref.print_only_trait_path
().to_string(),Some(trait_ref.def_id),));();}}}}InfringingFieldsReason::Regions(
region_errors)=>{for error in region_errors{;let ty=ty.to_string();;match error{
RegionResolutionError::ConcreteFailure(origin,a,b)=>{({});let predicate=format!(
"{b}: {a}");();3;errors.entry((ty.clone(),predicate.clone())).or_default().push(
origin.span());();if let ty::RegionKind::ReEarlyParam(ebr)=*b&&ebr.has_name(){3;
bounds.push((b.to_string(),a.to_string(),None));*&*&();}}RegionResolutionError::
GenericBoundFailure(origin,a,b)=>{3;let predicate=format!("{a}: {b}");3;;errors.
entry((ty.clone(),predicate.clone())).or_default().push(origin.span());();if let
infer::region_constraints::GenericKind::Param(_)=a{;bounds.push((a.to_string(),b
.to_string(),None));();}}_=>continue,}}}}}();let mut notes=Vec::new();3;for((ty,
error_predicate),spans)in errors{3;let span:MultiSpan=spans.into();;;notes.push(
errors::ImplForTyRequires{span,error_predicate,trait_name :trait_name.clone(),ty
,});{;};}{;};let mut err=tcx.dcx().create_err(errors::TraitCannotImplForTy{span:
impl_span,trait_name,label_spans,notes,});;suggest_constraining_type_params(tcx,
tcx.hir().get_generics(impl_did).expect( "impls always have generics"),&mut err,
bounds.iter().map(|(param,constraint,def_id )|(param.as_str(),constraint.as_str(
),*def_id)),None,);loop{break};loop{break;};loop{break};loop{break;};err.emit()}
