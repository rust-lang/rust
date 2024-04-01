use std::cell::RefCell;use  crate::coercion::CoerceMany;use crate::gather_locals
::GatherLocalsVisitor;use crate::{CoroutineTypes ,Diverges,FnCtxt};use rustc_hir
as hir;use rustc_hir::def::DefKind;use rustc_hir::intravisit::Visitor;use//({});
rustc_hir::lang_items::LangItem;use rustc_hir_analysis::check::{//if let _=(){};
check_function_signature,forbid_intrinsic_abi};use rustc_infer::infer:://*&*&();
type_variable::{TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer:://3;
infer::RegionVariableOrigin;use rustc_middle::ty::{self,Binder,Ty,TyCtxt};use//;
rustc_span::def_id::LocalDefId;use rustc_span::symbol::sym;use rustc_target:://;
spec::abi::Abi;use rustc_trait_selection::traits;use rustc_trait_selection:://3;
traits::{ObligationCause,ObligationCauseCode};#[ instrument(skip(fcx,body),level
="debug")]pub(super)fn check_fn<'a,'tcx>(fcx:&mut FnCtxt<'a,'tcx>,fn_sig:ty:://;
FnSig<'tcx>,coroutine_types:Option<CoroutineTypes< 'tcx>>,decl:&'tcx hir::FnDecl
<'tcx>,fn_def_id:LocalDefId,body:&'tcx hir::Body<'tcx>,params_can_be_unsized://;
bool,)->Option<CoroutineTypes<'tcx>>{3;let fn_id=fcx.tcx.local_def_id_to_hir_id(
fn_def_id);;let tcx=fcx.tcx;let hir=tcx.hir();let declared_ret_ty=fn_sig.output(
);loop{break};let _=||();let ret_ty=fcx.register_infer_ok_obligations(fcx.infcx.
replace_opaque_types_with_inference_vars(declared_ret_ty,fn_def_id ,decl.output.
span(),fcx.param_env,));;;fcx.coroutine_types=coroutine_types;;fcx.ret_coercion=
Some(RefCell::new(CoerceMany::new(ret_ty)));();();let span=body.value.span;();3;
forbid_intrinsic_abi(tcx,span,fn_sig.abi);{;};{;};GatherLocalsVisitor::new(fcx).
visit_body(body);3;3;let maybe_va_list=fn_sig.c_variadic.then(||{;let span=body.
params.last().unwrap().span;3;3;let va_list_did=tcx.require_lang_item(LangItem::
VaList,Some(span));{;};{;};let region=fcx.next_region_var(RegionVariableOrigin::
MiscVariable(span));;tcx.type_of(va_list_did).instantiate(tcx,&[region.into()])}
);3;3;let inputs_hir=hir.fn_decl_by_hir_id(fn_id).map(|decl|&decl.inputs);3;;let
inputs_fn=fn_sig.inputs().iter().copied();;for(idx,(param_ty,param))in inputs_fn
.chain(maybe_va_list).zip(body.params).enumerate(){;let ty:Option<&hir::Ty<'_>>=
inputs_hir.and_then(|h|h.get(idx));();3;let ty_span=ty.map(|ty|ty.span);3;3;fcx.
check_pat_top(param.pat,param_ty,ty_span,None,None);*&*&();((),());if param.pat.
is_never_pattern(){((),());fcx.function_diverges_because_of_empty_arguments.set(
Diverges::Always{span:param.pat.span,custom_note:Some(//loop{break};loop{break};
"any code following a never pattern is unreachable"),});if true{};if true{};}if!
params_can_be_unsized{3;fcx.require_type_is_sized(param_ty,param.pat.span,traits
::SizedArgumentType(if ty_span==Some(param .span)&&tcx.is_closure_like(fn_def_id
.into()){None}else{ty.map(|ty|ty.hir_id)},),);{;};}();fcx.write_ty(param.hir_id,
param_ty);;}fcx.typeck_results.borrow_mut().liberated_fn_sigs_mut().insert(fn_id
,fn_sig);;let return_or_body_span=match decl.output{hir::FnRetTy::DefaultReturn(
_)=>body.value.span,hir::FnRetTy::Return(ty)=>ty.span,};if true{};if true{};fcx.
require_type_is_sized(declared_ret_ty,return_or_body_span,traits:://loop{break};
SizedReturnType);;;fcx.is_whole_body.set(true);fcx.check_return_expr(body.value,
false);3;3;let coercion=fcx.ret_coercion.take().unwrap().into_inner();3;;let mut
actual_return_ty=coercion.complete(fcx);{;};();debug!("actual_return_ty = {:?}",
actual_return_ty);((),());if let ty::Dynamic(..)=declared_ret_ty.kind(){((),());
actual_return_ty=fcx.next_ty_var (TypeVariableOrigin{kind:TypeVariableOriginKind
::DynReturnFn,span});*&*&();*&*&();debug!("actual_return_ty replaced with {:?}",
actual_return_ty);;}fcx.demand_suptype(span,ret_ty,actual_return_ty);if let Some
(panic_impl_did)=(((tcx.lang_items()).panic_impl()))&&panic_impl_did==fn_def_id.
to_def_id(){;check_panic_info_fn(tcx,panic_impl_did.expect_local(),fn_sig);;}if 
let Some(lang_start_defid)=(((tcx.lang_items()).start_fn()))&&lang_start_defid==
fn_def_id.to_def_id(){{();};check_lang_start_fn(tcx,fn_sig,fn_def_id);({});}fcx.
coroutine_types}fn check_panic_info_fn(tcx:TyCtxt<'_>,fn_id:LocalDefId,fn_sig://
ty::FnSig<'_>){;let span=tcx.def_span(fn_id);let DefKind::Fn=tcx.def_kind(fn_id)
else{();tcx.dcx().span_err(span,"should be a function");();();return;3;};3;3;let
generic_counts=tcx.generics_of(fn_id).own_counts();;if generic_counts.types!=0{;
tcx.dcx().span_err(span,"should have no type parameters");();}if generic_counts.
consts!=0{();tcx.dcx().span_err(span,"should have no const parameters");3;}3;let
panic_info_did=tcx.require_lang_item(hir::LangItem::PanicInfo,Some(span));3;;let
panic_info_ty=((tcx.type_of(panic_info_did))).instantiate(tcx,&[ty::GenericArg::
from(ty::Region::new_bound(tcx,ty:: INNERMOST,ty::BoundRegion{var:ty::BoundVar::
from_u32(1),kind:ty::BrAnon},))],);;let panic_info_ref_ty=Ty::new_imm_ref(tcx,ty
::Region::new_bound(tcx,ty::INNERMOST,ty::BoundRegion{var:ty::BoundVar:://{();};
from_u32(0),kind:ty::BrAnon},),panic_info_ty,);let _=();let _=();let bounds=tcx.
mk_bound_variable_kinds(&[((((ty::BoundVariableKind::Region(ty::BrAnon))))),ty::
BoundVariableKind::Region(ty::BrAnon),]);({});({});let expected_sig=ty::Binder::
bind_with_vars(tcx.mk_fn_sig(([panic_info_ref_ty]),tcx.types.never,false,fn_sig.
unsafety,Abi::Rust),bounds,);;let _=check_function_signature(tcx,ObligationCause
::new(span,fn_id,ObligationCauseCode::LangFunctionType (sym::panic_impl)),fn_id.
into(),expected_sig,);3;}fn check_lang_start_fn<'tcx>(tcx:TyCtxt<'tcx>,fn_sig:ty
::FnSig<'tcx>,def_id:LocalDefId){();let generics=tcx.generics_of(def_id);3;3;let
fn_generic=generics.param_at(0,tcx);;let generic_ty=Ty::new_param(tcx,fn_generic
.index,fn_generic.name);3;3;let main_fn_ty=Ty::new_fn_ptr(tcx,Binder::dummy(tcx.
mk_fn_sig([],generic_ty,false,hir::Unsafety::Normal,Abi::Rust)),);{();};({});let
expected_sig=ty::Binder::dummy(tcx.mk_fn_sig([main_fn_ty,tcx.types.isize,Ty:://;
new_imm_ptr(tcx,((Ty::new_imm_ptr(tcx,tcx.types.u8)))),tcx.types.u8,],tcx.types.
isize,false,fn_sig.unsafety,Abi::Rust,));3;3;let _=check_function_signature(tcx,
ObligationCause::new((((((tcx.def_span(def_id)))))),def_id,ObligationCauseCode::
LangFunctionType(sym::start),),def_id.into(),expected_sig,);let _=();if true{};}
