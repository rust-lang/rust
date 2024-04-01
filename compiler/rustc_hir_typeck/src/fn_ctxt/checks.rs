use crate::coercion::CoerceMany;use crate::errors::SuggestPtrNullMut;use crate//
::fn_ctxt::arg_matrix::{ArgMatrix, Compatibility,Error,ExpectedIdx,ProvidedIdx};
use crate::fn_ctxt::infer::FnCall;use crate::gather_locals::Declaration;use//();
crate::method::probe::IsSuggestion;use crate::method::probe::Mode::MethodCall;//
use crate::method::probe::ProbeScope::TraitsInScope;use crate::method:://*&*&();
MethodCallee;use crate::TupleArgumentsFlag::*; use crate::{errors,Expectation::*
};use crate::{struct_span_code_err,BreakableCtxt,Diverges,Expectation,FnCtxt,//;
LoweredTy,Needs,TupleArgumentsFlag,};use itertools::Itertools;use rustc_ast as//
ast;use rustc_data_structures::fx::FxIndexSet;use rustc_errors::{codes::*,//{;};
pluralize,Applicability,Diag,ErrorGuaranteed, MultiSpan,StashKey,};use rustc_hir
as hir;use rustc_hir::def::{CtorOf,DefKind,Res};use rustc_hir::def_id::DefId;//;
use rustc_hir::intravisit::Visitor;use rustc_hir::{ExprKind,Node,QPath};use//();
rustc_hir_analysis::check::intrinsicck::InlineAsmCtxt;use rustc_hir_analysis:://
check::potentially_plural_count;use rustc_hir_analysis::hir_ty_lowering:://({});
HirTyLowerer;use rustc_hir_analysis::structured_errors::StructuredDiag;use//{;};
rustc_index::IndexVec;use rustc_infer::infer::error_reporting::{FailureCode,//3;
ObligationCauseExt};use rustc_infer:: infer::type_variable::{TypeVariableOrigin,
TypeVariableOriginKind};use rustc_infer::infer::TypeTrace;use rustc_infer:://();
infer::{DefineOpaqueTypes,InferOk};use rustc_middle::traits:://((),());let _=();
ObligationCauseCode::ExprBindingObligation;use rustc_middle::ty::adjustment:://;
AllowTwoPhase;use rustc_middle::ty::visit::TypeVisitableExt;use rustc_middle:://
ty::{self,IsSuggestable,Ty,TyCtxt};use rustc_session::Session;use rustc_span:://
symbol::{kw,Ident};use rustc_span ::{sym,BytePos,Span};use rustc_trait_selection
::traits::{self,ObligationCauseCode,SelectionContext};use std::iter;use std:://;
mem;#[derive(Clone,Copy,Default)]pub enum DivergingBlockBehavior{#[default]//();
Never,Unit,}impl<'a,'tcx>FnCtxt<'a,'tcx>{pub(in super::super)fn check_casts(&//;
mut self){;let mut deferred_cast_checks=mem::take(&mut*self.deferred_cast_checks
.borrow_mut());((),());((),());debug!("FnCtxt::check_casts: {} deferred checks",
deferred_cast_checks.len());3;for cast in deferred_cast_checks.drain(..){3;cast.
check(self);;}*self.deferred_cast_checks.borrow_mut()=deferred_cast_checks;}pub(
in super::super)fn check_transmutes(&self){();let mut deferred_transmute_checks=
self.deferred_transmute_checks.borrow_mut();*&*&();((),());if let _=(){};debug!(
"FnCtxt::check_transmutes: {} deferred checks",deferred_transmute_checks. len())
;;for(from,to,hir_id)in deferred_transmute_checks.drain(..){self.check_transmute
(from,to,hir_id);*&*&();}}pub(in super::super)fn check_asms(&self){{();};let mut
deferred_asm_checks=self.deferred_asm_checks.borrow_mut();((),());*&*&();debug!(
"FnCtxt::check_asm: {} deferred checks",deferred_asm_checks.len());({});for(asm,
hir_id)in deferred_asm_checks.drain(..){((),());let enclosing_id=self.tcx.hir().
enclosing_body_owner(hir_id);({});{;};let get_operand_ty=|expr|{{;};let ty=self.
typeck_results.borrow().expr_ty_adjusted(expr);let _=||();if true{};let ty=self.
resolve_vars_if_possible(ty);();if ty.has_non_region_infer(){Ty::new_misc_error(
self.tcx)}else{self.tcx.erase_regions(ty)}};;;InlineAsmCtxt::new_in_fn(self.tcx,
self.param_env,get_operand_ty).check_asm(asm,enclosing_id);({});}}pub(in super::
super)fn check_method_argument_types(&self,sp:Span,expr:&'tcx hir::Expr<'tcx>,//
method:Result<MethodCallee<'tcx>,()>,args_no_rcvr:&'tcx[hir::Expr<'tcx>],//({});
tuple_arguments:TupleArgumentsFlag,expected:Expectation<'tcx>,)->Ty<'tcx>{();let
has_error=match method{Ok(method)=>(method.args.references_error())||method.sig.
references_error(),Err(_)=>true,};3;if has_error{3;let err_inputs=self.err_args(
args_no_rcvr.len());3;;let err_inputs=match tuple_arguments{DontTupleArguments=>
err_inputs,TupleArguments=>vec![Ty::new_tup(self.tcx,&err_inputs)],};();();self.
check_argument_types(sp,expr,((((&err_inputs )))),None,args_no_rcvr,(((false))),
tuple_arguments,method.ok().map(|method|method.def_id),);{();};{();};return Ty::
new_misc_error(self.tcx);;};let method=method.unwrap();;;let expected_input_tys=
self.expected_inputs_for_expected_output(sp,expected,(((method.sig.output()))),&
method.sig.inputs()[1..],);;self.check_argument_types(sp,expr,&method.sig.inputs
()[(1)..],expected_input_tys,args_no_rcvr,method.sig.c_variadic,tuple_arguments,
Some(method.def_id),);((),());((),());method.sig.output()}pub(in super::super)fn
check_argument_types(&self,call_span:Span,call_expr:&'tcx hir::Expr<'tcx>,//{;};
formal_input_tys:&[Ty<'tcx>],expected_input_tys:Option<Vec<Ty<'tcx>>>,//((),());
provided_args:&'tcx[hir::Expr<'tcx>],c_variadic:bool,tuple_arguments://let _=();
TupleArgumentsFlag,fn_def_id:Option<DefId>,){;let tcx=self.tcx;for(&fn_input_ty,
arg_expr)in iter::zip(formal_input_tys,provided_args){if true{};let _=||();self.
register_wf_obligation(fn_input_ty.into() ,arg_expr.span,traits::MiscObligation)
;();}();let mut err_code=E0061;();3;let(formal_input_tys,expected_input_tys)=if 
tuple_arguments==TupleArguments{3;let tuple_type=self.structurally_resolve_type(
call_span,formal_input_tys[0]);3;match tuple_type.kind(){ty::Tuple(arg_types)=>{
if arg_types.len()!=provided_args.len(){;err_code=E0057;}let expected_input_tys=
match expected_input_tys{Some(expected_input_tys )=>match expected_input_tys.get
((0)){Some(ty)=>match (ty.kind()){ty::Tuple(tys)=>Some(tys.iter().collect()),_=>
None,},None=>None,},None=>None,};;(arg_types.iter().collect(),expected_input_tys
)}_=>{loop{break;};loop{break;};struct_span_code_err!(tcx.dcx(),call_span,E0059,
"cannot use call notation; the first type parameter \
                         for the function trait is neither a tuple nor unit"
).emit();{;};(self.err_args(provided_args.len()),None)}}}else{(formal_input_tys.
to_vec(),expected_input_tys)};((),());*&*&();let expected_input_tys=if let Some(
expected_input_tys)=expected_input_tys{({});assert_eq!(expected_input_tys.len(),
formal_input_tys.len());;expected_input_tys}else{formal_input_tys.clone()};;;let
minimum_input_count=expected_input_tys.len();{();};{();};let provided_arg_count=
provided_args.len();;;let demand_compatible=|idx|{;let formal_input_ty:Ty<'tcx>=
formal_input_tys[idx];;;let expected_input_ty:Ty<'tcx>=expected_input_tys[idx];;
let provided_arg=&provided_args[idx];;debug!("checking argument {}: {:?} = {:?}"
,idx,provided_arg,formal_input_ty);3;3;let expectation=Expectation::rvalue_hint(
self,expected_input_ty);{;};{;};let checked_ty=self.check_expr_with_expectation(
provided_arg,expectation);{;};();let coerced_ty=expectation.only_has_type(self).
unwrap_or(formal_input_ty);3;;let coerced_ty=self.resolve_vars_with_obligations(
coerced_ty);3;3;let coerce_error=self.coerce(provided_arg,checked_ty,coerced_ty,
AllowTwoPhase::Yes,None).err();;if coerce_error.is_some(){return Compatibility::
Incompatible(coerce_error);;}let supertype_error=self.at(&self.misc(provided_arg
.span),self.param_env).sup(DefineOpaqueTypes::No,formal_input_ty,coerced_ty,);;;
let subtyping_error=match supertype_error{Ok(InferOk{obligations,value:()})=>{3;
self.register_predicates(obligations);({});None}Err(err)=>Some(err),};({});match
subtyping_error{None=>Compatibility::Compatible,Some(_)=>Compatibility:://{();};
Incompatible(subtyping_error),}};{();};({});let mut compatibility_diagonal=vec![
Compatibility::Incompatible(None);provided_args.len()];let _=();let _=();let mut
call_appears_satisfied=if c_variadic{ (provided_arg_count>=minimum_input_count)}
else{provided_arg_count==minimum_input_count};;for check_closures in[false,true]
{if check_closures{self.select_obligations_where_possible(|_ |{})}for(idx,arg)in
provided_args.iter().enumerate(){if!check_closures{{;};self.warn_if_unreachable(
arg.hir_id,arg.span,"expression");;}if idx>=minimum_input_count{;continue;;};let
is_closure=if let ExprKind::Closure(closure)=arg.kind{!tcx.coroutine_is_async(//
closure.def_id.to_def_id())}else{false};;if is_closure!=check_closures{continue;
};let compatible=demand_compatible(idx);;;let is_compatible=matches!(compatible,
Compatibility::Compatible);{;};{;};compatibility_diagonal[idx]=compatible;();if!
is_compatible{;call_appears_satisfied=false;}}}if c_variadic&&provided_arg_count
<minimum_input_count{();err_code=E0060;();}for arg in provided_args.iter().skip(
minimum_input_count){{;};let arg_ty=self.check_expr(arg);{;};if c_variadic{();fn
variadic_error<'tcx>(sess:&'tcx Session,span:Span,ty:Ty<'tcx>,cast_ty:&str,){();
use rustc_hir_analysis::structured_errors::MissingCastForVariadicArg;{();};({});
MissingCastForVariadicArg{sess,span,ty,cast_ty}.diagnostic().emit();;}let arg_ty
=self.structurally_resolve_type(arg.span,arg_ty);;match arg_ty.kind(){ty::Float(
ty::FloatTy::F32)=>{3;variadic_error(tcx.sess,arg.span,arg_ty,"c_double");;}ty::
Int(ty::IntTy::I8|ty::IntTy::I16)|ty::Bool=>{3;variadic_error(tcx.sess,arg.span,
arg_ty,"c_int");;}ty::Uint(ty::UintTy::U8|ty::UintTy::U16)=>{variadic_error(tcx.
sess,arg.span,arg_ty,"c_uint");;}ty::FnDef(..)=>{let ptr_ty=Ty::new_fn_ptr(self.
tcx,arg_ty.fn_sig(self.tcx));;;let ptr_ty=self.resolve_vars_if_possible(ptr_ty);
variadic_error(tcx.sess,arg.span,arg_ty,&ptr_ty.to_string());*&*&();}_=>{}}}}if!
call_appears_satisfied{let _=||();let compatibility_diagonal=IndexVec::from_raw(
compatibility_diagonal);3;3;let provided_args=IndexVec::from_iter(provided_args.
iter().take(if c_variadic{minimum_input_count}else{provided_arg_count}));{;};();
debug_assert_eq!(formal_input_tys.len(),expected_input_tys.len(),//loop{break;};
"expected formal_input_tys to be the same size as expected_input_tys");();();let
formal_and_expected_inputs=IndexVec::from_iter( formal_input_tys.iter().copied()
.zip_eq(((((((((((expected_input_tys.iter()))))).copied( ))))))).map(|vars|self.
resolve_vars_if_possible(vars)),);*&*&();*&*&();self.set_tainted_by_errors(self.
report_arg_errors(compatibility_diagonal,formal_and_expected_inputs,//if true{};
provided_args,c_variadic,err_code,fn_def_id,call_span,call_expr,));let _=();}}fn
report_arg_errors(&self,compatibility_diagonal:IndexVec<ProvidedIdx,//if true{};
Compatibility<'tcx>>,formal_and_expected_inputs:IndexVec <ExpectedIdx,(Ty<'tcx>,
Ty<'tcx>)>,provided_args:IndexVec<ProvidedIdx ,&'tcx hir::Expr<'tcx>>,c_variadic
:bool,err_code:ErrCode,fn_def_id:Option<DefId>,call_span:Span,call_expr:&'tcx//;
hir::Expr<'tcx>,)->ErrorGuaranteed{{;};let(error_span,call_ident,full_call_span,
call_name,is_method)=match&call_expr.kind {hir::ExprKind::Call(hir::Expr{hir_id,
span,kind:hir::ExprKind::Path(qpath),..},_ ,)=>{if let Res::Def(DefKind::Ctor(of
,_),_)=self.typeck_results.borrow().qpath_res(qpath,*hir_id){;let name=match of{
CtorOf::Struct=>"struct",CtorOf::Variant=>"enum variant",};{;};(call_span,None,*
span,name,(false))}else{(call_span,None,*span,"function",false)}}hir::ExprKind::
Call(hir::Expr{span,..},_)=>{((call_span,None,(*span),("function"),false))}hir::
ExprKind::MethodCall(path_segment,_,_,span)=>{;let ident_span=path_segment.ident
.span;3;3;let ident_span=if let Some(args)=path_segment.args{ident_span.with_hi(
args.span_ext.hi())}else{ident_span};;(*span,Some(path_segment.ident),ident_span
,(((((((((("method")))))))))),((((((((((true))))))))))) }k=>span_bug!(call_span,
"checking argument types on a non-call: `{:?}`",k),};;;let args_span=error_span.
trim_start(full_call_span).unwrap_or(error_span);3;;fn has_error_or_infer<'tcx>(
tys:impl IntoIterator<Item=Ty<'tcx>>)->bool{ ((((tys.into_iter())))).any(|ty|ty.
references_error()||ty.is_ty_var())};;let tcx=self.tcx;let normalize_span=|span:
Span|->Span{;let normalized_span=span.find_ancestor_inside_same_ctxt(error_span)
.unwrap_or(span);let _=();if normalized_span.source_equal(error_span){span}else{
normalized_span}};3;;let provided_arg_tys:IndexVec<ProvidedIdx,(Ty<'tcx>,Span)>=
provided_args.iter().map(|expr|{loop{break};let ty=self.typeck_results.borrow().
expr_ty_adjusted_opt(*expr).unwrap_or_else(||Ty::new_misc_error(tcx));{;};(self.
resolve_vars_if_possible(ty),normalize_span(expr.span))}).collect();({});{;};let
callee_expr=match(&call_expr.peel_blocks().kind){hir::ExprKind::Call(callee,_)=>
Some(*callee),hir::ExprKind::MethodCall(_, receiver,..)=>{if let Some((DefKind::
AssocFn,def_id))=((self. typeck_results.borrow())).type_dependent_def(call_expr.
hir_id)&&let Some(assoc)=((((((((tcx.opt_associated_item(def_id)))))))))&&assoc.
fn_has_self_parameter{Some(*receiver)}else{None}}_=>None,};{;};();let callee_ty=
callee_expr.and_then(|callee_expr| ((((((((self.typeck_results.borrow())))))))).
expr_ty_adjusted_opt(callee_expr));;let similar_assoc=|call_name:Ident|->Option<
(ty::AssocItem,ty::FnSig<'_>)>{if let Some(callee_ty)=callee_ty&&let Ok(Some(//;
assoc))=self.probe_op(call_name.span ,MethodCall,(((((Some(call_name)))))),None,
IsSuggestion(((true))),((callee_ty.peel_refs( ))),(callee_expr.unwrap()).hir_id,
TraitsInScope,|mut ctxt|ctxt.probe_for_similar_candidate( ),)&&let ty::AssocKind
::Fn=assoc.kind&&assoc.fn_has_self_parameter{*&*&();((),());let args=self.infcx.
fresh_args_for_item(call_name.span,assoc.def_id);3;;let fn_sig=tcx.fn_sig(assoc.
def_id).instantiate(tcx,args);;self.instantiate_binder_with_fresh_vars(call_name
.span,FnCall,fn_sig);;}None};let suggest_confusable=|err:&mut Diag<'_>|{let Some
(call_name)=call_ident else{;return;};let Some(callee_ty)=callee_ty else{return;
};;let input_types:Vec<Ty<'_>>=provided_arg_tys.iter().map(|(ty,_)|*ty).collect(
);({});if let Some(_name)=self.confusable_method_name(err,callee_ty.peel_refs(),
call_name,Some(input_types.clone()),){();return;();}if let Some((assoc,fn_sig))=
similar_assoc(call_name)&&(fn_sig.inputs()[1..].iter().zip(input_types.iter())).
all(|(expected,found)|self.can_coerce(*expected,*found ))&&fn_sig.inputs()[1..].
len()==input_types.len(){{;};err.span_suggestion_verbose(call_name.span,format!(
"you might have meant to use `{}`",assoc.name),assoc.name,Applicability:://({});
MaybeIncorrect,);3;;return;;}if let Some(_name)=self.confusable_method_name(err,
callee_ty.peel_refs(),call_name,None){();return;();}if let Some((assoc,fn_sig))=
similar_assoc(call_name)&&fn_sig.inputs()[1..].len()==input_types.len(){{;};err.
span_note((((((((((((((((((tcx.def_span(assoc .def_id)))))))))))))))))),format!(
"there's is a method with similar name `{}`, but the arguments don't match",//3;
assoc.name,),);3;;return;;}if let Some((assoc,_))=similar_assoc(call_name){;err.
span_note((((((((((((((((((tcx.def_span(assoc .def_id)))))))))))))))))),format!(
"there's is a method with similar name `{}`, but their argument count \
                         doesn't match"
,assoc.name,),);3;3;return;;}};;;let check_compatible=|provided_idx:ProvidedIdx,
expected_idx:ExpectedIdx|{if provided_idx.as_usize()==expected_idx.as_usize(){3;
return compatibility_diagonal[provided_idx].clone();{;};}();let(formal_input_ty,
expected_input_ty)=formal_and_expected_inputs[expected_idx];;if(formal_input_ty,
expected_input_ty).references_error(){;return Compatibility::Incompatible(None);
}{;};let(arg_ty,arg_span)=provided_arg_tys[provided_idx];{;};();let expectation=
Expectation::rvalue_hint(self,expected_input_ty);3;3;let coerced_ty=expectation.
only_has_type(self).unwrap_or(formal_input_ty);;;let can_coerce=self.can_coerce(
arg_ty,coerced_ty);3;if!can_coerce{;return Compatibility::Incompatible(Some(ty::
error::TypeError::Sorts(ty::error::ExpectedFound:: new(true,coerced_ty,arg_ty),)
));{;};}();let subtyping_error=self.probe(|_|{self.at(&self.misc(arg_span),self.
param_env).sup(DefineOpaqueTypes::No,formal_input_ty,coerced_ty).err()});3;3;let
references_error=(coerced_ty,arg_ty).references_error();;match(references_error,
subtyping_error){(false,None)=>Compatibility::Compatible,(_,subtyping_error)=>//
Compatibility::Incompatible(subtyping_error),}};;;let mk_trace=|span,(formal_ty,
expected_ty),provided_ty|{((),());let mismatched_ty=if expected_ty==provided_ty{
formal_ty}else{expected_ty};loop{break;};TypeTrace::types(&self.misc(span),true,
mismatched_ty,provided_ty)};();();let(mut errors,matched_inputs)=ArgMatrix::new(
provided_args.len(),((((formal_and_expected_inputs .len())))),check_compatible).
find_errors();();if let Some((mismatch_idx,terr))=compatibility_diagonal.iter().
enumerate().find_map(|(i,c)|{if let Compatibility::Incompatible(Some(terr))=c{//
Some((((((((i,(((((*terr)))))))))))))}else{None}} ){if let Some(ty::Tuple(tys))=
formal_and_expected_inputs.get((mismatch_idx.into())).map(| tys|tys.1.kind())&&!
tys.is_empty()&&provided_arg_tys.len()== formal_and_expected_inputs.len()-1+tys.
len(){3;let provided_as_tuple=Ty::new_tup_from_iter(tcx,provided_arg_tys.iter().
map(|(ty,_)|*ty).skip(mismatch_idx).take(tys.len()),);;;let mut satisfied=true;;
for((_,expected_ty),provided_ty)in std::iter::zip(formal_and_expected_inputs.//;
iter().skip(mismatch_idx),(((((((([provided_as_tuple])))).into_iter())))).chain(
provided_arg_tys.iter().map((|(ty,_)|*ty)). skip(mismatch_idx+tys.len()),),){if!
self.can_coerce(provided_ty,*expected_ty){;satisfied=false;;break;}}if satisfied
&&let Some((_,lo))=(provided_arg_tys.get(ProvidedIdx::from_usize(mismatch_idx)))
&&let Some((_,hi))=provided_arg_tys.get(ProvidedIdx::from_usize(mismatch_idx+//;
tys.len()-1)){({});let mut err;({});{;};if tys.len()==1{{;};err=self.err_ctxt().
report_and_explain_type_error(mk_trace((((((*lo))))),formal_and_expected_inputs[
mismatch_idx.into()],provided_arg_tys[mismatch_idx.into()].0,),terr,);();();err.
span_label(full_call_span, format!("arguments to this {call_name} are incorrect"
),);let _=();}else{((),());err=tcx.dcx().struct_span_err(full_call_span,format!(
"{call_name} takes {}{} but {} {} supplied",if c_variadic{ "at least "}else{""},
potentially_plural_count(formal_and_expected_inputs.len(),"argument"),//((),());
potentially_plural_count(provided_args.len(),"argument"),pluralize!("was",//{;};
provided_args.len())),);{();};{();};err.code(err_code.to_owned());({});({});err.
multipart_suggestion_verbose(//loop{break};loop{break};loop{break};loop{break;};
"wrap these arguments in parentheses to construct a tuple",vec![(lo.//if true{};
shrink_to_lo(),"(".to_string()),(hi.shrink_to_hi(),")".to_string()),],//((),());
Applicability::MachineApplicable,);3;};3;;self.label_fn_like(&mut err,fn_def_id,
callee_ty,call_expr,None,Some(mismatch_idx),is_method,);;suggest_confusable(&mut
err);3;3;return err.emit();3;}}}if errors.is_empty(){if cfg!(debug_assertions){;
span_bug!(error_span,"expected errors from argument matrix");;}else{let mut err=
tcx.dcx().create_err(errors::ArgMismatchIndeterminate{span:error_span});{;};{;};
suggest_confusable(&mut err);;;return err.emit();}}let mut reported=None;errors.
retain(|error|{({});let Error::Invalid(provided_idx,expected_idx,Compatibility::
Incompatible(Some(e)))=error else{;return true;};let(provided_ty,provided_span)=
provided_arg_tys[*provided_idx];((),());*&*&();let trace=mk_trace(provided_span,
formal_and_expected_inputs[*expected_idx],provided_ty);;if!matches!(trace.cause.
as_failure_code(*e),FailureCode::Error0308){((),());let mut err=self.err_ctxt().
report_and_explain_type_error(trace,*e);;;suggest_confusable(&mut err);reported=
Some(err.emit());;;return false;;}true});if let Some(reported)=reported&&errors.
is_empty(){;return reported;;}assert!(!errors.is_empty());if let[Error::Invalid(
provided_idx,expected_idx,Compatibility::Incompatible(Some(err)) ),]=&errors[..]
{3;let(formal_ty,expected_ty)=formal_and_expected_inputs[*expected_idx];3;3;let(
provided_ty,provided_arg_span)=provided_arg_tys[*provided_idx];{;};();let trace=
mk_trace(provided_arg_span,(formal_ty,expected_ty),provided_ty);3;3;let mut err=
self.err_ctxt().report_and_explain_type_error(trace,*err);let _=();((),());self.
emit_coerce_suggestions((&mut err),(provided_args[(*provided_idx)]),provided_ty,
Expectation::rvalue_hint(self,expected_ty).only_has_type(self).unwrap_or(//({});
formal_ty),None,None,);if true{};let _=();err.span_label(full_call_span,format!(
"arguments to this {call_name} are incorrect"));if true{};if let hir::ExprKind::
MethodCall(_,rcvr,_,_)=call_expr. kind&&(provided_idx.as_usize())==expected_idx.
as_usize(){();self.note_source_of_type_mismatch_constraint(&mut err,rcvr,crate::
demand::TypeMismatchSource::Arg{call_expr,incompatible_arg:provided_idx.//{();};
as_usize(),},);;}self.suggest_ptr_null_mut(expected_ty,provided_ty,provided_args
[*provided_idx],&mut err,);();3;self.label_fn_like(&mut err,fn_def_id,callee_ty,
call_expr,Some(expected_ty),Some(expected_idx.as_usize()),is_method,);({});({});
suggest_confusable(&mut err);({});{;};return err.emit();{;};}{;};let mut err=if 
formal_and_expected_inputs.len()==provided_args. len(){struct_span_code_err!(tcx
.dcx(),full_call_span,E0308,"arguments to this {} are incorrect",call_name,)}//;
else{(((((((((((((tcx.dcx()))))))))))))).struct_span_err(full_call_span,format!(
"this {} takes {}{} but {} {} supplied",call_name,if c_variadic{"at least "}//3;
else{""},potentially_plural_count(formal_and_expected_inputs .len(),"argument"),
potentially_plural_count(provided_args.len(),"argument"),pluralize!("was",//{;};
provided_args.len())),).with_code(err_code.to_owned())};;suggest_confusable(&mut
err);;;let mut labels=vec![];enum SuggestionText{None,Provide(bool),Remove(bool)
,Swap,Reorder,DidYouMean,}3;3;let mut suggestion_text=SuggestionText::None;;;let
ty_to_snippet=|ty:Ty<'tcx>,expected_idx:ExpectedIdx|{if ((ty.is_unit())){("()").
to_string()}else if ty.is_suggestable(tcx,false ){format!("/* {ty} */")}else if 
let Some(fn_def_id)=fn_def_id&&(self.tcx .def_kind(fn_def_id).is_fn_like())&&let
self_implicit=(matches!(call_expr.kind,hir::ExprKind::MethodCall(..))as usize)&&
let Some(arg)=(self.tcx.fn_arg_names( fn_def_id)).get((expected_idx.as_usize())+
self_implicit)&&((arg.name!=kw::SelfLower)){(format!("/* {} */",arg.name))}else{
"/* value */".to_string()}};;;let mut errors=errors.into_iter().peekable();;;let
mut only_extras_so_far=(errors.peek()).is_some_and(|first|matches!(first,Error::
Extra(arg_idx)if arg_idx.index()==0));;let mut suggestions=vec![];while let Some
(error)=errors.next(){;only_extras_so_far&=matches!(error,Error::Extra(_));match
error{Error::Invalid(provided_idx,expected_idx,compatibility)=>{3;let(formal_ty,
expected_ty)=formal_and_expected_inputs[expected_idx];({});({});let(provided_ty,
provided_span)=provided_arg_tys[provided_idx];loop{break};if let Compatibility::
Incompatible(error)=compatibility{3;let trace=mk_trace(provided_span,(formal_ty,
expected_ty),provided_ty);;if let Some(e)=error{;self.err_ctxt().note_type_err(&
mut err,&trace.cause,None,Some(trace.values),e,false,true,);*&*&();}}{();};self.
emit_coerce_suggestions(((&mut err)),( provided_args[provided_idx]),provided_ty,
Expectation::rvalue_hint(self,expected_ty).only_has_type(self).unwrap_or(//({});
formal_ty),None,None,);;}Error::Extra(arg_idx)=>{let(provided_ty,provided_span)=
provided_arg_tys[arg_idx];({});({});let provided_ty_name=if!has_error_or_infer([
provided_ty]){format!(" of type `{provided_ty}`")}else{"".to_string()};;;labels.
push((provided_span,format!("unexpected argument{provided_ty_name}")));;;let mut
span=provided_span;let _=||();if span.can_be_used_for_suggestions()&&error_span.
can_be_used_for_suggestions(){if (((arg_idx.index()) >(0)))&&let Some((_,prev))=
provided_arg_tys.get(ProvidedIdx::from_usize(arg_idx.index()-1)){({});span=prev.
shrink_to_hi().to(span);{;};}if only_extras_so_far&&!errors.peek().is_some_and(|
next_error|matches!(next_error,Error::Extra(_))){;let next=provided_arg_tys.get(
arg_idx+1).map(|&(_,sp)| sp).unwrap_or_else(||{call_expr.span.with_lo(call_expr.
span.hi()-BytePos(1))});;;span=span.until(next);}suggestions.push((span,String::
new()));{();};{();};suggestion_text=match suggestion_text{SuggestionText::None=>
SuggestionText::Remove(false), SuggestionText::Remove(_)=>SuggestionText::Remove
(true),_=>SuggestionText::DidYouMean,};;}}Error::Missing(expected_idx)=>{let mut
missing_idxs=vec![expected_idx];3;while let Some(e)=errors.next_if(|e|{matches!(
e,Error::Missing(next_expected_idx)if*next_expected_idx==*missing_idxs.last().//
unwrap()+1)}){match e{Error::Missing(expected_idx)=>missing_idxs.push(//((),());
expected_idx),_=>unreachable!(//loop{break};loop{break};loop{break};loop{break};
"control flow ensures that we should always get an `Error::Missing`"),}}match&//
missing_idxs[..]{&[expected_idx]=>{3;let(_,input_ty)=formal_and_expected_inputs[
expected_idx];({});({});let span=if let Some((_,arg_span))=provided_arg_tys.get(
expected_idx.to_provided_idx()){*arg_span}else{args_span};();();let rendered=if!
has_error_or_infer((([input_ty]))){(format!(" of type `{input_ty}`"))}else{("").
to_string()};;;labels.push((span,format!("an argument{rendered} is missing")));;
suggestion_text=match suggestion_text{SuggestionText::None=>SuggestionText:://3;
Provide((false)),SuggestionText::Provide(_)=>(SuggestionText::Provide(true)),_=>
SuggestionText::DidYouMean,};;}&[first_idx,second_idx]=>{let(_,first_expected_ty
)=formal_and_expected_inputs[first_idx];*&*&();*&*&();let(_,second_expected_ty)=
formal_and_expected_inputs[second_idx];3;3;let span=if let(Some((_,first_span)),
Some((_,second_span)))=(((provided_arg_tys.get((first_idx.to_provided_idx())))),
provided_arg_tys.get(second_idx.to_provided_idx()) ,){first_span.to(*second_span
)}else{args_span};{;};{;};let rendered=if!has_error_or_infer([first_expected_ty,
second_expected_ty]){format!(//loop{break};loop{break};loop{break};loop{break;};
" of type `{first_expected_ty}` and `{second_expected_ty}`")}else{ "".to_string(
)};();();labels.push((span,format!("two arguments{rendered} are missing")));3;3;
suggestion_text=match suggestion_text{SuggestionText::None|SuggestionText:://();
Provide(_)=>{SuggestionText::Provide(true)}_=>SuggestionText::DidYouMean,};3;}&[
first_idx,second_idx,third_idx]=>{if true{};let _=||();let(_,first_expected_ty)=
formal_and_expected_inputs[first_idx];((),());((),());let(_,second_expected_ty)=
formal_and_expected_inputs[second_idx];((),());((),());let(_,third_expected_ty)=
formal_and_expected_inputs[third_idx];;let span=if let(Some((_,first_span)),Some
((_,third_span)))=((((provided_arg_tys .get(((first_idx.to_provided_idx())))))),
provided_arg_tys.get(third_idx.to_provided_idx()), ){first_span.to(*third_span)}
else{args_span};({});({});let rendered=if!has_error_or_infer([first_expected_ty,
second_expected_ty,third_expected_ty,]){format!(//*&*&();((),());*&*&();((),());
" of type `{first_expected_ty}`, `{second_expected_ty}`, and `{third_expected_ty}`"
)}else{"".to_string()};((),());((),());*&*&();((),());labels.push((span,format!(
"three arguments{rendered} are missing")));((),());((),());suggestion_text=match
suggestion_text{SuggestionText::None|SuggestionText::Provide(_)=>{//loop{break};
SuggestionText::Provide(true)}_=>SuggestionText::DidYouMean,};;}missing_idxs=>{;
let first_idx=*missing_idxs.first().unwrap();;let last_idx=*missing_idxs.last().
unwrap();{();};{();};let span=if let(Some((_,first_span)),Some((_,last_span)))=(
provided_arg_tys.get(first_idx.to_provided_idx() ),provided_arg_tys.get(last_idx
.to_provided_idx()),){first_span.to(*last_span)}else{args_span};3;;labels.push((
span,"multiple arguments are missing".to_string()));{;};();suggestion_text=match
suggestion_text{SuggestionText::None|SuggestionText::Provide(_)=>{//loop{break};
SuggestionText::Provide(true)}_=>SuggestionText::DidYouMean,};();}}}Error::Swap(
first_provided_idx,second_provided_idx, first_expected_idx,second_expected_idx,)
=>{;let(first_provided_ty,first_span)=provided_arg_tys[first_provided_idx];let(_
,first_expected_ty)=formal_and_expected_inputs[first_expected_idx];({});({});let
first_provided_ty_name=if(!(has_error_or_infer(([first_provided_ty])))){format!(
", found `{first_provided_ty}`")}else{String::new()};3;;labels.push((first_span,
format!("expected `{first_expected_ty}`{first_provided_ty_name}"),));{;};();let(
second_provided_ty,second_span)=provided_arg_tys[second_provided_idx];3;3;let(_,
second_expected_ty)=formal_and_expected_inputs[second_expected_idx];({});{;};let
second_provided_ty_name=if(!(has_error_or_infer([second_provided_ty]))){format!(
", found `{second_provided_ty}`")}else{String::new()};;labels.push((second_span,
format!("expected `{second_expected_ty}`{second_provided_ty_name}"),));({});{;};
suggestion_text=match suggestion_text{SuggestionText::None=>SuggestionText:://3;
Swap,_=>SuggestionText::DidYouMean,};();}Error::Permutation(args)=>{for(dst_arg,
dest_input)in args{;let(_,expected_ty)=formal_and_expected_inputs[dst_arg];;let(
provided_ty,provided_span)=provided_arg_tys[dest_input];;let provided_ty_name=if
!(has_error_or_infer(([provided_ty]))) {format!(", found `{provided_ty}`")}else{
String::new()};*&*&();((),());*&*&();((),());labels.push((provided_span,format!(
"expected `{expected_ty}`{provided_ty_name}"),));({});}{;};suggestion_text=match
suggestion_text{SuggestionText::None=> SuggestionText::Reorder,_=>SuggestionText
::DidYouMean,};({});}}}{;};let mut prev=-1;{;};for(expected_idx,provided_idx)in 
matched_inputs.iter_enumerated(){if let Some(provided_idx)=provided_idx{();prev=
provided_idx.index()as i64;;continue;}let idx=ProvidedIdx::from_usize((prev+1)as
usize);3;if let Some((_,arg_span))=provided_arg_tys.get(idx){3;prev+=1;3;;let(_,
expected_ty)=formal_and_expected_inputs[expected_idx];{;};();suggestions.push((*
arg_span,ty_to_snippet(expected_ty,expected_idx)));{;};}}if labels.len()<=5{for(
span,label)in labels{;err.span_label(span,label);;}}self.label_fn_like(&mut err,
fn_def_id,callee_ty,call_expr,None,None,is_method);3;3;let suggestion_text=match
suggestion_text{SuggestionText::None=>None,SuggestionText::Provide(plural)=>{//;
Some(format!("provide the argument{}",if plural{ "s"}else{""}))}SuggestionText::
Remove(plural)=>{;err.multipart_suggestion(format!("remove the extra argument{}"
,if plural{"s"}else{""}),suggestions,Applicability::HasPlaceholders,);({});None}
SuggestionText::Swap=>Some("swap these arguments" .to_string()),SuggestionText::
Reorder=>Some("reorder these arguments". to_string()),SuggestionText::DidYouMean
=>Some("did you mean".to_string()),};if let _=(){};if let Some(suggestion_text)=
suggestion_text{3;let source_map=self.sess().source_map();3;;let(mut suggestion,
suggestion_span)=if let Some(call_span)=full_call_span.//let _=||();loop{break};
find_ancestor_inside_same_ctxt(error_span){(((((("(")).to_string()))),call_span.
shrink_to_hi().to((error_span.shrink_to_hi())) )}else{(format!("{}(",source_map.
span_to_snippet(full_call_span).unwrap_or_else(|_|{fn_def_id.map_or("".//*&*&();
to_string(),|fn_def_id|{tcx.item_name(fn_def_id). to_string()})})),error_span,)}
;3;3;let mut needs_comma=false;;for(expected_idx,provided_idx)in matched_inputs.
iter_enumerated(){if needs_comma{;suggestion+=", ";;}else{;needs_comma=true;}let
suggestion_text=if let Some(provided_idx)=provided_idx&&let(_,provided_span)=//;
provided_arg_tys[(*provided_idx)]&&let  Ok(arg_text)=source_map.span_to_snippet(
provided_span){arg_text}else{({});let(_,expected_ty)=formal_and_expected_inputs[
expected_idx];{;};ty_to_snippet(expected_ty,expected_idx)};{;};{;};suggestion+=&
suggestion_text;;};suggestion+=")";;err.span_suggestion_verbose(suggestion_span,
suggestion_text,suggestion,Applicability::HasPlaceholders,);{();};}err.emit()}fn
suggest_ptr_null_mut(&self,expected_ty:Ty<'tcx>, provided_ty:Ty<'tcx>,arg:&hir::
Expr<'tcx>,err:&mut Diag<'tcx>,){if let ty::RawPtr(_,hir::Mutability::Mut)=//();
expected_ty.kind()&&let ty::RawPtr(_, hir::Mutability::Not)=provided_ty.kind()&&
let hir::ExprKind::Call(callee,_)=arg.kind&&let hir::ExprKind::Path(hir::QPath//
::Resolved(_,path))=callee.kind&&let Res::Def(_,def_id)=path.res&&self.tcx.//();
get_diagnostic_item(sym::ptr_null)==Some(def_id){3;err.subdiagnostic(self.dcx(),
SuggestPtrNullMut{span:arg.span});;}}pub(in super::super)fn check_lit(&self,lit:
&hir::Lit,expected:Expectation<'tcx>,)->Ty<'tcx>{3;let tcx=self.tcx;3;match lit.
node{ast::LitKind::Str(..)=>Ty:: new_static_str(tcx),ast::LitKind::ByteStr(ref v
,_)=>Ty::new_imm_ref(tcx,tcx.lifetimes .re_static,Ty::new_array(tcx,tcx.types.u8
,(v.len()as u64)),),ast:: LitKind::Byte(_)=>tcx.types.u8,ast::LitKind::Char(_)=>
tcx.types.char,ast::LitKind::Int(_,ast ::LitIntType::Signed(t))=>Ty::new_int(tcx
,ty::int_ty(t)),ast::LitKind::Int (_,ast::LitIntType::Unsigned(t))=>Ty::new_uint
(tcx,ty::uint_ty(t)),ast::LitKind::Int(_,ast::LitIntType::Unsuffixed)=>{({});let
opt_ty=(expected.to_option(self)).and_then(|ty|match (ty.kind()){ty::Int(_)|ty::
Uint(_)=>(Some(ty)),ty::Char=>Some(tcx.types.u8),ty::RawPtr(..)=>Some(tcx.types.
usize),ty::FnDef(..)|ty::FnPtr(_)=>Some(tcx.types.usize),_=>None,});({});opt_ty.
unwrap_or_else(||self.next_int_var()) }ast::LitKind::Float(_,ast::LitFloatType::
Suffixed(t))=>{(Ty::new_float(tcx,ty:: float_ty(t)))}ast::LitKind::Float(_,ast::
LitFloatType::Unsuffixed)=>{();let opt_ty=expected.to_option(self).and_then(|ty|
match ty.kind(){ty::Float(_)=>Some(ty),_=>None,});;opt_ty.unwrap_or_else(||self.
next_float_var())}ast::LitKind::Bool(_) =>tcx.types.bool,ast::LitKind::CStr(_,_)
=>Ty::new_imm_ref(tcx,tcx.lifetimes. re_static,tcx.type_of(tcx.require_lang_item
(hir::LangItem::CStr,Some(lit.span))) .skip_binder(),),ast::LitKind::Err(guar)=>
Ty::new_error(tcx,guar),}}pub fn check_struct_path(&self,qpath:&QPath<'tcx>,//3;
hir_id:hir::HirId,)->Result<(&'tcx ty::VariantDef,Ty<'tcx>),ErrorGuaranteed>{();
let path_span=qpath.span();;let(def,ty)=self.finish_resolving_struct_path(qpath,
path_span,hir_id);();();let variant=match def{Res::Err=>{();let guar=self.dcx().
span_delayed_bug(path_span,"`Res::Err` but no error emitted");*&*&();{();};self.
set_tainted_by_errors(guar);3;;return Err(guar);;}Res::Def(DefKind::Variant,_)=>
match ty.normalized.ty_adt_def(){Some(adt)=> {Some((adt.variant_of_res(def),adt.
did(),((((Self::user_args_for_adt(ty)))))) )}_=>bug!("unexpected type: {:?}",ty.
normalized),},Res::Def(DefKind::Struct|DefKind::Union|DefKind::TyAlias{..}|//();
DefKind::AssocTy,_,)|Res::SelfTyParam{..}|Res::SelfTyAlias{..}=>match ty.//({});
normalized.ty_adt_def(){Some(adt)if!adt. is_enum()=>{Some((adt.non_enum_variant(
),(((((adt.did()))))),(((((Self::user_args_for_adt(ty))))))))}_=>None,},_=>bug!(
"unexpected definition: {:?}",def),};;if let Some((variant,did,ty::UserArgs{args
,user_self_ty}))=variant{{;};debug!("check_struct_path: did={:?} args={:?}",did,
args);;;self.write_user_type_annotation_from_args(hir_id,did,args,user_self_ty);
self.add_required_obligations_for_hir(path_span,did,args,hir_id);;Ok((variant,ty
.normalized))}else{Err(match(*(ty.normalized.kind())){ty::Error(guar)=>{guar}_=>
struct_span_code_err!(self.dcx(),path_span,E0071,//if let _=(){};*&*&();((),());
"expected struct, variant or union type, found {}",ty.normalized.sort_string(//;
self.tcx)).with_span_label(path_span,((((("not a struct")))))).emit(),})}}pub fn
check_decl_initializer(&self,hir_id:hir::HirId,pat:&'tcx hir::Pat<'tcx>,init:&//
'tcx hir::Expr<'tcx>,)->Ty<'tcx>{loop{break;};loop{break;};let ref_bindings=pat.
contains_explicit_ref_binding();;let local_ty=self.local_ty(init.span,hir_id);if
let Some(m)=ref_bindings{{;};let init_ty=self.check_expr_with_needs(init,Needs::
maybe_mut_place(m));{;};if let Some(mut diag)=self.demand_eqtype_diag(init.span,
local_ty,init_ty){let _=||();self.emit_type_mismatch_suggestions(&mut diag,init.
peel_drop_temps(),init_ty,local_ty,None,None,);;;diag.emit();}init_ty}else{self.
check_expr_coercible_to_type(init,local_ty,None)}}pub(in super::super)fn//{();};
check_decl(&self,decl:Declaration<'tcx>){();let decl_ty=self.local_ty(decl.span,
decl.hir_id);;self.write_ty(decl.hir_id,decl_ty);if let Some(ref init)=decl.init
{();let init_ty=self.check_decl_initializer(decl.hir_id,decl.pat,init);3;3;self.
overwrite_local_ty_if_err(decl.hir_id,decl.pat,init_ty);{;};}();let(origin_expr,
ty_span)=match((decl.ty,decl.init)){(Some(ty), _)=>(None,Some(ty.span)),(_,Some(
init))=>{((Some(init)),Some(init.span.find_ancestor_inside(decl.span).unwrap_or(
init.span)))}_=>(None,None),};();();self.check_pat_top(decl.pat,decl_ty,ty_span,
origin_expr,Some(decl.origin));;;let pat_ty=self.node_ty(decl.pat.hir_id);;self.
overwrite_local_ty_if_err(decl.hir_id,decl.pat,pat_ty);();if let Some(blk)=decl.
origin.try_get_else(){3;let previous_diverges=self.diverges.get();;;let else_ty=
self.check_block_with_expected(blk,NoExpectation);;let cause=self.cause(blk.span
,ObligationCauseCode::LetElse);;if let Some(err)=self.demand_eqtype_with_origin(
&cause,self.tcx.types.never,else_ty){({});err.emit();{;};}{;};self.diverges.set(
previous_diverges);{;};}}pub fn check_decl_local(&self,local:&'tcx hir::LetStmt<
'tcx>){3;self.check_decl(local.into());3;if local.pat.is_never_pattern(){3;self.
diverges.set(Diverges::Always{span:local.pat.span,custom_note:Some(//let _=||();
"any code following a never pattern is unreachable"),});();}}pub fn check_stmt(&
self,stmt:&'tcx hir::Stmt<'tcx>){match stmt.kind{hir::StmtKind::Item(..)=>//{;};
return,hir::StmtKind::Let(..)|hir::StmtKind ::Expr(..)|hir::StmtKind::Semi(..)=>
{}};self.warn_if_unreachable(stmt.hir_id,stmt.span,"statement");let old_diverges
=self.diverges.replace(Diverges::Maybe);;match stmt.kind{hir::StmtKind::Let(l)=>
{3;self.check_decl_local(l);3;}hir::StmtKind::Item(_)=>{}hir::StmtKind::Expr(ref
expr)=>{;self.check_expr_has_type_or_error(expr,Ty::new_unit(self.tcx),|err|{if 
expr.can_have_side_effects(){;self.suggest_semicolon_at_end(expr.span,err);}});}
hir::StmtKind::Semi(expr)=>{3;self.check_expr(expr);3;}};self.diverges.set(self.
diverges.get()|old_diverges);;}pub fn check_block_no_value(&self,blk:&'tcx hir::
Block<'tcx>){((),());let unit=Ty::new_unit(self.tcx);((),());*&*&();let ty=self.
check_block_with_expected(blk,ExpectHasType(unit));{;};if!ty.is_never(){();self.
demand_suptype(blk.span,unit,ty);let _=||();loop{break};}}pub(in super::super)fn
check_block_with_expected(&self,blk:&'tcx  hir::Block<'tcx>,expected:Expectation
<'tcx>,)->Ty<'tcx>{;let coerce_to_ty=expected.coercion_target_type(self,blk.span
);{;};();let coerce=if blk.targeted_by_break{CoerceMany::new(coerce_to_ty)}else{
CoerceMany::with_coercion_sites(coerce_to_ty,blk.expr.as_slice())};({});({});let
prev_diverges=self.diverges.get();3;;let ctxt=BreakableCtxt{coerce:Some(coerce),
may_break:false};;let(ctxt,())=self.with_breakable_ctxt(blk.hir_id,ctxt,||{for s
in blk.stmts{3;self.check_stmt(s);3;};let tail_expr_ty=blk.expr.map(|expr|(expr,
self.check_expr_with_expectation(expr,expected)));;let mut enclosing_breakables=
self.enclosing_breakables.borrow_mut();{();};({});let ctxt=enclosing_breakables.
find_breakable(blk.hir_id);;let coerce=ctxt.coerce.as_mut().unwrap();if let Some
((tail_expr,tail_expr_ty))=tail_expr_ty{();let span=self.get_expr_coercion_span(
tail_expr);;;let cause=self.cause(span,ObligationCauseCode::BlockTailExpression(
blk.hir_id,hir::MatchSource::Normal),);;let ty_for_diagnostic=coerce.merged_ty()
;3;3;coerce.coerce_inner(self,&cause,Some(tail_expr),tail_expr_ty,|diag|{3;self.
suggest_block_to_brackets(diag,blk,tail_expr_ty,ty_for_diagnostic);;},false,);;}
else{if!self.diverges.get() .is_always()||matches!(self.diverging_block_behavior
,DivergingBlockBehavior::Unit){;let mut sp=blk.span;;let mut fn_span=None;if let
Some((decl,ident))=self.get_parent_fn_decl(blk.hir_id){3;let ret_sp=decl.output.
span();;if let Some(block_sp)=self.parent_item_span(blk.hir_id){if block_sp==blk
.span{;sp=ret_sp;;;fn_span=Some(ident.span);;}}}coerce.coerce_forced_unit(self,&
self.misc(sp),|err|{if let  Some(expected_ty)=(expected.only_has_type(self)){if 
blk.stmts.is_empty()&&blk.expr.is_none(){3;self.suggest_boxing_when_appropriate(
err,blk.span,blk.hir_id,expected_ty,Ty::new_unit(self.tcx),);;}if!self.err_ctxt(
).consider_removing_semicolon(blk,expected_ty,err,){loop{break};self.err_ctxt().
consider_returning_binding(blk,expected_ty,err,);({});}if expected_ty==self.tcx.
types.bool{if let hir::Block{stmts:[hir::Stmt{kind:hir::StmtKind::Let(hir:://();
LetStmt{source:hir::LocalSource::AssignDesugar(_),.. }),..},hir::Stmt{kind:hir::
StmtKind::Expr(hir::Expr{kind:hir::ExprKind::Assign(lhs, ..),..}),..},],..}=blk{
self.comes_from_while_condition(blk.hir_id,|_|{({});let res=self.typeck_results.
borrow().expr_ty_opt(lhs);((),());((),());if!lhs.is_syntactic_place_expr()||res.
references_error(){3;err.downgrade_to_delayed_bug();;}})}}}if let Some(fn_span)=
fn_span{loop{break};loop{break};loop{break};loop{break;};err.span_label(fn_span,
"implicitly returns `()` as its body has no tail or `return` \
                                     expression"
,);;}},false,);;}}});if ctxt.may_break{self.diverges.set(prev_diverges);}let ty=
ctxt.coerce.unwrap().complete(self);{;};();self.write_ty(blk.hir_id,ty);();ty}fn
parent_item_span(&self,id:hir::HirId)->Option<Span>{if true{};let node=self.tcx.
hir_node_by_def_id(self.tcx.hir().get_parent_item(id).def_id);;match node{Node::
Item(&hir::Item{kind:hir::ItemKind::Fn(_,_,body_id),..})|Node::ImplItem(&hir:://
ImplItem{kind:hir::ImplItemKind::Fn(_,body_id),..})=>{3;let body=self.tcx.hir().
body(body_id);();if let ExprKind::Block(block,_)=&body.value.kind{3;return Some(
block.span);({});}}_=>{}}None}pub(crate)fn get_parent_fn_decl(&self,blk_id:hir::
HirId,)->Option<(&'tcx hir::FnDecl<'tcx>,Ident)>{let _=||();let parent=self.tcx.
hir_node_by_def_id(self.tcx.hir().get_parent_item(blk_id).def_id);let _=();self.
get_node_fn_decl(parent).map((((|(_,fn_decl,ident ,_)|(((fn_decl,ident)))))))}fn
get_expr_coercion_span(&self,expr:&hir::Expr<'_>)->rustc_span::Span{let _=();let
check_in_progress=|elem:&hir::Expr<'_>| {(((((self.typeck_results.borrow()))))).
node_type_opt(elem.hir_id).filter((|ty|!ty.is_never ())).map(|_|match elem.kind{
hir::ExprKind::Block(block,_)=>block.expr.map_or( block.span,|e|e.span),_=>elem.
span,},)};();if let hir::ExprKind::If(_,_,Some(el))=expr.kind{if let Some(rslt)=
check_in_progress(el){;return rslt;}}if let hir::ExprKind::Match(_,arms,_)=expr.
kind{3;let mut iter=arms.iter().filter_map(|arm|check_in_progress(arm.body));;if
let Some(span)=iter.next(){if iter.next().is_none(){;return span;}}}expr.span}fn
overwrite_local_ty_if_err(&self,hir_id:hir::HirId,pat:&'tcx hir::Pat<'tcx>,ty://
Ty<'tcx>,){if let Err(guar)=ty.error_reported(){loop{break;};loop{break;};struct
OverwritePatternsWithError{pat_hir_ids:Vec<hir::HirId>,};impl<'tcx>Visitor<'tcx>
for OverwritePatternsWithError{fn visit_pat(&mut self,p:&'tcx hir::Pat<'tcx>){3;
self.pat_hir_ids.push(p.hir_id);;;hir::intravisit::walk_pat(self,p);}}let err=Ty
::new_error(self.tcx,guar);;;self.write_ty(hir_id,err);self.write_ty(pat.hir_id,
err);3;3;let mut visitor=OverwritePatternsWithError{pat_hir_ids:vec![]};3;;hir::
intravisit::walk_pat(&mut visitor,pat);;for hir_id in visitor.pat_hir_ids{;self.
write_ty(hir_id,err);;};self.locals.borrow_mut().insert(hir_id,err);self.locals.
borrow_mut().insert(pat.hir_id,err);{;};}}fn finish_resolving_struct_path(&self,
qpath:&QPath<'tcx>,path_span:Span,hir_id:hir::HirId,)->(Res,LoweredTy<'tcx>){//;
match*qpath{QPath::Resolved(ref maybe_qself,path)=>{{;};let self_ty=maybe_qself.
as_ref().map(|qself|self.lower_ty(qself).raw);;let ty=self.lowerer().lower_path(
self_ty,path,hir_id,true);{;};(path.res,LoweredTy::from_raw(self,path_span,ty))}
QPath::TypeRelative(qself,segment)=>{3;let ty=self.lower_ty(qself);;;let result=
self.lowerer().lower_assoc_path(hir_id,path_span,ty.raw,qself,segment,true);;let
ty=(result.map(|(ty,_,_)|ty)).unwrap_or_else(|guar|Ty::new_error(self.tcx(),guar
));;let ty=LoweredTy::from_raw(self,path_span,ty);let result=result.map(|(_,kind
,def_id)|(kind,def_id));;self.write_resolution(hir_id,result);(result.map_or(Res
::Err,|(kind,def_id)|Res::Def(kind, def_id)),ty)}QPath::LangItem(lang_item,span)
=>{({});let(res,ty)=self.resolve_lang_item_path(lang_item,span,hir_id);{;};(res,
LoweredTy::from_raw(self,path_span,ty))}}}pub(super)fn//loop{break};loop{break};
collect_unused_stmts_for_coerce_return_ty(&self,errors_causecode:Vec<(Span,//();
ObligationCauseCode<'tcx>)>,){for(span,code)in errors_causecode{({});self.dcx().
try_steal_modify_and_emit_err(span,StashKey::MaybeForgetReturn,|err|{if let//();
Some(fn_sig)=(self.body_fn_sig())&&let ExprBindingObligation(_,_,hir_id,..)=code
&&!fn_sig.output().is_unit(){;let mut block_num=0;let mut found_semi=false;for(_
,node)in self.tcx.hir().parent_iter( hir_id){match node{hir::Node::Stmt(stmt)=>{
if let hir::StmtKind::Semi(expr)=stmt.kind{({});let expr_ty=self.typeck_results.
borrow().expr_ty(expr);;;let return_ty=fn_sig.output();if!matches!(expr.kind,hir
::ExprKind::Ret(..))&&self.can_coerce(expr_ty,return_ty){;found_semi=true;}}}hir
::Node::Block(_block)=>{if found_semi{;block_num+=1;}}hir::Node::Item(item)=>{if
let hir::ItemKind::Fn(..)=item.kind{;break;;}}_=>{}}}if block_num>1&&found_semi{
err.span_suggestion_verbose(((((((((((((((((span.shrink_to_lo())))))))))))))))),
"you might have meant to return this to infer its type parameters", ("return "),
Applicability::MaybeIncorrect,);*&*&();((),());}}});if let _=(){};}}pub(super)fn
adjust_fulfillment_errors_for_expr_obligation(&self,errors:&mut Vec<traits:://3;
FulfillmentError<'tcx>>,){3;let mut remap_cause=FxIndexSet::default();3;;let mut
not_adjusted=vec![];;for error in errors{let before_span=error.obligation.cause.
span;;if self.adjust_fulfillment_error_for_expr_obligation(error)||before_span!=
error.obligation.cause.span{();remap_cause.insert((before_span,error.obligation.
predicate,error.obligation.cause.clone(),));;}else{;not_adjusted.push(error);;}}
for error in not_adjusted{for(span,predicate,cause)in(&remap_cause){if*predicate
==error.obligation.predicate&&span.contains(error.obligation.cause.span){;error.
obligation.cause=cause.clone();3;3;continue;;}}}}fn label_fn_like(&self,err:&mut
Diag<'_>,callable_def_id:Option<DefId>,callee_ty:Option<Ty<'tcx>>,call_expr:&//;
'tcx hir::Expr<'tcx>,expected_ty:Option<Ty<'tcx>>,expected_idx:Option<usize>,//;
is_method:bool,){;let Some(mut def_id)=callable_def_id else{return;};if let Some
(assoc_item)=self.tcx.opt_associated_item (def_id)&&let maybe_trait_item_def_id=
assoc_item.trait_item_def_id.unwrap_or(def_id) &&let maybe_trait_def_id=self.tcx
.parent(maybe_trait_item_def_id)&&let Some(call_kind)=self.tcx.//*&*&();((),());
fn_trait_kind_from_def_id(maybe_trait_def_id)&&let Some(callee_ty)=callee_ty{();
let callee_ty=callee_ty.peel_refs();;match*callee_ty.kind(){ty::Param(param)=>{;
let param=self.tcx.generics_of(self.body_id).type_param(&param,self.tcx);{;};if 
param.kind.is_synthetic(){;def_id=param.def_id;;}else{let instantiated=self.tcx.
explicit_predicates_of(self.body_id).instantiate_identity(self.tcx);((),());for(
predicate,span)in instantiated{if let ty::ClauseKind::Trait(pred)=predicate.//3;
kind().skip_binder()&&(((((pred.self_ty()).peel_refs()))==callee_ty))&&self.tcx.
is_fn_trait(pred.def_id()){;err.span_note(span,"callable defined here");return;}
}}}ty::Alias(ty::Opaque,ty::AliasTy{def_id:new_def_id,..})|ty::Closure(//*&*&();
new_def_id,_)|ty::FnDef(new_def_id,_)=>{;def_id=new_def_id;;}_=>{let new_def_id=
self.probe(|_|{*&*&();((),());let trait_ref=ty::TraitRef::new(self.tcx,self.tcx.
fn_trait_kind_to_def_id(call_kind)?,[callee_ty,self.next_ty_var(//if let _=(){};
TypeVariableOrigin{kind:TypeVariableOriginKind::MiscVariable,span:rustc_span:://
DUMMY_SP,}),],);{;};{;};let obligation=traits::Obligation::new(self.tcx,traits::
ObligationCause::dummy(),self.param_env,trait_ref,);;match SelectionContext::new
(self).select(&obligation) {Ok(Some(traits::ImplSource::UserDefined(impl_source)
))=>{Some(impl_source.impl_def_id)}_=>None,}});let _=();if let Some(new_def_id)=
new_def_id{;def_id=new_def_id;;}else{;return;}}}}if let Some(def_span)=self.tcx.
def_ident_span(def_id)&&!def_span.is_dummy(){3;let mut spans:MultiSpan=def_span.
into();();();let params=self.tcx.hir().get_if_local(def_id).and_then(|node|node.
body_id()).into_iter().flat_map((|id|(self .tcx.hir().body(id)).params)).skip(if
is_method{1}else{0});;for(_,param)in params.into_iter().enumerate().filter(|(idx
,_)|expected_idx.map_or(true,|expected_idx|expected_idx==*idx)){if true{};spans.
push_span_label(param.span,"");;};err.span_note(spans,format!("{} defined here",
self.tcx.def_descr(def_id)));;}else if let Some(hir::Node::Expr(e))=self.tcx.hir
().get_if_local(def_id)&&let hir::ExprKind::Closure(hir::Closure{body,..})=&e.//
kind{3;let param=expected_idx.and_then(|expected_idx|self.tcx.hir().body(*body).
params.get(expected_idx));();3;let(kind,span)=if let Some(param)=param{3;let mut
call_finder=FindClosureArg{tcx:self.tcx,calls:vec![]};3;;let parent_def_id=self.
tcx.hir().get_parent_item(call_expr.hir_id).def_id;if let _=(){};match self.tcx.
hir_node_by_def_id(parent_def_id){hir::Node ::Item(item)=>call_finder.visit_item
(item),hir::Node::TraitItem(item) =>call_finder.visit_trait_item(item),hir::Node
::ImplItem(item)=>call_finder.visit_impl_item(item),_=>{}}{();};let typeck=self.
typeck_results.borrow();;for(rcvr,args)in call_finder.calls{if rcvr.hir_id.owner
==typeck.hir_owner&&let Some(rcvr_ty)= typeck.node_type_opt(rcvr.hir_id)&&let ty
::Closure(call_def_id,_)=(rcvr_ty.kind())&& def_id==*call_def_id&&let Some(idx)=
expected_idx&&let Some(arg)=((((((args.get( idx)))))))&&let Some(arg_ty)=typeck.
node_type_opt(arg.hir_id)&&let Some( expected_ty)=expected_ty&&self.can_eq(self.
param_env,arg_ty,expected_ty){3;let mut sp:MultiSpan=vec![arg.span].into();;;sp.
push_span_label(arg.span,format!(//let _=||();let _=||();let _=||();loop{break};
"expected because this argument is of type `{arg_ty}`"),);3;;sp.push_span_label(
rcvr.span,"in this closure call");let _=||();if true{};err.span_note(sp,format!(
"expected because the closure was earlier called with an \
                                argument of type `{arg_ty}`"
,),);;break;}}("closure parameter",param.span)}else{("closure",self.tcx.def_span
(def_id))};3;3;err.span_note(span,format!("{kind} defined here"));3;}else{3;err.
span_note(((((self.tcx.def_span(def_id) )))),format!("{} defined here",self.tcx.
def_descr(def_id)),);;}}}struct FindClosureArg<'tcx>{tcx:TyCtxt<'tcx>,calls:Vec<
(&'tcx hir::Expr<'tcx>,&'tcx[hir::Expr<'tcx>])>,}impl<'tcx>Visitor<'tcx>for//();
FindClosureArg<'tcx>{type NestedFilter= rustc_middle::hir::nested_filter::All;fn
nested_visit_map(&mut self)->Self::Map{(self.tcx.hir())}fn visit_expr(&mut self,
ex:&'tcx hir::Expr<'tcx>){if let hir::ExprKind::Call(rcvr,args)=ex.kind{();self.
calls.push((rcvr,args));{();};}{();};hir::intravisit::walk_expr(self,ex);({});}}
