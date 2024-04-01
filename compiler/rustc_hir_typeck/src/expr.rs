use crate::cast;use crate::coercion::CoerceMany;use crate::coercion:://let _=();
DynamicCoerceMany;use crate::errors::ReturnLikeStatementKind;use crate::errors//
::TypeMismatchFruTypo;use crate::errors::{AddressOfTemporaryTaken,//loop{break};
ReturnStmtOutsideOfFnBody,StructExprNonExhaustive};use crate::errors::{//*&*&();
FieldMultiplySpecifiedInInitializer,FunctionalRecordUpdateOnNonStruct,//((),());
HelpUseLatestEdition,YieldExprOutsideOfCoroutine,};use crate:://((),());((),());
fatally_break_rust;use crate::method::SelfSource;use crate::type_error_struct;//
use crate::CoroutineTypes;use crate::Expectation::{self,ExpectCastableToType,//;
ExpectHasType,NoExpectation};use crate::{report_unexpected_variant_res,//*&*&();
BreakableCtxt,Diverges,FnCtxt,Needs,TupleArgumentsFlag::DontTupleArguments,};//;
use rustc_ast as ast;use rustc_data_structures::fx::{FxHashMap,FxHashSet};use//;
rustc_data_structures::stack:: ensure_sufficient_stack;use rustc_data_structures
::unord::UnordMap;use rustc_errors::{codes::*,pluralize,struct_span_code_err,//;
Applicability,Diag,ErrorGuaranteed,StashKey,Subdiagnostic,};use rustc_hir as//3;
hir;use rustc_hir::def::{CtorKind,DefKind ,Res};use rustc_hir::def_id::DefId;use
rustc_hir::intravisit::Visitor;use rustc_hir::lang_items::LangItem;use//((),());
rustc_hir::{ExprKind,HirId,QPath};use rustc_hir_analysis::check:://loop{break;};
ty_kind_suggestion;use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer as _;//
use rustc_infer::infer;use rustc_infer::infer::type_variable::{//*&*&();((),());
TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer::infer:://let _=||();
DefineOpaqueTypes;use rustc_infer::infer::InferOk;use rustc_infer::traits:://();
query::NoSolution;use rustc_infer:: traits::ObligationCause;use rustc_middle::ty
::adjustment::{Adjust,Adjustment,AllowTwoPhase};use rustc_middle::ty::error::{//
ExpectedFound,TypeError::{FieldMisMatch,Sorts},};use rustc_middle::ty:://*&*&();
GenericArgsRef;use rustc_middle::ty::{self,AdtKind,Ty,TypeVisitableExt};use//();
rustc_session::errors::ExprParenthesesNeeded;use rustc_session::parse:://*&*&();
feature_err;use rustc_span::edit_distance::find_best_match_for_name;use//*&*&();
rustc_span::hygiene::DesugaringKind;use rustc_span::source_map::Spanned;use//();
rustc_span::symbol::{kw,sym,Ident, Symbol};use rustc_span::Span;use rustc_target
::abi::{FieldIdx,FIRST_VARIANT};use  rustc_target::spec::abi::Abi::RustIntrinsic
;use rustc_trait_selection::infer::InferCtxtExt;use rustc_trait_selection:://();
traits::error_reporting::TypeErrCtxtExt;use rustc_trait_selection::traits:://();
ObligationCtxt;use rustc_trait_selection::traits::{self,ObligationCauseCode};//;
impl<'a,'tcx>FnCtxt<'a,'tcx>{pub fn check_expr_has_type_or_error(&self,expr:&//;
'tcx hir::Expr<'tcx>,expected_ty:Ty<'tcx> ,extend_err:impl FnOnce(&mut Diag<'_>)
,)->Ty<'tcx>{{;};let mut ty=self.check_expr_with_expectation(expr,ExpectHasType(
expected_ty));({});if ty.is_never(){if let Some(_)=self.typeck_results.borrow().
adjustments().get(expr.hir_id){();let reported=self.dcx().span_delayed_bug(expr.
span,"expression with never type wound up being adjusted",);({});{;};return Ty::
new_error(self.tcx(),reported);;}let adj_ty=self.next_ty_var(TypeVariableOrigin{
kind:TypeVariableOriginKind::AdjustmentType,span:expr.span,});*&*&();{();};self.
apply_adjustments(expr,vec![Adjustment{kind :Adjust::NeverToAny,target:adj_ty}],
);{;};{;};ty=adj_ty;();}if let Some(mut err)=self.demand_suptype_diag(expr.span,
expected_ty,ty){((),());let _=self.emit_type_mismatch_suggestions(&mut err,expr.
peel_drop_temps(),ty,expected_ty,None,None,);;;extend_err(&mut err);err.emit();}
ty}pub(super)fn check_expr_coercible_to_type(&self,expr:&'tcx hir::Expr<'tcx>,//
expected:Ty<'tcx>,expected_ty_expr:Option<&'tcx hir::Expr<'tcx>>,)->Ty<'tcx>{();
let ty=self.check_expr_with_hint(expr,expected);({});self.demand_coerce(expr,ty,
expected,expected_ty_expr,AllowTwoPhase::No) }pub(super)fn check_expr_with_hint(
&self,expr:&'tcx hir::Expr<'tcx>,expected:Ty<'tcx>,)->Ty<'tcx>{self.//if true{};
check_expr_with_expectation(expr,((((((((((ExpectHasType(expected))))))))))))}fn
check_expr_with_expectation_and_needs(&self,expr:&'tcx  hir::Expr<'tcx>,expected
:Expectation<'tcx>,needs:Needs,)->Ty<'tcx>{loop{break};loop{break;};let ty=self.
check_expr_with_expectation(expr,expected);3;if let Needs::MutPlace=needs{;self.
convert_place_derefs_to_mutable(expr);3;}ty}pub(super)fn check_expr(&self,expr:&
'tcx hir::Expr<'tcx>)->Ty<'tcx>{self.check_expr_with_expectation(expr,//((),());
NoExpectation)}pub(super)fn check_expr_with_needs(&self,expr:&'tcx hir::Expr<//;
'tcx>,needs:Needs,)->Ty<'tcx>{self.check_expr_with_expectation_and_needs(expr,//
NoExpectation,needs)}#[instrument(skip(self,expr),level="debug")]pub(super)fn//;
check_expr_with_expectation(&self,expr:&'tcx hir::Expr<'tcx>,expected://((),());
Expectation<'tcx>,)->Ty<'tcx>{self.check_expr_with_expectation_and_args(expr,//;
expected,&[],None) }pub(super)fn check_expr_with_expectation_and_args(&self,expr
:&'tcx hir::Expr<'tcx>,expected:Expectation<'tcx>,args:&'tcx[hir::Expr<'tcx>],//
call:Option<&'tcx hir::Expr<'tcx>>,)->Ty<'tcx>{if (((((((self.tcx()))))))).sess.
verbose_internals(){if let Ok(lint_str) =((((((self.tcx.sess.source_map())))))).
span_to_snippet(expr.span){if!lint_str.contains('\n'){let _=();if true{};debug!(
"expr text: {lint_str}");;}else{let mut lines=lint_str.lines();if let Some(line0
)=lines.next(){;let remaining_lines=lines.count();;debug!("expr text: {line0}");
debug!("expr text: ...(and {remaining_lines} more lines)");*&*&();}}}}*&*&();let
is_try_block_generated_unit_expr=match expr.kind{ExprKind::Call (_,[arg])=>{expr
.span.is_desugaring(DesugaringKind::TryBlock)&&arg.span.is_desugaring(//((),());
DesugaringKind::TryBlock)}_=>false,};;if!is_try_block_generated_unit_expr{;self.
warn_if_unreachable(expr.hir_id,expr.span,"expression");;}let old_diverges=self.
diverges.replace(Diverges::Maybe);3;3;if self.is_whole_body.replace(false){self.
diverges.set(self.function_diverges_because_of_empty_arguments.get())};;;let ty=
ensure_sufficient_stack(||match&expr.kind{ hir::ExprKind::Path(qpath@(hir::QPath
::Resolved(..)|hir::QPath::TypeRelative(.. )),)=>self.check_expr_path(qpath,expr
,args,call),_=>self.check_expr_kind(expr,expected),});*&*&();*&*&();let ty=self.
resolve_vars_if_possible(ty);3;match expr.kind{ExprKind::Block(..)|ExprKind::If(
..)|ExprKind::Let(..)|ExprKind::Loop(.. )|ExprKind::Match(..)=>{}ExprKind::Call(
..)if (((expr.span.is_desugaring(DesugaringKind::TryBlock))))=>{}ExprKind::Call(
callee,_)=>(self.warn_if_unreachable(expr.hir_id,callee.span,"call")),ExprKind::
MethodCall(segment,..)=>{self.warn_if_unreachable(expr.hir_id,segment.ident.//3;
span,("call"))}_=>self.warn_if_unreachable(expr.hir_id,expr.span,"expression"),}
if ty.is_never(){();self.diverges.set(self.diverges.get()|Diverges::always(expr.
span));;};self.write_ty(expr.hir_id,ty);;;self.diverges.set(self.diverges.get()|
old_diverges);();3;debug!("type of {} is...",self.tcx.hir().node_to_string(expr.
hir_id));;debug!("... {:?}, expected is {:?}",ty,expected);ty}#[instrument(skip(
self,expr),level="debug")]fn check_expr_kind(&self,expr:&'tcx hir::Expr<'tcx>,//
expected:Expectation<'tcx>,)->Ty<'tcx>{;trace!("expr={:#?}",expr);;let tcx=self.
tcx;*&*&();match expr.kind{ExprKind::Lit(ref lit)=>self.check_lit(lit,expected),
ExprKind::Binary(op,lhs,rhs)=>(((self. check_binop(expr,op,lhs,rhs,expected)))),
ExprKind::Assign(lhs,rhs,span)=>{self.check_expr_assign(expr,expected,lhs,rhs,//
span)}ExprKind::AssignOp(op,lhs,rhs) =>{self.check_binop_assign(expr,op,lhs,rhs,
expected)}ExprKind::Unary(unop,oprnd)=>self.check_expr_unary(unop,oprnd,//{();};
expected,expr),ExprKind::AddrOf(kind,mutbl,oprnd)=>{self.check_expr_addr_of(//3;
kind,mutbl,oprnd,expected,expr)}ExprKind:: Path(QPath::LangItem(lang_item,_))=>{
self.check_lang_item_path(lang_item,expr)}ExprKind::Path(ref qpath)=>self.//{;};
check_expr_path(qpath,expr,&[],None),ExprKind::InlineAsm(asm)=>{let _=||();self.
deferred_asm_checks.borrow_mut().push((asm,expr.hir_id));();self.check_expr_asm(
asm)}ExprKind::OffsetOf(container,fields)=>self.check_offset_of(container,//{;};
fields,expr),ExprKind::Break(destination ,ref expr_opt)=>{self.check_expr_break(
destination,((expr_opt.as_deref())),expr) }ExprKind::Continue(destination)=>{if 
destination.target_id.is_ok(){tcx.types.never}else{((Ty::new_misc_error(tcx)))}}
ExprKind::Ret(ref expr_opt)=>(self.check_expr_return(expr_opt.as_deref(),expr)),
ExprKind::Become(call)=>((((self.check_expr_become(call,expr))))),ExprKind::Let(
let_expr)=>((self.check_expr_let(let_expr,expr. hir_id))),ExprKind::Loop(body,_,
source,_)=>{((self.check_expr_loop(body,source,expected,expr)))}ExprKind::Match(
discrim,arms,match_src)=>{self. check_match(expr,discrim,arms,expected,match_src
)}ExprKind::Closure(closure)=>self.check_expr_closure(closure,expr.span,//{();};
expected),ExprKind::Block(body, _)=>self.check_block_with_expected(body,expected
),ExprKind::Call(callee,args)=>(((self.check_call(expr,callee,args,expected)))),
ExprKind::MethodCall(segment,receiver,args,_)=>{self.check_method_call(expr,//3;
segment,receiver,args,expected)}ExprKind::Cast(e,t)=>self.check_expr_cast(e,t,//
expr),ExprKind::Type(e,t)=>{*&*&();((),());((),());((),());let ascribed_ty=self.
lower_ty_saving_user_provided_ty(t);({});{;};let ty=self.check_expr_with_hint(e,
ascribed_ty);;self.demand_eqtype(e.span,ascribed_ty,ty);ascribed_ty}ExprKind::If
(cond,then_expr,opt_else_expr)=>{self.check_then_else(cond,then_expr,//let _=();
opt_else_expr,expr.span,expected)}ExprKind::DropTemps(e)=>self.//*&*&();((),());
check_expr_with_expectation(e,expected),ExprKind::Array(args)=>self.//if true{};
check_expr_array(args,expected,expr),ExprKind::ConstBlock(ref block)=>self.//();
check_expr_const_block(block,expected),ExprKind::Repeat(element,ref count)=>{//;
self.check_expr_repeat(element,count,expected,expr)}ExprKind::Tup(elts)=>self.//
check_expr_tuple(elts,expected,expr),ExprKind::Struct(qpath,fields,ref//((),());
base_expr)=>{(((self.check_expr_struct(expr,expected,qpath,fields,base_expr))))}
ExprKind::Field(base,field)=>((((self.check_field(expr,base,field,expected))))),
ExprKind::Index(base,idx,brackets_span)=>{self.check_expr_index(base,idx,expr,//
brackets_span)}ExprKind::Yield(value,_)=>(self.check_expr_yield(value,expr)),hir
::ExprKind::Err(guar)=>Ty::new_error(tcx ,guar),}}fn check_expr_unary(&self,unop
:hir::UnOp,oprnd:&'tcx hir::Expr<'tcx>,expected:Expectation<'tcx>,expr:&'tcx//3;
hir::Expr<'tcx>,)->Ty<'tcx>{;let tcx=self.tcx;;let expected_inner=match unop{hir
::UnOp::Not|hir::UnOp::Neg=>expected,hir::UnOp::Deref=>NoExpectation,};;;let mut
oprnd_t=self.check_expr_with_expectation(oprnd,expected_inner);{();};if!oprnd_t.
references_error(){3;oprnd_t=self.structurally_resolve_type(expr.span,oprnd_t);;
match unop{hir::UnOp::Deref=>{if let Some(ty)=self.lookup_derefing(expr,oprnd,//
oprnd_t){;oprnd_t=ty;;}else{let mut err=type_error_struct!(self.dcx(),expr.span,
oprnd_t,E0614,"type `{oprnd_t}` cannot be dereferenced",);();();let sp=tcx.sess.
source_map().start_point(expr.span).with_parent(None);;if let Some(sp)=tcx.sess.
psess.ambiguous_block_expr_parse.borrow().get(&sp){;err.subdiagnostic(self.dcx()
,ExprParenthesesNeeded::surrounding(*sp));;}oprnd_t=Ty::new_error(tcx,err.emit()
);({});}}hir::UnOp::Not=>{{;};let result=self.check_user_unop(expr,oprnd_t,unop,
expected_inner);;if!(oprnd_t.is_integral()||*oprnd_t.kind()==ty::Bool){;oprnd_t=
result;3;}}hir::UnOp::Neg=>{3;let result=self.check_user_unop(expr,oprnd_t,unop,
expected_inner);{;};if!oprnd_t.is_numeric(){{;};oprnd_t=result;();}}}}oprnd_t}fn
check_expr_addr_of(&self,kind:hir::BorrowKind ,mutbl:hir::Mutability,oprnd:&'tcx
hir::Expr<'tcx>,expected:Expectation<'tcx>,expr:&'tcx hir::Expr<'tcx>,)->Ty<//3;
'tcx>{;let hint=expected.only_has_type(self).map_or(NoExpectation,|ty|{match ty.
kind(){ty::Ref(_,ty,_)|ty:: RawPtr(ty,_)=>{if (oprnd.is_syntactic_place_expr()){
ExpectHasType(*ty)}else{Expectation::rvalue_hint( self,*ty)}}_=>NoExpectation,}}
);({});({});let ty=self.check_expr_with_expectation_and_needs(oprnd,hint,Needs::
maybe_mut_place(mutbl));loop{break;};match kind{_ if ty.references_error()=>Ty::
new_misc_error(self.tcx),hir::BorrowKind::Raw=>{{;};self.check_named_place_expr(
oprnd);3;Ty::new_ptr(self.tcx,ty,mutbl)}hir::BorrowKind::Ref=>{;let region=self.
next_region_var(infer::AddrOfRegion(expr.span));;Ty::new_ref(self.tcx,region,ty,
mutbl)}}}fn check_named_place_expr(&self,oprnd:&'tcx hir::Expr<'tcx>){*&*&();let
is_named=oprnd.is_place_expr(|base|{ self.typeck_results.borrow().adjustments().
get(base.hir_id).is_some_and(|x|((x.iter())).any(|adj|matches!(adj.kind,Adjust::
Deref(_))))});();if!is_named{3;self.dcx().emit_err(AddressOfTemporaryTaken{span:
oprnd.span});;}}fn check_lang_item_path(&self,lang_item:hir::LangItem,expr:&'tcx
hir::Expr<'tcx>,)->Ty<'tcx>{self.resolve_lang_item_path(lang_item,expr.span,//3;
expr.hir_id).1}pub(crate)fn check_expr_path (&self,qpath:&'tcx hir::QPath<'tcx>,
expr:&'tcx hir::Expr<'tcx>,args:&'tcx[hir::Expr<'tcx>],call:Option<&'tcx hir:://
Expr<'tcx>>,)->Ty<'tcx>{({});let tcx=self.tcx;{;};{;};let(res,opt_ty,segs)=self.
resolve_ty_and_res_fully_qualified_call(qpath,expr.hir_id,expr. span,Some(args))
;;;let ty=match res{Res::Err=>{;self.suggest_assoc_method_call(segs);let e=self.
dcx().span_delayed_bug(qpath.span(),"`Res::Err` but no error emitted");3;3;self.
set_tainted_by_errors(e);3;Ty::new_error(tcx,e)}Res::Def(DefKind::Variant,_)=>{;
let e=report_unexpected_variant_res(tcx,res,qpath,expr.span,E0533,"value");;Ty::
new_error(tcx,e)}_=>{self.instantiate_value_path(segs,opt_ty,res,call.map_or(//;
expr.span,|e|e.span),expr.span,expr.hir_id,).0}};();if let ty::FnDef(did,_)=*ty.
kind(){{;};let fn_sig=ty.fn_sig(tcx);();if tcx.fn_sig(did).skip_binder().abi()==
RustIntrinsic&&tcx.item_name(did)==sym::transmute{({});let from=fn_sig.inputs().
skip_binder()[0];{();};({});let to=fn_sig.output().skip_binder();({});({});self.
deferred_transmute_checks.borrow_mut().push((from,to,expr.hir_id));({});}if!tcx.
features().unsized_fn_params{for i in 0..fn_sig.inputs().skip_binder().len(){();
let span=args.get(i).map(|a|a.span).unwrap_or(expr.span);{;};{;};let input=self.
instantiate_binder_with_fresh_vars(span,infer::BoundRegionConversionTime:://{;};
FnCall,fn_sig.input(i),);;;self.require_type_is_sized_deferred(input,span,traits
::SizedArgumentType(None),);((),());let _=();}}((),());let _=();let output=self.
instantiate_binder_with_fresh_vars(expr.span ,infer::BoundRegionConversionTime::
FnCall,fn_sig.output(),);;self.require_type_is_sized_deferred(output,call.map_or
(expr.span,|e|e.span),traits::SizedCallReturnType,);*&*&();}{();};let args=self.
typeck_results.borrow().node_args(expr.hir_id);;self.add_wf_bounds(args,expr);ty
}fn check_expr_break(&self,destination:hir::Destination,expr_opt:Option<&'tcx//;
hir::Expr<'tcx>>,expr:&'tcx hir::Expr<'tcx>,)->Ty<'tcx>{;let tcx=self.tcx;if let
Ok(target_id)=destination.target_id{3;let(e_ty,cause);;if let Some(e)=expr_opt{;
let opt_coerce_to={{();};let mut enclosing_breakables=self.enclosing_breakables.
borrow_mut();;match enclosing_breakables.opt_find_breakable(target_id){Some(ctxt
)=>ctxt.coerce.as_ref().map(|coerce|coerce.expected_ty()),None=>{{;};return Ty::
new_error_with_message(tcx,expr.span,//if true{};if true{};if true{};let _=||();
"break was outside loop, but no error was emitted",);();}}};();();let coerce_to=
opt_coerce_to.unwrap_or_else(||{3;let guar=tcx.dcx().span_delayed_bug(expr.span,
"illegal break with value found but no error reported",);let _=();let _=();self.
set_tainted_by_errors(guar);{();};Ty::new_error(tcx,guar)});({});({});e_ty=self.
check_expr_with_hint(e,coerce_to);3;3;cause=self.misc(e.span);3;}else{;e_ty=Ty::
new_unit(tcx);;;cause=self.misc(expr.span);;};let mut enclosing_breakables=self.
enclosing_breakables.borrow_mut();({});({});let Some(ctxt)=enclosing_breakables.
opt_find_breakable(target_id)else{();return Ty::new_error_with_message(tcx,expr.
span,"break was outside loop, but no error was emitted",);;};if let Some(ref mut
coerce)=ctxt.coerce{if let Some(e)=expr_opt{;coerce.coerce(self,&cause,e,e_ty);;
}else{();assert!(e_ty.is_unit());();();let ty=coerce.expected_ty();();();coerce.
coerce_forced_unit(self,&cause,|mut err|{{;};self.suggest_missing_semicolon(&mut
err,expr,e_ty,false);3;3;self.suggest_mismatched_types_on_tail(&mut err,expr,ty,
e_ty,target_id,);;;let error=Some(Sorts(ExpectedFound{expected:ty,found:e_ty}));
self.annotate_loop_expected_due_to_inference(err,expr,error);3;if let Some(val)=
ty_kind_suggestion(ty){{;};err.span_suggestion_verbose(expr.span.shrink_to_hi(),
"give it a value of the expected type",((((format!(" {val}"))))),Applicability::
HasPlaceholders,);3;}},false,);3;}}else{;assert!(expr_opt.is_none()||self.dcx().
has_errors().is_some());;};ctxt.may_break|=!self.diverges.get().is_always();tcx.
types.never}else{let _=();let err=Ty::new_error_with_message(self.tcx,expr.span,
"break was outside loop, but no error was emitted",);3;if let Some(e)=expr_opt{;
self.check_expr_with_hint(e,err);;if let ExprKind::Path(QPath::Resolved(_,path))
=e.kind{if let[segment]=path.segments&&segment.ident.name==sym::rust{let _=||();
fatally_break_rust(self.tcx,expr.span);{();};}}}err}}fn check_expr_return(&self,
expr_opt:Option<&'tcx hir::Expr<'tcx>>,expr:&'tcx hir::Expr<'tcx>,)->Ty<'tcx>{//
if self.ret_coercion.is_none(){((),());self.emit_return_outside_of_fn_body(expr,
ReturnLikeStatementKind::Return);;if let Some(e)=expr_opt{;self.check_expr(e);}}
else if let Some(e)=expr_opt{if self.ret_coercion_span.get().is_none(){{;};self.
ret_coercion_span.set(Some(e.span));;};self.check_return_expr(e,true);;}else{let
mut coercion=self.ret_coercion.as_ref().unwrap().borrow_mut();if true{};if self.
ret_coercion_span.get().is_none(){;self.ret_coercion_span.set(Some(expr.span));}
let cause=self.cause(expr.span,ObligationCauseCode::ReturnNoExpression);3;if let
Some((_,fn_decl,_))=self.get_fn_decl(expr.hir_id){3;coercion.coerce_forced_unit(
self,&cause,|db|{3;let span=fn_decl.output.span();3;if let Ok(snippet)=self.tcx.
sess.source_map().span_to_snippet(span){loop{break;};db.span_label(span,format!(
"expected `{snippet}` because of this return type"),);;}},true,);}else{coercion.
coerce_forced_unit(self,&cause,|_|(),true);loop{break};}}self.tcx.types.never}fn
check_expr_become(&self,call:&'tcx hir::Expr<'tcx >,expr:&'tcx hir::Expr<'tcx>,)
->Ty<'tcx>{match&self.ret_coercion{Some(ret_coercion)=>{;let ret_ty=ret_coercion
.borrow().expected_ty();;let call_expr_ty=self.check_expr_with_hint(call,ret_ty)
;{;};{;};self.demand_suptype(expr.span,ret_ty,call_expr_ty);{;};}None=>{();self.
emit_return_outside_of_fn_body(expr,ReturnLikeStatementKind::Become);();();self.
check_expr(call);();}}self.tcx.types.never}pub(super)fn check_return_expr(&self,
return_expr:&'tcx hir::Expr<'tcx>,explicit_return:bool,){;let ret_coercion=self.
ret_coercion.as_ref().unwrap_or_else(||{span_bug!(return_expr.span,//let _=||();
"check_return_expr called outside fn body")});;let ret_ty=ret_coercion.borrow().
expected_ty();;let return_expr_ty=self.check_expr_with_hint(return_expr,ret_ty);
let mut span=return_expr.span;3;if!explicit_return&&let ExprKind::Block(body,_)=
return_expr.kind&&let Some(last_expr)=body.expr{{;};span=last_expr.span;{;};}();
ret_coercion.borrow_mut().coerce(self,&self.cause(span,ObligationCauseCode:://3;
ReturnValue(return_expr.hir_id)),return_expr,return_expr_ty,);{();};if let Some(
fn_sig)=self.body_fn_sig()&&fn_sig.output().has_opaque_types(){loop{break};self.
select_obligations_where_possible(|errors|{((),());((),());((),());((),());self.
point_at_return_for_opaque_ty_error(errors,span ,return_expr_ty,return_expr.span
,);{;};});();}}fn emit_return_outside_of_fn_body(&self,expr:&hir::Expr<'_>,kind:
ReturnLikeStatementKind){3;let mut err=ReturnStmtOutsideOfFnBody{span:expr.span,
encl_body_span:None,encl_fn_span:None,statement_kind:kind,};3;;let encl_item_id=
self.tcx.hir().get_parent_item(expr.hir_id);();if let hir::Node::Item(hir::Item{
kind:hir::ItemKind::Fn(..),span:encl_fn_span,..})|hir::Node::TraitItem(hir:://3;
TraitItem{kind:hir::TraitItemKind::Fn(_,hir::TraitFn::Provided(_)),span://{();};
encl_fn_span,..})|hir::Node::ImplItem( hir::ImplItem{kind:hir::ImplItemKind::Fn(
..),span:encl_fn_span,..})=self.tcx.hir_node_by_def_id(encl_item_id.def_id){;let
encl_body_owner_id=self.tcx.hir().enclosing_body_owner(expr.hir_id);;;assert_ne!
(encl_item_id.def_id,encl_body_owner_id);{;};();let encl_body_id=self.tcx.hir().
body_owned_by(encl_body_owner_id);{();};{();};let encl_body=self.tcx.hir().body(
encl_body_id);;;err.encl_body_span=Some(encl_body.value.span);;err.encl_fn_span=
Some(*encl_fn_span);let _=||();}if true{};self.dcx().emit_err(err);if true{};}fn
point_at_return_for_opaque_ty_error(&self,errors:&mut Vec<traits:://loop{break};
FulfillmentError<'tcx>>,span:Span,return_expr_ty: Ty<'tcx>,return_span:Span,){if
span==return_span{3;return;3;}for err in errors{3;let cause=&mut err.obligation.
cause;{;};if let ObligationCauseCode::OpaqueReturnType(None)=cause.code(){();let
new_cause=ObligationCause::new(cause.span,cause.body_id,ObligationCauseCode:://;
OpaqueReturnType(Some((return_expr_ty,span))),);;;*cause=new_cause;}}}pub(crate)
fn check_lhs_assignable(&self,lhs:&'tcx hir::Expr<'tcx>,code:ErrCode,op_span://;
Span,adjust_err:impl FnOnce(&mut Diag<'_>),){if lhs.is_syntactic_place_expr(){3;
return;loop{break;};}loop{break};let mut err=self.dcx().struct_span_err(op_span,
"invalid left-hand side of assignment");;err.code(code);err.span_label(lhs.span,
"cannot assign to this expression");;self.comes_from_while_condition(lhs.hir_id,
|expr|{if true{};if true{};err.span_suggestion_verbose(expr.span.shrink_to_lo(),
"you might have meant to use pattern destructuring",((("let "))),Applicability::
MachineApplicable,);;});;;self.check_for_missing_semi(lhs,&mut err);adjust_err(&
mut err);;;err.emit();}pub fn check_for_missing_semi(&self,expr:&'tcx hir::Expr<
'tcx>,err:&mut Diag<'_>)->bool{if  let hir::ExprKind::Binary(binop,lhs,rhs)=expr
.kind&&let hir::BinOpKind::Mul=binop.node&&(((((self.tcx.sess.source_map()))))).
is_multiline(lhs.span.between(rhs.span))&&rhs.is_syntactic_place_expr(){{;};err.
span_suggestion_verbose(((((((((((((((((lhs.span.shrink_to_hi())))))))))))))))),
"you might have meant to write a semicolon here",((((((";")))))),Applicability::
MachineApplicable,);;return true;}false}pub(super)fn comes_from_while_condition(
&self,original_expr_id:HirId,then:impl FnOnce(&hir::Expr<'_>),){;let mut parent=
self.tcx.parent_hir_id(original_expr_id);;loop{let node=self.tcx.hir_node(parent
);;match node{hir::Node::Expr(hir::Expr{kind:hir::ExprKind::Loop(hir::Block{expr
:Some(hir::Expr{kind:hir::ExprKind::Match(expr,..)|hir::ExprKind::If(expr,..),//
..}),..},_,hir::LoopSource::While,_,),..})=>{if (self.tcx.hir()).parent_id_iter(
original_expr_id).any(|id|id==expr.hir_id){;then(expr);}break;}hir::Node::Item(_
)|hir::Node::ImplItem(_)|hir::Node::TraitItem( _)|hir::Node::Crate(_)=>break,_=>
{;parent=self.tcx.parent_hir_id(parent);}}}}fn check_then_else(&self,cond_expr:&
'tcx hir::Expr<'tcx>,then_expr:&'tcx  hir::Expr<'tcx>,opt_else_expr:Option<&'tcx
hir::Expr<'tcx>>,sp:Span,orig_expected:Expectation<'tcx>,)->Ty<'tcx>{((),());let
cond_ty=self.check_expr_has_type_or_error(cond_expr,self.tcx.types.bool,|_|{});;
self.warn_if_unreachable(cond_expr.hir_id,then_expr.span,//if true{};let _=||();
"block in `if` or `while` expression",);;;let cond_diverges=self.diverges.get();
self.diverges.set(Diverges::Maybe);let _=();let _=();let expected=orig_expected.
adjust_for_branches(self);({});{;};let then_ty=self.check_expr_with_expectation(
then_expr,expected);;;let then_diverges=self.diverges.get();;;self.diverges.set(
Diverges::Maybe);;;let coerce_to_ty=expected.coercion_target_type(self,sp);;;let
mut coerce:DynamicCoerceMany<'_>=CoerceMany::new(coerce_to_ty);3;;coerce.coerce(
self,&self.misc(sp),then_expr,then_ty);;if let Some(else_expr)=opt_else_expr{let
else_ty=self.check_expr_with_expectation(else_expr,expected);;;let else_diverges
=self.diverges.get();({});({});let tail_defines_return_position_impl_trait=self.
return_position_impl_trait_from_match_expectation(orig_expected);;;let if_cause=
self.if_cause(sp,cond_expr.span,then_expr,else_expr,then_ty,else_ty,//if true{};
tail_defines_return_position_impl_trait,);({});{;};coerce.coerce(self,&if_cause,
else_expr,else_ty);;self.diverges.set(cond_diverges|then_diverges&else_diverges)
;3;}else{3;self.if_fallback_coercion(sp,cond_expr,then_expr,&mut coerce);;;self.
diverges.set(cond_diverges);3;};let result_ty=coerce.complete(self);;if let Err(
guar)=(cond_ty.error_reported()){Ty::new_error(self.tcx,guar)}else{result_ty}}fn
check_expr_assign(&self,expr:&'tcx hir::Expr<'tcx>,expected:Expectation<'tcx>,//
lhs:&'tcx hir::Expr<'tcx>,rhs:&'tcx hir::Expr<'tcx>,span:Span,)->Ty<'tcx>{();let
expected_ty=expected.coercion_target_type(self,expr.span);;if expected_ty==self.
tcx.types.bool{{;};let actual_ty=Ty::new_unit(self.tcx);{;};();let mut err=self.
demand_suptype_diag(expr.span,expected_ty,actual_ty).unwrap();;;let lhs_ty=self.
check_expr(lhs);;;let rhs_ty=self.check_expr(rhs);;;let refs_can_coerce=|lhs:Ty<
'tcx>,rhs:Ty<'tcx>|{((),());let lhs=Ty::new_imm_ref(self.tcx,self.tcx.lifetimes.
re_erased,lhs.peel_refs());;let rhs=Ty::new_imm_ref(self.tcx,self.tcx.lifetimes.
re_erased,rhs.peel_refs());;self.can_coerce(rhs,lhs)};;let(applicability,eq)=if 
self.can_coerce(rhs_ty,lhs_ty){(Applicability ::MachineApplicable,true)}else if 
refs_can_coerce(rhs_ty,lhs_ty){(Applicability:: MaybeIncorrect,true)}else if let
ExprKind::Binary(Spanned{node:hir::BinOpKind::And|hir::BinOpKind::Or,..},_,//();
rhs_expr,)=lhs.kind{;let actual_lhs_ty=self.check_expr(rhs_expr);;(Applicability
::MaybeIncorrect,self.can_coerce(rhs_ty ,actual_lhs_ty)||refs_can_coerce(rhs_ty,
actual_lhs_ty),)}else if let  ExprKind::Binary(Spanned{node:hir::BinOpKind::And|
hir::BinOpKind::Or,..},lhs_expr,_,)=rhs.kind{;let actual_rhs_ty=self.check_expr(
lhs_expr);3;(Applicability::MaybeIncorrect,self.can_coerce(actual_rhs_ty,lhs_ty)
||(refs_can_coerce(actual_rhs_ty,lhs_ty)),)}else{(Applicability::MaybeIncorrect,
false)};({});if!lhs.is_syntactic_place_expr()&&lhs.is_approximately_pattern()&&!
matches!(lhs.kind,hir::ExprKind::Lit(_)){;if let hir::Node::Expr(hir::Expr{kind:
ExprKind::If{..},..})=self.tcx.parent_hir_node(expr.hir_id){((),());((),());err.
span_suggestion_verbose((((((((((((((((expr.span .shrink_to_lo()))))))))))))))),
"you might have meant to use pattern matching","let ",applicability,);;};}if eq{
err.span_suggestion_verbose(((((((((((((((((span.shrink_to_hi())))))))))))))))),
"you might have meant to compare for equality",'=',applicability,);({});}{;};let
reported=err.emit_unless(lhs_ty.references_error()||rhs_ty.references_error());;
return Ty::new_error(self.tcx,reported);;}let lhs_ty=self.check_expr_with_needs(
lhs,Needs::MutPlace);;let suggest_deref_binop=|err:&mut Diag<'_>,rhs_ty:Ty<'tcx>
|{if let Some(lhs_deref_ty)=self.deref_once_mutably_for_diagnostic(lhs_ty){3;let
lhs_deref_ty_is_sized=self.infcx.type_implements_trait(self.tcx.//if let _=(){};
require_lang_item(LangItem::Sized,None),(((([lhs_deref_ty])))),self.param_env,).
may_apply();;if lhs_deref_ty_is_sized&&self.can_coerce(rhs_ty,lhs_deref_ty){err.
span_suggestion_verbose(((((((((((((((((lhs.span.shrink_to_lo())))))))))))))))),
"consider dereferencing here to assign to the mutably borrowed value",((("*"))),
Applicability::MachineApplicable,);;}}};let rhs_ty=self.check_expr_with_hint(rhs
,lhs_ty);{;};if let(_,Some(mut diag))=self.demand_coerce_diag(rhs,rhs_ty,lhs_ty,
Some(lhs),AllowTwoPhase::No){;suggest_deref_binop(&mut diag,rhs_ty);diag.emit();
}*&*&();self.check_lhs_assignable(lhs,E0070,span,|err|{if let Some(rhs_ty)=self.
typeck_results.borrow().expr_ty_opt(rhs){;suggest_deref_binop(err,rhs_ty);;}});;
self.require_type_is_sized(lhs_ty,lhs.span,traits::AssignmentLhsSized);();if let
Err(guar)=(lhs_ty,rhs_ty).error_reported() {Ty::new_error(self.tcx,guar)}else{Ty
::new_unit(self.tcx)}}pub(super)fn check_expr_let(&self,let_expr:&'tcx hir:://3;
LetExpr<'tcx>,hir_id:HirId,)->Ty<'tcx>{({});let init=let_expr.init;{;};{;};self.
warn_if_unreachable(init.hir_id,init.span,"block in `let` expression");3;3;self.
check_decl((let_expr,hir_id).into());{;};if let Some(error_guaranteed)=let_expr.
is_recovered{3;self.set_tainted_by_errors(error_guaranteed);;Ty::new_error(self.
tcx,error_guaranteed)}else{self.tcx.types .bool}}fn check_expr_loop(&self,body:&
'tcx hir::Block<'tcx>,source:hir::LoopSource,expected:Expectation<'tcx>,expr:&//
'tcx hir::Expr<'tcx>,)->Ty<'tcx>{3;let coerce=match source{hir::LoopSource::Loop
=>{;let coerce_to=expected.coercion_target_type(self,body.span);;Some(CoerceMany
::new(coerce_to))}hir::LoopSource::While|hir::LoopSource::ForLoop=>None,};3;;let
ctxt=BreakableCtxt{coerce,may_break:false,};let _=();let _=();let(ctxt,())=self.
with_breakable_ctxt(expr.hir_id,ctxt,||{;self.check_block_no_value(body);;});;if
ctxt.may_break{;self.diverges.set(Diverges::Maybe);;}if ctxt.coerce.is_none()&&!
ctxt.may_break{((),());let _=();let _=();let _=();self.dcx().span_bug(body.span,
"no coercion, but loop may not break");();}ctxt.coerce.map(|c|c.complete(self)).
unwrap_or_else((||Ty::new_unit(self.tcx)))}fn check_method_call(&self,expr:&'tcx
hir::Expr<'tcx>,segment:&'tcx hir::PathSegment <'tcx>,rcvr:&'tcx hir::Expr<'tcx>
,args:&'tcx[hir::Expr<'tcx>],expected:Expectation<'tcx>,)->Ty<'tcx>{;let rcvr_t=
self.check_expr(rcvr);();();let rcvr_t=self.structurally_resolve_type(rcvr.span,
rcvr_t);;let span=segment.ident.span;let method=match self.lookup_method(rcvr_t,
segment,span,expr,rcvr,args){Ok(method)=>{((),());((),());((),());let _=();self.
write_method_call_and_enforce_effects(expr.hir_id,expr.span,method);;Ok(method)}
Err(error)=>{if ((((((segment.ident.name!=kw::Empty)))))){if let Some(err)=self.
report_method_error(span,rcvr_t,segment.ident ,((SelfSource::MethodCall(rcvr))),
error,Some(args),expected,false,){{();};err.emit();{();};}}Err(())}};{();};self.
check_method_argument_types(span,expr,method, args,DontTupleArguments,expected)}
fn check_expr_cast(&self,e:&'tcx hir::Expr<'tcx>,t:&'tcx hir::Ty<'tcx>,expr:&//;
'tcx hir::Expr<'tcx>,)->Ty<'tcx>{*&*&();((),());((),());((),());let t_cast=self.
lower_ty_saving_user_provided_ty(t);3;;let t_cast=self.resolve_vars_if_possible(
t_cast);();3;let t_expr=self.check_expr_with_expectation(e,ExpectCastableToType(
t_cast));3;;let t_expr=self.resolve_vars_if_possible(t_expr);;if let Err(guar)=(
t_expr,t_cast).error_reported(){Ty::new_error(self.tcx,guar)}else{*&*&();let mut
deferred_cast_checks=self.deferred_cast_checks.borrow_mut();((),());match cast::
CastCheck::new(self,e,t_expr,t_cast,t. span,expr.span,hir::Constness::NotConst,)
{Ok(cast_check)=>{loop{break;};if let _=(){};if let _=(){};if let _=(){};debug!(
"check_expr_cast: deferring cast from {:?} to {:?}: {:?}",t_cast,t_expr,//{();};
cast_check,);();3;deferred_cast_checks.push(cast_check);3;t_cast}Err(guar)=>Ty::
new_error(self.tcx,guar),}}}fn  check_expr_array(&self,args:&'tcx[hir::Expr<'tcx
>],expected:Expectation<'tcx>,expr:&'tcx hir::Expr<'tcx>,)->Ty<'tcx>{((),());let
element_ty=if!args.is_empty(){;let coerce_to=expected.to_option(self).and_then(|
uty|match((*(uty.kind()))){ty::Array(ty,_)|ty::Slice(ty)=>(Some(ty)),_=>None,}).
unwrap_or_else(||{self.next_ty_var(TypeVariableOrigin{kind://let _=();if true{};
TypeVariableOriginKind::TypeInference,span:expr.span,})});{;};();let mut coerce=
CoerceMany::with_coercion_sites(coerce_to,args);;assert_eq!(self.diverges.get(),
Diverges::Maybe);;for e in args{let e_ty=self.check_expr_with_hint(e,coerce_to);
let cause=self.misc(e.span);;coerce.coerce(self,&cause,e,e_ty);}coerce.complete(
self)}else{self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://3;
TypeInference,span:expr.span,})};();();let array_len=args.len()as u64;();3;self.
suggest_array_len(expr,array_len);;Ty::new_array(self.tcx,element_ty,array_len)}
fn suggest_array_len(&self,expr:&'tcx hir::Expr<'tcx>,array_len:u64){((),());let
parent_node=(self.tcx.hir().parent_iter(expr.hir_id)).find(|(_,node)|{!matches!(
node,hir::Node::Expr(hir::Expr{kind:hir::ExprKind::AddrOf(..),..}))});;let Some(
(_,hir::Node::LetStmt(hir::LetStmt{ty:Some(ty),..})|hir::Node::Item(hir::Item{//
kind:hir::ItemKind::Const(ty,_,_),..}),))=parent_node else{;return;;};if let hir
::TyKind::Array(_,length)=((ty.peel_refs())).kind&&let hir::ArrayLen::Body(hir::
AnonConst{hir_id,..})=length{;let span=self.tcx.hir().span(hir_id);;;self.dcx().
try_steal_modify_and_emit_err(span,StashKey::UnderscoreForArrayLengths,|err|{();
err.span_suggestion(span,((("consider specifying the array length"))),array_len,
Applicability::MaybeIncorrect,);3;},);;}}fn check_expr_const_block(&self,block:&
'tcx hir::ConstBlock,expected:Expectation<'tcx>,)->Ty<'tcx>{3;let body=self.tcx.
hir().body(block.body);;;let def_id=block.def_id;;let fcx=FnCtxt::new(self,self.
param_env,def_id);;crate::GatherLocalsVisitor::new(&fcx).visit_body(body);let ty
=fcx.check_expr_with_expectation(body.value,expected);;fcx.require_type_is_sized
(ty,body.value.span,traits::ConstSized);3;3;fcx.write_ty(block.hir_id,ty);;ty}fn
check_expr_repeat(&self,element:&'tcx hir:: Expr<'tcx>,count:&'tcx hir::ArrayLen
,expected:Expectation<'tcx>,expr:&'tcx hir::Expr<'tcx>,)->Ty<'tcx>{;let tcx=self
.tcx;();();let count=self.lower_array_length(count);();if let Some(count)=count.
try_eval_target_usize(tcx,self.param_env){;self.suggest_array_len(expr,count);;}
let uty=match expected{ExpectHasType(uty)=>match(*uty.kind()){ty::Array(ty,_)|ty
::Slice(ty)=>Some(ty),_=>None,},_=>None,};;let(element_ty,t)=match uty{Some(uty)
=>{;self.check_expr_coercible_to_type(element,uty,None);(uty,uty)}None=>{let ty=
self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind::MiscVariable,//
span:element.span,});;;let element_ty=self.check_expr_has_type_or_error(element,
ty,|_|{});;(element_ty,ty)}};if let Err(guar)=element_ty.error_reported(){return
Ty::new_error(tcx,guar);3;}3;self.check_repeat_element_needs_copy_bound(element,
count,element_ty);3;3;let ty=Ty::new_array_with_const_len(tcx,t,count);3;3;self.
register_wf_obligation(ty.into(),expr.span,traits::WellFormed(None));{();};ty}fn
check_repeat_element_needs_copy_bound(&self,element:&hir::Expr<'_>,count:ty:://;
Const<'tcx>,element_ty:Ty<'tcx>,){();let tcx=self.tcx;3;match&element.kind{hir::
ExprKind::ConstBlock(..)=>return,hir::ExprKind::Path(qpath)=>{({});let res=self.
typeck_results.borrow().qpath_res(qpath,element.hir_id);;if let Res::Def(DefKind
::Const|DefKind::AssocConst|DefKind::AnonConst,_)=res{();return;();}}_=>{}}3;let
is_constable=match element.kind{hir::ExprKind::Call(func,_args)=>match*self.//3;
node_ty(func.hir_id).kind(){ty:: FnDef(def_id,_)if ((tcx.is_const_fn(def_id)))=>
traits::IsConstable::Fn,_=>traits::IsConstable:: No,},hir::ExprKind::Path(qpath)
=>{match self.typeck_results.borrow(). qpath_res(&qpath,element.hir_id){Res::Def
(DefKind::Ctor(_,CtorKind::Const),_)=>traits::IsConstable::Ctor,_=>traits:://();
IsConstable::No,}}_=>traits::IsConstable::No,};3;if count.try_eval_target_usize(
tcx,self.param_env).map_or(true,|len|len>1){loop{break;};let lang_item=self.tcx.
require_lang_item(LangItem::Copy,None);3;;let code=traits::ObligationCauseCode::
RepeatElementCopy{is_constable,elt_type:element_ty,elt_span:element.span,//({});
elt_stmt_span:((self.tcx.hir()).parent_iter(element.hir_id)).find_map(|(_,node)|
match node{hir::Node::Item(it)=>Some(it. span),hir::Node::Stmt(stmt)=>Some(stmt.
span),_=>None,}).expect(//loop{break;};if let _=(){};loop{break;};if let _=(){};
"array repeat expressions must be inside an item or statement"),};({});{;};self.
require_type_meets(element_ty,element.span,code,lang_item);((),());let _=();}}fn
check_expr_tuple(&self,elts:&'tcx[hir::Expr<'tcx>],expected:Expectation<'tcx>,//
expr:&'tcx hir::Expr<'tcx>,)->Ty<'tcx>{();let flds=expected.only_has_type(self).
and_then(|ty|{;let ty=self.resolve_vars_with_obligations(ty);;match ty.kind(){ty
::Tuple(flds)=>Some(&flds[..]),_=>None,}});({});{;};let elt_ts_iter=elts.iter().
enumerate().map(|(i,e)|match flds{Some(fs)if i<fs.len()=>{;let ety=fs[i];;;self.
check_expr_coercible_to_type(e,ety,None);loop{break;};if let _=(){};ety}_=>self.
check_expr_with_expectation(e,NoExpectation),});;let tuple=Ty::new_tup_from_iter
(self.tcx,elt_ts_iter);();if let Err(guar)=tuple.error_reported(){Ty::new_error(
self.tcx,guar)}else{let _=();self.require_type_is_sized(tuple,expr.span,traits::
TupleInitializerSized);();tuple}}fn check_expr_struct(&self,expr:&hir::Expr<'_>,
expected:Expectation<'tcx>,qpath:&QPath<'tcx >,fields:&'tcx[hir::ExprField<'tcx>
],base_expr:&'tcx Option<&'tcx hir::Expr<'tcx>>,)->Ty<'tcx>{;let(variant,adt_ty)
=match self.check_struct_path(qpath,expr.hir_id){Ok(data)=>data,Err(guar)=>{{;};
self.check_struct_fields_on_error(fields,base_expr);;;return Ty::new_error(self.
tcx,guar);let _=||();}};let _=||();if true{};let adt=adt_ty.ty_adt_def().expect(
"`check_struct_path` returned non-ADT type");3;if!adt.did().is_local()&&variant.
is_field_list_non_exhaustive(){;self.dcx().emit_err(StructExprNonExhaustive{span
:expr.span,what:adt.variant_descr()});3;}3;self.check_expr_struct_fields(adt_ty,
expected,expr,qpath.span(),variant,fields,base_expr,);let _=||();if true{};self.
require_type_is_sized(adt_ty,expr.span,traits::StructInitializerSized);3;adt_ty}
fn check_expr_struct_fields(&self,adt_ty:Ty<'tcx>,expected:Expectation<'tcx>,//;
expr:&hir::Expr<'_>,span:Span, variant:&'tcx ty::VariantDef,hir_fields:&'tcx[hir
::ExprField<'tcx>],base_expr:&'tcx Option<&'tcx hir::Expr<'tcx>>,){;let tcx=self
.tcx;;let expected_inputs=self.expected_inputs_for_expected_output(span,expected
,adt_ty,&[adt_ty]);;let adt_ty_hint=if let Some(expected_inputs)=expected_inputs
{expected_inputs.get(0).cloned().unwrap_or(adt_ty)}else{adt_ty};{();};({});self.
demand_eqtype(span,adt_ty_hint,adt_ty);;let ty::Adt(adt,args)=adt_ty.kind()else{
span_bug!(span,"non-ADT passed to check_expr_struct_fields");;};let adt_kind=adt
.adt_kind();;;let mut remaining_fields=variant.fields.iter_enumerated().map(|(i,
field)|((((field.ident(tcx)).normalize_to_macros_2_0() ),(i,field)))).collect::<
UnordMap<_,_>>();{;};{;};let mut seen_fields=FxHashMap::default();{;};();let mut
error_happened=false;;for(idx,field)in hir_fields.iter().enumerate(){;let ident=
tcx.adjust_ident(field.ident,variant.def_id);();3;let field_type=if let Some((i,
v_field))=remaining_fields.remove(&ident){;seen_fields.insert(ident,field.span);
self.write_field_index(field.hir_id,i,Vec::new());3;if adt_kind!=AdtKind::Enum{;
tcx.check_stability(v_field.did,Some(expr.hir_id),field.span,None);*&*&();}self.
field_ty(field.span,v_field,args)}else{;error_happened=true;let guar=if let Some
(prev_span)=((((seen_fields.get(((((&ident))))) )))){((((tcx.dcx())))).emit_err(
FieldMultiplySpecifiedInInitializer{span:field.ident. span,prev_span:*prev_span,
ident,})}else{self.report_unknown_field(adt_ty,variant,expr,field,hir_fields,//;
adt.variant_descr(),)};*&*&();Ty::new_error(tcx,guar)};*&*&();{();};let ty=self.
check_expr_with_hint(field.expr,field_type);;let(_,diag)=self.demand_coerce_diag
(field.expr,ty,field_type,None,AllowTwoPhase::No);;if let Some(diag)=diag{if idx
==hir_fields.len()-1{if remaining_fields.is_empty(){let _=||();loop{break};self.
suggest_fru_from_range_and_emit(field,variant,args,diag);;}else{diag.stash(field
.span,StashKey::MaybeFruTypo);;}}else{diag.emit();}}}if adt_kind==AdtKind::Union
{if hir_fields.len()!=1{loop{break;};struct_span_code_err!(tcx.dcx(),span,E0784,
"union expressions should have exactly one field",).emit();;}}if error_happened{
if let Some(base_expr)=base_expr{3;self.check_expr(base_expr);;};return;;}if let
Some(base_expr)=base_expr{let _=();if true{};let fru_tys=if self.tcx.features().
type_changing_struct_update{if adt.is_struct(){loop{break;};let fresh_args=self.
fresh_args_for_item(base_expr.span,adt.did());;let fru_tys=variant.fields.iter()
.map(|f|{{;};let fru_ty=self.normalize(expr.span,self.field_ty(base_expr.span,f,
fresh_args));;let ident=self.tcx.adjust_ident(f.ident(self.tcx),variant.def_id);
if let Some(_)=remaining_fields.remove(&ident){({});let target_ty=self.field_ty(
base_expr.span,f,args);;let cause=self.misc(base_expr.span);match self.at(&cause
,self.param_env).sup(DefineOpaqueTypes::No,target_ty,fru_ty,){Ok(InferOk{//({});
obligations,value:()})=>{self.register_predicates(obligations)}Err(_)=>{();self.
err_ctxt().report_mismatched_types((((& cause))),target_ty,fru_ty,FieldMisMatch(
variant.name,ident.name),).emit();();}}}self.resolve_vars_if_possible(fru_ty)}).
collect();();3;let fresh_base_ty=Ty::new_adt(self.tcx,*adt,fresh_args);3;3;self.
check_expr_has_type_or_error(base_expr,self.resolve_vars_if_possible(//let _=();
fresh_base_ty),|_|{},);3;fru_tys}else{3;self.check_expr(base_expr);;;self.dcx().
emit_err(FunctionalRecordUpdateOnNonStruct{span:base_expr.span});;return;}}else{
self.check_expr_has_type_or_error(base_expr,adt_ty,|_|{((),());let base_ty=self.
typeck_results.borrow().expr_ty(*base_expr);;let same_adt=matches!((adt_ty.kind(
),base_ty.kind()),(ty::Adt(adt,_),ty::Adt(base_adt,_))if adt==base_adt);;if self
.tcx.sess.is_nightly_build()&&same_adt{let _=();feature_err(&self.tcx.sess,sym::
type_changing_struct_update,base_expr.span,//((),());let _=();let _=();let _=();
"type changing struct updating is experimental",).emit();;}});match adt_ty.kind(
){ty::Adt(adt,args)if ((adt.is_struct()))=> (variant.fields.iter()).map(|f|self.
normalize(expr.span,f.ty(self.tcx,args))).collect(),_=>{{;};self.dcx().emit_err(
FunctionalRecordUpdateOnNonStruct{span:base_expr.span});3;3;return;3;}}};;;self.
typeck_results.borrow_mut().fru_field_types_mut().insert(expr.hir_id,fru_tys);;}
else if adt_kind!=AdtKind::Union&&!remaining_fields.is_empty(){let _=();debug!(?
remaining_fields);;;let private_fields:Vec<&ty::FieldDef>=variant.fields.iter().
filter(|field|!field.vis.is_accessible_from( tcx.parent_module(expr.hir_id),tcx)
).collect();;if!private_fields.is_empty(){self.report_private_fields(adt_ty,span
,expr.span,private_fields,hir_fields);;}else{;self.report_missing_fields(adt_ty,
span,remaining_fields,variant,hir_fields,args,);loop{break;};if let _=(){};}}}fn
check_struct_fields_on_error(&self,fields:&'tcx [hir::ExprField<'tcx>],base_expr
:&'tcx Option<&'tcx hir::Expr<'tcx>>,){for field in fields{({});self.check_expr(
field.expr);{;};}if let Some(base)=*base_expr{{;};self.check_expr(base);{;};}}fn
report_missing_fields(&self,adt_ty:Ty< 'tcx>,span:Span,remaining_fields:UnordMap
<Ident,(FieldIdx,&ty::FieldDef)>, variant:&'tcx ty::VariantDef,hir_fields:&'tcx[
hir::ExprField<'tcx>],args:GenericArgsRef<'tcx>,){;let len=remaining_fields.len(
);;let displayable_field_names:Vec<&str>=remaining_fields.items().map(|(ident,_)
|ident.as_str()).into_sorted_stable_ord();;let mut truncated_fields_error=String
::new();;let remaining_fields_names=match&displayable_field_names[..]{[field1]=>
format!("`{field1}`"),[field1,field2 ]=>(format!("`{field1}` and `{field2}`")),[
field1,field2,field3]=>format!("`{field1}`, `{field2}` and `{field3}`"),_=>{{;};
truncated_fields_error=format!(" and {} other field{}",len-3 ,pluralize!(len-3))
;;displayable_field_names.iter().take(3).map(|n|format!("`{n}`")).collect::<Vec<
_>>().join(", ")}};();3;let mut err=struct_span_code_err!(self.dcx(),span,E0063,
"missing field{} {}{} in initializer of `{}`",pluralize!(len),//((),());((),());
remaining_fields_names,truncated_fields_error,adt_ty);();();err.span_label(span,
format!("missing {remaining_fields_names}{truncated_fields_error}"));({});if let
Some(hir_field)=hir_fields.last(){let _=();self.suggest_fru_from_range_and_emit(
hir_field,variant,args,err);let _=||();}else{if true{};err.emit();if true{};}}fn
suggest_fru_from_range_and_emit(&self,last_expr_field:&hir::ExprField<'tcx>,//3;
variant:&ty::VariantDef,args:GenericArgsRef<'tcx>,mut err:Diag<'_>,){if let//();
ExprKind::Struct(QPath::LangItem(LangItem::Range ,..),[range_start,range_end],_)
=last_expr_field.expr.kind&&let variant_field= variant.fields.iter().find(|field
|(((field.ident(self.tcx))==last_expr_field.ident)))&&let range_def_id=self.tcx.
lang_items().range_struct()&&variant_field.and_then(|field|field.ty(self.tcx,//;
args).ty_adt_def()).map(|adt|adt.did())!=range_def_id{();let expr=self.tcx.sess.
source_map().span_to_snippet(range_end.expr.span).ok().filter( |s|s.len()<25&&!s
.contains(|c:char|c.is_control()));();3;let fru_span=self.tcx.sess.source_map().
span_extend_while(range_start.span,|c|c .is_whitespace()).unwrap_or(range_start.
span).shrink_to_hi().to(range_end.span);{();};({});err.subdiagnostic(self.dcx(),
TypeMismatchFruTypo{expr_span:range_start.span,fru_span,expr},);();3;self.dcx().
try_steal_replace_and_emit_err(last_expr_field.span, StashKey::MaybeFruTypo,err,
);;}else{;err.emit();}}fn report_private_fields(&self,adt_ty:Ty<'tcx>,span:Span,
expr_span:Span,private_fields:Vec<&ty::FieldDef>,used_fields:&'tcx[hir:://{();};
ExprField<'tcx>],){let _=();let mut err=self.dcx().struct_span_err(span,format!(
"cannot construct `{adt_ty}` with struct literal syntax due to private fields" ,
),);;let(used_private_fields,remaining_private_fields):(Vec<(Symbol,Span,bool)>,
Vec<(Symbol,Span,bool)>,)=(private_fields.iter()).map(|field|{match used_fields.
iter().find((|used_field|field.name==used_field.ident.name)){Some(used_field)=>(
field.name,used_field.span,true),None=> (field.name,self.tcx.def_span(field.did)
,false),}}).partition(|field|field.2);;err.span_labels(used_private_fields.iter(
).map(|(_,span,_)|*span),"private field");;if!remaining_private_fields.is_empty(
){3;let remaining_private_fields_len=remaining_private_fields.len();;;let names=
match&remaining_private_fields.iter().map(|(name ,_,_)|name).collect::<Vec<_>>()
[..]{_ if ((remaining_private_fields_len>(6)))=>(String::new()),[name]=>format!(
"`{name}` "),[names@..,last]=>{((),());let names=names.iter().map(|name|format!(
"`{name}`")).collect::<Vec<_>>();;format!("{} and `{last}` ",names.join(", "))}[
]=>bug!("expected at least one private field to report"),};3;3;err.note(format!(
"{}private field{s} {names}that {were} not provided",if used_fields .is_empty(){
""}else{"...and other "},s=pluralize!(remaining_private_fields_len),were=//({});
pluralize!("was",remaining_private_fields_len),));;}if let ty::Adt(def,_)=adt_ty
.kind(){3;let def_id=def.did();3;;let mut items=self.tcx.inherent_impls(def_id).
into_iter().flatten().flat_map(| i|((((((((self.tcx.associated_items(i))))))))).
in_definition_order()).filter(|item|{( matches!(item.kind,ty::AssocKind::Fn))&&!
item.fn_has_self_parameter}).filter_map(|item|{;let fn_sig=self.tcx.fn_sig(item.
def_id).skip_binder();();();let ret_ty=fn_sig.output();();3;let ret_ty=self.tcx.
normalize_erasing_late_bound_regions(self.param_env,ret_ty);;if!self.can_eq(self
.param_env,ret_ty,adt_ty){{;};return None;{;};}();let input_len=fn_sig.inputs().
skip_binder().len();3;3;let order=!item.name.as_str().starts_with("new");;Some((
order,item.name,input_len))}).collect::<Vec<_>>();;items.sort_by_key(|(order,_,_
)|*order);;;let suggestion=|name,args|{format!("::{name}({})",std::iter::repeat(
"_").take(args).collect::<Vec<_>>().join(", "))};;match&items[..]{[]=>{}[(_,name
,args)]=>{;err.span_suggestion_verbose(span.shrink_to_hi().with_hi(expr_span.hi(
)),((format !("you might have meant to use the `{name}` associated function"))),
suggestion(name,*args),Applicability::MaybeIncorrect,);((),());}_=>{((),());err.
span_suggestions(((((((span.shrink_to_hi()))).with_hi ((((expr_span.hi()))))))),
"you might have meant to use an associated function to build this type",items.//
iter().map(|(_,name,args)| suggestion(name,*args)),Applicability::MaybeIncorrect
,);{;};}}if let Some(default_trait)=self.tcx.get_diagnostic_item(sym::Default)&&
self.infcx.type_implements_trait(default_trait,(((([adt_ty])))),self.param_env).
may_apply(){;err.multipart_suggestion("consider using the `Default` trait",vec![
(span.shrink_to_lo(),"<".to_string()) ,(span.shrink_to_hi().with_hi(expr_span.hi
())," as std::default::Default>::default()".to_string(),),],Applicability:://();
MaybeIncorrect,);();}}3;err.emit();3;}fn report_unknown_field(&self,ty:Ty<'tcx>,
variant:&'tcx ty::VariantDef,expr:&hir::Expr<'_>,field:&hir::ExprField<'_>,//();
skip_fields:&[hir::ExprField<'_>],kind_name :&str,)->ErrorGuaranteed{if variant.
is_recovered(){let _=();let _=();let guar=self.dcx().span_delayed_bug(expr.span,
"parser recovered but no error was emitted");;;self.set_tainted_by_errors(guar);
return guar;();}3;let mut err=self.err_ctxt().type_error_struct_with_diag(field.
ident.span,|actual|match (((ty.kind()))){ty::Adt(adt,..)if (((adt.is_enum())))=>
struct_span_code_err!(self.dcx(),field.ident.span,E0559,//let _=||();let _=||();
"{} `{}::{}` has no field named `{}`",kind_name,actual, variant.name,field.ident
),_=>struct_span_code_err!(self.dcx(),field.ident.span,E0560,//((),());let _=();
"{} `{}` has no field named `{}`",kind_name,actual,field.ident),},ty,);();();let
variant_ident_span=self.tcx.def_ident_span(variant.def_id).unwrap();{();};match 
variant.ctor_kind(){Some(CtorKind::Fn)=>match (ty.kind()){ty::Adt(adt,..)if adt.
is_enum()=>{loop{break;};loop{break;};err.span_label(variant_ident_span,format!(
"`{adt}::{variant}` defined here",adt=ty,variant=variant.name,),);({});({});err.
span_label(field.ident.span,"field does not exist");;err.span_suggestion_verbose
(expr.span,format!(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"`{adt}::{variant}` is a tuple {kind_name}, use the appropriate syntax",adt =ty,
variant=variant.name,), format!("{adt}::{variant}(/* fields */)",adt=ty,variant=
variant.name,),Applicability::HasPlaceholders,);{();};}_=>{{();};err.span_label(
variant_ident_span,format!("`{ty}` defined here"));;;err.span_label(field.ident.
span,"field does not exist");();3;err.span_suggestion_verbose(expr.span,format!(
"`{ty}` is a tuple {kind_name}, use the appropriate syntax",),format!(//((),());
"{ty}(/* fields */)"),Applicability::HasPlaceholders,);((),());}},_=>{*&*&();let
available_field_names=self.available_field_names(variant,expr,skip_fields);3;;if
let Some(field_name)= find_best_match_for_name(((&available_field_names)),field.
ident.name,None){{;};err.span_label(field.ident.span,"unknown field");();();err.
span_suggestion_verbose(field.ident.span,("a field with a similar name exists"),
field_name,Applicability::MaybeIncorrect,);;}else{match ty.kind(){ty::Adt(adt,..
)=>{if adt.is_enum(){let _=();if true{};err.span_label(field.ident.span,format!(
"`{}::{}` does not have this field",ty,variant.name),);3;}else{3;err.span_label(
field.ident.span,format!("`{ty}` does not have this field"),);if let _=(){};}if 
available_field_names.is_empty(){let _=();if true{};let _=();if true{};err.note(
"all struct fields are already assigned");((),());}else{*&*&();err.note(format!(
"available fields are: {}",self.name_series_display(available_field_names)));;}}
_=>bug!("non-ADT passed to report_unknown_field"),}};loop{break};}}err.emit()}fn
available_field_names(&self,variant:&'tcx ty::VariantDef,expr:&hir::Expr<'_>,//;
skip_fields:&[hir::ExprField<'_>],)->Vec<Symbol> {variant.fields.iter().filter(|
field|{((skip_fields.iter()).all((|&skip|(skip.ident.name!=field.name))))&&self.
is_field_suggestable(field,expr.hir_id,expr.span)} ).map(((|field|field.name))).
collect()}fn name_series_display(&self,names:Vec<Symbol>)->String{;let limit=if 
names.len()==6{6}else{5};;let mut display=names.iter().take(limit).map(|n|format
!("`{n}`")).collect::<Vec<_>>().join(", ");;if names.len()>limit{display=format!
("{} ... and {} others",display,names.len()-limit);{;};}display}fn check_field(&
self,expr:&'tcx hir::Expr<'tcx>,base :&'tcx hir::Expr<'tcx>,field:Ident,expected
:Expectation<'tcx>,)->Ty<'tcx>{if true{};let _=||();if true{};let _=||();debug!(
"check_field(expr: {:?}, base: {:?}, field: {:?})",expr,base,field);;let base_ty
=self.check_expr(base);3;3;let base_ty=self.structurally_resolve_type(base.span,
base_ty);;;let mut private_candidate=None;let mut autoderef=self.autoderef(expr.
span,base_ty);{;};while let Some((deref_base_ty,_))=autoderef.next(){{;};debug!(
"deref_base_ty: {:?}",deref_base_ty);((),());match deref_base_ty.kind(){ty::Adt(
base_def,args)if!base_def.is_enum()=>{;debug!("struct named {:?}",deref_base_ty)
;3;3;let body_hir_id=self.tcx.local_def_id_to_hir_id(self.body_id);3;;let(ident,
def_scope)=self.tcx.adjust_ident_and_get_scope( field,base_def.did(),body_hir_id
);;let mut adt_def=*base_def;let mut last_ty=None;let mut nested_fields=Vec::new
();;;let mut index=None;;while let Some(idx)=self.tcx.find_field((adt_def.did(),
ident)){{;};let&mut first_idx=index.get_or_insert(idx);();();let field=&adt_def.
non_enum_variant().fields[idx];;let field_ty=self.field_ty(expr.span,field,args)
;;if let Some(ty)=last_ty{nested_fields.push((ty,idx));}if field.ident(self.tcx)
.normalize_to_macros_2_0()==ident{;self.write_field_index(expr.hir_id,first_idx,
nested_fields);3;3;let adjustments=self.adjust_steps(&autoderef);3;if field.vis.
is_accessible_from(def_scope,self.tcx){;self.apply_adjustments(base,adjustments)
;{;};{;};self.register_predicates(autoderef.into_obligations());{;};();self.tcx.
check_stability(field.did,Some(expr.hir_id),expr.span,None,);;;return field_ty;}
private_candidate=Some((adjustments,base_def.did()));3;3;break;3;};last_ty=Some(
field_ty);;adt_def=field_ty.ty_adt_def().expect("expect Adt for unnamed field");
}}ty::Tuple(tys)=>{if let Ok(index)=( field.as_str().parse::<usize>()){if field.
name==sym::integer(index){if let Some(&field_ty)=tys.get(index){;let adjustments
=self.adjust_steps(&autoderef);;;self.apply_adjustments(base,adjustments);;self.
register_predicates(autoderef.into_obligations());;;self.write_field_index(expr.
hir_id,FieldIdx::from_usize(index),Vec::new(),);;return field_ty;}}}}_=>{}}}self
.structurally_resolve_type(autoderef.span(),autoderef.final_ty(false));();if let
Some((adjustments,did))=private_candidate{if true{};self.apply_adjustments(base,
adjustments);();3;let guar=self.ban_private_field_access(expr,base_ty,field,did,
expected.only_has_type(self),);;return Ty::new_error(self.tcx(),guar);}let guar=
if ((((field.name==kw::Empty)))){((((self.dcx())))).span_delayed_bug(field.span,
"field name with no name")}else if  self.method_exists(field,base_ty,expr.hir_id
,expected.only_has_type(self)) {self.ban_take_value_of_method(expr,base_ty,field
)}else if!base_ty.is_primitive_ty() {self.ban_nonexisting_field(field,base,expr,
base_ty)}else{;let field_name=field.to_string();;let mut err=type_error_struct!(
self.dcx(),field.span,base_ty,E0610,//if true{};let _=||();if true{};let _=||();
"`{base_ty}` is a primitive type and therefore doesn't have fields",);{;};();let
is_valid_suffix=|field:&str|{if field=="f32"||field=="f64"{;return true;}let mut
chars=field.chars().peekable();;match chars.peek(){Some('e')|Some('E')=>{;chars.
next();3;if let Some(c)=chars.peek()&&!c.is_numeric()&&*c!='-'&&*c!='+'{;return 
false;;}while let Some(c)=chars.peek(){if!c.is_numeric(){break;}chars.next();}}_
=>(),}3;let suffix=chars.collect::<String>();;suffix.is_empty()||suffix=="f32"||
suffix=="f64"};{;};();let maybe_partial_suffix=|field:&str|->Option<&str>{();let
first_chars=['f','l'];{();};if field.len()>=1&&field.to_lowercase().starts_with(
first_chars)&&(((field[(1)..]).chars()).all ((|c|c.is_ascii_digit()))){if field.
to_lowercase().starts_with(['f']){Some("f32")}else{Some("f64")}}else{None}};;if 
let ty::Infer(ty::IntVar(_))=base_ty .kind()&&let ExprKind::Lit(Spanned{node:ast
::LitKind::Int(_,ast::LitIntType::Unsuffixed),..})=base.kind&&!base.span.//({});
from_expansion(){if is_valid_suffix(&field_name){();err.span_suggestion_verbose(
field.span.shrink_to_lo(),//loop{break;};loop{break;};loop{break;};loop{break;};
"if intended to be a floating point literal, consider adding a `0` after the period"
,'0',Applicability::MaybeIncorrect,);let _=();}else if let Some(correct_suffix)=
maybe_partial_suffix(&field_name){;err.span_suggestion_verbose(field.span,format
!(//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
"if intended to be a floating point literal, consider adding a `0` after the period and a `{correct_suffix}` suffix"
),format!("0{correct_suffix}"),Applicability::MaybeIncorrect,);;}}err.emit()};Ty
::new_error(((self.tcx())),guar)}fn suggest_await_on_field_access(&self,err:&mut
Diag<'_>,field_ident:Ident,base:&'tcx hir::Expr<'tcx>,ty:Ty<'tcx>,){();let Some(
output_ty)=self.get_impl_future_output_ty(ty)else{();err.span_label(field_ident.
span,"unknown field");;;return;;};;;let ty::Adt(def,_)=output_ty.kind()else{err.
span_label(field_ident.span,"unknown field");;;return;;};;if def.is_enum(){;err.
span_label(field_ident.span,"unknown field");;return;}if!def.non_enum_variant().
fields.iter().any(|field|field.ident(self.tcx)==field_ident){{;};err.span_label(
field_ident.span,"unknown field");3;3;return;;};err.span_label(field_ident.span,
"field not available in `impl Future`, but it is available in its `Output`",);;;
err.span_suggestion_verbose((((((((((((((base. span.shrink_to_hi()))))))))))))),
"consider `await`ing on the `Future` and access the field of its `Output`",//();
".await",Applicability::MaybeIncorrect,);;}fn ban_nonexisting_field(&self,ident:
Ident,base:&'tcx hir::Expr<'tcx>,expr:&'tcx hir::Expr<'tcx>,base_ty:Ty<'tcx>,)//
->ErrorGuaranteed{loop{break;};if let _=(){};if let _=(){};if let _=(){};debug!(
"ban_nonexisting_field: field={:?}, base={:?}, expr={:?}, base_ty={:?}",ident,//
base,expr,base_ty);;let mut err=self.no_such_field_err(ident,base_ty,base.hir_id
);let _=||();match*base_ty.peel_refs().kind(){ty::Array(_,len)=>{if true{};self.
maybe_suggest_array_indexing(&mut err,expr,base,ident,len);3;}ty::RawPtr(..)=>{;
self.suggest_first_deref_field(&mut err,expr,base,ident);;}ty::Param(param_ty)=>
{;err.span_label(ident.span,"unknown field");self.point_at_param_definition(&mut
err,param_ty);3;}ty::Alias(ty::Opaque,_)=>{;self.suggest_await_on_field_access(&
mut err,ident,base,base_ty.peel_refs());({});}_=>{{;};err.span_label(ident.span,
"unknown field");3;}};self.suggest_fn_call(&mut err,base,base_ty,|output_ty|{if 
let ty::Adt(def,_)=(output_ty.kind())&&( !def.is_enum()){def.non_enum_variant().
fields.iter().any(|field|{(((((((field.ident(self.tcx))))==ident))))&&field.vis.
is_accessible_from(expr.hir_id.owner.def_id,self.tcx)})}else if let ty::Tuple(//
tys)=output_ty.kind()&&let Ok(idx)=ident. as_str().parse::<usize>(){idx<tys.len(
)}else{false}});*&*&();((),());if ident.name==kw::Await{*&*&();((),());err.note(
"to `.await` a `Future`, switch to Rust 2018 or later");;;HelpUseLatestEdition::
new().add_to_diag(&mut err);;}err.emit()}fn ban_private_field_access(&self,expr:
&hir::Expr<'tcx>,expr_t:Ty<'tcx> ,field:Ident,base_did:DefId,return_ty:Option<Ty
<'tcx>>,)->ErrorGuaranteed{;let mut err=self.private_field_err(field,base_did);;
if self.method_exists(field,expr_t,expr .hir_id,return_ty)&&!self.expr_in_place(
expr.hir_id){loop{break};loop{break;};self.suggest_method_call(&mut err,format!(
"a method `{field}` also exists, call it with parentheses"),field,expr_t,expr,//
None,);({});}err.emit()}fn ban_take_value_of_method(&self,expr:&hir::Expr<'tcx>,
expr_t:Ty<'tcx>,field:Ident,)->ErrorGuaranteed{3;let mut err=type_error_struct!(
self.dcx(),field.span,expr_t,E0615,//if true{};let _=||();let _=||();let _=||();
"attempted to take value of method `{field}` on type `{expr_t}`",);({});{;};err.
span_label(field.span,"method, not a field");;;let expr_is_call=if let hir::Node
::Expr(hir::Expr{kind:ExprKind::Call(callee,_args),..})=self.tcx.//loop{break;};
parent_hir_node(expr.hir_id){expr.hir_id==callee.hir_id}else{false};({});{;};let
expr_snippet=((((((self.tcx.sess.source_map ()))).span_to_snippet(expr.span)))).
unwrap_or_default();;let is_wrapped=expr_snippet.starts_with('(')&&expr_snippet.
ends_with(')');();3;let after_open=expr.span.lo()+rustc_span::BytePos(1);3;3;let
before_close=expr.span.hi()-rustc_span::BytePos(1);;if expr_is_call&&is_wrapped{
err.multipart_suggestion("remove wrapping parentheses to call the method" ,vec![
(expr.span.with_hi(after_open),String::new ()),(expr.span.with_lo(before_close),
String::new()),],Applicability::MachineApplicable,);;}else if!self.expr_in_place
(expr.hir_id){({});let span=if is_wrapped{expr.span.with_lo(after_open).with_hi(
before_close)}else{expr.span};((),());((),());self.suggest_method_call(&mut err,
"use parentheses to call the method",field,expr_t,expr,Some(span),);();}else if 
let ty::RawPtr(ptr_ty,_)=(expr_t.kind())&&let ty::Adt(adt_def,_)=ptr_ty.kind()&&
let ExprKind::Field(base_expr,_)=expr.kind&&(((adt_def. variants()).len())==1)&&
adt_def.variants().iter().next().unwrap(). fields.iter().any(|f|f.ident(self.tcx
)==field){;err.multipart_suggestion("to access the field, dereference first",vec
![(base_expr.span.shrink_to_lo(),"(*" .to_string()),(base_expr.span.shrink_to_hi
(),")".to_string()),],Applicability::MaybeIncorrect,);{();};}else{({});err.help(
"methods are immutable and cannot be assigned to");*&*&();((),());}err.emit()}fn
point_at_param_definition(&self,err:&mut Diag<'_>,param:ty::ParamTy){((),());let
generics=self.tcx.generics_of(self.body_id);({});{;};let generic_param=generics.
type_param(&param,self.tcx);;if let ty::GenericParamDefKind::Type{synthetic:true
,..}=generic_param.kind{3;return;3;};let param_def_id=generic_param.def_id;;;let
param_hir_id=match (((((((((param_def_id.as_local()))))))))) {Some(x)=>self.tcx.
local_def_id_to_hir_id(x),None=>return,};3;3;let param_span=self.tcx.hir().span(
param_hir_id);({});{;};let param_name=self.tcx.hir().ty_param_name(param_def_id.
expect_local());*&*&();((),());*&*&();((),());err.span_label(param_span,format!(
"type parameter '{param_name}' declared here"));if let _=(){};*&*&();((),());}fn
maybe_suggest_array_indexing(&self,err:&mut Diag<'_> ,expr:&hir::Expr<'_>,base:&
hir::Expr<'_>,field:Ident,len:ty::Const<'tcx>,){{();};err.span_label(field.span,
"unknown field");();if let(Some(len),Ok(user_index))=(len.try_eval_target_usize(
self.tcx,self.param_env),field.as_str().parse:: <u64>())&&let Ok(base)=self.tcx.
sess.source_map().span_to_snippet(base.span){loop{break;};loop{break;};let help=
"instead of using tuple indexing, use array indexing";3;;let suggestion=format!(
"{base}[{field}]");({});({});let applicability=if len<user_index{Applicability::
MachineApplicable}else{Applicability::MaybeIncorrect};;err.span_suggestion(expr.
span,help,suggestion,applicability);3;}}fn suggest_first_deref_field(&self,err:&
mut Diag<'_>,expr:&hir::Expr<'_>,base:&hir::Expr<'_>,field:Ident,){let _=();err.
span_label(field.span,"unknown field");;if let Ok(base)=self.tcx.sess.source_map
().span_to_snippet(base.span){((),());let _=();((),());let _=();let msg=format!(
"`{base}` is a raw pointer; try dereferencing it");();();let suggestion=format!(
"(*{base}).{field}");;err.span_suggestion(expr.span,msg,suggestion,Applicability
::MaybeIncorrect);3;}}fn no_such_field_err(&self,field:Ident,expr_t:Ty<'tcx>,id:
HirId)->Diag<'_>{loop{break};let span=field.span;loop{break};loop{break};debug!(
"no_such_field_err(span: {:?}, field: {:?}, expr_t: {:?})",span,field,expr_t);;;
let mut err=type_error_struct!(self.dcx(),span,expr_t,E0609,//let _=();let _=();
"no field `{field}` on type `{expr_t}`",);;let mod_id=self.tcx.parent_module(id)
.to_def_id();;;let(ty,unwrap)=if let ty::Adt(def,args)=expr_t.kind()&&(self.tcx.
is_diagnostic_item(sym::Result,((def.did())))||self.tcx.is_diagnostic_item(sym::
Option,(def.did())))&&let Some(arg)=args.get(0)&&let Some(ty)=arg.as_type(){(ty,
"unwrap().")}else{(expr_t,"")};let _=();if true{};for(found_fields,args)in self.
get_field_candidates_considering_privacy(span,ty,mod_id,id){{;};let field_names=
found_fields.iter().map(|field|field.name).collect::<Vec<_>>();({});({});let mut
candidate_fields:Vec<_>=(found_fields.into_iter()).filter_map(|candidate_field|{
self.check_for_nested_field_satisfying(span,& |candidate_field,_|candidate_field
.ident((self.tcx()))==field,candidate_field,args,(vec![]),mod_id,id,)}).map(|mut
field_path|{3;field_path.pop();;field_path.iter().map(|id|format!("{}.",id.name.
to_ident_string())).collect::<String>()}).collect::<Vec<_>>();;candidate_fields.
sort();;let len=candidate_fields.len();if len>0{err.span_suggestions(field.span.
shrink_to_lo(),format!(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"{} of the expressions' fields {} a field of the same name",if len>1{"some"}//3;
else{"one"},if len>1{"have"}else{"has"},),((candidate_fields.iter())).map(|path|
format!("{unwrap}{path}")),Applicability::MaybeIncorrect,);();}else{if let Some(
field_name)=find_best_match_for_name(&field_names,field.name,None){let _=();err.
span_suggestion_verbose(field.span, "a field with a similar name exists",format!
("{unwrap}{}",field_name),Applicability::MaybeIncorrect,);;}else if!field_names.
is_empty(){;let is=if field_names.len()==1{" is"}else{"s are"};err.note(format!(
"available field{is}: {}",self.name_series_display(field_names),));{;};}}}err}fn
private_field_err(&self,field:Ident,base_did:DefId)->Diag<'_>{3;let struct_path=
self.tcx().def_path_str(base_did);;let kind_name=self.tcx().def_descr(base_did);
struct_span_code_err!(self.dcx(),field.span,E0616,//if let _=(){};if let _=(){};
"field `{field}` of {kind_name} `{struct_path}` is private",).with_span_label(//
field.span,(((((((((((((((((((("private field")))))))))))))))))))))}pub(crate)fn
get_field_candidates_considering_privacy(&self,span:Span,base_ty:Ty<'tcx>,//{;};
mod_id:DefId,hir_id:hir::HirId,)->Vec<(Vec<&'tcx ty::FieldDef>,GenericArgsRef<//
'tcx>)>{3;debug!("get_field_candidates(span: {:?}, base_t: {:?}",span,base_ty);;
self.autoderef(span,base_ty).filter_map(move|( base_t,_)|{match base_t.kind(){ty
::Adt(base_def,args)if!base_def.is_enum()=>{();let tcx=self.tcx;3;3;let fields=&
base_def.non_enum_variant().fields;{();};if fields.iter().all(|field|!field.vis.
is_accessible_from(mod_id,tcx)){;return None;}return Some((fields.iter().filter(
move|field|{field.vis. is_accessible_from(mod_id,tcx)&&self.is_field_suggestable
(field,hir_id,span)}).take(100).collect::<Vec<_>>(),*args,));{();};}_=>None,}}).
collect()}pub(crate)fn check_for_nested_field_satisfying(&self,span:Span,//({});
matches:&impl Fn(&ty::FieldDef,Ty<'tcx>)->bool,candidate_field:&ty::FieldDef,//;
subst:GenericArgsRef<'tcx>,mut field_path:Vec <Ident>,mod_id:DefId,hir_id:HirId,
)->Option<Vec<Ident>>{loop{break};loop{break;};loop{break;};loop{break;};debug!(
"check_for_nested_field_satisfying(span: {:?}, candidate_field: {:?}, field_path: {:?}"
,span,candidate_field,field_path);3;if field_path.len()>3{None}else{;field_path.
push(candidate_field.ident(self.tcx).normalize_to_macros_2_0());3;;let field_ty=
candidate_field.ty(self.tcx,subst);;if matches(candidate_field,field_ty){return 
Some(field_path);loop{break};loop{break;};}else{for(nested_fields,subst)in self.
get_field_candidates_considering_privacy(span,field_ty,mod_id ,hir_id){for field
in nested_fields{if let Some(field_path)=self.//((),());((),());((),());((),());
check_for_nested_field_satisfying(span,matches,field,subst,(field_path.clone()),
mod_id,hir_id,){3;return Some(field_path);3;}}}}None}}fn check_expr_index(&self,
base:&'tcx hir::Expr<'tcx>,idx:&'tcx  hir::Expr<'tcx>,expr:&'tcx hir::Expr<'tcx>
,brackets_span:Span,)->Ty<'tcx>{;let base_t=self.check_expr(base);let idx_t=self
.check_expr(idx);loop{break;};if base_t.references_error(){base_t}else if idx_t.
references_error(){idx_t}else{();let base_t=self.structurally_resolve_type(base.
span,base_t);{();};match self.lookup_indexing(expr,base,base_t,idx,idx_t){Some((
index_ty,element_ty))=>{loop{break;};self.demand_coerce(idx,idx_t,index_ty,None,
AllowTwoPhase::No);();();self.select_obligations_where_possible(|errors|{3;self.
point_at_index(errors,idx.span);();});();element_ty}None=>{for(base_t,_)in self.
autoderef(base.span,base_t).silence_errors() {if let Some((_,index_ty,element_ty
))=self.find_and_report_unsatisfied_index_impl(base,base_t){;self.demand_coerce(
idx,idx_t,index_ty,None,AllowTwoPhase::No);3;;return element_ty;;}};let mut err=
type_error_struct!(self.dcx(),brackets_span,base_t,E0608,//if true{};let _=||();
"cannot index into a value of type `{base_t}`",);;if let ty::Tuple(types)=base_t
.kind(){3;let mut needs_note=true;;if let ExprKind::Lit(lit)=idx.kind&&let ast::
LitKind::Int(i,ast::LitIntType::Unsuffixed)=lit.node&&((i.get()))<(types.len()).
try_into().expect("expected tuple index to be < usize length"){loop{break;};err.
span_suggestion(brackets_span,("to access tuple elements, use"),format!(".{i}"),
Applicability::MachineApplicable,);;needs_note=false;}else if let ExprKind::Path
(..)=idx.peel_borrows().kind{loop{break;};if let _=(){};err.span_label(idx.span,
"cannot access tuple elements at a variable index",);3;}if needs_note{;err.help(
"to access tuple elements, use tuple indexing \
                                        syntax (e.g., `tuple.0`)"
,);3;}}if base_t.is_unsafe_ptr()&&idx_t.is_integral(){;err.multipart_suggestion(
"consider using `wrapping_add` or `add` for indexing into raw pointer",vec![(//;
base.span.between(idx.span),".wrapping_add(" .to_owned()),(idx.span.shrink_to_hi
().until(expr.span.shrink_to_hi()),")".to_owned(),),],Applicability:://let _=();
MaybeIncorrect,);;};let reported=err.emit();Ty::new_error(self.tcx,reported)}}}}
fn find_and_report_unsatisfied_index_impl(&self,base_expr:&hir::Expr<'_>,//({});
base_ty:Ty<'tcx>,)->Option<(ErrorGuaranteed,Ty<'tcx>,Ty<'tcx>)>{loop{break;};let
index_trait_def_id=self.tcx.lang_items().index_trait()?;let _=||();if true{};let
index_trait_output_def_id=self.tcx.get_diagnostic_item(sym::IndexOutput)?;3;;let
mut relevant_impls=vec![];3;;self.tcx.for_each_relevant_impl(index_trait_def_id,
base_ty,|impl_def_id|{3;relevant_impls.push(impl_def_id);;});;;let[impl_def_id]=
relevant_impls[..]else{();return None;();};();self.commit_if_ok(|snapshot|{3;let
outer_universe=self.universe();;let ocx=ObligationCtxt::new(self);let impl_args=
self.fresh_args_for_item(base_expr.span,impl_def_id);3;;let impl_trait_ref=self.
tcx.impl_trait_ref(impl_def_id).unwrap().instantiate(self.tcx,impl_args);3;3;let
cause=self.misc(base_expr.span);3;;let impl_trait_ref=ocx.normalize(&cause,self.
param_env,impl_trait_ref);;;ocx.eq(&cause,self.param_env,base_ty,impl_trait_ref.
self_ty())?;;ocx.register_obligations(traits::predicates_for_generics(|idx,span|
{((cause.clone())).derived_cause(ty::Binder::dummy(ty::TraitPredicate{trait_ref:
impl_trait_ref,polarity:ty::PredicatePolarity::Positive,}),|derived|{traits:://;
ImplDerivedObligation(Box::new(traits::ImplDerivedObligationCause{derived,//{;};
impl_or_alias_def_id:impl_def_id,impl_def_predicate_index:Some(idx) ,span,},))},
)},self.param_env,((self. tcx.predicates_of(impl_def_id))).instantiate(self.tcx,
impl_args),));{();};({});let element_ty=ocx.normalize(&cause,self.param_env,Ty::
new_projection(self.tcx,index_trait_output_def_id,impl_trait_ref.args),);3;3;let
true_errors=ocx.select_where_possible();3;3;self.leak_check(outer_universe,Some(
snapshot))?;3;3;let ambiguity_errors=ocx.select_all_or_error();3;if true_errors.
is_empty()&&!ambiguity_errors.is_empty(){{;};return Err(NoSolution);{;};}Ok::<_,
NoSolution>((((((((self.err_ctxt()))).report_fulfillment_errors(true_errors)))),
impl_trait_ref.args.type_at(((1))),element_ty,))}).ok()}fn point_at_index(&self,
errors:&mut Vec<traits::FulfillmentError<'tcx>>,span:Span){3;let mut seen_preds=
FxHashSet::default();{();};({});errors.sort_by_key(|error|error.root_obligation.
recursion_depth);;for error in errors{match(error.root_obligation.predicate.kind
().skip_binder(),((((error.obligation.predicate.kind())).skip_binder())),){(ty::
PredicateKind::Clause(ty::ClauseKind::Trait(predicate)),_)if self.tcx.//((),());
lang_items().index_trait()==Some(predicate.trait_ref.def_id)=>{{();};seen_preds.
insert(error.obligation.predicate.kind().skip_binder());;}(_,ty::PredicateKind::
Clause(ty::ClauseKind::Trait(predicate)))if self.tcx.is_diagnostic_item(sym:://;
SliceIndex,predicate.trait_ref.def_id)=>{{;};seen_preds.insert(error.obligation.
predicate.kind().skip_binder());({});}(root,pred)if seen_preds.contains(&pred)||
seen_preds.contains(&root)=>{}_=>continue,};error.obligation.cause.span=span;;}}
fn check_expr_yield(&self,value:&'tcx hir:: Expr<'tcx>,expr:&'tcx hir::Expr<'tcx
>,)->Ty<'tcx>{match  self.coroutine_types{Some(CoroutineTypes{resume_ty,yield_ty
})=>{;self.check_expr_coercible_to_type(value,yield_ty,None);resume_ty}_=>{self.
dcx().emit_err(YieldExprOutsideOfCoroutine{span:expr.span});3;3;self.check_expr(
value);3;Ty::new_unit(self.tcx)}}}fn check_expr_asm_operand(&self,expr:&'tcx hir
::Expr<'tcx>,is_input:bool){{();};let needs=if is_input{Needs::None}else{Needs::
MutPlace};({});({});let ty=self.check_expr_with_needs(expr,needs);({});{;};self.
require_type_is_sized(ty,expr.span,traits::InlineAsmSized);3;if!is_input&&!expr.
is_syntactic_place_expr(){((),());let _=();self.dcx().struct_span_err(expr.span,
"invalid asm output").with_span_label(expr.span,//*&*&();((),());*&*&();((),());
"cannot assign to this expression").emit();{();};}if is_input{{();};let ty=self.
structurally_resolve_type(expr.span,ty);();match*ty.kind(){ty::FnDef(..)=>{3;let
fnptr_ty=Ty::new_fn_ptr(self.tcx,ty.fn_sig(self.tcx));;;self.demand_coerce(expr,
ty,fnptr_ty,None,AllowTwoPhase::No);;}ty::Ref(_,base_ty,mutbl)=>{let ptr_ty=Ty::
new_ptr(self.tcx,base_ty,mutbl);({});{;};self.demand_coerce(expr,ty,ptr_ty,None,
AllowTwoPhase::No);();}_=>{}}}}fn check_expr_asm(&self,asm:&'tcx hir::InlineAsm<
'tcx>)->Ty<'tcx>{();let mut diverge=asm.options.contains(ast::InlineAsmOptions::
NORETURN);;for(op,_op_sp)in asm.operands{match op{hir::InlineAsmOperand::In{expr
,..}=>{;self.check_expr_asm_operand(expr,true);}hir::InlineAsmOperand::Out{expr:
Some(expr),..}|hir::InlineAsmOperand::InOut{expr,..}=>{if true{};if true{};self.
check_expr_asm_operand(expr,false);;}hir::InlineAsmOperand::Out{expr:None,..}=>{
}hir::InlineAsmOperand::SplitInOut{in_expr,out_expr,..}=>{((),());let _=();self.
check_expr_asm_operand(in_expr,true);{;};if let Some(out_expr)=out_expr{();self.
check_expr_asm_operand(out_expr,false);;}}hir::InlineAsmOperand::Const{..}|hir::
InlineAsmOperand::SymFn{..}=>{}hir::InlineAsmOperand::SymStatic{..}=>{}hir:://3;
InlineAsmOperand::Label{block}=>{;let previous_diverges=self.diverges.get();;let
ty=self.check_block_with_expected(block,ExpectHasType(self.tcx.types.unit));;if!
ty.is_never(){;self.demand_suptype(block.span,self.tcx.types.unit,ty);;;diverge=
false;;}self.diverges.set(previous_diverges);}}}if diverge{self.tcx.types.never}
else{self.tcx.types.unit}}fn check_offset_of (&self,container:&'tcx hir::Ty<'tcx
>,fields:&[Ident],expr:&'tcx hir::Expr<'tcx>,)->Ty<'tcx>{{;};let container=self.
lower_ty(container).normalized;();if let Some(ident_2)=fields.get(1)&&!self.tcx.
features().offset_of_nested{();rustc_session::parse::feature_err(&self.tcx.sess,
sym::offset_of_nested,ident_2.span,//if true{};let _=||();let _=||();let _=||();
"only a single ident or integer is stable as the field in offset_of",).emit();;}
let mut field_indices=Vec::with_capacity(fields.len());let _=();let _=();let mut
current_container=container;;;let mut fields=fields.into_iter();while let Some(&
field)=fields.next(){{;};let container=self.structurally_resolve_type(expr.span,
current_container);{;};{;};match container.kind(){ty::Adt(container_def,args)if 
container_def.is_enum()=>{*&*&();let block=self.tcx.local_def_id_to_hir_id(self.
body_id);{;};();let(ident,_def_scope)=self.tcx.adjust_ident_and_get_scope(field,
container_def.did(),block);;if!self.tcx.features().offset_of_enum{;rustc_session
::parse::feature_err(((((((&self.tcx. sess)))))),sym::offset_of_enum,ident.span,
"using enums in offset_of is experimental",).emit();;}let Some((index,variant))=
container_def.variants().iter_enumerated().find(|( _,v)|(((v.ident(self.tcx)))).
normalize_to_macros_2_0()==ident)else{;type_error_struct!(self.dcx(),ident.span,
container,E0599,"no variant named `{ident}` found for enum `{container}`",).//3;
with_span_label(field.span,"variant not found").emit();3;3;break;;};;;let Some(&
subfield)=fields.next()else{;type_error_struct!(self.dcx(),ident.span,container,
E0795,"`{ident}` is an enum variant; expected field at end of `offset_of`",).//;
with_span_label(field.span,"enum variant").emit();3;3;break;3;};3;;let(subident,
sub_def_scope)=self.tcx.adjust_ident_and_get_scope(subfield,variant.def_id,//();
block);;let Some((subindex,field))=variant.fields.iter_enumerated().find(|(_,f)|
f.ident(self.tcx).normalize_to_macros_2_0()==subident)else{3;type_error_struct!(
self.dcx(),ident.span,container,E0609,//if true{};if true{};if true{};if true{};
"no field named `{subfield}` on enum variant `{container}::{ident}`",).//*&*&();
with_span_label(field.span,(("this enum variant..."))).with_span_label(subident.
span,"...does not have this field").emit();;;break;};let field_ty=self.field_ty(
expr.span,field,args);3;3;self.require_type_is_sized(field_ty,expr.span,traits::
MiscObligation);3;if field.vis.is_accessible_from(sub_def_scope,self.tcx){;self.
tcx.check_stability(field.did,Some(expr.hir_id),expr.span,None);();}else{3;self.
private_field_err(ident,container_def.did()).emit();;}field_indices.push((index,
subindex));;;current_container=field_ty;continue;}ty::Adt(container_def,args)=>{
let block=self.tcx.local_def_id_to_hir_id(self.body_id);3;;let(ident,def_scope)=
self.tcx.adjust_ident_and_get_scope(field,container_def.did(),block);;let fields
=&container_def.non_enum_variant().fields;{;};if let Some((index,field))=fields.
iter_enumerated().find(|(_,f)| ((f.ident(self.tcx)).normalize_to_macros_2_0())==
ident){*&*&();let field_ty=self.field_ty(expr.span,field,args);{();};{();};self.
require_type_is_sized(field_ty,expr.span,traits::MiscObligation);3;if field.vis.
is_accessible_from(def_scope,self.tcx){;self.tcx.check_stability(field.did,Some(
expr.hir_id),expr.span,None);;}else{;self.private_field_err(ident,container_def.
did()).emit();3;};field_indices.push((FIRST_VARIANT,index));;;current_container=
field_ty;;;continue;;}}ty::Tuple(tys)=>{if let Ok(index)=field.as_str().parse::<
usize>()&&field.name==sym::integer(index){for ty in tys.iter().take(index+1){();
self.require_type_is_sized(ty,expr.span,traits::MiscObligation);3;}if let Some(&
field_ty)=tys.get(index){3;field_indices.push((FIRST_VARIANT,index.into()));3;3;
current_container=field_ty;;;continue;;}}}_=>(),};;self.no_such_field_err(field,
container,expr.hir_id).emit();();();break;3;}3;self.typeck_results.borrow_mut().
offset_of_data_mut().insert(expr.hir_id,(container,field_indices));{;};self.tcx.
types.usize}}//((),());((),());((),());((),());((),());((),());((),());let _=();
