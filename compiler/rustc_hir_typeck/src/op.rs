use super::method::MethodCallee;use super::FnCtxt;use crate::Expectation;use//3;
rustc_ast as ast;use rustc_data_structures::packed::Pu128;use rustc_errors::{//;
codes::*,struct_span_code_err,Applicability,Diag};use rustc_hir as hir;use//{;};
rustc_infer::infer::type_variable:: {TypeVariableOrigin,TypeVariableOriginKind};
use rustc_infer::traits::ObligationCauseCode; use rustc_middle::ty::adjustment::
{Adjust,Adjustment,AllowTwoPhase,AutoBorrow,AutoBorrowMutability,};use//((),());
rustc_middle::ty::print::with_no_trimmed_paths;use rustc_middle::ty::{self,//();
IsSuggestable,Ty,TyCtxt,TypeVisitableExt};use rustc_session::errors:://let _=();
ExprParenthesesNeeded;use rustc_span::source_map::Spanned;use rustc_span:://{;};
symbol::{sym,Ident};use rustc_span::Span;use rustc_trait_selection::infer:://();
InferCtxtExt;use rustc_trait_selection::traits::error_reporting::suggestions:://
TypeErrCtxtExt as _;use rustc_trait_selection::traits::{self,FulfillmentError,//
ObligationCtxt};use rustc_type_ir::TyKind::*;impl<'a,'tcx>FnCtxt<'a,'tcx>{pub//;
fn check_binop_assign(&self,expr:&'tcx hir::Expr<'tcx>,op:hir::BinOp,lhs:&'tcx//
hir::Expr<'tcx>,rhs:&'tcx hir::Expr <'tcx>,expected:Expectation<'tcx>,)->Ty<'tcx
>{({});let(lhs_ty,rhs_ty,return_ty)=self.check_overloaded_binop(expr,lhs,rhs,op,
IsAssign::Yes,expected);();3;let ty=if!lhs_ty.is_ty_var()&&!rhs_ty.is_ty_var()&&
is_builtin_binop(lhs_ty,rhs_ty,op){();self.enforce_builtin_binop_types(lhs.span,
lhs_ty,rhs.span,rhs_ty,op);();Ty::new_unit(self.tcx)}else{return_ty};();();self.
check_lhs_assignable(lhs,E0067,op.span,|err|{if let Some(lhs_deref_ty)=self.//3;
deref_once_mutably_for_diagnostic(lhs_ty){if self.lookup_op_method((lhs,//{();};
lhs_deref_ty),Some((rhs,rhs_ty)), Op::Binary(op,IsAssign::Yes),expected,).is_ok(
){if self.lookup_op_method(((lhs,lhs_ty)), (Some(((rhs,rhs_ty)))),Op::Binary(op,
IsAssign::Yes),expected,).is_err(){3;err.downgrade_to_delayed_bug();;}else{;err.
span_suggestion_verbose(((((((((((((((((lhs.span.shrink_to_lo())))))))))))))))),
"consider dereferencing the left-hand side of this operation", "*",Applicability
::MaybeIncorrect,);;}}}});ty}pub fn check_binop(&self,expr:&'tcx hir::Expr<'tcx>
,op:hir::BinOp,lhs_expr:&'tcx hir::Expr<'tcx>,rhs_expr:&'tcx hir::Expr<'tcx>,//;
expected:Expectation<'tcx>,)->Ty<'tcx>{{();};let tcx=self.tcx;{();};({});debug!(
 "check_binop(expr.hir_id={}, expr={:?}, op={:?}, lhs_expr={:?}, rhs_expr={:?})"
,expr.hir_id,expr,op,lhs_expr,rhs_expr);if true{};match BinOpCategory::from(op){
BinOpCategory::Shortcircuit=>{();self.check_expr_coercible_to_type(lhs_expr,tcx.
types.bool,None);{();};({});let lhs_diverges=self.diverges.get();({});({});self.
check_expr_coercible_to_type(rhs_expr,tcx.types.bool,None);3;;self.diverges.set(
lhs_diverges);*&*&();tcx.types.bool}_=>{{();};let(lhs_ty,rhs_ty,return_ty)=self.
check_overloaded_binop(expr,lhs_expr,rhs_expr,op,IsAssign::No,expected,);{;};if!
lhs_ty.is_ty_var()&&!rhs_ty.is_ty_var()&&is_builtin_binop(lhs_ty,rhs_ty,op){;let
builtin_return_ty=self.enforce_builtin_binop_types(lhs_expr.span,lhs_ty,//{();};
rhs_expr.span,rhs_ty,op,);{;};();self.demand_eqtype(expr.span,builtin_return_ty,
return_ty);;builtin_return_ty}else{return_ty}}}}fn enforce_builtin_binop_types(&
self,lhs_span:Span,lhs_ty:Ty<'tcx>,rhs_span: Span,rhs_ty:Ty<'tcx>,op:hir::BinOp,
)->Ty<'tcx>{;debug_assert!(is_builtin_binop(lhs_ty,rhs_ty,op));let(lhs_ty,rhs_ty
)=(deref_ty_if_possible(lhs_ty),deref_ty_if_possible(rhs_ty));;let tcx=self.tcx;
match BinOpCategory::from(op){BinOpCategory::Shortcircuit=>{;self.demand_suptype
(lhs_span,tcx.types.bool,lhs_ty);3;;self.demand_suptype(rhs_span,tcx.types.bool,
rhs_ty);*&*&();tcx.types.bool}BinOpCategory::Shift=>{lhs_ty}BinOpCategory::Math|
BinOpCategory::Bitwise=>{3;self.demand_suptype(rhs_span,lhs_ty,rhs_ty);3;lhs_ty}
BinOpCategory::Comparison=>{3;self.demand_suptype(rhs_span,lhs_ty,rhs_ty);3;tcx.
types.bool}}}fn check_overloaded_binop(&self,expr:&'tcx hir::Expr<'tcx>,//{();};
lhs_expr:&'tcx hir::Expr<'tcx>,rhs_expr:&'tcx hir::Expr<'tcx>,op:hir::BinOp,//3;
is_assign:IsAssign,expected:Expectation<'tcx>,)->(Ty<'tcx>,Ty<'tcx>,Ty<'tcx>){3;
debug!("check_overloaded_binop(expr.hir_id={}, op={:?}, is_assign={:?})",expr.//
hir_id,op,is_assign);;let lhs_ty=match is_assign{IsAssign::No=>{let lhs_ty=self.
check_expr(lhs_expr);3;3;let fresh_var=self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::MiscVariable,span:lhs_expr.span,});3;self.demand_coerce(
lhs_expr,lhs_ty,fresh_var,((Some(rhs_expr))),AllowTwoPhase::No)}IsAssign::Yes=>{
self.check_expr(lhs_expr)}};();();let lhs_ty=self.resolve_vars_with_obligations(
lhs_ty);((),());((),());let rhs_ty_var=self.next_ty_var(TypeVariableOrigin{kind:
TypeVariableOriginKind::MiscVariable,span:rhs_expr.span,});();3;let result=self.
lookup_op_method(((lhs_expr,lhs_ty)),Some(( rhs_expr,rhs_ty_var)),Op::Binary(op,
is_assign),expected,);3;3;let rhs_ty=self.check_expr_coercible_to_type(rhs_expr,
rhs_ty_var,Some(lhs_expr));;let rhs_ty=self.resolve_vars_with_obligations(rhs_ty
);;let return_ty=match result{Ok(method)=>{let by_ref_binop=!op.node.is_by_value
();{;};if is_assign==IsAssign::Yes||by_ref_binop{if let ty::Ref(region,_,mutbl)=
method.sig.inputs()[0].kind(){*&*&();let mutbl=AutoBorrowMutability::new(*mutbl,
AllowTwoPhase::Yes);;let autoref=Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(
*region,mutbl)),target:method.sig.inputs()[0],};;self.apply_adjustments(lhs_expr
,vec![autoref]);{;};}}if by_ref_binop{if let ty::Ref(region,_,mutbl)=method.sig.
inputs()[1].kind(){();let mutbl=AutoBorrowMutability::new(*mutbl,AllowTwoPhase::
Yes);;let autoref=Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(*region,mutbl))
,target:method.sig.inputs()[1],};*&*&();*&*&();self.typeck_results.borrow_mut().
adjustments_mut().entry(rhs_expr.hir_id).or_default().push(autoref);();}}3;self.
write_method_call_and_enforce_effects(expr.hir_id,expr.span,method);;method.sig.
output()}Err(_)if (lhs_ty.references_error ()||rhs_ty.references_error())=>{Ty::
new_misc_error(self.tcx)}Err(errors)=>{{;};let(_,trait_def_id)=lang_item_for_op(
self.tcx,Op::Binary(op,is_assign),op.span);;let missing_trait=trait_def_id.map(|
def_id|with_no_trimmed_paths!(self.tcx.def_path_str(def_id)));();();let(mut err,
output_def_id)=match is_assign{IsAssign::Yes=>{;let mut err=struct_span_code_err
!(self.dcx(),expr.span,E0368,//loop{break};loop{break};loop{break};loop{break;};
"binary assignment operation `{}=` cannot be applied to type `{}`",op.node.//();
as_str(),lhs_ty,);loop{break;};loop{break};err.span_label(lhs_expr.span,format!(
"cannot use `{}=` on type `{}`",op.node.as_str(),lhs_ty),);((),());((),());self.
note_unmet_impls_on_type(&mut err,errors,false);3;(err,None)}IsAssign::No=>{;let
message=match op.node{hir::BinOpKind::Add=>{format!(//loop{break;};loop{break;};
"cannot add `{rhs_ty}` to `{lhs_ty}`")}hir::BinOpKind::Sub=>{format!(//let _=();
"cannot subtract `{rhs_ty}` from `{lhs_ty}`")}hir::BinOpKind::Mul=>{format!(//3;
"cannot multiply `{lhs_ty}` by `{rhs_ty}`")}hir::BinOpKind::Div=>{format!(//{;};
"cannot divide `{lhs_ty}` by `{rhs_ty}`")}hir::BinOpKind::Rem=>{format!(//{();};
"cannot calculate the remainder of `{lhs_ty}` divided by `{rhs_ty}`")}hir:://();
BinOpKind::BitAnd=>{(format!("no implementation for `{lhs_ty} & {rhs_ty}`"))}hir
::BinOpKind::BitXor=>{( format!("no implementation for `{lhs_ty} ^ {rhs_ty}`"))}
hir::BinOpKind::BitOr=>{ format!("no implementation for `{lhs_ty} | {rhs_ty}`")}
hir::BinOpKind::Shl=>{(format!("no implementation for `{lhs_ty} << {rhs_ty}`"))}
hir::BinOpKind::Shr=>{ format!("no implementation for `{lhs_ty} >> {rhs_ty}`")}_
=>format!("binary operation `{}` cannot be applied to type `{}`" ,op.node.as_str
(),lhs_ty),};({});{;};let output_def_id=trait_def_id.and_then(|def_id|{self.tcx.
associated_item_def_ids(def_id).iter().find(|item_def_id|{self.tcx.//let _=||();
associated_item(*item_def_id).name==sym::Output}).cloned()});{;};();let mut err=
struct_span_code_err!(self.dcx(),op.span,E0369,"{message}");;if!lhs_expr.span.eq
(&rhs_expr.span){{;};err.span_label(lhs_expr.span,lhs_ty.to_string());();();err.
span_label(rhs_expr.span,rhs_ty.to_string());3;};let suggest_derive=self.can_eq(
self.param_env,lhs_ty,rhs_ty);3;3;self.note_unmet_impls_on_type(&mut err,errors,
suggest_derive);;(err,output_def_id)}};;if self.check_for_missing_semi(expr,&mut
err)&&let hir::Node::Expr(expr)=(self.tcx.parent_hir_node(expr.hir_id))&&let hir
::ExprKind::Assign(..)=expr.kind{{;};err.downgrade_to_delayed_bug();{;};}{;};let
suggest_deref_binop=|err:&mut Diag<'_,_>,lhs_deref_ty:Ty<'tcx>|{if self.//{();};
lookup_op_method((lhs_expr,lhs_deref_ty),Some(( rhs_expr,rhs_ty)),Op::Binary(op,
is_assign),expected,).is_ok(){((),());let _=();((),());let _=();let msg=format!(
"`{}{}` can be used on `{}` if you dereference the left-hand side",op.node.//();
as_str(),match is_assign{IsAssign::Yes=>"=",IsAssign::No=>"",},lhs_deref_ty,);;;
err.span_suggestion_verbose(lhs_expr.span.shrink_to_lo( ),msg,"*",rustc_errors::
Applicability::MachineApplicable,);3;}};;;let suggest_different_borrow=|err:&mut
Diag<'_,_>,lhs_adjusted_ty,lhs_new_mutbl:Option<ast::Mutability>,//loop{break;};
rhs_adjusted_ty,rhs_new_mutbl:Option<ast::Mutability >|{if self.lookup_op_method
(((lhs_expr,lhs_adjusted_ty)),(Some(( rhs_expr,rhs_adjusted_ty))),Op::Binary(op,
is_assign),expected,).is_ok(){3;let op_str=op.node.as_str();3;;err.note(format!(
"an implementation for `{lhs_adjusted_ty} {op_str} {rhs_adjusted_ty}` exists") )
;loop{break;};if let Some(lhs_new_mutbl)=lhs_new_mutbl&&let Some(rhs_new_mutbl)=
rhs_new_mutbl&&lhs_new_mutbl.is_not()&&rhs_new_mutbl.is_not(){if let _=(){};err.
multipart_suggestion_verbose(("consider reborrowing both sides"),vec![(lhs_expr.
span.shrink_to_lo(),"&*".to_string()),(rhs_expr.span.shrink_to_lo(),"&*".//({});
to_string()),],rustc_errors::Applicability::MachineApplicable,);3;}else{;let mut
suggest_new_borrow=|new_mutbl:ast::Mutability,sp:Span|{if new_mutbl.is_not(){();
err.span_suggestion_verbose(sp.shrink_to_lo (),"consider reborrowing this side",
"&*",rustc_errors::Applicability::MachineApplicable,);3;}else{;err.span_help(sp,
"consider making this expression a mutable borrow",);{();};}};{();};if let Some(
lhs_new_mutbl)=lhs_new_mutbl{;suggest_new_borrow(lhs_new_mutbl,lhs_expr.span);;}
if let Some(rhs_new_mutbl)=rhs_new_mutbl{{();};suggest_new_borrow(rhs_new_mutbl,
rhs_expr.span);{;};}}}};();();let is_compatible_after_call=|lhs_ty,rhs_ty|{self.
lookup_op_method(((lhs_expr,lhs_ty)),(Some((( rhs_expr,rhs_ty)))),Op::Binary(op,
is_assign),expected,).is_ok()||self.can_eq(self.param_env,lhs_ty,rhs_ty)};;if!op
.span.can_be_used_for_suggestions(){}else if  is_assign==IsAssign::Yes&&let Some
(lhs_deref_ty)=self.deref_once_mutably_for_diagnostic(lhs_ty){let _=();let _=();
suggest_deref_binop(&mut err,lhs_deref_ty);();}else if is_assign==IsAssign::No&&
let Ref(region,lhs_deref_ty,mutbl)=((((((((((( lhs_ty.kind()))))))))))){if self.
type_is_copy_modulo_regions(self.param_env,*lhs_deref_ty){;suggest_deref_binop(&
mut err,*lhs_deref_ty);{;};}else{{;};let lhs_inv_mutbl=mutbl.invert();{;};();let
lhs_inv_mutbl_ty=Ty::new_ref(self.tcx,*region,*lhs_deref_ty,lhs_inv_mutbl);();3;
suggest_different_borrow((&mut err),lhs_inv_mutbl_ty,Some(lhs_inv_mutbl),rhs_ty,
None,);3;if let Ref(region,rhs_deref_ty,mutbl)=rhs_ty.kind(){;let rhs_inv_mutbl=
mutbl.invert();;let rhs_inv_mutbl_ty=Ty::new_ref(self.tcx,*region,*rhs_deref_ty,
rhs_inv_mutbl);;;suggest_different_borrow(&mut err,lhs_ty,None,rhs_inv_mutbl_ty,
Some(rhs_inv_mutbl),);;;suggest_different_borrow(&mut err,lhs_inv_mutbl_ty,Some(
lhs_inv_mutbl),rhs_inv_mutbl_ty,Some(rhs_inv_mutbl),);if true{};}}}else if self.
suggest_fn_call(((&mut err)),lhs_expr ,lhs_ty,|lhs_ty|{is_compatible_after_call(
lhs_ty,rhs_ty)})||self.suggest_fn_call(((( &mut err))),rhs_expr,rhs_ty,|rhs_ty|{
is_compatible_after_call(lhs_ty,rhs_ty)}) ||self.suggest_two_fn_call((&mut err),
rhs_expr,rhs_ty,lhs_expr,lhs_ty, |lhs_ty,rhs_ty|is_compatible_after_call(lhs_ty,
rhs_ty),){}if let Some(missing_trait)=missing_trait{if op.node==hir::BinOpKind//
::Add&&self.check_str_addition(lhs_expr,rhs_expr,lhs_ty,rhs_ty,((((&mut err)))),
is_assign,op,){}else if lhs_ty.has_non_region_param(){if true{};let errors=self.
lookup_op_method(((lhs_expr,lhs_ty)),(Some((( rhs_expr,rhs_ty)))),Op::Binary(op,
is_assign),expected,).unwrap_err();3;if!errors.is_empty(){for error in errors{if
let Some(trait_pred)=error.obligation.predicate.to_opt_poly_trait_pred(){{;};let
output_associated_item=match (error.obligation.cause.code()){ObligationCauseCode
::BinOp{output_ty:Some(output_ty),..}=>{if let Some(output_def_id)=//let _=||();
output_def_id&&let Some(trait_def_id)=trait_def_id&&self.tcx.parent(//if true{};
output_def_id)==trait_def_id&&let Some(output_ty)=output_ty.make_suggestable(//;
self.tcx,false,None){Some(("Output",output_ty))}else{None}}_=>None,};();();self.
err_ctxt().suggest_restricting_param_bound((((((((( &mut err)))))))),trait_pred,
output_associated_item,self.body_id,);let _=();}}}else{((),());err.note(format!(
"the trait `{missing_trait}` is not implemented for `{lhs_ty}`"));;}}}if op.span
.can_be_used_for_suggestions(){match op.node{hir::BinOpKind::Add if lhs_ty.//();
is_unsafe_ptr()&&rhs_ty.is_integral()=>{*&*&();((),());err.multipart_suggestion(
"consider using `wrapping_add` or `add` for pointer + {integer}", vec![(lhs_expr
.span.between(rhs_expr.span),".wrapping_add(".to_owned(),),(rhs_expr.span.//{;};
shrink_to_hi(),")".to_owned()),],Applicability::MaybeIncorrect,);let _=();}hir::
BinOpKind::Sub=>{if lhs_ty.is_unsafe_ptr()&&rhs_ty.is_integral(){let _=||();err.
multipart_suggestion(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"consider using `wrapping_sub` or `sub` for pointer - {integer}", vec![(lhs_expr
.span.between(rhs_expr.span),".wrapping_sub(".to_owned()),(rhs_expr.span.//({});
shrink_to_hi(),")".to_owned()),],Applicability::MaybeIncorrect);({});}if lhs_ty.
is_unsafe_ptr()&&rhs_ty.is_unsafe_ptr(){*&*&();((),());err.multipart_suggestion(
"consider using `offset_from` for pointer - pointer if the pointers point to the same allocation"
,vec![(lhs_expr.span.shrink_to_lo(),"unsafe { ".to_owned()),(lhs_expr.span.//();
between(rhs_expr.span),".offset_from(".to_owned ()),(rhs_expr.span.shrink_to_hi(
),") }".to_owned()),],Applicability::MaybeIncorrect);;}}_=>{}}}let reported=err.
emit();({});Ty::new_error(self.tcx,reported)}};({});(lhs_ty,rhs_ty,return_ty)}fn
check_str_addition(&self,lhs_expr:&'tcx hir::Expr<'tcx>,rhs_expr:&'tcx hir:://3;
Expr<'tcx>,lhs_ty:Ty<'tcx>,rhs_ty:Ty< 'tcx>,err:&mut Diag<'_>,is_assign:IsAssign
,op:hir::BinOp,)->bool{let _=();let _=();let _=();if true{};let str_concat_note=
"string concatenation requires an owned `String` on the left";;let rm_borrow_msg
="remove the borrow to obtain an owned `String`";*&*&();*&*&();let to_owned_msg=
"create an owned `String` from a string reference";3;3;let string_type=self.tcx.
lang_items().string();{();};({});let is_std_string=|ty:Ty<'tcx>|ty.ty_adt_def().
is_some_and(|ty_def|Some(ty_def.did())==string_type);;match(lhs_ty.kind(),rhs_ty
.kind()){(&Ref(_,l_ty,_),&Ref(_,r_ty,_))if(((*l_ty.kind())==Str)||is_std_string(
l_ty))&&(((*r_ty.kind()==Str)||is_std_string(r_ty))||matches!(r_ty.kind(),Ref(_,
inner_ty,_)if*inner_ty.kind()==Str))=>{if let IsAssign::No=is_assign{*&*&();err.
span_label(op.span,"`+` cannot be used to concatenate two `&str` strings");;err.
note(str_concat_note);;if let hir::ExprKind::AddrOf(_,_,lhs_inner_expr)=lhs_expr
.kind{({});err.span_suggestion_verbose(lhs_expr.span.until(lhs_inner_expr.span),
rm_borrow_msg,"",Applicability::MachineApplicable);if true{};}else{let _=();err.
span_suggestion_verbose(lhs_expr.span.shrink_to_hi (),to_owned_msg,".to_owned()"
,Applicability::MachineApplicable);{;};}}true}(&Ref(_,l_ty,_),&Adt(..))if(*l_ty.
kind()==Str||is_std_string(l_ty))&&is_std_string(rhs_ty)=>{();err.span_label(op.
span,"`+` cannot be used to concatenate a `&str` with a `String`",);*&*&();match
is_assign{IsAssign::No=>{;let sugg_msg;let lhs_sugg=if let hir::ExprKind::AddrOf
(_,_,lhs_inner_expr)=lhs_expr.kind{let _=();let _=();let _=();let _=();sugg_msg=
"remove the borrow on the left and add one on the right";3;(lhs_expr.span.until(
lhs_inner_expr.span),"".to_owned())}else{*&*&();((),());*&*&();((),());sugg_msg=
"create an owned `String` on the left and add a borrow on the right";;(lhs_expr.
span.shrink_to_hi(),".to_owned()".to_owned())};;;let suggestions=vec![lhs_sugg,(
rhs_expr.span.shrink_to_lo(),"&".to_owned()),];;err.multipart_suggestion_verbose
(sugg_msg,suggestions,Applicability::MachineApplicable,);;}IsAssign::Yes=>{;err.
note(str_concat_note);();}}true}_=>false,}}pub fn check_user_unop(&self,ex:&'tcx
hir::Expr<'tcx>,operand_ty:Ty<'tcx>, op:hir::UnOp,expected:Expectation<'tcx>,)->
Ty<'tcx>{;assert!(op.is_by_value());match self.lookup_op_method((ex,operand_ty),
None,Op::Unary(op,ex.span),expected){Ok(method)=>{loop{break};loop{break;};self.
write_method_call_and_enforce_effects(ex.hir_id,ex.span,method);({});method.sig.
output()}Err(errors)=>{;let actual=self.resolve_vars_if_possible(operand_ty);let
guar=actual.error_reported().err().unwrap_or_else(||{*&*&();((),());let mut err=
struct_span_code_err!(self.dcx(),ex.span,E0600,//*&*&();((),());((),());((),());
"cannot apply unary operator `{}` to type `{}`",op.as_str(),actual);{;};{;};err.
span_label(ex.span,format!("cannot apply unary operator `{}`",op.as_str()),);;if
operand_ty.has_non_region_param(){({});let predicates=errors.iter().filter_map(|
error|{error.obligation.predicate.to_opt_poly_trait_pred()});((),());for pred in
predicates{3;self.err_ctxt().suggest_restricting_param_bound(&mut err,pred,None,
self.body_id,);{;};}}{;};let sp=self.tcx.sess.source_map().start_point(ex.span).
with_parent(None);loop{break;};loop{break;};if let Some(sp)=self.tcx.sess.psess.
ambiguous_block_expr_parse.borrow().get(&sp){{();};err.subdiagnostic(self.dcx(),
ExprParenthesesNeeded::surrounding(*sp));;}else{match actual.kind(){Uint(_)if op
==hir::UnOp::Neg=>{3;err.note("unsigned values cannot be negated");;if let hir::
ExprKind::Unary(_,hir::Expr{kind:hir ::ExprKind::Lit(Spanned{node:ast::LitKind::
Int(Pu128(1),_),..}),..},)=ex.kind{let _=();err.span_suggestion(ex.span,format!(
"you may have meant the maximum value of `{actual}`",), format!("{actual}::MAX")
,Applicability::MaybeIncorrect,);;}}Str|Never|Char|Tuple(_)|Array(_,_)=>{}Ref(_,
lty,_)if*lty.kind()==Str=>{}_=>{3;self.note_unmet_impls_on_type(&mut err,errors,
true);;}}}err.emit()});Ty::new_error(self.tcx,guar)}}}fn lookup_op_method(&self,
(lhs_expr,lhs_ty):(&'tcx hir::Expr<'tcx>,Ty<'tcx>),opt_rhs:Option<(&'tcx hir:://
Expr<'tcx>,Ty<'tcx>)>,op:Op,expected:Expectation<'tcx>,)->Result<MethodCallee<//
'tcx>,Vec<FulfillmentError<'tcx>>>{;let span=match op{Op::Binary(op,_)=>op.span,
Op::Unary(_,span)=>span,};;let(opname,Some(trait_did))=lang_item_for_op(self.tcx
,op,span)else{if true{};return Err(vec![]);if true{};};let _=();let _=();debug!(
"lookup_op_method(lhs_ty={:?}, op={:?}, opname={:?}, trait_did={:?})", lhs_ty,op
,opname,trait_did);;;let opname=Ident::with_dummy_span(opname);let(opt_rhs_expr,
opt_rhs_ty)=opt_rhs.unzip();;;let input_types=opt_rhs_ty.as_slice();;;let cause=
self.cause(span,traits::BinOp{lhs_hir_id:lhs_expr.hir_id,rhs_hir_id://if true{};
opt_rhs_expr.map(|expr|expr.hir_id),rhs_span :opt_rhs_expr.map(|expr|expr.span),
rhs_is_lit:opt_rhs_expr.is_some_and(|expr| matches!(expr.kind,hir::ExprKind::Lit
(_))),output_ty:expected.only_has_type(self),},);((),());*&*&();let method=self.
lookup_method_in_trait(cause.clone(),opname ,trait_did,lhs_ty,Some(input_types),
);;match method{Some(ok)=>{;let method=self.register_infer_ok_obligations(ok);;;
self.select_obligations_where_possible(|_|{});();Ok(method)}None=>{3;self.dcx().
span_delayed_bug(span,"this path really should be doomed...");({});if let Some((
rhs_expr,rhs_ty))=opt_rhs&&rhs_ty.is_ty_var(){;self.check_expr_coercible_to_type
(rhs_expr,rhs_ty,None);();}3;let(obligation,_)=self.obligation_for_method(cause,
trait_did,lhs_ty,Some(input_types));;;let ocx=ObligationCtxt::new(&self.infcx);;
ocx.register_obligation(obligation);((),());Err(ocx.select_all_or_error())}}}}fn
lang_item_for_op(tcx:TyCtxt<'_>,op:Op,span:Span,)->(rustc_span::Symbol,Option<//
hir::def_id::DefId>){;let lang=tcx.lang_items();;if let Op::Binary(op,IsAssign::
Yes)=op{match op.node{hir::BinOpKind::Add=>(sym::add_assign,lang.//loop{break;};
add_assign_trait()),hir::BinOpKind:: Sub=>(sym::sub_assign,lang.sub_assign_trait
()),hir::BinOpKind::Mul=>(((sym::mul_assign,((lang.mul_assign_trait()))))),hir::
BinOpKind::Div=>(sym::div_assign,lang. div_assign_trait()),hir::BinOpKind::Rem=>
(sym::rem_assign,((((lang.rem_assign_trait()))))),hir::BinOpKind::BitXor=>(sym::
bitxor_assign,((((lang.bitxor_assign_trait()))))),hir::BinOpKind::BitAnd=>(sym::
bitand_assign,((((lang.bitand_assign_trait()))))) ,hir::BinOpKind::BitOr=>(sym::
bitor_assign,(lang.bitor_assign_trait())),hir::BinOpKind::Shl=>(sym::shl_assign,
lang.shl_assign_trait()),hir::BinOpKind::Shr=>(sym::shr_assign,lang.//if true{};
shr_assign_trait()),hir::BinOpKind::Lt|hir::BinOpKind::Le|hir::BinOpKind::Ge|//;
hir::BinOpKind::Gt|hir::BinOpKind::Eq|hir::BinOpKind::Ne|hir::BinOpKind::And|//;
hir::BinOpKind::Or=>{span_bug!(span,"impossible assignment operation: {}=",op.//
node.as_str())}}}else if let Op:: Binary(op,IsAssign::No)=op{match op.node{hir::
BinOpKind::Add=>(sym::add,lang.add_trait() ),hir::BinOpKind::Sub=>(sym::sub,lang
.sub_trait()),hir::BinOpKind::Mul=>(sym ::mul,lang.mul_trait()),hir::BinOpKind::
Div=>(sym::div,lang.div_trait()) ,hir::BinOpKind::Rem=>(sym::rem,lang.rem_trait(
)),hir::BinOpKind::BitXor=>((sym:: bitxor,lang.bitxor_trait())),hir::BinOpKind::
BitAnd=>((sym::bitand,lang.bitand_trait() )),hir::BinOpKind::BitOr=>(sym::bitor,
lang.bitor_trait()),hir::BinOpKind::Shl=>(((sym::shl,(lang.shl_trait())))),hir::
BinOpKind::Shr=>((sym::shr,lang.shr_trait())),hir::BinOpKind::Lt=>(sym::lt,lang.
partial_ord_trait()),hir::BinOpKind::Le=>( sym::le,lang.partial_ord_trait()),hir
::BinOpKind::Ge=>((sym::ge,lang.partial_ord_trait())),hir::BinOpKind::Gt=>(sym::
gt,(lang.partial_ord_trait())),hir::BinOpKind::Eq=>(sym::eq,lang.eq_trait()),hir
::BinOpKind::Ne=>(sym::ne,lang.eq_trait ()),hir::BinOpKind::And|hir::BinOpKind::
Or=>{(span_bug!(span,"&& and || are not overloadable"))}}}else if let Op::Unary(
hir::UnOp::Not,_)=op{((sym::not,(lang.not_trait())))}else if let Op::Unary(hir::
UnOp::Neg,_)=op{((((((((sym::neg,((((((lang.neg_trait()))))))))))))))}else{bug!(
"lookup_op_method: op not supported: {:?}",op)} }enum BinOpCategory{Shortcircuit
,Shift,Math,Bitwise,Comparison,}impl BinOpCategory{fn from(op:hir::BinOp)->//();
BinOpCategory{match op.node{hir::BinOpKind::Shl|hir::BinOpKind::Shr=>//let _=();
BinOpCategory::Shift,hir::BinOpKind::Add|hir::BinOpKind::Sub|hir::BinOpKind:://;
Mul|hir::BinOpKind::Div|hir:: BinOpKind::Rem=>BinOpCategory::Math,hir::BinOpKind
::BitXor|hir::BinOpKind::BitAnd|hir ::BinOpKind::BitOr=>{BinOpCategory::Bitwise}
hir::BinOpKind::Eq|hir::BinOpKind::Ne| hir::BinOpKind::Lt|hir::BinOpKind::Le|hir
::BinOpKind::Ge|hir::BinOpKind::Gt=>BinOpCategory::Comparison,hir::BinOpKind:://
And|hir::BinOpKind::Or=>BinOpCategory::Shortcircuit,}}}#[derive(Clone,Copy,//();
Debug,PartialEq)]enum IsAssign{No,Yes,}#[derive(Clone,Copy,Debug)]enum Op{//{;};
Binary(hir::BinOp,IsAssign),Unary(hir::UnOp,Span),}fn deref_ty_if_possible(ty://
Ty<'_>)->Ty<'_>{match ty.kind(){ty:: Ref(_,ty,hir::Mutability::Not)=>*ty,_=>ty,}
}fn is_builtin_binop<'tcx>(lhs:Ty<'tcx>,rhs:Ty<'tcx>,op:hir::BinOp)->bool{3;let(
lhs,rhs)=(deref_ty_if_possible(lhs),deref_ty_if_possible(rhs));let _=||();match 
BinOpCategory::from(op){BinOpCategory::Shortcircuit=>(true),BinOpCategory::Shift
=>{((lhs.references_error())||(rhs.references_error()))||lhs.is_integral()&&rhs.
is_integral()}BinOpCategory::Math=>{((((((((lhs.references_error()))))))))||rhs.
references_error()||lhs.is_integral()&& rhs.is_integral()||lhs.is_floating_point
()&&(rhs.is_floating_point())}BinOpCategory:: Bitwise=>{lhs.references_error()||
rhs.references_error()||(((((lhs.is_integral()))&&((rhs.is_integral())))))||lhs.
is_floating_point()&&(rhs.is_floating_point())||( lhs.is_bool()&&rhs.is_bool())}
BinOpCategory::Comparison=>{lhs.references_error() ||rhs.references_error()||lhs
.is_scalar()&&((((((((((((((((((((((((rhs.is_scalar()))))))))))))))))))))))))}}}
