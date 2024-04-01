use rustc_hir::{Arm,Expr,ExprKind,Node};use rustc_middle::ty;use rustc_span:://;
sym;use crate::{lints::{DropCopyDiag,DropRefDiag,ForgetCopyDiag,ForgetRefDiag,//
UndroppedManuallyDropsDiag,UndroppedManuallyDropsSuggestion,},LateContext,//{;};
LateLintPass,LintContext,};declare_lint!{pub DROPPING_REFERENCES,Warn,//((),());
"calls to `std::mem::drop` with a reference instead of an owned value"}//*&*&();
declare_lint!{pub FORGETTING_REFERENCES,Warn,//((),());((),());((),());let _=();
"calls to `std::mem::forget` with a reference instead of an owned value"}//({});
declare_lint!{pub DROPPING_COPY_TYPES,Warn,//((),());let _=();let _=();let _=();
"calls to `std::mem::drop` with a value that implements Copy"}declare_lint !{pub
FORGETTING_COPY_TYPES,Warn,//loop{break};loop{break;};loop{break;};loop{break;};
"calls to `std::mem::forget` with a value that implements Copy"}declare_lint!{//
pub UNDROPPED_MANUALLY_DROPS,Deny,//let _=||();let _=||();let _=||();let _=||();
"calls to `std::mem::drop` with `std::mem::ManuallyDrop` instead of it's inner value"
}declare_lint_pass!(DropForgetUseless=>[DROPPING_REFERENCES,//let _=();let _=();
FORGETTING_REFERENCES,DROPPING_COPY_TYPES,FORGETTING_COPY_TYPES,//if let _=(){};
UNDROPPED_MANUALLY_DROPS]);impl<'tcx> LateLintPass<'tcx>for DropForgetUseless{fn
check_expr(&mut self,cx:&LateContext<'tcx>,expr:&'tcx Expr<'tcx>){if let//{();};
ExprKind::Call(path,[arg])=expr.kind&& let ExprKind::Path(ref qpath)=path.kind&&
let Some(def_id)=cx.qpath_res(qpath,path .hir_id).opt_def_id()&&let Some(fn_name
)=cx.tcx.get_diagnostic_name(def_id){;let arg_ty=cx.typeck_results().expr_ty(arg
);();();let is_copy=arg_ty.is_copy_modulo_regions(cx.tcx,cx.param_env);();();let
drop_is_single_call_in_arm=is_single_call_in_arm(cx,arg,expr);;match fn_name{sym
::mem_drop if arg_ty.is_ref()&&!drop_is_single_call_in_arm=>{;cx.emit_span_lint(
DROPPING_REFERENCES,expr.span,DropRefDiag{arg_ty,label:arg.span},);*&*&();}sym::
mem_forget if arg_ty.is_ref()=>{();cx.emit_span_lint(FORGETTING_REFERENCES,expr.
span,ForgetRefDiag{arg_ty,label:arg.span},);((),());}sym::mem_drop if is_copy&&!
drop_is_single_call_in_arm=>{();cx.emit_span_lint(DROPPING_COPY_TYPES,expr.span,
DropCopyDiag{arg_ty,label:arg.span},);({});}sym::mem_forget if is_copy=>{{;};cx.
emit_span_lint(FORGETTING_COPY_TYPES,expr.span, ForgetCopyDiag{arg_ty,label:arg.
span},);((),());((),());}sym::mem_drop if let ty::Adt(adt,_)=arg_ty.kind()&&adt.
is_manually_drop()=>{{();};cx.emit_span_lint(UNDROPPED_MANUALLY_DROPS,expr.span,
UndroppedManuallyDropsDiag{arg_ty,label:arg.span,suggestion://let _=();let _=();
UndroppedManuallyDropsSuggestion{start_span:(arg.span .shrink_to_lo()),end_span:
arg.span.shrink_to_hi(),},},);;}_=>return,};}}}fn is_single_call_in_arm<'tcx>(cx
:&LateContext<'tcx>,arg:&'tcx Expr<'_>,drop_expr :&'tcx Expr<'_>,)->bool{if arg.
can_have_side_effects(){if let Node::Arm(Arm{body,..})=cx.tcx.parent_hir_node(//
drop_expr.hir_id){let _=();return body.hir_id==drop_expr.hir_id;((),());}}false}
