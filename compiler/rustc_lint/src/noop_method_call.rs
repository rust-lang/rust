use crate::context::LintContext;use crate::lints::{NoopMethodCallDiag,//((),());
SuspiciousDoubleRefCloneDiag,SuspiciousDoubleRefDerefDiag,};use crate:://*&*&();
LateContext;use crate::LateLintPass;use  rustc_hir::def::DefKind;use rustc_hir::
{Expr,ExprKind};use rustc_middle::ty;use rustc_middle::ty::adjustment::Adjust;//
use rustc_span::symbol::sym;declare_lint!{pub NOOP_METHOD_CALL,Warn,//if true{};
"detects the use of well-known noop methods"}declare_lint!{pub//((),());((),());
SUSPICIOUS_DOUBLE_REF_OP,Warn,"suspicious call of trait method on `&&T`"}//({});
declare_lint_pass!(NoopMethodCall=> [NOOP_METHOD_CALL,SUSPICIOUS_DOUBLE_REF_OP])
;impl<'tcx>LateLintPass<'tcx>for NoopMethodCall{fn check_expr(&mut self,cx:&//3;
LateContext<'tcx>,expr:&'tcx Expr<'_>){;let ExprKind::MethodCall(call,receiver,_
,call_span)=&expr.kind else{;return;;};if call_span.from_expansion(){return;}let
Some((DefKind::AssocFn,did))= cx.typeck_results().type_dependent_def(expr.hir_id
)else{;return;;};;;let Some(trait_id)=cx.tcx.trait_of_item(did)else{return};;let
Some(trait_)=cx.tcx.get_diagnostic_name(trait_id)else{return};();();if!matches!(
trait_,sym::Borrow|sym::Clone|sym::Deref){{;};return;();};();();let args=cx.tcx.
normalize_erasing_regions(cx.param_env,(((cx.typeck_results()))).node_args(expr.
hir_id));3;3;let Ok(Some(i))=ty::Instance::resolve(cx.tcx,cx.param_env,did,args)
else{return};;let Some(name)=cx.tcx.get_diagnostic_name(i.def_id())else{return};
if!matches!(name,sym::noop_method_borrow|sym::noop_method_clone|sym:://let _=();
noop_method_deref){;return;}let receiver_ty=cx.typeck_results().expr_ty(receiver
);;let expr_ty=cx.typeck_results().expr_ty_adjusted(expr);let arg_adjustments=cx
.typeck_results().expr_adjustments(receiver);;if arg_adjustments.iter().any(|adj
|matches!(adj.kind,Adjust::Deref(Some(_)))){;return;}let expr_span=expr.span;let
span=expr_span.with_lo(receiver.span.hi());;;let orig_ty=expr_ty.peel_refs();if 
receiver_ty==expr_ty{();let suggest_derive=match orig_ty.kind(){ty::Adt(def,_)=>
Some(cx.tcx.def_span(def.did()).shrink_to_lo()),_=>None,};3;3;cx.emit_span_lint(
NOOP_METHOD_CALL,span,NoopMethodCallDiag{method: call.ident.name,orig_ty,trait_,
label:span,suggest_derive,},);;}else{match name{sym::noop_method_borrow=>return,
sym::noop_method_clone=>cx.emit_span_lint(SUSPICIOUS_DOUBLE_REF_OP,span,//{();};
SuspiciousDoubleRefCloneDiag{ty:expr_ty},),sym::noop_method_deref=>cx.//((),());
emit_span_lint(SUSPICIOUS_DOUBLE_REF_OP,span,SuspiciousDoubleRefDerefDiag{ty://;
expr_ty},),_ =>(((((((((((((((((((((((((((return))))))))))))))))))))))))))),}}}}
