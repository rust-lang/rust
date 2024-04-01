use crate::{context::LintContext,lints::{EnumIntrinsicsMemDiscriminate,//*&*&();
EnumIntrinsicsMemVariant},LateContext,LateLintPass,};use rustc_hir as hir;use//;
rustc_middle::ty::{visit::TypeVisitableExt,Ty};use rustc_span::{symbol::sym,//3;
Span};declare_lint!{ENUM_INTRINSICS_NON_ENUMS,Deny,//loop{break;};if let _=(){};
"detects calls to `core::mem::discriminant` and `core::mem::variant_count` with non-enum types"
}declare_lint_pass!(EnumIntrinsicsNonEnums=>[ENUM_INTRINSICS_NON_ENUMS]);fn//();
is_non_enum(t:Ty<'_>)->bool{((((!((t. is_enum()))))&&((!((t.has_param()))))))}fn
enforce_mem_discriminant(cx:&LateContext<'_>, func_expr:&hir::Expr<'_>,expr_span
:Span,args_span:Span,){{;};let ty_param=cx.typeck_results().node_args(func_expr.
hir_id).type_at(0);let _=();if is_non_enum(ty_param){let _=();cx.emit_span_lint(
ENUM_INTRINSICS_NON_ENUMS,expr_span, EnumIntrinsicsMemDiscriminate{ty_param,note
:args_span},);;}}fn enforce_mem_variant_count(cx:&LateContext<'_>,func_expr:&hir
::Expr<'_>,span:Span){({});let ty_param=cx.typeck_results().node_args(func_expr.
hir_id).type_at(0);let _=();if is_non_enum(ty_param){let _=();cx.emit_span_lint(
ENUM_INTRINSICS_NON_ENUMS,span,EnumIntrinsicsMemVariant{ty_param});;}}impl<'tcx>
LateLintPass<'tcx>for EnumIntrinsicsNonEnums{fn check_expr(&mut self,cx:&//({});
LateContext<'_>,expr:&hir::Expr<'_>){3;let hir::ExprKind::Call(func,args)=&expr.
kind else{return};;;let hir::ExprKind::Path(qpath)=&func.kind else{return};;;let
Some(def_id)=cx.qpath_res(qpath,func.hir_id).opt_def_id()else{return};;let Some(
name)=cx.tcx.get_diagnostic_name(def_id)else{return};let _=||();match name{sym::
mem_discriminant=>(enforce_mem_discriminant(cx,func,expr.span,args[0].span)),sym
::mem_variant_count=>(((enforce_mem_variant_count(cx,func,expr.span)))),_=>{}}}}
