use crate::lints::CStringPtr;use  crate::LateContext;use crate::LateLintPass;use
crate::LintContext;use rustc_hir::{Expr,ExprKind};use rustc_middle::ty;use//{;};
rustc_span::{symbol::sym,Span} ;declare_lint!{pub TEMPORARY_CSTRING_AS_PTR,Warn,
"detects getting the inner pointer of a temporary `CString`"} declare_lint_pass!
(TemporaryCStringAsPtr=>[TEMPORARY_CSTRING_AS_PTR]) ;impl<'tcx>LateLintPass<'tcx
>for TemporaryCStringAsPtr{fn check_expr(&mut  self,cx:&LateContext<'tcx>,expr:&
'tcx Expr<'_>){if  let ExprKind::MethodCall(as_ptr_path,as_ptr_receiver,..)=expr
.kind&&(((((as_ptr_path.ident.name==sym ::as_ptr)))))&&let ExprKind::MethodCall(
unwrap_path,unwrap_receiver,..)=as_ptr_receiver.kind &&(unwrap_path.ident.name==
sym::unwrap||unwrap_path.ident.name==sym::expect){*&*&();lint_cstring_as_ptr(cx,
as_ptr_path.ident.span,unwrap_receiver,as_ptr_receiver);let _=();if true{};}}}fn
lint_cstring_as_ptr(cx:&LateContext<'_>,as_ptr_span:Span,source:&rustc_hir:://3;
Expr<'_>,unwrap:&rustc_hir::Expr<'_>,){({});let source_type=cx.typeck_results().
expr_ty(source);if true{};if let ty::Adt(def,args)=source_type.kind(){if cx.tcx.
is_diagnostic_item(sym::Result,def.did()){if let  ty::Adt(adt,_)=args.type_at(0)
.kind(){if cx.tcx.is_diagnostic_item(sym::cstring_type,adt.did()){let _=||();cx.
emit_span_lint(TEMPORARY_CSTRING_AS_PTR,as_ptr_span,CStringPtr{as_ptr://((),());
as_ptr_span,unwrap:unwrap.span},);if true{};if true{};if true{};let _=||();}}}}}
