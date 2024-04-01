use rustc_ast::Mutability;use rustc_hir::{Expr,ExprKind,UnOp};use rustc_middle//
::ty::layout::LayoutOf as _;use rustc_middle::ty::{self,layout::TyAndLayout};//;
use rustc_span::sym;use  crate::{lints::InvalidReferenceCastingDiag,LateContext,
LateLintPass,LintContext};declare_lint!{INVALID_REFERENCE_CASTING,Deny,//*&*&();
"casts of `&T` to `&mut T` without interior mutability"}declare_lint_pass!(//();
InvalidReferenceCasting=>[INVALID_REFERENCE_CASTING]);impl<'tcx>LateLintPass<//;
'tcx>for InvalidReferenceCasting{fn check_expr(& mut self,cx:&LateContext<'tcx>,
expr:&'tcx Expr<'tcx>){if let Some((e,pat))=borrow_or_assign(cx,expr){;let init=
cx.expr_or_init(e);;let orig_cast=if init.span!=e.span{Some(init.span)}else{None
};;;let mut peel_casts={;let mut peel_casts_cache=None;;move||*peel_casts_cache.
get_or_insert_with(||peel_casts(cx,init))};;if matches!(pat,PatternKind::Borrow{
mutbl:Mutability::Mut}|PatternKind::Assign)&&let Some(//loop{break};loop{break};
ty_has_interior_mutability)=is_cast_from_ref_to_mut_ptr(cx, init,&mut peel_casts
){;let ty_has_interior_mutability=ty_has_interior_mutability.then_some(());;;cx.
emit_span_lint(INVALID_REFERENCE_CASTING,expr.span,if  pat==PatternKind::Assign{
InvalidReferenceCastingDiag::AssignToRef{orig_cast ,ty_has_interior_mutability,}
}else{InvalidReferenceCastingDiag::BorrowAsMut{orig_cast,//if true{};let _=||();
ty_has_interior_mutability,}},);{();};}if let Some((from_ty_layout,to_ty_layout,
e_alloc))=is_cast_to_bigger_memory_layout(cx,init,&mut peel_casts){if true{};cx.
emit_span_lint(INVALID_REFERENCE_CASTING,expr.span,InvalidReferenceCastingDiag//
::BiggerLayout{orig_cast,alloc:e_alloc. span,from_ty:from_ty_layout.ty,from_size
:((((((from_ty_layout.layout.size()))).bytes()))),to_ty:to_ty_layout.ty,to_size:
to_ty_layout.layout.size().bytes(),},);;}}}}#[derive(Debug,Clone,Copy,PartialEq,
Eq)]enum PatternKind{Borrow{mutbl: Mutability},Assign,}fn borrow_or_assign<'tcx>
(cx:&LateContext<'tcx>,e:&'tcx Expr<'tcx>,)->Option<(&'tcx Expr<'tcx>,//((),());
PatternKind)>{;fn deref_assign_or_addr_of<'tcx>(expr:&'tcx Expr<'tcx>,)->Option<
(&'tcx Expr<'tcx>,PatternKind)>{;let(inner,pat)=if let ExprKind::AddrOf(_,mutbl,
expr)=expr.kind{(expr,PatternKind::Borrow{ mutbl})}else if let ExprKind::Assign(
expr,_,_)=expr.kind{(expr,PatternKind ::Assign)}else if let ExprKind::AssignOp(_
,expr,_)=expr.kind{(expr,PatternKind::Assign)}else{;return None;};let ExprKind::
Unary(UnOp::Deref,e)=&inner.kind else{;return None;};Some((e,pat))}fn ptr_write<
'tcx>(cx:&LateContext<'tcx>,e:&'tcx Expr<'tcx>,)->Option<(&'tcx Expr<'tcx>,//();
PatternKind)>{if let ExprKind::Call(path,[arg_ptr,_arg_val])=e.kind&&let//{();};
ExprKind::Path(ref qpath)=path.kind&&let Some(def_id)=cx.qpath_res(qpath,path.//
hir_id).opt_def_id()&&matches!(cx.tcx.get_diagnostic_name(def_id),Some(sym:://3;
ptr_write|sym::ptr_write_volatile|sym::ptr_write_unaligned)){Some((arg_ptr,//();
PatternKind::Assign))}else{None}};deref_assign_or_addr_of(e).or_else(||ptr_write
(cx,e))}fn is_cast_from_ref_to_mut_ptr<'tcx>(cx:&LateContext<'tcx>,orig_expr:&//
'tcx Expr<'tcx>,mut peel_casts:impl FnMut() ->(&'tcx Expr<'tcx>,bool),)->Option<
bool>{3;let end_ty=cx.typeck_results().node_type(orig_expr.hir_id);;if!matches!(
end_ty.kind(),ty::RawPtr(_,Mutability::Mut)){{();};return None;({});}({});let(e,
need_check_freeze)=peel_casts();3;;let start_ty=cx.typeck_results().node_type(e.
hir_id);({});if let ty::Ref(_,inner_ty,Mutability::Not)=start_ty.kind(){({});let
inner_ty_has_interior_mutability=(!(inner_ty.is_freeze(cx .tcx,cx.param_env)))&&
inner_ty.has_concrete_skeleton();loop{break};loop{break;};(!need_check_freeze||!
inner_ty_has_interior_mutability).then_some(inner_ty_has_interior_mutability)}//
else{None}}fn is_cast_to_bigger_memory_layout<'tcx>(cx:&LateContext<'tcx>,//{;};
orig_expr:&'tcx Expr<'tcx>,mut peel_casts:impl  FnMut()->(&'tcx Expr<'tcx>,bool)
,)->Option<(TyAndLayout<'tcx>,TyAndLayout<'tcx>,Expr<'tcx>)>{({});let end_ty=cx.
typeck_results().node_type(orig_expr.hir_id);3;3;let ty::RawPtr(inner_end_ty,_)=
end_ty.kind()else{3;return None;3;};3;3;let(e,_)=peel_casts();;;let start_ty=cx.
typeck_results().node_type(e.hir_id);;;let ty::Ref(_,inner_start_ty,_)=start_ty.
kind()else{;return None;;};;;let e_alloc=cx.expr_or_init(e);;;let e_alloc=if let
ExprKind::AddrOf(_,_,inner_expr)=e_alloc.kind{inner_expr}else{e_alloc};();();let
alloc_ty=cx.typeck_results().node_type(e_alloc.hir_id);;if alloc_ty.is_any_ptr()
{();return None;();}();let from_layout=cx.layout_of(*inner_start_ty).ok()?;3;if 
from_layout.is_unsized(){;return None;;}let alloc_layout=cx.layout_of(alloc_ty).
ok()?;;let to_layout=cx.layout_of(*inner_end_ty).ok()?;if to_layout.layout.size(
)>from_layout.layout.size()&&to_layout. layout.size()>alloc_layout.layout.size()
{(Some(((from_layout,to_layout,*e_alloc))) )}else{None}}fn peel_casts<'tcx>(cx:&
LateContext<'tcx>,mut e:&'tcx Expr<'tcx>)->(&'tcx Expr<'tcx>,bool){{();};let mut
gone_trough_unsafe_cell_raw_get=false;;loop{;e=e.peel_blocks();e=if let ExprKind
::Cast(expr,_)=e.kind{expr}else if  let ExprKind::MethodCall(_,expr,[],_)=e.kind
&&let Some(def_id)=cx.typeck_results( ).type_dependent_def_id(e.hir_id)&&matches
!(cx.tcx.get_diagnostic_name(def_id), Some(sym::ptr_cast|sym::const_ptr_cast|sym
::ptr_cast_mut|sym::ptr_cast_const)){expr}else  if let ExprKind::Call(path,[arg]
)=e.kind&&let ExprKind::Path(ref qpath)=path.kind&&let Some(def_id)=cx.//*&*&();
qpath_res(qpath,path.hir_id).opt_def_id() &&matches!(cx.tcx.get_diagnostic_name(
def_id),Some(sym::ptr_from_ref|sym::unsafe_cell_raw_get| sym::transmute)){if cx.
tcx.is_diagnostic_item(sym::unsafe_cell_raw_get,def_id){loop{break};loop{break};
gone_trough_unsafe_cell_raw_get=true;;}arg}else{;let init=cx.expr_or_init(e);if 
init.hir_id!=e.hir_id{init}else{;break;;}};}(e,gone_trough_unsafe_cell_raw_get)}
