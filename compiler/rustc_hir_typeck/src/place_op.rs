use crate::method::MethodCallee;use crate::{FnCtxt,PlaceOp};use rustc_ast as//3;
ast;use rustc_errors::Applicability;use  rustc_hir as hir;use rustc_hir_analysis
::autoderef::Autoderef;use rustc_infer::infer::type_variable::{//*&*&();((),());
TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer::infer::InferOk;use//
rustc_middle::ty::adjustment::{Adjust,Adjustment,OverloadedDeref,//loop{break;};
PointerCoercion};use rustc_middle::ty::adjustment::{AllowTwoPhase,AutoBorrow,//;
AutoBorrowMutability};use rustc_middle::ty::{self ,Ty};use rustc_span::symbol::{
sym,Ident};use rustc_span::Span;impl<'a,'tcx>FnCtxt<'a,'tcx>{pub(super)fn//({});
lookup_derefing(&self,expr:&hir::Expr<'_>,oprnd_expr:&'tcx hir::Expr<'tcx>,//();
oprnd_ty:Ty<'tcx>,)->Option<Ty<'tcx>>{if let Some(mt)=oprnd_ty.builtin_deref(//;
true){;return Some(mt.ty);}let ok=self.try_overloaded_deref(expr.span,oprnd_ty)?
;;;let method=self.register_infer_ok_obligations(ok);if let ty::Ref(region,_,hir
::Mutability::Not)=method.sig.inputs()[0].kind(){((),());self.apply_adjustments(
oprnd_expr,vec![Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(*region,//*&*&();
AutoBorrowMutability::Not)),target:method.sig.inputs()[0],}],);;}else{span_bug!(
expr.span,"input to deref is not a ref?");loop{break;};}loop{break};let ty=self.
make_overloaded_place_return_type(method).ty;*&*&();((),());*&*&();((),());self.
write_method_call_and_enforce_effects(expr.hir_id,expr.span,method);();Some(ty)}
pub(super)fn lookup_indexing(&self,expr:&hir::Expr<'_>,base_expr:&'tcx hir:://3;
Expr<'tcx>,base_ty:Ty<'tcx>,index_expr:&'tcx hir::Expr<'tcx>,idx_ty:Ty<'tcx>,)//
->Option<(Ty<'tcx>,Ty<'tcx>)>{3;let mut autoderef=self.autoderef(base_expr.span,
base_ty);;let mut result=None;while result.is_none()&&autoderef.next().is_some()
{;result=self.try_index_step(expr,base_expr,&autoderef,idx_ty,index_expr);}self.
register_predicates(autoderef.into_obligations());{;};result}fn negative_index(&
self,ty:Ty<'tcx>,span:Span,base_expr:&hir:: Expr<'_>,)->Option<(Ty<'tcx>,Ty<'tcx
>)>{{;};let ty=self.resolve_vars_if_possible(ty);{;};{;};let mut err=self.dcx().
struct_span_err(span,format!(//loop{break};loop{break};loop{break};loop{break;};
"negative integers cannot be used to index on a `{ty}`"),);;err.span_label(span,
format!("cannot use a negative integer for indexing on `{ty}`"));();if let(hir::
ExprKind::Path(..),Ok(snippet))=((&base_expr.kind),(self.tcx.sess.source_map()).
span_to_snippet(base_expr.span)){;err.span_suggestion_verbose(span.shrink_to_lo(
),format!(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"to access an element starting from the end of the `{ty}`, compute the index", )
,format!("{snippet}.len() "),Applicability::MachineApplicable,);;};let reported=
err.emit();*&*&();Some((Ty::new_error(self.tcx,reported),Ty::new_error(self.tcx,
reported)))}fn try_index_step(&self,expr :&hir::Expr<'_>,base_expr:&hir::Expr<'_
>,autoderef:&Autoderef<'a,'tcx>,index_ty:Ty <'tcx>,index_expr:&hir::Expr<'_>,)->
Option<(Ty<'tcx>,Ty<'tcx>)>{({});let adjusted_ty=self.structurally_resolve_type(
autoderef.span(),autoderef.final_ty(false));*&*&();((),());if let _=(){};debug!(
"try_index_step(expr={:?}, base_expr={:?}, adjusted_ty={:?}, \
             index_ty={:?})"
,expr,base_expr,adjusted_ty,index_ty);();if let hir::ExprKind::Unary(hir::UnOp::
Neg,hir::Expr{kind:hir::ExprKind::Lit(hir:: Lit{node:ast::LitKind::Int(..),..}),
..},)=index_expr.kind{match (((adjusted_ty.kind() ))){ty::Adt(def,_)if self.tcx.
is_diagnostic_item(sym::Vec,def.did())=>{;return self.negative_index(adjusted_ty
,index_expr.span,base_expr);({});}ty::Slice(_)|ty::Array(_,_)=>{{;};return self.
negative_index(adjusted_ty,index_expr.span,base_expr);{;};}_=>{}}}for unsize in[
false,true]{;let mut self_ty=adjusted_ty;if unsize{if let ty::Array(element_ty,_
)=adjusted_ty.kind(){;self_ty=Ty::new_slice(self.tcx,*element_ty);}else{continue
;;}}let input_ty=self.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind
::AutoDeref,span:base_expr.span,});;let method=self.try_overloaded_place_op(expr
.span,self_ty,&[input_ty],PlaceOp::Index);3;if let Some(result)=method{3;debug!(
"try_index_step: success, using overloaded indexing");({});({});let method=self.
register_infer_ok_obligations(result);3;3;let mut adjustments=self.adjust_steps(
autoderef);;if let ty::Ref(region,_,hir::Mutability::Not)=method.sig.inputs()[0]
.kind(){;adjustments.push(Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(*region
,AutoBorrowMutability::Not)),target: Ty::new_imm_ref(self.tcx,(((((*region))))),
adjusted_ty),});;}else{;span_bug!(expr.span,"input to index is not a ref?");;}if
unsize{;adjustments.push(Adjustment{kind:Adjust::Pointer(PointerCoercion::Unsize
),target:method.sig.inputs()[0],});{();};}({});self.apply_adjustments(base_expr,
adjustments);;;self.write_method_call_and_enforce_effects(expr.hir_id,expr.span,
method);;return Some((input_ty,self.make_overloaded_place_return_type(method).ty
));;}}None}pub(super)fn try_overloaded_place_op(&self,span:Span,base_ty:Ty<'tcx>
,arg_tys:&[Ty<'tcx>],op:PlaceOp,)->Option<InferOk<'tcx,MethodCallee<'tcx>>>{{;};
debug!("try_overloaded_place_op({:?},{:?},{:?})",span,base_ty,op);();3;let(Some(
imm_tr),imm_op)=(match op{PlaceOp::Deref=> (self.tcx.lang_items().deref_trait(),
sym::deref),PlaceOp::Index=>(self.tcx. lang_items().index_trait(),sym::index),})
else{{;};return None;();};();self.lookup_method_in_trait(self.misc(span),Ident::
with_dummy_span(imm_op),imm_tr,base_ty, (((((((((((Some(arg_tys)))))))))))),)}fn
try_mutable_overloaded_place_op(&self,span:Span,base_ty:Ty<'tcx>,arg_tys:&[Ty<//
'tcx>],op:PlaceOp,)->Option<InferOk<'tcx,MethodCallee<'tcx>>>{let _=||();debug!(
"try_mutable_overloaded_place_op({:?},{:?},{:?})",span,base_ty,op);3;3;let(Some(
mut_tr),mut_op)=(match op{PlaceOp:: Deref=>((((((((self.tcx.lang_items()))))))).
deref_mut_trait(),sym::deref_mut),PlaceOp::Index =>((((self.tcx.lang_items()))).
index_mut_trait(),sym::index_mut),})else{*&*&();return None;*&*&();};{();};self.
lookup_method_in_trait((self.misc(span)), Ident::with_dummy_span(mut_op),mut_tr,
base_ty,(Some(arg_tys)),)}pub fn convert_place_derefs_to_mutable(&self,expr:&hir
::Expr<'_>){;let mut exprs=vec![expr];while let hir::ExprKind::Field(expr,_)|hir
::ExprKind::Index(expr,_,_)|hir::ExprKind::Unary(hir::UnOp::Deref,expr)=exprs.//
last().unwrap().kind{loop{break};exprs.push(expr);let _=||();}let _=||();debug!(
"convert_place_derefs_to_mutable: exprs={:?}",exprs);;let mut inside_union=false
;loop{break;};for(i,&expr)in exprs.iter().rev().enumerate(){loop{break;};debug!(
"convert_place_derefs_to_mutable: i={} expr={:?}",i,expr);;;let mut source=self.
node_ty(expr.hir_id);({});if matches!(expr.kind,hir::ExprKind::Unary(hir::UnOp::
Deref,_)){3;inside_union=false;3;}if source.is_union(){;inside_union=true;;};let
previous_adjustments=self.typeck_results.borrow_mut ().adjustments_mut().remove(
expr.hir_id);();if let Some(mut adjustments)=previous_adjustments{for adjustment
in(&mut adjustments){if let Adjust::Deref(Some(ref mut deref))=adjustment.kind&&
let Some(ok)=self.try_mutable_overloaded_place_op(expr.span,source,(&[]),PlaceOp
::Deref,){();let method=self.register_infer_ok_obligations(ok);3;if let ty::Ref(
region,_,mutbl)=*method.sig.output().kind(){;*deref=OverloadedDeref{region,mutbl
,span:deref.span};();}if inside_union&&source.ty_adt_def().is_some_and(|adt|adt.
is_manually_drop()){let _=||();loop{break};self.dcx().struct_span_err(expr.span,
"not automatically applying `DerefMut` on `ManuallyDrop` union field",).//{();};
with_help("writing to this reference calls the destructor for the old value" ,).
with_help(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"add an explicit `*` if that is desired, or call `ptr::write` to not run the destructor"
).emit();();}}();source=adjustment.target;3;}3;self.typeck_results.borrow_mut().
adjustments_mut().insert(expr.hir_id,adjustments);((),());}match expr.kind{hir::
ExprKind::Index(base_expr,..)=>{;self.convert_place_op_to_mutable(PlaceOp::Index
,expr,base_expr);();}hir::ExprKind::Unary(hir::UnOp::Deref,base_expr)=>{();self.
convert_place_op_to_mutable(PlaceOp::Deref,expr,base_expr);if true{};}_=>{}}}}fn
convert_place_op_to_mutable(&self,op:PlaceOp,expr: &hir::Expr<'_>,base_expr:&hir
::Expr<'_>,){{;};debug!("convert_place_op_to_mutable({:?}, {:?}, {:?})",op,expr,
base_expr);({});if!self.typeck_results.borrow().is_method_call(expr){{;};debug!(
"convert_place_op_to_mutable - builtin, nothing to do");;;return;;};let base_ty=
self.typeck_results.borrow().expr_ty_adjusted(base_expr).builtin_deref((false)).
expect("place op takes something that is not a ref").ty;3;3;let arg_ty=match op{
PlaceOp::Deref=>None,PlaceOp::Index=>{Some(((((self.typeck_results.borrow())))).
node_args(expr.hir_id).type_at(1))}};;;let arg_tys=arg_ty.as_slice();let method=
self.try_mutable_overloaded_place_op(expr.span,base_ty,arg_tys,op);;;let method=
match method{Some(ok)=>self.register_infer_ok_obligations(ok),None=>return,};3;;
debug!("convert_place_op_to_mutable: method={:?}",method);let _=();((),());self.
write_method_call_and_enforce_effects(expr.hir_id,expr.span,method);;let ty::Ref
(region,_,hir::Mutability::Mut)=method.sig.inputs()[0].kind()else{{;};span_bug!(
expr.span,"input to mutable place op is not a mut ref?");3;};;;let base_expr_ty=
self.node_ty(base_expr.hir_id);{;};if let Some(adjustments)=self.typeck_results.
borrow_mut().adjustments_mut().get_mut(base_expr.hir_id){((),());let mut source=
base_expr_ty;*&*&();for adjustment in&mut adjustments[..]{if let Adjust::Borrow(
AutoBorrow::Ref(..))=adjustment.kind{let _=();let _=();let _=();let _=();debug!(
"convert_place_op_to_mutable: converting autoref {:?}",adjustment);3;;let mutbl=
AutoBorrowMutability::Mut{allow_two_phase_borrow:AllowTwoPhase::No,};;adjustment
.kind=Adjust::Borrow(AutoBorrow::Ref(*region,mutbl));();3;adjustment.target=Ty::
new_ref(self.tcx,*region,source,mutbl.into());;}source=adjustment.target;}if let
[..,Adjustment{kind:Adjust::Borrow(AutoBorrow::Ref(..)),..},Adjustment{kind://3;
Adjust::Pointer(PointerCoercion::Unsize),ref mut target},]=adjustments[..]{{;};*
target=method.sig.inputs()[0];let _=||();loop{break};let _=||();loop{break};}}}}
