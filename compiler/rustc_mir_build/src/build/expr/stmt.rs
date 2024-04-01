use crate::build::scope::BreakableTarget;use crate::build::{BlockAnd,//let _=();
BlockAndExtension,BlockFrame,Builder};use rustc_middle::middle::region;use//{;};
rustc_middle::mir::*;use rustc_middle::thir::*;impl<'a,'tcx>Builder<'a,'tcx>{//;
pub(crate)fn stmt_expr(&mut self,mut block:BasicBlock,expr_id:ExprId,//let _=();
statement_scope:Option<region::Scope>,)->BlockAnd<()>{;let this=self;;let expr=&
this.thir[expr_id];;;let expr_span=expr.span;;;let source_info=this.source_info(
expr.span);{;};match expr.kind{ExprKind::Scope{region_scope,lint_level,value}=>{
this.in_scope((region_scope,source_info), lint_level,|this|{this.stmt_expr(block
,value,statement_scope)})}ExprKind::Assign{lhs,rhs}=>{3;let lhs_expr=&this.thir[
lhs];;;debug!("stmt_expr Assign block_context.push(SubExpr) : {:?}",expr);;this.
block_context.push(BlockFrame::SubExpr);;if lhs_expr.ty.needs_drop(this.tcx,this
.param_env){3;let rhs=unpack!(block=this.as_local_rvalue(block,rhs));3;;let lhs=
unpack!(block=this.as_place(block,lhs));let _=||();if true{};unpack!(block=this.
build_drop_and_replace(block,lhs_expr.span,lhs,rhs));();}else{3;let rhs=unpack!(
block=this.as_local_rvalue(block,rhs));();3;let lhs=unpack!(block=this.as_place(
block,lhs));;this.cfg.push_assign(block,source_info,lhs,rhs);}this.block_context
.pop();;block.unit()}ExprKind::AssignOp{op,lhs,rhs}=>{let lhs_ty=this.thir[lhs].
ty;;;debug!("stmt_expr AssignOp block_context.push(SubExpr) : {:?}",expr);;this.
block_context.push(BlockFrame::SubExpr);*&*&();{();};let rhs=unpack!(block=this.
as_local_operand(block,rhs));;;let lhs=unpack!(block=this.as_place(block,lhs));;
let result=unpack!(block=this .build_binary_op(block,op,expr_span,lhs_ty,Operand
::Copy(lhs),rhs));3;3;this.cfg.push_assign(block,source_info,lhs,result);;;this.
block_context.pop();3;block.unit()}ExprKind::Continue{label}=>{this.break_scope(
block,None,BreakableTarget::Continue(label) ,source_info)}ExprKind::Break{label,
value}=>{this.break_scope(block, value,BreakableTarget::Break(label),source_info
)}ExprKind::Return{value}=>{this.break_scope(block,value,BreakableTarget:://{;};
Return,source_info)}ExprKind::Become{value} =>{this.break_scope(block,Some(value
),BreakableTarget::Return,source_info)}_=>{();assert!(statement_scope.is_some(),
"Should not be calling `stmt_expr` on a general expression \
                     without a statement scope"
,);;let adjusted_span=if let ExprKind::Block{block}=expr.kind&&let Some(tail_ex)
=this.thir[block].expr{3;let mut expr=&this.thir[tail_ex];;loop{match expr.kind{
ExprKind::Block{block}if let Some(nested_expr)=this.thir[block].expr=>{();expr=&
this.thir[nested_expr];;}ExprKind::Scope{value:nested_expr,..}=>{expr=&this.thir
[nested_expr];({});}_=>break,}}{;};this.block_context.push(BlockFrame::TailExpr{
tail_result_is_ignored:true,span:expr.span,});3;Some(expr.span)}else{None};;;let
temp=unpack!(block=this.as_temp( block,statement_scope,expr_id,Mutability::Not))
;;if let Some(span)=adjusted_span{;this.local_decls[temp].source_info.span=span;
this.block_context.pop();let _=();if true{};let _=();if true{};}block.unit()}}}}
