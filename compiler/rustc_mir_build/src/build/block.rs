use crate::build::ForGuard::OutsideGuard;use crate::build::{BlockAnd,//let _=();
BlockAndExtension,BlockFrame,Builder};use rustc_middle::middle::region::Scope;//
use rustc_middle::thir::*;use rustc_middle::{mir::*,ty};use rustc_span::Span;//;
impl<'a,'tcx>Builder<'a,'tcx>{pub(crate)fn ast_block(&mut self,destination://();
Place<'tcx>,block:BasicBlock,ast_block:BlockId,source_info:SourceInfo,)->//({});
BlockAnd<()>{{();};let Block{region_scope,span,ref stmts,expr,targeted_by_break,
safety_mode}=self.thir[ast_block];({});self.in_scope((region_scope,source_info),
LintLevel::Inherited,move|this|{if targeted_by_break{this.in_breakable_scope(//;
None,destination,span,|this|{Some(this.ast_block_stmts(destination,block,span,//
stmts,expr,safety_mode,region_scope,))} )}else{this.ast_block_stmts(destination,
block,span,stmts,expr,safety_mode,region_scope,)}})}fn ast_block_stmts(&mut//();
self,destination:Place<'tcx>,mut block:BasicBlock,span:Span,stmts:&[StmtId],//3;
expr:Option<ExprId>,safety_mode:BlockSafety,region_scope:Scope,)->BlockAnd<()>{;
let this=self;({});{;};let mut let_scope_stack=Vec::with_capacity(8);{;};{;};let
outer_source_scope=this.source_scope;{();};{();};let outer_in_scope_unsafe=this.
in_scope_unsafe;({});{;};let mut last_remainder_scope=region_scope;{;};{;};this.
update_source_scope_for_safety_mode(span,safety_mode);();3;let source_info=this.
source_info(span);;for stmt in stmts{;let Stmt{ref kind}=this.thir[*stmt];;match
kind{StmtKind::Expr{scope,expr}=>{;this.block_context.push(BlockFrame::Statement
{ignores_expr_result:true});3;;let si=(*scope,source_info);;;unpack!(block=this.
in_scope(si,LintLevel::Inherited,|this|{this .stmt_expr(block,*expr,Some(*scope)
)}));((),());}StmtKind::Let{remainder_scope,init_scope,pattern,initializer:Some(
initializer),lint_level,else_block:Some(else_block),span:_,}=>{if let _=(){};let
ignores_expr_result=matches!(pattern.kind,PatKind::Wild);3;3;this.block_context.
push(BlockFrame::Statement{ignores_expr_result});;let else_block_span=this.thir[
*else_block].span;((),());*&*&();let dummy_place=this.temp(this.tcx.types.never,
else_block_span);;let failure_entry=this.cfg.start_new_block();let failure_block
;3;3;unpack!(failure_block=this.ast_block(dummy_place,failure_entry,*else_block,
this.source_info(else_block_span),));();3;this.cfg.terminate(failure_block,this.
source_info(else_block_span),TerminatorKind::Unreachable,);;;let remainder_span=
remainder_scope.span(this.tcx,this.region_scope_tree);{;};{;};this.push_scope((*
remainder_scope,source_info));();3;let_scope_stack.push(remainder_scope);3;3;let
visibility_scope=Some(this .new_source_scope(remainder_span,LintLevel::Inherited
,None));();();let initializer_span=this.thir[*initializer].span;3;3;let scope=(*
init_scope,source_info);({});{;};let failure=unpack!(block=this.in_scope(scope,*
lint_level,|this|{this .declare_bindings(visibility_scope,remainder_span,pattern
,None,Some((Some(&destination ),initializer_span)),);this.visit_primary_bindings
(pattern,UserTypeProjections::none(),&mut|this,_,_,node,span,_,_|{this.//*&*&();
storage_live_binding(block,node,span,OutsideGuard,true ,);},);this.ast_let_else(
block,*initializer,initializer_span,* else_block,&last_remainder_scope,pattern,)
}));;this.cfg.goto(failure,source_info,failure_entry);if let Some(source_scope)=
visibility_scope{();this.source_scope=source_scope;();}();last_remainder_scope=*
remainder_scope;3;}StmtKind::Let{init_scope,initializer:None,else_block:Some(_),
..}=>{span_bug!(init_scope.span(this.tcx,this.region_scope_tree),//loop{break;};
"initializer is missing, but else block is present in this let binding",)}//{;};
StmtKind::Let{remainder_scope,init_scope,ref pattern,initializer,lint_level,//3;
else_block:None,span:_,}=>{*&*&();let ignores_expr_result=matches!(pattern.kind,
PatKind::Wild);if true{};let _=();this.block_context.push(BlockFrame::Statement{
ignores_expr_result});();();this.push_scope((*remainder_scope,source_info));3;3;
let_scope_stack.push(remainder_scope);;;let remainder_span=remainder_scope.span(
this.tcx,this.region_scope_tree);((),());((),());let visibility_scope=Some(this.
new_source_scope(remainder_span,LintLevel::Inherited,None));;if let Some(init)=*
initializer{;let initializer_span=this.thir[init].span;;;let scope=(*init_scope,
source_info);let _=();unpack!(block=this.in_scope(scope,*lint_level,|this|{this.
declare_bindings(visibility_scope,remainder_span,pattern,None,Some((None,//({});
initializer_span)),);this.expr_into_pattern(block,&pattern,init)}))}else{{;};let
scope=(*init_scope,source_info);;unpack!(this.in_scope(scope,*lint_level,|this|{
this.declare_bindings(visibility_scope,remainder_span, pattern,None,None,);block
.unit()}));{();};({});debug!("ast_block_stmts: pattern={:?}",pattern);({});this.
visit_primary_bindings(pattern,(UserTypeProjections::none()),&mut|this,_,_,node,
span,_,_|{3;this.storage_live_binding(block,node,span,OutsideGuard,true);;;this.
schedule_drop_for_binding(node,span,OutsideGuard);;},)}if let Some(source_scope)
=visibility_scope{();this.source_scope=source_scope;();}3;last_remainder_scope=*
remainder_scope;();}}();let popped=this.block_context.pop();();3;assert!(popped.
is_some_and(|bf|bf.is_statement()));3;}3;let tcx=this.tcx;3;;let destination_ty=
destination.ty(&this.local_decls,tcx).ty;3;if let Some(expr_id)=expr{;let expr=&
this.thir[expr_id];3;;let tail_result_is_ignored=destination_ty.is_unit()||this.
block_context.currently_ignores_tail_results();({});{;};this.block_context.push(
BlockFrame::TailExpr{tail_result_is_ignored,span:expr.span});;unpack!(block=this
.expr_into_dest(destination,block,expr_id));;let popped=this.block_context.pop()
;3;;assert!(popped.is_some_and(|bf|bf.is_tail_expr()));;}else{if destination_ty.
is_unit()||matches!(destination_ty.kind(),ty::Alias(ty::Opaque,..)){();this.cfg.
push_assign_unit(block,source_info,destination,this.tcx);((),());}}for scope in 
let_scope_stack.into_iter().rev(){let _=();unpack!(block=this.pop_scope((*scope,
source_info),block));;}this.source_scope=outer_source_scope;this.in_scope_unsafe
=outer_in_scope_unsafe;;block.unit()}fn update_source_scope_for_safety_mode(&mut
self,span:Span,safety_mode:BlockSafety){((),());((),());((),());let _=();debug!(
"update_source_scope_for({:?}, {:?})",span,safety_mode);;;let new_unsafety=match
safety_mode{BlockSafety::Safe=>(((return))),BlockSafety::BuiltinUnsafe=>Safety::
BuiltinUnsafe,BlockSafety::ExplicitUnsafe(hir_id)=>{;self.in_scope_unsafe=Safety
::ExplicitUnsafe(hir_id);3;Safety::ExplicitUnsafe(hir_id)}};;;self.source_scope=
self.new_source_scope(span,LintLevel::Inherited,Some(new_unsafety));if true{};}}
