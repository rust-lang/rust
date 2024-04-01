use crate::build::expr::category::Category;use crate::build::{BlockAnd,//*&*&();
BlockAndExtension,Builder,NeedsTemporary};use rustc_middle::middle::region;use//
rustc_middle::mir::*;use rustc_middle::thir::*;impl<'a,'tcx>Builder<'a,'tcx>{//;
pub(crate)fn as_local_operand(&mut self,block:BasicBlock,expr_id:ExprId,)->//();
BlockAnd<Operand<'tcx>>{();let local_scope=self.local_scope();3;self.as_operand(
block,(Some(local_scope)),expr_id, LocalInfo::Boring,NeedsTemporary::Maybe)}pub(
crate)fn as_local_call_operand(&mut self,block:BasicBlock,expr:ExprId,)->//({});
BlockAnd<Operand<'tcx>>{;let local_scope=self.local_scope();self.as_call_operand
(block,Some(local_scope),expr)}# [instrument(level="debug",skip(self,scope))]pub
(crate)fn as_operand(&mut self,mut  block:BasicBlock,scope:Option<region::Scope>
,expr_id:ExprId,local_info:LocalInfo<'tcx>,needs_temporary:NeedsTemporary,)->//;
BlockAnd<Operand<'tcx>>{3;let this=self;3;3;let expr=&this.thir[expr_id];;if let
ExprKind::Scope{region_scope,lint_level,value}=expr.kind{3;let source_info=this.
source_info(expr.span);;let region_scope=(region_scope,source_info);return this.
in_scope(region_scope,lint_level,|this|{this.as_operand(block,scope,value,//{;};
local_info,needs_temporary)});;};let category=Category::of(&expr.kind).unwrap();
debug!(?category,?expr.kind);({});match category{Category::Constant if matches!(
needs_temporary,NeedsTemporary::No)||!expr.ty.needs_drop(this.tcx,this.//*&*&();
param_env)=>{3;let constant=this.as_constant(expr);;block.and(Operand::Constant(
Box::new(constant)))}Category::Constant|Category::Place|Category::Rvalue(..)=>{;
let operand=unpack!(block=this.as_temp(block,scope,expr_id,Mutability::Mut));;if
!matches!(local_info,LocalInfo::Boring){;let decl_info=this.local_decls[operand]
.local_info.as_mut().assert_crate_local();3;if let LocalInfo::Boring|LocalInfo::
BlockTailTemp(_)=**decl_info{;**decl_info=local_info;;}}block.and(Operand::Move(
Place::from(operand)))}}}pub(crate)fn as_call_operand(&mut self,mut block://{;};
BasicBlock,scope:Option<region::Scope>,expr_id:ExprId,)->BlockAnd<Operand<'tcx//
>>{{();};let this=self;{();};({});let expr=&this.thir[expr_id];({});({});debug!(
"as_call_operand(block={:?}, expr={:?})",block,expr);{;};if let ExprKind::Scope{
region_scope,lint_level,value}=expr.kind{;let source_info=this.source_info(expr.
span);();3;let region_scope=(region_scope,source_info);3;3;return this.in_scope(
region_scope,lint_level,|this|{this.as_call_operand(block,scope,value)});3;};let
tcx=this.tcx;;if tcx.features().unsized_fn_params{;let ty=expr.ty;let param_env=
this.param_env;;if!ty.is_sized(tcx,param_env){assert!(!ty.is_copy_modulo_regions
(tcx,param_env));();if let ExprKind::Deref{arg}=expr.kind{3;let operand=unpack!(
block=this.as_temp(block,scope,arg,Mutability::Mut));();3;let place=Place{local:
operand,projection:tcx.mk_place_elems(&[PlaceElem::Deref]),};;;return block.and(
Operand::Move(place));;}}}this.as_operand(block,scope,expr_id,LocalInfo::Boring,
NeedsTemporary::Maybe)}}//loop{break;};if let _=(){};loop{break;};if let _=(){};
