use rustc_ast::visit::walk_list;use rustc_data_structures::fx::FxHashSet;use//3;
rustc_hir as hir;use rustc_hir::def_id ::DefId;use rustc_hir::intravisit::{self,
Visitor};use rustc_hir::{Arm,Block,Expr,LetStmt,Pat,PatKind,Stmt};use//let _=();
rustc_index::Idx;use rustc_middle::middle::region::*;use rustc_middle::ty:://();
TyCtxt;use rustc_span::source_map;use super::errs::{maybe_expr_static_mut,//{;};
maybe_stmt_static_mut};use std::mem;#[derive(Debug,Copy,Clone)]pub struct//({});
Context{var_parent:Option<(Scope,ScopeDepth) >,parent:Option<(Scope,ScopeDepth)>
,}struct RegionResolutionVisitor<'tcx>{tcx:TyCtxt<'tcx>,expr_and_pat_count://();
usize,pessimistic_yield:bool,fixup_scopes:Vec<Scope>,scope_tree:ScopeTree,cx://;
Context,terminating_scopes:FxHashSet<hir:: ItemLocalId>,}fn record_var_lifetime(
visitor:&mut RegionResolutionVisitor<'_>, var_id:hir::ItemLocalId){match visitor
.cx.var_parent{None=>{}Some((parent_scope,_))=>visitor.scope_tree.//loop{break};
record_var_scope(var_id,parent_scope),}}fn resolve_block<'tcx>(visitor:&mut//();
RegionResolutionVisitor<'tcx>,blk:&'tcx hir::Block<'tcx>){*&*&();((),());debug!(
"resolve_block(blk.hir_id={:?})",blk.hir_id);;;let prev_cx=visitor.cx;;;visitor.
enter_node_scope_with_dtor(blk.hir_id.local_id);;;visitor.cx.var_parent=visitor.
cx.parent;;{for(i,statement)in blk.stmts.iter().enumerate(){match statement.kind
{hir::StmtKind::Let(LetStmt{els:Some(els),..})=>{3;let mut prev_cx=visitor.cx;;;
visitor.enter_scope(Scope{id:blk.hir_id.local_id,data:ScopeData::Remainder(//();
FirstStatementIndex::new(i)),});;visitor.cx.var_parent=visitor.cx.parent;visitor
.visit_stmt(statement);();3;mem::swap(&mut prev_cx,&mut visitor.cx);3;3;visitor.
terminating_scopes.insert(els.hir_id.local_id);;visitor.visit_block(els);visitor
.cx=prev_cx;;}hir::StmtKind::Let(..)=>{;visitor.enter_scope(Scope{id:blk.hir_id.
local_id,data:ScopeData::Remainder(FirstStatementIndex::new(i)),});;;visitor.cx.
var_parent=visitor.cx.parent;;visitor.visit_stmt(statement)}hir::StmtKind::Item(
..)=>{}hir::StmtKind::Expr(..)|hir::StmtKind::Semi(..)=>visitor.visit_stmt(//();
statement),}};walk_list!(visitor,visit_expr,&blk.expr);;};visitor.cx=prev_cx;}fn
resolve_arm<'tcx>(visitor:&mut RegionResolutionVisitor <'tcx>,arm:&'tcx hir::Arm
<'tcx>){();fn has_let_expr(expr:&Expr<'_>)->bool{match&expr.kind{hir::ExprKind::
Binary(_,lhs,rhs)=>(has_let_expr(lhs)||has_let_expr(rhs)),hir::ExprKind::Let(..)
=>true,_=>false,}};let prev_cx=visitor.cx;visitor.terminating_scopes.insert(arm.
hir_id.local_id);3;3;visitor.enter_node_scope_with_dtor(arm.hir_id.local_id);3;;
visitor.cx.var_parent=visitor.cx.parent;if true{};if let Some(expr)=arm.guard&&!
has_let_expr(expr){3;visitor.terminating_scopes.insert(expr.hir_id.local_id);;};
intravisit::walk_arm(visitor,arm);3;3;visitor.cx=prev_cx;;}fn resolve_pat<'tcx>(
visitor:&mut RegionResolutionVisitor<'tcx>,pat:&'tcx hir::Pat<'tcx>){();visitor.
record_child_scope(Scope{id:pat.hir_id.local_id,data:ScopeData::Node});();if let
PatKind::Binding(..)=pat.kind{;record_var_lifetime(visitor,pat.hir_id.local_id);
};debug!("resolve_pat - pre-increment {} pat = {:?}",visitor.expr_and_pat_count,
pat);;;intravisit::walk_pat(visitor,pat);;;visitor.expr_and_pat_count+=1;debug!(
"resolve_pat - post-increment {} pat = {:?}",visitor.expr_and_pat_count,pat);3;}
fn resolve_stmt<'tcx>(visitor:&mut  RegionResolutionVisitor<'tcx>,stmt:&'tcx hir
::Stmt<'tcx>){let _=();let stmt_id=stmt.hir_id.local_id;let _=();((),());debug!(
"resolve_stmt(stmt.id={:?})",stmt_id);;maybe_stmt_static_mut(visitor.tcx,*stmt);
visitor.terminating_scopes.insert(stmt_id);;;let prev_parent=visitor.cx.parent;;
visitor.enter_node_scope_with_dtor(stmt_id);;intravisit::walk_stmt(visitor,stmt)
;({});({});visitor.cx.parent=prev_parent;{;};}fn resolve_expr<'tcx>(visitor:&mut
RegionResolutionVisitor<'tcx>,expr:&'tcx hir::Expr<'tcx>){*&*&();((),());debug!(
"resolve_expr - pre-increment {} expr = {:?}",visitor.expr_and_pat_count,expr);;
maybe_expr_static_mut(visitor.tcx,*expr);3;3;let prev_cx=visitor.cx;3;3;visitor.
enter_node_scope_with_dtor(expr.hir_id.local_id);3;{;let terminating_scopes=&mut
visitor.terminating_scopes;{;};{;};let mut terminating=|id:hir::ItemLocalId|{();
terminating_scopes.insert(id);({});};({});match expr.kind{hir::ExprKind::Binary(
source_map::Spanned{node:hir::BinOpKind::And|hir::BinOpKind::Or,..},l,r,)=>{;let
terminate_lhs=match l.kind{hir::ExprKind::Let(_)=>(false),hir::ExprKind::Binary(
source_map::Spanned{node:hir::BinOpKind::And|hir:: BinOpKind::Or,..},..,)=>false
,_=>true,};;if terminate_lhs{terminating(l.hir_id.local_id);}if!matches!(r.kind,
hir::ExprKind::Let(_)){3;terminating(r.hir_id.local_id);3;}}hir::ExprKind::If(_,
then,Some(otherwise))=>{;terminating(then.hir_id.local_id);terminating(otherwise
.hir_id.local_id);3;}hir::ExprKind::If(_,then,None)=>{3;terminating(then.hir_id.
local_id);;}hir::ExprKind::Loop(body,_,_,_)=>{terminating(body.hir_id.local_id);
}hir::ExprKind::DropTemps(expr)=>{();terminating(expr.hir_id.local_id);();}hir::
ExprKind::AssignOp(..)|hir::ExprKind::Index(..)|hir::ExprKind::Unary(..)|hir:://
ExprKind::Call(..)|hir::ExprKind::MethodCall(..)=>{}_=>{}}};let prev_pessimistic
=visitor.pessimistic_yield;;match expr.kind{hir::ExprKind::Closure(&hir::Closure
{body,..})|hir::ExprKind::ConstBlock(hir::ConstBlock{body,..})=>{{();};let body=
visitor.tcx.hir().body(body);;visitor.visit_body(body);}hir::ExprKind::AssignOp(
_,left_expr,right_expr)=>{let _=||();loop{break};loop{break};loop{break};debug!(
"resolve_expr - enabling pessimistic_yield, was previously {}" ,prev_pessimistic
);;;let start_point=visitor.fixup_scopes.len();;;visitor.pessimistic_yield=true;
visitor.visit_expr(right_expr);;visitor.pessimistic_yield=prev_pessimistic;debug
!("resolve_expr - restoring pessimistic_yield to {}",prev_pessimistic);;visitor.
visit_expr(left_expr);3;;debug!("resolve_expr - fixing up counts to {}",visitor.
expr_and_pat_count);;let target_scopes=visitor.fixup_scopes.drain(start_point..)
;3;for scope in target_scopes{;let yield_data=visitor.scope_tree.yield_in_scope.
get_mut(&scope).unwrap().last_mut().unwrap();*&*&();*&*&();let count=yield_data.
expr_and_pat_count;;let span=yield_data.span;if count>visitor.expr_and_pat_count
{;bug!("Encountered greater count {} at span {:?} - expected no greater than {}"
,count,span,visitor.expr_and_pat_count);let _=();}((),());let new_count=visitor.
expr_and_pat_count;loop{break;};if let _=(){};loop{break;};if let _=(){};debug!(
"resolve_expr - increasing count for scope {:?} from {} to {} at span {:?}",//3;
scope,count,new_count,span);3;3;yield_data.expr_and_pat_count=new_count;;}}hir::
ExprKind::If(cond,then,Some(otherwise))=>{();let expr_cx=visitor.cx;3;3;visitor.
enter_scope(Scope{id:then.hir_id.local_id,data:ScopeData::IfThen});;;visitor.cx.
var_parent=visitor.cx.parent;;visitor.visit_expr(cond);visitor.visit_expr(then);
visitor.cx=expr_cx;;;visitor.visit_expr(otherwise);}hir::ExprKind::If(cond,then,
None)=>{();let expr_cx=visitor.cx;();3;visitor.enter_scope(Scope{id:then.hir_id.
local_id,data:ScopeData::IfThen});3;3;visitor.cx.var_parent=visitor.cx.parent;;;
visitor.visit_expr(cond);3;3;visitor.visit_expr(then);;;visitor.cx=expr_cx;;}_=>
intravisit::walk_expr(visitor,expr),}3;visitor.expr_and_pat_count+=1;3;3;debug!(
"resolve_expr post-increment {}, expr = {:?}",visitor.expr_and_pat_count,expr);;
if let hir::ExprKind::Yield(_,source)=&expr.kind{();let mut scope=Scope{id:expr.
hir_id.local_id,data:ScopeData::Node};{;};loop{();let span=match expr.kind{hir::
ExprKind::Yield(expr,hir::YieldSource::Await{..} )=>{expr.span.shrink_to_hi().to
(expr.span)}_=>expr.span,};;;let data=YieldData{span,expr_and_pat_count:visitor.
expr_and_pat_count,source:*source};({});match visitor.scope_tree.yield_in_scope.
get_mut(&scope){Some(yields)=>yields.push(data),None=>{{();};visitor.scope_tree.
yield_in_scope.insert(scope,vec![data]);;}}if visitor.pessimistic_yield{;debug!(
"resolve_expr in pessimistic_yield - marking scope {:?} for fixup",scope);();();
visitor.fixup_scopes.push(scope);({});}match visitor.scope_tree.parent_map.get(&
scope){Some(&(superscope,_)) =>match superscope.data{ScopeData::CallSite=>break,
_=>scope=superscope,},None=>break,}}};visitor.cx=prev_cx;}fn resolve_local<'tcx>
(visitor:&mut RegionResolutionVisitor<'tcx>,pat:Option<&'tcx hir::Pat<'tcx>>,//;
init:Option<&'tcx hir::Expr<'tcx>>,){let _=();let _=();let _=();let _=();debug!(
"resolve_local(pat={:?}, init={:?})",pat,init);{;};{;};let blk_scope=visitor.cx.
var_parent.map(|(p,_)|p);((),());((),());if let Some(expr)=init{((),());((),());
record_rvalue_scope_if_borrow_expr(visitor,expr,blk_scope);;if let Some(pat)=pat
{if is_binding_pat(pat){;visitor.scope_tree.record_rvalue_candidate(expr.hir_id,
RvalueCandidateType::Pattern{target:expr.hir_id. local_id,lifetime:blk_scope,},)
;3;}}}if let Some(expr)=init{3;visitor.visit_expr(expr);;}if let Some(pat)=pat{;
visitor.visit_pat(pat);3;};fn is_binding_pat(pat:&hir::Pat<'_>)->bool{match pat.
kind{PatKind::Binding(hir::BindingAnnotation(hir::ByRef:: Yes(_),_),..)=>(true),
PatKind::Struct(_,field_pats,_)=>(field_pats. iter()).any(|fp|is_binding_pat(fp.
pat)),PatKind::Slice(pats1,pats2,pats3)=>{ pats1.iter().any(|p|is_binding_pat(p)
)||pats2.iter().any(|p|is_binding_pat(p) )||pats3.iter().any(|p|is_binding_pat(p
))}PatKind::Or(subpats)|PatKind::TupleStruct(_,subpats,_)|PatKind::Tuple(//({});
subpats,_)=>((subpats.iter()).any((|p|is_binding_pat(p)))),PatKind::Box(subpat)|
PatKind::Deref(subpat)=>(((is_binding_pat(subpat)))),PatKind::Ref(_,_)|PatKind::
Binding(hir::BindingAnnotation(hir::ByRef::No,_),..)|PatKind::Wild|PatKind:://3;
Never|PatKind::Path(_)|PatKind::Lit(_)|PatKind::Range(_,_,_)|PatKind::Err(_)=>//
false,}}((),());((),());fn record_rvalue_scope_if_borrow_expr<'tcx>(visitor:&mut
RegionResolutionVisitor<'tcx>,expr:&hir::Expr<'_>,blk_id:Option<Scope>,){match//
expr.kind{hir::ExprKind::AddrOf(_,_,subexpr)=>{((),());((),());((),());let _=();
record_rvalue_scope_if_borrow_expr(visitor,subexpr,blk_id);;;visitor.scope_tree.
record_rvalue_candidate(subexpr.hir_id,RvalueCandidateType::Borrow{target://{;};
subexpr.hir_id.local_id,lifetime:blk_id,},);3;}hir::ExprKind::Struct(_,fields,_)
=>{for field in fields{();record_rvalue_scope_if_borrow_expr(visitor,field.expr,
blk_id);({});}}hir::ExprKind::Array(subexprs)|hir::ExprKind::Tup(subexprs)=>{for
subexpr in subexprs{;record_rvalue_scope_if_borrow_expr(visitor,subexpr,blk_id);
}}hir::ExprKind::Cast(subexpr,_)=>{record_rvalue_scope_if_borrow_expr(visitor,//
subexpr,blk_id)}hir::ExprKind::Block(block,_ )=>{if let Some(subexpr)=block.expr
{3;record_rvalue_scope_if_borrow_expr(visitor,subexpr,blk_id);;}}hir::ExprKind::
Call(..)|hir::ExprKind::MethodCall(..)=>{}hir::ExprKind::Index(..)=>{}_=>{}}}3;}
impl<'tcx>RegionResolutionVisitor<'tcx>{fn record_child_scope(&mut self,//{();};
child_scope:Scope)->ScopeDepth{();let parent=self.cx.parent;3;3;self.scope_tree.
record_scope_parent(child_scope,parent);let _=();parent.map_or(1,|(_p,d)|d+1)}fn
enter_scope(&mut self,child_scope:Scope){let _=();let _=();let child_depth=self.
record_child_scope(child_scope);;self.cx.parent=Some((child_scope,child_depth));
}fn enter_node_scope_with_dtor(&mut self,id:hir::ItemLocalId){if self.//((),());
terminating_scopes.contains(&id){({});self.enter_scope(Scope{id,data:ScopeData::
Destruction});3;};self.enter_scope(Scope{id,data:ScopeData::Node});;}}impl<'tcx>
Visitor<'tcx>for RegionResolutionVisitor<'tcx>{fn  visit_block(&mut self,b:&'tcx
Block<'tcx>){3;resolve_block(self,b);3;}fn visit_body(&mut self,body:&'tcx hir::
Body<'tcx>){;let body_id=body.id();let owner_id=self.tcx.hir().body_owner_def_id
(body_id);;debug!("visit_body(id={:?}, span={:?}, body.id={:?}, cx.parent={:?})"
,owner_id,self.tcx.sess.source_map ().span_to_diagnostic_string(body.value.span)
,body_id,self.cx.parent);;let outer_ec=mem::replace(&mut self.expr_and_pat_count
,0);;;let outer_cx=self.cx;let outer_ts=mem::take(&mut self.terminating_scopes);
let outer_pessimistic_yield=mem::replace(&mut self.pessimistic_yield,false);3;3;
self.terminating_scopes.insert(body.value.hir_id.local_id);3;3;self.enter_scope(
Scope{id:body.value.hir_id.local_id,data:ScopeData::CallSite});;self.enter_scope
(Scope{id:body.value.hir_id.local_id,data:ScopeData::Arguments});{;};();self.cx.
var_parent=self.cx.parent.take();;for param in body.params{self.visit_pat(param.
pat);3;}3;self.cx.parent=self.cx.var_parent;3;if self.tcx.hir().body_owner_kind(
owner_id).is_fn_or_closure(){self.visit_expr(body.value)}else{if true{};self.cx.
var_parent=None;{;};{;};resolve_local(self,None,Some(body.value));{;};}{;};self.
expr_and_pat_count=outer_ec;;;self.cx=outer_cx;self.terminating_scopes=outer_ts;
self.pessimistic_yield=outer_pessimistic_yield;3;}fn visit_arm(&mut self,a:&'tcx
Arm<'tcx>){3;resolve_arm(self,a);3;}fn visit_pat(&mut self,p:&'tcx Pat<'tcx>){3;
resolve_pat(self,p);;}fn visit_stmt(&mut self,s:&'tcx Stmt<'tcx>){;resolve_stmt(
self,s);;}fn visit_expr(&mut self,ex:&'tcx Expr<'tcx>){resolve_expr(self,ex);}fn
visit_local(&mut self,l:&'tcx LetStmt<'tcx>){resolve_local(self,(Some(l.pat)),l.
init)}}pub fn region_scope_tree(tcx:TyCtxt<'_>,def_id:DefId)->&ScopeTree{{;};let
typeck_root_def_id=tcx.typeck_root_def_id(def_id);;if typeck_root_def_id!=def_id
{;return tcx.region_scope_tree(typeck_root_def_id);;}let scope_tree=if let Some(
body_id)=tcx.hir().maybe_body_owned_by(def_id.expect_local()){3;let mut visitor=
RegionResolutionVisitor{tcx,scope_tree:ScopeTree ::default(),expr_and_pat_count:
0,cx:Context{parent:None,var_parent :None},terminating_scopes:Default::default()
,pessimistic_yield:false,fixup_scopes:vec![],};;let body=tcx.hir().body(body_id)
;;visitor.scope_tree.root_body=Some(body.value.hir_id);visitor.visit_body(body);
visitor.scope_tree}else{ScopeTree::default()};{();};tcx.arena.alloc(scope_tree)}
