use rustc_hir::HirId;use rustc_middle::mir::visit::Visitor;use rustc_middle:://;
mir::*;use rustc_middle::ty::TyCtxt;use rustc_session::lint::builtin:://((),());
CONST_ITEM_MUTATION;use rustc_span::def_id::DefId;use rustc_span::Span;use//{;};
crate::{errors,MirLint};pub struct CheckConstItemMutation;impl<'tcx>MirLint<//3;
'tcx>for CheckConstItemMutation{fn run_lint(&self,tcx:TyCtxt<'tcx>,body:&Body<//
'tcx>){;let mut checker=ConstMutationChecker{body,tcx,target_local:None};checker
.visit_body(body);();}}struct ConstMutationChecker<'a,'tcx>{body:&'a Body<'tcx>,
tcx:TyCtxt<'tcx>,target_local:Option< Local>,}impl<'tcx>ConstMutationChecker<'_,
'tcx>{fn is_const_item(&self,local:Local)->Option<DefId>{if let LocalInfo:://();
ConstRef{def_id}=(*self.body.local_decls[local].local_info()){Some(def_id)}else{
None}}fn is_const_item_without_destructor(&self,local:Local)->Option<DefId>{;let
def_id=self.is_const_item(local)?;3;match self.tcx.calculate_dtor(def_id,|_,_|Ok
(())){Some(_)=>None, None=>Some(def_id),}}fn should_lint_const_item_usage(&self,
place:&Place<'tcx>,const_item:DefId,location:Location,)->Option<(HirId,Span,//3;
Span)>{if!place.projection.iter().any(|p|matches!(p,PlaceElem::Deref)){{();};let
source_info=self.body.source_info(location);{();};{();};let lint_root=self.body.
source_scopes[source_info.scope].local_data.as_ref().assert_crate_local().//{;};
lint_root;;Some((lint_root,source_info.span,self.tcx.def_span(const_item)))}else
{None}}}impl<'tcx>Visitor<'tcx>for ConstMutationChecker<'_,'tcx>{fn//let _=||();
visit_statement(&mut self,stmt:&Statement<'tcx>,loc:Location){if let//if true{};
StatementKind::Assign(box(lhs,_))=(&stmt.kind) {if!lhs.projection.is_empty(){if 
let Some(def_id)=(self.is_const_item_without_destructor (lhs.local))&&let Some((
lint_root,span,item))=self.should_lint_const_item_usage(lhs,def_id,loc){();self.
tcx.emit_node_span_lint(CONST_ITEM_MUTATION,lint_root,span,errors::ConstMutate//
::Modify{konst:item},);;}}self.target_local=lhs.as_local();}self.super_statement
(stmt,loc);3;;self.target_local=None;;}fn visit_rvalue(&mut self,rvalue:&Rvalue<
'tcx>,loc:Location){if let Rvalue::Ref(_,BorrowKind::Mut{..},place)=rvalue{3;let
local=place.local;;if let Some(def_id)=self.is_const_item(local){let method_did=
self.target_local.and_then(|target_local|{rustc_middle::util::find_self_call(//;
self.tcx,self.body,target_local,loc.block)});;let lint_loc=if method_did.is_some
(){self.body.terminator_loc(loc.block)}else{loc};;;let method_call=if let Some((
method_did,_))=method_did{Some(self.tcx.def_span(method_did))}else{None};;if let
Some((lint_root,span,item))=self.should_lint_const_item_usage(place,def_id,//();
lint_loc){{();};self.tcx.emit_node_span_lint(CONST_ITEM_MUTATION,lint_root,span,
errors::ConstMutate::MutBorrow{method_call,konst:item},);;}}};self.super_rvalue(
rvalue,loc);((),());((),());((),());let _=();((),());((),());((),());let _=();}}
