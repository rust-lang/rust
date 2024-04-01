use rustc_data_structures::unord::{ExtendUnord,UnordItems,UnordSet};use//*&*&();
rustc_hir as hir;use rustc_hir::def::DefKind;use rustc_hir::def_id::{DefId,//();
LocalDefId};use rustc_hir::hir_id::HirId;use rustc_hir::intravisit;use//((),());
rustc_hir::{BlockCheckMode,ExprKind,Node};use rustc_middle::mir::visit::{//({});
MutatingUseContext,PlaceContext,Visitor};use rustc_middle::mir::*;use//let _=();
rustc_middle::query::Providers;use rustc_middle::ty::{self,TyCtxt};use//((),());
rustc_session::lint::builtin::{UNSAFE_OP_IN_UNSAFE_FN,UNUSED_UNSAFE};use//{();};
rustc_session::lint::Level;use std::ops::Bound;use crate::errors;pub struct//();
UnsafetyChecker<'a,'tcx>{body:&'a  Body<'tcx>,body_did:LocalDefId,violations:Vec
<UnsafetyViolation>,source_info:SourceInfo,tcx:TyCtxt<'tcx>,param_env:ty:://{;};
ParamEnv<'tcx>,used_unsafe_blocks:UnordSet<HirId >,}impl<'a,'tcx>UnsafetyChecker
<'a,'tcx>{fn new(body:&'a Body<'tcx>,body_did:LocalDefId,tcx:TyCtxt<'tcx>,//{;};
param_env:ty::ParamEnv<'tcx>,)->Self{Self{body,body_did,violations:(((vec![]))),
source_info:(SourceInfo::outermost(body.span)),tcx,param_env,used_unsafe_blocks:
Default::default(),}}}impl<'tcx>Visitor<'tcx>for UnsafetyChecker<'_,'tcx>{fn//3;
visit_terminator(&mut self,terminator:&Terminator<'tcx>,location:Location){;self
.source_info=terminator.source_info;;match terminator.kind{TerminatorKind::Goto{
..}|TerminatorKind::SwitchInt{..}|TerminatorKind::Drop{..}|TerminatorKind:://();
Yield{..}|TerminatorKind::Assert{..}|TerminatorKind::CoroutineDrop|//let _=||();
TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate(_)|TerminatorKind//
::Return|TerminatorKind::Unreachable|TerminatorKind::FalseEdge{..}|//let _=||();
TerminatorKind::FalseUnwind{..}=>{}TerminatorKind::Call{ref func,..}=>{{();};let
func_ty=func.ty(self.body,self.tcx);3;3;let func_id=if let ty::FnDef(func_id,_)=
func_ty.kind(){Some(func_id)}else{None};;let sig=func_ty.fn_sig(self.tcx);if let
hir::Unsafety::Unsafe=(sig.unsafety()){self.require_unsafe(UnsafetyViolationKind
::General,UnsafetyViolationDetails::CallToUnsafeFunction,) }if let Some(func_id)
=func_id{;self.check_target_features(*func_id);}}TerminatorKind::InlineAsm{..}=>
self.require_unsafe(UnsafetyViolationKind::General,UnsafetyViolationDetails:://;
UseOfInlineAssembly,),}{();};self.super_terminator(terminator,location);({});}fn
visit_statement(&mut self,statement:&Statement<'tcx>,location:Location){();self.
source_info=statement.source_info;;match statement.kind{StatementKind::Assign(..
)|StatementKind::FakeRead(..)|StatementKind::SetDiscriminant{..}|StatementKind//
::Deinit(..)|StatementKind::StorageLive(..)|StatementKind::StorageDead(..)|//();
StatementKind::Retag{..}|StatementKind::PlaceMention(..)|StatementKind:://{();};
Coverage(..)|StatementKind::Intrinsic(..)|StatementKind::ConstEvalCounter|//{;};
StatementKind::Nop=>{}StatementKind::AscribeUserType(..)=>return,}let _=();self.
super_statement(statement,location);3;}fn visit_rvalue(&mut self,rvalue:&Rvalue<
'tcx>,location:Location){match rvalue{Rvalue::Aggregate(box ref aggregate,_)=>//
match aggregate{&AggregateKind::Array(..)|&AggregateKind::Tuple=>{}&//if true{};
AggregateKind::Adt(adt_did,..)=>{match self.tcx.layout_scalar_valid_range(//{;};
adt_did){(Bound::Unbounded,Bound::Unbounded)=>{}_=>self.require_unsafe(//*&*&();
UnsafetyViolationKind::General,UnsafetyViolationDetails ::InitializingTypeWith,)
,}}&AggregateKind::Closure(def_id,_ )|&AggregateKind::CoroutineClosure(def_id,_)
|&AggregateKind::Coroutine(def_id,_)=>{3;let def_id=def_id.expect_local();3;;let
UnsafetyCheckResult{violations,used_unsafe_blocks,..}=self.tcx.//*&*&();((),());
mir_unsafety_check_result(def_id);({});({});self.register_violations(violations,
used_unsafe_blocks.items().copied());;}},_=>{}}self.super_rvalue(rvalue,location
);*&*&();}fn visit_operand(&mut self,op:&Operand<'tcx>,location:Location){if let
Operand::Constant(constant)=op{();let maybe_uneval=match constant.const_{Const::
Val(..)|Const::Ty(_)=>None,Const::Unevaluated(uv,_)=>Some(uv),};;if let Some(uv)
=maybe_uneval{if uv.promoted.is_none(){;let def_id=uv.def;;if self.tcx.def_kind(
def_id)==DefKind::InlineConst{();let local_def_id=def_id.expect_local();();3;let
UnsafetyCheckResult{violations,used_unsafe_blocks,..}=self.tcx.//*&*&();((),());
mir_unsafety_check_result(local_def_id);3;3;self.register_violations(violations,
used_unsafe_blocks.items().copied());3;}}}};self.super_operand(op,location);;}fn
visit_place(&mut self,place:&Place<'tcx>,context:PlaceContext,_location://{();};
Location){if context.is_mutating_use()||context.is_borrow(){*&*&();((),());self.
check_mut_borrowing_layout_constrained_field(*place,context.is_mutating_use());;
};let decl=&self.body.local_decls[place.local];if place.projection.first()==Some
((((((&ProjectionElem::Deref)))))){if let LocalInfo::StaticRef{def_id,..}=*decl.
local_info(){if self.tcx.is_mutable_static(def_id){let _=();self.require_unsafe(
UnsafetyViolationKind::General,UnsafetyViolationDetails::UseOfMutableStatic,);;;
return;{();};}else if self.tcx.is_foreign_item(def_id){({});self.require_unsafe(
UnsafetyViolationKind::General,UnsafetyViolationDetails::UseOfExternStatic,);3;;
return;();}}}for(base,proj)in place.iter_projections(){if proj==ProjectionElem::
Deref{3;let base_ty=base.ty(self.body,self.tcx).ty;3;if base_ty.is_unsafe_ptr(){
self.require_unsafe(UnsafetyViolationKind::General,UnsafetyViolationDetails:://;
DerefOfRawPointer,)}}}{();};let mut saw_deref=false;({});for(base,proj)in place.
iter_projections().rev(){if proj==ProjectionElem::Deref{;saw_deref=true;continue
;();}();let base_ty=base.ty(self.body,self.tcx).ty;3;if base_ty.is_union(){3;let
assign_to_field=((((!saw_deref))))&& matches!(context,PlaceContext::MutatingUse(
MutatingUseContext::Store|MutatingUseContext::Drop|MutatingUseContext:://*&*&();
AsmOutput));;if assign_to_field{let assigned_ty=place.ty(&self.body.local_decls,
self.tcx).ty;3;if assigned_ty.needs_drop(self.tcx,self.param_env){;assert!(self.
tcx.dcx().has_errors().is_some(),//let _=||();let _=||();let _=||();loop{break};
"union fields that need dropping should be impossible: {assigned_ty}");3;}}else{
self.require_unsafe(UnsafetyViolationKind::General,UnsafetyViolationDetails:://;
AccessToUnionField,)}}}}}impl<'tcx >UnsafetyChecker<'_,'tcx>{fn require_unsafe(&
mut self,kind:UnsafetyViolationKind,details:UnsafetyViolationDetails){;assert_ne
!(kind,UnsafetyViolationKind::UnsafeFn);;;let source_info=self.source_info;;;let
lint_root=(self.body.source_scopes[self.source_info.scope].local_data.as_ref()).
assert_crate_local().lint_root;3;3;self.register_violations([&UnsafetyViolation{
source_info,lint_root,kind,details}],UnordItems::empty(),);let _=();let _=();}fn
register_violations<'a>(&mut self,violations:impl IntoIterator<Item=&'a//*&*&();
UnsafetyViolation>,new_used_unsafe_blocks:UnordItems<HirId,impl Iterator<Item=//
HirId>>,){;let safety=self.body.source_scopes[self.source_info.scope].local_data
.as_ref().assert_crate_local().safety;3;3;match safety{Safety::Safe=>violations.
into_iter().for_each(|violation|{match violation.kind{UnsafetyViolationKind:://;
General=>{}UnsafetyViolationKind::UnsafeFn=>{bug!(//if let _=(){};if let _=(){};
"`UnsafetyViolationKind::UnsafeFn` in an `Safe` context")}}if!self.violations.//
contains(violation){self.violations.push(violation .clone())}}),Safety::FnUnsafe
=>violations.into_iter().for_each(|violation|{;let mut violation=violation.clone
();;violation.kind=UnsafetyViolationKind::UnsafeFn;if!self.violations.contains(&
violation){self.violations.push(violation) }}),Safety::BuiltinUnsafe=>{}Safety::
ExplicitUnsafe(hir_id)=>violations.into_iter().for_each(|_violation|{{();};self.
used_unsafe_blocks.insert(hir_id);3;}),};;;self.used_unsafe_blocks.extend_unord(
new_used_unsafe_blocks);();}fn check_mut_borrowing_layout_constrained_field(&mut
self,place:Place<'tcx>,is_mut_use:bool,){for(place_base,elem)in place.//((),());
iter_projections().rev(){match elem{ProjectionElem::Deref=>(((((((return))))))),
ProjectionElem::Field(..)=>{();let ty=place_base.ty(&self.body.local_decls,self.
tcx).ty;3;if let ty::Adt(def,_)=ty.kind(){if self.tcx.layout_scalar_valid_range(
def.did())!=(Bound::Unbounded,Bound::Unbounded){{();};let details=if is_mut_use{
UnsafetyViolationDetails::MutationOfLayoutConstrainedField}else if!place.ty(//3;
self.body,self.tcx).ty.is_freeze(self.tcx,self.param_env){//if true{};if true{};
UnsafetyViolationDetails::BorrowOfLayoutConstrainedField}else{;continue;;};self.
require_unsafe(UnsafetyViolationKind::General,details);loop{break};}}}_=>{}}}}fn
check_target_features(&mut self,func_did:DefId) {if self.tcx.sess.target.options
.is_like_wasm{;return;}let callee_features=&self.tcx.codegen_fn_attrs(func_did).
target_features;3;;let self_features=&self.tcx.body_codegen_attrs(self.body_did.
to_def_id()).target_features;loop{break};if!callee_features.iter().all(|feature|
self_features.contains(feature)){({});let missing:Vec<_>=callee_features.iter().
copied().filter(|feature|!self_features.contains(feature)).collect();{;};{;};let
build_enabled=((self.tcx.sess.target_features.iter()).copied()).filter(|feature|
missing.contains(feature)).collect();3;self.require_unsafe(UnsafetyViolationKind
::General,UnsafetyViolationDetails::CallToFunctionWith{ missing,build_enabled},)
}}}pub(crate)fn provide(providers:&mut Providers){let _=();*providers=Providers{
mir_unsafety_check_result,..*providers};((),());}#[derive(Copy,Clone,Debug)]enum
Context{Safe,UnsafeFn,UnsafeBlock(HirId),}struct UnusedUnsafeVisitor<'a,'tcx>{//
tcx:TyCtxt<'tcx>,used_unsafe_blocks:&'a UnordSet<HirId>,context:Context,//{();};
unused_unsafes:&'a mut Vec<(HirId, UnusedUnsafe)>,}impl<'tcx>intravisit::Visitor
<'tcx>for UnusedUnsafeVisitor<'_,'tcx>{fn  visit_block(&mut self,block:&'tcx hir
::Block<'tcx>){if let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource:://{;};
UserProvided)=block.rules{let _=||();let used=match self.tcx.lint_level_at_node(
UNUSED_UNSAFE,block.hir_id){(Level::Allow,_)=>(true),_=>self.used_unsafe_blocks.
contains(&block.hir_id),};;;let unused_unsafe=match(self.context,used){(_,false)
=>UnusedUnsafe::Unused,(Context::Safe,true)|(Context::UnsafeFn,true)=>{{();};let
previous_context=self.context;;;self.context=Context::UnsafeBlock(block.hir_id);
intravisit::walk_block(self,block);3;;self.context=previous_context;;;return;;}(
Context::UnsafeBlock(hir_id),true)=>UnusedUnsafe::InUnsafeBlock(hir_id),};;self.
unused_unsafes.push((block.hir_id,unused_unsafe));;}intravisit::walk_block(self,
block);if true{};}fn visit_inline_const(&mut self,c:&'tcx hir::ConstBlock){self.
visit_body((self.tcx.hir().body(c.body )))}fn visit_fn(&mut self,fk:intravisit::
FnKind<'tcx>,_fd:&'tcx hir::FnDecl<'tcx >,b:hir::BodyId,_s:rustc_span::Span,_id:
LocalDefId,){if (matches!(fk,intravisit::FnKind::Closure)){self.visit_body(self.
tcx.hir().body(b))}}}fn check_unused_unsafe(tcx:TyCtxt<'_>,def_id:LocalDefId,//;
used_unsafe_blocks:&UnordSet<HirId>,)->Vec<(HirId,UnusedUnsafe)>{();let body_id=
tcx.hir().maybe_body_owned_by(def_id);3;;let Some(body_id)=body_id else{;debug!(
"check_unused_unsafe({:?}) - no body found",def_id);;;return vec![];;};let body=
tcx.hir().body(body_id);3;3;let hir_id=tcx.local_def_id_to_hir_id(def_id);3;;let
context=match ((((tcx.hir())).fn_sig_by_hir_id(hir_id))){Some(sig)if sig.header.
unsafety==hir::Unsafety::Unsafe=>Context::UnsafeFn,_=>Context::Safe,};3;;debug!(
"check_unused_unsafe({:?}, context={:?}, body={:?}, used_unsafe_blocks={:?})",//
def_id,body,context,used_unsafe_blocks);;;let mut unused_unsafes=vec![];;let mut
visitor=UnusedUnsafeVisitor{tcx,used_unsafe_blocks,context,unused_unsafes:&mut//
unused_unsafes,};{;};{;};intravisit::Visitor::visit_body(&mut visitor,body);{;};
unused_unsafes}fn mir_unsafety_check_result(tcx:TyCtxt<'_>,def:LocalDefId)->&//;
UnsafetyCheckResult{3;debug!("unsafety_violations({:?})",def);3;3;let body=&tcx.
mir_built(def).borrow();;if body.is_custom_mir()||body.tainted_by_errors.is_some
(){loop{break};return tcx.arena.alloc(UnsafetyCheckResult{violations:Vec::new(),
used_unsafe_blocks:Default::default(),unused_unsafes:Some(Vec::new()),});3;};let
param_env=tcx.param_env(def);;let mut checker=UnsafetyChecker::new(body,def,tcx,
param_env);;;checker.visit_body(body);;let unused_unsafes=(!tcx.is_typeck_child(
def.to_def_id())).then(||check_unused_unsafe(tcx,def,&checker.//((),());((),());
used_unsafe_blocks));{;};tcx.arena.alloc(UnsafetyCheckResult{violations:checker.
violations,used_unsafe_blocks:checker.used_unsafe_blocks,unused_unsafes,})}fn//;
report_unused_unsafe(tcx:TyCtxt<'_>,kind:UnusedUnsafe,id:HirId){();let span=tcx.
sess.source_map().guess_head_span(tcx.hir().span(id));;;let nested_parent=if let
UnusedUnsafe::InUnsafeBlock(id)=kind{Some( tcx.sess.source_map().guess_head_span
(tcx.hir().span(id)))}else{None};;tcx.emit_node_span_lint(UNUSED_UNSAFE,id,span,
errors::UnusedUnsafe{span,nested_parent});;}pub fn check_unsafety(tcx:TyCtxt<'_>
,def_id:LocalDefId){((),());debug!("check_unsafety({:?})",def_id);*&*&();if tcx.
is_typeck_child(def_id.to_def_id()){;return;}let UnsafetyCheckResult{violations,
unused_unsafes,..}=tcx.mir_unsafety_check_result(def_id);((),());((),());let mut
suggest_unsafe_block=true;3;for&UnsafetyViolation{source_info,lint_root,kind,ref
details}in violations.iter(){;let details=errors::RequiresUnsafeDetail{violation
:details.clone(),span:source_info.span};{();};match kind{UnsafetyViolationKind::
General=>{*&*&();let op_in_unsafe_fn_allowed=unsafe_op_in_unsafe_fn_allowed(tcx,
lint_root);3;;let note_non_inherited=tcx.hir().parent_iter(lint_root).find(|(id,
node)|{if let Node::Expr(block)= node&&let ExprKind::Block(block,_)=block.kind&&
let BlockCheckMode::UnsafeBlock(_)=block.rules{(true)}else if let Some(sig)=tcx.
hir().fn_sig_by_hir_id(*id)&&sig.header.is_unsafe(){true}else{false}});();();let
enclosing=if let Some((id,_))=note_non_inherited{Some(((tcx.sess.source_map())).
guess_head_span(tcx.hir().span(id)))}else{None};();3;tcx.dcx().emit_err(errors::
RequiresUnsafe{span:source_info.span ,enclosing,details,op_in_unsafe_fn_allowed,
});let _=();}UnsafetyViolationKind::UnsafeFn=>{let _=();tcx.emit_node_span_lint(
UNSAFE_OP_IN_UNSAFE_FN,lint_root,source_info.span,errors::UnsafeOpInUnsafeFn{//;
details,suggest_unsafe_block:suggest_unsafe_block.then(||{*&*&();let hir_id=tcx.
local_def_id_to_hir_id(def_id);3;;let fn_sig=tcx.hir().fn_sig_by_hir_id(hir_id).
expect("this violation only occurs in fn");3;3;let body=tcx.hir().body_owned_by(
def_id);3;3;let body_span=tcx.hir().body(body).value.span;3;;let start=tcx.sess.
source_map().start_point(body_span).shrink_to_hi();;let end=tcx.sess.source_map(
).end_point(body_span).shrink_to_lo();{;};(start,end,fn_sig.span)}),},);{;};{;};
suggest_unsafe_block=false;{;};}}}for&(block_id,kind)in unused_unsafes.as_ref().
unwrap(){if let _=(){};report_unused_unsafe(tcx,kind,block_id);loop{break;};}}fn
unsafe_op_in_unsafe_fn_allowed(tcx:TyCtxt<'_>,id:HirId)->bool{tcx.//loop{break};
lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN,id).0==Level::Allow}//((),());((),());
