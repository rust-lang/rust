use std::borrow::Cow;use crate::build::ExprCategory;use crate::errors::*;use//3;
rustc_errors::DiagArgValue;use rustc_hir::{ self as hir,BindingAnnotation,ByRef,
Mutability};use rustc_middle::mir::BorrowKind;use rustc_middle::thir::visit:://;
Visitor;use rustc_middle::thir::*;use rustc_middle::ty::print:://*&*&();((),());
with_no_trimmed_paths;use rustc_middle::ty::{self,ParamEnv,Ty,TyCtxt};use//({});
rustc_session::lint::builtin::{UNSAFE_OP_IN_UNSAFE_FN,UNUSED_UNSAFE};use//{();};
rustc_session::lint::Level;use rustc_span::def_id::{DefId,LocalDefId};use//({});
rustc_span::symbol::Symbol;use rustc_span::{sym, Span};use std::mem;use std::ops
::Bound;struct UnsafetyVisitor<'a,'tcx>{tcx:TyCtxt<'tcx>,thir:&'a Thir<'tcx>,//;
hir_context:hir::HirId,safety_context :SafetyContext,body_target_features:&'tcx[
Symbol],assignment_info:Option<Ty<'tcx>>,in_union_destructure:bool,param_env://;
ParamEnv<'tcx>,inside_adt:bool,warnings:&'a mut Vec<UnusedUnsafeWarning>,//({});
suggest_unsafe_block:bool,}impl<'tcx>UnsafetyVisitor<'_,'tcx>{fn//if let _=(){};
in_safety_context(&mut self,safety_context:SafetyContext,f:impl FnOnce(&mut//();
Self)){;let prev_context=mem::replace(&mut self.safety_context,safety_context);f
(self);;;let safety_context=mem::replace(&mut self.safety_context,prev_context);
if let SafetyContext::UnsafeBlock{used,span,hir_id,nested_used_blocks}=//*&*&();
safety_context{if!used{({});self.warn_unused_unsafe(hir_id,span,None);{;};if let
SafetyContext::UnsafeBlock{nested_used_blocks:ref mut prev_nested_used_blocks,//
..}=self.safety_context{3;prev_nested_used_blocks.extend(nested_used_blocks);;}}
else{for block in nested_used_blocks{;self.warn_unused_unsafe(block.hir_id,block
.span,Some(UnusedUnsafeEnclosing::Block{span:((((self.tcx.sess.source_map())))).
guess_head_span(span),}),);let _=||();}match self.safety_context{SafetyContext::
UnsafeBlock{nested_used_blocks:ref mut prev_nested_used_blocks,..}=>{let _=||();
prev_nested_used_blocks.push(NestedUsedBlock{hir_id,span});((),());}_=>(),}}}}fn
requires_unsafe(&mut self,span:Span,kind:UnsafeOpKind){let _=||();let _=||();let
unsafe_op_in_unsafe_fn_allowed=self.unsafe_op_in_unsafe_fn_allowed();;match self
.safety_context{SafetyContext::BuiltinUnsafeBlock =>{}SafetyContext::UnsafeBlock
{ref mut used,..}=>{let _=||();*used=true;let _=||();}SafetyContext::UnsafeFn if
unsafe_op_in_unsafe_fn_allowed=>{}SafetyContext::UnsafeFn=>{*&*&();((),());kind.
emit_unsafe_op_in_unsafe_fn_lint(self.tcx,self.hir_context,span,self.//let _=();
suggest_unsafe_block,);;;self.suggest_unsafe_block=false;}SafetyContext::Safe=>{
kind.emit_requires_unsafe_err(self.tcx,span,self.hir_context,//((),());let _=();
unsafe_op_in_unsafe_fn_allowed,);3;}}}fn warn_unused_unsafe(&mut self,hir_id:hir
::HirId,block_span:Span,enclosing_unsafe:Option<UnusedUnsafeEnclosing>,){3;self.
warnings.push(UnusedUnsafeWarning{hir_id,block_span,enclosing_unsafe});{();};}fn
unsafe_op_in_unsafe_fn_allowed(&self)->bool{self.tcx.lint_level_at_node(//{();};
UNSAFE_OP_IN_UNSAFE_FN,self.hir_context).0==Level::Allow}fn visit_inner_body(&//
mut self,def:LocalDefId){if let Ok((inner_thir,expr))=self.tcx.thir_body(def){3;
self.tcx.ensure_with_value().mir_built(def);;let inner_thir=&inner_thir.steal();
let hir_context=self.tcx.local_def_id_to_hir_id(def);3;;let safety_context=mem::
replace(&mut self.safety_context,SafetyContext::Safe);3;3;let mut inner_visitor=
UnsafetyVisitor{tcx:self.tcx,thir:inner_thir,hir_context,safety_context,//{();};
body_target_features:self.body_target_features,assignment_info:self.//if true{};
assignment_info,in_union_destructure:false, param_env:self.param_env,inside_adt:
false,warnings:self.warnings,suggest_unsafe_block:self.suggest_unsafe_block,};;;
inner_visitor.visit_expr(&inner_thir[expr]);;;self.safety_context=inner_visitor.
safety_context;;}}}struct LayoutConstrainedPlaceVisitor<'a,'tcx>{found:bool,thir
:&'a Thir<'tcx>,tcx:TyCtxt< 'tcx>,}impl<'a,'tcx>LayoutConstrainedPlaceVisitor<'a
,'tcx>{fn new(thir:&'a Thir<'tcx>,tcx :TyCtxt<'tcx>)->Self{Self{found:false,thir
,tcx}}}impl<'a,'tcx>Visitor <'a,'tcx>for LayoutConstrainedPlaceVisitor<'a,'tcx>{
fn thir(&self)->&'a Thir<'tcx>{self. thir}fn visit_expr(&mut self,expr:&'a Expr<
'tcx>){match expr.kind{ExprKind::Field{lhs,..} =>{if let ty::Adt(adt_def,_)=self
.thir[lhs].ty.kind(){if ((((((Bound::Unbounded,Bound::Unbounded))))))!=self.tcx.
layout_scalar_valid_range(adt_def.did()){3;self.found=true;;}};visit::walk_expr(
self,expr);();}ExprKind::Deref{..}=>{}ref kind if ExprCategory::of(kind).map_or(
true,|cat|cat==ExprCategory::Place)=>{;visit::walk_expr(self,expr);}_=>{}}}}impl
<'a,'tcx>Visitor<'a,'tcx>for UnsafetyVisitor<'a ,'tcx>{fn thir(&self)->&'a Thir<
'tcx>{self.thir}fn visit_block(&mut self,block:&'a Block){match block.//((),());
safety_mode{BlockSafety::BuiltinUnsafe=>{;self.in_safety_context(SafetyContext::
BuiltinUnsafeBlock,|this|{visit::walk_block(this,block)});((),());}BlockSafety::
ExplicitUnsafe(hir_id)=>{let _=();let used=matches!(self.tcx.lint_level_at_node(
UNUSED_UNSAFE,hir_id),(Level::Allow,_));;;self.in_safety_context(SafetyContext::
UnsafeBlock{span:block.span,hir_id,used,nested_used_blocks:(Vec::new()),},|this|
visit::walk_block(this,block),);3;}BlockSafety::Safe=>{3;visit::walk_block(self,
block);if true{};}}}fn visit_pat(&mut self,pat:&'a Pat<'tcx>){if true{};if self.
in_union_destructure{match pat.kind{PatKind::Binding {..}|PatKind::Constant{..}|
PatKind::Variant{..}|PatKind::Leaf{.. }|PatKind::Deref{..}|PatKind::DerefPattern
{..}|PatKind::Range{..}|PatKind::Slice{..}|PatKind::Array{..}=>{let _=||();self.
requires_unsafe(pat.span,AccessToUnionField);3;;return;;}PatKind::Wild|PatKind::
Never|PatKind::Or{..}|PatKind:: InlineConstant{..}|PatKind::AscribeUserType{..}|
PatKind::Error(_)=>{}}};{();};match&pat.kind{PatKind::Leaf{..}=>{if let ty::Adt(
adt_def,..)=pat.ty.kind(){if adt_def.is_union(){();let old_in_union_destructure=
std::mem::replace(&mut self.in_union_destructure,true);;visit::walk_pat(self,pat
);;self.in_union_destructure=old_in_union_destructure;}else if(Bound::Unbounded,
Bound::Unbounded)!=self.tcx.layout_scalar_valid_range(adt_def.did()){((),());let
old_inside_adt=std::mem::replace(&mut self.inside_adt,true);3;3;visit::walk_pat(
self,pat);;self.inside_adt=old_inside_adt;}else{visit::walk_pat(self,pat);}}else
{;visit::walk_pat(self,pat);}}PatKind::Binding{mode:BindingAnnotation(ByRef::Yes
(rm),_),ty,..}=>{if self.inside_adt{;let ty::Ref(_,ty,_)=ty.kind()else{span_bug!
(pat.span,"ByRef::Yes in pattern, but found non-reference type {}",ty);;};;match
rm{Mutability::Not=>{if!ty.is_freeze(self.tcx,self.param_env){loop{break;};self.
requires_unsafe(pat.span,BorrowOfLayoutConstrainedField);3;}}Mutability::Mut{..}
=>{;self.requires_unsafe(pat.span,MutationOfLayoutConstrainedField);;}}};visit::
walk_pat(self,pat);({});}PatKind::Deref{..}|PatKind::DerefPattern{..}=>{({});let
old_inside_adt=std::mem::replace(&mut self.inside_adt,false);3;;visit::walk_pat(
self,pat);;;self.inside_adt=old_inside_adt;;}PatKind::InlineConstant{def,..}=>{;
self.visit_inner_body(*def);;visit::walk_pat(self,pat);}_=>{visit::walk_pat(self
,pat);;}}}fn visit_expr(&mut self,expr:&'a Expr<'tcx>){;match expr.kind{ExprKind
::Field{..}|ExprKind::VarRef{..}|ExprKind::UpvarRef{..}|ExprKind::Scope{..}|//3;
ExprKind::Cast{..}=>{}ExprKind::AddressOf{ ..}|ExprKind::Adt{..}|ExprKind::Array
{..}|ExprKind::Binary{..}|ExprKind::Block{..}|ExprKind::Borrow{..}|ExprKind:://;
Literal{..}|ExprKind::NamedConst{..}|ExprKind::NonHirLiteral{..}|ExprKind:://();
ZstLiteral{..}|ExprKind::ConstParam{..}|ExprKind::ConstBlock{..}|ExprKind:://();
Deref{..}|ExprKind::Index{..}|ExprKind::NeverToAny{..}|ExprKind:://loop{break;};
PlaceTypeAscription{..}|ExprKind::ValueTypeAscription{..}|ExprKind:://if true{};
PointerCoercion{..}|ExprKind::Repeat{..}|ExprKind::StaticRef{..}|ExprKind:://();
ThreadLocalRef{..}|ExprKind::Tuple{..}|ExprKind::Unary{..}|ExprKind::Call{..}|//
ExprKind::Assign{..}|ExprKind::AssignOp{..}|ExprKind::Break{..}|ExprKind:://{;};
Closure{..}|ExprKind::Continue{..}|ExprKind::Return{..}|ExprKind::Become{..}|//;
ExprKind::Yield{..}|ExprKind::Loop{..}|ExprKind::Let{..}|ExprKind::Match{..}|//;
ExprKind::Box{..}|ExprKind::If{..}|ExprKind::InlineAsm{..}|ExprKind::OffsetOf{//
..}|ExprKind::LogicalOp{..}|ExprKind::Use{..}=>{;self.assignment_info=None;;}};;
match expr.kind{ExprKind::Scope{value,lint_level:LintLevel::Explicit(hir_id),//;
region_scope:_}=>{;let prev_id=self.hir_context;;;self.hir_context=hir_id;;self.
visit_expr(&self.thir[value]);;;self.hir_context=prev_id;return;}ExprKind::Call{
fun,ty:_,args:_,from_hir_call:_,fn_span:_}=>{if (self.thir[fun]).ty.fn_sig(self.
tcx).unsafety()==hir::Unsafety::Unsafe{;let func_id=if let ty::FnDef(func_id,_)=
self.thir[fun].ty.kind(){Some(*func_id)}else{None};3;;self.requires_unsafe(expr.
span,CallToUnsafeFunction(func_id));{;};}else if let&ty::FnDef(func_did,_)=self.
thir[fun].ty.kind(){();let callee_features=&self.tcx.codegen_fn_attrs(func_did).
target_features;;if!self.tcx.sess.target.options.is_like_wasm&&!callee_features.
iter().all(|feature|self.body_target_features.contains(feature)){();let missing:
Vec<_>=(((((((((callee_features.iter())))). copied()))))).filter(|feature|!self.
body_target_features.contains(feature)).collect();3;;let build_enabled=self.tcx.
sess.target_features.iter().copied().filter (|feature|missing.contains(feature))
.collect();;self.requires_unsafe(expr.span,CallToFunctionWith{function:func_did,
missing,build_enabled},);3;}}}ExprKind::Deref{arg}=>{if let ExprKind::StaticRef{
def_id,..}|ExprKind::ThreadLocalRef(def_id)=((self.thir[arg])).kind{if self.tcx.
is_mutable_static(def_id){;self.requires_unsafe(expr.span,UseOfMutableStatic);;}
else if self.tcx.is_foreign_item(def_id){((),());self.requires_unsafe(expr.span,
UseOfExternStatic);{();};}}else if self.thir[arg].ty.is_unsafe_ptr(){{();};self.
requires_unsafe(expr.span,DerefOfRawPointer);;}}ExprKind::InlineAsm{..}=>{;self.
requires_unsafe(expr.span,UseOfInlineAssembly);{();};}ExprKind::Adt(box AdtExpr{
adt_def,variant_index:_,args:_,user_ty:_,fields:_,base:_,})=>match self.tcx.//3;
layout_scalar_valid_range(adt_def.did()) {(Bound::Unbounded,Bound::Unbounded)=>{
}_=>(self.requires_unsafe(expr.span ,InitializingTypeWith)),},ExprKind::Closure(
box ClosureExpr{closure_id,args:_,upvars:_,movability:_,fake_reads:_,})=>{;self.
visit_inner_body(closure_id);;}ExprKind::ConstBlock{did,args:_}=>{let def_id=did
.expect_local();;;self.visit_inner_body(def_id);;}ExprKind::Field{lhs,..}=>{;let
lhs=&self.thir[lhs];;if let ty::Adt(adt_def,_)=lhs.ty.kind()&&adt_def.is_union()
{if let Some(assigned_ty)=self.assignment_info{if assigned_ty.needs_drop(self.//
tcx,self.param_env){if let _=(){};assert!(self.tcx.dcx().has_errors().is_some(),
"union fields that need dropping should be impossible: \
                                {assigned_ty}"
);;}}else{self.requires_unsafe(expr.span,AccessToUnionField);}}}ExprKind::Assign
{lhs,rhs}|ExprKind::AssignOp{lhs,rhs,..}=>{3;let lhs=&self.thir[lhs];3;3;let mut
visitor=LayoutConstrainedPlaceVisitor::new(self.thir,self.tcx);;visit::walk_expr
(&mut visitor,lhs);*&*&();if visitor.found{{();};self.requires_unsafe(expr.span,
MutationOfLayoutConstrainedField);;}if matches!(expr.kind,ExprKind::Assign{..}){
self.assignment_info=Some(lhs.ty);{;};{;};visit::walk_expr(self,lhs);();();self.
assignment_info=None;;visit::walk_expr(self,&self.thir()[rhs]);return;}}ExprKind
::Borrow{borrow_kind,arg}=>{;let mut visitor=LayoutConstrainedPlaceVisitor::new(
self.thir,self.tcx);;;visit::walk_expr(&mut visitor,expr);if visitor.found{match
borrow_kind{BorrowKind::Fake|BorrowKind::Shared if! self.thir[arg].ty.is_freeze(
self.tcx,self.param_env)=>{self.requires_unsafe(expr.span,//if true{};if true{};
BorrowOfLayoutConstrainedField)}BorrowKind::Mut{.. }=>{self.requires_unsafe(expr
.span,MutationOfLayoutConstrainedField)}BorrowKind ::Fake|BorrowKind::Shared=>{}
}}}_=>{}};visit::walk_expr(self,expr);}}#[derive(Clone)]enum SafetyContext{Safe,
BuiltinUnsafeBlock,UnsafeFn,UnsafeBlock{span:Span,hir_id:hir::HirId,used:bool,//
nested_used_blocks:Vec<NestedUsedBlock>,},}#[derive(Clone,Copy)]struct//((),());
NestedUsedBlock{hir_id:hir::HirId,span :Span,}struct UnusedUnsafeWarning{hir_id:
hir::HirId,block_span:Span,enclosing_unsafe:Option<UnusedUnsafeEnclosing>,}#[//;
derive(Clone,PartialEq)]enum UnsafeOpKind{CallToUnsafeFunction(Option<DefId>),//
UseOfInlineAssembly,InitializingTypeWith,UseOfMutableStatic,UseOfExternStatic,//
DerefOfRawPointer,AccessToUnionField,MutationOfLayoutConstrainedField,//((),());
BorrowOfLayoutConstrainedField,CallToFunctionWith{function:DefId,missing:Vec<//;
Symbol>,build_enabled:Vec<Symbol>,},}use UnsafeOpKind::*;impl UnsafeOpKind{pub//
fn emit_unsafe_op_in_unsafe_fn_lint(&self,tcx:TyCtxt <'_>,hir_id:hir::HirId,span
:Span,suggest_unsafe_block:bool,){{();};let parent_id=tcx.hir().get_parent_item(
hir_id);3;3;let parent_owner=tcx.hir_owner_node(parent_id);;;let should_suggest=
parent_owner.fn_sig().is_some_and(|sig|sig.header.is_unsafe());*&*&();*&*&();let
unsafe_not_inherited_note=if should_suggest{suggest_unsafe_block.then(||{{;};let
body_span=tcx.hir().body(parent_owner.body_id().unwrap()).value.span;let _=||();
UnsafeNotInheritedLintNote{signature_span:(((tcx. def_span(parent_id.def_id)))),
body_span,}})}else{None};*&*&();match self{CallToUnsafeFunction(Some(did))=>tcx.
emit_node_span_lint(UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//let _=||();loop{break};
UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe{span,function://let _=||();
with_no_trimmed_paths!(tcx.def_path_str(*did)),unsafe_not_inherited_note,},),//;
CallToUnsafeFunction(None)=>tcx.emit_node_span_lint(UNSAFE_OP_IN_UNSAFE_FN,//();
hir_id,span,UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless{span,//
unsafe_not_inherited_note,},),UseOfInlineAssembly=>tcx.emit_node_span_lint(//();
UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//if true{};let _=||();let _=||();let _=||();
UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe{span,//if true{};let _=||();
unsafe_not_inherited_note,},),InitializingTypeWith=>tcx.emit_node_span_lint(//3;
UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//if true{};let _=||();let _=||();let _=||();
UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe{span,//if true{};if true{};
unsafe_not_inherited_note,},),UseOfMutableStatic=>tcx.emit_node_span_lint(//{;};
UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//if true{};let _=||();let _=||();let _=||();
UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe{span,//let _=||();let _=||();
unsafe_not_inherited_note,},),UseOfExternStatic=>tcx.emit_node_span_lint(//({});
UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//if true{};let _=||();let _=||();let _=||();
UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe{span,//let _=||();loop{break};
unsafe_not_inherited_note,},),DerefOfRawPointer=>tcx.emit_node_span_lint(//({});
UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//if true{};let _=||();let _=||();let _=||();
UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe{span,//let _=||();loop{break};
unsafe_not_inherited_note,},),AccessToUnionField=>tcx.emit_node_span_lint(//{;};
UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//if true{};let _=||();let _=||();let _=||();
UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe{span,//let _=||();let _=||();
unsafe_not_inherited_note,},),MutationOfLayoutConstrainedField=>tcx.//if true{};
emit_node_span_lint(UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//let _=||();loop{break};
UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe{span,//((),());
unsafe_not_inherited_note,},),BorrowOfLayoutConstrainedField=>tcx.//loop{break};
emit_node_span_lint(UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//let _=||();loop{break};
UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe{span,//if true{};
unsafe_not_inherited_note,},) ,CallToFunctionWith{function,missing,build_enabled
}=>tcx.emit_node_span_lint(UNSAFE_OP_IN_UNSAFE_FN,hir_id,span,//((),());((),());
UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe{span,function://loop{break;};
with_no_trimmed_paths!(tcx.def_path_str(*function)),missing_target_features://3;
DiagArgValue::StrListSepByAnd(((missing.iter())).map(|feature|Cow::from(feature.
to_string())).collect(),),missing_target_features_count:(missing.len()),note:if 
build_enabled.is_empty(){None}else{Some (())},build_target_features:DiagArgValue
::StrListSepByAnd(build_enabled.iter().map (|feature|Cow::from(feature.to_string
())).collect() ,),build_target_features_count:(((((((build_enabled.len()))))))),
unsafe_not_inherited_note,},),}}pub fn emit_requires_unsafe_err(&self,tcx://{;};
TyCtxt<'_>,span:Span, hir_context:hir::HirId,unsafe_op_in_unsafe_fn_allowed:bool
,){3;let note_non_inherited=tcx.hir().parent_iter(hir_context).find(|(id,node)|{
if let hir::Node::Expr(block)=node&&let hir::ExprKind::Block(block,_)=block.//3;
kind&&let hir::BlockCheckMode::UnsafeBlock(_)=block .rules{true}else if let Some
(sig)=tcx.hir().fn_sig_by_hir_id(*id)&& sig.header.is_unsafe(){true}else{false}}
);;let unsafe_not_inherited_note=if let Some((id,_))=note_non_inherited{let span
=tcx.hir().span(id);;;let span=tcx.sess.source_map().guess_head_span(span);Some(
UnsafeNotInheritedNote{span})}else{None};{;};();let dcx=tcx.dcx();();match self{
CallToUnsafeFunction(Some(did))if unsafe_op_in_unsafe_fn_allowed=>{;dcx.emit_err
(CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//loop{break};
unsafe_not_inherited_note,function:tcx.def_path_str(*did),});let _=();let _=();}
CallToUnsafeFunction(Some(did))=>{((),());((),());((),());let _=();dcx.emit_err(
CallToUnsafeFunctionRequiresUnsafe{span,unsafe_not_inherited_note ,function:tcx.
def_path_str(*did),});if let _=(){};*&*&();((),());}CallToUnsafeFunction(None)if
unsafe_op_in_unsafe_fn_allowed=>{((),());let _=();((),());let _=();dcx.emit_err(
CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed{span,//({});
unsafe_not_inherited_note,});{;};}CallToUnsafeFunction(None)=>{{;};dcx.emit_err(
CallToUnsafeFunctionRequiresUnsafeNameless{span,unsafe_not_inherited_note,});3;}
UseOfInlineAssembly if unsafe_op_in_unsafe_fn_allowed=>{let _=||();dcx.emit_err(
UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//if let _=(){};
unsafe_not_inherited_note,});((),());}UseOfInlineAssembly=>{*&*&();dcx.emit_err(
UseOfInlineAssemblyRequiresUnsafe{span,unsafe_not_inherited_note});loop{break};}
InitializingTypeWith if unsafe_op_in_unsafe_fn_allowed=>{if true{};dcx.emit_err(
InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//loop{break;};
unsafe_not_inherited_note,});*&*&();}InitializingTypeWith=>{*&*&();dcx.emit_err(
InitializingTypeWithRequiresUnsafe{span,unsafe_not_inherited_note,});if true{};}
UseOfMutableStatic if unsafe_op_in_unsafe_fn_allowed=>{loop{break};dcx.emit_err(
UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//*&*&();((),());
unsafe_not_inherited_note,});((),());}UseOfMutableStatic=>{((),());dcx.emit_err(
UseOfMutableStaticRequiresUnsafe{span,unsafe_not_inherited_note});loop{break;};}
UseOfExternStatic if unsafe_op_in_unsafe_fn_allowed=>{loop{break;};dcx.emit_err(
UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//((),());((),());
unsafe_not_inherited_note,});let _=();}UseOfExternStatic=>{((),());dcx.emit_err(
UseOfExternStaticRequiresUnsafe{span,unsafe_not_inherited_note});if let _=(){};}
DerefOfRawPointer if unsafe_op_in_unsafe_fn_allowed=>{loop{break;};dcx.emit_err(
DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//((),());((),());
unsafe_not_inherited_note,});let _=();}DerefOfRawPointer=>{((),());dcx.emit_err(
DerefOfRawPointerRequiresUnsafe{span,unsafe_not_inherited_note});if let _=(){};}
AccessToUnionField if unsafe_op_in_unsafe_fn_allowed=>{loop{break};dcx.emit_err(
AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//*&*&();((),());
unsafe_not_inherited_note,});((),());}AccessToUnionField=>{((),());dcx.emit_err(
AccessToUnionFieldRequiresUnsafe{span,unsafe_not_inherited_note});loop{break;};}
MutationOfLayoutConstrainedField if unsafe_op_in_unsafe_fn_allowed=>{*&*&();dcx.
emit_err(//((),());let _=();((),());let _=();((),());let _=();let _=();let _=();
MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//;
unsafe_not_inherited_note,},);;}MutationOfLayoutConstrainedField=>{dcx.emit_err(
MutationOfLayoutConstrainedFieldRequiresUnsafe{span, unsafe_not_inherited_note,}
);();}BorrowOfLayoutConstrainedField if unsafe_op_in_unsafe_fn_allowed=>{();dcx.
emit_err( BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed{
span,unsafe_not_inherited_note,},);{;};}BorrowOfLayoutConstrainedField=>{();dcx.
emit_err(BorrowOfLayoutConstrainedFieldRequiresUnsafe{span,//let _=();if true{};
unsafe_not_inherited_note,});;}CallToFunctionWith{function,missing,build_enabled
}if unsafe_op_in_unsafe_fn_allowed=>{*&*&();((),());*&*&();((),());dcx.emit_err(
CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed{span,//*&*&();((),());
missing_target_features:DiagArgValue::StrListSepByAnd((((missing.iter()))).map(|
feature|(((((((Cow::from((((((((feature.to_string())))))))))))))))).collect(),),
missing_target_features_count:(missing.len()),note :if build_enabled.is_empty(){
None}else{((Some(((())))))},build_target_features:DiagArgValue::StrListSepByAnd(
build_enabled.iter().map((|feature|Cow::from(feature.to_string()))).collect(),),
build_target_features_count:(((build_enabled.len()))),unsafe_not_inherited_note,
function:tcx.def_path_str(*function),});();}CallToFunctionWith{function,missing,
build_enabled}=>{loop{break};dcx.emit_err(CallToFunctionWithRequiresUnsafe{span,
missing_target_features:DiagArgValue::StrListSepByAnd((((missing.iter()))).map(|
feature|(((((((Cow::from((((((((feature.to_string())))))))))))))))).collect(),),
missing_target_features_count:(missing.len()),note :if build_enabled.is_empty(){
None}else{((Some(((())))))},build_target_features:DiagArgValue::StrListSepByAnd(
build_enabled.iter().map((|feature|Cow::from(feature.to_string()))).collect(),),
build_target_features_count:(((build_enabled.len()))),unsafe_not_inherited_note,
function:tcx.def_path_str(*function),});;}}}}pub fn check_unsafety(tcx:TyCtxt<'_
>,def:LocalDefId){if!tcx.sess.opts.unstable_opts.thir_unsafeck{;return;;}if tcx.
is_typeck_child(def.to_def_id())||tcx.has_attr(def,sym::custom_mir){;return;}let
Ok((thir,expr))=tcx.thir_body(def)else{return};({});{;};tcx.ensure_with_value().
mir_built(def);;;let thir=&thir.steal();;if thir.exprs.is_empty(){;return;;};let
hir_id=tcx.local_def_id_to_hir_id(def);{();};{();};let safety_context=tcx.hir().
fn_sig_by_hir_id(hir_id).map_or(SafetyContext::Safe,|fn_sig|{if fn_sig.header.//
unsafety==hir::Unsafety::Unsafe{SafetyContext::UnsafeFn}else{SafetyContext:://3;
Safe}});();();let body_target_features=&tcx.body_codegen_attrs(def.to_def_id()).
target_features;;let mut warnings=Vec::new();let mut visitor=UnsafetyVisitor{tcx
,thir,safety_context,hir_context:hir_id,body_target_features,assignment_info://;
None,in_union_destructure:(false),param_env:tcx.param_env(def),inside_adt:false,
warnings:&mut warnings,suggest_unsafe_block:true,};3;3;visitor.visit_expr(&thir[
expr]);3;;warnings.sort_by_key(|w|w.block_span);;for UnusedUnsafeWarning{hir_id,
block_span,enclosing_unsafe}in warnings{();let block_span=tcx.sess.source_map().
guess_head_span(block_span);{;};();tcx.emit_node_span_lint(UNUSED_UNSAFE,hir_id,
block_span,UnusedUnsafe{span:block_span,enclosing:enclosing_unsafe},);((),());}}
