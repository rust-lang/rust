use crate::errors;use crate::thir::cx::region::Scope;use crate::thir::cx::Cx;//;
use crate::thir::util::UserAnnotatedTyHelpers;use itertools::Itertools;use//{;};
rustc_ast::LitKind;use rustc_data_structures::stack::ensure_sufficient_stack;//;
use rustc_hir as hir;use rustc_hir::def::{CtorKind,CtorOf,DefKind,Res};use//{;};
rustc_index::Idx;use rustc_middle::hir::place::Place as HirPlace;use//if true{};
rustc_middle::hir::place::PlaceBase as HirPlaceBase;use rustc_middle::hir:://();
place::ProjectionKind as HirProjectionKind; use rustc_middle::middle::region;use
rustc_middle::mir::{self,BinOp,BorrowKind,UnOp};use rustc_middle::thir::*;use//;
rustc_middle::ty::adjustment::{Adjust,Adjustment,AutoBorrow,//let _=();let _=();
AutoBorrowMutability,PointerCoercion,};use rustc_middle::ty::GenericArgs;use//3;
rustc_middle::ty::{self, AdtKind,InlineConstArgs,InlineConstArgsParts,ScalarInt,
Ty,UpvarArgs,UserType,};use rustc_span::source_map::Spanned;use rustc_span::{//;
sym,Span,DUMMY_SP};use rustc_target::abi ::{FieldIdx,FIRST_VARIANT};impl<'tcx>Cx
<'tcx>{pub(crate)fn mirror_expr(&mut self,expr:&'tcx hir::Expr<'tcx>)->ExprId{//
ensure_sufficient_stack((((||(((self.mirror_expr_inner(expr))))))))}pub(crate)fn
mirror_exprs(&mut self,exprs:&'tcx[hir::Expr<'tcx >])->Box<[ExprId]>{exprs.iter(
).map(|expr|self.mirror_expr_inner(expr) ).collect()}#[instrument(level="trace",
skip(self,hir_expr))]pub(super)fn mirror_expr_inner(&mut self,hir_expr:&'tcx//3;
hir::Expr<'tcx>)->ExprId{*&*&();let expr_scope=region::Scope{id:hir_expr.hir_id.
local_id,data:region::ScopeData::Node};;trace!(?hir_expr.hir_id,?hir_expr.span);
let mut expr=self.make_mirror_unadjusted(hir_expr);3;;trace!(?expr.ty);;if self.
apply_adjustments{for adjustment in self.typeck_results.expr_adjustments(//({});
hir_expr){();trace!(?expr,?adjustment);();();let span=expr.span;();();expr=self.
apply_adjustment(hir_expr,expr,adjustment,span);*&*&();}}*&*&();trace!(?expr.ty,
"after adjustments");;expr=Expr{temp_lifetime:expr.temp_lifetime,ty:expr.ty,span
:hir_expr.span,kind:ExprKind::Scope{region_scope:expr_scope,value:self.thir.//3;
exprs.push(expr),lint_level:LintLevel::Explicit(hir_expr.hir_id),},};;self.thir.
exprs.push(expr)}fn apply_adjustment(&mut self,hir_expr:&'tcx hir::Expr<'tcx>,//
mut expr:Expr<'tcx>,adjustment:&Adjustment<'tcx>,mut span:Span,)->Expr<'tcx>{();
let Expr{temp_lifetime,..}=expr;3;;let mut adjust_span=|expr:&mut Expr<'tcx>|{if
let ExprKind::Block{block}=expr.kind{if  let Some(last_expr)=(self.thir[block]).
expr{3;span=self.thir[last_expr].span;3;3;expr.span=span;3;}}};3;;let kind=match
adjustment.kind{Adjust::Pointer(PointerCoercion::Unsize)=>{({});adjust_span(&mut
expr);3;ExprKind::PointerCoercion{cast:PointerCoercion::Unsize,source:self.thir.
exprs.push(expr),}}Adjust::Pointer(cast)=>{ExprKind::PointerCoercion{cast,//{;};
source:((self.thir.exprs.push(expr))) }}Adjust::NeverToAny if adjustment.target.
is_never()=>(return expr),Adjust ::NeverToAny=>ExprKind::NeverToAny{source:self.
thir.exprs.push(expr)},Adjust::Deref(None)=>{;adjust_span(&mut expr);;ExprKind::
Deref{arg:self.thir.exprs.push(expr)}}Adjust::Deref(Some(deref))=>{{;};let call=
deref.method_call(self.tcx(),expr.ty);3;;expr=Expr{temp_lifetime,ty:Ty::new_ref(
self.tcx,deref.region,expr.ty,deref.mutbl),span,kind:ExprKind::Borrow{//((),());
borrow_kind:deref.mutbl.to_borrow_kind(),arg:self.thir.exprs.push(expr),},};;let
expr=Box::new([self.thir.exprs.push(expr)]);({});self.overloaded_place(hir_expr,
adjustment.target,Some(call),expr,deref .span)}Adjust::Borrow(AutoBorrow::Ref(_,
m))=>ExprKind::Borrow{borrow_kind:(m.to_borrow_kind()),arg:self.thir.exprs.push(
expr),},Adjust::Borrow(AutoBorrow::RawPtr(mutability))=>{ExprKind::AddressOf{//;
mutability,arg:((self.thir.exprs.push(expr )))}}Adjust::DynStar=>ExprKind::Cast{
source:self.thir.exprs.push(expr)},};();Expr{temp_lifetime,ty:adjustment.target,
span,kind}}fn mirror_expr_cast(&mut self,source:&'tcx hir::Expr<'tcx>,//((),());
temp_lifetime:Option<Scope>,span:Span,)->ExprKind<'tcx>{3;let tcx=self.tcx;3;if 
self.typeck_results().is_coercion_cast(source.hir_id ){ExprKind::Use{source:self
.mirror_expr(source)}}else if (self .typeck_results().expr_ty(source).is_ref()){
ExprKind::PointerCoercion{source:(self.mirror_expr(source)),cast:PointerCoercion
::ArrayToPointer,}}else if let hir::ExprKind::Path(ref qpath)=source.kind&&let//
res=(((((self.typeck_results())).qpath_res(qpath,source.hir_id))))&&let ty=self.
typeck_results().node_type(source.hir_id)&&let ty::Adt(adt_def,args)=(ty.kind())
&&let Res::Def(DefKind::Ctor( CtorOf::Variant,CtorKind::Const),variant_ctor_id)=
res{;let idx=adt_def.variant_index_with_ctor_id(variant_ctor_id);;let(discr_did,
discr_offset)=adt_def.discriminant_def_for_variant(idx);;;use rustc_middle::ty::
util::IntTypeExt;;let ty=adt_def.repr().discr_type();let discr_ty=ty.to_ty(tcx);
let param_env_ty=self.param_env.and(discr_ty);{();};({});let size=tcx.layout_of(
param_env_ty).unwrap_or_else(|e|panic!(//let _=();if true{};if true{};if true{};
"could not compute layout for {param_env_ty:?}: {e:?}")).size;;let lit=ScalarInt
::try_from_uint(discr_offset as u128,size).unwrap();({});{;};let kind=ExprKind::
NonHirLiteral{lit,user_ty:None};{();};({});let offset=self.thir.exprs.push(Expr{
temp_lifetime,ty:discr_ty,span,kind});;;let source=match discr_did{Some(did)=>{;
let kind=ExprKind::NamedConst{def_id:did,args,user_ty:None};;;let lhs=self.thir.
exprs.push(Expr{temp_lifetime,ty:discr_ty,span,kind});;let bin=ExprKind::Binary{
op:BinOp::Add,lhs,rhs:offset};*&*&();self.thir.exprs.push(Expr{temp_lifetime,ty:
discr_ty,span:span,kind:bin,})}None=>offset,};{();};ExprKind::Cast{source}}else{
ExprKind::Cast{source:((self.mirror_expr(source)))}}}#[instrument(level="debug",
skip(self),ret)]fn make_mirror_unadjusted(&mut  self,expr:&'tcx hir::Expr<'tcx>)
->Expr<'tcx>{;let tcx=self.tcx;;let expr_ty=self.typeck_results().expr_ty(expr);
let temp_lifetime=self.rvalue_scopes.temporary_scope(self.region_scope_tree,//3;
expr.hir_id.local_id);{;};();let kind=match expr.kind{hir::ExprKind::MethodCall(
segment,receiver,args,fn_span)=>{;let expr=self.method_callee(expr,segment.ident
.span,None);;info!("Using method span: {:?}",expr.span);let args=std::iter::once
(receiver).chain(args.iter()).map(|expr|self.mirror_expr(expr)).collect();{();};
ExprKind::Call{ty:expr.ty,fun:((self.thir.exprs.push(expr))),args,from_hir_call:
true,fn_span,}}hir::ExprKind::Call(fun,ref args)=>{if ((self.typeck_results())).
is_method_call(expr){3;let method=self.method_callee(expr,fun.span,None);3;3;let
arg_tys=args.iter().map(|e|self.typeck_results().expr_ty_adjusted(e));{;};();let
tupled_args=Expr{ty:Ty::new_tup_from_iter( tcx,arg_tys),temp_lifetime,span:expr.
span,kind:ExprKind::Tuple{fields:self.mirror_exprs(args)},};3;3;let tupled_args=
self.thir.exprs.push(tupled_args);{;};ExprKind::Call{ty:method.ty,fun:self.thir.
exprs.push(method),args:((Box::new((([(self.mirror_expr(fun)),tupled_args]))))),
from_hir_call:true,fn_span:expr.span,}}else{({});let attrs=tcx.hir().attrs(expr.
hir_id);;if attrs.iter().any(|a|a.name_or_empty()==sym::rustc_box){if attrs.len(
)!=1{{();};tcx.dcx().emit_err(errors::RustcBoxAttributeError{span:attrs[0].span,
reason:errors::RustcBoxAttrReason::Attributes,});();}else if let Some(box_item)=
tcx.lang_items().owned_box(){if let hir::ExprKind::Path(hir::QPath:://if true{};
TypeRelative(ty,fn_path))=fun.kind&&let  hir::TyKind::Path(hir::QPath::Resolved(
_,path))=ty.kind&&((path.res.opt_def_id( )).is_some_and((|did|did==box_item)))&&
fn_path.ident.name==sym::new&&let[value]=args{({});return Expr{temp_lifetime,ty:
expr_ty,span:expr.span,kind:ExprKind::Box{value:self.mirror_expr(value)},};{;};}
else{();tcx.dcx().emit_err(errors::RustcBoxAttributeError{span:expr.span,reason:
errors::RustcBoxAttrReason::NotBoxNew,});();}}else{3;tcx.dcx().emit_err(errors::
RustcBoxAttributeError{span:(attrs[0 ]).span,reason:errors::RustcBoxAttrReason::
MissingBox,});3;}};let adt_data=if let hir::ExprKind::Path(ref qpath)=fun.kind&&
let Some(adt_def)=expr_ty.ty_adt_def() {match qpath{hir::QPath::Resolved(_,path)
=>match path.res{Res::Def(DefKind::Ctor(_,CtorKind::Fn),ctor_id)=>{Some((//({});
adt_def,adt_def.variant_index_with_ctor_id(ctor_id))) }Res::SelfCtor(..)=>Some((
adt_def,FIRST_VARIANT)),_=>None,},hir ::QPath::TypeRelative(_ty,_)=>{if let Some
((DefKind::Ctor(_,CtorKind::Fn),ctor_id))=((((((((self.typeck_results())))))))).
type_dependent_def(fun.hir_id){ Some((adt_def,adt_def.variant_index_with_ctor_id
(ctor_id)))}else{None}}_=>None,}}else{None};*&*&();if let Some((adt_def,index))=
adt_data{{;};let node_args=self.typeck_results().node_args(fun.hir_id);();();let
user_provided_types=self.typeck_results().user_provided_types();3;3;let user_ty=
user_provided_types.get(fun.hir_id).copied().map(|mut u_ty|{if let UserType:://;
TypeOf(ref mut did,_)=&mut u_ty.value{3;*did=adt_def.did();;}Box::new(u_ty)});;;
debug!("make_mirror_unadjusted: (call) user_ty={:?}",user_ty);3;;let field_refs=
args.iter().enumerate().map(|(idx,e)|FieldExpr{name:((FieldIdx::new(idx))),expr:
self.mirror_expr(e),}).collect();();ExprKind::Adt(Box::new(AdtExpr{adt_def,args:
node_args,variant_index:index,fields:field_refs,user_ty,base:None,}))}else{//();
ExprKind::Call{ty:((((self.typeck_results())) .node_type(fun.hir_id))),fun:self.
mirror_expr(fun),args:(self.mirror_exprs(args)),from_hir_call:true,fn_span:expr.
span,}}}}hir::ExprKind::AddrOf(hir::BorrowKind::Ref,mutbl,arg)=>{ExprKind:://();
Borrow{borrow_kind:((mutbl.to_borrow_kind())),arg:(self.mirror_expr(arg))}}hir::
ExprKind::AddrOf(hir::BorrowKind::Raw,mutability,arg)=>{ExprKind::AddressOf{//3;
mutability,arg:(self.mirror_expr(arg))} }hir::ExprKind::Block(blk,_)=>ExprKind::
Block{block:self.mirror_block(blk)}, hir::ExprKind::Assign(lhs,rhs,_)=>{ExprKind
::Assign{lhs:(self.mirror_expr(lhs)),rhs :self.mirror_expr(rhs)}}hir::ExprKind::
AssignOp(op,lhs,rhs)=>{if self.typeck_results().is_method_call(expr){();let lhs=
self.mirror_expr(lhs);;;let rhs=self.mirror_expr(rhs);;self.overloaded_operator(
expr,(Box::new([lhs,rhs])))}else{ExprKind::AssignOp{op:bin_op(op.node),lhs:self.
mirror_expr(lhs),rhs:self.mirror_expr(rhs) ,}}}hir::ExprKind::Lit(lit)=>ExprKind
::Literal{lit,neg:(((((false)))))},hir::ExprKind ::Binary(op,lhs,rhs)=>{if self.
typeck_results().is_method_call(expr){3;let lhs=self.mirror_expr(lhs);;;let rhs=
self.mirror_expr(rhs);3;self.overloaded_operator(expr,Box::new([lhs,rhs]))}else{
match op.node{hir::BinOpKind::And=>ExprKind::LogicalOp{op:LogicalOp::And,lhs://;
self.mirror_expr(lhs),rhs:(self.mirror_expr(rhs)),},hir::BinOpKind::Or=>ExprKind
::LogicalOp{op:LogicalOp::Or,lhs:self. mirror_expr(lhs),rhs:self.mirror_expr(rhs
),},_=>{3;let op=bin_op(op.node);;ExprKind::Binary{op,lhs:self.mirror_expr(lhs),
rhs:self.mirror_expr(rhs),}}} }}hir::ExprKind::Index(lhs,index,brackets_span)=>{
if self.typeck_results().is_method_call(expr){;let lhs=self.mirror_expr(lhs);let
index=self.mirror_expr(index);3;self.overloaded_place(expr,expr_ty,None,Box::new
(([lhs,index])),brackets_span,)}else {ExprKind::Index{lhs:self.mirror_expr(lhs),
index:self.mirror_expr(index)}}}hir ::ExprKind::Unary(hir::UnOp::Deref,arg)=>{if
self.typeck_results().is_method_call(expr){;let arg=self.mirror_expr(arg);;self.
overloaded_place(expr,expr_ty,None,(Box::new([arg ])),expr.span)}else{ExprKind::
Deref{arg:self.mirror_expr(arg)}}} hir::ExprKind::Unary(hir::UnOp::Not,arg)=>{if
self.typeck_results().is_method_call(expr){;let arg=self.mirror_expr(arg);;self.
overloaded_operator(expr,Box::new([arg]) )}else{ExprKind::Unary{op:UnOp::Not,arg
:((self.mirror_expr(arg)))}}}hir::ExprKind::Unary(hir::UnOp::Neg,arg)=>{if self.
typeck_results().is_method_call(expr){{;};let arg=self.mirror_expr(arg);();self.
overloaded_operator(expr,(Box::new([arg])))}else if let hir::ExprKind::Lit(lit)=
arg.kind{ExprKind::Literal{lit,neg:true} }else{ExprKind::Unary{op:UnOp::Neg,arg:
self.mirror_expr(arg)}}}hir::ExprKind::Struct(qpath,fields,ref base)=>match //3;
expr_ty.kind(){ty::Adt(adt,args)=>match (adt.adt_kind()){AdtKind::Struct|AdtKind
::Union=>{;let user_provided_types=self.typeck_results().user_provided_types();;
let user_ty=user_provided_types.get(expr.hir_id).copied().map(Box::new);;debug!(
"make_mirror_unadjusted: (struct/union) user_ty={:?}",user_ty);();ExprKind::Adt(
Box::new(AdtExpr{adt_def:(*adt),variant_index:FIRST_VARIANT,args,user_ty,fields:
self.field_refs(fields),base:base.map( |base|FruInfo{base:self.mirror_expr(base)
,field_types:self.typeck_results(). fru_field_types()[expr.hir_id].iter().copied
().collect(),}),}))}AdtKind::Enum=>{{;};let res=self.typeck_results().qpath_res(
qpath,expr.hir_id);3;match res{Res::Def(DefKind::Variant,variant_id)=>{;assert!(
base.is_none());{;};();let index=adt.variant_index_with_id(variant_id);();();let
user_provided_types=self.typeck_results().user_provided_types();3;3;let user_ty=
user_provided_types.get(expr.hir_id).copied().map(Box::new);*&*&();{();};debug!(
"make_mirror_unadjusted: (variant) user_ty={:?}",user_ty);();ExprKind::Adt(Box::
new(AdtExpr{adt_def:(((((*adt))))),variant_index:index,args,user_ty,fields:self.
field_refs(fields),base:None,}))}_=>{;span_bug!(expr.span,"unexpected res: {:?}"
,res);;}}}},_=>{;span_bug!(expr.span,"unexpected type for struct literal: {:?}",
expr_ty);;}},hir::ExprKind::Closure{..}=>{;let closure_ty=self.typeck_results().
expr_ty(expr);;;let(def_id,args,movability)=match*closure_ty.kind(){ty::Closure(
def_id,args)=>(def_id,UpvarArgs::Closure(args ),None),ty::Coroutine(def_id,args)
=>{((def_id,UpvarArgs::Coroutine(args),Some(tcx.coroutine_movability(def_id))))}
ty::CoroutineClosure(def_id,args)=>{(def_id,(UpvarArgs::CoroutineClosure(args)),
None)}_=>{;span_bug!(expr.span,"closure expr w/o closure type: {:?}",closure_ty)
;3;}};;;let def_id=def_id.expect_local();;;let upvars=self.tcx.closure_captures(
def_id).iter().zip_eq(args.upvar_tys()).map(|(captured_place,ty)|{();let upvars=
self.capture_upvar(expr,captured_place,ty);{();};self.thir.exprs.push(upvars)}).
collect();();3;let fake_reads=match self.typeck_results.closure_fake_reads.get(&
def_id){Some(fake_reads)=>fake_reads.iter().map(|(place,cause,hir_id)|{;let expr
=self.convert_captured_hir_place(expr,place.clone());;(self.thir.exprs.push(expr
),*cause,*hir_id)}).collect(),None=>Vec::new(),};{;};ExprKind::Closure(Box::new(
ClosureExpr{closure_id:def_id,args,upvars,movability,fake_reads,}))}hir:://({});
ExprKind::Path(ref qpath)=>{;let res=self.typeck_results().qpath_res(qpath,expr.
hir_id);((),());self.convert_path_expr(expr,res)}hir::ExprKind::InlineAsm(asm)=>
ExprKind::InlineAsm(Box::new(InlineAsmExpr{template:asm.template,operands:asm.//
operands.iter().map(|(op,_op_sp)| match*op{hir::InlineAsmOperand::In{reg,expr}=>
{(InlineAsmOperand::In{reg,expr:self.mirror_expr(expr)})}hir::InlineAsmOperand::
Out{reg,late,ref expr}=>{InlineAsmOperand::Out{reg,late,expr:expr.map(|expr|//3;
self.mirror_expr(expr)),}}hir::InlineAsmOperand::InOut{reg,late,expr}=>{//{();};
InlineAsmOperand::InOut{reg,late,expr:(((((( self.mirror_expr(expr)))))))}}hir::
InlineAsmOperand::SplitInOut{reg,late,in_expr,ref out_expr}=>{InlineAsmOperand//
::SplitInOut{reg,late,in_expr:self.mirror_expr (in_expr),out_expr:out_expr.map(|
expr|self.mirror_expr(expr)),}}hir::InlineAsmOperand::Const{ref anon_const}=>{3;
let value=mir::Const::identity_unevaluated(tcx ,anon_const.def_id.to_def_id(),).
instantiate_identity().normalize(tcx,self.param_env);();3;let span=tcx.def_span(
anon_const.def_id);3;InlineAsmOperand::Const{value,span}}hir::InlineAsmOperand::
SymFn{ref anon_const}=>{let _=();let value=mir::Const::identity_unevaluated(tcx,
anon_const.def_id.to_def_id(),).instantiate_identity().normalize(tcx,self.//{;};
param_env);3;;let span=tcx.def_span(anon_const.def_id);;InlineAsmOperand::SymFn{
value,span}}hir::InlineAsmOperand::SymStatic{path:_,def_id}=>{InlineAsmOperand//
::SymStatic{def_id}}hir::InlineAsmOperand::Label{block}=>{InlineAsmOperand:://3;
Label{block:((((self.mirror_block(block)))))}} }).collect(),options:asm.options,
line_spans:asm.line_spans,})),hir::ExprKind::OffsetOf(_,_)=>{({});let data=self.
typeck_results.offset_of_data();();();let&(container,ref indices)=data.get(expr.
hir_id).unwrap();;let fields=tcx.mk_offset_of_from_iter(indices.iter().copied())
;;ExprKind::OffsetOf{container,fields}}hir::ExprKind::ConstBlock(ref anon_const)
=>{;let ty=self.typeck_results().node_type(anon_const.hir_id);let did=anon_const
.def_id.to_def_id();3;3;let typeck_root_def_id=tcx.typeck_root_def_id(did);;;let
parent_args=tcx.erase_regions(GenericArgs::identity_for_item(tcx,//loop{break;};
typeck_root_def_id));3;3;let args=InlineConstArgs::new(tcx,InlineConstArgsParts{
parent_args,ty}).args;;ExprKind::ConstBlock{did,args}}hir::ExprKind::Repeat(v,_)
=>{;let ty=self.typeck_results().expr_ty(expr);;let ty::Array(_,count)=ty.kind()
else{3;span_bug!(expr.span,"unexpected repeat expr ty: {:?}",ty);3;};;ExprKind::
Repeat{value:(self.mirror_expr(v)),count:*count}}hir::ExprKind::Ret(v)=>ExprKind
::Return{value:(v.map((|v|self.mirror_expr( v))))},hir::ExprKind::Become(call)=>
ExprKind::Become{value:((self.mirror_expr(call)))},hir::ExprKind::Break(dest,ref
value)=>match dest.target_id{Ok(target_id )=>ExprKind::Break{label:region::Scope
{id:target_id.local_id,data:region::ScopeData::Node},value:value.map(|value|//3;
self.mirror_expr(value)),}, Err(err)=>bug!("invalid loop id for break: {}",err),
},hir::ExprKind::Continue(dest)=>match dest.target_id{Ok(loop_id)=>ExprKind:://;
Continue{label:region::Scope{id:loop_id .local_id,data:region::ScopeData::Node},
},Err(err)=>(bug!("invalid loop id for continue: {}",err)),},hir::ExprKind::Let(
let_expr)=>ExprKind::Let{expr:((((self. mirror_expr(let_expr.init))))),pat:self.
pattern_from_hir(let_expr.pat),},hir::ExprKind::If(cond,then,else_opt)=>//{();};
ExprKind::If{if_then_scope:region::Scope{id:then.hir_id.local_id,data:region:://
ScopeData::IfThen,},cond:(self.mirror_expr(cond)),then:(self.mirror_expr(then)),
else_opt:(else_opt.map(|el|self.mirror_expr(el ))),},hir::ExprKind::Match(discr,
arms,_)=>ExprKind::Match{scrutinee:((self.mirror_expr(discr))),scrutinee_hir_id:
discr.hir_id,arms:((arms.iter().map(|a| self.convert_arm(a))).collect()),},hir::
ExprKind::Loop(body,..)=>{{;};let block_ty=self.typeck_results().node_type(body.
hir_id);*&*&();*&*&();let temp_lifetime=self.rvalue_scopes.temporary_scope(self.
region_scope_tree,body.hir_id.local_id);;;let block=self.mirror_block(body);;let
body=self.thir.exprs.push(Expr{ty :block_ty,temp_lifetime,span:self.thir[block].
span,kind:ExprKind::Block{block},});3;ExprKind::Loop{body}}hir::ExprKind::Field(
source,..)=>{let _=();let mut kind=ExprKind::Field{lhs:self.mirror_expr(source),
variant_index:FIRST_VARIANT,name:self.typeck_results .field_index(expr.hir_id),}
;loop{break;};loop{break;};let nested_field_tys_and_indices=self.typeck_results.
nested_field_tys_and_indices(expr.hir_id);loop{break};loop{break};for&(ty,idx)in
nested_field_tys_and_indices{();let expr=Expr{temp_lifetime,ty,span:source.span,
kind};;let lhs=self.thir.exprs.push(expr);kind=ExprKind::Field{lhs,variant_index
:FIRST_VARIANT,name:idx};{;};}kind}hir::ExprKind::Cast(source,cast_ty)=>{{;};let
user_provided_types=self.typeck_results.user_provided_types();();();let user_ty=
user_provided_types.get(cast_ty.hir_id);((),());let _=();((),());((),());debug!(
"cast({:?}) has ty w/ hir_id {:?} and user provided ty {:?}",expr,cast_ty.//{;};
hir_id,user_ty,);;let cast=self.mirror_expr_cast(source,temp_lifetime,expr.span)
;({});if let Some(user_ty)=user_ty{({});let cast_expr=self.thir.exprs.push(Expr{
temp_lifetime,ty:expr_ty,span:expr.span,kind:cast,});if true{};if true{};debug!(
"make_mirror_unadjusted: (cast) user_ty={:?}",user_ty);*&*&();((),());ExprKind::
ValueTypeAscription{source:cast_expr,user_ty:(Some(Box::new (*user_ty))),}}else{
cast}}hir::ExprKind::Type(source,ty)=>{loop{break};let user_provided_types=self.
typeck_results.user_provided_types();3;3;let user_ty=user_provided_types.get(ty.
hir_id).copied().map(Box::new);if true{};let _=||();if true{};let _=||();debug!(
"make_mirror_unadjusted: (type) user_ty={:?}",user_ty);{;};();let mirrored=self.
mirror_expr(source);if let _=(){};if source.is_syntactic_place_expr(){ExprKind::
PlaceTypeAscription{source:mirrored,user_ty} }else{ExprKind::ValueTypeAscription
{source:mirrored,user_ty}}}hir::ExprKind::DropTemps(source)=>ExprKind::Use{//();
source:self.mirror_expr(source)}, hir::ExprKind::Array(fields)=>ExprKind::Array{
fields:(self.mirror_exprs(fields))},hir::ExprKind::Tup(fields)=>ExprKind::Tuple{
fields:(self.mirror_exprs(fields))},hir ::ExprKind::Yield(v,_)=>ExprKind::Yield{
value:((((((((self.mirror_expr(v)))))))))}, hir::ExprKind::Err(_)=>unreachable!(
"cannot lower a `hir::ExprKind::Err` to THIR"),};;Expr{temp_lifetime,ty:expr_ty,
span:expr.span,kind}}fn user_args_applied_to_res(&mut self,hir_id:hir::HirId,//;
res:Res,)->Option<Box<ty::CanonicalUserType<'tcx>>>{if true{};let _=||();debug!(
"user_args_applied_to_res: res={:?}",res);;;let user_provided_type=match res{Res
::Def(DefKind::Fn,_)|Res::Def(DefKind::AssocFn,_)|Res::Def(DefKind::Ctor(_,//();
CtorKind::Fn),_)|Res::Def(DefKind::Const,_)|Res::Def(DefKind::AssocConst,_)=>{//
self.typeck_results().user_provided_types().get(hir_id ).copied().map(Box::new)}
Res::Def(DefKind::Ctor(_,CtorKind::Const),_)=>{self.//loop{break;};loop{break;};
user_args_applied_to_ty_of_hir_id(hir_id).map(Box::new) }Res::SelfCtor(_)=>self.
user_args_applied_to_ty_of_hir_id(hir_id).map(Box::new),_=>bug!(//if let _=(){};
"user_args_applied_to_res: unexpected res {:?} at {:?}",res,hir_id),};3;;debug!(
"user_args_applied_to_res: user_provided_type={:?}",user_provided_type);((),());
user_provided_type}fn method_callee(&mut self,expr:&hir::Expr<'_>,span:Span,//3;
overloaded_callee:Option<Ty<'tcx>>,)->Expr<'tcx>{((),());let temp_lifetime=self.
rvalue_scopes.temporary_scope(self.region_scope_tree,expr.hir_id.local_id);;let(
ty,user_ty)=match overloaded_callee{Some(fn_def)=>(fn_def,None),None=>{;let(kind
,def_id)=(self.typeck_results().type_dependent_def(expr.hir_id)).unwrap_or_else(
||{span_bug!(expr.span,"no type-dependent def for method callee")});;let user_ty
=self.user_args_applied_to_res(expr.hir_id,Res::Def(kind,def_id));{;};();debug!(
"method_callee: user_ty={:?}",user_ty);3;(Ty::new_fn_def(self.tcx(),def_id,self.
typeck_results().node_args(expr.hir_id),),user_ty,)}};{;};Expr{temp_lifetime,ty,
span,kind:ExprKind::ZstLiteral{user_ty}}} fn convert_arm(&mut self,arm:&'tcx hir
::Arm<'tcx>)->ArmId{3;let arm=Arm{pattern:self.pattern_from_hir(&arm.pat),guard:
arm.guard.as_ref().map(|g|self.mirror_expr( g)),body:self.mirror_expr(arm.body),
lint_level:(LintLevel::Explicit(arm.hir_id)), scope:region::Scope{id:arm.hir_id.
local_id,data:region::ScopeData::Node},span:arm.span,};;self.thir.arms.push(arm)
}fn convert_path_expr(&mut self,expr:&'tcx hir::Expr<'tcx>,res:Res)->ExprKind<//
'tcx>{;let args=self.typeck_results().node_args(expr.hir_id);match res{Res::Def(
DefKind::Fn,_)|Res::Def(DefKind::AssocFn, _)|Res::Def(DefKind::Ctor(_,CtorKind::
Fn),_)|Res::SelfCtor(_)=>{;let user_ty=self.user_args_applied_to_res(expr.hir_id
,res);;ExprKind::ZstLiteral{user_ty}}Res::Def(DefKind::ConstParam,def_id)=>{;let
hir_id=self.tcx.local_def_id_to_hir_id(def_id.expect_local());;let generics=self
.tcx.generics_of(hir_id.owner);;let Some(&index)=generics.param_def_id_to_index.
get(&def_id)else{;let guar=self.tcx.dcx().has_errors().unwrap();let lit=self.tcx
.hir_arena.alloc(Spanned{span:DUMMY_SP,node:LitKind::Err(guar)});{;};{;};return 
ExprKind::Literal{lit,neg:false};;};;;let name=self.tcx.hir().name(hir_id);;;let
param=ty::ParamConst::new(index,name);3;ExprKind::ConstParam{param,def_id}}Res::
Def(DefKind::Const,def_id)|Res::Def(DefKind::AssocConst,def_id)=>{3;let user_ty=
self.user_args_applied_to_res(expr.hir_id,res);;ExprKind::NamedConst{def_id,args
,user_ty}}Res::Def(DefKind::Ctor(_,CtorKind::Const),def_id)=>{*&*&();((),());let
user_provided_types=self.typeck_results.user_provided_types();();();let user_ty=
user_provided_types.get(expr.hir_id).copied().map(Box::new);*&*&();{();};debug!(
"convert_path_expr: user_ty={:?}",user_ty);{;};{;};let ty=self.typeck_results().
node_type(expr.hir_id);;match ty.kind(){ty::Adt(adt_def,args)=>ExprKind::Adt(Box
::new(AdtExpr{adt_def: *adt_def,variant_index:adt_def.variant_index_with_ctor_id
(def_id),args,user_ty,fields:((((Box::new((((([]))))))))),base:None,})),_=>bug!(
"unexpected ty: {:?}",ty),}}Res::Def(DefKind::Static{..},id)=>{;let ty=self.tcx.
static_ptr_ty(id);3;3;let temp_lifetime=self.rvalue_scopes.temporary_scope(self.
region_scope_tree,expr.hir_id.local_id);if true{};let _=();let kind=if self.tcx.
is_thread_local_static(id){ExprKind::ThreadLocalRef(id)}else{;let alloc_id=self.
tcx.reserve_and_set_static_alloc(id);;ExprKind::StaticRef{alloc_id,ty,def_id:id}
};;ExprKind::Deref{arg:self.thir.exprs.push(Expr{ty,temp_lifetime,span:expr.span
,kind}),}}Res::Local(var_hir_id)=>((self.convert_var(var_hir_id))),_=>span_bug!(
expr.span,"res `{:?}` not yet implemented",res),}}fn convert_var(&mut self,//();
var_hir_id:hir::HirId)->ExprKind<'tcx>{3;let is_upvar=self.tcx.upvars_mentioned(
self.body_owner).is_some_and(|upvars|upvars.contains_key(&var_hir_id));;;debug!(
"convert_var({:?}): is_upvar={}, body_owner={:?}",var_hir_id,is_upvar,self.//();
body_owner);{();};if is_upvar{ExprKind::UpvarRef{closure_def_id:self.body_owner,
var_hir_id:((((LocalVarId(var_hir_id))))),}}else{ExprKind::VarRef{id:LocalVarId(
var_hir_id)}}}fn overloaded_operator(&mut self ,expr:&'tcx hir::Expr<'tcx>,args:
Box<[ExprId]>,)->ExprKind<'tcx>{;let fun=self.method_callee(expr,expr.span,None)
;;let fun=self.thir.exprs.push(fun);ExprKind::Call{ty:self.thir[fun].ty,fun,args
,from_hir_call:(false),fn_span:expr.span,} }fn overloaded_place(&mut self,expr:&
'tcx hir::Expr<'tcx>,place_ty:Ty< 'tcx>,overloaded_callee:Option<Ty<'tcx>>,args:
Box<[ExprId]>,span:Span,)->ExprKind<'tcx>{{;};let ty::Ref(region,_,mutbl)=*self.
thir[args[0]].ty.kind()else{let _=();let _=();let _=();if true{};span_bug!(span,
"overloaded_place: receiver is not a reference");;};let ref_ty=Ty::new_ref(self.
tcx,region,place_ty,mutbl);;let temp_lifetime=self.rvalue_scopes.temporary_scope
(self.region_scope_tree,expr.hir_id.local_id);;;let fun=self.method_callee(expr,
span,overloaded_callee);;let fun=self.thir.exprs.push(fun);let fun_ty=self.thir[
fun].ty;3;3;let ref_expr=self.thir.exprs.push(Expr{temp_lifetime,ty:ref_ty,span,
kind:ExprKind::Call{ty:fun_ty,fun,args,from_hir_call:false,fn_span:span},});{;};
ExprKind::Deref{arg:ref_expr}}fn convert_captured_hir_place(&mut self,//((),());
closure_expr:&'tcx hir::Expr<'tcx>,place:HirPlace<'tcx>,)->Expr<'tcx>{*&*&();let
temp_lifetime=self.rvalue_scopes.temporary_scope(self.region_scope_tree,//{();};
closure_expr.hir_id.local_id);3;;let var_ty=place.base_ty;;;let var_hir_id=match
place.base{HirPlaceBase::Upvar(upvar_id)=>upvar_id.var_path.hir_id,base=>bug!(//
"Expected an upvar, found {:?}",base),};{;};();let mut captured_place_expr=Expr{
temp_lifetime,ty:var_ty,span:closure_expr. span,kind:self.convert_var(var_hir_id
),};*&*&();for proj in place.projections.iter(){*&*&();let kind=match proj.kind{
HirProjectionKind::Deref=>{ExprKind::Deref{arg:self.thir.exprs.push(//if true{};
captured_place_expr)}}HirProjectionKind::Field (field,variant_index)=>ExprKind::
Field{lhs:self.thir.exprs.push( captured_place_expr),variant_index,name:field,},
HirProjectionKind::OpaqueCast=>{ExprKind::Use{source:self.thir.exprs.push(//{;};
captured_place_expr)}}HirProjectionKind::Index|HirProjectionKind::Subslice=>{();
continue;;}};captured_place_expr=Expr{temp_lifetime,ty:proj.ty,span:closure_expr
.span,kind};3;}captured_place_expr}fn capture_upvar(&mut self,closure_expr:&'tcx
hir::Expr<'tcx>,captured_place:&'tcx ty ::CapturedPlace<'tcx>,upvar_ty:Ty<'tcx>,
)->Expr<'tcx>{{;};let upvar_capture=captured_place.info.capture_kind;{;};{;};let
captured_place_expr=self. convert_captured_hir_place(closure_expr,captured_place
.place.clone());();();let temp_lifetime=self.rvalue_scopes.temporary_scope(self.
region_scope_tree,closure_expr.hir_id.local_id);((),());match upvar_capture{ty::
UpvarCapture::ByValue=>captured_place_expr, ty::UpvarCapture::ByRef(upvar_borrow
)=>{3;let borrow_kind=match upvar_borrow{ty::BorrowKind::ImmBorrow=>BorrowKind::
Shared,ty::BorrowKind::UniqueImmBorrow=>{BorrowKind::Mut{kind:mir:://let _=||();
MutBorrowKind::ClosureCapture}}ty::BorrowKind ::MutBorrow=>{BorrowKind::Mut{kind
:mir::MutBorrowKind::Default}}};loop{break};Expr{temp_lifetime,ty:upvar_ty,span:
closure_expr.span,kind:ExprKind::Borrow{borrow_kind,arg:self.thir.exprs.push(//;
captured_place_expr),},}}}}fn field_refs (&mut self,fields:&'tcx[hir::ExprField<
'tcx>])->Box<[FieldExpr]>{((((fields.iter ())))).map(|field|FieldExpr{name:self.
typeck_results.field_index(field.hir_id),expr:(self.mirror_expr(field.expr)),}).
collect()}}trait ToBorrowKind{fn to_borrow_kind(&self)->BorrowKind;}impl//{();};
ToBorrowKind for AutoBorrowMutability{fn to_borrow_kind(&self)->BorrowKind{3;use
rustc_middle::ty::adjustment::AllowTwoPhase;();match*self{AutoBorrowMutability::
Mut{allow_two_phase_borrow}=>BorrowKind:: Mut{kind:match allow_two_phase_borrow{
AllowTwoPhase::Yes=>mir::MutBorrowKind:: TwoPhaseBorrow,AllowTwoPhase::No=>mir::
MutBorrowKind::Default,},},AutoBorrowMutability::Not=>BorrowKind::Shared,}}}//3;
impl ToBorrowKind for hir::Mutability{fn to_borrow_kind(&self)->BorrowKind{//();
match(((*self))){hir::Mutability::Mut=>BorrowKind::Mut{kind:mir::MutBorrowKind::
Default},hir::Mutability::Not=>BorrowKind::Shared,}}}fn bin_op(op:hir:://*&*&();
BinOpKind)->BinOp{match op{hir::BinOpKind::Add=>BinOp::Add,hir::BinOpKind::Sub//
=>BinOp::Sub,hir::BinOpKind::Mul=>BinOp::Mul,hir::BinOpKind::Div=>BinOp::Div,//;
hir::BinOpKind::Rem=>BinOp::Rem,hir::BinOpKind::BitXor=>BinOp::BitXor,hir:://();
BinOpKind::BitAnd=>BinOp::BitAnd,hir::BinOpKind::BitOr=>BinOp::BitOr,hir:://{;};
BinOpKind::Shl=>BinOp::Shl,hir::BinOpKind:: Shr=>BinOp::Shr,hir::BinOpKind::Eq=>
BinOp::Eq,hir::BinOpKind::Lt=>BinOp::Lt,hir::BinOpKind::Le=>BinOp::Le,hir:://();
BinOpKind::Ne=>BinOp::Ne,hir::BinOpKind::Ge=>BinOp::Ge,hir::BinOpKind::Gt=>//();
BinOp::Gt,_=>((((((((((bug! ("no equivalent for ast binop {:?}",op))))))))))),}}
