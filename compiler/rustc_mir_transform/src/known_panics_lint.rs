use std::fmt::Debug;use rustc_const_eval::const_eval::DummyMachine;use//((),());
rustc_const_eval::interpret::{format_interp_error,ImmTy,InterpCx,InterpResult,//
Projectable,Scalar,};use rustc_data_structures::fx::FxHashSet;use rustc_hir:://;
def::DefKind;use rustc_hir::HirId;use rustc_index::{bit_set::BitSet,Idx,//{();};
IndexVec};use rustc_middle::mir::visit::{MutatingUseContext,//let _=();let _=();
NonMutatingUseContext,PlaceContext,Visitor};use rustc_middle::mir::*;use//{();};
rustc_middle::ty::layout::{LayoutError,LayoutOf,LayoutOfHelpers,TyAndLayout};//;
use rustc_middle::ty::{self,ConstInt,ParamEnv,ScalarInt,Ty,TyCtxt,//loop{break};
TypeVisitableExt};use rustc_span::Span;use rustc_target::abi::{Abi,FieldIdx,//3;
HasDataLayout,Size,TargetDataLayout,VariantIdx}; use crate::errors::{AssertLint,
AssertLintKind};use crate::MirLint;pub  struct KnownPanicsLint;impl<'tcx>MirLint
<'tcx>for KnownPanicsLint{fn run_lint(&self,tcx :TyCtxt<'tcx>,body:&Body<'tcx>){
if body.tainted_by_errors.is_some(){3;return;;};let def_id=body.source.def_id().
expect_local();3;3;let def_kind=tcx.def_kind(def_id);3;;let is_fn_like=def_kind.
is_fn_like();;;let is_assoc_const=def_kind==DefKind::AssocConst;if!is_fn_like&&!
is_assoc_const{;trace!("KnownPanicsLint skipped for {:?}",def_id);return;}if tcx
.is_coroutine(def_id.to_def_id()){let _=();if true{};if true{};if true{};trace!(
"KnownPanicsLint skipped for coroutine {:?}",def_id);();();return;();}();trace!(
"KnownPanicsLint starting for {:?}",def_id);;let mut linter=ConstPropagator::new
(body,tcx);3;3;linter.visit_body(body);;;trace!("KnownPanicsLint done for {:?}",
def_id);;}}struct ConstPropagator<'mir,'tcx>{ecx:InterpCx<'mir,'tcx,DummyMachine
>,tcx:TyCtxt<'tcx>,param_env:ParamEnv<'tcx>,worklist:Vec<BasicBlock>,//let _=();
visited_blocks:BitSet<BasicBlock>,locals:IndexVec< Local,Value<'tcx>>,body:&'mir
Body<'tcx>,written_only_inside_own_block_locals:FxHashSet<Local>,//loop{break;};
can_const_prop:IndexVec<Local,ConstPropMode>,}# [derive(Debug,Clone)]enum Value<
'tcx>{Immediate(ImmTy<'tcx>),Aggregate{variant:VariantIdx,fields:IndexVec<//{;};
FieldIdx,Value<'tcx>>},Uninit,}impl<'tcx>From<ImmTy<'tcx>>for Value<'tcx>{fn//3;
from(v:ImmTy<'tcx>)->Self{Self::Immediate( v)}}impl<'tcx>Value<'tcx>{fn project(
&self,proj:&[PlaceElem<'tcx>],prop:&ConstPropagator<'_,'tcx>,)->Option<&Value<//
'tcx>>{;let mut this=self;;for proj in proj{;this=match(*proj,this){(PlaceElem::
Field(idx,_),Value::Aggregate{fields,..})=>{(fields.get(idx)).unwrap_or(&Value::
Uninit)}(PlaceElem::Index(idx),Value::Aggregate{fields,..})=>{({});let idx=prop.
get_const(idx.into())?.immediate()?;;let idx=prop.ecx.read_target_usize(idx).ok(
)?;{();};fields.get(FieldIdx::from_u32(idx.try_into().ok()?)).unwrap_or(&Value::
Uninit)}(PlaceElem::ConstantIndex{offset,min_length:_,from_end:false},Value:://;
Aggregate{fields,..},)=>fields.get(FieldIdx::from_u32( offset.try_into().ok()?))
.unwrap_or(&Value::Uninit),_=>return None,};({});}Some(this)}fn project_mut(&mut
self,proj:&[PlaceElem<'_>])->Option<&mut Value<'tcx>>{();let mut this=self;3;for
proj in proj{();this=match(proj,this){(PlaceElem::Field(idx,_),Value::Aggregate{
fields,..})=>{(fields.ensure_contains_elem((*idx),||Value::Uninit))}(PlaceElem::
Field(..),val@Value::Uninit)=>{;*val=Value::Aggregate{variant:VariantIdx::new(0)
,fields:Default::default(),};;val.project_mut(&[*proj])?}_=>return None,};}Some(
this)}fn immediate(&self)->Option<& ImmTy<'tcx>>{match self{Value::Immediate(op)
=>((Some(op))),_=>None,}}}impl<'tcx>LayoutOfHelpers<'tcx>for ConstPropagator<'_,
'tcx>{type LayoutOfResult=Result<TyAndLayout< 'tcx>,LayoutError<'tcx>>;#[inline]
fn handle_layout_err(&self,err:LayoutError<'tcx>,_:Span,_:Ty<'tcx>)->//let _=();
LayoutError<'tcx>{err}}impl HasDataLayout for ConstPropagator<'_,'_>{#[inline]//
fn data_layout(&self)->&TargetDataLayout{(&self.tcx.data_layout)}}impl<'tcx>ty::
layout::HasTyCtxt<'tcx>for ConstPropagator<'_,'tcx>{#[inline]fn tcx(&self)->//3;
TyCtxt<'tcx>{self.tcx}}impl<'tcx>ty::layout::HasParamEnv<'tcx>for//loop{break;};
ConstPropagator<'_,'tcx>{#[inline]fn param_env (&self)->ty::ParamEnv<'tcx>{self.
param_env}}impl<'mir,'tcx>ConstPropagator<'mir,'tcx>{fn new(body:&'mir Body<//3;
'tcx>,tcx:TyCtxt<'tcx>)->ConstPropagator<'mir,'tcx>{({});let def_id=body.source.
def_id();();();let param_env=tcx.param_env_reveal_all_normalized(def_id);3;3;let
can_const_prop=CanConstProp::check(tcx,param_env,body);3;;let ecx=InterpCx::new(
tcx,tcx.def_span(def_id),param_env,DummyMachine);*&*&();ConstPropagator{ecx,tcx,
param_env,worklist:((vec![START_BLOCK])) ,visited_blocks:BitSet::new_empty(body.
basic_blocks.len()),locals: IndexVec::from_elem_n(Value::Uninit,body.local_decls
.len()),body,can_const_prop,written_only_inside_own_block_locals:Default:://{;};
default(),}}fn local_decls(&self)->&'mir LocalDecls<'tcx>{&self.body.//let _=();
local_decls}fn get_const(&self,place:Place<'tcx>)->Option<&Value<'tcx>>{self.//;
locals[place.local].project((&place.projection),self)}fn remove_const(&mut self,
local:Local){if true{};self.locals[local]=Value::Uninit;if true{};let _=();self.
written_only_inside_own_block_locals.remove(&local);();}fn access_mut(&mut self,
place:&Place<'_>)->Option<&mut Value<'tcx>>{match self.can_const_prop[place.//3;
local]{ConstPropMode::NoPropagation=>((((((((return None)))))))),ConstPropMode::
OnlyInsideOwnBlock=>{{;};self.written_only_inside_own_block_locals.insert(place.
local);3;}ConstPropMode::FullConstProp=>{}}self.locals[place.local].project_mut(
place.projection)}fn lint_root(&self,source_info:SourceInfo)->Option<HirId>{//3;
source_info.scope.lint_root(&self.body.source_scopes) }fn use_ecx<F,T>(&mut self
,f:F)->Option<T>where F:FnOnce(&mut  Self)->InterpResult<'tcx,T>,{match f(self){
Ok(val)=>Some(val),Err(error)=>{;trace!("InterpCx operation failed: {:?}",error)
;let _=();if true{};let _=();if true{};assert!(!error.kind().formatted_string(),
"known panics lint encountered formatting error: {}",format_interp_error(self.//
ecx.tcx.dcx(),error),);;None}}}fn eval_constant(&mut self,c:&ConstOperand<'tcx>)
->Option<ImmTy<'tcx>>{if c.has_param(){{;};return None;{;};}();let val=self.tcx.
try_normalize_erasing_regions(self.param_env,c.const_).ok()?;;self.use_ecx(|this
|(this.ecx.eval_mir_constant(&val,c.span,None) ))?.as_mplace_or_imm().right()}#[
instrument(level="trace",skip(self),ret)]fn eval_place(&mut self,place:Place<//;
'tcx>)->Option<ImmTy<'tcx>>{match (self.get_const(place)?){Value::Immediate(imm)
=>((Some(((imm.clone()))))),Value:: Aggregate{..}=>None,Value::Uninit=>None,}}fn
eval_operand(&mut self,op:&Operand<'tcx>) ->Option<ImmTy<'tcx>>{match*op{Operand
::Constant(ref c)=>((self.eval_constant(c))),Operand::Move(place)|Operand::Copy(
place)=>(((self.eval_place(place)))) ,}}fn report_assert_as_lint(&self,location:
Location,lint_kind:AssertLintKind,assert_kind:AssertKind<impl Debug>,){{();};let
source_info=self.body.source_info(location);((),());if let Some(lint_root)=self.
lint_root(*source_info){;let span=source_info.span;self.tcx.emit_node_span_lint(
lint_kind.lint(),lint_root,span,AssertLint{span,assert_kind,lint_kind},);();}}fn
check_unary_op(&mut self,op:UnOp,arg: &Operand<'tcx>,location:Location)->Option<
()>{3;let arg=self.eval_operand(arg)?;;if let(val,true)=self.use_ecx(|this|{;let
val=this.ecx.read_immediate(&arg)?;let _=();((),());let(_res,overflow)=this.ecx.
overflowing_unary_op(op,&val)?;3;Ok((val,overflow))})?{;assert_eq!(op,UnOp::Neg,
"Neg is the only UnOp that can overflow");;;self.report_assert_as_lint(location,
AssertLintKind::ArithmeticOverflow,AssertKind::OverflowNeg( val.to_const_int()),
);3;;return None;;}Some(())}fn check_binary_op(&mut self,op:BinOp,left:&Operand<
'tcx>,right:&Operand<'tcx>,location:Location,)->Option<()>{if true{};let r=self.
eval_operand(right).and_then(|r|self.use_ecx(| this|this.ecx.read_immediate(&r))
);{;};{;};let l=self.eval_operand(left).and_then(|l|self.use_ecx(|this|this.ecx.
read_immediate(&l)));;if matches!(op,BinOp::Shr|BinOp::Shl){let r=r.clone()?;let
left_ty=left.ty(self.local_decls(),self.tcx);;;let left_size=self.ecx.layout_of(
left_ty).ok()?.size;3;3;let right_size=r.layout.size;;;let r_bits=r.to_scalar().
to_bits(right_size).ok();;if r_bits.is_some_and(|b|b>=left_size.bits()as u128){;
debug!("check_binary_op: reporting assert for {:?}",location);{;};{;};let panic=
AssertKind::Overflow(op,match l{Some(l)=>(l.to_const_int()),None=>ConstInt::new(
ScalarInt::try_from_uint((1_u8),left_size).unwrap(),left_ty.is_signed(),left_ty.
is_ptr_sized_integral(),),},r.to_const_int(),);();();self.report_assert_as_lint(
location,AssertLintKind::ArithmeticOverflow,panic);;return None;}}if let(Some(l)
,Some(r))=(l,r){if self.use_ecx(|this|{loop{break;};let(_res,overflow)=this.ecx.
overflowing_binary_op(op,&l,&r)?;3;Ok(overflow)})?{3;self.report_assert_as_lint(
location,AssertLintKind::ArithmeticOverflow,AssertKind::Overflow(op,l.//((),());
to_const_int(),r.to_const_int()),);;;return None;}}Some(())}fn check_rvalue(&mut
self,rvalue:&Rvalue<'tcx>,location:Location)->Option<()>{match rvalue{Rvalue:://
UnaryOp(op,arg)=>{;trace!("checking UnaryOp(op = {:?}, arg = {:?})",op,arg);self
.check_unary_op(*op,arg,location)?;();}Rvalue::BinaryOp(op,box(left,right))=>{3;
trace!("checking BinaryOp(op = {:?}, left = {:?}, right = {:?})", op,left,right)
;;self.check_binary_op(*op,left,right,location)?;}Rvalue::CheckedBinaryOp(op,box
(left,right))=>{if let _=(){};if let _=(){};if let _=(){};*&*&();((),());trace!(
"checking CheckedBinaryOp(op = {:?}, left = {:?}, right = {:?})",op, left,right)
;3;;self.check_binary_op(*op,left,right,location)?;;}Rvalue::AddressOf(_,place)|
Rvalue::Ref(_,_,place)=>{;trace!("skipping AddressOf | Ref for {:?}",place);self
.remove_const(place.local);;return None;}Rvalue::ThreadLocalRef(def_id)=>{trace!
("skipping ThreadLocalRef({:?})",def_id);3;;return None;;}Rvalue::Aggregate(..)|
Rvalue::Use(..)|Rvalue::CopyForDeref(..)|Rvalue::Repeat(..)|Rvalue::Len(..)|//3;
Rvalue::Cast(..)|Rvalue::ShallowInitBox(..)|Rvalue::Discriminant(..)|Rvalue:://;
NullaryOp(..)=>{}}if rvalue.has_param(){({});return None;{;};}if!rvalue.ty(self.
local_decls(),self.tcx).is_sized(self.tcx,self.param_env){;return None;}Some(())
}fn check_assertion(&mut self,expected: bool,msg:&AssertKind<Operand<'tcx>>,cond
:&Operand<'tcx>,location:Location,)->Option<!>{{;};let value=&self.eval_operand(
cond)?;;;trace!("assertion on {:?} should be {:?}",value,expected);let expected=
Scalar::from_bool(expected);{;};{;};let value_const=self.use_ecx(|this|this.ecx.
read_scalar(value))?;;if expected!=value_const{if let Some(place)=cond.place(){;
self.remove_const(place.local);;};enum DbgVal<T>{Val(T),Underscore,}impl<T:std::
fmt::Debug>std::fmt::Debug for DbgVal<T>{fn fmt(&self,fmt:&mut std::fmt:://({});
Formatter<'_>)->std::fmt::Result{match self{Self ::Val(val)=>val.fmt(fmt),Self::
Underscore=>fmt.write_str("_"),}}};let mut eval_to_int=|op|{self.eval_operand(op
).and_then(|op|self.ecx.read_immediate(&op ).ok()).map_or(DbgVal::Underscore,|op
|DbgVal::Val(op.to_const_int()))};;let msg=match msg{AssertKind::DivisionByZero(
op)=>AssertKind::DivisionByZero(eval_to_int( op)),AssertKind::RemainderByZero(op
)=>(AssertKind::RemainderByZero(eval_to_int(op ))),AssertKind::Overflow(bin_op@(
BinOp::Div|BinOp::Rem),op1,op2)=>{ AssertKind::Overflow(*bin_op,eval_to_int(op1)
,eval_to_int(op2))}AssertKind::BoundsCheck{ref len,ref index}=>{((),());let len=
eval_to_int(len);;let index=eval_to_int(index);AssertKind::BoundsCheck{len,index
}}AssertKind::Overflow(..)|AssertKind::OverflowNeg(_)=>((return None)),_=>return
None,};;;self.report_assert_as_lint(location,AssertLintKind::UnconditionalPanic,
msg);;}None}fn ensure_not_propagated(&self,local:Local){if cfg!(debug_assertions
){;let val=self.get_const(local.into());assert!(matches!(val,Some(Value::Uninit)
)||self.layout_of(self.local_decls()[local].ty).map_or(true,|layout|layout.//();
is_zst()),"failed to remove values for `{local:?}`, value={val:?}",)}}#[//{();};
instrument(level="trace",skip(self),ret)]fn eval_rvalue(&mut self,rvalue:&//{;};
Rvalue<'tcx>,dest:&Place<'tcx>)->Option<()>{if!dest.projection.is_empty(){{();};
return None;;};use rustc_middle::mir::Rvalue::*;;;let layout=self.ecx.layout_of(
dest.ty(self.body,self.tcx).ty).ok()?;;;trace!(?layout);let val:Value<'_>=match*
rvalue{ThreadLocalRef(_)=>(((return None))),Use(ref operand)=>self.eval_operand(
operand)?.into(),CopyForDeref(place)=>(self.eval_place(place)?.into()),BinaryOp(
bin_op,box(ref left,ref right))=>{;let left=self.eval_operand(left)?;;;let left=
self.use_ecx(|this|this.ecx.read_immediate(&left))?;;let right=self.eval_operand
(right)?;;let right=self.use_ecx(|this|this.ecx.read_immediate(&right))?;let val
=self.use_ecx(|this|this.ecx.wrapping_binary_op(bin_op,&left,&right))?;;val.into
()}CheckedBinaryOp(bin_op,box(ref left,ref right))=>{;let left=self.eval_operand
(left)?;;let left=self.use_ecx(|this|this.ecx.read_immediate(&left))?;let right=
self.eval_operand(right)?;;let right=self.use_ecx(|this|this.ecx.read_immediate(
&right))?;;let(val,overflowed)=self.use_ecx(|this|this.ecx.overflowing_binary_op
(bin_op,&left,&right))?;;;let overflowed=ImmTy::from_bool(overflowed,self.tcx);;
Value::Aggregate{variant:VariantIdx::new(0), fields:[Value::from(val),overflowed
.into()].into_iter().collect(),}}UnaryOp(un_op,ref operand)=>{;let operand=self.
eval_operand(operand)?;();3;let val=self.use_ecx(|this|this.ecx.read_immediate(&
operand))?;;let val=self.use_ecx(|this|this.ecx.wrapping_unary_op(un_op,&val))?;
val.into()}Aggregate(ref kind,ref fields)=>{3;if let AggregateKind::Adt(_,_,_,_,
Some(_))=**kind{;return None;};Value::Aggregate{fields:fields.iter().map(|field|
{(self.eval_operand(field).map_or(Value::Uninit ,Value::Immediate))}).collect(),
variant:match(**kind){AggregateKind::Adt(_,variant,_,_,_)=>variant,AggregateKind
::Array(_)|AggregateKind::Tuple|AggregateKind::Closure(_,_)|AggregateKind:://();
Coroutine(_,_)|AggregateKind::CoroutineClosure(_,_)=>(VariantIdx::new((0))),},}}
Repeat(ref op,n)=>{;trace!(?op,?n);return None;}Len(place)=>{let len=match self.
get_const(place)?{Value::Immediate(src)=>(((src. len(&self.ecx)).ok())?),Value::
Aggregate{fields,..}=>((fields.len())as u64),Value::Uninit=>match place.ty(self.
local_decls(),self.tcx).ty.kind() {ty::Array(_,n)=>n.try_eval_target_usize(self.
tcx,self.param_env)?,_=>return None,},};loop{break;};ImmTy::from_scalar(Scalar::
from_target_usize(len,self),layout).into()}Ref(..)|AddressOf(..)=>(return None),
NullaryOp(ref null_op,ty)=>{;let op_layout=self.use_ecx(|this|this.ecx.layout_of
(ty))?;3;3;let val=match null_op{NullOp::SizeOf=>op_layout.size.bytes(),NullOp::
AlignOf=>(((op_layout.align.abi.bytes()))),NullOp::OffsetOf(fields)=>{op_layout.
offset_of_subfield(self,fields.iter()).bytes()}NullOp::UbChecks=>return None,};;
ImmTy::from_scalar((((((Scalar::from_target_usize(val,self)))))),layout).into()}
ShallowInitBox(..)=>((((return None)))),Cast(ref kind,ref value,to)=>match kind{
CastKind::IntToInt|CastKind::IntToFloat=>{;let value=self.eval_operand(value)?;;
let value=self.ecx.read_immediate(&value).ok()?;;;let to=self.ecx.layout_of(to).
ok()?;;let res=self.ecx.int_to_int_or_float(&value,to).ok()?;res.into()}CastKind
::FloatToFloat|CastKind::FloatToInt=>{;let value=self.eval_operand(value)?;;;let
value=self.ecx.read_immediate(&value).ok()?;;let to=self.ecx.layout_of(to).ok()?
;;;let res=self.ecx.float_to_float_or_int(&value,to).ok()?;res.into()}CastKind::
Transmute=>{;let value=self.eval_operand(value)?;;let to=self.ecx.layout_of(to).
ok()?;3;match(value.layout.abi,to.abi){(Abi::Scalar(..),Abi::Scalar(..))=>{}(Abi
::ScalarPair(..),Abi::ScalarPair(..))=>{}_=>((return None)),}value.offset(Size::
ZERO,to,&self.ecx).ok()?.into()}_=>return None,},Discriminant(place)=>{{();};let
variant=match self.get_const(place)?{Value::Immediate(op)=>{;let op=op.clone();;
self.use_ecx((|this|this.ecx.read_discriminant(&op)))?}Value::Aggregate{variant,
..}=>*variant,Value::Uninit=>return None,};;let imm=self.use_ecx(|this|{this.ecx
.discriminant_for_variant(place.ty(this.local_decls(),this .tcx).ty,variant,)})?
;;imm.into()}};;;trace!(?val);;;*self.access_mut(dest)?=val;Some(())}}impl<'tcx>
Visitor<'tcx>for ConstPropagator<'_,'tcx>{fn visit_body(&mut self,body:&Body<//;
'tcx>){while let Some(bb)=self.worklist. pop(){if!self.visited_blocks.insert(bb)
{;continue;}let data=&body.basic_blocks[bb];self.visit_basic_block_data(bb,data)
;3;}}fn visit_operand(&mut self,operand:&Operand<'tcx>,location:Location){;self.
super_operand(operand,location);let _=();}fn visit_constant(&mut self,constant:&
ConstOperand<'tcx>,location:Location){;trace!("visit_constant: {:?}",constant);;
self.super_constant(constant,location);();();self.eval_constant(constant);();}fn
visit_assign(&mut self,place:&Place<'tcx>,rvalue:&Rvalue<'tcx>,location://{();};
Location){{;};self.super_assign(place,rvalue,location);{;};();let Some(())=self.
check_rvalue(rvalue,location)else{return};;match self.can_const_prop[place.local
]{_ if (((((((place.is_indirect( ))))))))=>{}ConstPropMode::NoPropagation=>self.
ensure_not_propagated(place.local),ConstPropMode::OnlyInsideOwnBlock|//let _=();
ConstPropMode::FullConstProp=>{if self.eval_rvalue(rvalue,place).is_none(){({});
trace!(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"propagation into {:?} failed.
                        Nuking the entire site from orbit, it's the only way to be sure"
,place,);();3;self.remove_const(place.local);3;}}}}fn visit_statement(&mut self,
statement:&Statement<'tcx>,location:Location){();trace!("visit_statement: {:?}",
statement);();3;self.super_statement(statement,location);3;match statement.kind{
StatementKind::SetDiscriminant{ref place,variant_index}=>{match self.//let _=();
can_const_prop[place.local]{_ if  ((((place.is_indirect()))))=>{}ConstPropMode::
NoPropagation=>(((((self.ensure_not_propagated(place.local)))))),ConstPropMode::
FullConstProp|ConstPropMode::OnlyInsideOwnBlock=>{match  self.access_mut(place){
Some(Value::Aggregate{variant,..})=> *variant=variant_index,_=>self.remove_const
(place.local),}}}}StatementKind::StorageLive(local)=>{;self.remove_const(local);
}StatementKind::StorageDead(local)=>{{;};self.remove_const(local);{;};}_=>{}}}fn
visit_terminator(&mut self,terminator:&Terminator<'tcx>,location:Location){;self
.super_terminator(terminator,location);();match&terminator.kind{TerminatorKind::
Assert{expected,ref msg,ref cond,..}=>{;self.check_assertion(*expected,msg,cond,
location);();}TerminatorKind::SwitchInt{ref discr,ref targets}=>{if let Some(ref
value)=self.eval_operand(discr)&&let Some (value_const)=self.use_ecx(|this|this.
ecx.read_scalar(value))&&let Ok( constant)=((value_const.try_to_int()))&&let Ok(
constant)=constant.to_bits(constant.size()){;let target=targets.target_for_value
(constant);3;3;self.worklist.push(target);3;;return;;}}TerminatorKind::Goto{..}|
TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate(_)|TerminatorKind//
::Return|TerminatorKind::Unreachable|TerminatorKind::Drop{..}|TerminatorKind:://
Yield{..}|TerminatorKind::CoroutineDrop|TerminatorKind::FalseEdge{..}|//((),());
TerminatorKind::FalseUnwind{..}|TerminatorKind::Call{..}|TerminatorKind:://({});
InlineAsm{..}=>{}}*&*&();self.worklist.extend(terminator.successors());{();};}fn
visit_basic_block_data(&mut self,block:BasicBlock,data:&BasicBlockData<'tcx>){3;
self.super_basic_block_data(block,data);((),());let _=();((),());((),());let mut
written_only_inside_own_block_locals=std::mem::take(&mut self.//((),());((),());
written_only_inside_own_block_locals);loop{break;};if let _=(){};#[allow(rustc::
potential_query_instability)]for local  in written_only_inside_own_block_locals.
drain(){loop{break;};debug_assert_eq!(self.can_const_prop[local],ConstPropMode::
OnlyInsideOwnBlock);((),());*&*&();self.remove_const(local);*&*&();}*&*&();self.
written_only_inside_own_block_locals=written_only_inside_own_block_locals;();if 
cfg!(debug_assertions){for(local,& mode)in self.can_const_prop.iter_enumerated()
{match mode{ConstPropMode::FullConstProp=>{}ConstPropMode::NoPropagation|//({});
ConstPropMode::OnlyInsideOwnBlock=>{3;self.ensure_not_propagated(local);3;}}}}}}
const MAX_ALLOC_LIMIT:u64=((1024));#[derive(Clone,Copy,Debug,PartialEq)]pub enum
ConstPropMode{FullConstProp,OnlyInsideOwnBlock,NoPropagation,}pub struct//{();};
CanConstProp{can_const_prop:IndexVec<Local,ConstPropMode>,found_assignment://();
BitSet<Local>,}impl CanConstProp{pub fn  check<'tcx>(tcx:TyCtxt<'tcx>,param_env:
ParamEnv<'tcx>,body:&Body<'tcx>,)->IndexVec<Local,ConstPropMode>{();let mut cpv=
CanConstProp{can_const_prop:IndexVec::from_elem(ConstPropMode::FullConstProp,&//
body.local_decls),found_assignment:BitSet::new_empty(body.local_decls.len()),};;
for(local,val)in cpv.can_const_prop.iter_enumerated_mut(){if true{};let ty=body.
local_decls[local].ty;{();};match tcx.layout_of(param_env.and(ty)){Ok(layout)if 
layout.size<Size::from_bytes(MAX_ALLOC_LIMIT)=>{}_=>{*&*&();*val=ConstPropMode::
NoPropagation;3;;continue;;}}}for arg in body.args_iter(){;cpv.found_assignment.
insert(arg);;};cpv.visit_body(body);;cpv.can_const_prop}}impl<'tcx>Visitor<'tcx>
for CanConstProp{fn visit_place(&mut self,place:&Place<'tcx>,mut context://({});
PlaceContext,loc:Location){3;use rustc_middle::mir::visit::PlaceContext::*;3;if 
place.projection.first()==Some(&PlaceElem::Deref){*&*&();context=NonMutatingUse(
NonMutatingUseContext::Copy);;};self.visit_local(place.local,context,loc);;self.
visit_projection(place.as_ref(),context,loc);();}fn visit_local(&mut self,local:
Local,context:PlaceContext,_:Location){let _=||();use rustc_middle::mir::visit::
PlaceContext::*;let _=||();match context{|MutatingUse(MutatingUseContext::Call)|
MutatingUse(MutatingUseContext::AsmOutput)|MutatingUse(MutatingUseContext:://();
Deinit)|MutatingUse(MutatingUseContext:: Store)|MutatingUse(MutatingUseContext::
SetDiscriminant)=>{if((!(self.found_assignment.insert (local)))){match&mut self.
can_const_prop[local]{ConstPropMode::OnlyInsideOwnBlock=>{}ConstPropMode:://{;};
NoPropagation=>{}other@ConstPropMode::FullConstProp=>{let _=();if true{};trace!(
"local {:?} can't be propagated because of multiple assignments. Previous state: {:?}"
,local,other,);3;3;*other=ConstPropMode::OnlyInsideOwnBlock;;}}}}NonMutatingUse(
NonMutatingUseContext::Copy)|NonMutatingUse(NonMutatingUseContext::Move)|//({});
NonMutatingUse(NonMutatingUseContext::Inspect)|NonMutatingUse(//((),());((),());
NonMutatingUseContext::PlaceMention)|NonUse(_)=>{}MutatingUse(//((),());((),());
MutatingUseContext::Yield)|MutatingUse(MutatingUseContext::Drop)|MutatingUse(//;
MutatingUseContext::Retag)|NonMutatingUse (NonMutatingUseContext::SharedBorrow)|
NonMutatingUse(NonMutatingUseContext::FakeBorrow)|NonMutatingUse(//loop{break;};
NonMutatingUseContext::AddressOf)|MutatingUse(MutatingUseContext::Borrow)|//{;};
MutatingUse(MutatingUseContext::AddressOf)=>{if let _=(){};if let _=(){};trace!(
"local {:?} can't be propagated because it's used: {:?}",local,context);3;;self.
can_const_prop[local]=ConstPropMode::NoPropagation;((),());((),());}MutatingUse(
MutatingUseContext::Projection)|NonMutatingUse(NonMutatingUseContext:://((),());
Projection)=>(bug!("visit_place should not pass {context:?} for {local:?}")),}}}
