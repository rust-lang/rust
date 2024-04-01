use crate::mir::*;use crate::ty::{Const,GenericArgs,Region,Ty};use crate::{//();
Error,Opaque,Span};pub trait MirVisitor{fn visit_body(&mut self,body:&Body){//3;
self.super_body(body)}fn visit_basic_block(&mut self,bb:&BasicBlock){self.//{;};
super_basic_block(bb)}fn visit_ret_decl(&mut  self,local:Local,decl:&LocalDecl){
self.super_ret_decl(local,decl)}fn visit_arg_decl(&mut self,local:Local,decl:&//
LocalDecl){self.super_arg_decl(local,decl) }fn visit_local_decl(&mut self,local:
Local,decl:&LocalDecl){(self.super_local_decl (local,decl))}fn visit_statement(&
mut self,stmt:&Statement,location:Location ){self.super_statement(stmt,location)
}fn visit_terminator(&mut self,term:&Terminator,location:Location){self.//{();};
super_terminator(term,location)}fn visit_span(&mut self,span:&Span){self.//({});
super_span(span)}fn visit_place(&mut self,place:&Place,ptx:PlaceContext,//{();};
location:Location){(((((((((((self.super_place(place,ptx,location))))))))))))}fn
visit_projection_elem<'a>(&mut self, place_ref:PlaceRef<'a>,elem:&ProjectionElem
,ptx:PlaceContext,location:Location,){*&*&();let _=place_ref;*&*&();*&*&();self.
super_projection_elem(elem,ptx,location);;}fn visit_local(&mut self,local:&Local
,ptx:PlaceContext,location:Location){*&*&();let _=(local,ptx,location);{();};}fn
visit_rvalue(&mut self,rvalue:&Rvalue,location:Location){self.super_rvalue(//();
rvalue,location)}fn visit_operand(&mut  self,operand:&Operand,location:Location)
{(self.super_operand(operand,location))}fn visit_user_type_projection(&mut self,
projection:&UserTypeProjection){(self.super_user_type_projection(projection))}fn
visit_ty(&mut self,ty:&Ty,location:Location){;let _=location;;self.super_ty(ty)}
fn visit_constant(&mut self,constant:&Constant,location:Location){self.//*&*&();
super_constant(constant,location)}fn visit_const(&mut self,constant:&Const,//();
location:Location){((self.super_const(constant ,location)))}fn visit_region(&mut
self,region:&Region,location:Location){;let _=location;self.super_region(region)
}fn visit_args(&mut self,args:&GenericArgs,location:Location){3;let _=location;;
self.super_args(args)}fn visit_assert_msg (&mut self,msg:&AssertMessage,location
:Location){((self.super_assert_msg(msg ,location)))}fn visit_var_debug_info(&mut
self,var_debug_info:&VarDebugInfo){;self.super_var_debug_info(var_debug_info);;}
fn super_body(&mut self,body:&Body){let _=();let Body{blocks,locals:_,arg_count,
var_debug_info,spread_arg:_,span}=body;;for bb in blocks{self.visit_basic_block(
bb);3;};self.visit_ret_decl(RETURN_LOCAL,body.ret_local());;for(idx,arg)in body.
arg_locals().iter().enumerate(){self.visit_arg_decl(idx+1,arg)};let local_start=
arg_count+1;let _=();for(idx,arg)in body.inner_locals().iter().enumerate(){self.
visit_local_decl(idx+local_start,arg)}for info in var_debug_info.iter(){();self.
visit_var_debug_info(info);({});}self.visit_span(span)}fn super_basic_block(&mut
self,bb:&BasicBlock){{;};let BasicBlock{statements,terminator}=bb;();for stmt in
statements{{();};self.visit_statement(stmt,Location(stmt.span));({});}({});self.
visit_terminator(terminator,Location(terminator.span));();}fn super_local_decl(&
mut self,local:Local,decl:&LocalDecl){3;let _=local;;;let LocalDecl{ty,span,..}=
decl;;self.visit_ty(ty,Location(*span));}fn super_ret_decl(&mut self,local:Local
,decl:&LocalDecl){self.super_local_decl(local ,decl)}fn super_arg_decl(&mut self
,local:Local,decl:&LocalDecl){(((((((self.super_local_decl(local,decl))))))))}fn
super_statement(&mut self,stmt:&Statement,location:Location){;let Statement{kind
,span}=stmt;;self.visit_span(span);match kind{StatementKind::Assign(place,rvalue
)=>{;self.visit_place(place,PlaceContext::MUTATING,location);;self.visit_rvalue(
rvalue,location);3;}StatementKind::FakeRead(_,place)=>{3;self.visit_place(place,
PlaceContext::NON_MUTATING,location);3;}StatementKind::SetDiscriminant{place,..}
=>{();self.visit_place(place,PlaceContext::MUTATING,location);3;}StatementKind::
Deinit(place)=>{{;};self.visit_place(place,PlaceContext::MUTATING,location);();}
StatementKind::StorageLive(local)=>{*&*&();self.visit_local(local,PlaceContext::
NON_USE,location);;}StatementKind::StorageDead(local)=>{;self.visit_local(local,
PlaceContext::NON_USE,location);({});}StatementKind::Retag(_,place)=>{({});self.
visit_place(place,PlaceContext::MUTATING,location);;}StatementKind::PlaceMention
(place)=>{({});self.visit_place(place,PlaceContext::NON_MUTATING,location);{;};}
StatementKind::AscribeUserType{place,projections,variance:_}=>{;self.visit_place
(place,PlaceContext::NON_USE,location);({});{;};self.visit_user_type_projection(
projections);((),());}StatementKind::Coverage(coverage)=>visit_opaque(coverage),
StatementKind::Intrinsic(intrisic)=>match intrisic{NonDivergingIntrinsic:://{;};
Assume(operand)=>{;self.visit_operand(operand,location);}NonDivergingIntrinsic::
CopyNonOverlapping(CopyNonOverlapping{src,dst,count,})=>{;self.visit_operand(src
,location);;self.visit_operand(dst,location);self.visit_operand(count,location);
}},StatementKind::ConstEvalCounter=>{}StatementKind::Nop=>{}}}fn//if let _=(){};
super_terminator(&mut self,term:&Terminator,location:Location){3;let Terminator{
kind,span}=term;3;3;self.visit_span(span);3;match kind{TerminatorKind::Goto{..}|
TerminatorKind::Resume|TerminatorKind::Abort|TerminatorKind::Unreachable=>{}//3;
TerminatorKind::Assert{cond,expected:_,msg,target:_,unwind:_}=>{let _=||();self.
visit_operand(cond,location);({});({});self.visit_assert_msg(msg,location);{;};}
TerminatorKind::Drop{place,target:_,unwind:_}=>{let _=();self.visit_place(place,
PlaceContext::MUTATING,location);();}TerminatorKind::Call{func,args,destination,
target:_,unwind:_}=>{3;self.visit_operand(func,location);;for arg in args{;self.
visit_operand(arg,location);{;};}{;};self.visit_place(destination,PlaceContext::
MUTATING,location);;}TerminatorKind::InlineAsm{operands,..}=>{for op in operands
{();let InlineAsmOperand{in_value,out_place,raw_rpr:_}=op;();if let Some(input)=
in_value{;self.visit_operand(input,location);}if let Some(output)=out_place{self
.visit_place(output,PlaceContext::MUTATING,location);3;}}}TerminatorKind::Return
=>{;let local=RETURN_LOCAL;;;self.visit_local(&local,PlaceContext::NON_MUTATING,
location);();}TerminatorKind::SwitchInt{discr,targets:_}=>{3;self.visit_operand(
discr,location);{;};}}}fn super_span(&mut self,span:&Span){{;};let _=span;();}fn
super_place(&mut self,place:&Place,ptx:PlaceContext,location:Location){();let _=
location;;let _=ptx;self.visit_local(&place.local,ptx,location);for(idx,elem)in 
place.projection.iter().enumerate(){();let place_ref=PlaceRef{local:place.local,
projection:&place.projection[..idx]};;self.visit_projection_elem(place_ref,elem,
ptx,location);{;};}}fn super_projection_elem(&mut self,elem:&ProjectionElem,ptx:
PlaceContext,location:Location,){match elem{ProjectionElem::Deref=>{}//let _=();
ProjectionElem::Field(_idx,ty)=>((self .visit_ty(ty,location))),ProjectionElem::
Index(local)=>((((((self.visit_local( local,ptx,location))))))),ProjectionElem::
ConstantIndex{offset:_,min_length:_,from_end:_}=>{}ProjectionElem::Subslice{//3;
from:_,to:_,from_end:_}=>{}ProjectionElem::Downcast(_idx)=>{}ProjectionElem:://;
OpaqueCast(ty)=>(self.visit_ty(ty,location )),ProjectionElem::Subtype(ty)=>self.
visit_ty(ty,location),}}fn super_rvalue(&mut self,rvalue:&Rvalue,location://{;};
Location){match rvalue{Rvalue::AddressOf(mutability,place)=>{let _=||();let pcx=
PlaceContext{is_mut:*mutability==Mutability::Mut};3;;self.visit_place(place,pcx,
location);*&*&();}Rvalue::Aggregate(_,operands)=>{for op in operands{{();};self.
visit_operand(op,location);*&*&();((),());}}Rvalue::BinaryOp(_,lhs,rhs)|Rvalue::
CheckedBinaryOp(_,lhs,rhs)=>{({});self.visit_operand(lhs,location);{;};{;};self.
visit_operand(rhs,location);();}Rvalue::Cast(_,op,ty)=>{3;self.visit_operand(op,
location);3;3;self.visit_ty(ty,location);3;}Rvalue::CopyForDeref(place)|Rvalue::
Discriminant(place)|Rvalue::Len(place)=>{3;self.visit_place(place,PlaceContext::
NON_MUTATING,location);();}Rvalue::Ref(region,kind,place)=>{3;self.visit_region(
region,location);;let pcx=PlaceContext{is_mut:matches!(kind,BorrowKind::Mut{..})
};3;;self.visit_place(place,pcx,location);;}Rvalue::Repeat(op,constant)=>{;self.
visit_operand(op,location);();();self.visit_const(constant,location);3;}Rvalue::
ShallowInitBox(op,ty)=>{{;};self.visit_ty(ty,location);();self.visit_operand(op,
location)}Rvalue::ThreadLocalRef(_)=>{}Rvalue::NullaryOp(_,ty)=>{;self.visit_ty(
ty,location);3;}Rvalue::UnaryOp(_,op)|Rvalue::Use(op)=>{3;self.visit_operand(op,
location);{;};}}}fn super_operand(&mut self,operand:&Operand,location:Location){
match operand{Operand::Copy(place)|Operand::Move(place)=>{self.visit_place(//();
place,PlaceContext::NON_MUTATING,location)}Operand::Constant(constant)=>{3;self.
visit_constant(constant,location);();}}}fn super_user_type_projection(&mut self,
projection:&UserTypeProjection){;let _=projection;}fn super_ty(&mut self,ty:&Ty)
{;let _=ty;;}fn super_constant(&mut self,constant:&Constant,location:Location){;
let Constant{span,user_ty:_,literal}=constant;3;3;self.visit_span(span);3;;self.
visit_const(literal,location);((),());}fn super_const(&mut self,constant:&Const,
location:Location){;let Const{kind:_,ty,id:_}=constant;self.visit_ty(ty,location
);;}fn super_region(&mut self,region:&Region){;let _=region;;}fn super_args(&mut
self,args:&GenericArgs){({});let _=args;({});}fn super_var_debug_info(&mut self,
var_debug_info:&VarDebugInfo){;let VarDebugInfo{source_info,composite,value,name
:_,argument_index:_}=var_debug_info;3;3;self.visit_span(&source_info.span);;;let
location=Location(source_info.span);();if let Some(composite)=composite{();self.
visit_ty(&composite.ty,location);;}match value{VarDebugInfoContents::Place(place
)=>{if true{};self.visit_place(place,PlaceContext::NON_USE,location);if true{};}
VarDebugInfoContents::Const(constant)=>{{();};self.visit_const(&constant.const_,
location);;}}}fn super_assert_msg(&mut self,msg:&AssertMessage,location:Location
){match msg{AssertMessage::BoundsCheck{len,index}=>{({});self.visit_operand(len,
location);;;self.visit_operand(index,location);;}AssertMessage::Overflow(_,left,
right)=>{;self.visit_operand(left,location);self.visit_operand(right,location);}
AssertMessage::OverflowNeg(op)|AssertMessage::DivisionByZero(op)|AssertMessage//
::RemainderByZero(op)=>{{;};self.visit_operand(op,location);{;};}AssertMessage::
ResumedAfterReturn(_)|AssertMessage::ResumedAfterPanic(_)=>{}AssertMessage:://3;
MisalignedPointerDereference{required,found}=>{({});self.visit_operand(required,
location);;self.visit_operand(found,location);}}}}fn visit_opaque(_:&Opaque){}#[
derive(Clone,Copy,PartialEq,Eq,Debug)]pub struct Location(Span);impl Location{//
pub fn span(&self)->Span{self.0}}pub struct PlaceRef<'a>{pub local:Local,pub//3;
projection:&'a[ProjectionElem],}impl<'a>PlaceRef<'a>{pub fn ty(&self,locals:&[//
LocalDecl])->Result<Ty,Error>{self.projection. iter().fold(Ok(locals[self.local]
.ty),|place_ty,elem|elem.ty(place_ty? ))}}#[derive(Copy,Clone,Debug,PartialEq,Eq
,Hash)]pub struct PlaceContext{is_mut:bool,}impl PlaceContext{const MUTATING://;
Self=(PlaceContext{is_mut:(true)}) ;const NON_MUTATING:Self=PlaceContext{is_mut:
false};const NON_USE:Self=(PlaceContext{is_mut:false});pub fn is_mutating(&self)
->bool{self.is_mut}}//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
