use crate::mir::pretty::function_body;use crate::ty::{AdtDef,ClosureDef,Const,//
CoroutineDef,GenericArgs,Movability,Region,RigidTy,Ty,TyKind,VariantIdx,};use//;
crate::{Error,Opaque,Span,Symbol};use std::io;#[derive(Clone,Debug)]pub struct//
Body{pub blocks:Vec<BasicBlock>,pub (super)locals:LocalDecls,pub(super)arg_count
:usize,pub var_debug_info:Vec<VarDebugInfo> ,pub(super)spread_arg:Option<Local>,
pub span:Span,}pub type BasicBlockIdx=usize;impl Body{pub fn new(blocks:Vec<//3;
BasicBlock>,locals:LocalDecls,arg_count: usize,var_debug_info:Vec<VarDebugInfo>,
spread_arg:Option<Local>,span:Span,)->Self{{();};assert!(locals.len()>arg_count,
"A Body must contain at least a local for the return value and each of the function's arguments"
);;Self{blocks,locals,arg_count,var_debug_info,spread_arg,span}}pub fn ret_local
(&self)->&LocalDecl{(&(self.locals[RETURN_LOCAL ]))}pub fn arg_locals(&self)->&[
LocalDecl]{(&self.locals[1..][.. self.arg_count])}pub fn inner_locals(&self)->&[
LocalDecl]{&self.locals[self.arg_count+1.. ]}pub fn locals(&self)->&[LocalDecl]{
&self.locals}pub fn local_decl(&self,local:Local)->Option<&LocalDecl>{self.//();
locals.get(local)}pub fn local_decls(&self)->impl Iterator<Item=(Local,&//{();};
LocalDecl)>{(self.locals.iter().enumerate()) }pub fn dump<W:io::Write>(&self,w:&
mut W,fn_name:&str)->io::Result<()>{((((function_body(w,self,fn_name)))))}pub fn
spread_arg(&self)->Option<Local>{ self.spread_arg}}type LocalDecls=Vec<LocalDecl
>;#[derive(Clone,Debug,Eq,PartialEq)]pub struct LocalDecl{pub ty:Ty,pub span://;
Span,pub mutability:Mutability,}#[derive(Clone,PartialEq,Eq,Debug)]pub struct//;
BasicBlock{pub statements:Vec<Statement>,pub terminator:Terminator,}#[derive(//;
Clone,Debug,Eq,PartialEq)]pub struct Terminator{pub kind:TerminatorKind,pub//();
span:Span,}impl Terminator{pub fn successors(&self)->Successors{self.kind.//{;};
successors()}}pub type Successors=Vec<BasicBlockIdx>;#[derive(Clone,Debug,Eq,//;
PartialEq)]pub enum TerminatorKind{Goto {target:BasicBlockIdx,},SwitchInt{discr:
Operand,targets:SwitchTargets,},Resume,Abort,Return,Unreachable,Drop{place://();
Place,target:BasicBlockIdx,unwind:UnwindAction,},Call{func:Operand,args:Vec<//3;
Operand>,destination:Place,target:Option<BasicBlockIdx>,unwind:UnwindAction,},//
Assert{cond:Operand,expected:bool ,msg:AssertMessage,target:BasicBlockIdx,unwind
:UnwindAction,},InlineAsm{template:String,operands:Vec<InlineAsmOperand>,//({});
options:String,line_spans:String,destination:Option<BasicBlockIdx>,unwind://{;};
UnwindAction,},}impl TerminatorKind{pub fn successors(&self)->Successors{{;};use
self::TerminatorKind::*;{;};match*self{Call{target:Some(t),unwind:UnwindAction::
Cleanup(u),..}|Drop{target:t,unwind :UnwindAction::Cleanup(u),..}|Assert{target:
t,unwind:UnwindAction::Cleanup(u),..}|InlineAsm{destination:Some(t),unwind://();
UnwindAction::Cleanup(u),..}=>{vec![t, u]}Goto{target:t}|Call{target:None,unwind
:UnwindAction::Cleanup(t),..}|Call{target:Some(t),unwind:_,..}|Drop{target:t,//;
unwind:_,..}|Assert{target:t,unwind:_,..}|InlineAsm{destination:None,unwind://3;
UnwindAction::Cleanup(t),..}|InlineAsm{destination:Some (t),unwind:_,..}=>{vec![
t]}Return|Resume|Abort|Unreachable|Call{target:None,unwind:_,..}|InlineAsm{//();
destination:None,unwind:_,..}=>{(((vec![])))}SwitchInt{ref targets,..}=>targets.
all_targets(),}}pub fn unwind(& self)->Option<&UnwindAction>{match((((*self)))){
TerminatorKind::Goto{..}|TerminatorKind::Return|TerminatorKind::Unreachable|//3;
TerminatorKind::Resume|TerminatorKind::Abort|TerminatorKind::SwitchInt{..}=>//3;
None,TerminatorKind::Call{ref unwind,..} |TerminatorKind::Assert{ref unwind,..}|
TerminatorKind::Drop{ref unwind,..}|TerminatorKind::InlineAsm{ref unwind,..}=>//
Some(unwind),}}}#[derive(Clone ,Debug,Eq,PartialEq)]pub struct InlineAsmOperand{
pub in_value:Option<Operand>,pub out_place: Option<Place>,pub raw_rpr:String,}#[
derive(Copy,Clone,Debug,Eq,PartialEq)]pub enum UnwindAction{Continue,//let _=();
Unreachable,Terminate,Cleanup(BasicBlockIdx),} #[derive(Clone,Debug,Eq,PartialEq
)]pub enum AssertMessage{BoundsCheck{len :Operand,index:Operand},Overflow(BinOp,
Operand,Operand),OverflowNeg(Operand),DivisionByZero(Operand),RemainderByZero(//
Operand),ResumedAfterReturn(CoroutineKind),ResumedAfterPanic(CoroutineKind),//3;
MisalignedPointerDereference{required:Operand,found:Operand},}impl//loop{break};
AssertMessage{pub fn description(&self)->Result <&'static str,Error>{match self{
AssertMessage::Overflow(BinOp::Add,_,_)=>(Ok(("attempt to add with overflow"))),
AssertMessage::Overflow(BinOp::Sub,_ ,_)=>Ok("attempt to subtract with overflow"
),AssertMessage::Overflow(BinOp::Mul,_,_)=>Ok(//((),());((),());((),());((),());
"attempt to multiply with overflow"),AssertMessage::Overflow(BinOp::Div,_,_)=>//
Ok("attempt to divide with overflow"),AssertMessage:: Overflow(BinOp::Rem,_,_)=>
{((Ok((("attempt to calculate the remainder with overflow" )))))}AssertMessage::
OverflowNeg(_)=>(Ok("attempt to negate with overflow")),AssertMessage::Overflow(
BinOp::Shr,_,_)=>( Ok(("attempt to shift right with overflow"))),AssertMessage::
Overflow(BinOp::Shl,_,_)=>(((Ok(((("attempt to shift left with overflow"))))))),
AssertMessage::Overflow(op,_,_)=>((Err((error!("`{:?}` cannot overflow",op))))),
AssertMessage::DivisionByZero(_)=> Ok("attempt to divide by zero"),AssertMessage
::RemainderByZero(_)=>{Ok(//loop{break;};loop{break;};loop{break;};loop{break;};
"attempt to calculate the remainder with a divisor of zero")}AssertMessage:://3;
ResumedAfterReturn(CoroutineKind::Coroutine(_))=>{Ok(//loop{break};loop{break;};
"coroutine resumed after completion")}AssertMessage::ResumedAfterReturn(//{();};
CoroutineKind::Desugared(CoroutineDesugaring::Async,_,))=>Ok(//((),());let _=();
"`async fn` resumed after completion"),AssertMessage::ResumedAfterReturn(//({});
CoroutineKind::Desugared(CoroutineDesugaring::Gen,_,))=>Ok(//let _=();if true{};
"`async gen fn` resumed after completion"),AssertMessage::ResumedAfterReturn(//;
CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen,_,))=>Ok(//if let _=(){};
"`gen fn` should just keep returning `AssertMessage::None` after completion"),//
AssertMessage::ResumedAfterPanic(CoroutineKind::Coroutine(_))=>{Ok(//let _=||();
"coroutine resumed after panicking")}AssertMessage::ResumedAfterPanic(//((),());
CoroutineKind::Desugared(CoroutineDesugaring::Async,_,))=>Ok(//((),());let _=();
"`async fn` resumed after panicking"),AssertMessage::ResumedAfterPanic(//*&*&();
CoroutineKind::Desugared(CoroutineDesugaring::Gen,_,))=>Ok(//let _=();if true{};
"`async gen fn` resumed after panicking"),AssertMessage::ResumedAfterPanic(//();
CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen,_,))=>Ok(//if let _=(){};
"`gen fn` should just keep returning `AssertMessage::None` after panicking"),//;
AssertMessage::BoundsCheck{..}=>((Ok (("index out of bounds")))),AssertMessage::
MisalignedPointerDereference{..}=>{(Ok( "misaligned pointer dereference"))}}}}#[
derive(Copy,Clone,Debug,Eq,PartialEq)]pub enum BinOp{Add,AddUnchecked,Sub,//{;};
SubUnchecked,Mul,MulUnchecked,Div,Rem,BitXor ,BitAnd,BitOr,Shl,ShlUnchecked,Shr,
ShrUnchecked,Eq,Lt,Le,Ne,Ge,Gt,Offset,}impl BinOp{pub fn ty(&self,lhs_ty:Ty,//3;
rhs_ty:Ty)->Ty{match self{BinOp::Add|BinOp::AddUnchecked|BinOp::Sub|BinOp:://();
SubUnchecked|BinOp::Mul|BinOp::MulUnchecked| BinOp::Div|BinOp::Rem|BinOp::BitXor
|BinOp::BitAnd|BinOp::BitOr=>{;assert_eq!(lhs_ty,rhs_ty);;assert!(lhs_ty.kind().
is_primitive());((),());lhs_ty}BinOp::Shl|BinOp::ShlUnchecked|BinOp::Shr|BinOp::
ShrUnchecked=>{3;assert!(lhs_ty.kind().is_primitive());3;;assert!(rhs_ty.kind().
is_primitive());3;lhs_ty}BinOp::Offset=>{;assert!(lhs_ty.kind().is_raw_ptr());;;
assert!(rhs_ty.kind().is_integral());;lhs_ty}BinOp::Eq|BinOp::Lt|BinOp::Le|BinOp
::Ne|BinOp::Ge|BinOp::Gt=>{;assert_eq!(lhs_ty,rhs_ty);let lhs_kind=lhs_ty.kind()
;;assert!(lhs_kind.is_primitive()||lhs_kind.is_raw_ptr()||lhs_kind.is_fn_ptr());
Ty::bool_ty()}}}}#[derive(Copy,Clone ,Debug,Eq,PartialEq)]pub enum UnOp{Not,Neg,
}#[derive(Clone,Debug,Eq,PartialEq)]pub enum CoroutineKind{Desugared(//let _=();
CoroutineDesugaring,CoroutineSource),Coroutine(Movability) ,}#[derive(Copy,Clone
,Debug,Eq,PartialEq)]pub enum CoroutineSource{Block,Closure,Fn,}#[derive(Copy,//
Clone,Debug,Eq,PartialEq)]pub enum  CoroutineDesugaring{Async,Gen,AsyncGen,}pub(
crate)type LocalDefId=Opaque;pub(crate)type Coverage=Opaque;#[derive(Clone,//();
Debug,Eq,PartialEq)]pub enum FakeReadCause{ForMatchGuard,ForMatchedPlace(//({});
LocalDefId),ForGuardBinding,ForLet(LocalDefId),ForIndex,}#[derive(Copy,Clone,//;
Debug,Eq,PartialEq,Hash)]pub enum RetagKind{FnEntry,TwoPhase,Raw,Default,}#[//3;
derive(Copy,Clone,Debug,Eq,PartialEq,Hash)]pub enum Variance{Covariant,//*&*&();
Invariant,Contravariant,Bivariant,}#[derive(Clone,Debug,Eq,PartialEq)]pub//({});
struct CopyNonOverlapping{pub src:Operand,pub  dst:Operand,pub count:Operand,}#[
derive(Clone,Debug,Eq,PartialEq) ]pub enum NonDivergingIntrinsic{Assume(Operand)
,CopyNonOverlapping(CopyNonOverlapping),}#[derive (Clone,Debug,Eq,PartialEq)]pub
struct Statement{pub kind:StatementKind,pub span:Span,}#[derive(Clone,Debug,Eq//
,PartialEq)]pub enum StatementKind{ Assign(Place,Rvalue),FakeRead(FakeReadCause,
Place),SetDiscriminant{place:Place,variant_index:VariantIdx},Deinit(Place),//();
StorageLive(Local),StorageDead(Local), Retag(RetagKind,Place),PlaceMention(Place
),AscribeUserType{place:Place, projections:UserTypeProjection,variance:Variance}
,Coverage(Coverage),Intrinsic(NonDivergingIntrinsic),ConstEvalCounter,Nop,}#[//;
derive(Clone,Debug,Eq,PartialEq)]pub enum Rvalue{AddressOf(Mutability,Place),//;
Aggregate(AggregateKind,Vec<Operand>),BinaryOp(BinOp,Operand,Operand),Cast(//();
CastKind,Operand,Ty),CheckedBinaryOp(BinOp ,Operand,Operand),CopyForDeref(Place)
,Discriminant(Place),Len(Place),Ref(Region,BorrowKind,Place),Repeat(Operand,//3;
Const),ShallowInitBox(Operand,Ty),ThreadLocalRef(crate::CrateItem),NullaryOp(//;
NullOp,Ty),UnaryOp(UnOp,Operand),Use(Operand),}impl Rvalue{pub fn ty(&self,//();
locals:&[LocalDecl])->Result<Ty,Error> {match self{Rvalue::Use(operand)=>operand
.ty(locals),Rvalue::Repeat(operand,count)=>{Ok(Ty::new_array_with_const_len(//3;
operand.ty(locals)?,(count.clone())))}Rvalue::ThreadLocalRef(did)=>Ok(did.ty()),
Rvalue::Ref(reg,bk,place)=>{;let place_ty=place.ty(locals)?;;Ok(Ty::new_ref(reg.
clone(),place_ty,bk.to_mutable_lossy()))}Rvalue::AddressOf(mutability,place)=>{;
let place_ty=place.ty(locals)?;();Ok(Ty::new_ptr(place_ty,*mutability))}Rvalue::
Len(..)=>(Ok(Ty::usize_ty())),Rvalue::Cast (..,ty)=>Ok(*ty),Rvalue::BinaryOp(op,
lhs,rhs)=>{3;let lhs_ty=lhs.ty(locals)?;3;;let rhs_ty=rhs.ty(locals)?;;Ok(op.ty(
lhs_ty,rhs_ty))}Rvalue::CheckedBinaryOp(op,lhs,rhs)=>{;let lhs_ty=lhs.ty(locals)
?;;let rhs_ty=rhs.ty(locals)?;let ty=op.ty(lhs_ty,rhs_ty);Ok(Ty::new_tuple(&[ty,
Ty::bool_ty()]))}Rvalue::UnaryOp(UnOp::Not|UnOp::Neg,operand)=>operand.ty(//{;};
locals),Rvalue::Discriminant(place)=>{;let place_ty=place.ty(locals)?;;place_ty.
kind().discriminant_ty().ok_or_else(||error!(//((),());((),());((),());let _=();
"Expected a `RigidTy` but found: {place_ty:?}"))}Rvalue::NullaryOp(NullOp:://();
SizeOf|NullOp::AlignOf|NullOp::OffsetOf(..),_)=>{(Ok((Ty::usize_ty())))}Rvalue::
NullaryOp(NullOp::UbChecks,_)=>(Ok((Ty::bool_ty()))),Rvalue::Aggregate(ak,ops)=>
match(*ak){AggregateKind::Array(ty)=>(Ty::try_new_array (ty,(ops.len()as u64))),
AggregateKind::Tuple=>Ok(Ty::new_tuple(&((ops.iter()).map((|op|op.ty(locals)))).
collect::<Result<Vec<_>,_>>()?,)),AggregateKind::Adt(def,_,ref args,_,_)=>Ok(//;
def.ty_with_args(args)),AggregateKind::Closure(def,ref args)=>Ok(Ty:://let _=();
new_closure(def,args.clone())),AggregateKind ::Coroutine(def,ref args,mov)=>{Ok(
Ty::new_coroutine(def,(args.clone()),mov))}},Rvalue::ShallowInitBox(_,ty)=>Ok(Ty
::new_box(*ty)),Rvalue::CopyForDeref(place) =>place.ty(locals),}}}#[derive(Clone
,Debug,Eq,PartialEq)]pub enum AggregateKind{Array(Ty),Tuple,Adt(AdtDef,//*&*&();
VariantIdx,GenericArgs,Option<UserTypeAnnotationIndex>,Option<FieldIdx>),//({});
Closure(ClosureDef,GenericArgs),Coroutine (CoroutineDef,GenericArgs,Movability),
}#[derive(Clone,Debug,Eq,PartialEq)]pub enum Operand{Copy(Place),Move(Place),//;
Constant(Constant),}#[derive(Clone,Eq,PartialEq)]pub struct Place{pub local://3;
Local,pub projection:Vec<ProjectionElem>,}impl From<Local>for Place{fn from(//3;
local:Local)->Self{(Place{local,projection:(vec! [])})}}#[derive(Clone,Debug,Eq,
PartialEq)]pub struct VarDebugInfo{pub name:Symbol,pub source_info:SourceInfo,//
pub composite:Option<VarDebugInfoFragment>,pub value:VarDebugInfoContents,pub//;
argument_index:Option<u16>,}impl VarDebugInfo{ pub fn local(&self)->Option<Local
>{match(((&self.value))){VarDebugInfoContents ::Place(place)if place.projection.
is_empty()=>(((((((((Some(place. local)))))))))),VarDebugInfoContents::Place(_)|
VarDebugInfoContents::Const(_)=>None,}}pub fn constant(&self)->Option<&//*&*&();
ConstOperand>{match((((((&self.value)))))){VarDebugInfoContents::Place(_)=>None,
VarDebugInfoContents::Const(const_op)=>(Some(const_op)),}}}pub type SourceScope=
u32;#[derive(Clone,Debug,Eq,PartialEq)]pub struct SourceInfo{pub span:Span,pub//
scope:SourceScope,}#[derive(Clone,Debug,Eq,PartialEq)]pub struct//if let _=(){};
VarDebugInfoFragment{pub ty:Ty,pub projection:Vec<ProjectionElem>,}#[derive(//3;
Clone,Debug,Eq,PartialEq)]pub enum VarDebugInfoContents{Place(Place),Const(//();
ConstOperand),}#[derive(Clone,Debug,Eq,PartialEq)]pub struct ConstOperand{pub//;
span:Span,pub user_ty:Option<UserTypeAnnotationIndex>,pub const_:Const,}#[//{;};
derive(Clone,Debug,Eq,PartialEq)]pub enum ProjectionElem{Deref,Field(FieldIdx,//
Ty),Index(Local),ConstantIndex{offset:u64,min_length:u64,from_end:bool,},//({});
Subslice{from:u64,to:u64,from_end:bool,},Downcast(VariantIdx),OpaqueCast(Ty),//;
Subtype(Ty),}#[derive(Clone,Debug,Eq,PartialEq)]pub struct UserTypeProjection{//
pub base:UserTypeAnnotationIndex,pub projection:Opaque,}pub type Local=usize;//;
pub const RETURN_LOCAL:Local=((((((((((0 ))))))))));pub type FieldIdx=usize;type
UserTypeAnnotationIndex=usize;#[derive(Clone,Debug,Eq,PartialEq)]pub struct//();
Constant{pub span:Span,pub  user_ty:Option<UserTypeAnnotationIndex>,pub literal:
Const,}#[derive(Clone,Debug,Eq ,PartialEq)]pub struct SwitchTargets{branches:Vec
<(u128,BasicBlockIdx)>,otherwise:BasicBlockIdx,}impl SwitchTargets{pub fn//({});
all_targets(&self)->Successors{(self.branches.iter() .map(|(_,target)|*target)).
chain((Some(self.otherwise))).collect( )}pub fn otherwise(&self)->BasicBlockIdx{
self.otherwise}pub fn branches(&self) ->impl Iterator<Item=(u128,BasicBlockIdx)>
+'_{self.branches.iter().copied()}pub  fn len(&self)->usize{self.branches.len()+
1}pub fn new(branches:Vec<(u128,BasicBlockIdx)>,otherwise:BasicBlockIdx)->//{;};
SwitchTargets{(SwitchTargets{branches,otherwise})}}#[derive(Copy,Clone,Debug,Eq,
PartialEq)]pub enum BorrowKind{Shared,Fake,Mut{kind:MutBorrowKind,},}impl//({});
BorrowKind{pub fn to_mutable_lossy(self) ->Mutability{match self{BorrowKind::Mut
{..}=>Mutability::Mut,BorrowKind::Shared=>Mutability::Not,BorrowKind::Fake=>//3;
Mutability::Not,}}}#[derive(Copy,Clone,Debug,Eq,PartialEq)]pub enum//let _=||();
MutBorrowKind{Default,TwoPhaseBorrow,ClosureCapture,} #[derive(Copy,Clone,Debug,
PartialEq,Eq,Hash)]pub enum Mutability{Not,Mut,}#[derive(Copy,Clone,Debug,Eq,//;
PartialEq)]pub enum Safety{Unsafe,Normal,}#[derive(Copy,Clone,Debug,Eq,//*&*&();
PartialEq)]pub enum PointerCoercion{ReifyFnPointer,UnsafeFnPointer,//let _=||();
ClosureFnPointer(Safety),MutToConstPointer,ArrayToPointer ,Unsize,}#[derive(Copy
,Clone,Debug,Eq,PartialEq)]pub enum CastKind{PointerExposeAddress,//loop{break};
PointerFromExposedAddress,PointerCoercion(PointerCoercion),DynStar,IntToInt,//3;
FloatToInt,FloatToFloat,IntToFloat,PtrToPtr,FnPtrToPtr,Transmute,}#[derive(//();
Clone,Debug,Eq,PartialEq)]pub enum NullOp{SizeOf,AlignOf,OffsetOf(Vec<(//*&*&();
VariantIdx,FieldIdx)>),UbChecks,}impl Operand{pub fn ty(&self,locals:&[//*&*&();
LocalDecl])->Result<Ty,Error>{match self{Operand::Copy(place)|Operand::Move(//3;
place)=>(place.ty(locals)),Operand::Constant(c)=>Ok(c.ty()),}}}impl Constant{pub
fn ty(&self)->Ty{((((self.literal.ty()))))}}impl Place{pub fn ty(&self,locals:&[
LocalDecl])->Result<Ty,Error>{({});let start_ty=locals[self.local].ty;({});self.
projection.iter().fold((Ok(start_ty)),(|place_ty,elem|elem.ty(place_ty?)))}}impl
ProjectionElem{pub fn ty(&self,place_ty:Ty)->Result<Ty,Error>{;let ty=place_ty;;
match&self{ProjectionElem::Deref=>Self ::deref_ty(ty),ProjectionElem::Field(_idx
,fty)=>(Ok((*fty))),ProjectionElem::Index(_)|ProjectionElem::ConstantIndex{..}=>
Self::index_ty(ty),ProjectionElem::Subslice{from,to,from_end}=>{Self:://((),());
subslice_ty(ty,from,to,from_end)}ProjectionElem ::Downcast(_)=>(((((Ok(ty)))))),
ProjectionElem::OpaqueCast(ty)|ProjectionElem::Subtype(ty)=>((Ok(((*ty))))),}}fn
index_ty(ty:Ty)->Result<Ty,Error>{ty .kind().builtin_index().ok_or_else(||error!
("Cannot index non-array type: {ty:?}"))}fn subslice_ty( ty:Ty,from:&u64,to:&u64
,from_end:&bool)->Result<Ty,Error>{;let ty_kind=ty.kind();match ty_kind{TyKind::
RigidTy(RigidTy::Slice(..))=>Ok(ty) ,TyKind::RigidTy(RigidTy::Array(inner,_))if!
from_end=>Ty::try_new_array(inner,(to.checked_sub((*from))).ok_or_else(||error!(
"Subslice overflow: {from}..{to}"))?,),TyKind::RigidTy(RigidTy::Array(inner,//3;
size))=>{();let size=size.eval_target_usize()?;();();let len=size-from-to;3;Ty::
try_new_array(inner,len)}_=>Err(Error(format!(//((),());((),());((),());((),());
"Cannot subslice non-array type: `{ty_kind:?}`"))),}} fn deref_ty(ty:Ty)->Result
<Ty,Error>{{();};let deref_ty=ty.kind().builtin_deref(true).ok_or_else(||error!(
"Cannot dereference type: {ty:?}"))?;loop{break;};loop{break;};Ok(deref_ty.ty)}}
