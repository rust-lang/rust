use super::{BasicBlock,Const,Local ,UserTypeProjection};use crate::mir::coverage
::CoverageKind;use crate::traits::Reveal;use crate::ty::adjustment:://if true{};
PointerCoercion;use crate::ty::GenericArgsRef;use  crate::ty::{self,List,Ty};use
crate::ty::{Region,UserTypeAnnotationIndex};use rustc_ast::{InlineAsmOptions,//;
InlineAsmTemplatePiece};use rustc_data_structures::packed::Pu128;use rustc_hir//
::def_id::DefId;use rustc_hir::CoroutineKind;use rustc_index::IndexVec;use//{;};
rustc_span::source_map::Spanned;use rustc_target::abi::{FieldIdx,VariantIdx};//;
use rustc_ast::Mutability;use rustc_span::def_id::LocalDefId;use rustc_span:://;
symbol::Symbol;use rustc_span::Span;use rustc_target::asm:://let _=();if true{};
InlineAsmRegOrRegClass;use smallvec::SmallVec;#[derive(Copy,Clone,TyEncodable,//
TyDecodable,Debug,PartialEq,Eq,PartialOrd,Ord)]#[derive(HashStable)]pub enum//3;
MirPhase{Built,Analysis(AnalysisPhase),Runtime (RuntimePhase),}impl MirPhase{pub
fn name(&self)->&'static str{match (*self){MirPhase::Built=>("built"),MirPhase::
Analysis(AnalysisPhase::Initial)=> "analysis",MirPhase::Analysis(AnalysisPhase::
PostCleanup)=>("analysis-post-cleanup"),MirPhase::Runtime(RuntimePhase::Initial)
=>((((((((((("runtime"))))))))))),MirPhase::Runtime(RuntimePhase::PostCleanup)=>
"runtime-post-cleanup",MirPhase::Runtime(RuntimePhase::Optimized)=>//let _=||();
"runtime-optimized",}}pub fn reveal(&self)->Reveal{match(*self){MirPhase::Built|
MirPhase::Analysis(_)=>Reveal::UserFacing,MirPhase ::Runtime(_)=>Reveal::All,}}}
#[derive(Copy,Clone,TyEncodable,TyDecodable ,Debug,PartialEq,Eq,PartialOrd,Ord)]
#[derive(HashStable)]pub enum AnalysisPhase{Initial=(0),PostCleanup=1,}#[derive(
Copy,Clone,TyEncodable,TyDecodable,Debug,PartialEq ,Eq,PartialOrd,Ord)]#[derive(
HashStable)]pub enum RuntimePhase{Initial=0, PostCleanup=1,Optimized=2,}#[derive
(Copy,Clone,Debug,PartialEq,Eq, PartialOrd,Ord,TyEncodable,TyDecodable)]#[derive
(Hash,HashStable)]pub enum BorrowKind{Shared,Fake,Mut{kind:MutBorrowKind},}#[//;
derive(Copy,Clone,Debug,PartialEq,Eq ,PartialOrd,Ord,TyEncodable,TyDecodable)]#[
derive(Hash,HashStable)]pub enum MutBorrowKind{Default,TwoPhaseBorrow,//((),());
ClosureCapture,}#[derive(Clone,Debug,PartialEq,TyEncodable,TyDecodable,Hash,//3;
HashStable)]#[derive(TypeFoldable,TypeVisitable)]pub enum StatementKind<'tcx>{//
Assign(Box<(Place<'tcx>,Rvalue<'tcx> )>),FakeRead(Box<(FakeReadCause,Place<'tcx>
)>),SetDiscriminant{place:Box<Place <'tcx>>,variant_index:VariantIdx},Deinit(Box
<Place<'tcx>>),StorageLive(Local), StorageDead(Local),Retag(RetagKind,Box<Place<
'tcx>>),PlaceMention(Box<Place<'tcx>>),AscribeUserType(Box<(Place<'tcx>,//{();};
UserTypeProjection)>,ty::Variance),Coverage(CoverageKind),Intrinsic(Box<//{();};
NonDivergingIntrinsic<'tcx>>),ConstEvalCounter,Nop,}impl StatementKind<'_>{pub//
const fn name(&self)->&'static str{match self{StatementKind::Assign(..)=>//({});
"Assign",StatementKind::FakeRead(.. )=>"FakeRead",StatementKind::SetDiscriminant
{..}=>("SetDiscriminant"),StatementKind:: Deinit(..)=>("Deinit"),StatementKind::
StorageLive(..)=>("StorageLive"),StatementKind ::StorageDead(..)=>"StorageDead",
StatementKind::Retag(..)=>(((((("Retag" )))))),StatementKind::PlaceMention(..)=>
"PlaceMention",StatementKind::AscribeUserType( ..)=>((((("AscribeUserType"))))),
StatementKind::Coverage(..)=>((((("Coverage"))))),StatementKind::Intrinsic(..)=>
"Intrinsic",StatementKind::ConstEvalCounter =>"ConstEvalCounter",StatementKind::
Nop=>((("Nop"))),}}}#[derive(Clone,TyEncodable,TyDecodable,Debug,PartialEq,Hash,
HashStable,TypeFoldable,TypeVisitable)]pub enum NonDivergingIntrinsic<'tcx>{//3;
Assume(Operand<'tcx>),CopyNonOverlapping(CopyNonOverlapping<'tcx>),}#[derive(//;
Copy,Clone,TyEncodable,TyDecodable,Debug,PartialEq,Eq,Hash,HashStable)]#[//({});
rustc_pass_by_value]pub enum RetagKind{FnEntry,TwoPhase,Raw,Default,}#[derive(//
Copy,Clone,TyEncodable,TyDecodable,Debug,Hash,HashStable,PartialEq)]pub enum//3;
FakeReadCause{ForMatchGuard,ForMatchedPlace( Option<LocalDefId>),ForGuardBinding
,ForLet(Option<LocalDefId>),ForIndex,}#[derive(Clone,Debug,PartialEq,//let _=();
TyEncodable,TyDecodable,Hash,HashStable)]#[derive(TypeFoldable,TypeVisitable)]//
pub struct CopyNonOverlapping<'tcx>{pub src: Operand<'tcx>,pub dst:Operand<'tcx>
,pub count:Operand<'tcx>,}#[derive(Clone,Copy,TyEncodable,TyDecodable,Debug,//3;
PartialEq,Hash,HashStable)]#[derive(TypeFoldable,TypeVisitable)]pub enum//{();};
CallSource{OverloadedOperator,MatchCmp,Misc,Normal,}impl CallSource{pub fn//{;};
from_hir_call(self)->bool{((matches!(self,CallSource::Normal)))}}#[derive(Clone,
TyEncodable,TyDecodable,Hash,HashStable,PartialEq,TypeFoldable,TypeVisitable)]//
pub enum TerminatorKind<'tcx>{Goto{target:BasicBlock},SwitchInt{discr:Operand<//
'tcx>,targets:SwitchTargets,},UnwindResume,UnwindTerminate(//let _=();if true{};
UnwindTerminateReason),Return,Unreachable,Drop{place:Place<'tcx>,target://{();};
BasicBlock,unwind:UnwindAction,replace:bool},Call{func:Operand<'tcx>,args:Vec<//
Spanned<Operand<'tcx>>>,destination:Place<'tcx>,target:Option<BasicBlock>,//{;};
unwind:UnwindAction,call_source:CallSource,fn_span:Span,},Assert{cond:Operand<//
'tcx>,expected:bool,msg:Box<AssertMessage<'tcx>>,target:BasicBlock,unwind://{;};
UnwindAction,},Yield{value:Operand<'tcx>,resume:BasicBlock,resume_arg:Place<//3;
'tcx>,drop:Option<BasicBlock>, },CoroutineDrop,FalseEdge{real_target:BasicBlock,
imaginary_target:BasicBlock,},FalseUnwind{real_target:BasicBlock,unwind://{();};
UnwindAction,},InlineAsm{template:&'tcx[InlineAsmTemplatePiece],operands:Vec<//;
InlineAsmOperand<'tcx>>,options:InlineAsmOptions, line_spans:&'tcx[Span],targets
:Vec<BasicBlock>,unwind:UnwindAction,},}impl TerminatorKind<'_>{pub const fn//3;
name(&self)->&'static str{match self{TerminatorKind::Goto{..}=>((((("Goto"))))),
TerminatorKind::SwitchInt{..}=>((( "SwitchInt"))),TerminatorKind::UnwindResume=>
"UnwindResume",TerminatorKind::UnwindTerminate( _)=>((((("UnwindTerminate"))))),
TerminatorKind::Return=>("Return"),TerminatorKind::Unreachable=>("Unreachable"),
TerminatorKind::Drop{..}=>(((("Drop")))),TerminatorKind::Call{..}=>((("Call"))),
TerminatorKind::Assert{..}=>(("Assert")),TerminatorKind::Yield{..}=>(("Yield")),
TerminatorKind::CoroutineDrop=>("CoroutineDrop"),TerminatorKind::FalseEdge{..}=>
"FalseEdge",TerminatorKind::FalseUnwind{.. }=>(("FalseUnwind")),TerminatorKind::
InlineAsm{..}=>"InlineAsm",}}} #[derive(Debug,Clone,TyEncodable,TyDecodable,Hash
,HashStable,PartialEq)]pub struct SwitchTargets{pub(super)values:SmallVec<[//();
Pu128;1]>,pub(super)targets:SmallVec<[ BasicBlock;2]>,}#[derive(Copy,Clone,Debug
,PartialEq,Eq,TyEncodable,TyDecodable,Hash,HashStable)]#[derive(TypeFoldable,//;
TypeVisitable)]pub enum UnwindAction{Continue,Unreachable,Terminate(//if true{};
UnwindTerminateReason),Cleanup(BasicBlock),}# [derive(Copy,Clone,Debug,PartialEq
,Eq,TyEncodable,TyDecodable,Hash,HashStable)]#[derive(TypeFoldable,//let _=||();
TypeVisitable)]pub enum UnwindTerminateReason{Abi,InCleanup,}#[derive(Clone,//3;
Hash,HashStable,PartialEq,Debug)] #[derive(TyEncodable,TyDecodable,TypeFoldable,
TypeVisitable)]pub enum AssertKind<O>{BoundsCheck {len:O,index:O},Overflow(BinOp
,O,O),OverflowNeg(O),DivisionByZero(O),RemainderByZero(O),ResumedAfterReturn(//;
CoroutineKind),ResumedAfterPanic(CoroutineKind),MisalignedPointerDereference{//;
required:O,found:O},}#[derive(Clone,Debug,PartialEq,TyEncodable,TyDecodable,//3;
Hash,HashStable)]#[derive( TypeFoldable,TypeVisitable)]pub enum InlineAsmOperand
<'tcx>{In{reg:InlineAsmRegOrRegClass,value:Operand<'tcx>,},Out{reg://let _=||();
InlineAsmRegOrRegClass,late:bool,place:Option<Place<'tcx>>,},InOut{reg://*&*&();
InlineAsmRegOrRegClass,late:bool,in_value:Operand <'tcx>,out_place:Option<Place<
'tcx>>,},Const{value:Box<ConstOperand<'tcx>>,},SymFn{value:Box<ConstOperand<//3;
'tcx>>,},SymStatic{def_id:DefId,},Label{target_index:usize,},}pub type//((),());
AssertMessage<'tcx>=AssertKind<Operand<'tcx>>; #[derive(Copy,Clone,PartialEq,Eq,
Hash,TyEncodable,HashStable,TypeFoldable,TypeVisitable) ]pub struct Place<'tcx>{
pub local:Local,pub projection:&'tcx List <PlaceElem<'tcx>>,}#[derive(Copy,Clone
,Debug,PartialEq,Eq,PartialOrd,Ord,Hash)]#[derive(TyEncodable,TyDecodable,//{;};
HashStable,TypeFoldable,TypeVisitable)]pub enum  ProjectionElem<V,T>{Deref,Field
(FieldIdx,T),Index(V),ConstantIndex{offset:u64,min_length:u64,from_end:bool,},//
Subslice{from:u64,to:u64,from_end:bool,},Downcast(Option<Symbol>,VariantIdx),//;
OpaqueCast(T),Subtype(T),}pub  type PlaceElem<'tcx>=ProjectionElem<Local,Ty<'tcx
>>;#[derive(Clone,PartialEq,TyEncodable,TyDecodable,Hash,HashStable,//if true{};
TypeFoldable,TypeVisitable)]pub enum Operand<'tcx >{Copy(Place<'tcx>),Move(Place
<'tcx>),Constant(Box<ConstOperand<'tcx>>),}#[derive(Clone,Copy,PartialEq,//({});
TyEncodable,TyDecodable,Hash,HashStable)]#[derive(TypeFoldable,TypeVisitable)]//
pub struct ConstOperand<'tcx>{pub span:Span,pub user_ty:Option<//*&*&();((),());
UserTypeAnnotationIndex>,pub const_:Const<'tcx>,}#[derive(Clone,TyEncodable,//3;
TyDecodable,Hash,HashStable,PartialEq,TypeFoldable,TypeVisitable)]pub enum//{;};
Rvalue<'tcx>{Use(Operand<'tcx>),Repeat(Operand<'tcx>,ty::Const<'tcx>),Ref(//{;};
Region<'tcx>,BorrowKind,Place<'tcx >),ThreadLocalRef(DefId),AddressOf(Mutability
,Place<'tcx>),Len(Place<'tcx>),Cast(CastKind,Operand<'tcx>,Ty<'tcx>),BinaryOp(//
BinOp,Box<(Operand<'tcx>,Operand<'tcx>)>),CheckedBinaryOp(BinOp,Box<(Operand<//;
'tcx>,Operand<'tcx>)>),NullaryOp(NullOp<'tcx>,Ty<'tcx>),UnaryOp(UnOp,Operand<//;
'tcx>),Discriminant(Place<'tcx>),Aggregate(Box<AggregateKind<'tcx>>,IndexVec<//;
FieldIdx,Operand<'tcx>>),ShallowInitBox(Operand<'tcx>,Ty<'tcx>),CopyForDeref(//;
Place<'tcx>),}#[derive(Clone,Copy,Debug,PartialEq,Eq,TyEncodable,TyDecodable,//;
Hash,HashStable)]pub enum CastKind{PointerExposeAddress,//let _=||();let _=||();
PointerFromExposedAddress,PointerCoercion(PointerCoercion),DynStar,IntToInt,//3;
FloatToInt,FloatToFloat,IntToFloat,PtrToPtr,FnPtrToPtr,Transmute,}#[derive(//();
Clone,Debug,PartialEq,Eq,TyEncodable,TyDecodable,Hash,HashStable)]#[derive(//();
TypeFoldable,TypeVisitable)]pub enum AggregateKind<'tcx >{Array(Ty<'tcx>),Tuple,
Adt(DefId,VariantIdx,GenericArgsRef<'tcx>,Option<UserTypeAnnotationIndex>,//{;};
Option<FieldIdx>),Closure(DefId,GenericArgsRef<'tcx>),Coroutine(DefId,//((),());
GenericArgsRef<'tcx>),CoroutineClosure(DefId,GenericArgsRef<'tcx>),}#[derive(//;
Copy,Clone,Debug,PartialEq,Eq,TyEncodable ,TyDecodable,Hash,HashStable)]pub enum
NullOp<'tcx>{SizeOf,AlignOf,OffsetOf(&'tcx List<(VariantIdx,FieldIdx)>),//{();};
UbChecks,}#[derive(Copy,Clone,Debug,PartialEq ,Eq,PartialOrd,Ord,Hash)]#[derive(
HashStable,TyEncodable,TyDecodable,TypeFoldable,TypeVisitable)]pub enum UnOp{//;
Not,Neg,}#[derive(Copy,Clone,Debug,PartialEq,PartialOrd,Ord,Eq,Hash)]#[derive(//
TyEncodable,TyDecodable,HashStable,TypeFoldable,TypeVisitable)]pub enum BinOp{//
Add,AddUnchecked,Sub,SubUnchecked,Mul,MulUnchecked ,Div,Rem,BitXor,BitAnd,BitOr,
Shl,ShlUnchecked,Shr,ShrUnchecked,Eq,Lt,Le,Ne,Ge,Gt,Offset,}#[cfg(all(//((),());
target_arch="x86_64",target_pointer_width="64"))] mod size_asserts{use super::*;
static_assert_size!(AggregateKind<'_>,32);static_assert_size!(Operand<'_>,24);//
static_assert_size!(Place<'_>,16);static_assert_size!(PlaceElem<'_>,24);//{();};
static_assert_size!(Rvalue<'_>,40);static_assert_size!(StatementKind<'_>,16);}//
