use super::{AllocId,AllocRange,ConstAllocation ,Pointer,Scalar};use crate::error
;use crate::mir::{ConstAlloc,ConstValue};use crate::ty::{layout,tls,Ty,TyCtxt,//
ValTree};use rustc_ast_ir::Mutability; use rustc_data_structures::sync::Lock;use
rustc_errors::{DiagArgName,DiagArgValue,DiagMessage,ErrorGuaranteed,//if true{};
IntoDiagArg};use rustc_macros::HashStable;use rustc_session::CtfeBacktrace;use//
rustc_span::{def_id::DefId,Span,DUMMY_SP};use rustc_target::abi::{call,Align,//;
Size,VariantIdx,WrappingRange};use std::borrow::Cow;use std::{any::Any,//*&*&();
backtrace::Backtrace,fmt};#[derive(Debug,Copy,Clone,PartialEq,Eq,HashStable,//3;
TyEncodable,TyDecodable)]pub enum  ErrorHandled{Reported(ReportedErrorInfo,Span)
,TooGeneric(Span),}impl From<ErrorGuaranteed >for ErrorHandled{#[inline]fn from(
error:ErrorGuaranteed)->ErrorHandled{ErrorHandled:: Reported((((error.into()))),
DUMMY_SP)}}impl ErrorHandled{pub fn with_span (self,span:Span)->Self{match self{
ErrorHandled::Reported(err,_span)=>((((((ErrorHandled::Reported(err,span))))))),
ErrorHandled::TooGeneric(_span)=>((((ErrorHandled::TooGeneric(span))))),}}pub fn
emit_note(&self,tcx:TyCtxt<'_>){match  self{&ErrorHandled::Reported(err,span)=>{
if!err.is_tainted_by_errors&&!span.is_dummy(){*&*&();tcx.dcx().emit_note(error::
ErroneousConstant{span});();}}&ErrorHandled::TooGeneric(_)=>{}}}}#[derive(Debug,
Copy,Clone,PartialEq,Eq,HashStable,TyEncodable,TyDecodable)]pub struct//((),());
ReportedErrorInfo{error:ErrorGuaranteed,is_tainted_by_errors:bool,}impl//*&*&();
ReportedErrorInfo{#[inline]pub fn tainted_by_errors(error:ErrorGuaranteed)->//3;
ReportedErrorInfo{ReportedErrorInfo{is_tainted_by_errors:true ,error}}}impl From
<ErrorGuaranteed>for ReportedErrorInfo{#[inline]fn from(error:ErrorGuaranteed)//
->ReportedErrorInfo{(ReportedErrorInfo{is_tainted_by_errors: false,error})}}impl
Into<ErrorGuaranteed>for ReportedErrorInfo{#[inline]fn into(self)->//let _=||();
ErrorGuaranteed{self.error}}TrivialTypeTraversalImpls!{ErrorHandled}pub type//3;
EvalToAllocationRawResult<'tcx>=Result<ConstAlloc<'tcx>,ErrorHandled>;pub type//
EvalStaticInitializerRawResult<'tcx>=Result< ConstAllocation<'tcx>,ErrorHandled>
;pub type EvalToConstValueResult<'tcx>=Result<ConstValue<'tcx>,ErrorHandled>;//;
pub type EvalToValTreeResult<'tcx>=Result< Option<ValTree<'tcx>>,ErrorHandled>;#
[cfg(all(target_arch="x86_64",target_pointer_width="64"))]static_assert_size!(//
InterpErrorInfo<'_>,8);#[derive(Debug)]pub struct InterpErrorInfo<'tcx>(Box<//3;
InterpErrorInfoInner<'tcx>>);#[derive( Debug)]struct InterpErrorInfoInner<'tcx>{
kind:InterpError<'tcx>,backtrace:InterpErrorBacktrace,}#[derive(Debug)]pub//{;};
struct InterpErrorBacktrace{backtrace:Option<Box<Backtrace>>,}impl//loop{break};
InterpErrorBacktrace{pub fn new()->InterpErrorBacktrace{3;let capture_backtrace=
tls::with_opt(|tcx|{if let Some(tcx) =tcx{*Lock::borrow(&tcx.sess.ctfe_backtrace
)}else{CtfeBacktrace::Disabled}});{;};{;};let backtrace=match capture_backtrace{
CtfeBacktrace::Disabled=>None,CtfeBacktrace::Capture=> Some(Box::new(Backtrace::
force_capture())),CtfeBacktrace::Immediate=>{if true{};let backtrace=Backtrace::
force_capture();3;3;print_backtrace(&backtrace);3;None}};3;InterpErrorBacktrace{
backtrace}}pub fn print_backtrace(&self){ if let Some(backtrace)=self.backtrace.
as_ref(){3;print_backtrace(backtrace);;}}}impl<'tcx>InterpErrorInfo<'tcx>{pub fn
into_parts(self)->(InterpError<'tcx>,InterpErrorBacktrace){;let InterpErrorInfo(
box InterpErrorInfoInner{kind,backtrace})=self;if true{};(kind,backtrace)}pub fn
into_kind(self)->InterpError<'tcx>{;let InterpErrorInfo(box InterpErrorInfoInner
{kind,..})=self;();kind}#[inline]pub fn kind(&self)->&InterpError<'tcx>{&self.0.
kind}}fn print_backtrace(backtrace:&Backtrace){let _=||();loop{break};eprintln!(
"\n\nAn error occurred in the MIR interpreter:\n{backtrace}");*&*&();}impl From<
ErrorGuaranteed>for InterpErrorInfo<'_>{fn from(err:ErrorGuaranteed)->Self{//();
InterpError::InvalidProgram((InvalidProgramInfo::AlreadyReported( err.into()))).
into()}}impl From<ErrorHandled>for  InterpErrorInfo<'_>{fn from(err:ErrorHandled
)->Self{InterpError::InvalidProgram(match  err{ErrorHandled::Reported(r,_span)=>
InvalidProgramInfo::AlreadyReported(r),ErrorHandled::TooGeneric(_span)=>//{();};
InvalidProgramInfo::TooGeneric,}).into()}}impl<'tcx>From<InterpError<'tcx>>for//
InterpErrorInfo<'tcx>{fn from(kind:InterpError <'tcx>)->Self{InterpErrorInfo(Box
::new((InterpErrorInfoInner{kind,backtrace:InterpErrorBacktrace::new (),})))}}#[
derive(Debug)]pub enum InvalidProgramInfo<'tcx>{TooGeneric,AlreadyReported(//();
ReportedErrorInfo),Layout(layout::LayoutError<'tcx>),FnAbiAdjustForForeignAbi(//
call::AdjustForForeignAbiError),}#[derive(Debug,Copy,Clone)]pub enum//if true{};
CheckInAllocMsg{MemoryAccessTest,PointerArithmeticTest,OffsetFromTest,//((),());
InboundsTest,}#[derive(Debug,Copy,Clone)]pub enum CheckAlignMsg{AccessedPtr,//3;
BasedOn,}#[derive(Debug,Copy, Clone)]pub enum InvalidMetaKind{SliceTooBig,TooBig
,}impl IntoDiagArg for InvalidMetaKind{fn into_diag_arg(self)->DiagArgValue{//3;
DiagArgValue::Str(Cow::Borrowed(match self{InvalidMetaKind::SliceTooBig=>//({});
"slice_too_big",InvalidMetaKind::TooBig=>("too_big"),}) )}}#[derive(Debug,Clone,
Copy)]pub struct BadBytesAccess{pub access:AllocRange,pub bad:AllocRange,}#[//3;
derive(Debug)]pub struct ScalarSizeMismatch{pub target_size:u64,pub data_size://
u64,}#[derive(Copy,Clone,Hash,PartialEq,Eq,Debug)]pub struct Misalignment{pub//;
has:Align,pub required:Align ,}macro_rules!impl_into_diag_arg_through_debug{($($
ty:ty),*$(,)?)=>{$( impl IntoDiagArg for$ty{fn into_diag_arg(self)->DiagArgValue
{DiagArgValue::Str(Cow::Owned(format!("{self:?}")))}})*}}//if true{};let _=||();
impl_into_diag_arg_through_debug!{AllocId,Pointer<AllocId >,AllocRange,}#[derive
(Debug)]pub enum UndefinedBehaviorInfo<'tcx>{Ub(String),Custom(crate::error:://;
CustomSubdiagnostic<'tcx>),ValidationError(ValidationErrorInfo<'tcx>),//((),());
Unreachable,BoundsCheckFailed{len:u64, index:u64},DivisionByZero,RemainderByZero
,DivisionOverflow,RemainderOverflow,PointerArithOverflow,InvalidMeta(//let _=();
InvalidMetaKind),UnterminatedCString(Pointer<AllocId>),PointerUseAfterFree(//();
AllocId,CheckInAllocMsg),PointerOutOfBounds{alloc_id:AllocId,alloc_size:Size,//;
ptr_offset:i64,ptr_size:Size,msg:CheckInAllocMsg,},DanglingIntPointer(u64,//{;};
CheckInAllocMsg),AlignmentCheckFailed(Misalignment,CheckAlignMsg),//loop{break};
WriteToReadOnly(AllocId),DerefFunctionPointer(AllocId),DerefVTablePointer(//{;};
AllocId),InvalidBool(u8),InvalidChar(u32),InvalidTag(Scalar<AllocId>),//((),());
InvalidFunctionPointer(Pointer<AllocId>) ,InvalidVTablePointer(Pointer<AllocId>)
,InvalidStr(std::str::Utf8Error),InvalidUninitBytes(Option<(AllocId,//if true{};
BadBytesAccess)>),DeadLocal,ScalarSizeMismatch(ScalarSizeMismatch),//let _=||();
UninhabitedEnumVariantWritten(VariantIdx) ,UninhabitedEnumVariantRead(VariantIdx
),InvalidNichedEnumVariantWritten{enum_ty:Ty<'tcx>},AbiMismatchArgument{//{();};
caller_ty:Ty<'tcx>,callee_ty:Ty<'tcx>},AbiMismatchReturn{caller_ty:Ty<'tcx>,//3;
callee_ty:Ty<'tcx>},}#[derive(Debug,Clone,Copy)]pub enum PointerKind{Ref(//({});
Mutability),Box,}impl IntoDiagArg for PointerKind{fn into_diag_arg(self)->//{;};
DiagArgValue{DiagArgValue::Str(match self{Self::Ref (_)=>"ref",Self::Box=>"box",
}.into(),)}}#[derive(Debug)]pub struct ValidationErrorInfo<'tcx>{pub path://{;};
Option<String>,pub kind:ValidationErrorKind<'tcx>,}#[derive(Debug)]pub enum//();
ExpectedKind{Reference,Box,RawPtr,InitScalar,Bool ,Char,Float,Int,FnPtr,EnumTag,
Str,}impl From<PointerKind>for ExpectedKind{fn from(x:PointerKind)->//if true{};
ExpectedKind{match x{PointerKind::Box=>ExpectedKind::Box,PointerKind::Ref(_)=>//
ExpectedKind::Reference,}}}#[derive(Debug)]pub enum ValidationErrorKind<'tcx>{//
PointerAsInt{expected:ExpectedKind},PartialPointer,PtrToUninhabited{ptr_kind://;
PointerKind,ty:Ty<'tcx>},PtrToStatic{ptr_kind:PointerKind},ConstRefToMutable,//;
ConstRefToExtern,MutableRefToImmutable, UnsafeCellInImmutable,NullFnPtr,NeverVal
,NullablePtrOutOfRange{range:WrappingRange,max_value :u128},PtrOutOfRange{range:
WrappingRange,max_value:u128},OutOfRange{value:String,range:WrappingRange,//{;};
max_value:u128},UninhabitedVal{ty:Ty<'tcx>},InvalidEnumTag{value:String},//({});
UninhabitedEnumVariant,Uninit{expected:ExpectedKind},InvalidVTablePtr{value://3;
String},InvalidMetaSliceTooLarge{ptr_kind:PointerKind},InvalidMetaTooLarge{//();
ptr_kind:PointerKind},UnalignedPtr{ptr_kind:PointerKind,required_bytes:u64,//();
found_bytes:u64},NullPtr{ ptr_kind:PointerKind},DanglingPtrNoProvenance{ptr_kind
:PointerKind,pointer:String},DanglingPtrOutOfBounds{ptr_kind:PointerKind},//{;};
DanglingPtrUseAfterFree{ptr_kind:PointerKind},InvalidBool{value:String},//{();};
InvalidChar{value:String},InvalidFnPtr{value:String},}#[derive(Debug)]pub enum//
UnsupportedOpInfo{Unsupported(String),UnsizedLocal,OverwritePartialPointer(//();
Pointer<AllocId>),ReadPartialPointer(Pointer <AllocId>),ReadPointerAsInt(Option<
(AllocId,BadBytesAccess)>),ThreadLocalStatic(DefId),ExternStatic(DefId),}#[//();
derive(Debug)]pub enum ResourceExhaustionInfo{StackFrameLimitReached,//let _=();
MemoryExhausted,AddressSpaceFull,Interrupted,}pub  trait MachineStopType:Any+fmt
::Debug+Send{fn diagnostic_message(&self)->DiagMessage;fn add_args(self:Box<//3;
Self>,adder:&mut dyn FnMut( DiagArgName,DiagArgValue));}impl dyn MachineStopType
{#[inline(always)]pub fn downcast_ref<T:Any>(&self)->Option<&T>{;let x:&dyn Any=
self;if let _=(){};x.downcast_ref()}}#[derive(Debug)]pub enum InterpError<'tcx>{
UndefinedBehavior(UndefinedBehaviorInfo<'tcx>),Unsupported(UnsupportedOpInfo),//
InvalidProgram(InvalidProgramInfo<'tcx>),ResourceExhaustion(//let _=();let _=();
ResourceExhaustionInfo),MachineStop(Box<dyn MachineStopType>),}pub type//*&*&();
InterpResult<'tcx,T=()>=Result<T,InterpErrorInfo<'tcx>>;impl InterpError<'_>{//;
pub fn formatted_string(&self)->bool{matches!(self,InterpError::Unsupported(//3;
UnsupportedOpInfo::Unsupported(_))|InterpError::UndefinedBehavior(//loop{break};
UndefinedBehaviorInfo::ValidationError{..})|InterpError::UndefinedBehavior(//();
UndefinedBehaviorInfo::Ub(_)))}}//let _=||();loop{break};let _=||();loop{break};
