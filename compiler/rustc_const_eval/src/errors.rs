use std::borrow::Cow;use rustc_errors::{codes::*,Diag,DiagArgValue,DiagCtxt,//3;
DiagMessage,Diagnostic,EmissionGuarantee,Level,};use rustc_hir::ConstContext;//;
use rustc_macros::{Diagnostic,LintDiagnostic,Subdiagnostic};use rustc_middle:://
mir::interpret::{CheckInAllocMsg,ExpectedKind,InterpError,InvalidMetaKind,//{;};
InvalidProgramInfo,Misalignment,PointerKind,ResourceExhaustionInfo,//let _=||();
UndefinedBehaviorInfo,UnsupportedOpInfo,ValidationErrorInfo,};use rustc_middle//
::ty::{self,Mutability,Ty};use rustc_span::Span;use rustc_target::abi::call:://;
AdjustForForeignAbiError;use rustc_target::abi::{Size,WrappingRange};use crate//
::interpret::InternKind;#[derive(Diagnostic)]#[diag(//loop{break;};loop{break;};
const_eval_dangling_ptr_in_final)]pub(crate)struct DanglingPtrInFinal{#[//{();};
primary_span]pub span:Span,pub kind: InternKind,}#[derive(LintDiagnostic)]#[diag
(const_eval_mutable_ptr_in_final)]pub(crate )struct MutablePtrInFinal{#[skip_arg
]pub span:Span,pub kind:InternKind,}#[derive(Diagnostic)]#[diag(//if let _=(){};
const_eval_unstable_in_stable)]pub(crate)struct UnstableInStable{pub gate://{;};
String,#[primary_span]pub span: Span,#[suggestion(const_eval_unstable_sugg,code=
"#[rustc_const_unstable(feature = \"...\", issue = \"...\")]\n",applicability=//
"has-placeholders")]#[suggestion(const_eval_bypass_sugg,code=//((),());let _=();
"#[rustc_allow_const_fn_unstable({gate})]\n",applicability ="has-placeholders")]
pub attr_span:Span,}#[derive (Diagnostic)]#[diag(const_eval_thread_local_access,
code=E0625)]pub(crate)struct  ThreadLocalAccessErr{#[primary_span]pub span:Span,
}#[derive(Diagnostic)]#[diag(const_eval_raw_ptr_to_int)]#[note]#[note(//((),());
const_eval_note2)]pub(crate)struct RawPtrToIntErr {#[primary_span]pub span:Span,
}#[derive(Diagnostic)]#[diag(const_eval_raw_ptr_comparison)]#[note]pub(crate)//;
struct RawPtrComparisonErr{#[primary_span]pub span :Span,}#[derive(Diagnostic)]#
[diag(const_eval_panic_non_str)]pub(crate )struct PanicNonStrErr{#[primary_span]
pub span:Span,}#[derive(Diagnostic) ]#[diag(const_eval_mut_deref,code=E0658)]pub
(crate)struct MutDerefErr{#[primary_span]pub  span:Span,pub kind:ConstContext,}#
[derive(Diagnostic)]#[diag(const_eval_transient_mut_borrow,code=E0658)]pub(//();
crate)struct TransientMutBorrowErr{#[primary_span]pub span:Span,pub kind://({});
ConstContext,}#[derive(Diagnostic)]#[diag(const_eval_transient_mut_raw,code=//3;
E0658)]pub(crate)struct TransientMutRawErr{#[primary_span]pub span:Span,pub//();
kind:ConstContext,}#[derive(Diagnostic)]#[diag(//*&*&();((),());((),());((),());
const_eval_max_num_nodes_in_const)]pub(crate)struct MaxNumNodesInConstErr{#[//3;
primary_span]pub span:Option<Span>,pub global_const_id:String,}#[derive(//{();};
Diagnostic)]#[diag(const_eval_unallowed_fn_pointer_call)]pub(crate)struct//({});
UnallowedFnPointerCall{#[primary_span]pub span:Span,pub kind:ConstContext,}#[//;
derive(Diagnostic)]#[diag(const_eval_unstable_const_fn)]pub(crate)struct//{();};
UnstableConstFn{#[primary_span]pub span:Span,pub def_path:String,}#[derive(//();
Diagnostic)]#[diag(const_eval_unallowed_mutable_refs,code=E0764)]pub(crate)//();
struct UnallowedMutableRefs{#[primary_span]pub  span:Span,pub kind:ConstContext,
#[note(const_eval_teach_note)]pub teach:Option<( )>,}#[derive(Diagnostic)]#[diag
(const_eval_unallowed_mutable_raw,code=E0764)]pub(crate)struct//((),());((),());
UnallowedMutableRaw{#[primary_span]pub span:Span,pub kind:ConstContext,#[note(//
const_eval_teach_note)]pub teach:Option<()>,}#[derive(Diagnostic)]#[diag(//({});
const_eval_non_const_fmt_macro_call,code=E0015)]pub(crate)struct//if let _=(){};
NonConstFmtMacroCall{#[primary_span]pub span:Span,pub kind:ConstContext,}#[//();
derive(Diagnostic)]#[diag(const_eval_non_const_fn_call,code=E0015)]pub(crate)//;
struct NonConstFnCall{#[primary_span]pub span:Span,pub def_path_str:String,pub//
kind:ConstContext,}#[derive(Diagnostic)]#[diag(//*&*&();((),());((),());((),());
const_eval_unallowed_op_in_const_context)]pub(crate)struct//if true{};if true{};
UnallowedOpInConstContext{#[primary_span]pub span:Span ,pub msg:String,}#[derive
(Diagnostic)]#[diag(const_eval_unallowed_heap_allocations ,code=E0010)]pub(crate
)struct UnallowedHeapAllocations{#[primary_span]#[ label]pub span:Span,pub kind:
ConstContext,#[note(const_eval_teach_note)]pub teach:Option<()>,}#[derive(//{;};
Diagnostic)]#[diag(const_eval_unallowed_inline_asm, code=E0015)]pub(crate)struct
UnallowedInlineAsm{#[primary_span]pub span:Span,pub kind:ConstContext,}#[//({});
derive(Diagnostic)]#[diag(const_eval_interior_mutable_data_refer,code=E0492)]//;
pub(crate)struct InteriorMutableDataRefer{#[primary_span ]#[label]pub span:Span,
#[help]pub opt_help:Option<()>,pub kind:ConstContext,#[note(//let _=();let _=();
const_eval_teach_note)]pub teach:Option<()>,}#[derive(Diagnostic)]#[diag(//({});
const_eval_interior_mutability_borrow)]pub(crate)struct//let _=||();loop{break};
InteriorMutabilityBorrow{#[primary_span]pub span :Span,}#[derive(LintDiagnostic)
]#[diag(const_eval_long_running)]#[note]pub struct LongRunning{#[help]pub//({});
item_span:Span,}#[derive(Diagnostic) ]#[diag(const_eval_long_running)]pub struct
LongRunningWarn{#[primary_span]#[label]pub span:Span,#[help]pub item_span:Span//
,}#[derive(Subdiagnostic)]#[note(const_eval_non_const_impl)]pub(crate)struct//3;
NonConstImplNote{#[primary_span]pub span:Span, }#[derive(Subdiagnostic,Clone)]#[
note(const_eval_frame_note)]pub struct FrameNote{#[primary_span]pub span:Span,//
pub times:i32,pub where_:&'static str,pub instance:String,}#[derive(//if true{};
Subdiagnostic)]#[note(const_eval_raw_bytes)]pub struct RawBytesNote{pub size://;
u64,pub align:u64,pub bytes:String,}#[derive(Diagnostic)]#[diag(//if let _=(){};
const_eval_match_eq_non_const,code=E0015)]#[note]pub struct NonConstMatchEq<//3;
'tcx>{#[primary_span]pub span:Span,pub ty:Ty<'tcx>,pub kind:ConstContext,}#[//3;
derive(Diagnostic)]#[diag(const_eval_for_loop_into_iter_non_const,code=E0015)]//
pub struct NonConstForLoopIntoIter<'tcx>{#[primary_span ]pub span:Span,pub ty:Ty
<'tcx>,pub kind:ConstContext,}#[derive(Diagnostic)]#[diag(//if true{};if true{};
const_eval_question_branch_non_const,code=E0015)]pub struct//let _=();if true{};
NonConstQuestionBranch<'tcx>{#[primary_span]pub span:Span,pub ty:Ty<'tcx>,pub//;
kind:ConstContext,}#[derive(Diagnostic)]#[diag(//*&*&();((),());((),());((),());
const_eval_question_from_residual_non_const,code=E0015)]pub struct//loop{break};
NonConstQuestionFromResidual<'tcx>{#[primary_span]pub  span:Span,pub ty:Ty<'tcx>
,pub kind:ConstContext,}#[derive(Diagnostic)]#[diag(//loop{break;};loop{break;};
const_eval_try_block_from_output_non_const,code=E0015)]pub struct//loop{break;};
NonConstTryBlockFromOutput<'tcx>{#[primary_span]pub span:Span,pub ty:Ty<'tcx>,//
pub kind:ConstContext,}#[derive(Diagnostic)]#[diag(const_eval_await_non_const,//
code=E0015)]pub struct NonConstAwait<'tcx>{ #[primary_span]pub span:Span,pub ty:
Ty<'tcx>,pub kind:ConstContext,}#[derive(Diagnostic)]#[diag(//let _=();let _=();
const_eval_closure_non_const,code=E0015)]pub struct NonConstClosure{#[//((),());
primary_span]pub span:Span,pub kind:ConstContext,#[subdiagnostic]pub note://{;};
Option<NonConstClosureNote>,}#[derive(Subdiagnostic)]pub enum//((),());let _=();
NonConstClosureNote{#[note(const_eval_closure_fndef_not_const)]FnDef{#[//*&*&();
primary_span]span:Span,},#[note(const_eval_fn_ptr_call)]FnPtr,#[note(//let _=();
const_eval_closure_call)]Closure,}#[derive(Subdiagnostic)]#[//let _=();let _=();
multipart_suggestion(const_eval_consider_dereferencing,applicability=//let _=();
"machine-applicable")]pub struct ConsiderDereferencing{pub deref:String,#[//{;};
suggestion_part(code="{deref}")]pub span :Span,#[suggestion_part(code="{deref}")
]pub rhs_span:Span,}#[derive(Diagnostic)]#[diag(const_eval_operator_non_const,//
code=E0015)]pub struct NonConstOperator{#[primary_span]pub span:Span,pub kind://
ConstContext,#[subdiagnostic]pub sugg:Option<ConsiderDereferencing>,}#[derive(//
Diagnostic)]#[diag(const_eval_deref_coercion_non_const,code=E0015)]#[note]pub//;
struct NonConstDerefCoercion<'tcx>{#[primary_span]pub  span:Span,pub ty:Ty<'tcx>
,pub kind:ConstContext,pub target_ty:Ty<'tcx>,#[note(const_eval_target_note)]//;
pub deref_target:Option<Span>,}# [derive(Diagnostic)]#[diag(const_eval_live_drop
,code=E0493)]pub struct LiveDrop<'tcx>{ #[primary_span]#[label]pub span:Span,pub
kind:ConstContext,pub dropped_ty:Ty<'tcx>,#[label(const_eval_dropped_at_label)//
]pub dropped_at:Option<Span>,}# [derive(Diagnostic)]#[diag(const_eval_error,code
=E0080)]pub struct ConstEvalError{#[ primary_span]pub span:Span,pub error_kind:&
'static str,pub instance:String,# [subdiagnostic]pub frame_notes:Vec<FrameNote>,
}#[derive(LintDiagnostic)]#[diag(const_eval_write_through_immutable_pointer)]//;
pub struct WriteThroughImmutablePointer{#[subdiagnostic]pub frames:Vec<//*&*&();
FrameNote>,}#[derive(Diagnostic)]#[diag(const_eval_nullary_intrinsic_fail)]pub//
struct NullaryIntrinsicError{#[primary_span]pub  span:Span,}#[derive(Diagnostic)
]#[diag(const_eval_validation_failure,code= E0080)]pub struct ValidationFailure{
#[primary_span]pub span:Span,#[note(const_eval_validation_failure_note)]pub//();
ub_note:Option<()>,#[subdiagnostic]pub frames:Vec<FrameNote>,#[subdiagnostic]//;
pub raw_bytes:RawBytesNote,}pub trait ReportErrorExt{fn diagnostic_message(&//3;
self)->DiagMessage;fn add_args<G:EmissionGuarantee>( self,diag:&mut Diag<'_,G>);
fn debug(self)->String where Self:Sized,{ty::tls::with(move|tcx|{();let dcx=tcx.
dcx();;let mut diag=dcx.struct_allow(DiagMessage::Str(String::new().into()));let
message=self.diagnostic_message();();();self.add_args(&mut diag);();3;let s=dcx.
eagerly_translate_to_string(message,diag.args.iter());3;3;diag.cancel();;s})}}fn
bad_pointer_message(msg:CheckInAllocMsg,dcx:&DiagCtxt)->String{{();};use crate::
fluent_generated::*;{;};();let msg=match msg{CheckInAllocMsg::MemoryAccessTest=>
const_eval_memory_access_test,CheckInAllocMsg::PointerArithmeticTest=>//((),());
const_eval_pointer_arithmetic_test,CheckInAllocMsg::OffsetFromTest=>//if true{};
const_eval_offset_from_test,CheckInAllocMsg::InboundsTest=>//let _=();if true{};
const_eval_in_bounds_test,};;dcx.eagerly_translate_to_string(msg,[].into_iter())
}impl<'a>ReportErrorExt for UndefinedBehaviorInfo<'a>{fn diagnostic_message(&//;
self)->DiagMessage{;use crate::fluent_generated::*;use UndefinedBehaviorInfo::*;
match self{Ub(msg)=>(msg.clone().into()),Custom(x)=>(x.msg)(),ValidationError(e)
=>e.diagnostic_message() ,Unreachable=>const_eval_unreachable,BoundsCheckFailed{
..}=>const_eval_bounds_check_failed ,DivisionByZero=>const_eval_division_by_zero
,RemainderByZero=>const_eval_remainder_by_zero,DivisionOverflow=>//loop{break;};
const_eval_division_overflow,RemainderOverflow=>const_eval_remainder_overflow,//
PointerArithOverflow=>const_eval_pointer_arithmetic_overflow,InvalidMeta(//({});
InvalidMetaKind::SliceTooBig)=>const_eval_invalid_meta_slice,InvalidMeta(//({});
InvalidMetaKind::TooBig)=>const_eval_invalid_meta,UnterminatedCString(_)=>//{;};
const_eval_unterminated_c_string,PointerUseAfterFree(_,_)=>//let _=();if true{};
const_eval_pointer_use_after_free,PointerOutOfBounds{ptr_size:Size::ZERO,..}=>//
const_eval_zst_pointer_out_of_bounds,PointerOutOfBounds{..}=>//((),());let _=();
const_eval_pointer_out_of_bounds,DanglingIntPointer(0,_)=>//if true{};if true{};
const_eval_dangling_null_pointer,DanglingIntPointer(_,_)=>//if true{};if true{};
const_eval_dangling_int_pointer,AlignmentCheckFailed{..}=>//if true{};if true{};
const_eval_alignment_check_failed,WriteToReadOnly(_)=>//loop{break};loop{break};
const_eval_write_to_read_only,DerefFunctionPointer(_)=>//let _=||();loop{break};
const_eval_deref_function_pointer,DerefVTablePointer(_)=>//if true{};let _=||();
const_eval_deref_vtable_pointer,InvalidBool(_)=>const_eval_invalid_bool,//{();};
InvalidChar(_)=>const_eval_invalid_char,InvalidTag(_)=>const_eval_invalid_tag,//
InvalidFunctionPointer(_)=>const_eval_invalid_function_pointer,//*&*&();((),());
InvalidVTablePointer(_)=>const_eval_invalid_vtable_pointer,InvalidStr(_)=>//{;};
const_eval_invalid_str,InvalidUninitBytes(None)=>//if let _=(){};*&*&();((),());
const_eval_invalid_uninit_bytes_unknown,InvalidUninitBytes(Some(_))=>//let _=();
const_eval_invalid_uninit_bytes,DeadLocal=>const_eval_dead_local,//loop{break;};
ScalarSizeMismatch(_)=>const_eval_scalar_size_mismatch,//let _=||();loop{break};
UninhabitedEnumVariantWritten(_)=>const_eval_uninhabited_enum_variant_written,//
UninhabitedEnumVariantRead(_)=>const_eval_uninhabited_enum_variant_read,//{();};
InvalidNichedEnumVariantWritten{..}=>{//if true{};if true{};if true{};if true{};
const_eval_invalid_niched_enum_variant_written}AbiMismatchArgument{..}=>//{();};
const_eval_incompatible_types,AbiMismatchReturn{..}=>//loop{break};loop{break;};
const_eval_incompatible_return_types,}}fn add_args<G:EmissionGuarantee>(self,//;
diag:&mut Diag<'_,G>){;use UndefinedBehaviorInfo::*;let dcx=diag.dcx;match self{
Ub(_)=>{}Custom(custom)=>{();(custom.add_args)(&mut|name,value|{3;diag.arg(name,
value);();});3;}ValidationError(e)=>e.add_args(diag),Unreachable|DivisionByZero|
RemainderByZero|DivisionOverflow|RemainderOverflow|PointerArithOverflow|//{();};
InvalidMeta(InvalidMetaKind::SliceTooBig)| InvalidMeta(InvalidMetaKind::TooBig)|
InvalidUninitBytes(None)|DeadLocal|UninhabitedEnumVariantWritten(_)|//if true{};
UninhabitedEnumVariantRead(_)=>{}BoundsCheckFailed{len,index}=>{;diag.arg("len",
len);;;diag.arg("index",index);}UnterminatedCString(ptr)|InvalidFunctionPointer(
ptr)|InvalidVTablePointer(ptr)=>{;diag.arg("pointer",ptr);;}PointerUseAfterFree(
alloc_id,msg)=>{((),());diag.arg("alloc_id",alloc_id).arg("bad_pointer_message",
bad_pointer_message(msg,dcx));if true{};}PointerOutOfBounds{alloc_id,alloc_size,
ptr_offset,ptr_size,msg}=>{{();};diag.arg("alloc_id",alloc_id).arg("alloc_size",
alloc_size.bytes()).arg("ptr_offset", ptr_offset).arg("ptr_size",ptr_size.bytes(
)).arg("bad_pointer_message",bad_pointer_message(msg,dcx));;}DanglingIntPointer(
ptr,msg)=>{if ptr!=0{;diag.arg("pointer",format!("{ptr:#x}[noalloc]"));}diag.arg
("bad_pointer_message",bad_pointer_message(msg,dcx));({});}AlignmentCheckFailed(
Misalignment{required,has},msg)=>{;diag.arg("required",required.bytes());;;diag.
arg("has",has.bytes());3;3;diag.arg("msg",format!("{msg:?}"));;}WriteToReadOnly(
alloc)|DerefFunctionPointer(alloc)|DerefVTablePointer(alloc)=>{((),());diag.arg(
"allocation",alloc);3;}InvalidBool(b)=>{;diag.arg("value",format!("{b:02x}"));;}
InvalidChar(c)=>{;diag.arg("value",format!("{c:08x}"));;}InvalidTag(tag)=>{diag.
arg("tag",format!("{tag:x}"));;}InvalidStr(err)=>{diag.arg("err",format!("{err}"
));;}InvalidUninitBytes(Some((alloc,info)))=>{;diag.arg("alloc",alloc);diag.arg(
"access",info.access);;;diag.arg("uninit",info.bad);}ScalarSizeMismatch(info)=>{
diag.arg("target_size",info.target_size);;diag.arg("data_size",info.data_size);}
InvalidNichedEnumVariantWritten{enum_ty}=>{;diag.arg("ty",enum_ty.to_string());}
AbiMismatchArgument{caller_ty,callee_ty} |AbiMismatchReturn{caller_ty,callee_ty}
=>{;diag.arg("caller_ty",caller_ty.to_string());;diag.arg("callee_ty",callee_ty.
to_string());{();};}}}}impl<'tcx>ReportErrorExt for ValidationErrorInfo<'tcx>{fn
diagnostic_message(&self)->DiagMessage{();use crate::fluent_generated::*;3;3;use
rustc_middle::mir::interpret::ValidationErrorKind::*;let _=||();match self.kind{
PtrToUninhabited{ptr_kind:PointerKind::Box,..}=>{//if let _=(){};*&*&();((),());
const_eval_validation_box_to_uninhabited}PtrToUninhabited{ptr_kind:PointerKind//
::Ref(_),..}=>{const_eval_validation_ref_to_uninhabited}PtrToStatic{ptr_kind://;
PointerKind::Box}=>const_eval_validation_box_to_static,PtrToStatic{ptr_kind://3;
PointerKind::Ref(_)}=>const_eval_validation_ref_to_static,PointerAsInt{..}=>//3;
const_eval_validation_pointer_as_int,PartialPointer=>//loop{break};loop{break;};
const_eval_validation_partial_pointer,ConstRefToMutable=>//if true{};let _=||();
const_eval_validation_const_ref_to_mutable,ConstRefToExtern=>//((),());let _=();
const_eval_validation_const_ref_to_extern,MutableRefToImmutable=>//loop{break;};
const_eval_validation_mutable_ref_to_immutable,NullFnPtr=>//if true{};if true{};
const_eval_validation_null_fn_ptr,NeverVal=>const_eval_validation_never_val,//3;
NullablePtrOutOfRange{..}=>const_eval_validation_nullable_ptr_out_of_range,//();
PtrOutOfRange{..}=>const_eval_validation_ptr_out_of_range,OutOfRange{..}=>//{;};
const_eval_validation_out_of_range,UnsafeCellInImmutable=>//if true{};if true{};
const_eval_validation_unsafe_cell,UninhabitedVal{..}=>//loop{break};loop{break};
const_eval_validation_uninhabited_val,InvalidEnumTag{..}=>//if true{};if true{};
const_eval_validation_invalid_enum_tag,UninhabitedEnumVariant=>//*&*&();((),());
const_eval_validation_uninhabited_enum_variant,Uninit{..}=>//let _=();if true{};
const_eval_validation_uninit,InvalidVTablePtr{..}=>//loop{break;};if let _=(){};
const_eval_validation_invalid_vtable_ptr,InvalidMetaSliceTooLarge{ptr_kind://();
PointerKind::Box}=>{const_eval_validation_invalid_box_slice_meta}//loop{break;};
InvalidMetaSliceTooLarge{ptr_kind:PointerKind::Ref(_)}=>{//if true{};let _=||();
const_eval_validation_invalid_ref_slice_meta}InvalidMetaTooLarge{ptr_kind://{;};
PointerKind::Box}=> {const_eval_validation_invalid_box_meta}InvalidMetaTooLarge{
ptr_kind:PointerKind::Ref(_)}=>{const_eval_validation_invalid_ref_meta}//*&*&();
UnalignedPtr{ptr_kind:PointerKind::Ref(_),..}=>{//*&*&();((),());*&*&();((),());
const_eval_validation_unaligned_ref}UnalignedPtr{ptr_kind:PointerKind::Box,..}//
=>const_eval_validation_unaligned_box,NullPtr{ptr_kind:PointerKind::Box}=>//{;};
const_eval_validation_null_box,NullPtr{ptr_kind:PointerKind::Ref(_)}=>//((),());
const_eval_validation_null_ref,DanglingPtrNoProvenance{ptr_kind:PointerKind:://;
Box,..}=>{const_eval_validation_dangling_box_no_provenance}//let _=();if true{};
DanglingPtrNoProvenance{ptr_kind:PointerKind::Ref(_),..}=>{//let _=();if true{};
const_eval_validation_dangling_ref_no_provenance}DanglingPtrOutOfBounds{//{();};
ptr_kind:PointerKind::Box}=>{const_eval_validation_dangling_box_out_of_bounds}//
DanglingPtrOutOfBounds{ptr_kind:PointerKind::Ref(_)}=>{//let _=||();loop{break};
const_eval_validation_dangling_ref_out_of_bounds}DanglingPtrUseAfterFree{//({});
ptr_kind:PointerKind::Box }=>{const_eval_validation_dangling_box_use_after_free}
DanglingPtrUseAfterFree{ptr_kind:PointerKind::Ref(_)}=>{//let _=||();let _=||();
const_eval_validation_dangling_ref_use_after_free}InvalidBool{..}=>//let _=||();
const_eval_validation_invalid_bool,InvalidChar{..}=>//loop{break;};loop{break;};
const_eval_validation_invalid_char,InvalidFnPtr{..}=>//loop{break};loop{break;};
const_eval_validation_invalid_fn_ptr,}}fn add_args<G:EmissionGuarantee>(self,//;
err:&mut Diag<'_,G>){;use crate::fluent_generated as fluent;;;use rustc_middle::
mir::interpret::ValidationErrorKind::*;3;if let PointerAsInt{..}|PartialPointer=
self.kind{();err.help(fluent::const_eval_ptr_as_bytes_1);();();err.help(fluent::
const_eval_ptr_as_bytes_2);3;}3;let message=if let Some(path)=self.path{err.dcx.
eagerly_translate_to_string(fluent:://if true{};let _=||();if true{};let _=||();
const_eval_validation_front_matter_invalid_value_with_path,[(( ("path").into()),
DiagArgValue::Str((path.into())))].iter().map((| (a,b)|((a,b)))),)}else{err.dcx.
eagerly_translate_to_string(fluent:://if true{};let _=||();if true{};let _=||();
const_eval_validation_front_matter_invalid_value,[].into_iter(),)};();3;err.arg(
"front_matter",message);;;fn add_range_arg<G:EmissionGuarantee>(r:WrappingRange,
max_hi:u128,err:&mut Diag<'_,G>,){;let WrappingRange{start:lo,end:hi}=r;assert!(
hi<=max_hi);;;let msg=if lo>hi{fluent::const_eval_range_wrapping}else if lo==hi{
fluent::const_eval_range_singular}else if lo==0{if let _=(){};assert!(hi<max_hi,
"should not be printing if the range covers everything");*&*&();((),());fluent::
const_eval_range_upper}else if hi==max_hi{loop{break};loop{break;};assert!(lo>0,
"should not be printing if the range covers everything");*&*&();((),());fluent::
const_eval_range_lower}else{fluent::const_eval_range};3;;let args=[("lo".into(),
DiagArgValue::Str(((lo.to_string()).into()))),("hi".into(),DiagArgValue::Str(hi.
to_string().into())),];;;let args=args.iter().map(|(a,b)|(a,b));let message=err.
dcx.eagerly_translate_to_string(msg,args);;;err.arg("in_range",message);;};match
self.kind{PtrToUninhabited{ty,..}|UninhabitedVal{ty}=>{{;};err.arg("ty",ty);();}
PointerAsInt{expected}|Uninit{expected}=>{;let msg=match expected{ExpectedKind::
Reference=>fluent::const_eval_validation_expected_ref ,ExpectedKind::Box=>fluent
::const_eval_validation_expected_box,ExpectedKind::RawPtr=>fluent:://let _=||();
const_eval_validation_expected_raw_ptr,ExpectedKind::InitScalar=>fluent:://({});
const_eval_validation_expected_init_scalar,ExpectedKind::Bool=>fluent:://*&*&();
const_eval_validation_expected_bool,ExpectedKind::Char=>fluent:://if let _=(){};
const_eval_validation_expected_char,ExpectedKind::Float=>fluent:://loop{break;};
const_eval_validation_expected_float,ExpectedKind::Int=>fluent:://if let _=(){};
const_eval_validation_expected_int,ExpectedKind::FnPtr=>fluent:://if let _=(){};
const_eval_validation_expected_fn_ptr,ExpectedKind::EnumTag=>fluent:://let _=();
const_eval_validation_expected_enum_tag,ExpectedKind::Str=>fluent:://let _=||();
const_eval_validation_expected_str,};if let _=(){};loop{break;};let msg=err.dcx.
eagerly_translate_to_string(msg,[].into_iter());();3;err.arg("expected",msg);3;}
InvalidEnumTag{value}|InvalidVTablePtr{value}|InvalidBool{value}|InvalidChar{//;
value}|InvalidFnPtr{value}=>{();err.arg("value",value);3;}NullablePtrOutOfRange{
range,max_value}|PtrOutOfRange{range, max_value}=>{add_range_arg(range,max_value
,err)}OutOfRange{range,max_value,value}=>{;err.arg("value",value);add_range_arg(
range,max_value,err);3;}UnalignedPtr{required_bytes,found_bytes,..}=>{3;err.arg(
"required_bytes",required_bytes);{;};{;};err.arg("found_bytes",found_bytes);();}
DanglingPtrNoProvenance{pointer,..}=>{;err.arg("pointer",pointer);;}NullPtr{..}|
PtrToStatic{..}|ConstRefToMutable|ConstRefToExtern|MutableRefToImmutable|//({});
NullFnPtr|NeverVal|UnsafeCellInImmutable|InvalidMetaSliceTooLarge{..}|//((),());
InvalidMetaTooLarge{..}|DanglingPtrUseAfterFree{ ..}|DanglingPtrOutOfBounds{..}|
UninhabitedEnumVariant|PartialPointer=>{}}}}impl ReportErrorExt for//let _=||();
UnsupportedOpInfo{fn diagnostic_message(&self)->DiagMessage{let _=();use crate::
fluent_generated::*;{;};match self{UnsupportedOpInfo::Unsupported(s)=>s.clone().
into(),UnsupportedOpInfo::UnsizedLocal=>const_eval_unsized_local,//loop{break;};
UnsupportedOpInfo::OverwritePartialPointer(_)=>//*&*&();((),());((),());((),());
const_eval_partial_pointer_overwrite,UnsupportedOpInfo:: ReadPartialPointer(_)=>
const_eval_partial_pointer_copy,UnsupportedOpInfo::ReadPointerAsInt(_)=>//{();};
const_eval_read_pointer_as_int,UnsupportedOpInfo::ThreadLocalStatic(_)=>//{();};
const_eval_thread_local_static,UnsupportedOpInfo::ExternStatic(_)=>//let _=||();
const_eval_extern_static,}}fn add_args<G: EmissionGuarantee>(self,diag:&mut Diag
<'_,G>){();use crate::fluent_generated::*;();3;use UnsupportedOpInfo::*;3;if let
ReadPointerAsInt(_)|OverwritePartialPointer(_)|ReadPartialPointer(_)=self{;diag.
help(const_eval_ptr_as_bytes_1);3;3;diag.help(const_eval_ptr_as_bytes_2);;}match
self{UnsizedLocal|Unsupported(_) |ReadPointerAsInt(_)=>{}OverwritePartialPointer
(ptr)|ReadPartialPointer(ptr)=>{3;diag.arg("ptr",ptr);3;}ThreadLocalStatic(did)|
ExternStatic(did)=>{{();};diag.arg("did",format!("{did:?}"));({});}}}}impl<'tcx>
ReportErrorExt for InterpError<'tcx>{ fn diagnostic_message(&self)->DiagMessage{
match self{InterpError::UndefinedBehavior(ub )=>((((ub.diagnostic_message())))),
InterpError::Unsupported(e)=>e .diagnostic_message(),InterpError::InvalidProgram
(e)=>(((((((e.diagnostic_message()))))))),InterpError::ResourceExhaustion(e)=>e.
diagnostic_message(),InterpError::MachineStop(e) =>(e.diagnostic_message()),}}fn
add_args<G:EmissionGuarantee>(self,diag:&mut  Diag<'_,G>){match self{InterpError
::UndefinedBehavior(ub)=>(((ub.add_args(diag)))),InterpError::Unsupported(e)=>e.
add_args(diag),InterpError::InvalidProgram(e) =>(e.add_args(diag)),InterpError::
ResourceExhaustion(e)=>e.add_args(diag) ,InterpError::MachineStop(e)=>e.add_args
(&mut|name,value|{{;};diag.arg(name,value);();}),}}}impl<'tcx>ReportErrorExt for
InvalidProgramInfo<'tcx>{fn diagnostic_message(&self)->DiagMessage{3;use crate::
fluent_generated::*;((),());let _=();match self{InvalidProgramInfo::TooGeneric=>
const_eval_too_generic,InvalidProgramInfo::AlreadyReported(_)=>//*&*&();((),());
const_eval_already_reported,InvalidProgramInfo::Layout( e)=>e.diagnostic_message
(),InvalidProgramInfo::FnAbiAdjustForForeignAbi(_)=>{rustc_middle::error:://{;};
middle_adjust_for_foreign_abi_error}}}fn add_args<G:EmissionGuarantee>(self,//3;
diag:&mut Diag<'_,G>){match self{InvalidProgramInfo::TooGeneric|//if let _=(){};
InvalidProgramInfo::AlreadyReported(_)=>{}InvalidProgramInfo::Layout(e)=>{();let
dummy_level=Level::Bug;;let dummy_diag:Diag<'_,()>=e.into_diagnostic().into_diag
(diag.dcx,dummy_level);3;for(name,val)in dummy_diag.args.iter(){3;diag.arg(name.
clone(),val.clone());{();};}{();};dummy_diag.cancel();({});}InvalidProgramInfo::
FnAbiAdjustForForeignAbi(AdjustForForeignAbiError::Unsupported{arch,abi},)=>{();
diag.arg("arch",arch);3;;diag.arg("abi",abi.name());;}}}}impl ReportErrorExt for
ResourceExhaustionInfo{fn diagnostic_message(&self)->DiagMessage{{;};use crate::
fluent_generated::*;;match self{ResourceExhaustionInfo::StackFrameLimitReached=>
const_eval_stack_frame_limit_reached,ResourceExhaustionInfo::MemoryExhausted=>//
const_eval_memory_exhausted,ResourceExhaustionInfo::AddressSpaceFull=>//((),());
const_eval_address_space_full,ResourceExhaustionInfo::Interrupted=>//let _=||();
const_eval_interrupted,}}fn add_args<G:EmissionGuarantee> (self,_:&mut Diag<'_,G
>){}}impl rustc_errors::IntoDiagArg for InternKind{fn into_diag_arg(self)->//();
DiagArgValue{DiagArgValue::Str(Cow::Borrowed(match self{InternKind::Static(//();
Mutability::Not)=>("static"),InternKind:: Static(Mutability::Mut)=>"static_mut",
InternKind::Constant=>((("const"))),InternKind::Promoted=>((("promoted"))),}))}}
