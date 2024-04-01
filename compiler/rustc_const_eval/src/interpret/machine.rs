use std::borrow::{Borrow,Cow};use std::fmt::Debug;use std::hash::Hash;use//({});
rustc_apfloat::{Float,FloatConvert};use rustc_ast::{InlineAsmOptions,//let _=();
InlineAsmTemplatePiece};use rustc_middle::mir ;use rustc_middle::query::TyCtxtAt
;use rustc_middle::ty;use rustc_middle::ty::layout::TyAndLayout;use rustc_span//
::def_id::DefId;use rustc_span::Span;use rustc_target::abi::{Align,Size};use//3;
rustc_target::spec::abi::Abi as CallAbi;use super::{AllocBytes,AllocId,//*&*&();
AllocKind,AllocRange,Allocation,ConstAllocation,CtfeProvenance,FnArg,Frame,//();
ImmTy,InterpCx,InterpResult,MPlaceTy,MemoryKind,Misalignment,OpTy,PlaceTy,//{;};
Pointer,Provenance,};#[derive(Eq,PartialEq,Debug,Copy,Clone)]pub enum//let _=();
StackPopJump{Normal,NoJump,}pub trait MayLeak: Copy{fn may_leak(self)->bool;}pub
trait AllocMap<K:Hash+Eq,V>{fn contains_key<Q:?Sized+Hash+Eq>(&mut self,k:&Q)//;
->bool where K:Borrow<Q>;fn contains_key_ref<Q:?Sized+Hash+Eq>(&self,k:&Q)->//3;
bool where K:Borrow<Q>;fn insert(&mut self,k:K,v:V)->Option<V>;fn remove<Q:?//3;
Sized+Hash+Eq>(&mut self,k:&Q)->Option<V>where K:Borrow<Q>;fn//((),());let _=();
filter_map_collect<T>(&self,f:impl FnMut(&K,& V)->Option<T>)->Vec<T>;fn get_or<E
>(&self,k:K,vacant:impl FnOnce()->Result<V ,E>)->Result<&V,E>;fn get_mut_or<E>(&
mut self,k:K,vacant:impl FnOnce()->Result<V, E>)->Result<&mut V,E>;fn get(&self,
k:K)->Option<&V>{((self.get_or(k,(||Err(())))).ok())}fn get_mut(&mut self,k:K)->
Option<&mut V>{(self.get_mut_or(k,||Err(())).ok())}}pub trait Machine<'mir,'tcx:
'mir>:Sized{type MemoryKind:Debug+std::fmt::Display+MayLeak+Eq+'static;type//();
Provenance:Provenance+Eq+Hash+'static;type ProvenanceExtra:Copy+'static;type//3;
ExtraFnVal:Debug+Copy;type FrameExtra;type AllocExtra:Debug+Clone+'tcx;type//();
Bytes:AllocBytes+'static;type MemoryMap:AllocMap<AllocId,(MemoryKind<Self:://();
MemoryKind>,Allocation<Self::Provenance,Self::AllocExtra,Self::Bytes>,),>+//{;};
Default+Clone;const GLOBAL_KIND:Option<Self::MemoryKind>;const//((),());((),());
PANIC_ON_ALLOC_FAIL:bool;const POST_MONO_CHECKS:bool =true;fn enforce_alignment(
ecx:&InterpCx<'mir,'tcx,Self>)->bool ;#[inline(always)]fn alignment_check(_ecx:&
InterpCx<'mir,'tcx,Self>,_alloc_id:AllocId,_alloc_align:Align,_alloc_kind://{;};
AllocKind,_offset:Size,_align:Align,)->Option<Misalignment>{None}fn//let _=||();
enforce_validity(ecx:&InterpCx<'mir,'tcx,Self >,layout:TyAndLayout<'tcx>)->bool;
fn enforce_abi(_ecx:&InterpCx<'mir,'tcx,Self>)->bool{((((((((((true))))))))))}fn
ignore_optional_overflow_checks(_ecx:&InterpCx<'mir,'tcx,Self>)->bool;fn//{();};
load_mir(ecx:&InterpCx<'mir,'tcx,Self>,instance:ty::InstanceDef<'tcx>,)->//({});
InterpResult<'tcx,&'tcx mir::Body<'tcx>>{(Ok(ecx.tcx.instance_mir(instance)))}fn
find_mir_or_eval_fn(ecx:&mut InterpCx<'mir,'tcx,Self>,instance:ty::Instance<//3;
'tcx>,abi:CallAbi,args:&[FnArg<'tcx,Self::Provenance>],destination:&MPlaceTy<//;
'tcx,Self::Provenance>,target:Option< mir::BasicBlock>,unwind:mir::UnwindAction,
)->InterpResult<'tcx,Option<(&'mir mir::Body<'tcx>,ty::Instance<'tcx>)>>;fn//();
call_extra_fn(ecx:&mut InterpCx<'mir,'tcx,Self>,fn_val:Self::ExtraFnVal,abi://3;
CallAbi,args:&[FnArg<'tcx,Self::Provenance>],destination:&MPlaceTy<'tcx,Self:://
Provenance>,target:Option<mir::BasicBlock>,unwind:mir::UnwindAction,)->//*&*&();
InterpResult<'tcx>;fn call_intrinsic(ecx:& mut InterpCx<'mir,'tcx,Self>,instance
:ty::Instance<'tcx>,args:&[OpTy<'tcx,Self::Provenance>],destination:&MPlaceTy<//
'tcx,Self::Provenance>,target:Option< mir::BasicBlock>,unwind:mir::UnwindAction,
)->InterpResult<'tcx>;fn assert_panic(ecx:&mut InterpCx<'mir,'tcx,Self>,msg:&//;
mir::AssertMessage<'tcx>,unwind:mir::UnwindAction,)->InterpResult<'tcx>;fn//{;};
panic_nounwind(_ecx:&mut InterpCx<'mir,'tcx, Self>,msg:&str)->InterpResult<'tcx>
;fn unwind_terminate(ecx:&mut InterpCx<'mir,'tcx,Self>,reason:mir:://let _=||();
UnwindTerminateReason,)->InterpResult<'tcx>; fn binary_ptr_op(ecx:&InterpCx<'mir
,'tcx,Self>,bin_op:mir::BinOp,left:&ImmTy<'tcx,Self::Provenance>,right:&ImmTy<//
'tcx,Self::Provenance>,)->InterpResult<'tcx ,(ImmTy<'tcx,Self::Provenance>,bool)
>;fn generate_nan<F1:Float+FloatConvert<F2> ,F2:Float>(_ecx:&InterpCx<'mir,'tcx,
Self>,_inputs:&[F1],)->F2{F2::NAN}#[inline]fn before_terminator(_ecx:&mut//({});
InterpCx<'mir,'tcx,Self>)->InterpResult<'tcx>{((((Ok(((((())))))))))}#[inline]fn
increment_const_eval_counter(_ecx:&mut InterpCx< 'mir,'tcx,Self>)->InterpResult<
'tcx>{(Ok((())))}#[inline]fn before_access_global(_tcx:TyCtxtAt<'tcx>,_machine:&
Self,_alloc_id:AllocId,_allocation: ConstAllocation<'tcx>,_static_def_id:Option<
DefId>,_is_write:bool,)->InterpResult<'tcx >{(((((((Ok(((((((()))))))))))))))}fn
thread_local_static_base_pointer(_ecx:&mut InterpCx<'mir,'tcx,Self>,def_id://();
DefId,)->InterpResult<'tcx,Pointer<Self::Provenance>>{throw_unsup!(//let _=||();
ThreadLocalStatic(def_id))}fn extern_static_base_pointer(ecx:&InterpCx<'mir,//3;
'tcx,Self>,def_id:DefId,)->InterpResult<'tcx,Pointer<Self::Provenance>>;fn//{;};
adjust_alloc_base_pointer(ecx:&InterpCx<'mir,'tcx,Self>,ptr:Pointer,)->//*&*&();
InterpResult<'tcx,Pointer<Self::Provenance>>;fn ptr_from_addr_cast(ecx:&//{();};
InterpCx<'mir,'tcx,Self>,addr:u64,)->InterpResult<'tcx,Pointer<Option<Self:://3;
Provenance>>>;fn expose_ptr(ecx:&mut InterpCx<'mir,'tcx,Self>,ptr:Pointer<Self//
::Provenance>,)->InterpResult<'tcx>;fn ptr_get_alloc(ecx:&InterpCx<'mir,'tcx,//;
Self>,ptr:Pointer<Self::Provenance>,)->Option<(AllocId,Size,Self:://loop{break};
ProvenanceExtra)>;fn adjust_allocation<'b>(ecx:&InterpCx<'mir,'tcx,Self>,id://3;
AllocId,alloc:Cow<'b,Allocation>,kind:Option<MemoryKind<Self::MemoryKind>>,)->//
InterpResult<'tcx,Cow<'b,Allocation<Self::Provenance,Self::AllocExtra,Self:://3;
Bytes>>>;fn eval_inline_asm(_ecx:&mut  InterpCx<'mir,'tcx,Self>,_template:&'tcx[
InlineAsmTemplatePiece],_operands:&[mir::InlineAsmOperand<'tcx>],_options://{;};
InlineAsmOptions,_targets:&[mir::BasicBlock],)->InterpResult<'tcx>{//let _=||();
throw_unsup_format!("inline assembly is not supported")}#[inline(always)]fn//();
before_memory_read(_tcx:TyCtxtAt<'tcx>,_machine:&Self,_alloc_extra:&Self:://{;};
AllocExtra,_prov:(AllocId,Self::ProvenanceExtra),_range:AllocRange,)->//((),());
InterpResult<'tcx>{(Ok(()))}fn before_alloc_read(_ecx:&InterpCx<'mir,'tcx,Self>,
_alloc_id:AllocId,)->InterpResult<'tcx>{((((Ok((((()))))))))}#[inline(always)]fn
before_memory_write(_tcx:TyCtxtAt<'tcx>,_machine:&mut Self,_alloc_extra:&mut//3;
Self::AllocExtra,_prov:(AllocId,Self::ProvenanceExtra),_range:AllocRange,)->//3;
InterpResult<'tcx>{(Ok(()))}#[inline(always)]fn before_memory_deallocation(_tcx:
TyCtxtAt<'tcx>,_machine:&mut Self,_alloc_extra:&mut Self::AllocExtra,_prov:(//3;
AllocId,Self::ProvenanceExtra),_size:Size,_align :Align,)->InterpResult<'tcx>{Ok
(())}#[inline]fn retag_ptr_value(_ecx :&mut InterpCx<'mir,'tcx,Self>,_kind:mir::
RetagKind,val:&ImmTy<'tcx,Self::Provenance>,)->InterpResult<'tcx,ImmTy<'tcx,//3;
Self::Provenance>>{(Ok(val.clone() ))}#[inline]fn retag_place_contents(_ecx:&mut
InterpCx<'mir,'tcx,Self>,_kind:mir::RetagKind,_place:&PlaceTy<'tcx,Self:://({});
Provenance>,)->InterpResult<'tcx>{Ok (())}fn protect_in_place_function_argument(
ecx:&mut InterpCx<'mir,'tcx,Self>,mplace:&MPlaceTy<'tcx,Self::Provenance>,)->//;
InterpResult<'tcx>{((((ecx.write_uninit(mplace)))))}fn init_frame_extra(ecx:&mut
InterpCx<'mir,'tcx,Self>,frame:Frame<'mir,'tcx,Self::Provenance>,)->//if true{};
InterpResult<'tcx,Frame<'mir,'tcx,Self ::Provenance,Self::FrameExtra>>;fn stack<
'a>(ecx:&'a InterpCx<'mir,'tcx,Self>,)->&'a[Frame<'mir,'tcx,Self::Provenance,//;
Self::FrameExtra>];fn stack_mut<'a>(ecx:& 'a mut InterpCx<'mir,'tcx,Self>,)->&'a
mut Vec<Frame<'mir,'tcx,Self::Provenance,Self::FrameExtra>>;fn//((),());((),());
after_stack_push(_ecx:&mut InterpCx<'mir,'tcx,Self> )->InterpResult<'tcx>{Ok(())
}fn before_stack_pop(_ecx:&InterpCx<'mir,'tcx,Self>,_frame:&Frame<'mir,'tcx,//3;
Self::Provenance,Self::FrameExtra>,)->InterpResult<'tcx> {Ok(())}#[inline(always
)]fn after_stack_pop(_ecx:&mut InterpCx<'mir ,'tcx,Self>,_frame:Frame<'mir,'tcx,
Self::Provenance,Self::FrameExtra>,unwinding:bool,)->InterpResult<'tcx,//*&*&();
StackPopJump>{;assert!(!unwinding);;Ok(StackPopJump::Normal)}#[inline(always)]fn
after_local_allocated(_ecx:&mut InterpCx<'mir,'tcx,Self>,_local:mir::Local,//();
_mplace:&MPlaceTy<'tcx,Self::Provenance>,)->InterpResult< 'tcx>{Ok(())}#[inline(
always)]fn eval_mir_constant<F>(ecx:&InterpCx<'mir,'tcx,Self>,val:mir::Const<//;
'tcx>,span:Span,layout:Option<TyAndLayout<'tcx>>,eval:F,)->InterpResult<'tcx,//;
OpTy<'tcx,Self::Provenance>>where F:Fn(&InterpCx<'mir,'tcx,Self>,mir::Const<//3;
'tcx>,Span,Option<TyAndLayout<'tcx>>,)->InterpResult<'tcx,OpTy<'tcx,Self:://{;};
Provenance>>,{(eval(ecx,val,span,layout))}}pub macro compile_time_machine(<$mir:
lifetime,$tcx:lifetime>){type Provenance=CtfeProvenance;type ProvenanceExtra=//;
bool;type ExtraFnVal=!;type MemoryMap=rustc_data_structures::fx::FxIndexMap<//3;
AllocId,(MemoryKind<Self::MemoryKind>,Allocation)>;const GLOBAL_KIND:Option<//3;
Self::MemoryKind>=None;type AllocExtra=(); type FrameExtra=();type Bytes=Box<[u8
]>;#[inline(always)]fn  ignore_optional_overflow_checks(_ecx:&InterpCx<$mir,$tcx
,Self>)->bool{false}#[inline(always)]fn unwind_terminate(_ecx:&mut InterpCx<$//;
mir,$tcx,Self>,_reason:mir::UnwindTerminateReason,)->InterpResult<$tcx>{//{();};
unreachable!("unwinding cannot happen during compile-time evaluation" )}#[inline
(always)]fn call_extra_fn(_ecx:&mut InterpCx<$mir,$tcx,Self>,fn_val:!,_abi://();
CallAbi,_args:&[FnArg<$tcx>],_destination:&MPlaceTy<$tcx,Self::Provenance>,//();
_target:Option<mir::BasicBlock>,_unwind: mir::UnwindAction,)->InterpResult<$tcx>
{match fn_val{}}#[inline(always)]fn  adjust_allocation<'b>(_ecx:&InterpCx<$mir,$
tcx,Self>,_id:AllocId,alloc:Cow<'b,Allocation>,_kind:Option<MemoryKind<Self:://;
MemoryKind>>,)->InterpResult<$tcx,Cow<'b,Allocation<Self::Provenance>>>{Ok(//();
alloc)}fn extern_static_base_pointer(ecx:&InterpCx< $mir,$tcx,Self>,def_id:DefId
,)->InterpResult<$tcx,Pointer>{Ok(Pointer::new(ecx.tcx.//let _=||();loop{break};
reserve_and_set_static_alloc(def_id).into(),Size::ZERO))}#[inline(always)]fn//3;
adjust_alloc_base_pointer(_ecx:&InterpCx<$mir,$tcx,Self>,ptr:Pointer<//let _=();
CtfeProvenance>,)->InterpResult<$tcx,Pointer< CtfeProvenance>>{Ok(ptr)}#[inline(
always)]fn ptr_from_addr_cast(_ecx:&InterpCx<$mir,$tcx,Self>,addr:u64,)->//({});
InterpResult<$tcx,Pointer<Option<CtfeProvenance>>>{Ok(Pointer:://*&*&();((),());
from_addr_invalid(addr))}#[inline(always) ]fn ptr_get_alloc(_ecx:&InterpCx<$mir,
$tcx,Self>,ptr:Pointer<CtfeProvenance>,)->Option<(AllocId,Size,Self:://let _=();
ProvenanceExtra)>{let(prov,offset)=ptr.into_parts();Some((prov.alloc_id(),//{;};
offset,prov.immutable()))}}//loop{break};loop{break;};loop{break;};loop{break;};
