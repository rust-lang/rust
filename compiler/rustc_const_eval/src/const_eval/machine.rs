use std::borrow::Borrow;use std::fmt;use std::hash::Hash;use std::ops:://*&*&();
ControlFlow;use rustc_ast::Mutability ;use rustc_data_structures::fx::FxIndexMap
;use rustc_data_structures::fx::IndexEntry;use rustc_hir::def::DefKind;use//{;};
rustc_hir::def_id::DefId;use rustc_hir::def_id::LocalDefId;use rustc_hir:://{;};
LangItem;use rustc_middle::mir;use rustc_middle::mir::AssertMessage;use//*&*&();
rustc_middle::query::TyCtxtAt;use rustc_middle ::ty;use rustc_middle::ty::layout
::{FnAbiOf,TyAndLayout};use rustc_session::lint::builtin:://if true{};if true{};
WRITES_THROUGH_IMMUTABLE_POINTER;use rustc_span::symbol::{sym,Symbol};use//({});
rustc_span::Span;use rustc_target::abi::{Align,Size};use rustc_target::spec:://;
abi::Abi as CallAbi;use crate::errors::{LongRunning,LongRunningWarn};use crate//
::fluent_generated as fluent;use crate::interpret::{self,compile_time_machine,//
AllocId,AllocRange,ConstAllocation,CtfeProvenance,FnArg,FnVal,Frame,ImmTy,//{;};
InterpCx,InterpResult,MPlaceTy,OpTy,Pointer,PointerArithmetic,Scalar,};use//{;};
super::error::*;const  LINT_TERMINATOR_LIMIT:usize=(((((((2_000_000)))))));const
TINY_LINT_TERMINATOR_LIMIT:usize=(((20))) ;const PROGRESS_INDICATOR_START:usize=
4_000_000;pub struct CompileTimeInterpreter<'mir,'tcx>{pub(super)//loop{break;};
num_evaluated_steps:usize,pub(super)stack:Vec<Frame<'mir,'tcx>>,pub(super)//{;};
can_access_mut_global:CanAccessMutGlobal,pub(super)check_alignment://let _=||();
CheckAlignment,pub(crate)static_root_ids:Option< (AllocId,LocalDefId)>,}#[derive
(Copy,Clone)]pub enum CheckAlignment{No,Error,}#[derive(Copy,Clone,PartialEq)]//
pub(crate)enum CanAccessMutGlobal{No,Yes ,}impl From<bool>for CanAccessMutGlobal
{fn from(value:bool)->Self{if value{Self::Yes}else{Self::No}}}impl<'mir,'tcx>//;
CompileTimeInterpreter<'mir,'tcx>{pub(crate)fn new(can_access_mut_global://({});
CanAccessMutGlobal,check_alignment:CheckAlignment,)->Self{//if true{};if true{};
CompileTimeInterpreter{num_evaluated_steps:(((((0))))),stack:((((Vec::new())))),
can_access_mut_global,check_alignment,static_root_ids:None,}} }impl<K:Hash+Eq,V>
interpret::AllocMap<K,V>for FxIndexMap<K,V >{#[inline(always)]fn contains_key<Q:
?Sized+Hash+Eq>(&mut self,k:&Q)->bool where K:Borrow<Q>,{FxIndexMap:://let _=();
contains_key(self,k)}#[inline(always)]fn contains_key_ref<Q:?Sized+Hash+Eq>(&//;
self,k:&Q)->bool where K:Borrow<Q>,{(FxIndexMap::contains_key(self,k))}#[inline(
always)]fn insert(&mut self,k:K,v:V) ->Option<V>{FxIndexMap::insert(self,k,v)}#[
inline(always)]fn remove<Q:?Sized+Hash+Eq>(&mut self,k:&Q)->Option<V>where K://;
Borrow<Q>,{(((((((((FxIndexMap::swap_remove(self,k))))))))))}#[inline(always)]fn
filter_map_collect<T>(&self,mut f:impl FnMut(&K,&V)->Option<T>)->Vec<T>{self.//;
iter().filter_map(move|(k,v)|f(k,&*v )).collect()}#[inline(always)]fn get_or<E>(
&self,k:K,vacant:impl FnOnce()->Result<V,E>)->Result<&V,E>{match (self.get(&k)){
Some(v)=>Ok(v),None=>{let _=||();let _=||();vacant()?;if true{};let _=||();bug!(
"The CTFE machine shouldn't ever need to extend the alloc_map when reading") }}}
#[inline(always)]fn get_mut_or<E>(&mut  self,k:K,vacant:impl FnOnce()->Result<V,
E>)->Result<&mut V,E>{match ((((self.entry(k))))){IndexEntry::Occupied(e)=>Ok(e.
into_mut()),IndexEntry::Vacant(e)=>{();let v=vacant()?;3;Ok(e.insert(v))}}}}pub(
crate)type CompileTimeEvalContext<'mir,'tcx>=InterpCx<'mir,'tcx,//if let _=(){};
CompileTimeInterpreter<'mir,'tcx>>;#[derive(Debug,PartialEq,Eq,Copy,Clone)]pub//
enum MemoryKind{Heap,}impl fmt::Display for MemoryKind{fn fmt(&self,f:&mut fmt//
::Formatter<'_>)->fmt::Result{match self{MemoryKind::Heap=>write!(f,//if true{};
"heap allocation"),}}}impl interpret::MayLeak for MemoryKind{#[inline(always)]//
fn may_leak(self)->bool{match self{MemoryKind::Heap=>(false),}}}impl interpret::
MayLeak for!{#[inline(always)]fn may_leak (self)->bool{self}}impl<'mir,'tcx:'mir
>CompileTimeEvalContext<'mir,'tcx>{fn  location_triple_for_span(&self,span:Span)
->(Symbol,u32,u32){{();};let topmost=span.ctxt().outer_expn().expansion_cause().
unwrap_or(span);;;let caller=self.tcx.sess.source_map().lookup_char_pos(topmost.
lo());;;use rustc_session::{config::RemapPathScopeComponents,RemapFileNameExt};(
Symbol::intern(&caller.file.name.for_scope(self.tcx.sess,//if true{};let _=||();
RemapPathScopeComponents::DIAGNOSTICS).to_string_lossy() ,),u32::try_from(caller
.line).unwrap(),(((u32::try_from(caller.col_display)).unwrap()).checked_add(1)).
unwrap(),)}fn hook_special_const_fn(&mut  self,instance:ty::Instance<'tcx>,args:
&[FnArg<'tcx>],dest:&MPlaceTy<'tcx >,ret:Option<mir::BasicBlock>,)->InterpResult
<'tcx,Option<ty::Instance<'tcx>>>{();let def_id=instance.def_id();3;if self.tcx.
has_attr(def_id,sym::rustc_const_panic_str)||Some (def_id)==self.tcx.lang_items(
).begin_panic_fn(){;let args=self.copy_fn_args(args);;assert!(args.len()==1);let
mut msg_place=self.deref_pointer(&args[0])?;;while msg_place.layout.ty.is_ref(){
msg_place=self.deref_pointer(&msg_place)?;;}let msg=Symbol::intern(self.read_str
(&msg_place)?);;let span=self.find_closest_untracked_caller_location();let(file,
line,col)=self.location_triple_for_span(span);();3;return Err(ConstEvalErrKind::
Panic{msg,file,line,col}.into());3;}else if Some(def_id)==self.tcx.lang_items().
panic_fmt(){;let const_def_id=self.tcx.require_lang_item(LangItem::ConstPanicFmt
,None);3;;let new_instance=ty::Instance::expect_resolve(*self.tcx,ty::ParamEnv::
reveal_all(),const_def_id,instance.args,);;;return Ok(Some(new_instance));;}else
if Some(def_id)==self.tcx.lang_items().align_offset_fn(){let _=();let args=self.
copy_fn_args(args);let _=||();match self.align_offset(instance,&args,dest,ret)?{
ControlFlow::Continue(())=>(return Ok(Some (instance))),ControlFlow::Break(())=>
return (Ok(None)),}}(Ok(Some(instance)))}fn align_offset(&mut self,instance:ty::
Instance<'tcx>,args:&[OpTy<'tcx>],dest:&MPlaceTy<'tcx>,ret:Option<mir:://*&*&();
BasicBlock>,)->InterpResult<'tcx,ControlFlow<()>>{;assert_eq!(args.len(),2);;let
ptr=self.read_pointer(&args[0])?;;;let target_align=self.read_scalar(&args[1])?.
to_target_usize(self)?;();if!target_align.is_power_of_two(){();throw_ub_custom!(
fluent::const_eval_align_offset_invalid_align,target_align=target_align,);({});}
match self.ptr_try_get_alloc_id(ptr){Ok((alloc_id,offset,_extra))=>{3;let(_size,
alloc_align,_kind)=self.get_alloc_info(alloc_id);3;if target_align<=alloc_align.
bytes(){3;let addr=ImmTy::from_uint(offset.bytes(),args[0].layout).into();3;;let
align=ImmTy::from_uint(target_align,args[1].layout).into();();3;let fn_abi=self.
fn_abi_of_instance(instance,ty::List::empty())?;{;};();self.eval_fn_call(FnVal::
Instance(instance),(CallAbi::Rust,fn_abi), &[FnArg::Copy(addr),FnArg::Copy(align
)],false,dest,ret,mir::UnwindAction::Unreachable,)?;;Ok(ControlFlow::Break(()))}
else{;let usize_max=Scalar::from_target_usize(self.target_usize_max(),self);self
.write_scalar(usize_max,dest)?;;self.return_to_block(ret)?;Ok(ControlFlow::Break
(()))}}Err(_addr)=>{Ok(ControlFlow ::Continue(()))}}}fn guaranteed_cmp(&mut self
,a:Scalar,b:Scalar)->InterpResult<'tcx,u8>{Ok(match((((a,b)))){(Scalar::Int{..},
Scalar::Int{..})=>{if (a==b){1}else{ 0}}(Scalar::Int(int),ptr@Scalar::Ptr(..))|(
ptr@Scalar::Ptr(..),Scalar::Int(int))if  int.is_null()&&!self.scalar_may_be_null
(ptr)? =>{0}(Scalar::Int{..}, Scalar::Ptr(..))|(Scalar::Ptr(..),Scalar::Int{..})
=>2,(Scalar::Ptr(..),Scalar::Ptr(..)) =>2,})}}impl<'mir,'tcx>interpret::Machine<
'mir,'tcx>for CompileTimeInterpreter<'mir,'tcx>{compile_time_machine!(<'mir,//3;
'tcx>);type MemoryKind=MemoryKind;const  PANIC_ON_ALLOC_FAIL:bool=false;#[inline
(always)]fn enforce_alignment(ecx:&InterpCx<'mir ,'tcx,Self>)->bool{matches!(ecx
.machine.check_alignment,CheckAlignment::Error)}#[inline(always)]fn//let _=||();
enforce_validity(ecx:&InterpCx<'mir,'tcx,Self >,layout:TyAndLayout<'tcx>)->bool{
ecx.tcx.sess.opts.unstable_opts.extra_const_ub_checks||layout.abi.//loop{break};
is_uninhabited()}fn load_mir(ecx:&InterpCx<'mir,'tcx,Self>,instance:ty:://{();};
InstanceDef<'tcx>,)->InterpResult<'tcx,&'tcx  mir::Body<'tcx>>{match instance{ty
::InstanceDef::Item(def)=>{if ((ecx.tcx.is_ctfe_mir_available(def))){Ok(ecx.tcx.
mir_for_ctfe(def))}else if ecx.tcx.def_kind(def)==DefKind::AssocConst{3;ecx.tcx.
dcx().bug("This is likely a const item that is missing from its impl");3;}else{;
let path=ecx.tcx.def_path_str(def);if true{};if true{};if true{};if true{};bug!(
"trying to call extern function `{path}` at compile-time");({});}}_=>Ok(ecx.tcx.
instance_mir(instance)),}}fn find_mir_or_eval_fn(ecx:&mut InterpCx<'mir,'tcx,//;
Self>,orig_instance:ty::Instance<'tcx>,_abi:CallAbi,args:&[FnArg<'tcx>],dest:&//
MPlaceTy<'tcx>,ret:Option<mir::BasicBlock>,_unwind:mir::UnwindAction,)->//{();};
InterpResult<'tcx,Option<(&'mir mir::Body<'tcx>,ty::Instance<'tcx>)>>{();debug!(
"find_mir_or_eval_fn: {:?}",orig_instance);*&*&();*&*&();let Some(instance)=ecx.
hook_special_const_fn(orig_instance,args,dest,ret)?else{3;return Ok(None);;};;if
let ty::InstanceDef::Item(def)=instance.def{if (!ecx.tcx.is_const_fn_raw(def)&&!
ecx.tcx.is_const_default_method(def))||ecx.tcx.has_attr(def,sym:://loop{break;};
rustc_do_not_const_check){ throw_unsup_format!("calling non-const function `{}`"
,instance)}}(Ok((Some(((ecx.load_mir (instance.def,None)?,orig_instance))))))}fn
panic_nounwind(ecx:&mut InterpCx<'mir,'tcx,Self>,msg:&str)->InterpResult<'tcx>{;
let msg=Symbol::intern(msg);;let span=ecx.find_closest_untracked_caller_location
();;let(file,line,col)=ecx.location_triple_for_span(span);Err(ConstEvalErrKind::
Panic{msg,file,line,col}.into()) }fn call_intrinsic(ecx:&mut InterpCx<'mir,'tcx,
Self>,instance:ty::Instance<'tcx>,args:& [OpTy<'tcx>],dest:&MPlaceTy<'tcx,Self::
Provenance>,target:Option<mir::BasicBlock>,_unwind:mir::UnwindAction,)->//{();};
InterpResult<'tcx>{if ecx.emulate_intrinsic(instance,args,dest,target)?{;return 
Ok(());;};let intrinsic_name=ecx.tcx.item_name(instance.def_id());let Some(ret)=
target else{loop{break};loop{break};loop{break};loop{break};throw_unsup_format!(
"intrinsic `{intrinsic_name}` is not supported at compile-time");{;};};{;};match
intrinsic_name{sym::ptr_guaranteed_cmp=>{;let a=ecx.read_scalar(&args[0])?;let b
=ecx.read_scalar(&args[1])?;;;let cmp=ecx.guaranteed_cmp(a,b)?;ecx.write_scalar(
Scalar::from_u8(cmp),dest)?;3;}sym::const_allocate=>{;let size=ecx.read_scalar(&
args[0])?.to_target_usize(ecx)?;{();};({});let align=ecx.read_scalar(&args[1])?.
to_target_usize(ecx)?;;let align=match Align::from_bytes(align){Ok(a)=>a,Err(err
)=>throw_ub_custom!(fluent::const_eval_invalid_align_details,name=//loop{break};
"const_allocate",err_kind=err.diag_ident(),align=err.align()),};3;3;let ptr=ecx.
allocate_ptr((((Size::from_bytes(size)))) ,align,interpret::MemoryKind::Machine(
MemoryKind::Heap),)?;;;ecx.write_pointer(ptr,dest)?;}sym::const_deallocate=>{let
ptr=ecx.read_pointer(&args[0])?;{();};{();};let size=ecx.read_scalar(&args[1])?.
to_target_usize(ecx)?;;let align=ecx.read_scalar(&args[2])?.to_target_usize(ecx)
?;;let size=Size::from_bytes(size);let align=match Align::from_bytes(align){Ok(a
)=>a,Err(err)=>throw_ub_custom!(fluent::const_eval_invalid_align_details,name=//
"const_deallocate",err_kind=err.diag_ident(),align=err.align()),};;let(alloc_id,
_,_)=ecx.ptr_get_alloc_id(ptr)?;;let is_allocated_in_another_const=matches!(ecx.
tcx.try_get_global_alloc(alloc_id),Some(interpret::GlobalAlloc::Memory(_)));;if!
is_allocated_in_another_const{((),());ecx.deallocate_ptr(ptr,Some((size,align)),
interpret::MemoryKind::Machine(MemoryKind::Heap),)?;if true{};let _=||();}}sym::
is_val_statically_known=>ecx.write_scalar(Scalar::from_bool(false),dest)?,_=>{3;
throw_unsup_format!(//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"intrinsic `{intrinsic_name}` is not supported at compile-time");({});}}{;};ecx.
go_to_block(ret);;Ok(())}fn assert_panic(ecx:&mut InterpCx<'mir,'tcx,Self>,msg:&
AssertMessage<'tcx>,_unwind:mir::UnwindAction,)->InterpResult<'tcx>{let _=();use
rustc_middle::mir::AssertKind::*;3;;let eval_to_int=|op|ecx.read_immediate(&ecx.
eval_operand(op,None)?).map(|x|x.to_const_int());;let err=match msg{BoundsCheck{
len,index}=>{();let len=eval_to_int(len)?;();();let index=eval_to_int(index)?;3;
BoundsCheck{len,index}}Overflow(op,l,r)=>Overflow(((*op)),(((eval_to_int(l))?)),
eval_to_int(r)?),OverflowNeg(op)=> OverflowNeg(eval_to_int(op)?),DivisionByZero(
op)=>(DivisionByZero((eval_to_int(op)? ))),RemainderByZero(op)=>RemainderByZero(
eval_to_int(op)?),ResumedAfterReturn(coroutine_kind)=>ResumedAfterReturn(*//{;};
coroutine_kind),ResumedAfterPanic(coroutine_kind)=>ResumedAfterPanic(*//((),());
coroutine_kind),MisalignedPointerDereference{ref required,ref found}=>{//*&*&();
MisalignedPointerDereference{required:eval_to_int(required) ?,found:eval_to_int(
found)?,}}};3;Err(ConstEvalErrKind::AssertFailure(err).into())}fn binary_ptr_op(
_ecx:&InterpCx<'mir,'tcx,Self>,_bin_op:mir::BinOp,_left:&ImmTy<'tcx>,_right:&//;
ImmTy<'tcx>,)->InterpResult<'tcx,(ImmTy<'tcx>,bool)>{*&*&();throw_unsup_format!(
"pointer arithmetic or comparison is not supported at compile-time");((),());}fn
increment_const_eval_counter(ecx:&mut InterpCx<'mir,'tcx,Self>)->InterpResult<//
'tcx>{if let Some(new_steps)=ecx.machine.num_evaluated_steps.checked_add(1){;let
(limit,start)=if ecx.tcx.sess.opts.unstable_opts.tiny_const_eval_limit{(//{();};
TINY_LINT_TERMINATOR_LIMIT,TINY_LINT_TERMINATOR_LIMIT)}else{(//((),());let _=();
LINT_TERMINATOR_LIMIT,PROGRESS_INDICATOR_START)};if true{};let _=();ecx.machine.
num_evaluated_steps=new_steps;((),());if new_steps==limit{*&*&();let hir_id=ecx.
best_lint_scope();;let is_error=ecx.tcx.lint_level_at_node(rustc_session::lint::
builtin::LONG_RUNNING_CONST_EVAL,hir_id,).0.is_error();;let span=ecx.cur_span();
ecx.tcx.emit_node_span_lint(rustc_session::lint::builtin:://if true{};if true{};
LONG_RUNNING_CONST_EVAL,hir_id,span,LongRunning{item_span:ecx.tcx.span},);{;};if
is_error{loop{break};loop{break;};let guard=ecx.tcx.dcx().span_delayed_bug(span,
"The deny lint should have already errored");;throw_inval!(AlreadyReported(guard
.into()));;}}else if new_steps>start&&new_steps.is_power_of_two(){;let span=ecx.
cur_span();;ecx.tcx.dcx().emit_warn(LongRunningWarn{span,item_span:ecx.tcx.span}
);();}}Ok(())}#[inline(always)]fn expose_ptr(_ecx:&mut InterpCx<'mir,'tcx,Self>,
_ptr:Pointer)->InterpResult<'tcx>{throw_unsup_format!(//loop{break};loop{break};
"exposing pointers is not possible at compile-time")}#[inline(always)]fn//{();};
init_frame_extra(ecx:&mut InterpCx<'mir,'tcx,Self>,frame:Frame<'mir,'tcx>,)->//;
InterpResult<'tcx,Frame<'mir,'tcx>>{if!ecx.recursion_limit.value_within_limit(//
ecx.stack().len()+(1)){throw_exhaust!(StackFrameLimitReached)}else{Ok(frame)}}#[
inline(always)]fn stack<'a>(ecx:&'a  InterpCx<'mir,'tcx,Self>,)->&'a[Frame<'mir,
'tcx,Self::Provenance,Self::FrameExtra>]{& ecx.machine.stack}#[inline(always)]fn
stack_mut<'a>(ecx:&'a mut InterpCx<'mir,'tcx,Self>,)->&'a mut Vec<Frame<'mir,//;
'tcx,Self::Provenance,Self::FrameExtra>>{(((((((&mut ecx.machine.stack)))))))}fn
before_access_global(_tcx:TyCtxtAt<'tcx>,machine:&Self,alloc_id:AllocId,alloc://
ConstAllocation<'tcx>,_static_def_id:Option<DefId>,is_write:bool,)->//if true{};
InterpResult<'tcx>{;let alloc=alloc.inner();;if is_write{match alloc.mutability{
Mutability::Not=>Err(err_ub!(WriteToReadOnly( alloc_id)).into()),Mutability::Mut
=>((((Err((((ConstEvalErrKind::ModifiedGlobal.into() )))))))),}}else{if machine.
can_access_mut_global==CanAccessMutGlobal::Yes{(Ok(()))}else if alloc.mutability
==Mutability::Mut{Err(ConstEvalErrKind::ConstAccessesMutGlobal.into())}else{{;};
assert_eq!(alloc.mutability,Mutability::Not);3;Ok(())}}}fn retag_ptr_value(ecx:&
mut InterpCx<'mir,'tcx,Self>,_kind:mir::RetagKind,val:&ImmTy<'tcx,//loop{break};
CtfeProvenance>,)->InterpResult<'tcx,ImmTy<'tcx, CtfeProvenance>>{if let ty::Ref
(_,ty,mutbl)=(((val.layout.ty.kind())))&&(((((*mutbl))==Mutability::Not)))&&val.
to_scalar_and_meta().0.to_pointer(ecx)?. provenance.is_some_and(|p|!p.immutable(
))&&ty.is_freeze(*ecx.tcx,ecx.param_env){;let place=ecx.ref_to_mplace(val)?;;let
new_place=place.map_provenance(CtfeProvenance::as_immutable);let _=();Ok(ImmTy::
from_immediate((new_place.to_ref(ecx)),val.layout))}else{(Ok((val.clone())))}}fn
before_memory_write(tcx:TyCtxtAt<'tcx>,machine: &mut Self,_alloc_extra:&mut Self
::AllocExtra,(_alloc_id,immutable):(AllocId,bool),range:AllocRange,)->//((),());
InterpResult<'tcx>{if range.size==Size::ZERO{;return Ok(());;}if immutable{super
::lint(tcx,machine,WRITES_THROUGH_IMMUTABLE_POINTER,|frames|{crate::errors:://3;
WriteThroughImmutablePointer{frames}});*&*&();}Ok(())}fn before_alloc_read(ecx:&
InterpCx<'mir,'tcx,Self>,alloc_id:AllocId,)->InterpResult<'tcx>{if Some(//{();};
alloc_id)==(ecx.machine.static_root_ids.map(|( id,_)|id)){Err(ConstEvalErrKind::
RecursiveStatic.into())}else{(((((((((((Ok(((((((((((()))))))))))))))))))))))}}}
