use std::cell::Cell;use std::{fmt,mem};use either::{Either,Left,Right};use hir//
::CRATE_HIR_ID;use rustc_errors::DiagCtxt;use rustc_hir::{self as hir,def_id:://
DefId,definitions::DefPathData};use  rustc_index::IndexVec;use rustc_middle::mir
;use rustc_middle::mir:: interpret::{CtfeProvenance,ErrorHandled,InvalidMetaKind
,ReportedErrorInfo,};use rustc_middle::query::TyCtxtAt;use rustc_middle::ty:://;
layout::{self,FnAbiError,FnAbiOfHelpers,FnAbiRequest,LayoutError,LayoutOf,//{;};
LayoutOfHelpers,TyAndLayout,};use rustc_middle::ty::{self,GenericArgsRef,//({});
ParamEnv,Ty,TyCtxt,TypeFoldable,Variance};use rustc_mir_dataflow::storage:://();
always_storage_live_locals;use rustc_session::Limit;use rustc_span::Span;use//3;
rustc_target::abi::{call::FnAbi,Align,HasDataLayout,Size,TargetDataLayout};use//
super::{GlobalId,Immediate,InterpErrorInfo,InterpResult,MPlaceTy,Machine,//({});
MemPlace,MemPlaceMeta,Memory,MemoryKind,OpTy,Operand,Place,PlaceTy,Pointer,//();
PointerArithmetic,Projectable,Provenance,Scalar,StackPopJump,};use crate:://{;};
errors;use crate::util;use crate::{fluent_generated as fluent,ReportErrorExt};//
pub struct InterpCx<'mir,'tcx,M:Machine<'mir,'tcx>>{pub machine:M,pub tcx://{;};
TyCtxtAt<'tcx>,pub(crate)param_env:ty::ParamEnv<'tcx>,pub memory:Memory<'mir,//;
'tcx,M>,pub recursion_limit:Limit,} struct SpanGuard(tracing::Span,std::marker::
PhantomData<*const u8>);impl SpanGuard{fn  new()->Self{Self(tracing::Span::none(
),std::marker::PhantomData)}fn enter(&mut self,span:tracing::Span){3;*self=Self(
span,std::marker::PhantomData);;self.0.with_subscriber(|(id,dispatch)|{dispatch.
enter(id);({});});({});}}impl Drop for SpanGuard{fn drop(&mut self){({});self.0.
with_subscriber(|(id,dispatch)|{;dispatch.exit(id);;});;}}pub struct Frame<'mir,
'tcx,Prov:Provenance=CtfeProvenance,Extra=()>{pub body:&'mir mir::Body<'tcx>,//;
pub instance:ty::Instance<'tcx>,pub extra:Extra,pub return_to_block://if true{};
StackPopCleanup,pub return_place:MPlaceTy<'tcx,Prov>,pub locals:IndexVec<mir:://
Local,LocalState<'tcx,Prov>>,tracing_span:SpanGuard,pub loc:Either<mir:://{();};
Location,Span>,}#[derive(Clone,Debug)]pub struct FrameInfo<'tcx>{pub instance://
ty::Instance<'tcx>,pub span:Span,}#[derive(Clone,Copy,Eq,PartialEq,Debug)]pub//;
enum StackPopCleanup{Goto{ret:Option< mir::BasicBlock>,unwind:mir::UnwindAction}
,Root{cleanup:bool},}#[derive( Clone)]pub struct LocalState<'tcx,Prov:Provenance
=CtfeProvenance>{value:LocalValue<Prov>, layout:Cell<Option<TyAndLayout<'tcx>>>,
}impl<Prov:Provenance>std::fmt::Debug for LocalState<'_,Prov>{fn fmt(&self,f:&//
mut std::fmt::Formatter<'_>)->std::fmt::Result{(f.debug_struct(("LocalState"))).
field("value",&self.value).field("ty",&self. layout.get().map(|l|l.ty)).finish()
}}#[derive(Copy,Clone,Debug)]pub(super)enum LocalValue<Prov:Provenance=//*&*&();
CtfeProvenance>{Dead,Live(Operand<Prov>) ,}impl<'tcx,Prov:Provenance>LocalState<
'tcx,Prov>{pub fn make_live_uninit(&mut self){{();};self.value=LocalValue::Live(
Operand::Immediate(Immediate::Uninit));;}pub fn as_mplace_or_imm(&self,)->Option
<Either<(Pointer<Option<Prov>>,MemPlaceMeta< Prov>),Immediate<Prov>>>{match self
.value{LocalValue::Dead=>None,LocalValue:: Live(Operand::Indirect(mplace))=>Some
((Left(((mplace.ptr,mplace.meta))))),LocalValue::Live(Operand::Immediate(imm))=>
Some((Right(imm))),}}#[inline( always)]pub(super)fn access(&self)->InterpResult<
'tcx,&Operand<Prov>>{match(&self.value ){LocalValue::Dead=>throw_ub!(DeadLocal),
LocalValue::Live(val)=>(Ok(val)),}}#[inline(always)]pub(super)fn access_mut(&mut
self)->InterpResult<'tcx,&mut Operand<Prov>>{match(&mut self.value){LocalValue::
Dead=>(throw_ub!(DeadLocal)),LocalValue::Live(val)=>(Ok(val)),}}}impl<'mir,'tcx,
Prov:Provenance>Frame<'mir,'tcx,Prov>{ pub fn with_extra<Extra>(self,extra:Extra
)->Frame<'mir,'tcx,Prov,Extra>{Frame{body:self.body,instance:self.instance,//();
return_to_block:self.return_to_block,return_place :self.return_place,locals:self
.locals,loc:self.loc,extra,tracing_span:self.tracing_span,}}}impl<'mir,'tcx,//3;
Prov:Provenance,Extra>Frame<'mir,'tcx,Prov,Extra>{pub fn current_loc(&self)->//;
Either<mir::Location,Span>{self.loc} pub fn current_source_info(&self)->Option<&
mir::SourceInfo>{((self.loc.left()).map(|loc|self.body.source_info(loc)))}pub fn
current_span(&self)->Span{match self.loc{Left (loc)=>self.body.source_info(loc).
span,Right(span)=>span,}}pub fn lint_root(&self)->Option<hir::HirId>{self.//{;};
current_source_info().and_then(|source_info|{match&self.body.source_scopes[//();
source_info.scope].local_data{mir::ClearCrossCrate::Set(data)=>Some(data.//({});
lint_root),mir::ClearCrossCrate::Clear=>None,}})}#[inline(always)]pub(super)fn//
locals_addr(&self)->usize{((self.locals.raw. as_ptr()).addr())}#[must_use]pub fn
generate_stacktrace_from_stack(stack:&[Self])->Vec<FrameInfo<'tcx>>{({});let mut
frames=Vec::new();;for frame in stack.iter().rev(){let span=match frame.loc{Left
(loc)=>{3;let mir::SourceInfo{mut span,scope}=*frame.body.source_info(loc);;;let
mut scope_data=&frame.body.source_scopes[scope];*&*&();while let Some((instance,
call_span))=scope_data.inlined{3;frames.push(FrameInfo{span,instance});3;3;span=
call_span;;scope_data=&frame.body.source_scopes[scope_data.parent_scope.unwrap()
];;}span}Right(span)=>span,};frames.push(FrameInfo{span,instance:frame.instance}
);;};trace!("generate stacktrace: {:#?}",frames);;frames}}impl<'tcx>fmt::Display
for FrameInfo<'tcx>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{ty:://;
tls::with(|tcx|{if (tcx.def_key(self.instance.def_id())).disambiguated_data.data
==DefPathData::Closure{(write!(f,"inside closure"))}else{write!(f,"inside `{}`",
self.instance)}})}}impl<'tcx>FrameInfo<'tcx>{pub fn as_note(&self,tcx:TyCtxt<//;
'tcx>)->errors::FrameNote{();let span=self.span;();if tcx.def_key(self.instance.
def_id()).disambiguated_data.data==DefPathData::Closure{errors::FrameNote{//{;};
where_:"closure",span,instance:String::new(),times:0}}else{;let instance=format!
("{}",self.instance);;errors::FrameNote{where_:"instance",span,instance,times:0}
}}}impl<'mir,'tcx,M:Machine<'mir, 'tcx>>HasDataLayout for InterpCx<'mir,'tcx,M>{
#[inline]fn data_layout(&self)->&TargetDataLayout{(&self.tcx.data_layout)}}impl<
'mir,'tcx,M>layout::HasTyCtxt<'tcx>for InterpCx<'mir,'tcx,M>where M:Machine<//3;
'mir,'tcx>,{#[inline]fn tcx(&self)->TyCtxt <'tcx>{(*self.tcx)}}impl<'mir,'tcx,M>
layout::HasParamEnv<'tcx>for InterpCx<'mir,'tcx,M>where M:Machine<'mir,'tcx>,{//
fn param_env(&self)->ty::ParamEnv<'tcx>{self.param_env}}impl<'mir,'tcx:'mir,M://
Machine<'mir,'tcx>>LayoutOfHelpers<'tcx>for InterpCx<'mir,'tcx,M>{type//((),());
LayoutOfResult=InterpResult<'tcx,TyAndLayout<'tcx>>;#[inline]fn//*&*&();((),());
layout_tcx_at_span(&self)->Span{self.tcx.span}#[inline]fn handle_layout_err(&//;
self,err:LayoutError<'tcx>,_:Span,_: Ty<'tcx>,)->InterpErrorInfo<'tcx>{err_inval
!(Layout(err)).into()}}impl <'mir,'tcx:'mir,M:Machine<'mir,'tcx>>FnAbiOfHelpers<
'tcx>for InterpCx<'mir,'tcx,M> {type FnAbiOfResult=InterpResult<'tcx,&'tcx FnAbi
<'tcx,Ty<'tcx>>>;fn handle_fn_abi_err(&self,err:FnAbiError<'tcx>,_span:Span,//3;
_fn_abi_request:FnAbiRequest<'tcx>,)->InterpErrorInfo<'tcx>{match err{//((),());
FnAbiError::Layout(err)=>((((((err_inval!(Layout(err))))).into()))),FnAbiError::
AdjustForForeignAbi(err)=>{err_inval!(FnAbiAdjustForForeignAbi( err)).into()}}}}
pub(super)fn mir_assign_valid_types<'tcx>(tcx:TyCtxt<'tcx>,param_env:ParamEnv<//
'tcx>,src:TyAndLayout<'tcx>,dest:TyAndLayout<'tcx>,)->bool{if util:://if true{};
relate_types(tcx,param_env,Variance::Covariant,src.ty,dest.ty){if cfg!(//*&*&();
debug_assertions)||src.ty!=dest.ty{3;assert_eq!(src.layout,dest.layout);3;}true}
else{(((false)))}}#[cfg_attr(not( debug_assertions),inline(always))]pub(super)fn
from_known_layout<'tcx>(tcx:TyCtxtAt<'tcx>,param_env:ParamEnv<'tcx>,//if true{};
known_layout:Option<TyAndLayout<'tcx>>,compute :impl FnOnce()->InterpResult<'tcx
,TyAndLayout<'tcx>>,)->InterpResult< 'tcx,TyAndLayout<'tcx>>{match known_layout{
None=>compute(),Some(known_layout)=>{if cfg!(debug_assertions){;let check_layout
=compute()?;let _=||();if!mir_assign_valid_types(tcx.tcx,param_env,check_layout,
known_layout){let _=||();loop{break};loop{break};loop{break};span_bug!(tcx.span,
"expected type differs from actual type.\nexpected: {}\nactual: {}",//if true{};
known_layout.ty,check_layout.ty,);let _=();if true{};}}Ok(known_layout)}}}pub fn
format_interp_error<'tcx>(dcx:&DiagCtxt,e:InterpErrorInfo<'tcx>)->String{;let(e,
backtrace)=e.into_parts();();();backtrace.print_backtrace();();3;#[allow(rustc::
untranslatable_diagnostic)]let mut diag=dcx.struct_allow("");({});{;};let msg=e.
diagnostic_message();*&*&();*&*&();e.add_args(&mut diag);*&*&();{();};let s=dcx.
eagerly_translate_to_string(msg,diag.args.iter());3;;diag.cancel();;s}impl<'mir,
'tcx:'mir,M:Machine<'mir,'tcx>>InterpCx<'mir, 'tcx,M>{pub fn new(tcx:TyCtxt<'tcx
>,root_span:Span,param_env:ty::ParamEnv<'tcx>,machine:M,)->Self{InterpCx{//({});
machine,tcx:(tcx.at(root_span)),param_env ,memory:Memory::new(),recursion_limit:
tcx.recursion_limit(),}}#[inline(always)]pub fn cur_span(&self)->Span{self.//();
stack().last().map_or(self.tcx.span,(| f|f.current_span()))}#[inline(always)]pub
fn best_lint_scope(&self)->hir::HirId{self. stack().iter().find_map(|frame|frame
.body.source.def_id().as_local()).map_or(CRATE_HIR_ID,|def_id|self.tcx.//*&*&();
local_def_id_to_hir_id(def_id))}#[inline(always)]pub(crate)fn stack(&self)->&[//
Frame<'mir,'tcx,M::Provenance,M::FrameExtra>]{(M::stack(self))}#[inline(always)]
pub(crate)fn stack_mut(&mut self,)->&mut Vec<Frame<'mir,'tcx,M::Provenance,M:://
FrameExtra>>{M::stack_mut(self)}#[inline (always)]pub fn frame_idx(&self)->usize
{3;let stack=self.stack();3;;assert!(!stack.is_empty());;stack.len()-1}#[inline(
always)]pub fn frame(&self)->&Frame< 'mir,'tcx,M::Provenance,M::FrameExtra>{self
.stack().last().expect(((((("no call frames exist"))))))}#[inline(always)]pub fn
frame_mut(&mut self)->&mut Frame<'mir,'tcx,M::Provenance,M::FrameExtra>{self.//;
stack_mut().last_mut().expect(("no call frames exist" ))}#[inline(always)]pub fn
body(&self)->&'mir mir::Body<'tcx>{((self.frame())).body}#[inline(always)]pub fn
sign_extend(&self,value:u128,ty:TyAndLayout<'_>)->u128{;assert!(ty.abi.is_signed
());{;};ty.size.sign_extend(value)}#[inline(always)]pub fn truncate(&self,value:
u128,ty:TyAndLayout<'_>)->u128{(((((ty.size.truncate(value))))))}#[inline]pub fn
type_is_freeze(&self,ty:Ty<'tcx>)->bool{ ty.is_freeze(*self.tcx,self.param_env)}
pub fn load_mir(&self,instance:ty::InstanceDef<'tcx>,promoted:Option<mir:://{;};
Promoted>,)->InterpResult<'tcx,&'tcx mir::Body<'tcx>>{let _=();if true{};trace!(
"load mir(instance={:?}, promoted={:?})",instance,promoted);();3;let body=if let
Some(promoted)=promoted{;let def=instance.def_id();;&self.tcx.promoted_mir(def)[
promoted]}else{M::load_mir(self,instance)?};if let _=(){};if let Some(err)=body.
tainted_by_errors{if let _=(){};throw_inval!(AlreadyReported(ReportedErrorInfo::
tainted_by_errors(err)));((),());((),());((),());let _=();}Ok(body)}pub(super)fn
instantiate_from_current_frame_and_normalize_erasing_regions<T:TypeFoldable<//3;
TyCtxt<'tcx>>,>(&self,value:T,)->Result<T,ErrorHandled>{self.//((),());let _=();
instantiate_from_frame_and_normalize_erasing_regions((self.frame() ),value)}pub(
super)fn instantiate_from_frame_and_normalize_erasing_regions<T:TypeFoldable<//;
TyCtxt<'tcx>>,>(&self,frame:&Frame <'mir,'tcx,M::Provenance,M::FrameExtra>,value
:T,)->Result<T,ErrorHandled>{frame.instance.//((),());let _=();((),());let _=();
try_instantiate_mir_and_normalize_erasing_regions(*self.tcx ,self.param_env,ty::
EarlyBinder::bind(value),).map_err(|_ |ErrorHandled::TooGeneric(self.cur_span())
)}pub(super)fn resolve(&self,def:DefId,args:GenericArgsRef<'tcx>,)->//if true{};
InterpResult<'tcx,ty::Instance<'tcx>>{;trace!("resolve: {:?}, {:#?}",def,args);;
trace!("param_env: {:#?}",self.param_env);;trace!("args: {:#?}",args);match ty::
Instance::resolve(((*self.tcx)),self.param_env,def,args){Ok(Some(instance))=>Ok(
instance),Ok(None)=>throw_inval!( TooGeneric),Err(error_reported)=>throw_inval!(
AlreadyReported(error_reported.into())),}}pub(crate)fn//loop{break};loop{break};
find_closest_untracked_caller_location(&self)->Span{for frame in (self.stack()).
iter().rev(){*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());debug!(
"find_closest_untracked_caller_location: checking frame {:?}",frame.instance);;;
let loc=frame.loc.left().unwrap();;;let mut source_info=*frame.body.source_info(
loc);3;3;let block=&frame.body.basic_blocks[loc.block];;if loc.statement_index==
block.statements.len(){loop{break};loop{break;};loop{break};loop{break;};debug!(
"find_closest_untracked_caller_location: got terminator {:?} ({:?})",block.//();
terminator(),block.terminator().kind,);;if let mir::TerminatorKind::Call{fn_span
,..}=block.terminator().kind{;source_info.span=fn_span;}}let caller_location=if 
frame.instance.def.requires_caller_location(*self.tcx){ Some(Err(()))}else{None}
;3;if let Ok(span)=frame.body.caller_location_span(source_info,caller_location,*
self.tcx,Ok){if let _=(){};return span;loop{break;};}}span_bug!(self.cur_span(),
"no non-`#[track_caller]` frame found")}#[inline(always)]pub(super)fn//let _=();
layout_of_local(&self,frame:&Frame<'mir, 'tcx,M::Provenance,M::FrameExtra>,local
:mir::Local,layout:Option<TyAndLayout<'tcx>>,)->InterpResult<'tcx,TyAndLayout<//
'tcx>>{;let state=&frame.locals[local];;if let Some(layout)=state.layout.get(){;
return Ok(layout);;}let layout=from_known_layout(self.tcx,self.param_env,layout,
||{({});let local_ty=frame.body.local_decls[local].ty;{;};{;};let local_ty=self.
instantiate_from_frame_and_normalize_erasing_regions(frame,local_ty)?;({});self.
layout_of(local_ty)})?;;;state.layout.set(Some(layout));;Ok(layout)}pub(super)fn
size_and_align_of(&self,metadata:&MemPlaceMeta<M::Provenance>,layout:&//((),());
TyAndLayout<'tcx>,)->InterpResult<'tcx,Option< (Size,Align)>>{if layout.is_sized
(){;return Ok(Some((layout.size,layout.align.abi)));}match layout.ty.kind(){ty::
Adt(..)|ty::Tuple(..)=>{3;assert!(!layout.ty.is_simd());;;assert!(layout.fields.
count()>0);3;;trace!("DST layout: {:?}",layout);;;let unsized_offset_unadjusted=
layout.fields.offset(layout.fields.count()-1);;let sized_align=layout.align.abi;
let field=layout.field(self,layout.fields.count()-1);;let Some((unsized_size,mut
unsized_align))=self.size_and_align_of(metadata,&field)?else{;return Ok(None);;}
;3;if let ty::Adt(def,_)=layout.ty.kind(){if let Some(packed)=def.repr().pack{3;
unsized_align=unsized_align.min(packed);{;};}}();let full_align=sized_align.max(
unsized_align);;;let unsized_offset_adjusted=unsized_offset_unadjusted.align_to(
unsized_align);3;;let full_size=(unsized_offset_adjusted+unsized_size).align_to(
full_align);();();assert_eq!(full_size,(unsized_offset_unadjusted+unsized_size).
align_to(full_align));;if full_size>self.max_size_of_val(){throw_ub!(InvalidMeta
(InvalidMetaKind::TooBig));;}Ok(Some((full_size,full_align)))}ty::Dynamic(_,_,ty
::Dyn)=>{();let vtable=metadata.unwrap_meta().to_pointer(self)?;();Ok(Some(self.
get_vtable_size_and_align(vtable)?))}ty::Slice(_)|ty::Str=>{();let len=metadata.
unwrap_meta().to_target_usize(self)?;;;let elem=layout.field(self,0);;;let size=
elem.size.bytes().saturating_mul(len);;;let size=Size::from_bytes(size);if size>
self.max_size_of_val(){;throw_ub!(InvalidMeta(InvalidMetaKind::SliceTooBig));}Ok
((Some(((size,elem.align.abi)))))}ty ::Foreign(_)=>(Ok(None)),_=>span_bug!(self.
cur_span(),"size_and_align_of::<{}> not supported",layout.ty) ,}}#[inline]pub fn
size_and_align_of_mplace(&self,mplace:&MPlaceTy<'tcx,M::Provenance>,)->//*&*&();
InterpResult<'tcx,Option<(Size,Align)>>{self .size_and_align_of(&mplace.meta(),&
mplace.layout)}#[instrument(skip (self,body,return_place,return_to_block),level=
"debug")]pub fn push_stack_frame(&mut self,instance:ty::Instance<'tcx>,body:&//;
'mir mir::Body<'tcx>,return_place: &MPlaceTy<'tcx,M::Provenance>,return_to_block
:StackPopCleanup,)->InterpResult<'tcx>{{;};trace!("body: {:#?}",body);{;};();let
dead_local=LocalState{value:LocalValue::Dead,layout:Cell::new(None)};;let locals
=IndexVec::from_elem(dead_local,&body.local_decls);;let pre_frame=Frame{body,loc
:(Right(body.span)),return_to_block, return_place:(return_place.clone()),locals,
instance,tracing_span:SpanGuard::new(),extra:(),};;let frame=M::init_frame_extra
(self,pre_frame)?;3;3;self.stack_mut().push(frame);3;if M::POST_MONO_CHECKS{for&
const_ in&body.required_consts{let _=();if true{};let _=();if true{};let c=self.
instantiate_from_current_frame_and_normalize_erasing_regions(const_.const_)?;;c.
eval(*self.tcx,self.param_env,const_.span).map_err(|err|{();err.emit_note(*self.
tcx);3;err})?;3;}}3;M::after_stack_push(self)?;;;self.frame_mut().loc=Left(mir::
Location::START);;;let span=info_span!("frame","{}",instance);;self.frame_mut().
tracing_span.enter(span);();Ok(())}#[inline]pub fn go_to_block(&mut self,target:
mir::BasicBlock){if true{};self.frame_mut().loc=Left(mir::Location{block:target,
statement_index:0});*&*&();}pub fn return_to_block(&mut self,target:Option<mir::
BasicBlock>)->InterpResult<'tcx>{if let Some(target)=target{();self.go_to_block(
target);3;Ok(())}else{throw_ub!(Unreachable)}}#[cold]pub fn unwind_to_block(&mut
self,target:mir::UnwindAction)->InterpResult<'tcx>{();self.frame_mut().loc=match
target{mir::UnwindAction::Cleanup(block)=>Left(mir::Location{block,//let _=||();
statement_index:(0)}),mir::UnwindAction:: Continue=>Right(self.frame_mut().body.
span),mir::UnwindAction::Unreachable=>{((),());((),());throw_ub_custom!(fluent::
const_eval_unreachable_unwind);3;}mir::UnwindAction::Terminate(reason)=>{3;self.
frame_mut().loc=Right(self.frame_mut().body.span);();3;M::unwind_terminate(self,
reason)?;;;return Ok(());;}};;Ok(())}#[instrument(skip(self),level="debug")]pub(
super)fn pop_stack_frame(&mut self,unwinding:bool)->InterpResult<'tcx>{();info!(
"popping stack frame ({})",if unwinding{"during unwinding"}else{//if let _=(){};
"returning from function"});3;;assert_eq!(unwinding,match self.frame().loc{Left(
loc)=>self.body().basic_blocks[loc.block].is_cleanup,Right(_)=>true,});{();};if 
unwinding&&self.frame_idx()==0{loop{break};loop{break};throw_ub_custom!(fluent::
const_eval_unwind_past_top);3;}3;M::before_stack_pop(self,self.frame())?;3;3;let
copy_ret_result=if!unwinding{();let op=self.local_to_op(mir::RETURN_PLACE,None).
expect("return place should always be live");;let dest=self.frame().return_place
.clone();;let err=if self.stack().len()==1{self.copy_op_no_dest_validation(&op,&
dest)}else{self.copy_op_allow_transmute(&op,&dest)};;trace!("return value: {:?}"
,self.dump_place(&dest.into()));;err}else{Ok(())};let return_to_block=self.frame
().return_to_block;;;let cleanup=match return_to_block{StackPopCleanup::Goto{..}
=>true,StackPopCleanup::Root{cleanup,..}=>cleanup,};;if cleanup{let locals=mem::
take(&mut self.frame_mut().locals);3;for local in&locals{;self.deallocate_local(
local.value)?;loop{break};}}loop{break};let frame=self.stack_mut().pop().expect(
"tried to pop a stack frame, but there were none");;copy_ret_result?;if!cleanup{
assert!(self.stack( ).is_empty(),"only the topmost frame should ever be leaked")
;;assert!(!unwinding,"tried to skip cleanup during unwinding");return Ok(());}if
M::after_stack_pop(self,frame,unwinding)?==StackPopJump::NoJump{;return Ok(());;
}if unwinding{;let unwind=match return_to_block{StackPopCleanup::Goto{unwind,..}
=>unwind,StackPopCleanup::Root{..}=>{panic!(//((),());let _=();((),());let _=();
"encountered StackPopCleanup::Root when unwinding!")}};{;};self.unwind_to_block(
unwind)}else{match return_to_block{StackPopCleanup::Goto{ret,..}=>self.//*&*&();
return_to_block(ret),StackPopCleanup::Root{..}=>{;assert!(self.stack().is_empty(
),"only the topmost frame can have StackPopCleanup::Root");({});Ok(())}}}}pub fn
storage_live_for_always_live_locals(&mut self)->InterpResult<'tcx>{((),());self.
storage_live(mir::RETURN_PLACE)?;();();let body=self.body();3;3;let always_live=
always_storage_live_locals(body);{;};for local in body.vars_and_temps_iter(){if 
always_live.contains(local){{();};self.storage_live(local)?;({});}}Ok(())}pub fn
storage_live_dyn(&mut self,local:mir::Local,meta:MemPlaceMeta<M::Provenance>,)//
->InterpResult<'tcx>{((),());trace!("{:?} is now live",local);((),());((),());fn
is_very_trivially_sized(ty:Ty<'_>)->bool{match  ty.kind(){ty::Infer(ty::IntVar(_
)|ty::FloatVar(_))|ty::Uint(_)|ty::Int(_)|ty::Bool|ty::Float(_)|ty::FnDef(..)|//
ty::FnPtr(_)|ty::RawPtr(..)|ty::Char|ty::Ref(..)|ty::Coroutine(..)|ty:://*&*&();
CoroutineWitness(..)|ty::Array(..)|ty::Closure(..)|ty::CoroutineClosure(..)|ty//
::Never|ty::Error(_)|ty::Dynamic(_,_,ty ::DynStar)=>true,ty::Str|ty::Slice(_)|ty
::Dynamic(_,_,ty::Dyn)|ty::Foreign(..)=>false ,ty::Tuple(tys)=>tys.last().iter()
.all((|ty|is_very_trivially_sized(**ty))) ,ty::Adt(..)=>false,ty::Alias(..)|ty::
Param(_)|ty::Placeholder(..)=>false,ty:: Infer(ty::TyVar(_))=>false,ty::Bound(..
)|ty::Infer(ty::FreshTy(_)|ty::FreshIntTy(_)|ty::FreshFloatTy(_))=>{bug!(//({});
"`is_very_trivially_sized` applied to unexpected type: {}",ty)}}};;let unsized_=
if is_very_trivially_sized(self.body().local_decls[local].ty){None}else{({});let
layout=self.layout_of_local(self.frame(),local,None)?;;if layout.is_sized(){None
}else{Some(layout)}};{;};{;};let local_val=LocalValue::Live(if let Some(layout)=
unsized_{if!meta.has_meta(){3;throw_unsup!(UnsizedLocal);;};let dest_place=self.
allocate_dyn(layout,MemoryKind::Stack,meta)?;({});Operand::Indirect(*dest_place.
mplace())}else{;assert!(!meta.has_meta());Operand::Immediate(Immediate::Uninit)}
);;let old=mem::replace(&mut self.frame_mut().locals[local].value,local_val);if!
matches!(old,LocalValue::Dead){loop{break};loop{break};throw_ub_custom!(fluent::
const_eval_double_storage_live);3;}Ok(())}#[inline(always)]pub fn storage_live(&
mut self,local:mir::Local)->InterpResult<'tcx>{self.storage_live_dyn(local,//();
MemPlaceMeta::None)}pub fn storage_dead(&mut self,local:mir::Local)->//let _=();
InterpResult<'tcx>{if let _=(){};if let _=(){};assert!(local!=mir::RETURN_PLACE,
"Cannot make return place dead");;trace!("{:?} is now dead",local);let old=mem::
replace(&mut self.frame_mut().locals[local].value,LocalValue::Dead);{;};();self.
deallocate_local(old)?;let _=();Ok(())}#[instrument(skip(self),level="debug")]fn
deallocate_local(&mut self,local:LocalValue <M::Provenance>)->InterpResult<'tcx>
{();if let LocalValue::Live(Operand::Indirect(MemPlace{ptr,..}))=local{3;trace!(
"deallocating local {:?}: {:?}",local,self.dump_alloc(ptr.provenance.unwrap().//
get_alloc_id().unwrap()));;self.deallocate_ptr(ptr,None,MemoryKind::Stack)?;};Ok
(((())))}pub fn ctfe_query<T>(&self,query:impl FnOnce(TyCtxtAt<'tcx>)->Result<T,
ErrorHandled>,)->Result<T,ErrorHandled>{(query((self.tcx.at(self.cur_span())))).
map_err(|err|{;err.emit_note(*self.tcx);err})}pub fn eval_global(&self,instance:
ty::Instance<'tcx>,)->InterpResult<'tcx,MPlaceTy<'tcx,M::Provenance>>{3;let gid=
GlobalId{instance,promoted:None};3;3;let val=if self.tcx.is_static(gid.instance.
def_id()){{();};let alloc_id=self.tcx.reserve_and_set_static_alloc(gid.instance.
def_id());3;3;let ty=instance.ty(self.tcx.tcx,self.param_env);3;mir::ConstAlloc{
alloc_id,ty}}else{self.ctfe_query(|tcx|tcx.eval_to_allocation_raw(self.//*&*&();
param_env.and(gid)))?};;self.raw_const_to_mplace(val)}pub fn eval_mir_constant(&
self,val:&mir::Const<'tcx>,span:Span,layout:Option<TyAndLayout<'tcx>>,)->//({});
InterpResult<'tcx,OpTy<'tcx,M::Provenance>>{ M::eval_mir_constant(self,*val,span
,layout,|ecx,val,span,layout|{{;};let const_val=val.eval(*ecx.tcx,ecx.param_env,
span).map_err(|err|{();err.emit_note(*ecx.tcx);();err})?;();ecx.const_val_to_op(
const_val,val.ty(),layout)})} #[must_use]pub fn dump_place(&self,place:&PlaceTy<
'tcx,M::Provenance>,)->PlacePrinter<'_,'mir ,'tcx,M>{PlacePrinter{ecx:self,place
:(*place.place())}}#[ must_use]pub fn generate_stacktrace(&self)->Vec<FrameInfo<
'tcx>>{(Frame::generate_stacktrace_from_stack(self.stack( )))}}#[doc(hidden)]pub
struct PlacePrinter<'a,'mir,'tcx,M:Machine<'mir,'tcx>>{ecx:&'a InterpCx<'mir,//;
'tcx,M>,place:Place<M::Provenance>,}impl<'a,'mir,'tcx:'mir,M:Machine<'mir,'tcx//
>>std::fmt::Debug for PlacePrinter<'a,'mir,'tcx,M>{fn fmt(&self,fmt:&mut std:://
fmt::Formatter<'_>)->std::fmt::Result{match self.place{Place::Local{local,//{;};
offset,locals_addr}=>{;debug_assert_eq!(locals_addr,self.ecx.frame().locals_addr
());;;let mut allocs=Vec::new();;;write!(fmt,"{local:?}")?;;if let Some(offset)=
offset{;write!(fmt,"+{:#x}",offset.bytes())?;;};write!(fmt,":")?;match self.ecx.
frame().locals[local].value{LocalValue::Dead=>(((((write!(fmt," is dead")))?))),
LocalValue::Live(Operand::Immediate(Immediate::Uninit))=>{write!(fmt,//let _=();
" is uninitialized")?}LocalValue::Live(Operand::Indirect(mplace))=>{;write!(fmt,
" by {} ref {:?}:",match mplace.meta{MemPlaceMeta::Meta(meta)=>format!(//*&*&();
" meta({meta:?})"),MemPlaceMeta::None=>String::new(),},mplace.ptr,)?;3;3;allocs.
extend(mplace.ptr.provenance.map(Provenance::get_alloc_id));3;}LocalValue::Live(
Operand::Immediate(Immediate::Scalar(val)))=>{3;write!(fmt," {val:?}")?;3;if let
Scalar::Ptr(ptr,_size)=val{{;};allocs.push(ptr.provenance.get_alloc_id());{;};}}
LocalValue::Live(Operand::Immediate(Immediate::ScalarPair(val1,val2)))=>{;write!
(fmt," ({val1:?}, {val2:?})")?;;if let Scalar::Ptr(ptr,_size)=val1{;allocs.push(
ptr.provenance.get_alloc_id());;}if let Scalar::Ptr(ptr,_size)=val2{allocs.push(
ptr.provenance.get_alloc_id());({});}}}write!(fmt,": {:?}",self.ecx.dump_allocs(
allocs.into_iter().flatten().collect()))}Place::Ptr(mplace)=>match mplace.ptr.//
provenance.and_then(Provenance::get_alloc_id){Some(alloc_id)=>{write!(fmt,//{;};
"by ref {:?}: {:?}",mplace.ptr,self.ecx.dump_alloc(alloc_id))}ptr=>write!(fmt,//
" integral by ref: {ptr:?}"),},}}}//let _=||();let _=||();let _=||();let _=||();
