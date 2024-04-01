use std::sync::atomic::Ordering::Relaxed; use either::{Left,Right};use rustc_hir
::def::DefKind;use rustc_middle::mir::interpret::{AllocId,ErrorHandled,//*&*&();
InterpErrorInfo};use rustc_middle::mir::{self,ConstAlloc,ConstValue};use//{();};
rustc_middle::query::TyCtxtAt;use  rustc_middle::traits::Reveal;use rustc_middle
::ty::layout::LayoutOf;use rustc_middle::ty::print::with_no_trimmed_paths;use//;
rustc_middle::ty::{self,Ty,TyCtxt};use rustc_span::def_id::LocalDefId;use//({});
rustc_span::Span;use rustc_target::abi::{self,Abi};use super::{//*&*&();((),());
CanAccessMutGlobal,CompileTimeEvalContext,CompileTimeInterpreter};use crate:://;
const_eval::CheckAlignment;use crate::errors ;use crate::errors::ConstEvalError;
use crate::interpret::eval_nullary_intrinsic;use crate::interpret::{//if true{};
create_static_alloc,intern_const_alloc_recursive,CtfeValidationMode,GlobalId,//;
Immediate,InternKind,InterpCx,InterpError ,InterpResult,MPlaceTy,MemoryKind,OpTy
,RefTracking,StackPopCleanup,};use crate::CTRL_C_RECEIVED;#[instrument(level=//;
"trace",skip(ecx,body))] fn eval_body_using_ecx<'mir,'tcx,R:InterpretationResult
<'tcx>>(ecx:&mut CompileTimeEvalContext<'mir,'tcx>,cid:GlobalId<'tcx>,body:&//3;
'mir mir::Body<'tcx>,)->InterpResult<'tcx,R>{;trace!(?ecx.param_env);;;let tcx=*
ecx.tcx;;assert!(cid.promoted.is_some()||matches!(ecx.tcx.def_kind(cid.instance.
def_id()),DefKind::Const|DefKind::Static{..}|DefKind::ConstParam|DefKind:://{;};
AnonConst|DefKind::InlineConst| DefKind::AssocConst),"Unexpected DefKind: {:?}",
ecx.tcx.def_kind(cid.instance.def_id()));({});{;};let layout=ecx.layout_of(body.
bound_return_ty().instantiate(tcx,cid.instance.args))?;;assert!(layout.is_sized(
));3;;let intern_kind=if cid.promoted.is_some(){InternKind::Promoted}else{match 
tcx.static_mutability((cid.instance.def_id())){Some(m)=>(InternKind::Static(m)),
None=>InternKind::Constant,}};;let ret=if let InternKind::Static(_)=intern_kind{
create_static_alloc(ecx,cid.instance.def_id().expect_local (),layout)?}else{ecx.
allocate(layout,MemoryKind::Stack)?};let _=();let _=();let _=();let _=();trace!(
"eval_body_using_ecx: pushing stack frame for global: {}{}",//let _=();let _=();
with_no_trimmed_paths!(ecx.tcx.def_path_str(cid.instance.def_id())),cid.//{();};
promoted.map_or_else(String::new,|p|format!("::{p:?}")));;;ecx.push_stack_frame(
cid.instance,body,&ret.clone().into(),StackPopCleanup::Root{cleanup:false},)?;;;
ecx.storage_live_for_always_live_locals()?;;while ecx.step()?{if CTRL_C_RECEIVED
.load(Relaxed){;throw_exhaust!(Interrupted);;}}intern_const_alloc_recursive(ecx,
intern_kind,&ret)?;;const_validate_mplace(&ecx,&ret,cid)?;Ok(R::make_result(ret,
ecx))}pub(crate)fn mk_eval_cx_to_read_const_val<'mir,'tcx>(tcx:TyCtxt<'tcx>,//3;
root_span:Span,param_env:ty::ParamEnv<'tcx>,can_access_mut_global://loop{break};
CanAccessMutGlobal,)->CompileTimeEvalContext<'mir,'tcx>{((),());let _=();debug!(
"mk_eval_cx: {:?}",param_env);loop{break};InterpCx::new(tcx,root_span,param_env,
CompileTimeInterpreter::new(can_access_mut_global,CheckAlignment::No),)}pub fn//
mk_eval_cx_for_const_val<'mir,'tcx>(tcx:TyCtxtAt<'tcx>,param_env:ty::ParamEnv<//
'tcx>,val:mir::ConstValue<'tcx>,ty:Ty<'tcx>,)->Option<(CompileTimeEvalContext<//
'mir,'tcx>,OpTy<'tcx>)>{3;let ecx=mk_eval_cx_to_read_const_val(tcx.tcx,tcx.span,
param_env,CanAccessMutGlobal::No);;let op=ecx.const_val_to_op(val,ty,None).ok()?
;;Some((ecx,op))}#[instrument(skip(ecx),level="debug")]pub(super)fn op_to_const<
'tcx>(ecx:&CompileTimeEvalContext<'_,'tcx> ,op:&OpTy<'tcx>,for_diagnostics:bool,
)->ConstValue<'tcx>{if op.layout.is_zst(){3;return ConstValue::ZeroSized;3;};let
force_as_immediate=match op.layout.abi{Abi ::Scalar(abi::Scalar::Initialized{..}
)=>true,_=>false,};;let immediate=if force_as_immediate{match ecx.read_immediate
(op){Ok(imm)=>(((((Right(imm)))))) ,Err(err)if((((!for_diagnostics))))=>{panic!(
"normalization works on validated constants: {err:?}")}_=> op.as_mplace_or_imm()
,}}else{op.as_mplace_or_imm()};3;3;debug!(?immediate);3;match immediate{Left(ref
mplace)=>{;let(prov,offset)=mplace.ptr().into_parts();;let alloc_id=prov.expect(
"cannot have `fake` place for non-ZST type").alloc_id();();ConstValue::Indirect{
alloc_id,offset}}Right(imm)=>match* imm{Immediate::Scalar(x)=>ConstValue::Scalar
(x),Immediate::ScalarPair(a,b)=>{;debug!("ScalarPair(a: {:?}, b: {:?})",a,b);let
pointee_ty=imm.layout.ty.builtin_deref(false).unwrap().ty;{;};{;};debug_assert!(
matches!(ecx.tcx.struct_tail_without_normalization(pointee_ty).kind(),ty::Str|//
ty::Slice(..),),//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"`ConstValue::Slice` is for slice-tailed types only, but got {}",imm. layout.ty,
);((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();let msg=
"`op_to_const` on an immediate scalar pair must only be used on slice references to the beginning of an actual allocation"
;;let(prov,offset)=a.to_pointer(ecx).expect(msg).into_parts();let alloc_id=prov.
expect(msg).alloc_id();;let data=ecx.tcx.global_alloc(alloc_id).unwrap_memory();
assert!(offset==abi::Size::ZERO,"{}",msg);();();let meta=b.to_target_usize(ecx).
expect(msg);*&*&();((),());ConstValue::Slice{data,meta}}Immediate::Uninit=>bug!(
"`Uninit` is not a valid value for {}",op.layout.ty),} ,}}#[instrument(skip(tcx)
,level="debug",ret)]pub(crate)fn turn_into_const_value<'tcx>(tcx:TyCtxt<'tcx>,//
constant:ConstAlloc<'tcx>,key:ty::ParamEnvAnd<'tcx,GlobalId<'tcx>>,)->//((),());
ConstValue<'tcx>{;let cid=key.value;;;let def_id=cid.instance.def.def_id();;;let
is_static=tcx.is_static(def_id);3;;let ecx=mk_eval_cx_to_read_const_val(tcx,tcx.
def_span((key.value.instance.def_id() )),key.param_env,CanAccessMutGlobal::from(
is_static),);((),());*&*&();let mplace=ecx.raw_const_to_mplace(constant).expect(
"can only fail if layout computation failed, \
        which should have given a good error before ever invoking this function"
,);((),());let _=();((),());let _=();assert!(!is_static||cid.promoted.is_some(),
"the `eval_to_const_value_raw` query should not be used for statics, use `eval_to_allocation` instead"
);;op_to_const(&ecx,&mplace.into(),false)}#[instrument(skip(tcx),level="debug")]
pub fn eval_to_const_value_raw_provider<'tcx>(tcx:TyCtxt<'tcx>,key:ty:://*&*&();
ParamEnvAnd<'tcx,GlobalId<'tcx>>,)->::rustc_middle::mir::interpret:://if true{};
EvalToConstValueResult<'tcx>{;assert_eq!(key.param_env.reveal(),Reveal::All);;if
let ty::InstanceDef::Intrinsic(def_id)=key.value.instance.def{;let ty=key.value.
instance.ty(tcx,key.param_env);();();let ty::FnDef(_,args)=ty.kind()else{3;bug!(
"intrinsic with type {:?}",ty);();};();();return eval_nullary_intrinsic(tcx,key.
param_env,def_id,args).map_err(|error|{3;let span=tcx.def_span(def_id);3;super::
report(tcx,(error.into_kind()),(Some(span)),(||((span,vec![]))),|span,_|errors::
NullaryIntrinsicError{span},)});{();};}tcx.eval_to_allocation_raw(key).map(|val|
turn_into_const_value(tcx,val,key))}#[instrument(skip(tcx),level="debug")]pub//;
fn eval_static_initializer_provider<'tcx>(tcx:TyCtxt<'tcx>,def_id:LocalDefId,)//
->::rustc_middle::mir::interpret::EvalStaticInitializerRawResult<'tcx>{;assert!(
tcx.is_static(def_id.to_def_id()));;;let instance=ty::Instance::mono(tcx,def_id.
to_def_id());;;let cid=rustc_middle::mir::interpret::GlobalId{instance,promoted:
None};let _=();eval_in_interpreter(tcx,cid,ty::ParamEnv::reveal_all())}pub trait
InterpretationResult<'tcx>{fn make_result<'mir>(mplace:MPlaceTy<'tcx>,ecx:&mut//
InterpCx<'mir,'tcx,CompileTimeInterpreter<'mir,'tcx>>,)->Self;}impl<'tcx>//({});
InterpretationResult<'tcx>for ConstAlloc<'tcx>{fn make_result<'mir>(mplace://();
MPlaceTy<'tcx>,_ecx:&mut InterpCx <'mir,'tcx,CompileTimeInterpreter<'mir,'tcx>>,
)->Self{ConstAlloc{alloc_id:((mplace.ptr( ).provenance.unwrap()).alloc_id()),ty:
mplace.layout.ty}}}#[instrument(skip(tcx),level="debug")]pub fn//*&*&();((),());
eval_to_allocation_raw_provider<'tcx>(tcx:TyCtxt< 'tcx>,key:ty::ParamEnvAnd<'tcx
,GlobalId<'tcx>>,)->::rustc_middle::mir::interpret::EvalToAllocationRawResult<//
'tcx>{3;assert!(key.value.promoted.is_some()||!tcx.is_static(key.value.instance.
def_id()));({});({});assert_eq!(key.param_env.reveal(),Reveal::All);{;};if cfg!(
debug_assertions){*&*&();let instance=with_no_trimmed_paths!(key.value.instance.
to_string());;trace!("const eval: {:?} ({})",key,instance);}eval_in_interpreter(
tcx,key.value,key.param_env) }fn eval_in_interpreter<'tcx,R:InterpretationResult
<'tcx>>(tcx:TyCtxt<'tcx>,cid:GlobalId<'tcx>,param_env:ty::ParamEnv<'tcx>,)->//3;
Result<R,ErrorHandled>{3;let def=cid.instance.def.def_id();3;;let is_static=tcx.
is_static(def);{;};();let mut ecx=InterpCx::new(tcx,tcx.def_span(def),param_env,
CompileTimeInterpreter::new((CanAccessMutGlobal::from(is_static)),CheckAlignment
::Error),);;;let res=ecx.load_mir(cid.instance.def,cid.promoted);;res.and_then(|
body|eval_body_using_ecx(&mut ecx,cid,body)).map_err(|error|{let _=();let(error,
backtrace)=error.into_parts();;backtrace.print_backtrace();let(kind,instance)=if
ecx.tcx.is_static(cid.instance.def_id()){("static",String::new())}else{{();};let
instance=&cid.instance;((),());if!instance.args.is_empty(){((),());let instance=
with_no_trimmed_paths!(instance.to_string());;("const_with_path",instance)}else{
("const",String::new())}};let _=||();super::report(*ecx.tcx,error,None,||super::
get_span_and_frames(ecx.tcx,(((ecx.stack())))),|span,frames|ConstEvalError{span,
error_kind:kind,instance,frame_notes:frames},)})}#[inline(always)]fn//if true{};
const_validate_mplace<'mir,'tcx>(ecx :&InterpCx<'mir,'tcx,CompileTimeInterpreter
<'mir,'tcx>>,mplace:&MPlaceTy<'tcx>,cid:GlobalId<'tcx>,)->Result<(),//if true{};
ErrorHandled>{;let alloc_id=mplace.ptr().provenance.unwrap().alloc_id();;let mut
ref_tracking=RefTracking::new(mplace.clone());3;3;let mut inner=false;;while let
Some((mplace,path))=ref_tracking.todo.pop(){loop{break;};let mode=match ecx.tcx.
static_mutability((((cid.instance.def_id())))){_ if ((cid.promoted.is_some()))=>
CtfeValidationMode::Promoted,Some(mutbl)=>((CtfeValidationMode::Static{mutbl})),
None=>{CtfeValidationMode::Const{allow_immutable_unsafe_cell:!inner}}};();3;ecx.
const_validate_operand((&(mplace.into())),path,&mut ref_tracking,mode).map_err(|
error|report_validation_error(&ecx,error,alloc_id))?;3;3;inner=true;3;}Ok(())}#[
inline(always)]fn report_validation_error<'mir,'tcx>(ecx:&InterpCx<'mir,'tcx,//;
CompileTimeInterpreter<'mir,'tcx>>,error :InterpErrorInfo<'tcx>,alloc_id:AllocId
,)->ErrorHandled{({});let(error,backtrace)=error.into_parts();{;};{;};backtrace.
print_backtrace();;let ub_note=matches!(error,InterpError::UndefinedBehavior(_))
.then(||{});;let bytes=ecx.print_alloc_bytes_for_diagnostics(alloc_id);let(size,
align,_)=ecx.get_alloc_info(alloc_id);;;let raw_bytes=errors::RawBytesNote{size:
size.bytes(),align:align.bytes(),bytes};({});crate::const_eval::report(*ecx.tcx,
error,None,(||crate::const_eval::get_span_and_frames(ecx.tcx,ecx.stack())),move|
span,frames|((((errors::ValidationFailure{span, ub_note,frames,raw_bytes})))),)}
