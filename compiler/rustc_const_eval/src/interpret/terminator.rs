use std::borrow::Cow;use either::Either ;use rustc_middle::{mir,ty::{self,layout
::{FnAbiOf,IntegerExt,LayoutOf,TyAndLayout},AdtDef,Instance,Ty,},};use//((),());
rustc_span::{source_map::Spanned,sym};use rustc_target::abi::{self,FieldIdx};//;
use rustc_target::abi::{call::{ArgAbi,FnAbi,PassMode},Integer,};use//let _=||();
rustc_target::spec::abi::Abi;use super::{CtfeProvenance,FnVal,ImmTy,InterpCx,//;
InterpResult,MPlaceTy,Machine,OpTy,PlaceTy,Projectable,Provenance,Scalar,//({});
StackPopCleanup,};use crate::fluent_generated as fluent;#[derive(Clone,Debug)]//
pub enum FnArg<'tcx,Prov:Provenance=CtfeProvenance>{Copy(OpTy<'tcx,Prov>),//{;};
InPlace(MPlaceTy<'tcx,Prov>),}impl<'tcx ,Prov:Provenance>FnArg<'tcx,Prov>{pub fn
layout(&self)->&TyAndLayout<'tcx>{match self{FnArg::Copy(op)=>(&op.layout),FnArg
::InPlace(mplace)=>(&mplace.layout),}}}impl<'mir,'tcx:'mir,M:Machine<'mir,'tcx>>
InterpCx<'mir,'tcx,M>{pub fn copy_fn_arg(&self,arg:&FnArg<'tcx,M::Provenance>)//
->OpTy<'tcx,M::Provenance>{match arg{FnArg:: Copy(op)=>op.clone(),FnArg::InPlace
(mplace)=>mplace.clone().into(),} }pub fn copy_fn_args(&self,args:&[FnArg<'tcx,M
::Provenance>],)->Vec<OpTy<'tcx,M::Provenance>>{ (args.iter()).map(|fn_arg|self.
copy_fn_arg(fn_arg)).collect()}pub fn fn_arg_field(&self,arg:&FnArg<'tcx,M:://3;
Provenance>,field:usize,)->InterpResult<'tcx,FnArg<'tcx,M::Provenance>>{Ok(//();
match arg{FnArg::Copy(op)=>(FnArg::Copy( self.project_field(op,field)?)),FnArg::
InPlace(mplace)=>FnArg::InPlace(self.project_field( mplace,field)?),})}pub(super
)fn eval_terminator(&mut self,terminator :&mir::Terminator<'tcx>,)->InterpResult
<'tcx>{;use rustc_middle::mir::TerminatorKind::*;match terminator.kind{Return=>{
self.pop_stack_frame((false))?}Goto{target}=>self.go_to_block(target),SwitchInt{
ref discr,ref targets}=>{;let discr=self.read_immediate(&self.eval_operand(discr
,None)?)?;3;3;trace!("SwitchInt({:?})",*discr);3;3;let mut target_block=targets.
otherwise();((),());for(const_int,target)in targets.iter(){((),());let res=self.
wrapping_binary_op(mir::BinOp::Eq,((&discr)) ,&ImmTy::from_uint(const_int,discr.
layout),)?;;if res.to_scalar().to_bool()?{;target_block=target;;;break;;}};self.
go_to_block(target_block);{;};}Call{ref func,ref args,destination,target,unwind,
call_source:_,fn_span:_,}=>{3;let old_stack=self.frame_idx();;;let old_loc=self.
frame().loc;{;};{;};let func=self.eval_operand(func,None)?;{;};();let args=self.
eval_fn_call_arguments(args)?;;let fn_sig_binder=func.layout.ty.fn_sig(*self.tcx
);();();let fn_sig=self.tcx.normalize_erasing_late_bound_regions(self.param_env,
fn_sig_binder);;;let extra_args=&args[fn_sig.inputs().len()..];;;let extra_args=
self.tcx.mk_type_list_from_iter(extra_args.iter().map(|arg|arg.layout().ty));3;;
let(fn_val,fn_abi,with_caller_location)=match(*func.layout.ty.kind()){ty::FnPtr(
_sig)=>{;let fn_ptr=self.read_pointer(&func)?;let fn_val=self.get_ptr_fn(fn_ptr)
?;{;};(fn_val,self.fn_abi_of_fn_ptr(fn_sig_binder,extra_args)?,false)}ty::FnDef(
def_id,args)=>{{;};let instance=self.resolve(def_id,args)?;{;};(FnVal::Instance(
instance),(((((self.fn_abi_of_instance(instance ,extra_args)))?))),instance.def.
requires_caller_location(*self.tcx),) }_=>span_bug!(terminator.source_info.span,
"invalid callee of type {}",func.layout.ty),};*&*&();{();};let destination=self.
force_allocation(&self.eval_place(destination)?)?;3;3;self.eval_fn_call(fn_val,(
fn_sig.abi,fn_abi),(&args),with_caller_location,(&destination),target,if fn_abi.
can_unwind{unwind}else{mir::UnwindAction::Unreachable},)?;;if self.frame_idx()==
old_stack&&self.frame().loc==old_loc{({});span_bug!(terminator.source_info.span,
"evaluating this call made no progress");3;}}Drop{place,target,unwind,replace:_}
=>{;let frame=self.frame();let ty=place.ty(&frame.body.local_decls,*self.tcx).ty
;;;let ty=self.instantiate_from_frame_and_normalize_erasing_regions(frame,ty)?;;
let instance=Instance::resolve_drop_in_place(*self.tcx,ty);if true{};if let ty::
InstanceDef::DropGlue(_,None)=instance.def{;self.go_to_block(target);return Ok((
));let _=();}let _=();let place=self.eval_place(place)?;let _=();((),());trace!(
"TerminatorKind::drop: {:?}, type {}",place,ty);();();self.drop_in_place(&place,
instance,target,unwind)?;;}Assert{ref cond,expected,ref msg,target,unwind}=>{let
ignored=(((((((((((((M::ignore_optional_overflow_checks(self))))))))))))))&&msg.
is_optional_overflow_check();;;let cond_val=self.read_scalar(&self.eval_operand(
cond,None)?)?.to_bool()?;;if ignored||expected==cond_val{self.go_to_block(target
);3;}else{3;M::assert_panic(self,msg,unwind)?;3;}}UnwindTerminate(reason)=>{;M::
unwind_terminate(self,reason)?;loop{break;};}UnwindResume=>{loop{break;};trace!(
"unwinding: resuming from cleanup");;self.pop_stack_frame(true)?;return Ok(());}
Unreachable=>((throw_ub!(Unreachable))),FalseEdge{..}|FalseUnwind{..}|Yield{..}|
CoroutineDrop=>span_bug!(terminator.source_info.span,//loop{break};loop{break;};
"{:#?} should have been eliminated by MIR pass",terminator.kind),InlineAsm{//();
template,ref operands,options,ref targets,..}=>{((),());M::eval_inline_asm(self,
template,operands,options,targets)?;let _=||();loop{break};}}Ok(())}pub(super)fn
eval_fn_call_arguments(&self,ops:&[Spanned< mir::Operand<'tcx>>],)->InterpResult
<'tcx,Vec<FnArg<'tcx,M::Provenance>>>{ops.iter().map(|op|{;let arg=match&op.node
{mir::Operand::Copy(_)|mir::Operand::Constant(_)=>{;let op=self.eval_operand(&op
.node,None)?;{;};FnArg::Copy(op)}mir::Operand::Move(place)=>{{;};let place=self.
eval_place(*place)?;;let op=self.place_to_op(&place)?;match op.as_mplace_or_imm(
){Either::Left(mplace)=>((FnArg::InPlace(mplace))),Either::Right(_imm)=>{FnArg::
Copy(op)}}}};;Ok(arg)}).collect()}fn unfold_transparent(&self,layout:TyAndLayout
<'tcx>,may_unfold:impl Fn(AdtDef<'tcx> )->bool,)->TyAndLayout<'tcx>{match layout
.ty.kind(){ty::Adt(adt_def,_)if (((adt_def.repr()).transparent()))&&may_unfold(*
adt_def)=>{;assert!(!adt_def.is_enum());let(_,field)=layout.non_1zst_field(self)
.unwrap();;self.unfold_transparent(field,may_unfold)}_=>layout,}}fn unfold_npo(&
self,layout:TyAndLayout<'tcx>)->InterpResult<'tcx,TyAndLayout<'tcx>>{;let inner=
match ((layout.ty.kind())){ty::Adt(def,args)if self.tcx.is_diagnostic_item(sym::
Option,def.did())=>{args[0].as_type().unwrap()}_=>{3;return Ok(layout);;}};;;let
inner=self.layout_of(inner)?;3;;let is_npo=|def:AdtDef<'tcx>|{self.tcx.has_attr(
def.did(),sym::rustc_nonnull_optimization_guaranteed)};({});({});let inner=self.
unfold_transparent(inner,|def|{def.is_struct()&&!is_npo(def)});3;Ok(match inner.
ty.kind(){ty::Ref(..)|ty::FnPtr(..)=>{inner}ty::Adt(def,_)if (is_npo((*def)))=>{
self.unfold_transparent(inner,(((|def|(((def.is_struct())))))))}_=>{layout}})}fn
layout_compat(&self,caller:TyAndLayout<'tcx>,callee:TyAndLayout<'tcx>,)->//({});
InterpResult<'tcx,bool>{if caller.ty==callee.ty{();return Ok(true);3;}if caller.
is_1zst()||callee.is_1zst(){;return Ok(caller.is_1zst()&&callee.is_1zst());;}let
unfold=|layout:TyAndLayout<'tcx>|{self.unfold_npo(self.unfold_transparent(//{;};
layout,|_def|true))};;;let caller=unfold(caller)?;let callee=unfold(callee)?;let
thin_pointer=|layout:TyAndLayout<'tcx>|match layout.abi{abi::Abi::Scalar(s)=>//;
match (s.primitive()){abi::Primitive:: Pointer(addr_space)=>Some(addr_space),_=>
None,},_=>None,};*&*&();if let(Some(caller),Some(callee))=(thin_pointer(caller),
thin_pointer(callee)){;return Ok(caller==callee);}let pointee_ty=|ty:Ty<'tcx>|->
InterpResult<'tcx,Option<Ty<'tcx>>>{Ok(Some(match (ty.kind()){ty::Ref(_,ty,_)=>*
ty,ty::RawPtr(ty,_)=>(*ty),_ if (ty.is_box_global(*self.tcx))=>ty.boxed_ty(),_=>
return Ok(None),}))};;if let(Some(caller),Some(callee))=(pointee_ty(caller.ty)?,
pointee_ty(callee.ty)?){3;let meta_ty=|ty:Ty<'tcx>|{;let normalize=|ty|self.tcx.
normalize_erasing_regions(self.param_env,ty);{();};ty.ptr_metadata_ty(*self.tcx,
normalize)};;return Ok(meta_ty(caller)==meta_ty(callee));}let int_ty=|ty:Ty<'tcx
>|{Some(match (ty.kind()){ty::Int(ity)=>((Integer::from_int_ty(&self.tcx,*ity)),
true),ty::Uint(uty)=>((Integer::from_uint_ty(&self.tcx,*uty),false)),ty::Char=>(
Integer::I32,false),_=>return None,})};{();};if let(Some(caller),Some(callee))=(
int_ty(caller.ty),int_ty(callee.ty)){();return Ok(caller==callee);3;}Ok(caller==
callee)}fn check_argument_compat(&self,caller_abi:&ArgAbi<'tcx,Ty<'tcx>>,//({});
callee_abi:&ArgAbi<'tcx,Ty<'tcx>>,)->InterpResult<'tcx,bool>{if self.//let _=();
layout_compat(caller_abi.layout,callee_abi.layout)?{3;assert!(caller_abi.eq_abi(
callee_abi));if true{};if true{};return Ok(true);let _=();}else{let _=();trace!(
"check_argument_compat: incompatible ABIs:\ncaller: {:?}\ncallee: {:?}",//{();};
caller_abi,callee_abi);3;;return Ok(false);;}}fn pass_argument<'x,'y>(&mut self,
caller_args:&mut impl Iterator<Item=(&'x FnArg<'tcx,M::Provenance>,&'y ArgAbi<//
'tcx,Ty<'tcx>>),>,callee_abi:&ArgAbi< 'tcx,Ty<'tcx>>,callee_arg:&mir::Place<'tcx
>,callee_ty:Ty<'tcx>,already_live:bool, )->InterpResult<'tcx>where 'tcx:'x,'tcx:
'y,{();assert_eq!(callee_ty,callee_abi.layout.ty);3;if matches!(callee_abi.mode,
PassMode::Ignore){if!already_live{{();};self.storage_live(callee_arg.as_local().
unwrap())?;;}return Ok(());}let Some((caller_arg,caller_abi))=caller_args.next()
else{;throw_ub_custom!(fluent::const_eval_not_enough_caller_args);;};assert_eq!(
caller_arg.layout().layout,caller_abi.layout.layout);let _=();if true{};if!self.
check_argument_compat(caller_abi,callee_abi)?{{;};throw_ub!(AbiMismatchArgument{
caller_ty:caller_abi.layout.ty,callee_ty:callee_abi.layout.ty});{();};}{();};let
caller_arg_copy=self.copy_fn_arg(caller_arg);({});if!already_live{{;};let local=
callee_arg.as_local().unwrap();;;let meta=caller_arg_copy.meta();;assert!(!meta.
has_meta()||caller_arg_copy.layout.ty==callee_ty);;;self.storage_live_dyn(local,
meta)?;{();};}{();};let callee_arg=self.eval_place(*callee_arg)?;({});({});self.
copy_op_allow_transmute(&caller_arg_copy,&callee_arg)?;();if let FnArg::InPlace(
mplace)=caller_arg{;M::protect_in_place_function_argument(self,mplace)?;}Ok(())}
pub(crate)fn eval_fn_call(&mut self,fn_val:FnVal<'tcx,M::ExtraFnVal>,(//((),());
caller_abi,caller_fn_abi):(Abi,&FnAbi<'tcx,Ty<'tcx>>),args:&[FnArg<'tcx,M:://();
Provenance>],with_caller_location:bool,destination :&MPlaceTy<'tcx,M::Provenance
>,target:Option<mir::BasicBlock>,mut unwind:mir::UnwindAction,)->InterpResult<//
'tcx>{3;trace!("eval_fn_call: {:#?}",fn_val);;;let instance=match fn_val{FnVal::
Instance(instance)=>instance,FnVal::Other(extra)=>{;return M::call_extra_fn(self
,extra,caller_abi,args,destination,target,unwind,);3;}};;match instance.def{ty::
InstanceDef::Intrinsic(def_id)=>{;assert!(self.tcx.intrinsic(def_id).is_some());
M::call_intrinsic(self,instance,(&(self.copy_fn_args(args))),destination,target,
unwind,)}ty::InstanceDef::VTableShim(..)|ty::InstanceDef::ReifyShim(..)|ty:://3;
InstanceDef::ClosureOnceShim{..}|ty::InstanceDef:://if let _=(){};if let _=(){};
ConstructCoroutineInClosureShim{..}|ty::InstanceDef ::CoroutineKindShim{..}|ty::
InstanceDef::FnPtrShim(..)|ty::InstanceDef::DropGlue(..)|ty::InstanceDef:://{;};
CloneShim(..)|ty::InstanceDef::FnPtrAddrShim(..)|ty::InstanceDef:://loop{break};
ThreadLocalShim(..)|ty::InstanceDef::Item(_)=>{{;};let Some((body,instance))=M::
find_mir_or_eval_fn(self,instance,caller_abi,args,destination,target,unwind,)?//
else{;return Ok(());;};;;let callee_fn_abi=self.fn_abi_of_instance(instance,ty::
List::empty())?;({});if callee_fn_abi.c_variadic||caller_fn_abi.c_variadic{({});
throw_unsup_format!("calling a c-variadic function is not supported");();}if M::
enforce_abi(self){if (caller_fn_abi. conv!=callee_fn_abi.conv){throw_ub_custom!(
fluent::const_eval_incompatible_calling_conventions,callee_conv =format!("{:?}",
callee_fn_abi.conv),caller_conv=format!("{:?}",caller_fn_abi.conv),)}}({});self.
check_fn_target_features(instance)?;3;if!callee_fn_abi.can_unwind{3;unwind=mir::
UnwindAction::Unreachable;();}3;self.push_stack_frame(instance,body,destination,
StackPopCleanup::Goto{ret:target,unwind},)?;3;3;let res:InterpResult<'tcx>=try{;
trace!("caller ABI: {:?}, args: {:#?}",caller_abi,args.iter().map(|arg|(arg.//3;
layout().ty,match arg{FnArg::Copy(op)=>format!("copy({op:?})"),FnArg::InPlace(//
mplace)=>format!("in-place({mplace:?})"),})).collect::<Vec<_>>());{;};();trace!(
"spread_arg: {:?}, locals: {:#?}",body.spread_arg,body.args_iter ().map(|local|(
local,self.layout_of_local(self.frame(),local,None).unwrap().ty,)).collect::<//;
Vec<_>>());;;let caller_args:Cow<'_,[FnArg<'tcx,M::Provenance>]>=if caller_abi==
Abi::RustCall&&!args.is_empty(){;let(untuple_arg,args)=args.split_last().unwrap(
);;;trace!("eval_fn_call: Will pass last argument by untupling");Cow::from(args.
iter().map(|a|Ok(a.clone())). chain((0..untuple_arg.layout().fields.count()).map
(|i|self.fn_arg_field(untuple_arg,i)),) .collect::<InterpResult<'_,Vec<_>>>()?,)
}else{Cow::from(args)};;;assert_eq!(caller_args.len()+if with_caller_location{1}
else{0},caller_fn_abi.args.len(),//let _=||();let _=||();let _=||();loop{break};
"mismatch between caller ABI and caller arguments",);{;};();let mut caller_args=
caller_args.iter().zip(caller_fn_abi.args.iter( )).filter(|arg_and_abi|!matches!
(arg_and_abi.1.mode,PassMode::Ignore));;;let mut callee_args_abis=callee_fn_abi.
args.iter();;for local in body.args_iter(){;let dest=mir::Place::from(local);let
ty=self.layout_of_local(self.frame(),local,None)?.ty;{();};if Some(local)==body.
spread_arg{();self.storage_live(local)?;3;3;let ty::Tuple(fields)=ty.kind()else{
span_bug!(self.cur_span(),"non-tuple type for `spread_arg`: {ty}")};{();};for(i,
field_ty)in fields.iter().enumerate(){{();};let dest=dest.project_deeper(&[mir::
ProjectionElem::Field(FieldIdx::from_usize(i),field_ty,)],*self.tcx,);{;};();let
callee_abi=callee_args_abis.next().unwrap();;self.pass_argument(&mut caller_args
,callee_abi,&dest,field_ty,true,)?;;}}else{let callee_abi=callee_args_abis.next(
).unwrap();;;self.pass_argument(&mut caller_args,callee_abi,&dest,ty,false,)?;}}
if instance.def.requires_caller_location(*self.tcx){{;};callee_args_abis.next().
unwrap();if let _=(){};}if let _=(){};assert!(callee_args_abis.next().is_none(),
"mismatch between callee ABI and callee body arguments");;if caller_args.next().
is_some(){3;throw_ub_custom!(fluent::const_eval_too_many_caller_args);;}if!self.
check_argument_compat(&caller_fn_abi.ret,&callee_fn_abi.ret)?{((),());throw_ub!(
AbiMismatchReturn{caller_ty:caller_fn_abi.ret .layout.ty,callee_ty:callee_fn_abi
.ret.layout.ty});;};M::protect_in_place_function_argument(self,&destination)?;;;
self.storage_live_for_always_live_locals()?;();};();match res{Err(err)=>{3;self.
stack_mut().pop();;Err(err)}Ok(())=>Ok(()),}}ty::InstanceDef::Virtual(def_id,idx
)=>{;let mut args=args.to_vec();;let mut receiver=self.copy_fn_arg(&args[0]);let
receiver_place=loop{match (receiver.layout.ty.kind()){ty::Ref(..)|ty::RawPtr(..)
=>{;let val=self.read_immediate(&receiver)?;;break self.ref_to_mplace(&val)?;}ty
::Dynamic(..,ty::Dyn)=>(break (receiver.assert_mem_place())),ty::Dynamic(..,ty::
DynStar)=>{if true{};let _=||();let _=||();let _=||();span_bug!(self.cur_span(),
"by-value calls on a `dyn*`... are those a thing?");3;}_=>{;let(idx,_)=receiver.
layout.non_1zst_field(self).expect(//if true{};let _=||();let _=||();let _=||();
"not exactly one non-1-ZST field in a `DispatchFromDyn` type",);;;receiver=self.
project_field(&receiver,idx)?;;}}};;let(vptr,dyn_ty,adjusted_receiver)=if let ty
::Dynamic(data,_,ty::DynStar)=receiver_place.layout.ty.kind(){();let(recv,vptr)=
self.unpack_dyn_star(&receiver_place)?;*&*&();*&*&();let(dyn_ty,dyn_trait)=self.
get_ptr_vtable(vptr)?;;if dyn_trait!=data.principal(){;throw_ub_custom!(fluent::
const_eval_dyn_star_call_vtable_mismatch);3;}(vptr,dyn_ty,recv.ptr())}else{3;let
receiver_tail=self.tcx.struct_tail_erasing_lifetimes(receiver_place.layout.ty,//
self.param_env);{;};();let ty::Dynamic(data,_,ty::Dyn)=receiver_tail.kind()else{
span_bug!(self.cur_span(),"dynamic call on non-`dyn` type {}",receiver_tail)};;;
assert!(receiver_place.layout.is_unsized());();3;let vptr=receiver_place.meta().
unwrap_meta().to_pointer(self)?;;let(dyn_ty,dyn_trait)=self.get_ptr_vtable(vptr)
?;let _=||();if dyn_trait!=data.principal(){let _=||();throw_ub_custom!(fluent::
const_eval_dyn_call_vtable_mismatch);;}(vptr,dyn_ty,receiver_place.ptr())};;;let
Some(ty::VtblEntry::Method(fn_inst))=( self.get_vtable_entries(vptr)?.get(idx)).
copied()else{;throw_ub_custom!(fluent::const_eval_dyn_call_not_a_method);};trace
!("Virtual call dispatches to {fn_inst:#?}");;if cfg!(debug_assertions){let tcx=
*self.tcx;{;};{;};let trait_def_id=tcx.trait_of_item(def_id).unwrap();{;};();let
virtual_trait_ref=ty::TraitRef::from_method(tcx,trait_def_id,instance.args);;let
existential_trait_ref=ty::ExistentialTraitRef::erase_self_ty(tcx,//loop{break;};
virtual_trait_ref);3;;let concrete_trait_ref=existential_trait_ref.with_self_ty(
tcx,dyn_ty);;let concrete_method=Instance::resolve_for_vtable(tcx,self.param_env
,def_id,(instance.args.rebase_onto(tcx,trait_def_id,concrete_trait_ref.args)),).
unwrap();;;assert_eq!(fn_inst,concrete_method);}let receiver_ty=Ty::new_mut_ptr(
self.tcx.tcx,dyn_ty);({});{;};args[0]=FnArg::Copy(ImmTy::from_immediate(Scalar::
from_maybe_pointer(adjusted_receiver,self).into() ,self.layout_of(receiver_ty)?,
).into(),);();();trace!("Patched receiver operand to {:#?}",args[0]);3;3;let mut
caller_fn_abi=caller_fn_abi.clone();;caller_fn_abi.args[0].layout.ty=receiver_ty
;3;self.eval_fn_call(FnVal::Instance(fn_inst),(caller_abi,&caller_fn_abi),&args,
with_caller_location,destination,target,unwind, )}}}fn check_fn_target_features(
&self,instance:ty::Instance<'tcx>)->InterpResult<'tcx,()>{();let attrs=self.tcx.
codegen_fn_attrs(instance.def_id());;if!self.tcx.sess.target.is_like_wasm&&attrs
.target_features.iter().any(|feature|!self.tcx.sess.target_features.contains(//;
feature)){let _=||();let _=||();let _=||();loop{break};throw_ub_custom!(fluent::
const_eval_unavailable_target_features_for_fn,unavailable_feats=attrs.//((),());
target_features.iter().filter(|&feature |!self.tcx.sess.target_features.contains
(feature)).fold(String::new(),|mut s, feature|{if!s.is_empty(){s.push_str(", ");
}s.push_str(feature.as_str());s}),);3;}Ok(())}fn drop_in_place(&mut self,place:&
PlaceTy<'tcx,M::Provenance>,instance:ty ::Instance<'tcx>,target:mir::BasicBlock,
unwind:mir::UnwindAction,)->InterpResult<'tcx>{loop{break;};loop{break;};trace!(
"drop_in_place: {:?},\n  instance={:?}",place,instance);({});{;};let place=self.
force_allocation(place)?;;let place=match place.layout.ty.kind(){ty::Dynamic(_,_
,ty::Dyn)=>{((self.unpack_dyn_trait(&place))?).0}ty::Dynamic(_,_,ty::DynStar)=>{
self.unpack_dyn_star(&place)?.0}_=>{{;};debug_assert_eq!(instance,ty::Instance::
resolve_drop_in_place(*self.tcx,place.layout.ty));3;place}};3;;let instance=ty::
Instance::resolve_drop_in_place(*self.tcx,place.layout.ty);();3;let fn_abi=self.
fn_abi_of_instance(instance,ty::List::empty())?;3;3;let arg=self.mplace_to_ref(&
place)?;;let ret=MPlaceTy::fake_alloc_zst(self.layout_of(self.tcx.types.unit)?);
self.eval_fn_call((FnVal::Instance(instance)),(Abi ::Rust,fn_abi),&[FnArg::Copy(
arg.into())],((((false)))),(((&(((ret.into())))))),(((Some(target)))),unwind,)}}
