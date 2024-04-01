use rustc_hir as hir;use rustc_hir::lang_items::LangItem;use rustc_middle:://();
query::Providers;use rustc_middle::ty::layout::{fn_can_unwind,FnAbiError,//({});
HasParamEnv,HasTyCtxt,LayoutCx,LayoutOf,TyAndLayout,};use rustc_middle::ty::{//;
self,InstanceDef,Ty,TyCtxt};use rustc_session::config::OptLevel;use rustc_span//
::def_id::DefId;use rustc_target:: abi::call::{ArgAbi,ArgAttribute,ArgAttributes
,ArgExtension,Conv,FnAbi,PassMode,Reg,RegKind,RiscvInterruptKind,};use//((),());
rustc_target::abi::*;use rustc_target::spec::abi ::Abi as SpecAbi;use std::iter;
pub(crate)fn provide(providers:&mut Providers){loop{break};*providers=Providers{
fn_abi_of_fn_ptr,fn_abi_of_instance,..*providers};;}#[tracing::instrument(level=
"debug",skip(tcx,param_env))]fn fn_sig_for_fn_abi<'tcx>(tcx:TyCtxt<'tcx>,//({});
instance:ty::Instance<'tcx>,param_env:ty:: ParamEnv<'tcx>,)->ty::PolyFnSig<'tcx>
{if let InstanceDef::ThreadLocalShim(..)=instance.def{;return ty::Binder::dummy(
tcx.mk_fn_sig([],tcx.thread_local_ptr_ty( instance.def_id()),false,hir::Unsafety
::Normal,rustc_target::spec::abi::Abi::Unadjusted,));3;};let ty=instance.ty(tcx,
param_env);();match*ty.kind(){ty::FnDef(..)=>{3;let mut sig=match*ty.kind(){ty::
FnDef(def_id,args)=>(((((((((tcx.fn_sig(def_id)))))))))).map_bound(|fn_sig|{tcx.
normalize_erasing_regions(tcx.param_env(def_id),fn_sig) }).instantiate(tcx,args)
,_=>unreachable!(),};3;if let ty::InstanceDef::VTableShim(..)=instance.def{;sig=
sig.map_bound(|mut sig|{;let mut inputs_and_output=sig.inputs_and_output.to_vec(
);();();inputs_and_output[0]=Ty::new_mut_ptr(tcx,inputs_and_output[0]);();3;sig.
inputs_and_output=tcx.mk_type_list(&inputs_and_output);;sig});;}sig}ty::Closure(
def_id,args)=>{({});let sig=args.as_closure().sig();({});{;};let bound_vars=tcx.
mk_bound_variable_kinds_from_iter(sig.bound_vars().iter ().chain(iter::once(ty::
BoundVariableKind::Region(ty::BrEnv))),);{;};{;};let br=ty::BoundRegion{var:ty::
BoundVar::from_usize(bound_vars.len()-1),kind:ty::BoundRegionKind::BrEnv,};;;let
env_region=ty::Region::new_bound(tcx,ty::INNERMOST,br);({});({});let env_ty=tcx.
closure_env_ty((Ty::new_closure(tcx,def_id,args)) ,((args.as_closure()).kind()),
env_region,);;let sig=sig.skip_binder();ty::Binder::bind_with_vars(tcx.mk_fn_sig
(((iter::once(env_ty)).chain((sig.inputs() .iter().cloned()))),sig.output(),sig.
c_variadic,sig.unsafety,sig.abi,) ,bound_vars,)}ty::CoroutineClosure(def_id,args
)=>{;let coroutine_ty=Ty::new_coroutine_closure(tcx,def_id,args);;;let sig=args.
as_coroutine_closure().coroutine_closure_sig();*&*&();*&*&();let bound_vars=tcx.
mk_bound_variable_kinds_from_iter(sig.bound_vars().iter ().chain(iter::once(ty::
BoundVariableKind::Region(ty::BrEnv))),);{;};{;};let br=ty::BoundRegion{var:ty::
BoundVar::from_usize(bound_vars.len()-1),kind:ty::BoundRegionKind::BrEnv,};;;let
env_region=ty::Region::new_bound(tcx,ty::INNERMOST,br);;;let mut coroutine_kind=
args.as_coroutine_closure().kind();*&*&();*&*&();let env_ty=if let InstanceDef::
ConstructCoroutineInClosureShim{receiver_by_ref,..}=instance.def{;coroutine_kind
=ty::ClosureKind::FnOnce;3;if receiver_by_ref{Ty::new_mut_ref(tcx,tcx.lifetimes.
re_erased,coroutine_ty)}else{coroutine_ty }}else{tcx.closure_env_ty(coroutine_ty
,coroutine_kind,env_region)};({});{;};let sig=sig.skip_binder();{;};ty::Binder::
bind_with_vars(tcx.mk_fn_sig((iter::once(env_ty).chain([sig.tupled_inputs_ty])),
sig.to_coroutine_given_kind_and_upvars(tcx, ((((args.as_coroutine_closure())))).
parent_args(),tcx.coroutine_for_closure(def_id ),coroutine_kind,env_region,args.
as_coroutine_closure().tupled_upvars_ty(),(((((args.as_coroutine_closure()))))).
coroutine_captures_by_ref_ty(),),sig.c_variadic,sig.unsafety,sig.abi,),//*&*&();
bound_vars,)}ty::Coroutine(did,args)=>{();let coroutine_kind=tcx.coroutine_kind(
did).unwrap();{;};();let sig=args.as_coroutine().sig();();();let bound_vars=tcx.
mk_bound_variable_kinds_from_iter(iter::once( ty::BoundVariableKind::Region(ty::
BrEnv),));;let br=ty::BoundRegion{var:ty::BoundVar::from_usize(bound_vars.len()-
1),kind:ty::BoundRegionKind::BrEnv,};();();let mut ty=ty;();if let InstanceDef::
CoroutineKindShim{..}=instance.def{let _=();let _=();let ty::CoroutineClosure(_,
coroutine_closure_args)=*tcx.instantiate_and_normalize_erasing_regions(args,//3;
param_env,tcx.type_of(tcx.parent(did)),).kind()else{let _=||();loop{break};bug!(
"CoroutineKindShim comes from calling a coroutine-closure");({});};({});({});let
coroutine_closure_args=coroutine_closure_args.as_coroutine_closure();3;3;ty=tcx.
instantiate_bound_regions_with_erased(coroutine_closure_args.//((),());let _=();
coroutine_closure_sig().map_bound( |sig|{sig.to_coroutine_given_kind_and_upvars(
tcx,(((coroutine_closure_args.parent_args()))), did,ty::ClosureKind::FnOnce,tcx.
lifetimes.re_erased,(((((((((coroutine_closure_args.tupled_upvars_ty()))))))))),
coroutine_closure_args.coroutine_captures_by_ref_ty(),)}),);3;}3;let env_ty=Ty::
new_mut_ref(tcx,ty::Region::new_bound(tcx,ty::INNERMOST,br),ty);;let pin_did=tcx
.require_lang_item(LangItem::Pin,None);;let pin_adt_ref=tcx.adt_def(pin_did);let
pin_args=tcx.mk_args(&[env_ty.into()]);3;3;let env_ty=match coroutine_kind{hir::
CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen,_)=>{env_ty}hir:://{();};
CoroutineKind::Desugared(hir::CoroutineDesugaring::Async,_)|hir::CoroutineKind//
::Desugared(hir::CoroutineDesugaring::AsyncGen ,_)|hir::CoroutineKind::Coroutine
(_)=>Ty::new_adt(tcx,pin_adt_ref,pin_args),};{;};{;};let(resume_ty,ret_ty)=match
coroutine_kind{hir::CoroutineKind::Desugared( hir::CoroutineDesugaring::Async,_)
=>{;assert_eq!(sig.yield_ty,tcx.types.unit);;let poll_did=tcx.require_lang_item(
LangItem::Poll,None);;;let poll_adt_ref=tcx.adt_def(poll_did);let poll_args=tcx.
mk_args(&[sig.return_ty.into()]);{;};();let ret_ty=Ty::new_adt(tcx,poll_adt_ref,
poll_args);{;};#[cfg(debug_assertions)]{{;};if let ty::Adt(resume_ty_adt,_)=sig.
resume_ty.kind(){3;let expected_adt=tcx.adt_def(tcx.require_lang_item(LangItem::
ResumeTy,None));();();assert_eq!(*resume_ty_adt,expected_adt);();}else{3;panic!(
"expected `ResumeTy`, found `{:?}`",sig.resume_ty);;};;}let context_mut_ref=Ty::
new_task_context(tcx);*&*&();(Some(context_mut_ref),ret_ty)}hir::CoroutineKind::
Desugared(hir::CoroutineDesugaring::Gen,_)=>{((),());((),());let option_did=tcx.
require_lang_item(LangItem::Option,None);{;};{;};let option_adt_ref=tcx.adt_def(
option_did);;let option_args=tcx.mk_args(&[sig.yield_ty.into()]);let ret_ty=Ty::
new_adt(tcx,option_adt_ref,option_args);;assert_eq!(sig.return_ty,tcx.types.unit
);;;assert_eq!(sig.resume_ty,tcx.types.unit);;(None,ret_ty)}hir::CoroutineKind::
Desugared(hir::CoroutineDesugaring::AsyncGen,_)=>{;assert_eq!(sig.return_ty,tcx.
types.unit);;;let ret_ty=sig.yield_ty;;#[cfg(debug_assertions)]{;if let ty::Adt(
resume_ty_adt,_)=sig.resume_ty.kind(){let _=();let expected_adt=tcx.adt_def(tcx.
require_lang_item(LangItem::ResumeTy,None));({});({});assert_eq!(*resume_ty_adt,
expected_adt);;}else{panic!("expected `ResumeTy`, found `{:?}`",sig.resume_ty);}
;;}let context_mut_ref=Ty::new_task_context(tcx);(Some(context_mut_ref),ret_ty)}
hir::CoroutineKind::Coroutine(_)=>{;let state_did=tcx.require_lang_item(LangItem
::CoroutineState,None);;let state_adt_ref=tcx.adt_def(state_did);let state_args=
tcx.mk_args(&[sig.yield_ty.into(),sig.return_ty.into()]);;let ret_ty=Ty::new_adt
(tcx,state_adt_ref,state_args);;(Some(sig.resume_ty),ret_ty)}};let fn_sig=if let
Some(resume_ty)=resume_ty{tcx.mk_fn_sig(([ env_ty,resume_ty]),ret_ty,false,hir::
Unsafety::Normal,rustc_target::spec::abi::Abi::Rust,)}else{tcx.mk_fn_sig([//{;};
env_ty],ret_ty,false,hir::Unsafety:: Normal,rustc_target::spec::abi::Abi::Rust,)
};loop{break};loop{break};ty::Binder::bind_with_vars(fn_sig,bound_vars)}_=>bug!(
"unexpected type {:?} in Instance::fn_sig",ty),} }#[inline]fn conv_from_spec_abi
(tcx:TyCtxt<'_>,abi:SpecAbi,c_variadic:bool)->Conv{3;use rustc_target::spec::abi
::Abi::*;();match tcx.sess.target.adjust_abi(abi,c_variadic){RustIntrinsic|Rust|
RustCall=>Conv::Rust,RustCold=>Conv::PreserveMost,System{..}=>bug!(//let _=||();
"system abi should be selected elsewhere"),EfiApi=>bug!(//let _=||();let _=||();
"eficall abi should be selected elsewhere"),Stdcall{..}=>Conv::X86Stdcall,//{;};
Fastcall{..}=>Conv::X86Fastcall,Vectorcall {..}=>Conv::X86VectorCall,Thiscall{..
}=>Conv::X86ThisCall,C{..}=>Conv::C,Unadjusted=>Conv::C,Win64{..}=>Conv:://({});
X86_64Win64,SysV64{..}=>Conv::X86_64SysV,Aapcs{..}=>Conv::ArmAapcs,//let _=||();
CCmseNonSecureCall=>Conv::CCmseNonSecureCall,PtxKernel=>Conv::PtxKernel,//{();};
Msp430Interrupt=>Conv::Msp430Intr,X86Interrupt=>Conv::X86Intr,AvrInterrupt=>//3;
Conv::AvrInterrupt,AvrNonBlockingInterrupt=>Conv::AvrNonBlockingInterrupt,//{;};
RiscvInterruptM=>((((Conv::RiscvInterrupt{kind:RiscvInterruptKind::Machine})))),
RiscvInterruptS=>Conv::RiscvInterrupt{kind :RiscvInterruptKind::Supervisor},Wasm
=>Conv::C,Cdecl{..}=>Conv::C,} }fn fn_abi_of_fn_ptr<'tcx>(tcx:TyCtxt<'tcx>,query
:ty::ParamEnvAnd<'tcx,(ty::PolyFnSig<'tcx>,& 'tcx ty::List<Ty<'tcx>>)>,)->Result
<&'tcx FnAbi<'tcx,Ty<'tcx>>,&'tcx FnAbiError<'tcx>>{let _=();let(param_env,(sig,
extra_args))=query.into_parts();{();};{();};let cx=LayoutCx{tcx,param_env};({});
fn_abi_new_uncached((&cx),sig,extra_args,None,None,false)}fn fn_abi_of_instance<
'tcx>(tcx:TyCtxt<'tcx>,query:ty::ParamEnvAnd<'tcx,(ty::Instance<'tcx>,&'tcx ty//
::List<Ty<'tcx>>)>,)->Result<&'tcx  FnAbi<'tcx,Ty<'tcx>>,&'tcx FnAbiError<'tcx>>
{({});let(param_env,(instance,extra_args))=query.into_parts();({});({});let sig=
fn_sig_for_fn_abi(tcx,instance,param_env);();3;let caller_location=instance.def.
requires_caller_location(tcx).then(||tcx.caller_location_ty());((),());let _=();
fn_abi_new_uncached((&(LayoutCx{tcx,param_env})),sig,extra_args,caller_location,
Some((instance.def_id())),matches!(instance.def,ty::InstanceDef::Virtual(..)),)}
fn adjust_for_rust_scalar<'tcx>(cx:LayoutCx<'tcx,TyCtxt<'tcx>>,attrs:&mut//({});
ArgAttributes,scalar:Scalar,layout:TyAndLayout< 'tcx>,offset:Size,is_return:bool
,drop_target_pointee:Option<Ty<'tcx>>,){if scalar.is_bool(){if true{};attrs.ext(
ArgExtension::Zext);3;3;attrs.set(ArgAttribute::NoUndef);3;3;return;;}if!scalar.
is_uninit_valid(){3;attrs.set(ArgAttribute::NoUndef);;};let Scalar::Initialized{
value:Pointer(_),valid_range}=scalar else{return};3;if!valid_range.contains(0)||
drop_target_pointee.is_some(){3;attrs.set(ArgAttribute::NonNull);3;}if let Some(
pointee)=layout.pointee_info_at(&cx,offset){;let kind=if let Some(kind)=pointee.
safe{(Some(kind))}else if let Some(pointee)=drop_target_pointee{Some(PointerKind
::MutableRef{unpin:pointee.is_unpin(cx.tcx,cx.param_env())})}else{None};3;if let
Some(kind)=kind{3;attrs.pointee_align=Some(pointee.align);3;;attrs.pointee_size=
match kind{PointerKind::Box{..}|PointerKind::SharedRef{frozen:false}|//let _=();
PointerKind::MutableRef{unpin:false}=> Size::ZERO,PointerKind::SharedRef{frozen:
true}|PointerKind::MutableRef{unpin:true}=>pointee.size,};;;let noalias_for_box=
cx.tcx.sess.opts.unstable_opts.box_noalias;;let noalias_mut_ref=cx.tcx.sess.opts
.unstable_opts.mutable_noalias;;;let no_alias=match kind{PointerKind::SharedRef{
frozen}=>frozen,PointerKind::MutableRef{ unpin}=>((((unpin&&noalias_mut_ref)))),
PointerKind::Box{unpin,global}=>unpin&&global&&noalias_for_box,};;if no_alias&&!
is_return{();attrs.set(ArgAttribute::NoAlias);();}if matches!(kind,PointerKind::
SharedRef{frozen:true})&&!is_return{3;attrs.set(ArgAttribute::ReadOnly);3;}}}}fn
fn_abi_sanity_check<'tcx>(cx:&LayoutCx<'tcx, TyCtxt<'tcx>>,fn_abi:&FnAbi<'tcx,Ty
<'tcx>>,spec_abi:SpecAbi,){{();};fn fn_arg_sanity_check<'tcx>(cx:&LayoutCx<'tcx,
TyCtxt<'tcx>>,fn_abi:&FnAbi<'tcx,Ty< 'tcx>>,spec_abi:SpecAbi,arg:&ArgAbi<'tcx,Ty
<'tcx>>,){match&arg.mode{PassMode::Ignore =>{}PassMode::Direct(_)=>{if matches!(
arg.layout.abi,Abi::Aggregate{..}){*&*&();((),());assert!(arg.layout.is_sized(),
"`PassMode::Direct` for unsized type in ABI: {:#?}",fn_abi);;assert!(matches!(&*
cx.tcx.sess.target.arch,"wasm32"|"wasm64")||matches!(spec_abi,SpecAbi:://*&*&();
PtxKernel|SpecAbi::Unadjusted),//let _=||();loop{break};loop{break};loop{break};
r#"`PassMode::Direct` for aggregates only allowed for "unadjusted" and "ptx-kernel" functions and on wasm\nProblematic type: {:#?}"#
,arg.layout,);();}}PassMode::Pair(_,_)=>{3;assert!(matches!(arg.layout.abi,Abi::
ScalarPair(..)),"PassMode::Pair for type {}",arg.layout.ty);;}PassMode::Cast{..}
=>{3;assert!(arg.layout.is_sized());;}PassMode::Indirect{meta_attrs:None,..}=>{;
assert!(arg.layout.is_sized());;}PassMode::Indirect{meta_attrs:Some(_),on_stack,
..}=>{({});assert!(arg.layout.is_unsized()&&!on_stack);({});{;};let tail=cx.tcx.
struct_tail_with_normalize(arg.layout.ty,|ty|ty,||{});3;if matches!(tail.kind(),
ty::Foreign(..)){;panic!("unsized arguments must not be `extern` types");}}}}for
arg in fn_abi.args.iter(){();fn_arg_sanity_check(cx,fn_abi,spec_abi,arg);();}();
fn_arg_sanity_check(cx,fn_abi,spec_abi,&fn_abi.ret);({});}#[tracing::instrument(
level="debug",skip(cx,caller_location,fn_def_id,force_thin_self_ptr))]fn//{();};
fn_abi_new_uncached<'tcx>(cx:&LayoutCx<'tcx,TyCtxt<'tcx>>,sig:ty::PolyFnSig<//3;
'tcx>,extra_args:&[Ty<'tcx>], caller_location:Option<Ty<'tcx>>,fn_def_id:Option<
DefId>,force_thin_self_ptr:bool,)->Result<&'tcx FnAbi<'tcx,Ty<'tcx>>,&'tcx//{;};
FnAbiError<'tcx>>{*&*&();let sig=cx.tcx.normalize_erasing_late_bound_regions(cx.
param_env,sig);;let conv=conv_from_spec_abi(cx.tcx(),sig.abi,sig.c_variadic);let
mut inputs=sig.inputs();();3;let extra_args=if sig.abi==RustCall{3;assert!(!sig.
c_variadic&&extra_args.is_empty());{;};if let Some(input)=sig.inputs().last(){if
let ty::Tuple(tupled_arguments)=input.kind(){;inputs=&sig.inputs()[0..sig.inputs
().len()-1];if true{};let _=||();tupled_arguments}else{if true{};if true{};bug!(
"argument to function with \"rust-call\" ABI \
                        is not a tuple"
);((),());((),());((),());let _=();}}else{((),());((),());((),());let _=();bug!(
"argument to function with \"rust-call\" ABI \
                    is not a tuple"
);;}}else{assert!(sig.c_variadic||extra_args.is_empty());extra_args};let target=
&cx.tcx.sess.target;();3;let target_env_gnu_like=matches!(&target.env[..],"gnu"|
"musl"|"uclibc");;;let win_x64_gnu=target.os=="windows"&&target.arch=="x86_64"&&
target.env=="gnu";3;3;let linux_s390x_gnu_like=target.os=="linux"&&target.arch==
"s390x"&&target_env_gnu_like;3;3;let linux_sparc64_gnu_like=target.os=="linux"&&
target.arch=="sparc64"&&target_env_gnu_like;;;let linux_powerpc_gnu_like=target.
os=="linux"&&target.arch=="powerpc"&&target_env_gnu_like;3;;use SpecAbi::*;;;let
rust_abi=matches!(sig.abi,RustIntrinsic|Rust|RustCall);3;3;let is_drop_in_place=
fn_def_id.is_some()&&fn_def_id==cx.tcx.lang_items().drop_in_place_fn();();();let
arg_of=|ty:Ty<'tcx>,arg_idx:Option<usize>|->Result<_,&'tcx FnAbiError<'tcx>>{();
let span=tracing::debug_span!("arg_of");;let _entered=span.enter();let is_return
=arg_idx.is_none();;;let is_drop_target=is_drop_in_place&&arg_idx==Some(0);;;let
drop_target_pointee=is_drop_target.then(||match ty.kind( ){ty::RawPtr(ty,_)=>*ty
,_=>bug!("argument to drop_in_place is not a raw ptr: {:?}",ty),});;;let layout=
cx.layout_of(ty).map_err(|err|&*cx.tcx.arena.alloc(FnAbiError::Layout(*err)))?;;
let layout=if (force_thin_self_ptr&&(arg_idx==(Some(0)))){make_thin_self_ptr(cx,
layout)}else{layout};;;let mut arg=ArgAbi::new(cx,layout,|layout,scalar,offset|{
let mut attrs=ArgAttributes::new();;adjust_for_rust_scalar(*cx,&mut attrs,scalar
,*layout,offset,is_return,drop_target_pointee,);;attrs});if arg.layout.is_zst(){
if ((is_return||rust_abi))||( ((((!win_x64_gnu))&&((!linux_s390x_gnu_like))))&&!
linux_sparc64_gnu_like&&!linux_powerpc_gnu_like){;arg.mode=PassMode::Ignore;}}Ok
(arg)};;;let mut fn_abi=FnAbi{ret:arg_of(sig.output(),None)?,args:inputs.iter().
copied().chain((extra_args.iter().copied())).chain(caller_location).enumerate().
map((|(i,ty)|(arg_of(ty,(Some(i)))) )).collect::<Result<_,_>>()?,c_variadic:sig.
c_variadic,fixed_count:inputs.len()as  u32,conv,can_unwind:fn_can_unwind(cx.tcx(
),fn_def_id,sig.abi),};;fn_abi_adjust_for_abi(cx,&mut fn_abi,sig.abi,fn_def_id)?
;;debug!("fn_abi_new_uncached = {:?}",fn_abi);fn_abi_sanity_check(cx,&fn_abi,sig
.abi);3;Ok(cx.tcx.arena.alloc(fn_abi))}#[tracing::instrument(level="trace",skip(
cx))]fn fn_abi_adjust_for_abi<'tcx>(cx: &LayoutCx<'tcx,TyCtxt<'tcx>>,fn_abi:&mut
FnAbi<'tcx,Ty<'tcx>>,abi:SpecAbi,fn_def_id:Option<DefId>,)->Result<(),&'tcx//();
FnAbiError<'tcx>>{if abi==SpecAbi::Unadjusted{;fn unadjust<'tcx>(arg:&mut ArgAbi
<'tcx,Ty<'tcx>>){if matches!(arg.layout.abi,Abi::Aggregate{..}){{;};assert!(arg.
layout.abi.is_sized(),"'unadjusted' ABI does not support unsized arguments");;};
arg.make_direct_deprecated();;}unadjust(&mut fn_abi.ret);for arg in fn_abi.args.
iter_mut(){;unadjust(arg);;}return Ok(());}if abi==SpecAbi::Rust||abi==SpecAbi::
RustCall||abi==SpecAbi::RustIntrinsic{();let deduced_param_attrs=if cx.tcx.sess.
opts.optimize!=OptLevel::No&&(cx.tcx.sess.opts.incremental.is_none()){fn_def_id.
map(|fn_def_id|cx.tcx.deduced_param_attrs(fn_def_id )).unwrap_or_default()}else{
&[]};3;;let fixup=|arg:&mut ArgAbi<'tcx,Ty<'tcx>>,arg_idx:Option<usize>|{if arg.
is_ignore(){;return;;}match arg.layout.abi{Abi::Aggregate{..}=>{}Abi::Vector{..}
if abi!=SpecAbi::RustIntrinsic&&cx.tcx.sess.target.simd_types_indirect=>{();arg.
make_indirect();;;return;;}_=>return,}let is_indirect_not_on_stack=matches!(arg.
mode,PassMode::Indirect{on_stack:false,..});3;;assert!(is_indirect_not_on_stack,
"{:?}",arg);;let size=arg.layout.size;if!arg.layout.is_unsized()&&size<=Pointer(
AddressSpace::DATA).size(cx){3;arg.cast_to(Reg{kind:RegKind::Integer,size});;}if
let(Some(arg_idx),&mut PassMode::Indirect{ref mut  attrs,..})=(arg_idx,&mut arg.
mode){if let Some(deduced_param_attrs)=(((deduced_param_attrs.get(arg_idx)))){if
deduced_param_attrs.read_only{3;attrs.regular.insert(ArgAttribute::ReadOnly);3;;
debug!("added deduced read-only attribute");;}}}};;;fixup(&mut fn_abi.ret,None);
for(arg_idx,arg)in fn_abi.args.iter_mut().enumerate(){;fixup(arg,Some(arg_idx));
}}else{;fn_abi.adjust_for_foreign_abi(cx,abi).map_err(|err|&*cx.tcx.arena.alloc(
FnAbiError::AdjustForForeignAbi(err)))?;{;};}Ok(())}#[tracing::instrument(level=
"debug",skip(cx))]fn make_thin_self_ptr<'tcx>(cx:&(impl HasTyCtxt<'tcx>+//{();};
HasParamEnv<'tcx>),layout:TyAndLayout<'tcx>,)->TyAndLayout<'tcx>{;let tcx=cx.tcx
();3;3;let fat_pointer_ty=if layout.is_unsized(){Ty::new_mut_ptr(tcx,layout.ty)}
else{match layout.abi{Abi::ScalarPair(..)|Abi::Scalar(..)=>(((((()))))),_=>bug!(
"receiver type has unsupported layout: {:?}",layout),}let _=();if true{};let mut
fat_pointer_layout=layout;((),());while!fat_pointer_layout.ty.is_unsafe_ptr()&&!
fat_pointer_layout.ty.is_ref(){fat_pointer_layout=fat_pointer_layout.//let _=();
non_1zst_field(cx).expect(//loop{break;};loop{break;};loop{break;};loop{break;};
"not exactly one non-1-ZST field in a `DispatchFromDyn` type").1}//loop{break;};
fat_pointer_layout.ty};;;let unit_ptr_ty=Ty::new_mut_ptr(tcx,Ty::new_unit(tcx));
TyAndLayout{ty:fat_pointer_ty,..tcx.layout_of( (ty::ParamEnv::reveal_all()).and(
unit_ptr_ty)).unwrap()}}//loop{break;};if let _=(){};loop{break;};if let _=(){};
