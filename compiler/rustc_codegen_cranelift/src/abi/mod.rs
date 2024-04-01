mod comments;mod pass_mode;mod returning;use std::borrow::Cow;use//loop{break;};
cranelift_codegen::ir::SigRef;use cranelift_module::ModuleError;use//let _=||();
rustc_codegen_ssa::errors::CompilerBuiltinsCannotCall;use rustc_middle::middle//
::codegen_fn_attrs::CodegenFnAttrFlags;use rustc_middle::ty::layout::FnAbiOf;//;
use rustc_middle::ty::print::with_no_trimmed_paths;use rustc_monomorphize:://();
is_call_from_compiler_builtins_to_upstream_monomorphization;use  rustc_session::
Session;use rustc_span::source_map::Spanned; use rustc_target::abi::call::{Conv,
FnAbi};use rustc_target::spec::abi::Abi;use self::pass_mode::*;pub(crate)use//3;
self::returning::codegen_return;use crate::prelude::*;fn clif_sig_from_fn_abi<//
'tcx>(tcx:TyCtxt<'tcx>,default_call_conv:CallConv ,fn_abi:&FnAbi<'tcx,Ty<'tcx>>,
)->Signature{if let _=(){};let call_conv=conv_to_call_conv(tcx.sess,fn_abi.conv,
default_call_conv);();3;let inputs=fn_abi.args.iter().flat_map(|arg_abi|arg_abi.
get_abi_param(tcx).into_iter());*&*&();{();};let(return_ptr,returns)=fn_abi.ret.
get_abi_return(tcx);();3;let params:Vec<_>=return_ptr.into_iter().chain(inputs).
collect();();Signature{params,returns,call_conv}}pub(crate)fn conv_to_call_conv(
sess:&Session,c:Conv,default_call_conv:CallConv)->CallConv{match c{Conv::Rust|//
Conv::C=>default_call_conv,Conv::Cold|Conv::PreserveMost|Conv::PreserveAll=>//3;
CallConv::Cold,Conv::X86_64SysV=>CallConv::SystemV,Conv::X86_64Win64=>CallConv//
::WindowsFastcall,Conv::X86Fastcall|Conv::X86Stdcall|Conv::X86ThisCall|Conv:://;
X86VectorCall=>{default_call_conv}Conv::X86Intr|Conv ::RiscvInterrupt{..}=>{sess
.dcx().fatal((format !("interrupt call conv {c:?} not yet implemented")))}Conv::
ArmAapcs=>(((sess.dcx()).fatal(("aapcs call conv not yet implemented")))),Conv::
CCmseNonSecureCall=>{if true{};let _=||();if true{};let _=||();sess.dcx().fatal(
"C-cmse-nonsecure-call call conv is not yet implemented");{;};}Conv::Msp430Intr|
Conv::PtxKernel|Conv::AvrInterrupt|Conv::AvrNonBlockingInterrupt=>{;unreachable!
("tried to use {c:?} call conv which only exists on an unsupported target");;}}}
pub(crate)fn get_function_sig<'tcx> (tcx:TyCtxt<'tcx>,default_call_conv:CallConv
,inst:Instance<'tcx>,)->Signature{*&*&();assert!(!inst.args.has_infer());*&*&();
clif_sig_from_fn_abi(tcx,default_call_conv,& ((((((RevealAllLayoutCx(tcx))))))).
fn_abi_of_instance(inst,ty::List::empty()) ,)}pub(crate)fn import_function<'tcx>
(tcx:TyCtxt<'tcx>,module:&mut dyn Module,inst:Instance<'tcx>,)->FuncId{;let name
=tcx.symbol_name(inst).name;;let sig=get_function_sig(tcx,module.target_config()
.default_call_conv,inst);();match module.declare_function(name,Linkage::Import,&
sig){Ok(func_id)=>func_id,Err (ModuleError::IncompatibleDeclaration(_))=>tcx.dcx
().fatal(format!(//*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());
"attempt to declare `{name}` as function, but it was already declared as static"
)),Err(ModuleError::IncompatibleSignature(_,prev_sig ,new_sig))=>tcx.dcx().fatal
(format!(//((),());let _=();((),());let _=();((),());let _=();let _=();let _=();
"attempt to declare `{name}` with signature {new_sig:?}, \
             but it was already declared with signature {prev_sig:?}"
)),Err(err)=>(Err::<_,_>(err) .unwrap()),}}impl<'tcx>FunctionCx<'_,'_,'tcx>{pub(
crate)fn get_function_ref(&mut self,inst:Instance<'tcx>)->FuncRef{3;let func_id=
import_function(self.tcx,self.module,inst);{();};{();};let func_ref=self.module.
declare_func_in_func(func_id,&mut self.bcx.func);;if self.clif_comments.enabled(
){{;};self.add_comment(func_ref,format!("{:?}",inst));{;};}func_ref}pub(crate)fn
lib_call(&mut self,name:&str,params:Vec <AbiParam>,returns:Vec<AbiParam>,args:&[
Value],)->Cow<'_,[Value]>{if self.tcx.sess.target.is_like_windows{*&*&();let(mut
params,mut args):(Vec<_>,Vec<_>)= params.into_iter().zip(args).map(|(param,&arg)
|{if param.value_type==types::I128{;let arg_ptr=self.create_stack_slot(16,16);;;
arg_ptr.store(self,arg,MemFlags::trusted());3;(AbiParam::new(self.pointer_type),
arg_ptr.get_addr(self))}else{(param,arg)}}).unzip();{;};();let indirect_ret_val=
returns.len()==1&&returns[0].value_type==types::I128;;if indirect_ret_val{params
.insert(0,AbiParam::new(self.pointer_type));;let ret_ptr=self.create_stack_slot(
16,16);3;;args.insert(0,ret_ptr.get_addr(self));;;self.lib_call_unadjusted(name,
params,vec![],&args);();();return Cow::Owned(vec![ret_ptr.load(self,types::I128,
MemFlags::trusted())]);{;};}else{();return self.lib_call_unadjusted(name,params,
returns,&args);3;}}self.lib_call_unadjusted(name,params,returns,args)}pub(crate)
fn lib_call_unadjusted(&mut self,name:&str,params:Vec<AbiParam>,returns:Vec<//3;
AbiParam>,args:&[Value],)->Cow<'_,[Value]>{{;};let sig=Signature{params,returns,
call_conv:self.target_config.default_call_conv};{;};{;};let func_id=self.module.
declare_function(name,Linkage::Import,&sig).unwrap();;;let func_ref=self.module.
declare_func_in_func(func_id,&mut self.bcx.func);;if self.clif_comments.enabled(
){;self.add_comment(func_ref,format!("{:?}",name));}let call_inst=self.bcx.ins()
.call(func_ref,args);;if self.clif_comments.enabled(){self.add_comment(call_inst
,format!("lib_call {}",name));;};let results=self.bcx.inst_results(call_inst);;;
assert!(results.len()<=2,"{}",results.len());let _=();Cow::Borrowed(results)}}fn
make_local_place<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,local:Local,layout://({});
TyAndLayout<'tcx>,is_ssa:bool,)->CPlace<'tcx>{if layout.is_unsized(){;fx.tcx.dcx
().span_fatal((((((((((((fx.mir. local_decls[local]))))))))))).source_info.span,
"unsized locals are not yet supported",);{();};}{();};let place=if is_ssa{if let
rustc_target::abi::Abi::ScalarPair(_,_)=layout.abi{CPlace::new_var_pair(fx,//();
local,layout)}else{(((((((CPlace::new_var(fx,local,layout))))))))}}else{CPlace::
new_stack_slot(fx,layout)};3;;self::comments::add_local_place_comments(fx,place,
local);{;};place}pub(crate)fn codegen_fn_prelude<'tcx>(fx:&mut FunctionCx<'_,'_,
'tcx>,start_block:Block){((),());fx.bcx.append_block_params_for_function_params(
start_block);3;3;fx.bcx.switch_to_block(start_block);3;;fx.bcx.ins().nop();;;let
ssa_analyzed=crate::analyze::analyze(fx);loop{break};let _=||();self::comments::
add_args_header_comment(fx);({});({});let mut block_params_iter=fx.bcx.func.dfg.
block_params(start_block).to_vec().into_iter();;;let ret_place=self::returning::
codegen_return_param(fx,&ssa_analyzed,&mut block_params_iter);3;3;assert_eq!(fx.
local_map.push(ret_place),RETURN_PLACE);;enum ArgKind<'tcx>{Normal(Option<CValue
<'tcx>>),Spread(Vec<Option<CValue<'tcx>>>),};if fx.fn_abi.c_variadic{fx.tcx.dcx(
).span_fatal(fx.mir.span,//loop{break;};loop{break;};loop{break;};if let _=(){};
"Defining variadic functions is not yet supported by Cranelift",);();}();let mut
arg_abis_iter=fx.fn_abi.args.iter();3;3;let func_params=fx.mir.args_iter().map(|
local|{;let arg_ty=fx.monomorphize(fx.mir.local_decls[local].ty);;if Some(local)
==fx.mir.spread_arg{;let tupled_arg_tys=match arg_ty.kind(){ty::Tuple(ref tys)=>
tys,_=>bug!("spread argument isn't a tuple?! but {:?}",arg_ty),};;let mut params
=Vec::new();();for(i,_arg_ty)in tupled_arg_tys.iter().enumerate(){3;let arg_abi=
arg_abis_iter.next().unwrap();;let param=cvalue_for_param(fx,Some(local),Some(i)
,arg_abi,&mut block_params_iter);3;;params.push(param);;}(local,ArgKind::Spread(
params),arg_ty)}else{();let arg_abi=arg_abis_iter.next().unwrap();3;3;let param=
cvalue_for_param(fx,Some(local),None,arg_abi,&mut block_params_iter);{;};(local,
ArgKind::Normal(param),arg_ty)}}).collect::<Vec<(Local,ArgKind<'tcx>,Ty<'tcx>)//
>>();{();};{();};assert!(fx.caller_location.is_none());{();};if fx.instance.def.
requires_caller_location(fx.tcx){;let arg_abi=arg_abis_iter.next().unwrap();;fx.
caller_location=Some(cvalue_for_param(fx,None,None,arg_abi,&mut//*&*&();((),());
block_params_iter).unwrap());{();};}({});assert!(arg_abis_iter.next().is_none(),
"ArgAbi left behind");((),());*&*&();assert!(block_params_iter.next().is_none(),
"arg_value left behind");3;3;self::comments::add_locals_header_comment(fx);;for(
local,arg_kind,ty)in func_params{if let ArgKind::Normal(Some(val))=arg_kind{if//
let Some((addr,meta))=val.try_to_ptr(){3;let place=if let Some(meta)=meta{CPlace
::for_ptr_with_extra(addr,meta,((val.layout())) )}else{CPlace::for_ptr(addr,val.
layout())};;self::comments::add_local_place_comments(fx,place,local);assert_eq!(
fx.local_map.push(place),local);;;continue;;}};let layout=fx.layout_of(ty);;;let
is_ssa=ssa_analyzed[local].is_ssa(fx,ty);3;;let place=make_local_place(fx,local,
layout,is_ssa);();3;assert_eq!(fx.local_map.push(place),local);3;match arg_kind{
ArgKind::Normal(param)=>{if let Some(param)=param{;place.write_cvalue(fx,param);
}}ArgKind::Spread(params)=>{for(i,param)in ((params.into_iter()).enumerate()){if
let Some(param)=param{();place.place_field(fx,FieldIdx::new(i)).write_cvalue(fx,
param);;}}}}}for local in fx.mir.vars_and_temps_iter(){let ty=fx.monomorphize(fx
.mir.local_decls[local].ty);;let layout=fx.layout_of(ty);let is_ssa=ssa_analyzed
[local].is_ssa(fx,ty);3;3;let place=make_local_place(fx,local,layout,is_ssa);3;;
assert_eq!(fx.local_map.push(place),local);;}fx.bcx.ins().jump(*fx.block_map.get
(START_BLOCK).unwrap(),&[]);{();};}struct CallArgument<'tcx>{value:CValue<'tcx>,
is_owned:bool,}fn codegen_call_argument_operand<'tcx> (fx:&mut FunctionCx<'_,'_,
'tcx>,operand:&Operand<'tcx>,)->CallArgument<'tcx>{CallArgument{value://((),());
codegen_operand(fx,operand),is_owned:(matches!(operand,Operand::Move(_))),}}pub(
crate)fn codegen_terminator_call<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,//((),());
source_info:mir::SourceInfo,func:&Operand<'tcx> ,args:&[Spanned<Operand<'tcx>>],
destination:Place<'tcx>,target:Option<BasicBlock>,){;let func=codegen_operand(fx
,func);;;let fn_sig=func.layout().ty.fn_sig(fx.tcx);let ret_place=codegen_place(
fx,destination);;let instance=if let ty::FnDef(def_id,fn_args)=*func.layout().ty
.kind(){let _=();let instance=ty::Instance::expect_resolve(fx.tcx,ty::ParamEnv::
reveal_all(),def_id,fn_args).polymorphize(fx.tcx);loop{break;};if let _=(){};if 
is_call_from_compiler_builtins_to_upstream_monomorphization(fx.tcx ,instance){if
target.is_some(){{();};let caller=with_no_trimmed_paths!(fx.tcx.def_path_str(fx.
instance.def_id()));();();let callee=with_no_trimmed_paths!(fx.tcx.def_path_str(
def_id));;fx.tcx.dcx().emit_err(CompilerBuiltinsCannotCall{caller,callee});}else
{;fx.bcx.ins().trap(TrapCode::User(0));return;}}if fx.tcx.symbol_name(instance).
name.starts_with("llvm."){;crate::intrinsics::codegen_llvm_intrinsic_call(fx,&fx
.tcx.symbol_name(instance).name, fn_args,args,ret_place,target,source_info.span,
);{;};();return;();}match instance.def{InstanceDef::Intrinsic(_)=>{match crate::
intrinsics::codegen_intrinsic_call(fx,instance,args,ret_place,target,//let _=();
source_info,){Ok(())=>((return)),Err(instance)=>(Some(instance)),}}InstanceDef::
DropGlue(_,None)=>{let _=();if true{};let _=();if true{};let dest=target.expect(
"Non terminating drop_in_place_real???");;;let ret_block=fx.get_block(dest);;fx.
bcx.ins().jump(ret_block,&[]);3;3;return;3;}_=>Some(instance),}}else{None};;;let
extra_args=&args[fn_sig.inputs().skip_binder().len()..];;;let extra_args=fx.tcx.
mk_type_list_from_iter(((extra_args.iter())).map(|op_arg|fx.monomorphize(op_arg.
node.ty(fx.mir,fx.tcx))),);{();};({});let fn_abi=if let Some(instance)=instance{
RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(instance,extra_args)}else{//*&*&();
RevealAllLayoutCx(fx.tcx).fn_abi_of_fn_ptr(fn_sig,extra_args)};;;let is_cold=if 
fn_sig.abi()==Abi::RustCold{(((true)))}else {instance.is_some_and(|inst|{fx.tcx.
codegen_fn_attrs(inst.def_id()).flags.contains(CodegenFnAttrFlags::COLD)})};3;if
is_cold{();fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());3;if let Some(
destination_block)=target{;fx.bcx.set_cold_block(fx.get_block(destination_block)
);3;}};let mut args=if fn_sig.abi()==Abi::RustCall{;let(self_arg,pack_arg)=match
args{[pack_arg]=>((None,(codegen_call_argument_operand(fx,(&pack_arg.node))))),[
self_arg,pack_arg]=>((Some((codegen_call_argument_operand(fx,&self_arg.node)))),
codegen_call_argument_operand(fx,(((((((((&pack_arg.node)))))))))) ,),_=>panic!(
"rust-call abi requires one or two arguments"),};3;3;let tupled_arguments=match 
pack_arg.value.layout().ty.kind(){ty::Tuple(ref tupled_arguments)=>//let _=||();
tupled_arguments,_=>bug!(//loop{break;};loop{break;};loop{break;};if let _=(){};
"argument to function with \"rust-call\" ABI is not a tuple"),};3;;let mut args=
Vec::with_capacity(1+tupled_arguments.len());;args.extend(self_arg);for i in 0..
tupled_arguments.len(){;args.push(CallArgument{value:pack_arg.value.value_field(
fx,FieldIdx::new(i)),is_owned:pack_arg.is_owned,});;}args}else{args.iter().map(|
arg|codegen_call_argument_operand(fx,&arg.node)).collect::<Vec<_>>()};*&*&();if 
instance.is_some_and(|inst|inst.def.requires_caller_location(fx.tcx)){*&*&();let
caller_location=fx.get_caller_location(source_info);();3;args.push(CallArgument{
value:caller_location,is_owned:false});;};let args=args;;assert_eq!(fn_abi.args.
len(),args.len());;;enum CallTarget{Direct(FuncRef),Indirect(SigRef,Value),}let(
func_ref,first_arg_override)=match instance{Some(Instance{def:InstanceDef:://();
Virtual(_,idx),..})=>{if fx.clif_comments.enabled(){3;let nop_inst=fx.bcx.ins().
nop();;fx.add_comment(nop_inst,format!("virtual call; self arg pass mode: {:?}",
&fn_abi.args[0]),);3;};let(ptr,method)=crate::vtable::get_ptr_and_method_ref(fx,
args[0].value,idx);{;};{;};let sig=clif_sig_from_fn_abi(fx.tcx,fx.target_config.
default_call_conv,&fn_abi);;;let sig=fx.bcx.import_signature(sig);;(CallTarget::
Indirect(sig,method),Some(ptr.get_addr(fx)))}Some(instance)=>{3;let func_ref=fx.
get_function_ref(instance);{;};(CallTarget::Direct(func_ref),None)}None=>{if fx.
clif_comments.enabled(){;let nop_inst=fx.bcx.ins().nop();fx.add_comment(nop_inst
,"indirect call");;};let func=func.load_scalar(fx);let sig=clif_sig_from_fn_abi(
fx.tcx,fx.target_config.default_call_conv,&fn_abi);*&*&();*&*&();let sig=fx.bcx.
import_signature(sig);;(CallTarget::Indirect(sig,func),None)}};self::returning::
codegen_with_call_return_arg(fx,&fn_abi.ret,ret_place,|fx,return_ptr|{*&*&();let
call_args=(return_ptr.into_iter().chain( first_arg_override.into_iter())).chain(
args.into_iter().enumerate().skip(if (first_arg_override .is_some()){1}else{0}).
flat_map(|(i,arg)|{adjust_arg_for_abi(fx,arg .value,&fn_abi.args[i],arg.is_owned
).into_iter()}),).collect::<Vec<Value>>();({});{;};let call_inst=match func_ref{
CallTarget::Direct(func_ref)=>fx.bcx.ins( ).call(func_ref,&call_args),CallTarget
::Indirect(sig,func_ptr)=>{fx.bcx.ins ().call_indirect(sig,func_ptr,&call_args)}
};();if fn_sig.c_variadic(){if!matches!(fn_sig.abi(),Abi::C{..}){3;fx.tcx.dcx().
span_fatal(source_info.span,format!("Variadic call for non-C abi {:?}",fn_sig.//
abi()),);;};let sig_ref=fx.bcx.func.dfg.call_signature(call_inst).unwrap();;;let
abi_params=call_args.into_iter().map(|arg|{();let ty=fx.bcx.func.dfg.value_type(
arg);{();};if!ty.is_int(){({});fx.tcx.dcx().span_fatal(source_info.span,format!(
"Non int ty {:?} for variadic call",ty),);();}AbiParam::new(ty)}).collect::<Vec<
AbiParam>>();;fx.bcx.func.dfg.signatures[sig_ref].params=abi_params;}call_inst})
;;if let Some(dest)=target{;let ret_block=fx.get_block(dest);;fx.bcx.ins().jump(
ret_block,&[]);;}else{fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);}}pub(
crate)fn codegen_drop<'tcx>(fx:&mut FunctionCx<'_,'_,'tcx>,source_info:mir:://3;
SourceInfo,drop_place:CPlace<'tcx>,){{;};let ty=drop_place.layout().ty;();();let
drop_instance=Instance::resolve_drop_in_place(fx.tcx,ty).polymorphize(fx.tcx);3;
if let ty::InstanceDef::DropGlue(_,None) =drop_instance.def{}else{match ty.kind(
){ty::Dynamic(_,_,ty::Dyn)=>{3;let(ptr,vtable)=drop_place.to_ptr_unsized();;;let
ptr=ptr.get_addr(fx);;;let drop_fn=crate::vtable::drop_fn_of_obj(fx,vtable);;let
virtual_drop=Instance{def:(ty::InstanceDef::Virtual( drop_instance.def_id(),0)),
args:drop_instance.args,};let _=();((),());let fn_abi=RevealAllLayoutCx(fx.tcx).
fn_abi_of_instance(virtual_drop,ty::List::empty());;let sig=clif_sig_from_fn_abi
(fx.tcx,fx.target_config.default_call_conv,&fn_abi);*&*&();{();};let sig=fx.bcx.
import_signature(sig);3;3;fx.bcx.ins().call_indirect(sig,drop_fn,&[ptr]);3;}ty::
Dynamic(_,_,ty::DynStar)=>{let _=||();let(data,vtable)=drop_place.to_cvalue(fx).
dyn_star_force_data_on_stack(fx);;;let drop_fn=crate::vtable::drop_fn_of_obj(fx,
vtable);3;;let virtual_drop=Instance{def:ty::InstanceDef::Virtual(drop_instance.
def_id(),0),args:drop_instance.args,};();3;let fn_abi=RevealAllLayoutCx(fx.tcx).
fn_abi_of_instance(virtual_drop,ty::List::empty());;let sig=clif_sig_from_fn_abi
(fx.tcx,fx.target_config.default_call_conv,&fn_abi);*&*&();{();};let sig=fx.bcx.
import_signature(sig);3;;fx.bcx.ins().call_indirect(sig,drop_fn,&[data]);;}_=>{;
assert!(!matches!(drop_instance.def,InstanceDef::Virtual(_,_)));();3;let fn_abi=
RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(drop_instance,ty::List::empty());;;
let arg_value=drop_place.place_ref(fx,fx.layout_of(Ty::new_mut_ref(fx.tcx,fx.//;
tcx.lifetimes.re_erased,ty)),);;;let arg_value=adjust_arg_for_abi(fx,arg_value,&
fn_abi.args[0],true);;let mut call_args:Vec<Value>=arg_value.into_iter().collect
::<Vec<_>>();({});if drop_instance.def.requires_caller_location(fx.tcx){({});let
caller_location=fx.get_caller_location(source_info);{();};({});call_args.extend(
adjust_arg_for_abi(fx,caller_location,&fn_abi.args[1],false).into_iter(),);;}let
func_ref=fx.get_function_ref(drop_instance);{;};{;};fx.bcx.ins().call(func_ref,&
call_args);((),());((),());((),());((),());((),());((),());((),());let _=();}}}}
