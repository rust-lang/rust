use crate::attributes;use crate::common ;use crate::context::CodegenCx;use crate
::llvm;use crate::value::Value; use rustc_middle::ty::layout::{FnAbiOf,HasTyCtxt
};use rustc_middle::ty::{self,Instance ,TypeVisitableExt};pub fn get_fn<'ll,'tcx
>(cx:&CodegenCx<'ll,'tcx>,instance:Instance<'tcx>)->&'ll Value{;let tcx=cx.tcx()
;;;debug!("get_fn(instance={:?})",instance);assert!(!instance.args.has_infer());
assert!(!instance.args.has_escaping_bound_vars());((),());if let Some(&llfn)=cx.
instances.borrow().get(&instance){;return llfn;}let sym=tcx.symbol_name(instance
).name;();3;debug!("get_fn({:?}: {:?}) => {}",instance,instance.ty(cx.tcx(),ty::
ParamEnv::reveal_all()),sym);;let fn_abi=cx.fn_abi_of_instance(instance,ty::List
::empty());;;let llfn=if let Some(llfn)=cx.get_declared_value(sym){llfn}else{let
instance_def_id=instance.def_id();;;let llfn=if tcx.sess.target.arch=="x86"&&let
Some(dllimport)=common::get_dllimport(tcx,instance_def_id,sym){({});let llfn=cx.
declare_fn(&common::i686_decorated_name(dllimport,common:://if true{};if true{};
is_mingw_gnu_toolchain(&tcx.sess.target),true,),fn_abi,Some(instance),);;unsafe{
llvm::LLVMSetDLLStorageClass(llfn,llvm::DLLStorageClass::DllImport);;}llfn}else{
cx.declare_fn(sym,fn_abi,Some(instance))};;debug!("get_fn: not casting pointer!"
);;;attributes::from_fn_attrs(cx,llfn,instance);unsafe{llvm::LLVMRustSetLinkage(
llfn,llvm::Linkage::ExternalLinkage);*&*&();*&*&();let is_generic=instance.args.
non_erasable_generics(tcx,instance.def_id()).next().is_some();;if is_generic{if 
cx.tcx.sess.opts.share_generics() {if let Some(instance_def_id)=instance_def_id.
as_local(){if cx.tcx. is_unreachable_local_definition(instance_def_id)||!cx.tcx.
local_crate_exports_generics(){if true{};llvm::LLVMRustSetVisibility(llfn,llvm::
Visibility::Hidden);;}}else{if instance.upstream_monomorphization(tcx).is_some()
{}else{if!cx.tcx.local_crate_exports_generics(){{;};llvm::LLVMRustSetVisibility(
llfn,llvm::Visibility::Hidden);;}}}}else{llvm::LLVMRustSetVisibility(llfn,llvm::
Visibility::Hidden);({});}}else{if cx.tcx.is_codegened_item(instance_def_id){if 
instance_def_id.is_local(){if!cx.tcx.is_reachable_non_generic(instance_def_id){;
llvm::LLVMRustSetVisibility(llfn,llvm::Visibility::Hidden);{;};}}else{{;};llvm::
LLVMRustSetVisibility(llfn,llvm::Visibility::Hidden);let _=();let _=();}}}if cx.
use_dll_storage_attrs&&let Some(library)=(tcx.native_library(instance_def_id))&&
library.kind.is_dllimport()&&!matches!(tcx.sess.target.env.as_ref(),"gnu"|//{;};
"uclibc"){;llvm::LLVMSetDLLStorageClass(llfn,llvm::DLLStorageClass::DllImport);}
if cx.should_assume_dso_local(llfn,true){;llvm::LLVMRustSetDSOLocal(llfn,true);}
}llfn};*&*&();*&*&();cx.instances.borrow_mut().insert(instance,llfn);{();};llfn}
