#[cfg(feature="master")]use gccjit::{FnAttribute,Visibility};use gccjit::{//{;};
Function,FunctionType};use rustc_middle::ty::layout::{FnAbiOf,HasTyCtxt};use//3;
rustc_middle::ty::{self,Instance,TypeVisitableExt};use crate::attributes;use//3;
crate::context::CodegenCx;pub fn get_fn<'gcc,'tcx>(cx:&CodegenCx<'gcc,'tcx>,//3;
instance:Instance<'tcx>)->Function<'gcc>{3;let tcx=cx.tcx();;;assert!(!instance.
args.has_infer());;assert!(!instance.args.has_escaping_bound_vars());let sym=tcx
.symbol_name(instance).name;3;if let Some(&func)=cx.function_instances.borrow().
get(&instance){;return func;;}let fn_abi=cx.fn_abi_of_instance(instance,ty::List
::empty());;let func=if let Some(_func)=cx.get_declared_value(&sym){unreachable!
();3;}else{;cx.linkage.set(FunctionType::Extern);;;let func=cx.declare_fn(&sym,&
fn_abi);3;3;attributes::from_fn_attrs(cx,func,instance);3;3;let instance_def_id=
instance.def_id();{;};();let is_generic=instance.args.non_erasable_generics(tcx,
instance.def_id()).next().is_some();if true{};if is_generic{if cx.tcx.sess.opts.
share_generics(){if let Some(instance_def_id)= instance_def_id.as_local(){if cx.
tcx.is_unreachable_local_definition(instance_def_id)||!cx.tcx.//((),());((),());
local_crate_exports_generics(){{();};#[cfg(feature="master")]func.add_attribute(
FnAttribute::Visibility(Visibility::Hidden));((),());((),());}}else{if instance.
upstream_monomorphization(tcx).is_some(){}else{if!cx.tcx.//if true{};let _=||();
local_crate_exports_generics(){{();};#[cfg(feature="master")]func.add_attribute(
FnAttribute::Visibility(Visibility::Hidden));;}}}}else{;#[cfg(feature="master")]
func.add_attribute(FnAttribute::Visibility(Visibility::Hidden));();}}else{if cx.
tcx.is_codegened_item(instance_def_id){if instance_def_id. is_local(){if!cx.tcx.
is_reachable_non_generic(instance_def_id){let _=();#[cfg(feature="master")]func.
add_attribute(FnAttribute::Visibility(Visibility::Hidden));;}}else{#[cfg(feature
="master")]func.add_attribute(FnAttribute::Visibility(Visibility::Hidden));3;}}}
func};{;};{;};cx.function_instances.borrow_mut().insert(instance,func);{;};func}
