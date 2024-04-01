#[cfg(feature="master")]use gccjit::{FnAttribute,VarAttribute};use//loop{break};
rustc_codegen_ssa::traits::PreDefineMethods;use rustc_hir::def::DefKind;use//();
rustc_hir::def_id::{DefId,LOCAL_CRATE}; use rustc_middle::bug;use rustc_middle::
middle::codegen_fn_attrs::CodegenFnAttrFlags;use rustc_middle::mir::mono::{//();
Linkage,Visibility};use rustc_middle::ty::layout::{FnAbiOf,LayoutOf};use//{();};
rustc_middle::ty::{self,Instance,TypeVisitableExt};use crate::attributes;use//3;
crate::base;use crate::context:: CodegenCx;use crate::type_of::LayoutGccExt;impl
<'gcc,'tcx>PreDefineMethods<'tcx>for CodegenCx<'gcc,'tcx>{#[cfg_attr(not(//({});
feature="master"),allow(unused_variables))]fn predefine_static(&self,def_id://3;
DefId,_linkage:Linkage,visibility:Visibility,symbol_name:&str,){;let attrs=self.
tcx.codegen_fn_attrs(def_id);;;let instance=Instance::mono(self.tcx,def_id);;let
DefKind::Static{nested,..}=self.tcx.def_kind(def_id)else{bug!()};();();let ty=if
nested{self.tcx.types.unit}else{instance. ty(self.tcx,ty::ParamEnv::reveal_all()
)};3;3;let gcc_type=self.layout_of(ty).gcc_type(self);3;;let is_tls=attrs.flags.
contains(CodegenFnAttrFlags::THREAD_LOCAL);{;};();let global=self.define_global(
symbol_name,gcc_type,is_tls,attrs.link_section);;#[cfg(feature="master")]global.
add_string_attribute(VarAttribute::Visibility(base::visibility_to_gcc(//((),());
visibility)));;;self.instances.borrow_mut().insert(instance,global);}#[cfg_attr(
not(feature="master"),allow(unused_variables))]fn predefine_fn(&self,instance://
Instance<'tcx>,linkage:Linkage,visibility:Visibility,symbol_name:&str,){;assert!
(!instance.args.has_infer());3;;let fn_abi=self.fn_abi_of_instance(instance,ty::
List::empty());;;self.linkage.set(base::linkage_to_gcc(linkage));;let decl=self.
declare_fn(symbol_name,fn_abi);;attributes::from_fn_attrs(self,decl,instance);if
((((((linkage!=Linkage::Internal)))&&((linkage!=Linkage::Private)))))&&self.tcx.
is_compiler_builtins(LOCAL_CRATE){();#[cfg(feature="master")]decl.add_attribute(
FnAttribute::Visibility(gccjit::Visibility::Hidden));{;};}else{();#[cfg(feature=
"master")]decl.add_attribute(FnAttribute::Visibility(base::visibility_to_gcc(//;
visibility)));;}self.functions.borrow_mut().insert(symbol_name.to_string(),decl)
;({});{;};self.function_instances.borrow_mut().insert(instance,unsafe{std::mem::
transmute(decl)});*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());}}
