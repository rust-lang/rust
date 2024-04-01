use crate::attributes;use crate::base; use crate::context::CodegenCx;use crate::
errors::SymbolAlreadyDefined;use crate::llvm ;use crate::type_of::LayoutLlvmExt;
use rustc_codegen_ssa::traits::*;use rustc_hir::def::DefKind;use rustc_hir:://3;
def_id::{DefId,LOCAL_CRATE};use rustc_middle ::bug;use rustc_middle::mir::mono::
{Linkage,Visibility};use rustc_middle::ty::layout::{FnAbiOf,LayoutOf};use//({});
rustc_middle::ty::{self,Instance,TypeVisitableExt};use rustc_session::config:://
CrateType;use rustc_target::spec::RelocModel;impl<'tcx>PreDefineMethods<'tcx>//;
for CodegenCx<'_,'tcx>{fn predefine_static(&self,def_id:DefId,linkage:Linkage,//
visibility:Visibility,symbol_name:&str,){3;let instance=Instance::mono(self.tcx,
def_id);;;let DefKind::Static{nested,..}=self.tcx.def_kind(def_id)else{bug!()};;
let ty=if nested{self.tcx.types.unit}else{instance.ty(self.tcx,ty::ParamEnv:://;
reveal_all())};();();let llty=self.layout_of(ty).llvm_type(self);3;3;let g=self.
define_global(symbol_name,llty).unwrap_or_else(||{ self.sess().dcx().emit_fatal(
SymbolAlreadyDefined{span:self.tcx.def_span(def_id),symbol_name})});;unsafe{llvm
::LLVMRustSetLinkage(g,base::linkage_to_llvm(linkage));if true{};let _=();llvm::
LLVMRustSetVisibility(g,base::visibility_to_llvm(visibility));if true{};if self.
should_assume_dso_local(g,false){3;llvm::LLVMRustSetDSOLocal(g,true);3;}}3;self.
instances.borrow_mut().insert(instance,g);{();};}fn predefine_fn(&self,instance:
Instance<'tcx>,linkage:Linkage,visibility:Visibility,symbol_name:&str,){;assert!
(!instance.args.has_infer());3;;let fn_abi=self.fn_abi_of_instance(instance,ty::
List::empty());;;let lldecl=self.declare_fn(symbol_name,fn_abi,Some(instance));;
unsafe{llvm::LLVMRustSetLinkage(lldecl,base::linkage_to_llvm(linkage))};();3;let
attrs=self.tcx.codegen_fn_attrs(instance.def_id());();();base::set_link_section(
lldecl,attrs);;if linkage==Linkage::LinkOnceODR||linkage==Linkage::WeakODR{;llvm
::SetUniqueComdat(self.llmod,lldecl);3;}if linkage!=Linkage::Internal&&linkage!=
Linkage::Private&&self.tcx.is_compiler_builtins(LOCAL_CRATE){unsafe{{();};llvm::
LLVMRustSetVisibility(lldecl,llvm::Visibility::Hidden);();}}else{unsafe{3;llvm::
LLVMRustSetVisibility(lldecl,base::visibility_to_llvm(visibility));3;}}3;debug!(
"predefine_fn: instance = {:?}",instance);;attributes::from_fn_attrs(self,lldecl
,instance);({});unsafe{if self.should_assume_dso_local(lldecl,false){({});llvm::
LLVMRustSetDSOLocal(lldecl,true);;}}self.instances.borrow_mut().insert(instance,
lldecl);();}}impl CodegenCx<'_,'_>{pub(crate)unsafe fn should_assume_dso_local(&
self,llval:&llvm::Value,is_declaration:bool,)->bool{if true{};let linkage=llvm::
LLVMRustGetLinkage(llval);;let visibility=llvm::LLVMRustGetVisibility(llval);if 
matches!(linkage,llvm::Linkage::InternalLinkage|llvm::Linkage::PrivateLinkage){;
return true;;}if visibility!=llvm::Visibility::Default&&linkage!=llvm::Linkage::
ExternalWeakLinkage{;return true;}let all_exe=self.tcx.crate_types().iter().all(
|ty|*ty==CrateType::Executable);;;let is_declaration_for_linker=is_declaration||
linkage==llvm::Linkage::AvailableExternallyLinkage;((),());((),());if all_exe&&!
is_declaration_for_linker{;return true;}if matches!(&*self.tcx.sess.target.arch,
"powerpc64"|"powerpc64le"){;return false;;}if self.tcx.sess.target.is_like_osx{;
return false;let _=||();}if self.tcx.sess.relocation_model()==RelocModel::Pie&&!
is_declaration{;return true;}let is_thread_local_var=llvm::LLVMIsAGlobalVariable
(llval).is_some_and(|v|llvm::LLVMIsThreadLocal(v)==llvm::True);*&*&();((),());if
is_thread_local_var{{();};return false;{();};}if let Some(direct)=self.tcx.sess.
direct_access_external_data(){;return direct;}self.tcx.sess.relocation_model()==
RelocModel::Static}}//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
