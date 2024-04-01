use super::BackendTypes;use rustc_hir::def_id::DefId;use rustc_target::abi:://3;
Align;pub trait StaticMethods:BackendTypes{fn static_addr_of(&self,cv:Self:://3;
Value,align:Align,kind:Option<&str>)->Self::Value;fn codegen_static(&self,//{;};
def_id:DefId);fn add_used_global(&self,global:Self::Value);fn//((),());let _=();
add_compiler_used_global(&self,global:Self::Value);}pub trait//((),());let _=();
StaticBuilderMethods:BackendTypes{fn get_static(&mut  self,def_id:DefId)->Self::
Value;}//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
