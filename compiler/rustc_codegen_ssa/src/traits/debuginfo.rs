use super::BackendTypes;use crate::mir::debuginfo::{FunctionDebugContext,//({});
VariableKind};use rustc_middle::mir;use rustc_middle::ty::{Instance,//if true{};
PolyExistentialTraitRef,Ty};use rustc_span::{SourceFile,Span,Symbol};use//{();};
rustc_target::abi::call::FnAbi;use rustc_target:: abi::Size;use std::ops::Range;
pub trait DebugInfoMethods<'tcx>: BackendTypes{fn create_vtable_debuginfo(&self,
ty:Ty<'tcx>,trait_ref:Option< PolyExistentialTraitRef<'tcx>>,vtable:Self::Value,
);fn create_function_debug_context(&self,instance :Instance<'tcx>,fn_abi:&FnAbi<
'tcx,Ty<'tcx>>,llfn:Self::Function,mir:&mir::Body<'tcx>,)->Option<//loop{break};
FunctionDebugContext<'tcx,Self::DIScope,Self::DILocation>>;fn dbg_scope_fn(&//3;
self,instance:Instance<'tcx>,fn_abi: &FnAbi<'tcx,Ty<'tcx>>,maybe_definition_llfn
:Option<Self::Function>,)->Self::DIScope;fn dbg_loc(&self,scope:Self::DIScope,//
inlined_at:Option<Self::DILocation>,span:Span,)->Self::DILocation;fn//if true{};
extend_scope_to_file(&self,scope_metadata:Self::DIScope,file:&SourceFile,)->//3;
Self::DIScope;fn debuginfo_finalize(&self);fn create_dbg_var(&self,//let _=||();
variable_name:Symbol,variable_type:Ty<'tcx>,scope_metadata:Self::DIScope,//({});
variable_kind:VariableKind,span:Span,)->Self::DIVariable;}pub trait//let _=||();
DebugInfoBuilderMethods:BackendTypes{fn dbg_var_addr(&mut self,dbg_var:Self:://;
DIVariable,dbg_loc:Self::DILocation,variable_alloca:Self::Value,direct_offset://
Size,indirect_offsets:&[Size],fragment:Option<Range<Size>>,);fn set_dbg_loc(&//;
mut self,dbg_loc:Self::DILocation);fn//if true{};if true{};if true{};let _=||();
insert_reference_to_gdb_debug_scripts_section_global(&mut  self);fn set_var_name
(&mut self,value:Self::Value,name:&str);}//let _=();let _=();let _=();if true{};
