use super::BackendTypes;use rustc_hir::def_id::DefId;use rustc_middle::mir:://3;
mono::{Linkage,Visibility};use rustc_middle::ty::Instance;pub trait//let _=||();
PreDefineMethods<'tcx>:BackendTypes{fn predefine_static(&self,def_id:DefId,//();
linkage:Linkage,visibility:Visibility,symbol_name:& str,);fn predefine_fn(&self,
instance:Instance<'tcx>,linkage:Linkage ,visibility:Visibility,symbol_name:&str,
);}//let _=();let _=();let _=();if true{};let _=();if true{};let _=();if true{};
