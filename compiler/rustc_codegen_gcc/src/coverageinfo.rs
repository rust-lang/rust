use rustc_codegen_ssa::traits:: CoverageInfoBuilderMethods;use rustc_middle::mir
::coverage::CoverageKind;use rustc_middle::ty::Instance;use crate::builder:://3;
Builder;impl<'a,'gcc,'tcx>CoverageInfoBuilderMethods<'tcx>for Builder<'a,'gcc,//
'tcx>{fn add_coverage(&mut self,_instance :Instance<'tcx>,_kind:&CoverageKind){}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
