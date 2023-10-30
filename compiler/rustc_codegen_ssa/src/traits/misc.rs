use super::BackendTypes;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::mono::CodegenUnit;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::Session;
use std::cell::RefCell;

pub trait MiscMethods<'tcx>: BackendTypes {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), Self::Value>>;
    fn check_overflow(&self) -> bool;
    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Function;
    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Self::Value;
    fn eh_personality(&self) -> Self::Value;
    fn sess(&self) -> &Session;
    fn codegen_unit(&self) -> &'tcx CodegenUnit<'tcx>;
    fn set_frame_pointer_type(&self, llfn: Self::Function);
    fn apply_target_cpu_attr(&self, llfn: Self::Function);
    /// Declares the extern "C" main function for the entry point. Returns None if the symbol already exists.
    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function>;
    fn create_autodiff(&self) -> Vec<Self::Function>;
}
