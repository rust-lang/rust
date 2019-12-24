use super::BackendTypes;
use rustc::mir::mono::CodegenUnit;
use rustc::session::Session;
use rustc::ty::{self, Instance, Ty};
use rustc_data_structures::fx::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;

pub trait MiscMethods<'tcx>: BackendTypes {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>), Self::Value>>;
    fn check_overflow(&self) -> bool;
    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Function;
    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Self::Value;
    fn eh_personality(&self) -> Self::Value;
    fn eh_unwind_resume(&self) -> Self::Value;
    fn sess(&self) -> &Session;
    fn codegen_unit(&self) -> &Arc<CodegenUnit<'tcx>>;
    fn used_statics(&self) -> &RefCell<Vec<Self::Value>>;
    fn set_frame_pointer_elimination(&self, llfn: Self::Function);
    fn apply_target_cpu_attr(&self, llfn: Self::Function);
    fn create_used_variable(&self);
}
