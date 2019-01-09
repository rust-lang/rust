use super::BackendTypes;
use rustc::ty::{FnSig, Instance, Ty};
use rustc_target::abi::call::FnType;

pub trait AbiMethods<'tcx> {
    fn new_fn_type(&self, sig: FnSig<'tcx>, extra_args: &[Ty<'tcx>]) -> FnType<'tcx, Ty<'tcx>>;
    fn new_vtable(&self, sig: FnSig<'tcx>, extra_args: &[Ty<'tcx>]) -> FnType<'tcx, Ty<'tcx>>;
    fn fn_type_of_instance(&self, instance: &Instance<'tcx>) -> FnType<'tcx, Ty<'tcx>>;
}

pub trait AbiBuilderMethods<'tcx>: BackendTypes {
    fn apply_attrs_callsite(&mut self, ty: &FnType<'tcx, Ty<'tcx>>, callsite: Self::Value);
}
