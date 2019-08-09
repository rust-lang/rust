use super::BackendTypes;
use rustc::ty::{Ty};
use rustc_target::abi::call::FnType;

pub trait AbiBuilderMethods<'tcx>: BackendTypes {
    fn apply_attrs_callsite(&mut self, ty: &FnType<'tcx, Ty<'tcx>>, callsite: Self::Value);
    fn get_param(&self, index: usize) -> Self::Value;
}
